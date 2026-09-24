#!/usr/bin/env python3
"""Test this checkout's Python operator using the image's existing binary library."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

MODULE = "sgl_kernel_npu.norm.split_qkv_rmsnorm_rope"
RELATIVE_SOURCE = "python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py"
TESTS = (
    "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope.py",
    "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope_head192.py",
)
BASELINE = "9bc1ac431e39760fc8e8d82922f39334cd248b24"
# This sync-branch ancestor contains the exact same operator blob as the PR.
BASELINE_SOURCE_COMMIT = "884908f16ca3304d34090beb53cd60f7010cf2c8"
BASELINE_SOURCE_BLOB = "ad6bc2bf008232a10e572cb99e494a1dacf06737"
WARMUP, ROUNDS, CALLS = 10, 5, 30


def baseline_supports(head_dim):
    return head_dim == 192 or (head_dim > 0 and head_dim & (head_dim - 1) == 0)


def load_baseline(root, output):
    """Load the agreed PR revision without changing the image or checkout."""
    result = subprocess.run(
        ["git", "show", f"{BASELINE_SOURCE_COMMIT}:{RELATIVE_SOURCE}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Cannot read baseline source {BASELINE_SOURCE_COMMIT} from {root}; "
            "fetch that commit first. " + result.stderr.decode(errors="replace")
        )
    blob = hashlib.sha1(
        b"blob " + str(len(result.stdout)).encode() + b"\0" + result.stdout
    ).hexdigest()
    if blob != BASELINE_SOURCE_BLOB:
        raise RuntimeError(f"Baseline source differs from the verified PR blob: {blob}")
    path = output / "baseline_module.py"
    path.write_bytes(result.stdout)
    name = "_head192_pr_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, hashlib.sha256(result.stdout).hexdigest()


def torch_reference(torch, case, *, cast_output=True):
    """Native split, FP32 RMSNorm/RoPE, output cast and V copy on input device."""
    tokens, dim = case["input"].shape[0], case["head_dim"]
    q_width, kv_width = case["q_hidden_size"], case["kv_hidden_size"]
    q, k, v = case["input"].split((q_width, kv_width, kv_width), dim=-1)
    rope_dim = case["sin"].shape[-1]
    sin = case["sin"].float().reshape(tokens, 1, rope_dim)
    cos = case["cos"].float().reshape(tokens, 1, rope_dim)
    outputs = []
    for name, values, width in (("q", q, q_width), ("k", k, kv_width)):
        values = values.float().reshape(tokens, width // dim, dim)
        if case["eps"] is not None:
            values = values / torch.sqrt(
                values.square().mean(dim=-1, keepdim=True) + case["eps"]
            )
            values = values * case[name + "_weight"].float().reshape(dim)
            if case[name + "_bias"] is not None:
                values = values + case[name + "_bias"].float().reshape(dim)
        rot, half = values[..., :rope_dim], rope_dim // 2
        if case["is_neox_style"]:
            left, right = rot[..., :half], rot[..., half:]
            roped = torch.cat(
                (
                    left * cos[..., :half] - right * sin[..., :half],
                    right * cos[..., half:] + left * sin[..., half:],
                ),
                dim=-1,
            )
        else:
            even, odd = rot[..., 0::2], rot[..., 1::2]
            pair_sin, pair_cos = sin[..., :half], cos[..., :half]
            roped = torch.stack(
                (even * pair_cos - odd * pair_sin, odd * pair_cos + even * pair_sin),
                dim=-1,
            ).flatten(start_dim=-2)
        values = torch.cat((roped, values[..., rope_dim:]), dim=-1)
        values = values.reshape(tokens, width)
        outputs.append(values.to(case["input"].dtype) if cast_output else values)
    return (*outputs, v.clone(memory_format=torch.contiguous_format))


def check_outputs(torch, outputs, expected, dtype):
    errors = []
    for actual, wanted in zip(outputs[:2], expected[:2]):
        if actual.dtype != dtype or actual.shape != wanted.shape:
            raise AssertionError("Q/K output dtype or shape changed")
        actual, wanted = actual.detach().float().cpu(), wanted.float().cpu()
        torch.testing.assert_close(actual, wanted, rtol=0, atol=5e-2)
        errors.append(float((actual - wanted).abs().max()) if actual.numel() else 0.0)
    actual_v, wanted_v = outputs[2].detach().cpu(), expected[2].cpu()
    if actual_v.dtype != wanted_v.dtype or actual_v.shape != wanted_v.shape:
        raise AssertionError("V output dtype or shape changed")
    if not torch.equal(
        actual_v.contiguous().view(torch.uint8), wanted_v.contiguous().view(torch.uint8)
    ):
        raise AssertionError("V differs from the input bytes")
    return {"q_max_abs_error": errors[0], "k_max_abs_error": errors[1], "v_exact": True}


def measure_calls(call, synchronize, *, warmup=WARMUP, rounds=ROUNDS, calls=CALLS):
    """Synchronized wall time of the public call, not a pure device-kernel timer."""
    for _ in range(warmup):
        call()
    timings = []
    for _ in range(rounds):
        synchronize()
        start = time.perf_counter()
        for _ in range(calls):
            call()
        synchronize()
        timings.append((time.perf_counter() - start) * 1e6 / calls)
    return {
        "round_us_per_call": timings,
        "median_us_per_call": statistics.median(timings),
    }


def benchmark_implementation(torch, function, case, expected):
    results = {}

    def invoke():
        return function(**case)

    try:
        eager_outputs = invoke()
        torch.npu.synchronize()
        accuracy = check_outputs(torch, eager_outputs, expected, case["input"].dtype)
        eager_cpu = tuple(value.detach().cpu() for value in eager_outputs)
        results["eager"] = {
            "status": "MEASURED",
            "accuracy": accuracy,
            **measure_calls(invoke, torch.npu.synchronize),
        }
    except Exception:
        results["eager"] = {"status": "FAILED", "error": traceback.format_exc()}
        # A graph comparison is invalid if this implementation's eager output failed.
        results["graph"] = {"status": "NOT_RUN", "reason": "Eager validation failed"}
        return results
    try:
        stream = torch.npu.Stream()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for _ in range(WARMUP):
                invoke()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            graph_outputs = invoke()
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        accuracy = check_outputs(torch, graph_outputs, expected, case["input"].dtype)
        check_outputs(torch, graph_outputs, eager_cpu, case["input"].dtype)
        with torch.npu.stream(stream):
            timings = measure_calls(graph.replay, torch.npu.synchronize)
        # Recheck retained graph outputs after the timed replays as well.
        check_outputs(torch, graph_outputs, expected, case["input"].dtype)
        results["graph"] = {"status": "MEASURED", "accuracy": accuracy, **timings}
    except Exception:
        results["graph"] = {"status": "FAILED", "error": traceback.format_exc()}
    return results


class BenchmarkAudit:
    """Reuse the exact final operator inputs from each passing pytest item."""

    def __init__(self, torch, candidate, baseline):
        self.torch, self.candidate, self.baseline = torch, candidate, baseline
        self.signature = inspect.signature(candidate)
        self.last_case = None
        self.calls = 0
        self.results = []

    def wrap(self):
        @wraps(self.candidate)
        def captured(*args, **kwargs):
            outputs = self.candidate(*args, **kwargs)
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.last_case = dict(bound.arguments)
            self.calls += 1
            return outputs

        return captured

    def pytest_runtest_setup(self, item):
        self.last_case, self.calls = None, 0

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        row = {"test": report.nodeid, "operator_calls_in_test": self.calls}
        self.results.append(row)
        if not report.passed:
            row.update(status="NOT_RUN", reason="Pytest case did not pass")
        elif self.last_case is None:
            if report.nodeid.endswith("::test_split_qkvgate_gemma_rmsnorm_rope"):
                row.update(
                    status="NOT_APPLICABLE",
                    reason="Separate Gemma operator; correctness regression only",
                )
            else:
                row.update(
                    status="FAILED",
                    reason="Passing case did not call the requested operator",
                )
        else:
            try:
                self.run_case(row, self.last_case)
            except Exception:
                row.update(status="FAILED", error=traceback.format_exc())
        self.last_case = None
        print(f"\nBenchmark {row['status']}: {report.nodeid}", flush=True)
        for mode in ("eager", "graph"):
            timings = []
            for name, implementation in row.get("implementations", {}).items():
                result = implementation.get(mode, implementation)
                value = (
                    f"{result['median_us_per_call']:.3f} us"
                    if result.get("status") == "MEASURED"
                    else result.get("status", "NOT_RUN")
                )
                timings.append(f"{name}={value}")
            if timings:
                ratios = [
                    f"{name}={value:.3f}"
                    for name, value in row.get("ratios", {}).get(mode, {}).items()
                ]
                print(f"  {mode}: " + ", ".join(timings + ratios), flush=True)

    def run_case(self, row, case):
        torch = self.torch
        row["parameters"] = {
            key: (
                {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in case.items()
        }
        cpu_case = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in case.items()
        }
        expected = torch_reference(torch, cpu_case, cast_output=False)
        implementations = {
            "candidate": self.candidate,
            "torch_native": lambda **inputs: torch_reference(torch, inputs),
        }
        measurements = {}
        row["implementations"] = measurements
        if baseline_supports(case["head_dim"]):
            implementations["pr_baseline"] = self.baseline
        else:
            measurements["pr_baseline"] = {
                "status": "UNSUPPORTED",
                "reason": "Baseline permits only powers of two and head_dim=192",
            }
        for name, function in implementations.items():
            measurements[name] = benchmark_implementation(
                torch, function, case, expected
            )
        row["ratios"] = {}
        for mode in ("eager", "graph"):
            candidate = measurements["candidate"][mode]
            native = measurements["torch_native"][mode]
            ratios = row["ratios"][mode] = {}
            if candidate["status"] == native["status"] == "MEASURED":
                ratios["torch_over_candidate"] = (
                    native["median_us_per_call"] / candidate["median_us_per_call"]
                )
            baseline = measurements["pr_baseline"].get(mode, {})
            if candidate["status"] == baseline.get("status") == "MEASURED":
                ratios["candidate_over_baseline"] = (
                    candidate["median_us_per_call"] / baseline["median_us_per_call"]
                )
        row["status"] = (
            "MEASURED"
            if all(
                measurements[name][mode]["status"] == "MEASURED"
                for name in implementations
                for mode in ("eager", "graph")
            )
            else "FAILED"
        )

    @property
    def complete(self):
        return any(row["status"] == "MEASURED" for row in self.results) and all(
            row["status"] in ("MEASURED", "NOT_APPLICABLE") for row in self.results
        )


class SelectionAudit:
    """Do not treat an environment-filtered subset as the full regression run."""

    def __init__(self, expected_files):
        self.expected_files = {str(Path(path).resolve()) for path in expected_files}
        self.collected_files = set()
        self.deselected = 0

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collection_finish(self, session):
        self.collected_files = {
            str(Path(item.path).resolve()) for item in session.items
        }

    @property
    def complete(self):
        return self.deselected == 0 and self.expected_files <= self.collected_files


def load_candidate(path):
    """Replace one module in this process; importing the image first loads its .so."""
    installed = importlib.import_module(MODULE)
    spec = importlib.util.spec_from_file_location(MODULE, path)
    candidate = importlib.util.module_from_spec(spec)
    parent_name, child_name = MODULE.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    original_attribute = getattr(parent, child_name)
    sys.modules[MODULE] = candidate
    try:
        spec.loader.exec_module(candidate)
        setattr(parent, child_name, candidate)
        if importlib.import_module(MODULE) is not candidate:
            raise RuntimeError("Candidate module was not selected")
    except BaseException:
        sys.modules[MODULE] = installed
        setattr(parent, child_name, original_attribute)
        raise

    def restore():
        sys.modules[MODULE] = installed
        setattr(parent, child_name, original_attribute)

    return installed, candidate, restore


def summarize_junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return {
        "total": len(cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "failed": sum(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible NPU index")
    parser.add_argument("--out", type=Path, help="New evidence directory")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare exact pytest inputs against the PR baseline and native Torch",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.out or Path(f"/home/tyj/glm52-ms1/evidence/head192-host-{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "device": args.device,
        "performance": "NOT_EVALUATED",
    }
    restore = None
    print(f"Evidence: {output.resolve()}", flush=True)
    try:
        import pytest
        import torch
        import torch_npu  # noqa: F401

        if (
            args.device < 0
            or not torch.npu.is_available()
            or args.device >= torch.npu.device_count()
        ):
            raise RuntimeError("Requested visible NPU is not available")
        torch.npu.set_device(args.device)
        source = root / RELATIVE_SOURCE
        installed, candidate, restore = load_candidate(source)
        installed_source = Path(installed.__file__).read_bytes()
        candidate_source = source.read_bytes()
        (output / "installed_module.py").write_bytes(installed_source)
        (output / "candidate_module.py").write_bytes(candidate_source)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        report.update(
            {
                "git_head": revision,
                "python": sys.version,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "torch-npu",
                        "triton-ascend",
                        "sgl-kernel-npu",
                        "pytest",
                    )
                },
                "device_properties": str(torch.npu.get_device_properties(args.device)),
                "installed_module": installed.__file__,
                "installed_package": importlib.import_module("sgl_kernel_npu").__file__,
                "candidate_module": candidate.__file__,
                "candidate_host_module": candidate.split_qkv_rmsnorm_rope.__module__,
                "device_properties_helper": inspect.getsourcefile(
                    inspect.unwrap(candidate.get_device_properties)
                ),
                "installed_source_sha256": hashlib.sha256(installed_source).hexdigest(),
                "candidate_source_sha256": hashlib.sha256(candidate_source).hexdigest(),
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "test_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in TESTS
                },
            }
        )
        print(f"Installed binary package: {installed.__file__}", flush=True)
        print(f"Candidate Python module: {candidate.__file__}", flush=True)
        selection = SelectionAudit(root / name for name in TESTS)
        plugins = [selection]
        benchmark = None
        if args.benchmark:
            baseline, baseline_hash = load_baseline(root, output)
            benchmark = BenchmarkAudit(
                torch, candidate.split_qkv_rmsnorm_rope, baseline.split_qkv_rmsnorm_rope
            )
            candidate.split_qkv_rmsnorm_rope = benchmark.wrap()
            plugins.append(benchmark)
            report["performance"] = {
                "status": "RUNNING",
                "baseline_commit": BASELINE,
                "baseline_source_commit": BASELINE_SOURCE_COMMIT,
                "baseline_source_git_blob": BASELINE_SOURCE_BLOB,
                "baseline_source_sha256": baseline_hash,
                "warmup_calls": WARMUP,
                "rounds": ROUNDS,
                "calls_per_round": CALLS,
                "measurement_order": "Per case: candidate, Torch native, supported PR baseline; each eager then graph",
                "metric": "Synchronized wall-clock us per public call (not pure kernel time)",
                "input_selection": "Last real operator call in each passing pytest case",
                "graph_inputs": "Static; capture and compilation excluded; replay warmup before timing",
                "ground_truth": "Native Torch FP32 on CPU; native NPU implementation casts Q/K to input dtype and copies V",
                "accuracy": "Q/K rtol=0 atol=0.05, V byte-exact; existing pytest checks also retained",
                "performance_acceptance": "No threshold supplied; measurements are not a performance PASS",
                "results": benchmark.results,
            }
        exit_code = int(
            pytest.main(
                [
                    *(str(root / name) for name in TESTS),
                    "-v",
                    "-ra",
                    f"--junitxml={output / 'pytest.xml'}",
                ],
                plugins=plugins,
            )
        )
        report["pytest_exit_code"] = exit_code
        summary = summarize_junit(output / "pytest.xml")
        report["tests"] = summary
        report["test_selection"] = {
            "deselected": selection.deselected,
            "collected_files": sorted(selection.collected_files),
            "complete": selection.complete,
        }
        if (
            exit_code == 0
            and summary["total"] > 0
            and not summary["skipped"]
            and not summary["failed"]
            and selection.complete
        ):
            report["status"] = "PUBLIC_HOST_TESTS_PASS"
        else:
            report["status"] = "INCOMPLETE" if exit_code == 0 else "FAILED"
            exit_code = exit_code or 1
        if benchmark is not None:
            complete = benchmark.complete and selection.complete and exit_code == 0
            report["performance"]["status"] = (
                "MEASURED" if complete else "FAILED_OR_INCOMPLETE"
            )
            if not complete:
                report["status"] = "FAILED_OR_INCOMPLETE"
                exit_code = exit_code or 1
    except (Exception, KeyboardInterrupt):
        report["status"] = "FAILED"
        if isinstance(report["performance"], dict):
            report["performance"]["status"] = "FAILED_OR_INCOMPLETE"
        (output / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        exit_code = 1
    finally:
        if restore is not None:
            restore()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
