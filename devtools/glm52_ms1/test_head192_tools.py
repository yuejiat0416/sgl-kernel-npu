"""CPU-only dispatch and test-loader checks; no NPU computation is simulated."""

import ast
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py"
BASELINE = "d974d3de5b7b0d6586a41f227cba93a861f07fe1"
SPEC = importlib.util.spec_from_file_location(
    "head192_runner", Path(__file__).with_name("run_head192_tests.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@dataclass(frozen=True)
class Device:
    type: str = "npu"
    index: int = 0


@dataclass
class Tensor:
    shape: tuple
    dtype: str = "bf16"
    device: Device = Device()
    contiguous: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return math.prod(self.shape)


class Capture:
    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.args, self.kwargs = args, kwargs

        return launch


def host_from_source(source):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "split_qkv_rmsnorm_rope"
    )
    capture = Capture()
    environment = {
        "math": math,
        "torch": SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            empty=lambda *shape, **kw: Tensor(shape, **kw),
        ),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "get_device_properties": lambda: (20, 40),
        "split_qkv_rmsnorm_rope_kernel": capture,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "host", "exec"), environment)
    return environment["split_qkv_rmsnorm_rope"], capture


def inputs(
    dim,
    *,
    heads=4,
    ratio=1,
    dtype="bf16",
    norm=True,
    bias=False,
    partial=False,
    neox=True,
):
    q_width, kv_width = heads * ratio * dim, heads * dim
    rope = dim // 2 if partial else dim
    return (
        (
            Tensor((11, q_width + 2 * kv_width), dtype),
            Tensor((11, 1, 1, rope)),
            Tensor((11, 1, 1, rope)),
            q_width,
            kv_width,
            dim,
        ),
        {
            "eps": 1e-5 if norm else None,
            "q_weight": Tensor((dim,), dtype) if norm else None,
            "k_weight": Tensor((dim,), dtype) if norm else None,
            "q_bias": Tensor((dim,), dtype) if bias else None,
            "k_bias": Tensor((dim,), dtype) if bias else None,
            "is_neox_style": neox,
        },
    )


class HostDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = subprocess.check_output(
            ["git", "show", f"{BASELINE}:{SOURCE}"], cwd=ROOT, text=True
        )
        cls.after = (ROOT / SOURCE).read_text()

    def test_old_dimensions_keep_allocations_grid_and_launch_arguments(self):
        for dim in (64, 128, 256):
            for options in (
                {},
                {"ratio": 6, "bias": True},
                {"partial": True},
                {"norm": False, "dtype": "fp16", "neox": False},
                {"dtype": "fp32", "partial": True, "neox": False},
            ):
                with self.subTest(dim=dim, options=options):
                    args, kwargs = inputs(dim, **options)
                    old, old_launch = host_from_source(self.before)
                    new, new_launch = host_from_source(self.after)
                    self.assertEqual(old(*args, **kwargs), new(*args, **kwargs))
                    self.assertEqual(old_launch.grid, new_launch.grid)
                    self.assertEqual(old_launch.args, new_launch.args)
                    self.assertEqual(old_launch.kwargs, new_launch.kwargs)

    def test_other_non_power_of_two_dimensions_use_complete_heads(self):
        for dim in (96, 160, 384):
            with self.subTest(dim=dim):
                args, kwargs = inputs(dim)
                old, _ = host_from_source(self.before)
                with self.assertRaises(AssertionError):
                    old(*args, **kwargs)
                new, call = host_from_source(self.after)
                outputs = new(*args, **kwargs)
                self.assertEqual(call.grid, (10, 4, 1))
                self.assertEqual(call.args[15:17], (dim, dim))
                self.assertEqual(call.args[19:23], (dim, dim, dim // 2, 0))
                self.assertEqual(call.kwargs, {"DO_PARTIAL": False, "DO_HALF": True})
                self.assertEqual([o.shape for o in outputs], [(11, 4 * dim)] * 3)

    def test_192_uses_complete_heads_without_compiler_override(self):
        for heads in (1, 2, 3, 4, 8, 16, 32, 64):
            with self.subTest(heads=heads):
                args, kwargs = inputs(192, heads=heads)
                old, _ = host_from_source(self.before)
                with self.assertRaises(AssertionError):
                    old(*args, **kwargs)
                new, call = host_from_source(self.after)
                outputs = new(*args, **kwargs)
                self.assertEqual(call.grid, (math.ceil(40 / heads), heads, 1))
                self.assertEqual(call.args[15:17], (192, 192))
                self.assertEqual(call.args[19:23], (192, 192, 96, 0))
                self.assertEqual(call.kwargs, {"DO_PARTIAL": False, "DO_HALF": True})
                self.assertEqual([o.shape for o in outputs], [(11, heads * 192)] * 3)

    def test_all_kernel_functions_remain_identical(self):
        def functions(source):
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name != "split_qkv_rmsnorm_rope"
            }

        self.assertEqual(functions(self.before), functions(self.after))

    def test_192_preserves_existing_parameter_modes(self):
        for options in (
            {"ratio": 2},
            {"ratio": 3, "bias": True},
            {"norm": False},
            {"partial": True},
            {"neox": False},
            {"dtype": "fp16"},
            {"dtype": "fp32", "norm": False, "neox": False},
        ):
            with self.subTest(options=options):
                args, kwargs = inputs(192, **options)
                host, call = host_from_source(self.after)
                outputs = host(*args, **kwargs)
                self.assertEqual(
                    [o.shape for o in outputs],
                    [(11, args[3]), (11, args[4]), (11, args[4])],
                )
                self.assertEqual([o.dtype for o in outputs], [args[0].dtype] * 3)
                self.assertEqual(call.grid, (10, 4, 1))
                self.assertEqual(
                    call.args[6:10],
                    (
                        kwargs["q_weight"],
                        kwargs["q_bias"],
                        kwargs["k_weight"],
                        kwargs["k_bias"],
                    ),
                )
                self.assertEqual(
                    call.args[14:19],
                    (
                        kwargs["eps"],
                        options.get("ratio", 1) * 192,
                        192,
                        kwargs["q_bias"] is not None,
                        kwargs["eps"] is not None,
                    ),
                )
                self.assertEqual(
                    call.kwargs,
                    {
                        "DO_PARTIAL": options.get("partial", False),
                        "DO_HALF": kwargs["is_neox_style"],
                    },
                )


class NativeReferenceTests(unittest.TestCase):
    def test_native_math_on_cpu_matches_existing_fp32_oracle(self):
        """Real Torch math only: this does not emulate or validate NPU execution."""
        try:
            import pytest  # noqa: F401
            import torch
        except ImportError:
            self.skipTest("CPU mathematical check requires Torch and pytest")
        path = ROOT / runner.TESTS[1]
        spec = importlib.util.spec_from_file_location("head192_cpu_oracle", path)
        oracle = importlib.util.module_from_spec(spec)
        # Import existing case construction without importing any NPU binary.
        kernel = ModuleType(runner.MODULE)
        kernel.split_qkv_rmsnorm_rope = None
        utilities = ModuleType("sgl_kernel_npu.utils.triton_utils")
        utilities.get_device_properties = mock.Mock(
            side_effect=AssertionError("CPU math must not invent NPU core counts")
        )
        with mock.patch.dict(
            sys.modules,
            {
                "torch_npu": ModuleType("torch_npu"),
                runner.MODULE: kernel,
                "sgl_kernel_npu.utils.triton_utils": utilities,
            },
        ):
            spec.loader.exec_module(oracle)

        checked = 0

        def native_on_cpu(**case):
            nonlocal checked
            expected = oracle._reference(case)
            fp32 = runner.torch_reference(torch, case, cast_output=False)
            for actual, wanted in zip(fp32[:2], expected[:2]):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            output = runner.torch_reference(torch, case)
            runner.check_outputs(torch, output, expected, case["input"].dtype)
            if output[2].numel():
                self.assertNotEqual(
                    output[2].untyped_storage().data_ptr(),
                    case["input"].untyped_storage().data_ptr(),
                )
            checked += 1
            return output

        # Preserve all source-defined shapes, seeds, dtypes, caches and options.
        oracle.split_qkv_rmsnorm_rope = native_on_cpu
        oracle._to_device = lambda case: case
        functions = (
            oracle.test_public_host_shapes,
            oracle.test_head192_public_host_2d_cache,
            oracle.test_head192_public_host_norm_boundaries,
            oracle.test_existing_head_dimensions_public_host,
            oracle.test_head192_public_host_existing_modes,
        )
        with mock.patch.object(
            torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True
        ):
            for function in functions:
                axes = []
                for mark in function.pytestmark:
                    if mark.name != "parametrize":
                        continue
                    names, values = mark.args[:2]
                    names = [name.strip() for name in names.split(",")]
                    axes.append(
                        [
                            dict(
                                zip(
                                    names,
                                    (
                                        value.values
                                        if hasattr(value, "values")
                                        else (value,) if len(names) == 1 else value
                                    ),
                                )
                            )
                            for value in values
                        ]
                    )
                for combination in itertools.product(*axes):
                    parameters = {
                        key: value
                        for axis in combination
                        for key, value in axis.items()
                    }
                    with self.subTest(
                        function=function.__name__, parameters=parameters
                    ):
                        function(**parameters)
        self.assertGreater(checked, 0)
        utilities.get_device_properties.assert_not_called()


class RunnerTests(unittest.TestCase):
    def test_benchmark_capture_preserves_exact_inputs_and_defaults(self):
        def candidate(input, head_dim, eps=None, is_neox_style=True):
            return input

        audit = runner.BenchmarkAudit(None, candidate, None)
        first, last = object(), object()
        wrapped = audit.wrap()
        self.assertIs(wrapped(first, 128), first)
        self.assertIs(wrapped(last, 192, eps=1e-5, is_neox_style=False), last)
        self.assertEqual(audit.calls, 2)
        self.assertIs(audit.last_case["input"], last)
        self.assertEqual(audit.last_case["head_dim"], 192)
        self.assertEqual(audit.last_case["eps"], 1e-5)
        self.assertFalse(audit.last_case["is_neox_style"])
        audit.pytest_runtest_setup(None)
        self.assertIsNone(audit.last_case)
        self.assertEqual(audit.calls, 0)
        wrapped(first, 64)
        self.assertIsNone(audit.last_case["eps"])
        self.assertTrue(audit.last_case["is_neox_style"])

    def test_benchmark_timer_excludes_warmup_and_reports_each_round(self):
        call, synchronize = mock.Mock(), mock.Mock()
        with mock.patch.object(
            runner.time, "perf_counter", side_effect=[0, 0.003, 1, 1.006]
        ):
            result = runner.measure_calls(
                call, synchronize, warmup=10, rounds=2, calls=30
            )
        self.assertEqual(call.call_count, 70)
        self.assertEqual(synchronize.call_count, 4)
        self.assertAlmostEqual(result["round_us_per_call"][0], 100)
        self.assertAlmostEqual(result["round_us_per_call"][1], 200)
        self.assertAlmostEqual(result["median_us_per_call"], 150)
        self.assertEqual((runner.WARMUP, runner.ROUNDS, runner.CALLS), (10, 5, 30))

    def test_benchmark_baseline_support_is_not_a_shape_substitution(self):
        for dim in (64, 128, 192, 256):
            self.assertTrue(runner.baseline_supports(dim))
        for dim in (0, 96, 160, 384):
            self.assertFalse(runner.baseline_supports(dim))

    def test_benchmark_records_native_ratios_and_unsupported_baseline(self):
        candidate, baseline = mock.Mock(), mock.Mock()
        # Signature is needed only for the recorder; no operator is executed.
        audit = runner.BenchmarkAudit(
            SimpleNamespace(Tensor=Tensor), candidate, baseline
        )

        def measured(torch, function, case, expected):
            self.assertEqual(case, {"head_dim": 96})
            median = 2 if function is candidate else 10
            return {
                mode: {"status": "MEASURED", "median_us_per_call": median}
                for mode in ("eager", "graph")
            }

        row = {}
        with (
            mock.patch.object(runner, "torch_reference", return_value="truth"),
            mock.patch.object(
                runner, "benchmark_implementation", side_effect=measured
            ) as run,
        ):
            audit.run_case(row, {"head_dim": 96})
        self.assertEqual(row["status"], "MEASURED")
        self.assertEqual(row["implementations"]["pr_baseline"]["status"], "UNSUPPORTED")
        self.assertEqual(row["ratios"]["graph"]["torch_over_candidate"], 5)
        self.assertNotIn("candidate_over_baseline", row["ratios"]["eager"])
        self.assertEqual(run.call_count, 2)
        baseline.assert_not_called()

    def test_benchmark_failure_cannot_be_reported_complete(self):
        audit = runner.BenchmarkAudit(
            SimpleNamespace(Tensor=Tensor), mock.Mock(), mock.Mock()
        )
        results = {
            "eager": {"status": "MEASURED", "median_us_per_call": 1},
            "graph": {"status": "FAILED", "error": "capture failed"},
        }
        row = {}
        with (
            mock.patch.object(runner, "torch_reference", return_value="truth"),
            mock.patch.object(runner, "benchmark_implementation", return_value=results),
        ):
            audit.run_case(row, {"head_dim": 192})
        audit.results.append(row)
        self.assertEqual(row["status"], "FAILED")
        self.assertFalse(audit.complete)

    def test_benchmark_only_excludes_the_separate_gemma_operator(self):
        audit = runner.BenchmarkAudit(None, mock.Mock(), None)
        audit.pytest_runtest_logreport(
            SimpleNamespace(
                when="call",
                passed=True,
                nodeid="file.py::test_split_qkvgate_gemma_rmsnorm_rope",
            )
        )
        self.assertEqual(audit.results[0]["status"], "NOT_APPLICABLE")
        self.assertFalse(audit.complete)
        audit.pytest_runtest_logreport(
            SimpleNamespace(
                when="call", passed=True, nodeid="file.py::test_missing_operator"
            )
        )
        self.assertEqual(audit.results[1]["status"], "FAILED")

    def test_missing_benchmark_baseline_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=128, stderr=b"missing object"),
            ):
                with self.assertRaisesRegex(RuntimeError, "fetch that commit first"):
                    runner.load_baseline(ROOT, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_changed_baseline_blob_cannot_be_used_for_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=b"unverified source"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "differs from the verified PR blob"
                ):
                    runner.load_baseline(ROOT, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_accuracy_failure_prevents_timing_and_graph_measurement(self):
        runtime = SimpleNamespace(npu=SimpleNamespace(synchronize=mock.Mock()))
        function = mock.Mock(return_value="wrong output")
        with (
            mock.patch.object(
                runner, "check_outputs", side_effect=AssertionError("wrong values")
            ),
            mock.patch.object(runner, "measure_calls") as timer,
        ):
            result = runner.benchmark_implementation(
                runtime, function, {"input": SimpleNamespace(dtype="bf16")}, "truth"
            )
        self.assertEqual(result["eager"]["status"], "FAILED")
        self.assertIn("wrong values", result["eager"]["error"])
        self.assertEqual(result["graph"]["status"], "NOT_RUN")
        timer.assert_not_called()

    def test_cached_helper_metadata_allows_runner_to_reach_pytest(self):
        @cache
        def cached_properties():
            raise AssertionError("Source inspection must not execute the helper")

        # Exercise main's actual metadata path with the same wrapper type as the
        # image. The fake runtime does not perform or validate NPU computation.
        candidate = SimpleNamespace(
            __file__=str(ROOT / SOURCE),
            get_device_properties=cached_properties,
            split_qkv_rmsnorm_rope=SimpleNamespace(__module__=runner.MODULE),
        )
        installed = SimpleNamespace(__file__=str(ROOT / SOURCE))
        restore = mock.Mock()
        fake_npu = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            set_device=mock.Mock(),
            get_device_properties=lambda device: "test device",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"

            def run_pytest(arguments, plugins):
                self.assertTrue(any(arg.startswith("--junitxml=") for arg in arguments))
                plugins[0].pytest_collection_finish(
                    SimpleNamespace(
                        items=[
                            SimpleNamespace(path=ROOT / name) for name in runner.TESTS
                        ]
                    )
                )
                (output / "pytest.xml").write_text(
                    "<testsuites><testsuite><testcase/></testsuite></testsuites>"
                )
                return 0

            pytest_main = mock.Mock(side_effect=run_pytest)
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "torch": SimpleNamespace(npu=fake_npu),
                        "torch_npu": ModuleType("torch_npu"),
                        "pytest": SimpleNamespace(main=pytest_main),
                    },
                ),
                mock.patch.object(
                    runner,
                    "load_candidate",
                    return_value=(installed, candidate, restore),
                ),
                mock.patch.object(
                    runner.importlib, "import_module", return_value=installed
                ),
                mock.patch.object(
                    runner.importlib.metadata, "version", return_value="test"
                ),
            ):
                code = runner.main(["--out", str(output)])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "PUBLIC_HOST_TESTS_PASS")
            self.assertEqual(
                Path(report["device_properties_helper"]).resolve(),
                Path(__file__).resolve(),
            )
            self.assertEqual(cached_properties.cache_info().currsize, 0)
            pytest_main.assert_called_once()
            restore.assert_called_once()

    def test_filtered_or_missing_test_files_cannot_pass_selection_audit(self):
        files = [ROOT / name for name in runner.TESTS]
        audit = runner.SelectionAudit(files)
        audit.pytest_collection_finish(
            SimpleNamespace(items=[SimpleNamespace(path=files[0])])
        )
        self.assertFalse(audit.complete)
        audit.pytest_collection_finish(
            SimpleNamespace(items=[SimpleNamespace(path=path) for path in files])
        )
        self.assertTrue(audit.complete)
        audit.pytest_deselected([object()])
        self.assertFalse(audit.complete)

    def test_missing_runtime_records_failure_without_false_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(Path(runner.__file__)),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(
                json.loads((output / "report.json").read_text())["status"], "FAILED"
            )
            self.assertTrue((output / "traceback.txt").is_file())
            self.assertNotIn("PUBLIC_HOST_TESTS_PASS", result.stdout)

    def test_module_override_and_restore_are_process_local(self):
        parent_name, child = runner.MODULE.rsplit(".", 1)
        parent = ModuleType(parent_name)
        installed = ModuleType(runner.MODULE)
        setattr(parent, child, installed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("MARKER = 'candidate'\n")
            with mock.patch.dict(
                sys.modules, {parent_name: parent, runner.MODULE: installed}
            ):
                original, candidate, restore = runner.load_candidate(path)
                self.assertIs(original, installed)
                self.assertIs(sys.modules[runner.MODULE], candidate)
                self.assertIs(getattr(parent, child), candidate)
                self.assertEqual(candidate.MARKER, "candidate")
                restore()
                self.assertIs(sys.modules[runner.MODULE], installed)
                self.assertIs(getattr(parent, child), installed)

    def test_failed_candidate_import_restores_installed_module(self):
        parent_name, child = runner.MODULE.rsplit(".", 1)
        parent, installed = ModuleType(parent_name), ModuleType(runner.MODULE)
        setattr(parent, child, installed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("raise RuntimeError('load failure')\n")
            with mock.patch.dict(
                sys.modules, {parent_name: parent, runner.MODULE: installed}
            ):
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    runner.load_candidate(path)
                self.assertIs(sys.modules[runner.MODULE], installed)
                self.assertIs(getattr(parent, child), installed)

    def test_junit_summary_exposes_skips_failures_and_empty_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            path.write_text(
                "<testsuites><testsuite><testcase/><testcase><skipped/></testcase><testcase><failure/></testcase></testsuite></testsuites>"
            )
            self.assertEqual(
                runner.summarize_junit(path), {"total": 3, "skipped": 1, "failed": 1}
            )
            path.write_text("<testsuites><testsuite/></testsuites>")
            self.assertEqual(
                runner.summarize_junit(path), {"total": 0, "skipped": 0, "failed": 0}
            )


if __name__ == "__main__":
    unittest.main()
