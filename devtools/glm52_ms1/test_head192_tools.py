"""CPU-only dispatch and test-loader checks; no NPU computation is simulated."""

import ast
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
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

    def test_other_non_power_of_two_dimensions_still_rejected(self):
        for dim in (96, 160, 384):
            for source in (self.before, self.after):
                with self.subTest(dim=dim, baseline=source is self.before):
                    host, _ = host_from_source(source)
                    args, kwargs = inputs(dim)
                    with self.assertRaises(AssertionError):
                        host(*args, **kwargs)

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


class RunnerTests(unittest.TestCase):
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
