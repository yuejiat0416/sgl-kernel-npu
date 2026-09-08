#!/usr/bin/env python3
"""Test this checkout's Python operator using the image's existing binary library."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

MODULE = "sgl_kernel_npu.norm.split_qkv_rmsnorm_rope"
RELATIVE_SOURCE = "python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py"
TESTS = (
    "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope.py",
    "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope_head192.py",
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
                    candidate.get_device_properties
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
        exit_code = int(
            pytest.main(
                [
                    *(str(root / name) for name in TESTS),
                    "-v",
                    "-ra",
                    f"--junitxml={output / 'pytest.xml'}",
                ],
                plugins=[selection],
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
    except (Exception, KeyboardInterrupt):
        report["status"] = "FAILED"
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
