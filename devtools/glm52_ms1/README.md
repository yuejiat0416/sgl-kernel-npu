# Native 192 public-host validation

These tools belong to the temporary synchronization branch. The operator change
and its NPU tests are separate from the tools intended for local development.

Inside the existing CANN 9.1 / Triton-Ascend 3.2.2 test container, from this
repository's root, run:

```bash
python3 devtools/glm52_ms1/run_head192_tests.py
```

Use an available NPU; `--device 1` selects another visible device. This does not
load model weights or start distributed workers. Do not put the checkout's
`python/sgl_kernel_npu` directory on `PYTHONPATH` for this command: that source tree
does not contain the image's compiled binary library.

The runner first imports the installed package and its binary library. It then
loads this checkout's `split_qkv_rmsnorm_rope.py` in the current test process,
before pytest collects the existing operator tests and the new head-192 tests.
Other Python helpers and binary operators continue to come from the installed
package. No installed files are replaced; no package is installed or rebuilt.
Triton may compile new specializations into its normal cache.

Results are saved under `/home/tyj/glm52-ms1/evidence/head192-host-*`, or a new
directory supplied with `--out`. Inspect:

- `report.json`: source paths/hashes, Git commit, package versions, test counts,
  and exit status. Skipped or deselected tests, a missing test file, or an empty
  suite prevent a PASS result, including filters inherited through pytest options.
- `pytest.xml`: individual results, including failures and captured output.
- `installed_module.py` and `candidate_module.py`: the exact compared sources.
- `traceback.txt`: runner setup errors, if any.

The current suite includes the four existing cases and 43 additional cases.
These exercise public-host 192 execution, BF16/FP32 position caches,
changed-input graph replay, old dimensions, and representative existing operator
options with 192 heads: GQA, optional normalization, bias, partial and interleaved
RoPE, floating-point dtypes, and equivalent contiguous views.

The 192 path follows the existing operator's input contract. The earlier
192-only dtype/norm/RoPE restrictions and their rejection tests have been removed.
Callers still need to supply valid buffer sizes, layouts, and optional parameter
combinations, as they do for existing dimensions. This suite does not pass
malformed buffers or invalid device pointers to the kernel to test for rejection.
A pass does not establish model accuracy, distributed operation, or performance.
No performance threshold is introduced by this runner.

CPU-only checks for dispatch compatibility and the temporary module loader:

```bash
python3 -m unittest discover -s devtools/glm52_ms1 -p test_head192_tools.py -v
```

Those checks compare dispatch with Git baseline
`d974d3de5b7b0d6586a41f227cba93a861f07fe1`; they do not execute or emulate NPU
arithmetic. Keep the checkout's Git history available for that comparison.
