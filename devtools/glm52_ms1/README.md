# Native head-dimension validation tools

These temporary tools belong only to the synchronization branch. Production
operator changes and formal NPU tests are committed separately.

The single execution guide is the [complete operation manual](https://github.com/yuejiat0416/sglang/blob/sync/glm52-dspark-ms1/devtools/glm52_ms1/README.md).
Its September 24 operator section contains the confirmed A3 command, input
matrix, timing settings, output locations, and interpretation. The local copy is
`/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/devtools/glm52_ms1/README.md`.

`run_head192_tests.py` keeps its existing name and correctness-only default.
With `--benchmark`, it also compares exact test inputs against native Torch on
the NPU and the preceding PR implementation. Dimensions are 64, 96, 128, 160,
192, 256, and 384. The suite contains 143 correctness cases: 142 exercise the
requested operator and receive eager/graph timing; one separate Gemma case
provides correctness regression only. Existing option and changed-input graph
tests remain in the suite.

The runner imports the installed binary package, then selects this checkout's
Python operator only within the test process. It does not install packages,
replace image files, or change the checkout. Triton may compile specializations
into its normal cache. Baseline source comes from existing sync history and is
verified against the exact operator blob in PR commit `9bc1ac4`.

Timing uses 10 warmup calls and five rounds of 30 calls, reporting all five
averages and their median. It measures synchronized public-call wall time,
excluding compilation, graph capture, CPU reference calculation, and output
validation. Graph timing calls `graph.replay`. No performance acceptance
threshold has been supplied; `MEASURED` is not a performance PASS.

Skipped/deselected tests, missing files, an empty suite, accuracy failures, and
incomplete required timing prevent an overall PASS. Unsupported old-PR
dimensions are explicitly recorded, while candidate and native Torch still run.
The results directory retains input metadata, exact sources, numerical errors,
timings, JUnit output, and setup tracebacks where applicable.

`test_head192_tools.py` contains CPU checks for dispatch, source loading, test
selection, timing accounting, and Torch reference math. These checks do not
execute or emulate NPU arithmetic. Keep Git history available for baseline
comparisons. They do not establish A3 compilation or numerical correctness.
