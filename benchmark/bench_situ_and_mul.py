"""Performance benchmark for SituAndMul (Ascend 910C, BF16).

Run on an NPU host:
    python benchmark/bench_situ_and_mul.py

Covers both call shapes: dense (group_list=None, KimiMLP layer0/shared) and routed
(group_list, MoE routed experts). Reports per-shape kernel time (us, NPU events),
speedup over the PyTorch reference, and achieved memory bandwidth (elementwise -> BW-bound).
"""

import argparse

import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul import situ_and_mul


def situ_and_mul_native(x, beta=4.0, linear_beta=25.0):
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def _bench(fn, warmup=5, iters=30):
    """Time ``fn`` on-device with NPU events; return the min kernel time (us)
    over ``iters`` runs after ``warmup``. Host wall-clock is not used."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # us, device-side
    return min(times)


def _report(tag, t_kernel, t_native, rows, h):
    # read 2d per row, write d per row (BF16).
    bytes_moved = rows * (h * 2 + (h // 2) * 2)
    bw = bytes_moved / (t_kernel * 1e-6) / 1e9  # GB/s (t_kernel in us)
    native_str = f"native={t_native:8.1f} us  speedup={t_native / t_kernel:5.2f}x" if t_native else "n/a"
    print(f"[{tag}] kernel={t_kernel:8.1f} us  {native_str}  bw={bw:7.1f} GB/s")


def bench_dense(N, d, beta=4.0, linear_beta=25.0):
    # group_list=None: dense KimiMLP (layer0 / shared) -- ALL rows processed.
    h = 2 * d
    x = torch.randn((N, h), dtype=torch.bfloat16).npu()
    t_native = _bench(lambda: situ_and_mul_native(x, beta, linear_beta))
    t_kernel = _bench(lambda: situ_and_mul(x))                       # dense path
    _report(f"dense   N={N:>5}, d={d:>6}", t_kernel, t_native, N, h)


def bench_shape(s, h, beta=4.0, linear_beta=25.0):
    # Routed, full capacity: one expert holds all rows -> all rows are real.
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = torch.Tensor([s] + [0] * 15).npu().to(torch.int64)
    t_native = _bench(lambda: situ_and_mul_native(x, beta, linear_beta))
    t_kernel = _bench(lambda: situ_and_mul(x, group_list, 1, beta, linear_beta))
    _report(f"routed  s={s:>5}, d={h // 2:>6}", t_kernel, t_native, s, h)


# Partial utilization: ~157 real rows out of a much larger capacity (realistic MoE dispatch
# where many vector cores idle). Useful-work BW only counts the real rows.
_PARTIAL_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]  # sum 157


def bench_partial(s, h, beta=4.0, linear_beta=25.0):
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = torch.Tensor(_PARTIAL_COUNTS).npu().to(torch.int64)
    real = sum(_PARTIAL_COUNTS)
    t_kernel = _bench(lambda: situ_and_mul(x, group_list, 1, beta, linear_beta))
    _report(f"partial s={s:>5}, real={real:>4}, d={h // 2:>6}", t_kernel, None, real, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--prod-h", type=int, default=3072 * 2)
    args = parser.parse_args()

    print("=== SituAndMul BF16 (Ascend 910C) ===")
    print("-- dense (group_list=None: KimiMLP layer0 / shared) --")
    bench_dense(1, 3072)        # decode, routed-sized d
    bench_dense(args.capacity, 3072)   # prefill, routed d (moe_intermediate_size)
    bench_dense(1, 6144)        # decode, shared-expert d (num_shared*moe_intermediate)
    bench_dense(args.capacity, 6144)   # prefill, shared d
    bench_dense(1, 33792)       # decode, dense layer0 d (H-tiled -- the UB case)
    bench_dense(512, 33792)     # prefill-ish, dense layer0 d (N capped: [N,67584] is big)

    print("-- routed (group_list: MoE routed experts), full capacity --")
    bench_shape(args.capacity, args.prod_h)  # prod: d = 3072
    bench_shape(1, args.prod_h)              # N=1: single decode token (launch overhead)
    bench_shape(32768, args.prod_h)          # N=32768: large prefill (bandwidth saturation)
    print("-- routed, partial utilization (realistic MoE dispatch) --")
    bench_partial(args.capacity, args.prod_h)


if __name__ == "__main__":
    main()
