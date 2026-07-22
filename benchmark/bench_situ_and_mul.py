"""Performance benchmark for SituAndMul (Ascend 910C, BF16).

Run on an NPU host:
    python benchmark/bench_situ_and_mul.py

Reports per-shape kernel time, speedup over the PyTorch reference, and achieved
memory bandwidth (the op is elementwise, so it is memory-bandwidth bound).
"""

import argparse
import time

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
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/op


def bench_shape(s, h, beta=4.0, linear_beta=25.0):
    # Full capacity: one expert holds all rows -> all rows are real.
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = torch.Tensor([s] + [0] * 15).npu().to(torch.int64)
    # bf16: read 2d per row, write d per row.
    bytes_moved = s * (h * 2 + (h // 2) * 2)

    t_native = _bench(lambda: situ_and_mul_native(x, beta, linear_beta))
    t_kernel = _bench(lambda: situ_and_mul(x, group_list, 1, beta, linear_beta))
    bw = bytes_moved / (t_kernel * 1e-3) / 1e9  # GB/s

    print(
        f"[s={s:>5}, h={h:>6}, d={h // 2:>6}] "
        f"kernel={t_kernel:8.3f} ms  native={t_native:8.3f} ms  "
        f"speedup={t_native / t_kernel:5.2f}x  bw={bw:7.1f} GB/s"
    )


# Partial utilization: ~157 real rows out of a much larger capacity (one token
# per expert block), the realistic MoE-dispatch case where many vector cores idle.
_PARTIAL_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]  # sum 157


def bench_partial(s, h, beta=4.0, linear_beta=25.0):
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = torch.Tensor(_PARTIAL_COUNTS).npu().to(torch.int64)
    real = sum(_PARTIAL_COUNTS)
    bytes_moved = real * (h * 2 + (h // 2) * 2)  # only real rows do work
    t_kernel = _bench(lambda: situ_and_mul(x, group_list, 1, beta, linear_beta))
    bw = bytes_moved / (t_kernel * 1e-3) / 1e9  # GB/s
    print(
        f"[partial s={s:>5}, real={real:>4}, h={h:>6}] "
        f"kernel={t_kernel:8.3f} ms  useful_bw={bw:7.1f} GB/s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--prod-h", type=int, default=3072 * 2)
    args = parser.parse_args()

    print("=== SituAndMul BF16 (Ascend 910C) ===")
    print("-- full capacity --")
    # Production-aligned: d = moe_intermediate_size = 3072 (input last-dim 6144).
    bench_shape(args.capacity, args.prod_h)  # prod: d = 3072 (default --prod-h 6144)
    bench_shape(2048, 8192)                  # nearby d = 4096
    bench_shape(512, 6144)                   # small capacity, prod d = 3072
    print("-- partial utilization (realistic MoE dispatch) --")
    bench_partial(args.capacity, args.prod_h)


if __name__ == "__main__":
    main()
