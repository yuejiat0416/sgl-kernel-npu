"""Perf benchmark for apply_attn_res (Ascend 910C, BF16).

Run on NPU host: python benchmark/bench_apply_attn_res.py
Reports kernel time (us, npu.Event device-side), speedup over PyTorch, bandwidth.
"""

import torch
import torch.nn as nn
import torch_npu

from sgl_kernel_npu.norm.apply_attn_res import apply_attn_res


def apply_attn_res_native(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Reference — mirrors modeling_kimi.py _apply_attn_res."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_f = v.float()
    variance = v_f.pow(2).mean(-1, keepdim=True)
    k = v_f * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_f).squeeze(1).to(v.dtype)


def _bench(fn, warmup=5, iters=30):
    """Device-side timing with npu.Event; returns min kernel time in MICROSECONDS."""
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


class _FakeNorm:
    def __init__(self, weight, eps):
        self.weight = weight
        self.variance_epsilon = eps


def bench_shape(N, B, H=7168):
    prefix_sum = torch.randn(N, H, dtype=torch.bfloat16).npu()
    block_residual = torch.randn(N, B, H, dtype=torch.bfloat16).npu()
    proj = nn.Linear(H, 1, bias=False).to(torch.bfloat16).npu()
    norm = _FakeNorm(torch.randn(H, dtype=torch.bfloat16).npu(), 1e-5)
    proj_w = proj.weight.detach().squeeze(0)

    t_kernel = _bench(lambda: apply_attn_res(prefix_sum, block_residual, proj, norm))
    t_native = _bench(lambda: apply_attn_res_native(prefix_sum, block_residual, proj_w, norm.weight, 1e-5))
    bytes_moved = N * (B + 1) * H * 2 + N * H * 2  # BF16 read (B+1 streams) + write (out)
    bw = bytes_moved / (t_kernel * 1e-6) / 1e9  # GB/s (t_kernel in us)
    print(
        f"[N={N:>5}, B={B}, H={H}] "
        f"kernel={t_kernel:8.1f} us  native={t_native:8.1f} us  "
        f"speedup={t_native / t_kernel:5.2f}x  bw={bw:7.1f} GB/s"
    )


if __name__ == "__main__":
    print("=== apply_attn_res BF16 (Ascend 910C) ===")
    bench_shape(N=4096, B=8)   # late-layer (B=8, 9 streams)
    bench_shape(N=4096, B=4)   # mid-layer
    bench_shape(N=4096, B=1)   # early-layer
    bench_shape(N=1, B=8)      # N=1: single decode token (max streams)
    bench_shape(N=32768, B=8)  # N=32768: large prefill
    bench_shape(N=4096, B=0)   # B=0: first layer (identity)
