"""Bench: situ_and_mul (no-quant) vs situ_and_mul_quant vs eager. Ascend 910C.
situ_and_mul_quant does int8 quant for d<=6144, unquant fallback for d=33792.
N sweep {1,4k,8k,32k,128k} for dense (d=3072/6144/33792) and routed (d=3072/6144)."""
import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul import situ_and_mul
from sgl_kernel_npu.activation.situ_and_mul_quant import situ_and_mul_quant

BETA = 4.0
LINEAR_BETA = 25.0
N_SWEEP = (1, 4096, 8192, 32768, 128000)


def _bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # us
    return min(times)


def _situ(seg, d):
    gate = seg[..., :d].to(torch.float32)
    up = seg[..., d:].to(torch.float32)
    return BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate) * (
        LINEAR_BETA * torch.tanh(up / LINEAR_BETA))


def _native_situ(x, chunk=1024):
    # eager situ (BF16 out) -- baseline for the d=33792 unquant path
    x2 = x.reshape(-1, x.shape[-1])
    d = x2.shape[1] // 2
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    for i in range(0, x2.shape[0], chunk):
        out[i:i + chunk] = _situ(x2[i:i + chunk], d).to(x.dtype)
    return out.reshape(*x.shape[:-1], d)


def _native_situ_quant(x, chunk=1024):
    # eager situ + int8 quant -- baseline for the d<=6144 quant path
    x2 = x.reshape(-1, x.shape[-1])
    N, two_d = x2.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=torch.int8, device=x.device)
    scale = torch.empty((N,), dtype=torch.float32, device=x.device)
    for i in range(0, N, chunk):
        situ = _situ(x2[i:i + chunk], d)
        s_row = torch.maximum(situ.abs().amax(-1, keepdim=True) / 127.0,
                              torch.tensor(1e-30, device=x.device))
        out[i:i + chunk] = torch.clamp(torch.floor(situ / s_row + 0.5), -128, 127).to(torch.int8)
        scale[i:i + chunk] = s_row.squeeze(-1)
    return out, scale


def _bw(rows, d, us, quant):
    if quant:  # read[N,2d]BF16 + write[N,d]int8 + [N]fp32 (full-row load: 1 read)
        bytes_ = rows * (2 * d * 2 + d * 1 + 4)
    else:      # unquant: read[N,2d]BF16 + write[N,d]BF16
        bytes_ = rows * (2 * d * 2 + d * 2)
    return bytes_ / (us * 1e-6) / 1e9


def _cmp(tag, t_nq, t_q, t_e, rows, d, quant):
    print(f"[{tag}] no_quant={t_nq:8.1f}us  quant={t_q:8.1f}us  eager={t_e:9.1f}us  "
          f"overhead={t_q / t_nq:4.2f}x  vs_eager={t_e / t_q:4.2f}x  "
          f"bw={_bw(rows, d, t_q, quant):7.1f}GB/s")


def bench_dense(N, d):
    torch.npu.empty_cache()
    x = torch.randn((N, 2 * d), dtype=torch.bfloat16).npu()
    quant = d <= 6144
    t_nq = _bench(lambda: situ_and_mul(x))
    t_q = _bench(lambda: situ_and_mul_quant(x))
    t_e = _bench(lambda: _native_situ_quant(x) if quant else _native_situ(x))
    _cmp(f"dense   N={N:>6}, d={d:>5}", t_nq, t_q, t_e, N, d, quant)
    del x
    torch.npu.empty_cache()


def bench_routed_full(s, d):
    # full capacity: one expert holds all rows -> all s rows real
    torch.npu.empty_cache()
    x = torch.randn((s, 2 * d), dtype=torch.bfloat16).npu()
    gl = torch.Tensor([s] + [0] * 15).npu().to(torch.int64)
    quant = d <= 6144
    t_nq = _bench(lambda: situ_and_mul(x, gl, 1))
    t_q = _bench(lambda: situ_and_mul_quant(x, gl, 1))
    t_e = _bench(lambda: _native_situ_quant(x) if quant else _native_situ(x))
    _cmp(f"routed  s={s:>6}, d={d:>5}", t_nq, t_q, t_e, s, d, quant)
    del x
    torch.npu.empty_cache()


def _try(fn, *args):
    try:
        fn(*args)
    except torch.OutOfMemoryError:
        torch.npu.empty_cache()
        print(f"[{fn.__name__} {args}] OOM (skipped) -- tensor > shared NPU free mem")


def main():
    print("=== situ_and_mul (no-quant) vs situ_and_mul_quant vs eager (Ascend 910C) ===")
    print("    (situ_and_mul_quant: int8 for d<=6144, unquant fallback for d=33792)")
    print("-- dense (d=3072/6144/33792) --")
    for d in (3072, 6144, 33792):
        for N in N_SWEEP:
            _try(bench_dense, N, d)
    print("-- routed full capacity (d=3072/6144) --")
    for d in (3072, 6144):
        for s in N_SWEEP:
            _try(bench_routed_full, s, d)


if __name__ == "__main__":
    main()
