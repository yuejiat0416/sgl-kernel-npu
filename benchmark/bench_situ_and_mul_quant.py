"""Bench: situ_and_mul_quant kernel (need_quant T/F) vs eager. Ascend 910C.
Grid: D in {3072, 6144, 33792} x N in {1, 8k, 32k, 128k}.
d<=6144: no_quant (BF16) + quant (int8) + eager (situ / situ+quant).
d=33792: no_quant (BF16) + eager (situ)."""
import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul_quant import situ_and_mul_quant

BETA = 4.0
LINEAR_BETA = 25.0


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
        times.append(s.elapsed_time(e) * 1000.0)
    return min(times)


def _native_situ(x, chunk=1024):
    x2 = x.reshape(-1, x.shape[-1])
    d = x2.shape[1] // 2
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    for i in range(0, x2.shape[0], chunk):
        gate = x2[i:i + chunk, :d].to(torch.float32)
        up = x2[i:i + chunk, d:].to(torch.float32)
        out[i:i + chunk] = (BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate) *
                             (LINEAR_BETA * torch.tanh(up / LINEAR_BETA))).to(x.dtype)
    return out.reshape(*x.shape[:-1], d)


def _native_situ_quant(x, chunk=1024):
    x2 = x.reshape(-1, x.shape[-1])
    N, two_d = x2.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=torch.int8, device=x.device)
    scale = torch.empty((N,), dtype=torch.float32, device=x.device)
    for i in range(0, N, chunk):
        gate = x2[i:i + chunk, :d].to(torch.float32)
        up = x2[i:i + chunk, d:].to(torch.float32)
        situ = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate) * (
            LINEAR_BETA * torch.tanh(up / LINEAR_BETA))
        s_row = torch.maximum(situ.abs().amax(-1, keepdim=True) / 127.0,
                              torch.tensor(1e-30, device=x.device))
        out[i:i + chunk] = torch.clamp(torch.floor(situ / s_row + 0.5), -128, 127).to(torch.int8)
        scale[i:i + chunk] = s_row.squeeze(-1)
    return out, scale


def bench(N, d):
    torch.npu.empty_cache()
    x = torch.randn((N, 2 * d), dtype=torch.bfloat16).npu()
    quant = d <= 6144
    t_nq = _bench(lambda: situ_and_mul_quant(x, need_quant=False))
    t_e_nq = _bench(lambda: _native_situ(x))
    if quant:
        t_q = _bench(lambda: situ_and_mul_quant(x, need_quant=True))
        t_e_q = _bench(lambda: _native_situ_quant(x))
        print(f"[d={d:>5} N={N:>6}] no_quant={t_nq:7.1f}us  quant={t_q:7.1f}us  "
              f"eager_situ={t_e_nq:8.1f}us  eager_quant={t_e_q:8.1f}us  "
              f"overhead={t_q / t_nq:4.2f}x  vs_eager={t_e_q / t_q:4.2f}x")
    else:
        print(f"[d={d:>5} N={N:>6}] no_quant={t_nq:7.1f}us  "
              f"eager_situ={t_e_nq:8.1f}us  vs_eager={t_e_nq / t_nq:4.2f}x")
    del x
    torch.npu.empty_cache()


def main():
    print("=== situ_and_mul_quant: kernel (need_quant T/F) vs eager (Ascend 910C) ===")
    for d in (3072, 6144):
        for N in (1, 8192, 32768, 128000):
            bench(N, d)
    for N in (1, 8192, 32768):  # d=33792: N=128K -> 17GB OOM, excluded
        bench(N, 33792)


if __name__ == "__main__":
    main()
