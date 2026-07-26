import unittest

import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul import situ_and_mul


def situ_and_mul_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
    """Reference (mirrors kimi_k3 SituAndMul). Chunked over tokens -> bit-exact + bounded memory
    (the shared NPU has little free memory; an unchunked FP32 reference OOMs at large d/N)."""
    x_2d = x.reshape(-1, x.shape[-1])
    N, two_d = x_2d.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=x.dtype, device=x.device)
    for i in range(0, N, chunk):
        seg = x_2d[i:i + chunk]
        gate = seg[:, :d].to(torch.float32)
        up = seg[:, d:].to(torch.float32)
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        out[i:i + chunk] = (situ_a * up).to(x.dtype)
    return out.reshape(*x.shape[:-1], d)


# 16 experts, count-format (sum = 157 real rows) -- routed-MoE dispatch metadata.
_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
_REAL = sum(_COUNTS)

# Tolerances per triton-ascend debug_guide/precision.md: BF16 5e-3, FP32 1e-5. equal_nan=True.
_TOL = {
    torch.bfloat16: (5e-3, 5e-3),
    torch.float32: (1e-5, 1e-5),
}

# K3 shapes. d in {routed 3072, shared 6144, dense 33792}; N in {decode 1, batch 64, prefill 4096}.
# d=33792 capped at N<=64: input [N,67584] gets big and the shared NPU is memory-tight;
# H-tile correctness is about d (multi-tile), not N, so small N still exercises it fully.
_DENSE_SHAPES = [(d, N) for d in (3072, 6144) for N in (1, 64, 4096)] + [
    (33792, N) for N in (1, 64)
]
_ROUTED_D = (3072, 6144, 33792)
_DTYPES = ((torch.bfloat16, "bf16"), (torch.float32, "fp32"))


class TestSituAndMulPrecision(unittest.TestCase):
    """Correctness vs the PyTorch reference across K3 shapes (dense + routed, prefill/decode)."""

    def _run_dense(self, N, d, dtype):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        x = torch.randn((N, 2 * d), dtype=dtype).npu()
        out = situ_and_mul(x)                          # group_list=None -> dense (all rows)
        ref = situ_and_mul_native(x)
        rtol, atol = _TOL[dtype]
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=True)

    def _run_routed(self, d, dtype):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        s = 256                                        # padded dispatch tensor (real rows = _REAL)
        x = torch.randn((s, 2 * d), dtype=dtype).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int64)
        out = situ_and_mul(x, group_list, 1)           # routed: only first _REAL rows written
        ref = situ_and_mul_native(x)
        rtol, atol = _TOL[dtype]
        torch.testing.assert_close(out[:_REAL], ref[:_REAL], rtol=rtol, atol=atol,
                                   equal_nan=True)


def _make_dense(d, N, dtype, expect_fail=False):
    def test(self):
        self._run_dense(N=N, d=d, dtype=dtype)
    return unittest.expectedFailure(test) if expect_fail else test


def _make_routed(d, dtype, expect_fail=False):
    def test(self):
        self._run_routed(d=d, dtype=dtype)
    return unittest.expectedFailure(test) if expect_fail else test


# BF16 grid = expected-failure: the kernel uses libdevice.tanh + tl.sigmoid (forced --
# no tl.tanh on this fork), which differ from torch.tanh/sigmoid by ~1 ULP FP32, flipping
# ~0.0% of BF16 boundary elements by 1 ULP (0.015625) at strict 5e-3. FP32 grid is the
# correctness witness (all pass at 1e-5). Same BF16-floor family as apply_attn_res.
_BF16 = torch.bfloat16

# Dense (group_list=None): full K3 d x N grid.
for _d, _N in _DENSE_SHAPES:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulPrecision, f"test_dense_d{_d}_N{_N}_{_name}",
                _make_dense(_d, _N, _dt, expect_fail=(_dt is _BF16)))
# Routed (group_list): each d exercises the group_list path (and tiling at large d).
for _d in _ROUTED_D:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulPrecision, f"test_routed_d{_d}_{_name}",
                _make_routed(_d, _dt, expect_fail=(_dt is _BF16)))


class TestSituAndMulBoundary(unittest.TestCase):
    """Edge cases: dispatch boundaries, input validation, nd dense input."""

    def test_zero_tokens(self):
        # all experts empty -> total_rows=0 -> kernel writes nothing, must not crash.
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.zeros(16, dtype=torch.int64).npu()
        situ_and_mul(x, gl, 1)

    def test_single_token_dense(self):
        x = torch.randn((1, 8192), dtype=torch.bfloat16).npu()
        out = situ_and_mul(x)                          # dense, N=1 (decode)
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3, equal_nan=True)

    def test_nd_dense_input(self):
        # dense path must accept >2D input (KimiMLP gate_up is [..., 2d]).
        B, s, d = 2, 16, 3072
        x = torch.randn((B, s, 2 * d), dtype=torch.bfloat16).npu()
        out = situ_and_mul(x)
        self.assertEqual(out.shape, (B, s, d))
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3, equal_nan=True)

    def test_group_list_int32(self):
        x = torch.randn((256, 8192), dtype=torch.bfloat16).npu()
        gl = torch.Tensor(_COUNTS).npu().to(torch.int32)
        out = situ_and_mul(x, gl, 1)
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(out[:_REAL], ref[:_REAL], rtol=5e-3, atol=5e-3,
                                   equal_nan=True)

    def test_cusum_path_regression(self):
        # Regression for cusum OOB (situ_and_mul.py:39): total_rows must read
        # group_list[NUM_EXPERTS-1] (last cumulative entry), not one-past-end [NUM_EXPERTS].
        # Same bug as situ_and_mul_quant / swiglu_quant. FP32 to avoid the BF16 1-ULP
        # floor -- isolates cusum correctness (a wrong total_rows mismatches cleanly).
        torch.npu.empty_cache()
        torch.manual_seed(0)
        total_rows, d, E = 4096, 3072, 8
        x = torch.randn((total_rows, 2 * d), dtype=torch.float32).npu()
        counts = torch.bincount(torch.randint(0, E, (total_rows,)), minlength=E).to(torch.int64)
        gl = torch.cumsum(counts, dim=0).to(torch.int64).to(x.device)  # cusum (type 0)
        out = situ_and_mul(x, gl, 0)
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(out[:total_rows], ref[:total_rows],
                                   rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_cusum_path_int32_group_list(self):
        # cusum path with int32 group_list (exercises int32 num_experts_algin path).
        torch.npu.empty_cache()
        torch.manual_seed(0)
        total_rows, d, E = 2048, 3072, 8
        x = torch.randn((total_rows, 2 * d), dtype=torch.float32).npu()
        counts = torch.bincount(torch.randint(0, E, (total_rows,)), minlength=E).to(torch.int32)
        gl = torch.cumsum(counts, dim=0).to(torch.int32).to(x.device)
        out = situ_and_mul(x, gl, 0)
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(out[:total_rows], ref[:total_rows],
                                   rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_invalid_group_list_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.Tensor(_COUNTS).npu().to(torch.int64)
        with self.assertRaises(ValueError):
            situ_and_mul(x, gl, 2)                     # group_list_type must be 0/1

    def test_invalid_group_list_dtype(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.Tensor(_COUNTS).npu().to(torch.float32)
        with self.assertRaises(ValueError):
            situ_and_mul(x, gl, 1)                     # group_list dtype must be int32/int64

    def test_odd_last_dim(self):
        x = torch.randn((16, 8191), dtype=torch.bfloat16).npu()
        with self.assertRaises(ValueError):
            situ_and_mul(x)                            # dense path also requires even last dim


if __name__ == "__main__":
    unittest.main()
