import unittest

import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul import situ_and_mul


def situ_and_mul_native(x, beta=4.0, linear_beta=25.0):
    """Reference implementation (mirrors the source PyTorch SituAndMul)."""
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


# 16 experts; count-format per-expert token counts (sum = 157 real rows).
_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
_REAL_TOKENS = sum(_COUNTS)
PROD_H = 3072 * 2

# Tolerances per triton-ascend debug_guide/precision.md:
#   BF16 → rtol=atol=5e-3; FP32 → rtol=atol=1e-5; FP16 uses BF16 grade (doc
#   has no FP16 entry). equal_nan=True (doc default).
_TOL = {
    torch.bfloat16: (5e-3, 5e-3),
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-5, 1e-5),
}


class TestSituAndMulPrecision(unittest.TestCase):
    """Correctness vs the PyTorch reference across shapes, params, and dtypes."""

    def _check(self, h, s=4096, beta=4.0, linear_beta=25.0, counts=None,
               scale=1.0, dtype=torch.bfloat16):
        x = torch.randn((s, h), dtype=dtype).npu() * scale
        c = counts if counts is not None else _COUNTS
        group_list = torch.Tensor(c).npu().to(torch.int64)
        real = sum(c)

        out = situ_and_mul(x, group_list, 1, beta=beta, linear_beta=linear_beta)
        ref = situ_and_mul_native(x, beta=beta, linear_beta=linear_beta)
        rtol, atol = _TOL[dtype]
        torch.testing.assert_close(out[:real], ref[:real], rtol=rtol, atol=atol,
                                   equal_nan=True)

    # ---- BF16 (doc: rtol=atol=5e-3) ----
    def test_prod_shape_bf16(self):
        self._check(h=PROD_H, dtype=torch.bfloat16)

    def test_small_bf16(self):
        self._check(h=8192, dtype=torch.bfloat16)

    def test_mid_bf16(self):
        self._check(h=3072, dtype=torch.bfloat16)

    def test_without_linear_beta_bf16(self):
        self._check(h=8192, linear_beta=None, dtype=torch.bfloat16)

    def test_custom_beta_bf16(self):
        self._check(h=8192, beta=2.0, linear_beta=10.0, dtype=torch.bfloat16)

    def test_saturation_bf16(self):
        self._check(h=8192, scale=50.0, dtype=torch.bfloat16)

    def test_nd_input_bf16(self):
        B, s, h = 2, 256, 8192
        x = torch.randn((B, s, h), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int64)
        real = sum(_COUNTS)
        out = situ_and_mul(x, group_list, 1)
        self.assertEqual(out.shape, (B, s, h // 2))
        ref = situ_and_mul_native(x)
        rtol, atol = _TOL[torch.bfloat16]
        torch.testing.assert_close(out.reshape(-1, h // 2)[:real],
                                   ref.reshape(-1, h // 2)[:real],
                                   rtol=rtol, atol=atol, equal_nan=True)

    # ---- FP16 (BF16-grade tolerance; doc has no FP16 entry) ----
    def test_fp16(self):
        self._check(h=8192, dtype=torch.float16)

    # ---- FP32 (doc: rtol=atol=1e-5) ----
    def test_prod_shape_fp32(self):
        self._check(h=PROD_H, dtype=torch.float32)

    def test_saturation_fp32(self):
        self._check(h=8192, scale=50.0, dtype=torch.float32)


class TestSituAndMulBoundary(unittest.TestCase):
    """Edge cases: token-count boundaries, dtypes, and input validation."""

    def _run(self, counts, h=8192, s=4096, beta=4.0, linear_beta=25.0):
        x = torch.randn((s, h), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(counts).npu().to(torch.int64)
        real = sum(counts)
        out = situ_and_mul(x, group_list, 1, beta=beta, linear_beta=linear_beta)
        self.assertEqual(out.shape, (s, h // 2))
        if real > 0:
            ref = situ_and_mul_native(x, beta=beta, linear_beta=linear_beta)
            rtol, atol = _TOL[torch.bfloat16]
            torch.testing.assert_close(out[:real], ref[:real], rtol=rtol, atol=atol,
                                       equal_nan=True)
        return out, real

    def test_zero_tokens(self):
        out, real = self._run([0] * 16)
        self.assertEqual(real, 0)

    def test_single_token(self):
        out, real = self._run([1] + [0] * 15)
        self.assertEqual(real, 1)

    def test_full_capacity(self):
        s = 256
        out, real = self._run([s] + [0] * 15, s=s)
        self.assertEqual(real, s)

    def test_all_in_one_expert(self):
        out, real = self._run([0, 0, 200] + [0] * 13)
        self.assertEqual(real, 200)

    def test_group_list_int32(self):
        x = torch.randn((4096, 8192), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int32)
        out = situ_and_mul(x, group_list, 1)
        ref = situ_and_mul_native(x)
        rtol, atol = _TOL[torch.bfloat16]
        torch.testing.assert_close(out[:_REAL_TOKENS], ref[:_REAL_TOKENS],
                                   rtol=rtol, atol=atol, equal_nan=True)

    def test_invalid_group_list_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int64)
        with self.assertRaises(ValueError):
            situ_and_mul(x, group_list, 2)

    def test_invalid_group_list_dtype(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.float32)
        with self.assertRaises(ValueError):
            situ_and_mul(x, group_list, 1)

    def test_odd_last_dim(self):
        x = torch.randn((16, 8191), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int64)
        with self.assertRaises(ValueError):
            situ_and_mul(x, group_list, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
