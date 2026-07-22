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


# 16 experts; count-format per-expert token counts (sum = 162 real rows).
_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
_REAL_TOKENS = sum(_COUNTS)
# Production shape: last dim = moe_intermediate_size * 2 = 3072 * 2 = 6144 (d = 3072).
PROD_H = 3072 * 2

# BF16: both kernel and reference compute in FP32 then round to BF16, so
# tolerances reflect BF16 rounding (~2^-8 mantissa). Tighten on NPU if easy.
RTOL, ATOL = 1e-2, 2e-2


class TestSituAndMulPrecision(unittest.TestCase):
    """Correctness vs the PyTorch reference across shapes and params."""

    def _check(self, h, s=4096, beta=4.0, linear_beta=25.0, counts=None):
        x = torch.randn((s, h), dtype=torch.bfloat16).npu()
        c = counts if counts is not None else _COUNTS
        group_list = torch.Tensor(c).npu().to(torch.int64)
        real = sum(c)

        out = situ_and_mul(x, group_list, 1, beta=beta, linear_beta=linear_beta)
        ref = situ_and_mul_native(x, beta=beta, linear_beta=linear_beta)
        torch.testing.assert_close(out[:real], ref[:real], rtol=RTOL, atol=ATOL)

    def test_prod_shape(self):
        # d = 3072 (moe_intermediate_size); input last-dim 6144.
        self._check(h=PROD_H)

    def test_small(self):
        self._check(h=8192)  # d = 4096

    def test_mid(self):
        self._check(h=3072)  # d = 1536

    def test_min_aligned(self):
        self._check(h=1024, s=512)  # d = 512

    def test_without_linear_beta(self):
        self._check(h=8192, linear_beta=None)  # DO_LINEAR_BETA = False

    def test_custom_beta(self):
        self._check(h=8192, beta=2.0, linear_beta=10.0)


class TestSituAndMulBoundary(unittest.TestCase):
    """Edge cases: token-count boundaries, dtypes, and input validation."""

    def _run(self, counts, h=8192, s=4096, beta=4.0, linear_beta=25.0):
        x = torch.randn((s, h), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(counts).npu().to(torch.int64)
        real = sum(counts)

        out = situ_and_mul(x, group_list, 1, beta=beta, linear_beta=linear_beta)

        # Output is always the full (capacity, d) shape; only `real` rows valid.
        self.assertEqual(out.shape, (s, h // 2))
        if real > 0:
            ref = situ_and_mul_native(x, beta=beta, linear_beta=linear_beta)
            torch.testing.assert_close(out[:real], ref[:real], rtol=RTOL, atol=ATOL)
        return out, real

    def test_zero_tokens(self):
        # All experts get 0 tokens -> kernel must no-op without crashing.
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
        # Host has a dedicated int32 alignment branch.
        x = torch.randn((4096, 8192), dtype=torch.bfloat16).npu()
        group_list = torch.Tensor(_COUNTS).npu().to(torch.int32)
        out = situ_and_mul(x, group_list, 1)
        ref = situ_and_mul_native(x)
        torch.testing.assert_close(
            out[:_REAL_TOKENS], ref[:_REAL_TOKENS], rtol=RTOL, atol=ATOL
        )

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

    # NOTE: group_list_type=0 (cusum) is not unit-tested here. Its buffer layout
    # (where the running total lives relative to num_experts) is a pipeline
    # contract; validate it against the real dispatch group_list once confirmed.


if __name__ == "__main__":
    unittest.main()
