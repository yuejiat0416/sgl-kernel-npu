import unittest

import numpy as np
import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul import situ_and_mul
from sgl_kernel_npu.activation.situ_and_mul_quant import situ_and_mul_quant


def situ_and_mul_quant_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
    """Reference: SituAndMul (FP32) + dynamic int8 quant (scale=max/127, round-half-up, clamp).
    For the d<=6144 quant path. Chunked over tokens -> bounded memory."""
    x_2d = x.reshape(-1, x.shape[-1])
    N, two_d = x_2d.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=torch.int8, device=x.device)
    scale = torch.empty((N,), dtype=torch.float32, device=x.device)
    for i in range(0, N, chunk):
        seg = x_2d[i:i + chunk]
        gate = seg[:, :d].to(torch.float32)
        up = seg[:, d:].to(torch.float32)
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        situ = situ_a * up
        s_row = torch.maximum(                              # mirror kernel's 1e-30 div-zero floor
            situ.abs().amax(dim=-1) / 127.0,
            torch.tensor(1e-30, device=x.device),
        )
        q = torch.floor(situ / s_row.unsqueeze(-1) + 0.5)
        q = torch.clamp(q, -128, 127).to(torch.int8)
        out[i:i + chunk] = q
        scale[i:i + chunk] = s_row
    return out.reshape(*x.shape[:-1], d), scale


_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
_REAL = sum(_COUNTS)

_SCALE_RTOL = 5e-3
_INT8_MAX_DIFF = 1
_INT8_DIFF_RATE = 2e-2
_TOL = {torch.bfloat16: (5e-3, 5e-3), torch.float32: (1e-5, 1e-5)}

# d=3072 up to N=128000; d=6144 capped at N=8192 (N=128000 OOMs the FP32 ref+diff on shared NPU);
# d=33792 capped at N<=64 ([N,67584] big). d=33792 is the unquant-fallback path.
_DENSE_SHAPES = ([(3072, N) for N in (1, 64, 4096, 8192, 128000)] +
                 [(6144, N) for N in (1, 64, 4096, 8192)] +
                 [(33792, N) for N in (1, 64)])
_ROUTED_D = (3072, 6144, 33792)
_DTYPES = ((torch.bfloat16, "bf16"), (torch.float32, "fp32"))


class TestSituAndMulQuantPrecision(unittest.TestCase):
    """d<=6144: int8 quant vs FP32 reference. d=33792: unquant fallback vs situ_and_mul (same kernel)."""

    def _assert_quant(self, out, scale, ref_out, ref_scale, real):
        np.testing.assert_allclose(
            scale[:real].to(torch.float32).cpu().numpy(),
            ref_scale[:real].cpu().numpy(), rtol=_SCALE_RTOL)
        diff = (out[:real].to(torch.int32) - ref_out[:real].to(torch.int32)).abs().cpu().numpy()
        self.assertLessEqual(int(diff.max()), _INT8_MAX_DIFF)
        self.assertLess(float((diff > 0).mean()), _INT8_DIFF_RATE)
        self.assertGreaterEqual(int(out[:real].min()), -128)
        self.assertLessEqual(int(out[:real].max()), 127)

    def _run_dense(self, N, d, dtype):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        x = torch.randn((N, 2 * d), dtype=dtype).npu()
        out, scale = situ_and_mul_quant(x)
        if d > 6144:  # unquant fallback: same kernel as situ_and_mul -> bit-identical
            self.assertEqual(out.dtype, dtype)
            torch.testing.assert_close(out, situ_and_mul(x), rtol=_TOL[dtype][0],
                                       atol=_TOL[dtype][1], equal_nan=True)
        else:         # int8 quant
            ref_out, ref_scale = situ_and_mul_quant_native(x)
            self._assert_quant(out, scale, ref_out, ref_scale, N)
        del x, out, scale
        torch.npu.empty_cache()

    def _run_routed(self, d, dtype):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        s = 256
        x = torch.randn((s, 2 * d), dtype=dtype).npu()
        gl = torch.Tensor(_COUNTS).npu().to(torch.int64)
        out, scale = situ_and_mul_quant(x, gl, 1)
        if d > 6144:
            self.assertEqual(out.dtype, dtype)
            torch.testing.assert_close(out[:_REAL], situ_and_mul(x, gl, 1)[:_REAL],
                                       rtol=_TOL[dtype][0], atol=_TOL[dtype][1], equal_nan=True)
        else:
            ref_out, ref_scale = situ_and_mul_quant_native(x)
            self._assert_quant(out, scale, ref_out, ref_scale, _REAL)
        del x, out, scale
        torch.npu.empty_cache()


def _make_dense(d, N, dtype):
    def test(self):
        self._run_dense(N=N, d=d, dtype=dtype)
    return test


def _make_routed(d, dtype):
    def test(self):
        self._run_routed(d=d, dtype=dtype)
    return test


for _d, _N in _DENSE_SHAPES:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulQuantPrecision, f"test_dense_d{_d}_N{_N}_{_name}",
                _make_dense(_d, _N, _dt))
for _d in _ROUTED_D:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulQuantPrecision, f"test_routed_d{_d}_{_name}",
                _make_routed(_d, _dt))


class TestSituAndMulQuantBoundary(unittest.TestCase):

    def test_large_d_falls_back_to_unquant(self):
        # d=33792 + need_quant=True -> unquant (int8 not supported for d>6144).
        x = torch.randn((16, 2 * 33792), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x)
        self.assertEqual(out.dtype, torch.bfloat16)  # NOT int8

    def test_fp8_not_implemented(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        with self.assertRaises(NotImplementedError):
            situ_and_mul_quant(x, quant_type=1)

    def test_fp8_not_raised_when_not_quantizing(self):
        # need_quant=False + fp8 -> fp8 not attempted -> no NotImplementedError, unquant out.
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False, quant_type=1)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_bad_quant_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x, quant_type=2)

    def test_need_quant_false_is_unquant(self):
        x = torch.randn((64, 2 * 3072), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, situ_and_mul(x), rtol=5e-3, atol=5e-3, equal_nan=True)

    def test_all_zero_row(self):
        x = torch.zeros((4, 2 * 3072), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x)
        self.assertTrue(torch.all(out == 0).item())
        self.assertTrue(torch.all(scale > 0).item())

    def test_zero_tokens(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.zeros(16, dtype=torch.int64).npu()
        situ_and_mul_quant(x, gl, 1)

    def test_invalid_group_list_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.Tensor(_COUNTS).npu().to(torch.int64)
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x, gl, 2)

    def test_odd_last_dim(self):
        x = torch.randn((16, 8191), dtype=torch.bfloat16).npu()
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x)


if __name__ == "__main__":
    unittest.main()
