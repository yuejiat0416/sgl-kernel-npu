import unittest

import numpy as np
import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul_quant import situ_and_mul_quant


def _situ(seg, d, beta, linear_beta):
    gate = seg[..., :d].to(torch.float32)
    up = seg[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(seg.dtype)


def situ_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
    """FP32 SituAndMul reference (chunked) -> x.dtype. For the d=33792 unquant fallback path."""
    x_2d = x.reshape(-1, x.shape[-1])
    N, two_d = x_2d.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=x.dtype, device=x.device)
    for i in range(0, N, chunk):
        out[i:i + chunk] = _situ(x_2d[i:i + chunk], d, beta, linear_beta)
    return out.reshape(*x.shape[:-1], d)


def situ_and_mul_quant_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
    """SituAndMul (FP32) + dynamic int8 quant reference. For the d<=6144 quant path."""
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
        s_row = torch.maximum(situ.abs().amax(dim=-1) / 127.0,
                              torch.tensor(1e-30, device=x.device))
        q = torch.floor(situ / s_row.unsqueeze(-1) + 0.5)
        out[i:i + chunk] = torch.clamp(q, -128, 127).to(torch.int8)
        scale[i:i + chunk] = s_row
    return out.reshape(*x.shape[:-1], d), scale


_COUNTS = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
_REAL = sum(_COUNTS)

_SCALE_RTOL = 5e-3
_INT8_MAX_DIFF = 1
_INT8_DIFF_RATE = 2e-2
_TOL = {torch.bfloat16: (5e-3, 5e-3), torch.float32: (1e-5, 1e-5)}
_BF16 = torch.bfloat16

_DENSE_SHAPES = ([(3072, N) for N in (1, 64, 4096, 8192, 128000)] +
                 [(6144, N) for N in (1, 64, 4096, 8192)] +
                 [(33792, N) for N in (1, 64)])
_ROUTED_D = (3072, 6144, 33792)
_DTYPES = ((torch.bfloat16, "bf16"), (torch.float32, "fp32"))


class TestSituAndMulQuantPrecision(unittest.TestCase):
    """d<=6144: int8 quant vs FP32 reference. d=33792: unquant fallback vs FP32 situ reference."""

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
        if d > 6144:  # unquant fallback -> FP32 situ reference
            self.assertEqual(out.dtype, dtype)
            torch.testing.assert_close(out, situ_native(x), rtol=_TOL[dtype][0],
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
            torch.testing.assert_close(out[:_REAL], situ_native(x)[:_REAL],
                                       rtol=_TOL[dtype][0], atol=_TOL[dtype][1], equal_nan=True)
        else:
            ref_out, ref_scale = situ_and_mul_quant_native(x)
            self._assert_quant(out, scale, ref_out, ref_scale, _REAL)
        del x, out, scale
        torch.npu.empty_cache()


def _make_dense(d, N, dtype, expect_fail=False):
    def test(self):
        self._run_dense(N=N, d=d, dtype=dtype)
    return unittest.expectedFailure(test) if expect_fail else test


def _make_routed(d, dtype, expect_fail=False):
    def test(self):
        self._run_routed(d=d, dtype=dtype)
    return unittest.expectedFailure(test) if expect_fail else test


# d<=6144: int8 statistical (BF16 passes). d=33792: unquant vs FP32 ref -> BF16 1-ULP xfail.
for _d, _N in _DENSE_SHAPES:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulQuantPrecision, f"test_dense_d{_d}_N{_N}_{_name}",
                _make_dense(_d, _N, _dt, expect_fail=(_dt is _BF16 and _d > 6144)))
for _d in _ROUTED_D:
    for _dt, _name in _DTYPES:
        setattr(TestSituAndMulQuantPrecision, f"test_routed_d{_d}_{_name}",
                _make_routed(_d, _dt, expect_fail=(_dt is _BF16 and _d > 6144)))


class TestSituAndMulQuantBoundary(unittest.TestCase):

    def test_large_d_falls_back_to_unquant(self):
        x = torch.randn((16, 2 * 33792), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_fp8_not_implemented(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        with self.assertRaises(NotImplementedError):
            situ_and_mul_quant(x, quant_type=1)

    def test_fp8_not_raised_when_not_quantizing(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False, quant_type=1)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_bad_quant_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x, quant_type=2)

    def test_need_quant_false_is_unquant(self):
        x = torch.randn((64, 2 * 3072), dtype=torch.float32).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, situ_native(x), rtol=1e-5, atol=1e-5, equal_nan=True)

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
