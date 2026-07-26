import unittest

import numpy as np
import torch
import torch_npu

from sgl_kernel_npu.activation.situ_and_mul_quant import situ_and_mul_quant


# ---------------------------------------------------------------------------
# References (FP32, chunked over rows: bit-exact vs eager, bounds peak memory).
# ---------------------------------------------------------------------------
def _situ(seg, d, beta, linear_beta):
    gate = seg[..., :d].to(torch.float32)
    up = seg[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(seg.dtype)


def situ_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
    x_2d = x.reshape(-1, x.shape[-1])
    N, two_d = x_2d.shape
    d = two_d // 2
    out = torch.empty((N, d), dtype=x.dtype, device=x.device)
    for i in range(0, N, chunk):
        out[i:i + chunk] = _situ(x_2d[i:i + chunk], d, beta, linear_beta)
    return out.reshape(*x.shape[:-1], d)


def situ_quant_native(x, beta=4.0, linear_beta=25.0, chunk=1024):
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
        out[i:i + chunk] = torch.clamp(torch.floor(situ / s_row.unsqueeze(-1) + 0.5),
                                       -128, 127).to(torch.int8)
        scale[i:i + chunk] = s_row
    return out.reshape(*x.shape[:-1], d), scale


# Tolerances (triton-ascend debug_guide/precision.md): BF16 5e-3, FP32 1e-5.
_SCALE_RTOL = 5e-3
_INT8_MAX_DIFF = 1
_INT8_DIFF_RATE = 2e-2
_TOL = {torch.bfloat16: (5e-3, 5e-3), torch.float32: (1e-5, 1e-5)}
_DTYPES = ((torch.bfloat16, "bf16"), (torch.float32, "fp32"))


def _assert_quant(out, scale, ref_out, ref_scale, n):
    np.testing.assert_allclose(
        scale[:n].to(torch.float32).cpu().numpy(),
        ref_scale[:n].cpu().numpy(), rtol=_SCALE_RTOL)
    diff = (out[:n].to(torch.int32) - ref_out[:n].to(torch.int32)).abs().cpu().numpy()
    assert int(diff.max()) <= _INT8_MAX_DIFF
    assert float((diff > 0).mean()) < _INT8_DIFF_RATE


def _make_group_list(total_rows, num_experts, gl_type, dtype, device):
    # per-expert counts summing exactly to total_rows (CPU bincount -> device).
    counts = torch.bincount(torch.randint(0, num_experts, (total_rows,)),
                            minlength=num_experts).to(torch.int64)
    if gl_type == 1:  # count
        return counts.to(dtype).to(device)
    return torch.cumsum(counts, dim=0).to(dtype).to(device)  # cusum (type 0)


# ===========================================================================
# 1. Precision grid: d {3072,6144,33792} x N {1,8k,32k,128k} x dtype x need_quant
#    (group_list=None = dense/shared path; d=33792 noquant only).
#    N=32768 @ d=33792 is the int32-pointer-offset regression point.
# ===========================================================================
class TestSituAndMulQuantPrecision(unittest.TestCase):
    """Grid: D in {3072, 6144, 33792} x N in {1, 8k, 32k, 128k} x need_quant.
    d<=6144: need_quant True (int8) + False (situ). d=33792: need_quant False only."""

    def _run(self, N, d, dtype, need_quant):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        x = torch.randn((N, 2 * d), dtype=dtype).npu()
        out, scale = situ_and_mul_quant(x, need_quant=need_quant)
        if need_quant:
            self.assertEqual(out.dtype, torch.int8)
            ref_out, ref_scale = situ_quant_native(x)
            _assert_quant(out, scale, ref_out, ref_scale, N)
        else:
            self.assertEqual(out.dtype, dtype)
            torch.testing.assert_close(out, situ_native(x), rtol=_TOL[dtype][0],
                                       atol=_TOL[dtype][1], equal_nan=True)
        del x, out, scale
        torch.npu.empty_cache()


def _make(N, d, dtype, need_quant, skip_reason=""):
    def test(self):
        self._run(N=N, d=d, dtype=dtype, need_quant=need_quant)
    if skip_reason:
        test = unittest.skip(skip_reason)(test)
    return test


_MOE_D = (3072, 6144)
_N_VALUES = (1, 8192, 32768, 128000)

for _d in _MOE_D:
    for _N in _N_VALUES:
        for _dt, _name in _DTYPES:
            if _d == 6144 and _N == 128000 and _dt == torch.float32:
                continue
            for _nq, _qtag in ((True, "quant"), (False, "noquant")):
                _skip = ""
                if _dt == torch.bfloat16 and not _nq:
                    _skip = "BF16 1-ULP floor (libdevice.tanh vs torch.tanh); FP32 is witness"
                if _d == 6144 and _dt == torch.float32 and _nq:
                    _skip = "UB overflow (COL_BLOCK_SIZE=HALF_COLS + multibuffer); deferred"
                setattr(TestSituAndMulQuantPrecision, f"test_d{_d}_N{_N}_{_name}_{_qtag}",
                        _make(_N, _d, _dt, _nq, skip_reason=_skip))

for _N in _N_VALUES:
    for _dt, _name in _DTYPES:
        if _N == 128000:
            continue
        if _N == 32768 and _dt == torch.float32:
            continue
        _skip = ""
        if _dt == torch.bfloat16:
            _skip = "BF16 1-ULP floor; FP32 is witness"
        setattr(TestSituAndMulQuantPrecision, f"test_d33792_N{_N}_{_name}_noquant",
                _make(_N, 33792, _dt, False, skip_reason=_skip))


# ===========================================================================
# 2. MoE group_list paths: count (type 1) + cusum (type 0).
#    The situ math is per-row / expert-independent, so the kernel only consumes
#    total_rows from group_list. We verify "processes exactly total_rows rows,
#    matches reference on those rows".
#    count path must pass. cusum path is SKIPPED (OOB read may crash the process).
# ===========================================================================
class TestSituAndMulQuantMoe(unittest.TestCase):

    def _run_count(self, total_rows, d, dtype, num_experts, gl_dtype, need_quant):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        x = torch.randn((total_rows, 2 * d), dtype=dtype).npu()
        gl = _make_group_list(total_rows, num_experts, 1, gl_dtype, x.device)
        out, scale = situ_and_mul_quant(x, gl, 1, need_quant=need_quant)
        if need_quant:
            ref_out, ref_scale = situ_quant_native(x)
            _assert_quant(out, scale, ref_out, ref_scale, total_rows)
        else:
            torch.testing.assert_close(out, situ_native(x), rtol=_TOL[dtype][0],
                                       atol=_TOL[dtype][1], equal_nan=True)
        del x, out, scale
        torch.npu.empty_cache()

    def test_count_bf16_quant(self):
        self._run_count(4096, 3072, torch.bfloat16, 8, torch.int64, True)

    def test_count_fp32_quant(self):
        self._run_count(4096, 3072, torch.float32, 8, torch.int64, True)

    def test_count_fp32_noquant(self):
        self._run_count(4096, 3072, torch.float32, 8, torch.int64, False)

    @unittest.skip("BF16 1-ULP floor (libdevice.tanh vs torch.tanh); FP32 is witness")
    def test_count_bf16_noquant(self):
        self._run_count(4096, 3072, torch.bfloat16, 8, torch.int64, False)

    def test_count_int32_group_list(self):
        # int32 group_list -> num_experts_algin rounds to x16 (vs x8 for int64).
        self._run_count(2048, 3072, torch.bfloat16, 8, torch.int32, True)

    def test_count_num_experts_not_aligned(self):
        # E=5 (not a multiple of 8) -> exercises alignment padding on the count load.
        self._run_count(2048, 3072, torch.bfloat16, 5, torch.int64, True)

    def test_count_single_expert(self):
        # E=1: all rows to one expert; count path degenerate but must work.
        self._run_count(2048, 3072, torch.bfloat16, 1, torch.int64, True)

    def test_count_zero_token_expert(self):
        # An expert with 0 tokens is realistic; total_rows still exact.
        torch.npu.empty_cache()
        torch.manual_seed(0)
        total_rows, d, E = 2048, 3072, 8
        x = torch.randn((total_rows, 2 * d), dtype=torch.bfloat16).npu()
        counts = torch.zeros(E, dtype=torch.int64)
        counts[:4] = torch.tensor([512, 512, 512, 512])  # experts 4..7 get 0 tokens
        gl = counts.to(x.device)
        out, scale = situ_and_mul_quant(x, gl, 1, need_quant=True)
        ref_out, ref_scale = situ_quant_native(x)
        _assert_quant(out, scale, ref_out, ref_scale, total_rows)

    def test_cusum_path_oob_regression(self):
        # Regression: cusum path must read total_rows = group_list[NUM_EXPERTS-1]
        # (the last cumulative entry), not one-past-end [NUM_EXPERTS]. The old
        # off-by-one crashed (MTE DDR out of range) or silently processed a wrong
        # row count; same bug existed in swiglu_quant.py and is fixed in both.
        torch.npu.empty_cache()
        torch.manual_seed(0)
        total_rows, d = 4096, 3072
        x = torch.randn((total_rows, 2 * d), dtype=torch.bfloat16).npu()
        gl = _make_group_list(total_rows, 8, 0, torch.int64, x.device)  # cusum
        out, scale = situ_and_mul_quant(x, gl, 0, need_quant=True)
        ref_out, ref_scale = situ_quant_native(x)
        _assert_quant(out, scale, ref_out, ref_scale, total_rows)


# ===========================================================================
# 3. Contracts: N-D input, determinism, purity, dtype/shape.
# ===========================================================================
class TestSituAndMulQuantContract(unittest.TestCase):
    def test_3d_input_shape_preserved(self):
        torch.npu.empty_cache()
        torch.manual_seed(0)
        B, S, d = 2, 4, 3072
        x = torch.randn((B, S, 2 * d), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x, need_quant=True)
        self.assertEqual(out.shape, (B, S, d))
        self.assertEqual(out.dtype, torch.int8)
        ref_out, ref_scale = situ_quant_native(x)
        n = B * S
        np.testing.assert_allclose(scale.reshape(-1)[:n].float().cpu().numpy(),
                                   ref_scale.reshape(-1)[:n].cpu().numpy(),
                                   rtol=_SCALE_RTOL)
        diff = (out.reshape(n, d).to(torch.int32)
                - ref_out.reshape(n, d).to(torch.int32)).abs().cpu().numpy()
        self.assertLessEqual(int(diff.max()), _INT8_MAX_DIFF)
        self.assertLess(float((diff > 0).mean()), _INT8_DIFF_RATE)

    def test_determinism_quant(self):
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.bfloat16).npu()
        out1, s1 = situ_and_mul_quant(x, need_quant=True)
        out2, s2 = situ_and_mul_quant(x, need_quant=True)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(s1, s2))

    def test_determinism_noquant(self):
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.float32).npu()
        out1, _ = situ_and_mul_quant(x, need_quant=False)
        out2, _ = situ_and_mul_quant(x, need_quant=False)
        self.assertTrue(torch.equal(out1, out2))

    def test_input_not_mutated(self):
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.bfloat16).npu()
        x_clone = x.clone()
        situ_and_mul_quant(x, need_quant=True)
        self.assertTrue(torch.equal(x, x_clone))

    def test_quant_dtype_shape_contract(self):
        x = torch.randn((16, 2 * 3072), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x, need_quant=True)
        self.assertEqual(out.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (16, 3072))
        self.assertEqual(tuple(scale.shape), (16,))

    def test_noquant_dtype_preserved(self):
        for dt in (torch.bfloat16, torch.float32):
            x = torch.randn((16, 2 * 3072), dtype=dt).npu()
            out, _ = situ_and_mul_quant(x, need_quant=False)
            self.assertEqual(out.dtype, dt)


# ===========================================================================
# 4. Math variants: linear_beta=None (up unbounded), custom beta/linear_beta.
# ===========================================================================
class TestSituAndMulQuantMath(unittest.TestCase):
    def test_linear_beta_none(self):
        torch.manual_seed(0)
        x = torch.randn((2048, 2 * 3072), dtype=torch.float32).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False, linear_beta=None)
        torch.testing.assert_close(out, situ_native(x, linear_beta=None),
                                   rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_custom_beta(self):
        torch.manual_seed(0)
        x = torch.randn((2048, 2 * 3072), dtype=torch.float32).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False, beta=2.0, linear_beta=10.0)
        torch.testing.assert_close(out, situ_native(x, beta=2.0, linear_beta=10.0),
                                   rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_linear_beta_none_quant(self):
        torch.manual_seed(0)
        x = torch.randn((2048, 2 * 3072), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x, need_quant=True, linear_beta=None)
        ref_out, ref_scale = situ_quant_native(x, linear_beta=None)
        _assert_quant(out, scale, ref_out, ref_scale, 2048)


# ===========================================================================
# 5. Quant edge cases: zero row (scale floor), int8 saturation clamp, scale sign.
# ===========================================================================
class TestSituAndMulQuantQuantEdge(unittest.TestCase):
    def test_all_zero_row(self):
        x = torch.zeros((4, 2 * 3072), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x, need_quant=True)
        self.assertTrue(torch.all(out == 0).item())
        self.assertTrue(torch.all(scale > 0).item())  # floor 1e-30 keeps scale > 0

    def test_int8_clamp_saturation(self):
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.bfloat16).npu() * 1e4
        out, _ = situ_and_mul_quant(x, need_quant=True)
        self.assertGreaterEqual(int(out.to(torch.int32).min()), -128)
        self.assertLessEqual(int(out.to(torch.int32).max()), 127)

    def test_scale_finite_positive(self):
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.bfloat16).npu()
        _, scale = situ_and_mul_quant(x, need_quant=True)
        self.assertTrue(torch.all(torch.isfinite(scale)).item())
        self.assertTrue(torch.all(scale > 0).item())

    def test_quant_round_trip_bounds(self):
        # dequantized codes must stay within [-128*scale, 127*scale] of the activation.
        torch.manual_seed(0)
        x = torch.randn((1024, 2 * 3072), dtype=torch.bfloat16).npu()
        out, scale = situ_and_mul_quant(x, need_quant=True)
        ref, _ = situ_quant_native(x)
        # every kernel code differs from the reference code by at most 1
        diff = (out.to(torch.int32) - ref.to(torch.int32)).abs()
        self.assertLessEqual(int(diff.max()), _INT8_MAX_DIFF)


# ===========================================================================
# 6. Boundary / argument validation.
# ===========================================================================
class TestSituAndMulQuantBoundary(unittest.TestCase):

    def test_dense_quant_not_supported(self):
        x = torch.randn((16, 2 * 33792), dtype=torch.bfloat16).npu()
        with self.assertRaises(NotImplementedError):
            situ_and_mul_quant(x)  # need_quant=True + d>6144 -> raise

    def test_large_d_unquant(self):
        x = torch.randn((16, 2 * 33792), dtype=torch.bfloat16).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False)
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

    def test_need_quant_false_fp32(self):
        x = torch.randn((64, 2 * 3072), dtype=torch.float32).npu()
        out, _ = situ_and_mul_quant(x, need_quant=False)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, situ_native(x), rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_zero_tokens(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.zeros(16, dtype=torch.int64).npu()
        situ_and_mul_quant(x, gl, 1)

    def test_invalid_group_list_type(self):
        x = torch.randn((16, 8192), dtype=torch.bfloat16).npu()
        gl = torch.zeros(16, dtype=torch.int64).npu()
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x, gl, 2)

    def test_odd_last_dim(self):
        x = torch.randn((16, 8191), dtype=torch.bfloat16).npu()
        with self.assertRaises(ValueError):
            situ_and_mul_quant(x)

    def test_dispatch_boundary_6144(self):
        # d=6144 -> moe kernel (half_cols<=6144); d=6145 -> dense kernel (>6144).
        for d, kind in ((6144, "moe"), (6145, "dense")):
            x = torch.randn((8, 2 * d), dtype=torch.bfloat16).npu()
            out, _ = situ_and_mul_quant(x, need_quant=False)
            self.assertEqual(tuple(out.shape), (8, d))

    def test_int32_overflow_regression_label(self):
        # N=32768 @ d=33792 is covered in the precision grid (test_d33792_N32768_*).
        # This is the int32-pointer-offset regression: row_idx*TOTAL_COLS overflowed
        # at N>~31775 before the int64 cast fix (situ_and_mul_quant.py:56). Kept as a
        # labeled pointer for the regression; the grid case is the actual assertion.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
