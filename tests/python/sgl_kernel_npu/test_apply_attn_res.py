import unittest

import torch
import torch.nn as nn
import torch_npu

from sgl_kernel_npu.norm.apply_attn_res import apply_attn_res


def apply_attn_res_native(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Reference — mirrors modeling_kimi.py _apply_attn_res (line 1119)."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [N, B+1, H]
    v_f = v.float()
    variance = v_f.pow(2).mean(-1, keepdim=True)
    k = v_f * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_f).squeeze(1).to(v.dtype)


class _FakeNorm:
    """Stand-in for KimiRMSNorm: just .weight + .variance_epsilon."""
    def __init__(self, weight, eps):
        self.weight = weight
        self.variance_epsilon = eps


H = 7168

# Tolerances per triton-ascend debug_guide/precision.md:
#   BF16 → rtol=atol=5e-3; FP32 → rtol=atol=1e-5. equal_nan=True (doc default).
_TOL = {
    torch.bfloat16: (5e-3, 5e-3),
    torch.float32: (1e-5, 1e-5),
}


class TestApplyAttnRes(unittest.TestCase):
    def _run(self, N, B, beta_scale=1.0, dtype=torch.bfloat16):
        torch.manual_seed(0)
        prefix_sum = (torch.randn(N, H, dtype=dtype).npu() * beta_scale)
        block_residual = (torch.randn(N, B, H, dtype=dtype).npu() * beta_scale)
        proj = nn.Linear(H, 1, bias=False).to(dtype).npu()
        norm_w = torch.randn(H, dtype=dtype).npu()
        norm = _FakeNorm(norm_w, 1e-5)
        proj_w = proj.weight.detach().squeeze(0)

        out = apply_attn_res(prefix_sum, block_residual, proj, norm)
        ref = apply_attn_res_native(prefix_sum, block_residual, proj_w, norm_w, 1e-5)
        rtol, atol = _TOL[dtype]
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=True)

    # ---- BF16 (doc: rtol=atol=5e-3) ----
    def test_b1_bf16(self):
        self._run(N=128, B=1, dtype=torch.bfloat16)

    def test_b4_bf16(self):
        self._run(N=256, B=4, dtype=torch.bfloat16)

    def test_b8_bf16(self):
        self._run(N=512, B=8, dtype=torch.bfloat16)

    def test_saturation_bf16(self):
        self._run(N=128, B=4, beta_scale=50.0, dtype=torch.bfloat16)

    # ---- FP32 (doc: rtol=atol=1e-5) ----
    def test_b1_fp32(self):
        self._run(N=128, B=1, dtype=torch.float32)

    def test_b4_fp32(self):
        self._run(N=256, B=4, dtype=torch.float32)

    def test_b8_fp32(self):
        self._run(N=512, B=8, dtype=torch.float32)

    def test_saturation_fp32(self):
        self._run(N=128, B=4, beta_scale=50.0, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
