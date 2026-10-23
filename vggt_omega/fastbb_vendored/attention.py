# vendored from fastforks/fastbb: fastbb/attention.py @ a61ba9c -- do not edit here, see fastbb/VENDORED.md
"""Flash attention for head_dim 64 with optional fp16 accumulation (prebuilt kernel, any N, no per-size work).

attention(q, k, v, scale=None, out=None, variant="hh")
    q, k, v: [B, N, H, 64] fp16 views (d contiguous, other strides multiples of 8 elements; e.g. the unbound packed qkv GEMM
    output [B, N, 3, H, 64]); returns O as [B, N, H, 64] (contiguous -> .view(B, N, H*64) for the projection, no transpose).
variant: "hh" fp16-accumulate QK^T and PV (fastest), "fh" fp32 QK^T + fp16 PV, "ff" fp32 both (= SDPA flash precision).
Falls back to torch SDPA for anything the kernel does not cover (other head dims, dtypes, strides, CPU).
"""
import math

import torch
import torch.nn.functional as F

from . import kernels as _k

_KERN = [None]
_PACKED = __import__("os").environ.get("FASTBB_PACKED", "0") == "1"  # packed-buffer launch (lower CPU cost), validating


def _kern():
    if _KERN[0] is None:
        _KERN[0] = _k.Kernels.get()
    return _KERN[0]


def _ok(t):
    return (t.is_cuda and t.dtype == torch.float16 and t.dim() == 4 and t.shape[-1] == 64 and t.stride(-1) == 1
            and all(s % 8 == 0 for s in t.stride()[:3]) and t.data_ptr() % 16 == 0)


def attention(q, k, v, scale=None, out=None, variant="hh", rows=None):
    """q, k, v [B, N, H, 64] -> [B, N, H, 64]."""
    B, N, H, D = q.shape
    if out is not None and (out.shape != (*q.shape[:-1], v.shape[-1]) or out.device != q.device or out.dtype != q.dtype):
        raise ValueError("out must match the attention output shape, dtype and device")
    if scale is None:
        scale = D ** -0.5
    if scale <= 0:
        # Flash implementations may scale a masked -inf tail to NaN or +inf.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        o = o.transpose(1, 2)
        return out.copy_(o) if out is not None else o
    if not (scale > 0 and _ok(q) and _ok(k) and _ok(v) and k.shape == q.shape and v.shape == q.shape
            and k.device == q.device == v.device and (out is None or _ok(out))):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        o = o.transpose(1, 2)
        return out.copy_(o) if out is not None else o
    if out is None:
        out = torch.empty((B, N, H, D), device=q.device, dtype=q.dtype)
    rm = rows or (2 if N * B * H >= 36 * 2 * 128 * 4 else 1)  # small problems: 64-row CTAs fill the GPU better
    kk = _kern()
    name = f"flash64_r{rm}_{variant}"
    br = 64 * rm
    grid, block = ((N + br - 1) // br, B * H, 1), (128, 1, 1)
    (kk.launch_packed if _PACKED else (lambda n, g, b, *x: kk.launch(n, (g, b), *x)))(name, grid, block, q, k, v, out, N, H,
              q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
              v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
              float(scale * math.log2(math.e)))
    return out


def attention_bhnd(q, k, v, scale=None, variant="hh"):
    """SDPA-layout drop-in: q, k, v [B, H, N, 64] -> [B, H, N, 64] (strided views, no copies)."""
    return attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale, variant=variant).transpose(1, 2)
