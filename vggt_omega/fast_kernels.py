"""Triton kernels for the "max" variant of vggt_omega.fast (any size; JIT-compiled once per dtype / flag combination).

- layer_norm_f32_to_f16: LayerNorm of an fp32 residual stream (rows addressed as [R, T] with strides) -> contiguous fp16
- add_layer_norm: x1 = x + o * g (fp32, new tensor) and LayerNorm(x1) -> fp16 in one pass
- qk_norm_rope_: in place on the packed qkv GEMM output [B, N, 3, H, 64]: q/k LayerNorm (optional), 2D RoPE (rotate_half
  layout, tables [HW, 64] fp32, first P tokens not rotated), q multiplied by qscale (softmax scale folded in)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(X, Y, W, B, T, sR, sT, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xp = X + (row // T) * sR + (row % T) * sT
    c = tl.arange(0, BLOCK)
    m = c < C
    x = tl.load(xp + c, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 0) / C + eps)
    y = xc * rstd * tl.load(W + c, mask=m, other=0.0) + tl.load(B + c, mask=m, other=0.0)
    tl.store(Y + row * C + c, y.to(tl.float16), mask=m)


def layer_norm_f32_to_f16(x, w, b, eps):
    """x [..., C] fp32 (any view whose leading dims collapse to [R, T] with strides sR, sT and unit stride in C)"""
    C = x.shape[-1]
    assert x.stride(-1) == 1 and w.dtype == torch.float32
    if x.dim() == 2:
        R, T, sR, sT = x.shape[0], 1, x.stride(0), 0
    else:
        x3 = x.reshape(-1, x.shape[-2], C) if x.dim() > 3 else x
        if x.dim() > 3:  # leading dims must collapse to one stride
            assert x3.data_ptr() == x.data_ptr()
        R, T, sR, sT = x3.shape[0], x3.shape[1], x3.stride(0), x3.stride(1)
        x = x3
    y = torch.empty(R * T, C, device=x.device, dtype=torch.float16)
    _ln_kernel[(R * T,)](x, y, w, b, T, sR, sT, C, eps, BLOCK=triton.next_power_of_2(C), num_warps=4 if C <= 1024 else 8)
    return y


@triton.jit
def _add_ln_kernel(X, O, G, XO, Y, W, B, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < C
    x = tl.load(X + row * C + c, mask=m, other=0.0) + tl.load(O + row * C + c, mask=m, other=0.0).to(tl.float32) * \
        tl.load(G + c, mask=m, other=0.0)
    tl.store(XO + row * C + c, x, mask=m)
    mean = tl.sum(x, 0) / C
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 0) / C + eps)
    y = xc * rstd * tl.load(W + c, mask=m, other=0.0) + tl.load(B + c, mask=m, other=0.0)
    tl.store(Y + row * C + c, y.to(tl.float16), mask=m)


def add_layer_norm(x, o, g, w, b, eps):
    """x fp32 [.., C] contiguous, o fp16 same shape, g fp32 [C] -> (x + o * g fp32 new tensor, LayerNorm fp16 [rows, C])"""
    C = x.shape[-1]
    assert x.is_contiguous() and o.is_contiguous()
    R = x.numel() // C
    xo = torch.empty_like(x)
    y = torch.empty(R, C, device=x.device, dtype=torch.float16)
    _add_ln_kernel[(R,)](x, o, g, xo, y, w, b, C, eps, BLOCK=triton.next_power_of_2(C), num_warps=4 if C <= 1024 else 8)
    return xo, y


@triton.jit
def _qk_kernel(QKV, SIN, COS, QW, QB, KW, KB, NROWS, N, H, P, eps, qscale,
               HAS_NORM: tl.constexpr, HAS_ROPE: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    valid = r < NROWS
    h = r % H
    t = r // H
    w = t % 2  # 0 = q, 1 = k
    tok = t // 2
    n = tok % N
    base = tok * (3 * H * 64) + w * (H * 64) + h * 64
    d = tl.arange(0, 32)
    p1 = QKV + base[:, None] + d[None, :]
    x1 = tl.load(p1, mask=valid[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(p1 + 32, mask=valid[:, None], other=0.0).to(tl.float32)
    isq = (w == 0)[:, None]
    if HAS_NORM:
        mean = (tl.sum(x1, 1) + tl.sum(x2, 1)) / 64
        x1 = x1 - mean[:, None]
        x2 = x2 - mean[:, None]
        rstd = 1.0 / tl.sqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / 64 + eps)
        w1 = tl.where(isq, tl.load(QW + d)[None, :], tl.load(KW + d)[None, :])
        w2 = tl.where(isq, tl.load(QW + 32 + d)[None, :], tl.load(KW + 32 + d)[None, :])
        b1 = tl.where(isq, tl.load(QB + d)[None, :], tl.load(KB + d)[None, :])
        b2 = tl.where(isq, tl.load(QB + 32 + d)[None, :], tl.load(KB + 32 + d)[None, :])
        x1 = x1 * rstd[:, None] * w1 + b1
        x2 = x2 * rstd[:, None] * w2 + b2
    if HAS_ROPE:
        pr = n - P
        ro = (pr >= 0) & valid
        tp = pr[:, None] * 64 + d[None, :]
        c1 = tl.load(COS + tp, mask=ro[:, None], other=1.0)
        c2 = tl.load(COS + tp + 32, mask=ro[:, None], other=1.0)
        s1 = tl.load(SIN + tp, mask=ro[:, None], other=0.0)
        s2 = tl.load(SIN + tp + 32, mask=ro[:, None], other=0.0)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        x1, x2 = y1, y2
    sc = tl.where(isq, qscale, 1.0)
    tl.store(p1, (x1 * sc).to(tl.float16), mask=valid[:, None])
    tl.store(p1 + 32, (x2 * sc).to(tl.float16), mask=valid[:, None])


def qk_norm_rope_(qkv, norm=None, rope=None, P=0, qscale=1.0, eps=1e-5):
    """qkv [B, N, 3, H, 64] contiguous fp16, in place. norm = (qw, qb, kw, kb) fp32 [64] or None; rope = (sin, cos) fp32
    [N - P, 64] or None."""
    B, N, _, H, D = qkv.shape
    assert D == 64 and qkv.is_contiguous() and qkv.dtype == torch.float16
    nrows = B * N * 2 * H
    BR = 32
    dummy = qkv
    qw, qb, kw, kb = norm if norm is not None else (dummy,) * 4
    sin, cos = rope if rope is not None else (dummy, dummy)
    if rope is not None:
        assert sin.shape == (N - P, 64) and sin.is_contiguous() and cos.is_contiguous()
    _qk_kernel[(triton.cdiv(nrows, BR),)](qkv, sin, cos, qw, qb, kw, kb, nrows, N, H, P, eps, float(qscale),
                                          HAS_NORM=norm is not None, HAS_ROPE=rope is not None, BR=BR, num_warps=4)
    return qkv
