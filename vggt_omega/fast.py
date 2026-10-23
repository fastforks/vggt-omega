"""Fast inference for VGGT-Omega: any frame count / image size, same outputs as VGGTOmega.forward.

    from vggt_omega.fast import FastVGGTOmega
    fm = FastVGGTOmega(model, variant="friendly" | "max")   # converts the loaded VGGTOmega in place
    preds = fm(images)                                        # images [S, 3, H, W] or [B, S, 3, H, W] in [0, 1]

Changes vs upstream (network math unchanged; precision / kernels only). Upstream runs the DINOv3 patch embed and the
aggregator under bf16 autocast with an fp32 residual stream (fp32 LayerNorms, fp32 RoPE) and the heads in fp32.
- patch embed (24 blocks) + aggregator (24 frame + 24 inter-frame blocks): fp16 weights / GEMMs / attention, the residual
  stream stays fp32 (as upstream; DINOv3 ViT-L streams exceed the fp16 range), LayerNorm fp32 in -> fp16 out; k-bias mask
  folded into the qkv bias once; LayerScale + residual as addcmul into the fp32 stream
- q/k LayerNorm + 2D RoPE on the packed qkv output (friendly: plain PyTorch; max: one Triton kernel, in place, softmax scale
  folded into q), attention: friendly torch SDPA, max vendored fastbb flash64 (`attn`="hh"/"fh") with fp16-accumulating GEMMs (`gacc`)
- DPT dense head fp16 channels-last (upstream fp32): token LayerNorm fp32 -> fp16 NHWC (free channels-last view), the two
  final 1x1 convs (depth / confidence) as one fp32 linear on NHWC, exp in fp32; camera head fp32 as upstream
friendly = plain PyTorch ops only (ONNX-exportable); max = + Triton kernels (LayerNorm, add+LayerNorm, q/k norm + RoPE),
fastbb flash attention, fp16-accumulate GEMMs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.aggregator import slice_expand_and_flatten
from .models.heads.utils import create_uv_grid, position_grid_to_embed

VARIANTS = {
    "friendly": dict(attn="sdpa", gacc=False, kern=False),
    "max": dict(attn="hh", gacc="nofc2", kern=True),
}
SMALL_WORK = 2.2e5  # B*N*C below this: torch SDPA (fastbb's threshold)


class _Blk:
    """prepared parameters of one SelfAttentionBlock"""

    def __init__(self, b, keep_fp32_norm):
        a = b.attn
        h = lambda t: t.detach().half().contiguous()
        f = lambda t: t.detach().float().contiguous()
        self.H = a.num_heads
        self.eps1, self.eps2 = b.norm1.eps, b.norm2.eps
        self.n1 = (f(b.norm1.weight), f(b.norm1.bias))
        self.n2 = (f(b.norm2.weight), f(b.norm2.bias))
        bias = a.qkv.bias * a.qkv.bias_mask if hasattr(a.qkv, "bias_mask") else a.qkv.bias
        self.wqkv, self.bqkv = h(a.qkv.weight), h(bias)
        self.wp, self.bp = h(a.proj.weight), h(a.proj.bias)
        self.qn = None
        if a.use_qk_norm:
            cv = f if keep_fp32_norm else h
            self.qn = (cv(a.q_norm.weight), cv(a.q_norm.bias), cv(a.k_norm.weight), cv(a.k_norm.bias))
            self.qk_eps = a.q_norm.eps
            assert a.k_norm.eps == a.q_norm.eps
        self.scale = a.scale
        self.g1, self.g2 = f(b.ls1.gamma), f(b.ls2.gamma)
        m = b.mlp
        assert isinstance(m.act, nn.GELU) and m.act.approximate == "none"
        self.w1, self.b1, self.w2, self.b2 = h(m.fc1.weight), h(m.fc1.bias), h(m.fc2.weight), h(m.fc2.bias)


def _rot(x, P, sin, cos):
    """rotate_half RoPE on tokens >= P of x [B, N, H, 64] (sin / cos [N - P, 64] in x's dtype)"""
    xp = x[:, P:]
    r = torch.cat((-xp[..., 32:], xp[..., :32]), -1)
    y = xp * cos[None, :, None] + r * sin[None, :, None]
    return torch.cat((x[:, :P], y), 1) if P else y


class FastVGGTOmega:
    def __init__(self, model, variant="friendly", dense_chunk=8, **over):
        cfg = dict(VARIANTS[variant], **over)
        self.m, self.variant, self.cfg, self.dense_chunk = model.eval(), variant, cfg, dense_chunk
        self.attn, self.gacc, self.kern = cfg["attn"], cfg["gacc"], cfg["kern"]
        if self.attn != "sdpa":
            from .fastbb_vendored.attention import attention
            self._fb_attention = attention
        if self.kern:
            from . import fast_kernels as K
            self.K = K
        agg = model.aggregator
        pe = agg.patch_embed
        assert isinstance(pe.patch_embed.norm, nn.Identity) and not pe.untie_cls_and_patch_norms
        self.pe_blocks = [_Blk(b, self.kern) for b in pe.blocks]
        self.frame = [_Blk(b, self.kern) for b in agg.frame_blocks]
        self.inter = [_Blk(b, self.kern) for b in agg.inter_frame_blocks]
        self.pe_conv = (pe.patch_embed.proj.weight.detach().half(), pe.patch_embed.proj.bias.detach().half())
        self.pe_norm = (pe.norm.weight.detach().float(), pe.norm.bias.detach().float(), pe.norm.eps)
        # free the fp32 copies of the converted blocks (camera head / dense head stay)
        for mod in (pe.blocks, agg.frame_blocks, agg.inter_frame_blocks):
            for p in mod.parameters():
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)
        dh = model.dense_head
        if dh is not None:
            last = (dh.proj, dh.proj_conf)
            dh.proj = dh.proj_conf = None
            dh.half().to(memory_format=torch.channels_last)
            dh.proj, dh.proj_conf = last
            self.head_w = torch.cat([last[0].weight.flatten(1), last[1].weight.flatten(1)]).float()   # [2 * r^2, 256]
            self.head_b = torch.cat([last[0].bias, last[1].bias]).float()
            self.head_norm = (dh.norm.weight.detach().float(), dh.norm.bias.detach().float(), dh.norm.eps)
        self._pos = {}
        self.settings = (f"FastVGGTOmega variant={variant} {cfg} dense_chunk={dense_chunk}; fp16 patch embed + aggregator "
                         "(fp32 residual stream), fp16 DPT head (final 1x1 convs fp32), camera head fp32")

    # ------------------------------------------------------------------ pieces
    def ln(self, x, wb, eps):
        """fp32 [.., C] -> fp16 [rows, C] (contiguous NHWC-ready)"""
        if self.kern:
            return self.K.layer_norm_f32_to_f16(x, wb[0], wb[1], eps)
        C = x.shape[-1]
        return F.layer_norm(x, (C,), wb[0], wb[1], eps).half().reshape(-1, C)

    def _qk_attn(self, p, qkv, rope, P):
        B, N, _, H, D = qkv.shape
        C = H * D
        fast = self.attn != "sdpa" and D == 64 and B * N * C >= SMALL_WORK
        if self.kern:
            self.K.qk_norm_rope_(qkv, p.qn, rope, P, qscale=p.scale if fast else 1.0, eps=getattr(p, "qk_eps", 1e-5))
            q, k, v = qkv.unbind(2)
        else:
            q, k, v = qkv.unbind(2)  # [B, N, H, 64]
            if p.qn is not None:
                q = F.layer_norm(q, (D,), p.qn[0], p.qn[1], p.qk_eps)
                k = F.layer_norm(k, (D,), p.qn[2], p.qn[3], p.qk_eps)
            if rope is not None:
                sin, cos = rope
                q, k = _rot(q, P, sin, cos), _rot(k, P, sin, cos)
        if fast:
            return self._fb_attention(q, k, v, scale=1.0 if self.kern else p.scale, variant=self.attn).view(B, N, C)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=p.scale)
        return o.transpose(1, 2).reshape(B, N, C)

    def block(self, p, x, rope=None, P=0):
        """x fp32 [B, N, C] (not modified) -> new fp32 tensor"""
        B, N, C = x.shape
        qkv = F.linear(self.ln(x, p.n1, p.eps1), p.wqkv, p.bqkv).view(B, N, 3, p.H, C // p.H)
        o = F.linear(self._qk_attn(p, qkv, rope, P), p.wp, p.bp)
        if self.kern:
            x, y = self.K.add_layer_norm(x, o.view(B, N, C), p.g1, p.n2[0], p.n2[1], p.eps2)
        else:
            x = torch.addcmul(x, o.view(B, N, C), p.g1)
            y = self.ln(x, p.n2, p.eps2)
        h = F.gelu(F.linear(y, p.w1, p.b1))
        if self.gacc == "nofc2":
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            o = F.linear(h, p.w2, p.b2)
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            o = F.linear(h, p.w2, p.b2)
        return x.addcmul_(o.view(B, N, C), p.g2)

    def patch_embed(self, x):
        pe = self.m.aggregator.patch_embed
        t = F.conv2d(x.half(), *self.pe_conv, stride=pe.patch_size)
        BS, C, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)
        cls = (pe.cls_token + 0 * pe.mask_token).float().expand(BS, -1, -1)
        tok = torch.cat([cls, pe.storage_tokens.float().expand(BS, -1, -1), t.float()], 1)
        sin, cos = pe.rope_embed(H=h, W=w)
        rope = (sin.float().contiguous(), cos.float().contiguous())
        if not self.kern:
            rope = tuple(r.half() for r in rope)
        P = 1 + pe.n_storage_tokens
        for p in self.pe_blocks:
            tok = self.block(p, tok, rope, P)
        w_, b_, eps = self.pe_norm
        return F.layer_norm(tok[:, P:], (C,), w_, b_, eps)  # x_norm_patchtokens, fp32 like upstream AMP

    def aggregate(self, images):
        agg = self.m.aggregator
        B, S, _, H, W = images.shape
        x = ((images - agg._resnet_mean) / agg._resnet_std).view(B * S, 3, H, W)
        pt = self.patch_embed(x)
        cam = slice_expand_and_flatten(agg.camera_token, B, S)
        reg = slice_expand_and_flatten(agg.register_token, B, S)
        tokens = torch.cat([cam, reg, pt], 1).float()
        _, N, C = tokens.shape
        sin, cos = agg.rope_embed(H=H // agg.patch_size, W=W // agg.patch_size)
        rope = (sin.float().contiguous(), cos.float().contiguous())
        if not self.kern:
            rope = tuple(r.half() for r in rope)
        P = agg.patch_token_start
        out = []
        for i in range(agg.depth):
            tokens = self.block(self.frame[i], tokens.view(B * S, N, C), rope, P)
            fr = tokens.view(B, S, N, C)
            t4 = fr
            if agg.inter_frame_attention_types[i] == "global":
                t4 = self.block(self.inter[i], fr.reshape(B, S * N, C)).view(B, S, N, C)
            else:
                cr = self.block(self.inter[i], fr[:, :, :P].reshape(B, S * P, C)).view(B, S, P, C)
                t4 = torch.cat([cr, fr[:, :, P:]], 2)
            tokens = t4
            out.append(torch.cat([fr, t4], -1) if i in agg.cached_layer_indices else None)
        return out, P

    def _pos_embed(self, x, width, height):
        k = (x.shape[1:], width, height)
        if k not in self._pos:
            pw, ph = x.shape[-1], x.shape[-2]
            pe = create_uv_grid(pw, ph, aspect_ratio=width / height, dtype=torch.float32, device=x.device)
            pe = position_grid_to_embed(pe, x.shape[1]) * 0.1
            self._pos[k] = pe.permute(2, 0, 1)[None].to(x.dtype).contiguous(memory_format=torch.channels_last)
        return x + self._pos[k]

    def _dense_chunk(self, toks, P, B, s0, s1, H, W):
        dh = self.m.dense_head
        ps = dh.patch_size
        ph, pw = H // ps, W // ps
        n = (s1 - s0) * B
        feats = []
        for fi, li in enumerate(dh.intermediate_layer_idx):
            x = toks[li][:, s0:s1, P:]
            y = self.ln(x, self.head_norm[:2], self.head_norm[2])  # fp16 [n * ph * pw, 2C] = NHWC
            y = y.view(n, ph, pw, -1).permute(0, 3, 1, 2)
            y = dh.projects[fi](y)
            y = self._pos_embed(y, W, H)
            feats.append(dh.resize_layers[fi](y))
        fused = dh.scratch_forward(feats)
        fused = self._pos_embed(fused, W, H)
        r = dh.final_shuffle_factor
        o = F.linear(fused.permute(0, 2, 3, 1).float(), self.head_w, self.head_b)  # [n, h, w, 2 r^2] fp32
        o = F.pixel_shuffle(o.permute(0, 3, 1, 2), r)  # [n, 2, H, W]
        depth = torch.exp(o[:, 0])[..., None]
        conf = 1.0 + torch.exp(o[:, 1])
        return depth.view(B, s1 - s0, H, W, 1), conf.view(B, s1 - s0, H, W)

    @torch.inference_mode()
    def __call__(self, images):
        m = self.m
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape
        old = torch.backends.cuda.matmul.allow_fp16_accumulation
        try:
            with torch.autocast("cuda", enabled=False):
                torch.backends.cuda.matmul.allow_fp16_accumulation = self.gacc is not False
                toks, P = self.aggregate(images)
                torch.backends.cuda.matmul.allow_fp16_accumulation = old
                preds = {"camera_and_register_tokens": toks[-1][:, :, :P].contiguous()}
                if m.camera_head is not None:
                    preds["pose_enc"] = m.camera_head(toks, patch_token_start=P)
                if m.dense_head is not None:
                    c = self.dense_chunk or S
                    ds, cs = zip(*[self._dense_chunk(toks, P, B, s, min(s + c, S), H, W) for s in range(0, S, c)])
                    preds["depth"], preds["depth_conf"] = torch.cat(ds, 1), torch.cat(cs, 1)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = old
        preds["images"] = images
        return preds
