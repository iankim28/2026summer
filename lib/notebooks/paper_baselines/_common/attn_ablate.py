"""Shared attention-head ablation helpers (Dyslexify / SamplingTAR ports)."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def fix_cls_attn_heads(visual, layer_spec, alpha=1.0):
    """Redirect CLS attention of selected heads to self (SamplingTAR-style).

    layer_spec: dict[int, list[int]]  layer -> head indices
    open_clip openai ViT-B/32 uses nn.MultiheadAttention(batch_first=True)
    and returns (attn_output, attn_weights_or_None).
    """
    hooks = []

    def hook_fn(module, inputs, output, heads):
        x = inputs[0]  # (B, L, C) when batch_first=True
        if not hasattr(module, "in_proj_weight") or module.in_proj_weight is None:
            return output
        if module.batch_first:
            x_nld = x
        else:
            x_nld = x.permute(1, 0, 2)
        B, L, C = x_nld.shape
        n_heads = module.num_heads
        qkv = F.linear(x_nld, module.in_proj_weight, module.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = C // n_heads
        q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, n_heads, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        att = att.softmax(dim=-1)
        # Dyslexify CLS mode: zero CLS←spatial for selected heads (equiv. alpha=1 self-attn)
        factors = att[:, :, :1, 1:].sum(dim=-1, keepdim=True)
        for h in heads:
            att[:, h, :1, 0] = alpha
            att[:, h, :1, 1:] = att[:, h, :1, 1:] * (1.0 - alpha) / (factors[:, h, :1, :] + 1e-6)
        out = (att @ v).transpose(1, 2).reshape(B, L, C)
        out = module.out_proj(out)
        if not module.batch_first:
            out = out.permute(1, 0, 2)
        # Critical: MHA returns a tuple; ResidualAttentionBlock indexes [0]
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    for layer, heads in layer_spec.items():
        if not heads:
            continue
        attn = visual.transformer.resblocks[int(layer)].attn
        hooks.append(attn.register_forward_hook(partial(hook_fn, heads=list(heads))))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def heads_to_layer_spec(heads):
    """list[(layer, head)] -> dict[layer, list[head]]"""
    spec = {}
    for layer, head in heads:
        spec.setdefault(int(layer), []).append(int(head))
    return spec


def patch_mask_from_rects(rects, grid=7, display=224):
    """Boolean mask over patches (grid*grid,) True if patch overlaps any rect."""
    ps = display // grid
    mask = np.zeros(grid * grid, dtype=bool)
    for x0, y0, x1, y1 in rects:
        for py in range(grid):
            for px in range(grid):
                bx0, by0 = px * ps, py * ps
                bx1, by1 = bx0 + ps, by0 + ps
                if not (x1 <= bx0 or bx1 <= x0 or y1 <= by0 or by1 <= y0):
                    mask[py * grid + px] = True
    return mask


@torch.no_grad()
def cls_to_patch_attn(visual, x, layer):
    """Return CLS→patch attention (B, n_heads, n_patches) for one layer."""
    captured = {}

    def hook_fn(module, inputs, output):
        inp = inputs[0]
        x_nld = inp if module.batch_first else inp.permute(1, 0, 2)
        B, L, C = x_nld.shape
        n_heads = module.num_heads
        head_dim = C // n_heads
        qkv = F.linear(x_nld, module.in_proj_weight, module.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        att = att.softmax(dim=-1)
        captured["attn"] = att[:, :, 0, 1:].detach()
        return output

    handle = visual.transformer.resblocks[layer].attn.register_forward_hook(hook_fn)
    try:
        _ = visual(x)
    finally:
        handle.remove()
    return captured["attn"]
