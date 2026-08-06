"""Attention-guided blur + head ablation hybrid for Dyslexify / SamplingTAR."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from .attn_ablate import fix_cls_attn_heads, heads_to_layer_spec
from .protocol import BLUR_RADIUS, DEVICE, blur_regions


GRID = 7
DISPLAY = 224
DEFAULT_TOP_K = 4
DEFAULT_SCORE_FRAC = 0.35


@torch.no_grad()
def cls_to_patch_attn_multi(visual, x, layers):
    """One forward: dict[layer] -> (B, n_heads, n_patches) CLS→patch attn."""
    layers = sorted({int(L) for L in layers})
    captured = {}

    def hook_fn(module, inputs, output, layer):
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
        captured[layer] = att[:, :, 0, 1:].detach()
        return output

    handles = []
    for layer in layers:
        attn = visual.transformer.resblocks[layer].attn
        handles.append(
            attn.register_forward_hook(partial(hook_fn, layer=layer))
        )
    try:
        _ = visual(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def attn_patch_mask(
    visual,
    pil_img,
    heads,
    top_k_patches=DEFAULT_TOP_K,
    score_frac=DEFAULT_SCORE_FRAC,
    pp=None,
    grid=GRID,
):
    """Bool mask over patches from mean CLS→patch attn of heads.

    Keeps patches with score >= score_frac * max(score), capped at top_k.
    Peakier sticker maps → more patches; flat clean maps → fewer.
    """
    n_patches = int(grid) * int(grid)
    if not heads:
        return np.zeros(n_patches, dtype=bool)
    if pp is None:
        raise ValueError("pp (open_clip preprocess) required")
    x = pp(pil_img).unsqueeze(0).to(DEVICE)
    layer_spec = heads_to_layer_spec(heads)
    attn_by_layer = cls_to_patch_attn_multi(visual, x, layer_spec.keys())
    scores = np.zeros(n_patches, dtype=np.float64)
    n = 0
    for layer, hlist in layer_spec.items():
        a = attn_by_layer[int(layer)][0].float().cpu().numpy()  # (H, P)
        for h in hlist:
            scores += a[int(h)]
            n += 1
    if n == 0:
        return np.zeros(n_patches, dtype=bool)
    scores /= n
    mx = float(scores.max())
    if mx <= 0:
        return np.zeros(n_patches, dtype=bool)
    cand = np.where(scores >= float(score_frac) * mx)[0]
    if cand.size == 0:
        cand = np.array([int(scores.argmax())])
    k = min(int(top_k_patches), cand.size)
    # among candidates, keep the top-k by score
    order = cand[np.argsort(-scores[cand])[:k]]
    mask = np.zeros(n_patches, dtype=bool)
    mask[order] = True
    return mask


def patch_mask_to_rects(mask, grid=GRID, display=DISPLAY):
    """Convert bool patch mask → list of (x0,y0,x1,y1) pixel boxes."""
    ps = display // grid
    rects = []
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    for i, on in enumerate(flat):
        if not on:
            continue
        py, px = i // grid, i % grid
        x0, y0 = px * ps, py * ps
        rects.append((x0, y0, x0 + ps, y0 + ps))
    return rects


def blur_by_attn_heads(
    visual,
    pil_img,
    heads,
    top_k_patches=DEFAULT_TOP_K,
    score_frac=DEFAULT_SCORE_FRAC,
    pp=None,
    radius=BLUR_RADIUS,
    grid=GRID,
):
    """Blur patches with highest mean CLS attn under selected heads."""
    mask = attn_patch_mask(
        visual,
        pil_img,
        heads,
        top_k_patches=top_k_patches,
        score_frac=score_frac,
        pp=pp,
        grid=grid,
    )
    rects = patch_mask_to_rects(mask, grid=grid)
    return blur_regions(pil_img, rects, radius=radius), mask


@torch.no_grad()
def classify_hybrid(
    en,
    imgs,
    text_emb,
    heads,
    top_k_patches=DEFAULT_TOP_K,
    score_frac=DEFAULT_SCORE_FRAC,
    batch_size=32,
    ablate=True,
    grid=None,
):
    """Attn-guided blur (+ optional head ablation) classify. Returns preds, mean_patches."""
    visual = en.m.visual
    g = int(grid if grid is not None else getattr(en, "grid", GRID))
    blurred = []
    n_patches = []
    for im in imgs:
        bim, mask = blur_by_attn_heads(
            visual,
            im,
            heads,
            top_k_patches=top_k_patches,
            score_frac=score_frac,
            pp=en.pp,
            grid=g,
        )
        blurred.append(bim)
        n_patches.append(int(mask.sum()))
    preds = []
    if ablate and heads:
        spec = heads_to_layer_spec(heads)
        with fix_cls_attn_heads(visual, spec, alpha=1.0):
            for i in range(0, len(blurred), batch_size):
                imf = en.embed_images(blurred[i : i + batch_size])
                preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    else:
        for i in range(0, len(blurred), batch_size):
            imf = en.embed_images(blurred[i : i + batch_size])
            preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds), float(np.mean(n_patches)) if n_patches else 0.0
