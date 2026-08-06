"""HF CLIP ViT attention mining / ablation (for Korean Bingsu CLIP)."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from .attn_ablate import heads_to_layer_spec, patch_mask_from_rects
from .protocol import BLUR_RADIUS, DEVICE, blur_regions


def _vision_layers(model):
    return model.vision_model.encoder.layers


def _qkv(attn, hidden):
    """HF CLIPAttention: separate q/k/v projections."""
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    v = attn.v_proj(hidden)
    return q, k, v


def _hidden_from_hook(args, kwargs):
    """HF CLIPAttention is called with hidden_states=... (keyword-only)."""
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise RuntimeError("CLIPAttention hook: no hidden_states in args/kwargs")


@torch.no_grad()
def cls_to_patch_attn_hf(model, pixel_values, layer):
    """Return CLS→patch attn (B, n_heads, n_patches) for one HF vision layer."""
    captured = {}

    def hook_fn(module, args, kwargs, output):
        hidden = _hidden_from_hook(args, kwargs)
        B, L, C = hidden.shape
        n_heads = module.num_heads
        head_dim = C // n_heads
        q, k, _ = _qkv(module, hidden)
        q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
        scale = head_dim**-0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.softmax(dim=-1)
        captured["attn"] = att[:, :, 0, 1:].detach()
        return output

    layer_mod = _vision_layers(model)[int(layer)].self_attn
    handle = layer_mod.register_forward_hook(hook_fn, with_kwargs=True)
    try:
        _ = model.vision_model(pixel_values=pixel_values)
    finally:
        handle.remove()
    return captured["attn"]


@contextmanager
def fix_cls_attn_heads_hf(model, layer_spec, alpha=1.0):
    """Redirect CLS attention of selected HF vision heads to self (alpha=1)."""
    hooks = []

    def hook_fn(module, args, kwargs, output, heads):
        hidden = _hidden_from_hook(args, kwargs)
        B, L, C = hidden.shape
        n_heads = module.num_heads
        head_dim = C // n_heads
        q, k, v = _qkv(module, hidden)
        q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, n_heads, head_dim).transpose(1, 2)
        scale = head_dim**-0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.softmax(dim=-1)
        factors = att[:, :, :1, 1:].sum(dim=-1, keepdim=True)
        for h in heads:
            att[:, h, :1, 0] = alpha
            att[:, h, :1, 1:] = att[:, h, :1, 1:] * (1.0 - alpha) / (
                factors[:, h, :1, :] + 1e-6
            )
        out = (att @ v).transpose(1, 2).reshape(B, L, C)
        out = module.out_proj(out)
        # HF CLIPAttention returns (attn_output, attn_weights) or just tensor
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    for layer, heads in layer_spec.items():
        if not heads:
            continue
        attn = _vision_layers(model)[int(layer)].self_attn
        hooks.append(
            attn.register_forward_hook(
                partial(hook_fn, heads=list(heads)), with_kwargs=True
            )
        )
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def attn_patch_mask_hf(
    model,
    pil_img,
    heads,
    processor,
    top_k_patches=4,
    score_frac=0.35,
    grid=7,
):
    n_patches = grid * grid
    if not heads:
        return np.zeros(n_patches, dtype=bool)
    pv = processor(images=[pil_img], return_tensors="pt").pixel_values.to(DEVICE)
    layer_spec = heads_to_layer_spec(heads)
    scores = np.zeros(n_patches, dtype=np.float64)
    n = 0
    for layer, hlist in layer_spec.items():
        a = cls_to_patch_attn_hf(model, pv, layer)[0].float().cpu().numpy()
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
    order = cand[np.argsort(-scores[cand])[:k]]
    mask = np.zeros(n_patches, dtype=bool)
    mask[order] = True
    return mask


def patch_mask_to_rects(mask, grid=7, display=224):
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


@torch.no_grad()
def classify_hybrid_hf(
    ko,
    imgs,
    text_emb,
    heads,
    top_k_patches=4,
    score_frac=0.35,
    batch_size=32,
    grid=7,
):
    """Attn-guided blur + HF head ablation. Returns preds, mean_patches."""
    blurred = []
    n_patches = []
    for im in imgs:
        mask = attn_patch_mask_hf(
            ko.m,
            im,
            heads,
            ko.p,
            top_k_patches=top_k_patches,
            score_frac=score_frac,
            grid=grid,
        )
        rects = patch_mask_to_rects(mask, grid=grid)
        blurred.append(blur_regions(im, rects, radius=BLUR_RADIUS))
        n_patches.append(int(mask.sum()))
    preds = []
    spec = heads_to_layer_spec(heads)
    with fix_cls_attn_heads_hf(ko.m, spec, alpha=1.0):
        for i in range(0, len(blurred), batch_size):
            imf = ko.embed_images(blurred[i : i + batch_size])
            preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds), float(np.mean(n_patches)) if n_patches else 0.0


@torch.no_grad()
def mine_head_scores_hf(ko, attacked, rects, max_images=50, n_layers=12, n_heads=12, grid=7):
    scores = {(L, H): [] for L in range(n_layers) for H in range(n_heads)}
    n = min(max_images, len(attacked))
    for i in range(n):
        pv = ko.p(images=[attacked[i]], return_tensors="pt").pixel_values.to(DEVICE)
        mask = patch_mask_from_rects(rects[i], grid=grid)
        if not mask.any():
            continue
        for layer in range(n_layers):
            attn = cls_to_patch_attn_hf(ko.m, pv, layer)[0].float().cpu().numpy()
            for h in range(n_heads):
                scores[(layer, h)].append(float(attn[h, mask].sum()))
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    vals = np.array(list(means.values()))
    mu, sigma = float(vals.mean()), float(vals.std() + 1e-8)
    print(f"HF mined on {n} imgs; score mean={mu:.4f} std={sigma:.4f}")
    return means, mu, sigma


@torch.no_grad()
def typographic_scores_hf(ko, attacked, rects, max_images=50, n_layers=12, n_heads=12, grid=7):
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    counts = 0
    n = min(max_images, len(attacked))
    for i in range(n):
        pv = ko.p(images=[attacked[i]], return_tensors="pt").pixel_values.to(DEVICE)
        mask = patch_mask_from_rects(rects[i], grid=grid)
        if not mask.any():
            continue
        for layer in range(n_layers):
            a = cls_to_patch_attn_hf(ko.m, pv, layer)[0].float().cpu().numpy()
            total = a.sum(axis=1) + 1e-8
            typo = a[:, mask].sum(axis=1)
            scores[layer] += typo / total
        counts += 1
    scores /= max(counts, 1)
    ranked = []
    for L in range(n_layers):
        for H in range(n_heads):
            ranked.append((float(scores[L, H]), L, H))
    ranked.sort(reverse=True)
    print(f"HF typo scores from {counts} images; top5={ranked[:5]}")
    return ranked, scores


@torch.no_grad()
def classify_with_heads_hf(ko, imgs, text_emb, heads, batch_size=32):
    spec = heads_to_layer_spec(heads)
    preds = []
    with fix_cls_attn_heads_hf(ko.m, spec, alpha=1.0):
        for i in range(0, len(imgs), batch_size):
            imf = ko.embed_images(imgs[i : i + batch_size])
            preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)
