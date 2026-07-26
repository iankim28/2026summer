"""Mask fills + per-backbone ViT patch-token neglect for partner fill ablation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy import ndimage
from open_clip.transformer import _expand_token

DISPLAY_SIZE = 224
BLUR_RADIUS = 12

# patch_size, grid for each scoring model
BACKBONE = {
    "en": {"kind": "open_clip", "patch": 32, "grid": 7},
    "zh": {"kind": "hf", "patch": 16, "grid": 14},
    "ko": {"kind": "hf", "patch": 32, "grid": 7},
    "ja": {"kind": "open_clip", "patch": 16, "grid": 14},
}


def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad[:-2, :-2]
            | pad[:-2, 1:-1]
            | pad[:-2, 2:]
            | pad[1:-1, :-2]
            | pad[1:-1, 1:-1]
            | pad[1:-1, 2:]
            | pad[2:, :-2]
            | pad[2:, 1:-1]
            | pad[2:, 2:]
        )
    return m


def align_cam(cam, size=DISPLAY_SIZE):
    cam = np.asarray(cam, dtype=np.float64)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    mx = cam.max()
    if mx > 0:
        cam = cam / mx
    return (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
        )
        / 255.0
    )


def n_cam_intersection(cam_en, cam_l):
    return np.minimum(align_cam(cam_en), align_cam(cam_l))


def cam_to_mask(saliency, threshold=0.95, dilate=3):
    thr = np.percentile(saliency, threshold * 100)
    mask = saliency >= thr
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask


def filter_mask_components(mask, top_k=2, bbox_snap=False):
    labeled, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = set(np.argsort(sizes)[::-1][:top_k] + 1)
    out = np.zeros_like(mask, dtype=bool)
    for i in keep:
        comp = labeled == i
        if bbox_snap:
            ys, xs = np.where(comp)
            out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        else:
            out |= comp
    return out


def build_cc_bbox_mask(cam_en, cam_l, threshold=0.95, dilate=3, top_k=2):
    inter = n_cam_intersection(cam_en, cam_l)
    mask = cam_to_mask(inter, threshold, dilate=dilate)
    return filter_mask_components(mask, top_k=top_k, bbox_snap=True)


def apply_mask(pil_img, mask, fill="blur"):
    """Image-space fill: blur | mean | black."""
    arr = np.array(pil_img.convert("RGB"))
    m = mask.astype(bool)
    if m.shape != arr.shape[:2]:
        m = (
            np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(arr.shape[1::-1], Image.NEAREST)
            )
            > 127
        )
    out = arr.copy()
    if not m.any():
        return Image.fromarray(out)
    if fill == "blur":
        blurred = np.array(
            Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        )
        out[m] = blurred[m]
    elif fill == "mean":
        mean = arr[~m].mean(0) if (~m).any() else arr.reshape(-1, 3).mean(0)
        out[m] = mean
    elif fill == "black":
        out[m] = 0
    else:
        raise ValueError(f"unknown fill={fill}")
    return Image.fromarray(out.astype(np.uint8))


def mask_to_patch_flags(mask, patch, grid, patch_thr=0.5):
    """Map 224 mask → length-(grid*grid) bool (True = neglect)."""
    m = mask.astype(bool)
    if m.shape != (DISPLAY_SIZE, DISPLAY_SIZE):
        m = (
            np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST
                )
            )
            > 127
        )
    flags = np.zeros(grid * grid, dtype=bool)
    cov = np.zeros(grid * grid, dtype=np.float64)
    for p in range(grid * grid):
        r, c = divmod(p, grid)
        cell = m[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch]
        cov[p] = float(cell.mean())
        flags[p] = cov[p] >= patch_thr
    return flags, cov


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


@torch.no_grad()
def encode_image_neglect_openclip(model, pil_img, mask, patch, grid, patch_thr=0.5, device="cuda"):
    """Zero open_clip ViT patch tokens (after pos-embed) for patches overlapping mask."""
    flags, cov = mask_to_patch_flags(mask, patch=patch, grid=grid, patch_thr=patch_thr)
    visual = model.m.visual
    x = model.pp(pil_img).unsqueeze(0).to(device)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([_expand_token(visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    if flags.any():
        idx = torch.tensor(np.where(flags)[0] + 1, device=x.device, dtype=torch.long)
        x[:, idx, :] = 0
    x = visual.patch_dropout(x)
    x = visual.ln_pre(x)
    x = visual.transformer(x)
    pooled, _tokens = visual._pool(x)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1), int(flags.sum()), cov


@torch.no_grad()
def encode_image_neglect_hf(model, pil_img, mask, patch, grid, patch_thr=0.5, device="cuda"):
    """Zero HF CLIP / ChineseCLIP vision patch tokens after embeddings."""
    flags, cov = mask_to_patch_flags(mask, patch=patch, grid=grid, patch_thr=patch_thr)
    pv = model.p(images=[pil_img], return_tensors="pt").pixel_values.to(device)
    # ChineseCLIP wraps vision in ChineseCLIPVisionModel; CLIPModel uses .vision_model
    vis = model.m.vision_model
    emb = vis.embeddings(pv)
    if flags.any():
        idx = torch.tensor(np.where(flags)[0] + 1, device=emb.device, dtype=torch.long)
        emb = emb.clone()
        emb[:, idx, :] = 0
    # both ChineseCLIP and CLIPVision use the historical typo pre_layrnorm
    x = vis.pre_layrnorm(emb)
    enc = vis.encoder(x)
    hs = enc.last_hidden_state
    pooled = vis.post_layernorm(hs[:, 0, :])
    if hasattr(model.m, "visual_projection"):
        pooled = model.m.visual_projection(pooled)
    return F.normalize(pooled.float(), dim=-1), int(flags.sum()), cov


@torch.no_grad()
def encode_image_neglect(model, lang, pil_img, mask, patch_thr=0.5, device="cuda"):
    spec = BACKBONE[lang]
    if spec["kind"] == "open_clip":
        return encode_image_neglect_openclip(
            model,
            pil_img,
            mask,
            patch=spec["patch"],
            grid=spec["grid"],
            patch_thr=patch_thr,
            device=device,
        )
    return encode_image_neglect_hf(
        model,
        pil_img,
        mask,
        patch=spec["patch"],
        grid=spec["grid"],
        patch_thr=patch_thr,
        device=device,
    )


@torch.no_grad()
def classify_neglect_batch(model, lang, text_emb, imgs, masks, patch_thr=0.5, device="cuda"):
    preds = []
    n_patches = []
    for img, mask in zip(imgs, masks):
        feat, n_p, _ = encode_image_neglect(
            model, lang, img, mask, patch_thr=patch_thr, device=device
        )
        pred = int((feat @ text_emb.t()).squeeze().argmax().item())
        preds.append(pred)
        n_patches.append(n_p)
    return np.array(preds), np.array(n_patches)
