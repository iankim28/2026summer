"""Mask fills + ViT patch-token neglect for EN neglect-vs-blur experiment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy import ndimage

from open_clip.transformer import _expand_token

DISPLAY_SIZE = 224
BLUR_RADIUS = 12
PATCH = 32
GRID = 7  # ViT-B/32


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


def rects_to_mask(rects, size=DISPLAY_SIZE):
    m = np.zeros((size, size), dtype=bool)
    for box in rects:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(size, x1), min(size, y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


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


def mask_to_patch_flags(mask, patch_thr=0.5):
    """Map 224 mask → length-49 bool (True = neglect)."""
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
    flags = np.zeros(GRID * GRID, dtype=bool)
    cov = np.zeros(GRID * GRID, dtype=np.float64)
    for p in range(GRID * GRID):
        r, c = divmod(p, GRID)
        cell = m[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
        cov[p] = float(cell.mean())
        flags[p] = cov[p] >= patch_thr
    return flags, cov


@torch.no_grad()
def encode_image_neglect(en_model, pil_img, mask, patch_thr=0.5, device="cuda"):
    """Zero ViT patch tokens (after pos-embed) for patches overlapping mask."""
    flags, cov = mask_to_patch_flags(mask, patch_thr=patch_thr)
    visual = en_model.m.visual
    x = en_model.pp(pil_img).unsqueeze(0).to(device)
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
def classify_neglect_batch(en_model, text_emb, imgs, masks, patch_thr=0.5, device="cuda"):
    preds = []
    n_patches = []
    for img, mask in zip(imgs, masks):
        feat, n_p, _ = encode_image_neglect(
            en_model, img, mask, patch_thr=patch_thr, device=device
        )
        pred = int((feat @ text_emb.t()).squeeze().argmax().item())
        preds.append(pred)
        n_patches.append(n_p)
    return np.array(preds), np.array(n_patches)


def _make_openclip_hook(collector):
    def hook(module, inputs, output):
        q_in = inputs[0]
        if getattr(module, "batch_first", False):
            B, L, D = q_in.shape
        else:
            L, B, D = q_in.shape
            q_in = q_in.transpose(0, 1).contiguous()
        n_head = module.num_heads
        hd = D // n_head
        with torch.no_grad():
            qkv = F.linear(q_in, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * (hd**-0.5)
            attn = attn.softmax(dim=-1)
        collector.append(attn[0].detach().cpu())

    return hook


def _norm_cam(cam):
    cam = np.maximum(cam if isinstance(cam, np.ndarray) else cam.cpu().numpy(), 0)
    cam -= cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def _build_attn_cam(all_attns):
    a = all_attns[-1]
    cls_row = a.mean(0)[0, 1:]
    n = int(round(cls_row.shape[0] ** 0.5))
    return _norm_cam(cls_row.reshape(n, n).numpy())


@torch.no_grad()
def classify_and_attn_en(en_model, text_emb, pil_img, device="cuda"):
    x = en_model.pp(pil_img).unsqueeze(0).to(device)
    collector = []
    handles = [
        rb.attn.register_forward_hook(_make_openclip_hook(collector))
        for rb in en_model.m.visual.transformer.resblocks
    ]
    feat = en_model.m.visual(x)
    imf = F.normalize(feat, dim=-1)
    pred = int((imf @ text_emb.t()).squeeze().argmax().item())
    for h in handles:
        h.remove()
    return pred, _build_attn_cam(collector)


@torch.no_grad()
def classify_and_attn_zh(zh_model, text_emb, pil_img, device="cuda"):
    pv = zh_model.p(images=[pil_img], return_tensors="pt").pixel_values.to(device)
    vis_out = zh_model.m.vision_model(pixel_values=pv, output_attentions=True)
    if hasattr(zh_model.m, "visual_projection"):
        proj = zh_model.m.visual_projection(vis_out.pooler_output)
    else:
        proj = vis_out.pooler_output
    imf = F.normalize(proj, dim=-1)
    pred = int((imf @ text_emb.t()).squeeze().argmax().item())
    attns = [a[0].cpu() for a in vis_out.attentions]
    return pred, _build_attn_cam(attns)
