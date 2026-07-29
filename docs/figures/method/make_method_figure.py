"""Make Method overview figure for the paper (gated cc_bbox_black).

Outputs (this directory):
  method_overview.png — two-panel schematic:
    (a) Localization: Attn-last EN ∩ L → thr / dilate / cc_bbox
    (b) Gated occlusion: attack detector → black fill | skip → reclassify

Uses frozen CIFAR-10 sample + attack_pos. Partner L = ZH (same recipe for KO/JA).
EN: OpenAI ViT-B/32 (open_clip). ZH: Chinese-CLIP ViT-B/16. CUDA required.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib import cm
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(
    HERE, "..", "..", "..", "lib", "notebooks", "image_samples",
    "CIFAR10_BALANCED_1000_SAMPLE.json",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_SIZE = 224
FONT_SIZE = 24
PAD = 8
THRESHOLD = 0.95
DILATE = 3
TOP_K = 2
TMPL_EN = "a photo of a {}."
TMPL_ZH = "一张{}的照片。"

CLASSES = {
    "en": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}

# Visual language (Defense-Prefix-style schematic)
C_TEXT = "#E8D5F2"       # lavender — text / labels
C_IMG = "#D4EDDA"        # green — image path
C_ENC_EN = "#C5B4E3"     # purple trapezoid EN
C_ENC_L = "#A8D5A2"      # green trapezoid L
C_LOSS = "#FFF3CD"       # yellow — decision / pred boxes
C_GATE = "#FDE2E4"       # soft pink — detector
C_ARROW = "#555555"
C_PANEL = "#1a1a1a"


def _font_paths():
    wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    return os.path.join(wf, "msyh.ttc"), os.path.join(wf, "arial.ttf")


_CJK, _LAT = _font_paths()
_FONT_CACHE = {}


def _get_font(fp, size=FONT_SIZE):
    key = (fp, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y


def require_cuda():
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is required for method-figure CLIP attention, but "
            f"torch.cuda.is_available() is False (torch={torch.__version__}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Device: cuda ({torch.__version__})")


def draw_e_plus_l(img, en_word, zh_word, xy_en, xy_l):
    img = img.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    for word, fp, xy in [(en_word, _LAT, xy_en), (zh_word, _CJK, xy_l)]:
        font = _get_font(fp)
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text(
            (rx + PAD - bb[0], ry + PAD - bb[1]),
            word, fill="black", font=font,
        )
    return img


def _norm_cam(cam):
    cam = np.maximum(cam if isinstance(cam, np.ndarray) else cam.cpu().numpy(), 0)
    cam -= cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def align_cam(cam, size=DISPLAY_SIZE):
    return np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0


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
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.softmax(dim=-1)
        collector.append(attn[0].detach().cpu())
    return hook


def _build_attn_cam(all_attns):
    a = all_attns[-1]
    cls_row = a.mean(0)[0, 1:]
    n = int(round(cls_row.shape[0] ** 0.5))
    return _norm_cam(cls_row.reshape(n, n).numpy())


def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:]
            | pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:]
            | pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:]
        )
    return m


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
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        else:
            out |= comp
    return out


def apply_black(pil_img, mask):
    arr = np.array(pil_img.convert("RGB"))
    m = mask.astype(bool)
    if mask.shape != arr.shape[:2]:
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize(
                arr.shape[1::-1], Image.NEAREST)
        ) > 127
    out = arr.copy()
    out[m] = 0
    return Image.fromarray(out.astype(np.uint8))


def overlay_heatmap(pil_img, cam, alpha=0.45):
    base = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    heat = cm.jet(align_cam(cam))[:, :, :3]
    mix = (1 - alpha) * base + alpha * heat
    return Image.fromarray((mix * 255).astype(np.uint8))


def overlay_mask(pil_img, mask, color=(255, 40, 40), alpha=0.55):
    arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    tint = np.zeros_like(arr)
    tint[:] = color
    arr[m] = (1 - alpha) * arr[m] + alpha * tint[m]
    return Image.fromarray(arr.astype(np.uint8))


def outline_bbox_mask(pil_img, mask, color=(255, 255, 255), width=2):
    """Draw white outlines around cc_bbox rectangles on a copy of the image."""
    out = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    labeled, n = ndimage.label(mask.astype(bool))
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        box = [xs.min(), ys.min(), xs.max(), ys.max()]
        for t in range(width):
            draw.rectangle(
                [box[0] - t, box[1] - t, box[2] + t, box[3] + t],
                outline=color,
            )
    return out


@torch.no_grad()
def en_attn_and_pred(model, preprocess, text_emb, pil_img):
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    collector = []
    handles = [
        rb.attn.register_forward_hook(_make_openclip_hook(collector))
        for rb in model.visual.transformer.resblocks
    ]
    feat = model.visual(x)
    imf = F.normalize(feat, dim=-1)
    logits = (100.0 * imf @ text_emb.T).squeeze(0)
    pred = int(logits.argmax().item())
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    for h in handles:
        h.remove()
    return pred, probs, _build_attn_cam(collector)


@torch.no_grad()
def zh_attn_and_pred(model, processor, text_emb, pil_img):
    pv = processor(images=[pil_img], return_tensors="pt").pixel_values.to(DEVICE)
    vis_out = model.vision_model(pixel_values=pv, output_attentions=True)
    proj = model.visual_projection(vis_out.pooler_output)
    imf = F.normalize(proj, dim=-1)
    logits = (100.0 * imf @ text_emb.T).squeeze(0)
    pred = int(logits.argmax().item())
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    attns = [a[0].cpu() for a in vis_out.attentions]
    return pred, probs, _build_attn_cam(attns)


@torch.no_grad()
def en_two_class(model, preprocess, tok, pil_img, true_name, attack_name):
    texts = tok([TMPL_EN.format(w) for w in (true_name, attack_name)]).to(DEVICE)
    text_feat = F.normalize(model.encode_text(texts), dim=-1)
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    img_feat = F.normalize(model.encode_image(x), dim=-1)
    logits = (100.0 * img_feat @ text_feat.T).squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu().numpy() * 100.0
    return {true_name: float(probs[0]), attack_name: float(probs[1])}


def pick_example(true, target, prefer_true="cat", prefer_tgt="bird"):
    prefer_t = CLASSES["en"].index(prefer_true)
    prefer_g = CLASSES["en"].index(prefer_tgt)
    fallback = None
    for i in range(len(true)):
        if int(true[i]) == prefer_t and int(target[i]) == prefer_g:
            return i
        if fallback is None and int(true[i]) == prefer_t:
            fallback = i
    return fallback if fallback is not None else 0


# ── drawing helpers ──────────────────────────────────────────────────────────

def _rounded(ax, xy, w, h, facecolor, edgecolor="#333", lw=1.2, radius=0.02, z=2):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=z,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    return box


def _trapezoid(ax, cx, cy, w, h, facecolor, edgecolor="#333", lw=1.2, z=2):
    """Isosceles trapezoid centered at (cx, cy) in axes fraction coords."""
    top = w * 0.72
    pts = np.array([
        [cx - top / 2, cy + h / 2],
        [cx + top / 2, cy + h / 2],
        [cx + w / 2, cy - h / 2],
        [cx - w / 2, cy - h / 2],
    ])
    poly = Polygon(
        pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=z, transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(poly)
    return poly


def _arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes, clip_on=False, zorder=5,
    )
    ax.add_patch(arr)
    return arr


def _text(ax, x, y, s, **kwargs):
    defaults = dict(
        ha="center", va="center", fontsize=8, color="#222",
        transform=ax.transAxes, zorder=6, clip_on=False,
    )
    defaults.update(kwargs)
    return ax.text(x, y, s, **defaults)


def _freeze_badge(ax, x, y):
    """Small freeze marker (works without emoji fonts)."""
    circ = plt.Circle(
        (x, y), 0.018, facecolor="#E3F2FD", edgecolor="#3A7CA5",
        linewidth=1.0, transform=ax.transAxes, clip_on=False, zorder=7,
    )
    ax.add_patch(circ)
    _text(ax, x, y, "F", fontsize=6.5, color="#3A7CA5", fontweight="bold")


def _imshow_axes(fig, rect, pil_img, title=None, title_fs=8):
    """Add an axes at figure-fraction rect and show PIL image. rect=(l,b,w,h)."""
    ax = fig.add_axes(rect)
    ax.set_zorder(10)
    ax.imshow(np.array(pil_img.convert("RGB")))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444")
        spine.set_linewidth(0.9)
    if title:
        ax.set_title(title, fontsize=title_fs, pad=4, fontweight="bold")
    return ax


def _mini_bars(fig, rect, labels, values, colors, title=None):
    ax = fig.add_axes(rect)
    ax.set_zorder(10)
    ax.set_facecolor("white")
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.65, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, pad=3, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, min(v + 2, 105),
            f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    return ax


def _fig_arrow(fig, x1, y1, x2, y2, color=C_ARROW, lw=1.5, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=fig.transFigure, clip_on=False, zorder=5,
    )
    fig.add_artist(arr)
    return arr


def _fig_box(fig, xy, w, h, facecolor, edgecolor="#333", lw=1.1, radius=0.012):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=0,
    )
    fig.add_artist(box)
    return box


def _fig_trap(fig, cx, cy, w, h, facecolor, edgecolor="#333", lw=1.1):
    top = w * 0.70
    pts = np.array([
        [cx - top / 2, cy + h / 2],
        [cx + top / 2, cy + h / 2],
        [cx + w / 2, cy - h / 2],
        [cx - w / 2, cy - h / 2],
    ])
    poly = Polygon(
        pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, transform=fig.transFigure, clip_on=False, zorder=2,
    )
    fig.add_artist(poly)
    return poly


def _fig_text(fig, x, y, s, **kwargs):
    defaults = dict(
        ha="center", va="center", fontsize=8, color="#222",
        transform=fig.transFigure, zorder=6, clip_on=False,
    )
    defaults.update(kwargs)
    return fig.text(x, y, s, **defaults)


def compute_stages(attacked, clean, en_model, en_pp, en_emb, zh_model, zh_proc, zh_emb):
    pred_en, _, cam_en = en_attn_and_pred(en_model, en_pp, en_emb, attacked)
    pred_zh, _, cam_zh = zh_attn_and_pred(zh_model, zh_proc, zh_emb, attacked)
    inter = np.minimum(align_cam(cam_en), align_cam(cam_zh))
    raw = cam_to_mask(inter, THRESHOLD, dilate=DILATE)
    cc_bbox = filter_mask_components(raw, top_k=TOP_K, bbox_snap=True)
    defended = apply_black(attacked, cc_bbox)
    return {
        "attacked": attacked,
        "pred_en": pred_en,
        "pred_zh": pred_zh,
        "cam_en": cam_en,
        "cam_zh": cam_zh,
        "inter": inter,
        "raw": raw,
        "cc_bbox": cc_bbox,
        "attn_en_img": overlay_heatmap(attacked, cam_en),
        "attn_zh_img": overlay_heatmap(attacked, cam_zh),
        "inter_img": overlay_heatmap(attacked, inter),
        "cc_bbox_img": outline_bbox_mask(overlay_mask(attacked, cc_bbox), cc_bbox),
        "defended": defended,
        "clean": clean,
    }


def draw_method_overview(path, stages, true_name, attack_name, probs_atk, probs_def):
    """Two-panel schematic with non-overlapping vertical zones in panel (b).

      Panel (a)  x in [0.01, 0.54]     Panel (b)  x in [0.56, 0.99]
      attacked -> encoders -> attn     detector
        -> EN∩L -> cc_bbox               /              \
                                     Black fill     No occlusion
                                           Reclassify
                                    before bars | after bars
    """
    fig = plt.figure(figsize=(17.5, 9.0), facecolor="white")
    tw, th = 0.090, 0.145

    fig.add_artist(
        plt.Line2D(
            [0.555, 0.555], [0.07, 0.96],
            transform=fig.transFigure, color="#999",
            linestyle="--", linewidth=1.3, clip_on=False,
        )
    )

    _fig_text(fig, 0.28, 0.975,
              "(a) Localization  —  Attn-last EN ∩ L",
              fontsize=13, fontweight="bold")
    _fig_text(fig, 0.78, 0.975,
              "(b) Gated occlusion  —  cc_bbox_black",
              fontsize=13, fontweight="bold")

    # ── Panel (a) — more vertical gap between EN and L lanes ────────────
    y_top, y_bot = 0.66, 0.24

    _fig_text(fig, 0.070, 0.91, "Attacked image", fontsize=9, fontweight="bold")
    _imshow_axes(fig, [0.025, 0.44, tw + 0.01, th + 0.02], stages["attacked"])
    _fig_text(fig, 0.070, 0.40, f"E+L  '{attack_name}'", fontsize=7.5, color="#555")

    _fig_trap(fig, 0.195, 0.76, 0.095, 0.085, C_ENC_EN)
    _fig_text(fig, 0.195, 0.775, "EN CLIP", fontsize=8, fontweight="bold")
    _fig_text(fig, 0.195, 0.742, "image encoder", fontsize=6, color="#333")
    _fig_text(fig, 0.235, 0.795, "F", fontsize=6.5, color="#3A7CA5", fontweight="bold",
              bbox=dict(boxstyle="circle,pad=0.12", fc="#E3F2FD", ec="#3A7CA5", lw=0.8))

    _fig_trap(fig, 0.195, 0.34, 0.095, 0.085, C_ENC_L)
    _fig_text(fig, 0.195, 0.355, "L CLIP (ZH)", fontsize=8, fontweight="bold")
    _fig_text(fig, 0.195, 0.322, "image encoder", fontsize=6, color="#333")
    _fig_text(fig, 0.235, 0.375, "F", fontsize=6.5, color="#3A7CA5", fontweight="bold",
              bbox=dict(boxstyle="circle,pad=0.12", fc="#E3F2FD", ec="#3A7CA5", lw=0.8))

    _fig_arrow(fig, 0.125, 0.58, 0.150, 0.76)
    _fig_arrow(fig, 0.125, 0.50, 0.150, 0.34)

    _imshow_axes(fig, [0.270, y_top, tw, th], stages["attn_en_img"],
                 title="Attn-last EN")
    _imshow_axes(fig, [0.270, y_bot, tw, th], stages["attn_zh_img"],
                 title="Attn-last L")
    _fig_arrow(fig, 0.243, 0.76, 0.270, 0.74)
    _fig_arrow(fig, 0.243, 0.34, 0.270, 0.32)

    _imshow_axes(fig, [0.400, y_top, tw, th], stages["inter_img"],
                 title="EN ∩ L")
    _fig_text(fig, 0.445, 0.635, "thr ≥ 0.95", fontsize=7, color="#555")

    _imshow_axes(fig, [0.400, y_bot, tw, th], stages["cc_bbox_img"],
                 title="cc_bbox")
    _fig_text(fig, 0.445, 0.215, "dilate · top-2 · bbox", fontsize=7, color="#555")

    _fig_arrow(fig, 0.360, 0.74, 0.400, 0.74)
    _fig_arrow(fig, 0.360, 0.32, 0.400, 0.32)
    _fig_arrow(fig, 0.445, 0.66, 0.445, 0.40)

    _fig_text(fig, 0.035, 0.10, "F", fontsize=7, color="#3A7CA5", fontweight="bold",
              bbox=dict(boxstyle="circle,pad=0.12", fc="#E3F2FD", ec="#3A7CA5", lw=0.8))
    _fig_text(fig, 0.065, 0.10, "= Freeze", fontsize=7.5, ha="left", color="#3A7CA5")

    _fig_arrow(fig, 0.495, 0.74, 0.600, 0.88, color="#666", lw=1.4)

    # ── Panel (b) — clear vertical gaps between zones ──────────────────
    # Zone 1 y≈0.84–0.94: detector
    _fig_box(fig, (0.615, 0.84), 0.23, 0.10, C_GATE, radius=0.01)
    _fig_text(fig, 0.730, 0.915, "Attack detector", fontsize=10, fontweight="bold")
    _fig_text(fig, 0.730, 0.880, "Attn-shape features", fontsize=7, color="#444")
    _fig_text(fig, 0.730, 0.855, "26 scalars → SVM / logistic", fontsize=6.5, color="#666")

    # Zone 2 y≈0.50–0.78: black fill | no occlusion
    _fig_text(fig, 0.640, 0.800, "if attacked", fontsize=8, color="#B00020",
              fontweight="bold")
    _fig_box(fig, (0.570, 0.52), 0.155, 0.25, "#F0F0F0", radius=0.01)
    _fig_text(fig, 0.647, 0.745, "Black fill", fontsize=9, fontweight="bold")
    _imshow_axes(fig, [0.585, 0.545, 0.125, 0.165], stages["defended"])
    _fig_text(fig, 0.647, 0.530, "cc_bbox_black", fontsize=7, color="#555")
    _fig_arrow(fig, 0.680, 0.84, 0.647, 0.77, color="#B00020")

    _fig_text(fig, 0.855, 0.800, "if clean", fontsize=8, color="#2E7D32",
              fontweight="bold")
    _fig_box(fig, (0.775, 0.52), 0.155, 0.25, C_IMG, radius=0.01)
    _fig_text(fig, 0.852, 0.745, "No occlusion", fontsize=9, fontweight="bold")
    _imshow_axes(fig, [0.790, 0.545, 0.125, 0.165], stages["clean"])
    _fig_text(fig, 0.852, 0.530, "pass-through", fontsize=7, color="#555")
    _fig_arrow(fig, 0.790, 0.84, 0.852, 0.77, color="#2E7D32")

    # Zone 3 y≈0.38–0.48: reclassify (gap below branches)
    _fig_trap(fig, 0.750, 0.430, 0.14, 0.075, C_ENC_EN)
    _fig_text(fig, 0.750, 0.445, "Reclassify", fontsize=9, fontweight="bold")
    _fig_text(fig, 0.750, 0.412, "frozen CLIP", fontsize=6.5, color="#333")
    _fig_text(fig, 0.808, 0.455, "F", fontsize=6.5, color="#3A7CA5", fontweight="bold",
              bbox=dict(boxstyle="circle,pad=0.12", fc="#E3F2FD", ec="#3A7CA5", lw=0.8))
    _fig_arrow(fig, 0.647, 0.52, 0.715, 0.455)
    _fig_arrow(fig, 0.852, 0.52, 0.785, 0.455, color="#2E7D32")

    # Zone 4 y≈0.06–0.30: before | after bars (gap below reclassify)
    _fig_arrow(fig, 0.750, 0.392, 0.750, 0.320)

    _fig_box(fig, (0.570, 0.07), 0.155, 0.22, "#EEF4FA", radius=0.01)
    _fig_text(fig, 0.647, 0.265, "Before (attacked)", fontsize=8, fontweight="bold")
    _mini_bars(
        fig, [0.585, 0.090, 0.125, 0.155],
        [true_name, attack_name],
        [probs_atk[true_name], probs_atk[attack_name]],
        ["#A8C8E8", "#F4A6A6"],
    )

    _fig_box(fig, (0.775, 0.07), 0.155, 0.22, C_LOSS, radius=0.01)
    _fig_text(fig, 0.852, 0.265, "After defense", fontsize=8, fontweight="bold")
    _mini_bars(
        fig, [0.790, 0.090, 0.125, 0.155],
        [true_name, attack_name],
        [probs_def[true_name], probs_def[attack_name]],
        ["#A8D5A2", "#F4A6A6"],
    )

    _fig_text(
        fig, 0.50, 0.012,
        "Figure. Method overview. CLIP image encoders stay frozen (F). "
        "(a) Attn-last maps from EN and partner L are intersected, thresholded, and shaped with cc_bbox. "
        "(b) An Attn-shape attack detector gates occlusion: black fill on attacked images; skip on clean "
        f"(L = ZH shown; same recipe for KO / JA).  Example: true={true_name}, stickers='{attack_name}'.",
        fontsize=7.5, color="#333", ha="center", va="bottom",
    )

    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print("Saved", path)



def main():
    require_cuda()

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    attack_pos = saved["attack_pos"]
    idx = saved["idx"]

    hf = load_dataset("uoft-cs/cifar10", split="test")
    label_key = "label" if "label" in hf.column_names else "labels"
    image_key = "img" if "img" in hf.column_names else "image"
    rows = hf.select(idx)
    true = np.array(rows[label_key])
    rng = random.Random(0)
    target = np.array([
        rng.choice([c for c in range(10) if c != int(true[k])])
        for k in range(len(idx))
    ])

    i = pick_example(true, target)
    t, g = int(true[i]), int(target[i])
    true_name = CLASSES["en"][t]
    en_w, zh_w = CLASSES["en"][g], CLASSES["zh"][g]
    xy_en = attack_pos["en"][i]
    xy_l = attack_pos["l"][i]

    clean = rows[i][image_key].convert("RGB").resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    attacked = draw_e_plus_l(rows[i][image_key], en_w, zh_w, xy_en, xy_l)

    print(f"Example i={i} true={true_name} attack=E+L en='{en_w}' zh_idx={g}")
    print("Loading EN CLIP...")
    en_model, _, en_pp = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    en_model = en_model.to(DEVICE).eval()
    en_tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        en_emb = F.normalize(
            en_model.encode_text(
                en_tok([TMPL_EN.format(w) for w in CLASSES["en"]]).to(DEVICE)
            ),
            dim=-1,
        )

    print("Loading ZH CLIP...")
    zh_model = ChineseCLIPModel.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16",
        attn_implementation="eager",
    ).to(DEVICE).eval()
    zh_proc = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16")
    with torch.no_grad():
        t_zh = zh_proc(
            text=[TMPL_ZH.format(w) for w in CLASSES["zh"]],
            padding=True, return_tensors="pt",
        ).to(DEVICE)
        zh_out = zh_model.get_text_features(
            input_ids=t_zh["input_ids"],
            attention_mask=t_zh["attention_mask"],
            token_type_ids=t_zh.get("token_type_ids"),
        )
        if not torch.is_tensor(zh_out):
            zh_out = zh_out.pooler_output
        zh_emb = F.normalize(zh_out, dim=-1)

    print("Computing Attn-last / intersection / cc_bbox...")
    stages = compute_stages(
        attacked, clean, en_model, en_pp, en_emb, zh_model, zh_proc, zh_emb,
    )
    print(
        f"  preds attacked: EN={CLASSES['en'][stages['pred_en']]} "
        f"ZH={CLASSES['en'][stages['pred_zh']]}"
    )

    probs_atk = en_two_class(en_model, en_pp, en_tok, attacked, true_name, en_w)
    probs_def = en_two_class(
        en_model, en_pp, en_tok, stages["defended"], true_name, en_w,
    )
    print(
        f"  2-class % attacked: {true_name}={probs_atk[true_name]:.1f} "
        f"{en_w}={probs_atk[en_w]:.1f}"
    )
    print(
        f"  2-class % defended: {true_name}={probs_def[true_name]:.1f} "
        f"{en_w}={probs_def[en_w]:.1f}"
    )

    out = os.path.join(HERE, "method_overview.png")
    draw_method_overview(out, stages, true_name, en_w, probs_atk, probs_def)

    # Individual assets for manual recreation
    assets = {
        "1_attacked.png": stages["attacked"],
        "2_attn_en.png": stages["attn_en_img"],
        "3_attn_l.png": stages["attn_zh_img"],
        "4_intersection.png": stages["inter_img"],
        "5_cc_bbox.png": stages["cc_bbox_img"],
        "6_black_fill.png": stages["defended"],
        "7_clean.png": stages["clean"],
    }
    for name, pil in assets.items():
        path = os.path.join(HERE, name)
        pil.convert("RGB").resize(
            (DISPLAY_SIZE * 3, DISPLAY_SIZE * 3), Image.NEAREST
        ).save(path)
        print("Saved", path)

    # Probability JSON for the before/after bars
    probs_path = os.path.join(HERE, "8_probs.json")
    with open(probs_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "true": true_name,
                "attack": en_w,
                "before_attacked": probs_atk,
                "after_defense": probs_def,
            },
            f,
            indent=2,
        )
    print("Saved", probs_path)
    print("Done.")


if __name__ == "__main__":
    main()
