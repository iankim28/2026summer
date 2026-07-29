"""Make Introduction figures for the paper.

Outputs (this directory):
  1_attacked.png          — typographic E+L dual-box attack
  2_occluded.png          — same image after solid-black sticker occlusion
  3_class_distribution.png — EN CLIP 2-class probs (true vs sticker) on both images
  intro_figure.png        — composite (a)(b)(c) for the paper

Uses frozen CIFAR-10 sample + attack_pos. Occlusion uses the drawn sticker
bounding boxes (illustrative GT occlusion matching production black fill).
EN CLIP: OpenAI ViT-B/32 via open_clip (CUDA required).
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
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(
    HERE, "..", "..", "..", "lib", "notebooks", "image_samples",
    "CIFAR10_BALANCED_1000_SAMPLE.json",
)

DISPLAY_SIZE = 224
UPSCALE = 3  # save larger for paper
FONT_SIZE = 24
PAD = 8
TMPL_EN = "a photo of a {}."

CLASSES = {
    "en": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}

COLOR_CLIP = "#B8D4EA"       # light blue — attacked / CLIP
COLOR_OURS = "#B8DDB4"       # light green — occluded / CLIP+Ours
EDGE_CLIP = "#5B8DB8"
EDGE_OURS = "#5A9A55"


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


def draw_e_plus_l(img, en_word, zh_word, xy_en, xy_l):
    """Draw dual-box E+L attack; return (pil, list of (x0,y0,x1,y1) boxes)."""
    img = img.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    boxes = []
    for word, fp, xy in [
        (en_word, _LAT, xy_en),
        (zh_word, _CJK, xy_l),
    ]:
        font = _get_font(fp)
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        box = (rx, ry, rx + bw, ry + bh)
        draw.rectangle(list(box), fill="white")
        draw.text(
            (rx + PAD - bb[0], ry + PAD - bb[1]),
            word, fill="black", font=font,
        )
        boxes.append(box)
    return img, boxes


def black_occlude(img, boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle(list(box), fill="black")
    return out


def upscale(pil):
    w = DISPLAY_SIZE * UPSCALE
    return pil.resize((w, w), Image.NEAREST)


def save_panel(pil, path, title, subtitle):
    """Save image with a clean title bar for paper use."""
    img = upscale(pil)
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        0.5, -0.04, subtitle,
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", path)


def require_cuda():
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is required for intro CLIP scoring, but "
            f"torch.cuda.is_available() is False (torch={torch.__version__}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Device: cuda ({torch.__version__})")


@torch.no_grad()
def en_clip_two_class_probs(attacked, occluded, true_name, attack_name, device="cuda"):
    """EN OpenAI ViT-B/32: 2-class softmax % over {true, attack} for both images.

    Returns dict with keys attacked/occluded → {true_name: %, attack_name: %}.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai",
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    texts = tok([TMPL_EN.format(w) for w in (true_name, attack_name)]).to(device)
    text_feat = F.normalize(model.encode_text(texts), dim=-1)

    out = {}
    for key, pil in (("attacked", attacked), ("occluded", occluded)):
        x = preprocess(pil.convert("RGB")).unsqueeze(0).to(device)
        img_feat = F.normalize(model.encode_image(x), dim=-1)
        logits = (100.0 * img_feat @ text_feat.T).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy() * 100.0
        out[key] = {true_name: float(probs[0]), attack_name: float(probs[1])}
    return out


def _draw_distribution_ax(ax, true_name, attack_name, probs, title="Probability (%)"):
    """Grouped bars with per-bar value labels (avoids y-axis overlap on extreme %)."""
    classes = [true_name, attack_name]
    clip_vals = [round(probs["attacked"][c], 1) for c in classes]
    ours_vals = [round(probs["occluded"][c], 1) for c in classes]

    x = np.arange(len(classes), dtype=float) * 1.5
    width = 0.32
    bars_clip = ax.bar(
        x - width / 2, clip_vals, width,
        label="CLIP", color=COLOR_CLIP, edgecolor=EDGE_CLIP, linewidth=1.1, zorder=3,
    )
    bars_ours = ax.bar(
        x + width / 2, ours_vals, width,
        label="CLIP + Ours", color=COLOR_OURS, edgecolor=EDGE_OURS, linewidth=1.1, zorder=3,
    )

    def _label_bar(bar, value):
        cx = bar.get_x() + bar.get_width() / 2
        if value >= 50:
            # Inside tall bars — keeps title clear
            ax.text(
                cx, value - 3.5, f"{value:.1f}",
                ha="center", va="top", fontsize=10, fontweight="bold",
                color="#1a1a1a", zorder=4,
            )
        else:
            ax.text(
                cx, value + 2.5, f"{value:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#1a1a1a", zorder=4,
            )

    for bar, v in zip(bars_clip, clip_vals):
        _label_bar(bar, v)
    for bar, v in zip(bars_ours, ours_vals):
        _label_bar(bar, v)

    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=9, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_ylim(0, 112)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    # Legend in the gap under the title (above the bars)
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=2,
        handlelength=1.15, handleheight=0.85, columnspacing=1.2,
        borderaxespad=0.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ax.plot(xmin, 112, "^", color="black", markersize=5, clip_on=False, zorder=5)
    ax.plot(xmax, 0, ">", color="black", markersize=5, clip_on=False, zorder=5)
    ax.tick_params(axis="y", length=3, colors="#444")
    ax.tick_params(axis="x", length=0, pad=6)


def save_class_distribution(path, true_name, attack_name, probs):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    _draw_distribution_ax(ax, true_name, attack_name, probs, title="Probability (%)")
    fig.subplots_adjust(left=0.22, right=0.96, top=0.82, bottom=0.12)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print("Saved", path)


def save_composite(path, attacked, occluded, true_name, attack_name, probs):
    """(a) attacked | (b) occluded | (c) class distribution."""
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4], wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(upscale(attacked))
    ax_a.set_title("(a) Typographic attack", fontsize=12, fontweight="bold", pad=6)
    ax_a.text(
        0.5, -0.02,
        f"true = {true_name}  ·  stickers = '{attack_name}' + ZH",
        transform=ax_a.transAxes, ha="center", va="top", fontsize=8, color="#333",
    )
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(upscale(occluded))
    ax_b.set_title("(b) After black occlusion", fontsize=12, fontweight="bold", pad=6)
    ax_b.text(
        0.5, -0.02,
        "solid black fill over sticker boxes",
        transform=ax_b.transAxes, ha="center", va="top", fontsize=8, color="#333",
    )
    ax_b.axis("off")

    ax_c = fig.add_subplot(gs[0, 2])
    _draw_distribution_ax(ax_c, true_name, attack_name, probs, title="(c) Probability (%)")

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print("Saved", path)


def pick_example(rows, image_key, true, target, attack_pos, prefer_true="dog", prefer_tgt="ship"):
    """Prefer dog←ship (matches existing pipeline viz); else first sample."""
    prefer_t = CLASSES["en"].index(prefer_true)
    prefer_g = CLASSES["en"].index(prefer_tgt)
    fallback = None
    for i in range(len(true)):
        if int(true[i]) == prefer_t and int(target[i]) == prefer_g:
            return i
        if fallback is None and int(true[i]) == prefer_t:
            fallback = i
    return fallback if fallback is not None else 0


def main():
    assert platform.system() == "Windows" or os.path.isfile(_LAT)
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

    i = pick_example(rows, image_key, true, target, attack_pos)
    t, g = int(true[i]), int(target[i])
    true_name = CLASSES["en"][t]
    en_w, zh_w = CLASSES["en"][g], CLASSES["zh"][g]
    xy_en = attack_pos["en"][i]
    xy_l = attack_pos["l"][i]

    attacked, boxes = draw_e_plus_l(rows[i][image_key], en_w, zh_w, xy_en, xy_l)
    occluded = black_occlude(attacked, boxes)

    print(
        f"Example i={i} true={true_name} "
        f"attack=E+L en='{en_w}' zh_idx={g} boxes={boxes}"
    )

    save_panel(
        attacked,
        os.path.join(HERE, "1_attacked.png"),
        "Typographic attack (E + L)",
        f"true = {true_name}   ·   stickers = '{en_w}' + ZH class name",
    )
    save_panel(
        occluded,
        os.path.join(HERE, "2_occluded.png"),
        "After black occlusion",
        "solid black fill over localized sticker boxes (cc_bbox_black)",
    )

    probs = en_clip_two_class_probs(attacked, occluded, true_name, en_w)
    print(
        f"2-class softmax %  |  CLIP attacked: "
        f"{true_name}={probs['attacked'][true_name]:.1f}  "
        f"{en_w}={probs['attacked'][en_w]:.1f}"
    )
    print(
        f"2-class softmax %  |  CLIP+Ours occluded: "
        f"{true_name}={probs['occluded'][true_name]:.1f}  "
        f"{en_w}={probs['occluded'][en_w]:.1f}"
    )

    save_class_distribution(
        os.path.join(HERE, "3_class_distribution.png"),
        true_name, en_w, probs,
    )
    save_composite(
        os.path.join(HERE, "intro_figure.png"),
        attacked, occluded, true_name, en_w, probs,
    )
    print("Done.")


if __name__ == "__main__":
    main()
