"""Make Dataset / evaluation-protocol figure for the paper.

Outputs (this directory):
  dataset_figure.png   — 2x8 grid: two class groups per row x (Clean | Pure E | E+L | Pure L)
  picks.json           — chosen indices / true / target / attack_pos
  Optional single-row panels kept for the dog example (1_clean … 4_pure_l)

Uses frozen CIFAR-10 balanced n=1000 sample + attack_pos.
Render-only (no CLIP / CUDA).
"""
from __future__ import annotations

import json
import os
import platform
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(
    HERE, "..", "..", "..", "lib", "notebooks", "image_samples",
    "CIFAR10_BALANCED_1000_SAMPLE.json",
)

DISPLAY_SIZE = 224
UPSCALE = 3
FONT_SIZE = 24
PAD = 8
BH_EXTRA = 12

CLASSES = {
    "en": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}

# Preferred (true, target) pairs — diverse classes + stickers.
# First row matches intro/method/dataset dog<-ship continuity.
PREFERRED_ROWS = [
    ("dog", "ship"),
    ("airplane", "truck"),
    ("frog", "cat"),
    ("truck", "bird"),
]

COL_TITLES = [
    "(a) Clean",
    "(b) Pure E (EN+EN)",
    "(c) E + L (EN+ZH)",
    "(d) Pure L (ZH+ZH)",
]


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


def to_display(img):
    return img.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)


def draw_dual_box(img, word0, word1, xy_en, xy_l, font0, font1):
    img = to_display(img)
    draw = ImageDraw.Draw(img)
    boxes = []
    for word, font, xy in [
        (word0, font0, xy_en),
        (word1, font1, xy_l),
    ]:
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + BH_EXTRA
        rx, ry = _clamp_xy(xy, bw, bh)
        box = (rx, ry, rx + bw, ry + bh)
        draw.rectangle(list(box), fill="white")
        draw.text(
            (rx + PAD - bb[0], ry + PAD - bb[1]),
            word, fill="black", font=font,
        )
        boxes.append(box)
    return img, boxes


def upscale(pil):
    w = DISPLAY_SIZE * UPSCALE
    return pil.resize((w, w), Image.NEAREST)


def _pos_key(xy_en, xy_l, bucket=48):
    """Coarse layout signature so rows get visually distinct sticker placements."""
    return (
        int(xy_en[0]) // bucket,
        int(xy_en[1]) // bucket,
        int(xy_l[0]) // bucket,
        int(xy_l[1]) // bucket,
    )


def pick_rows(true, target, attack_pos, preferred=PREFERRED_ROWS):
    """One index per preferred (true, target); enforce distinct attack_pos layouts."""
    used_pos = set()
    picks = []
    for true_name, tgt_name in preferred:
        prefer_t = CLASSES["en"].index(true_name)
        prefer_g = CLASSES["en"].index(tgt_name)
        chosen = None
        # Exact true+target with unused layout
        for i in range(len(true)):
            if int(true[i]) != prefer_t or int(target[i]) != prefer_g:
                continue
            key = _pos_key(attack_pos["en"][i], attack_pos["l"][i])
            if key in used_pos:
                continue
            chosen = i
            break
        # Fallback: any image of that true class with unused layout
        if chosen is None:
            for i in range(len(true)):
                if int(true[i]) != prefer_t:
                    continue
                key = _pos_key(attack_pos["en"][i], attack_pos["l"][i])
                if key in used_pos:
                    continue
                chosen = i
                break
        # Last resort: first of that class
        if chosen is None:
            for i in range(len(true)):
                if int(true[i]) == prefer_t:
                    chosen = i
                    break
        if chosen is None:
            raise RuntimeError(f"no sample for true={true_name}")
        used_pos.add(_pos_key(attack_pos["en"][chosen], attack_pos["l"][chosen]))
        picks.append(chosen)
    return picks


def render_row(raw, en_w, zh_w, xy_en, xy_l, font_en, font_zh):
    clean = to_display(raw)
    pure_e, _ = draw_dual_box(raw, en_w, en_w, xy_en, xy_l, font_en, font_en)
    e_plus_l, _ = draw_dual_box(raw, en_w, zh_w, xy_en, xy_l, font_en, font_zh)
    pure_l, _ = draw_dual_box(raw, zh_w, zh_w, xy_en, xy_l, font_zh, font_zh)
    return [clean, pure_e, e_plus_l, pure_l]


def save_composite(path, grid, row_labels):
    """grid: list of 4 class-rows (4 PILs each) → flatten to 2x8 (two classes per row)."""
    assert len(grid) == 4 and all(len(r) == 4 for r in grid)
    # Row 0: classes 0|1 ; Row 1: classes 2|3
    flat = [
        grid[0] + grid[1],
        grid[2] + grid[3],
    ]
    flat_labels = [
        [row_labels[0], row_labels[1]],
        [row_labels[2], row_labels[3]],
    ]

    fig, axes = plt.subplots(2, 8, figsize=(22.0, 5.8))
    for r in range(2):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(upscale(flat[r][c]))
            ax.axis("off")
            if r == 0:
                ax.set_title(COL_TITLES[c % 4], fontsize=10, fontweight="bold", pad=6)
        # Class labels under each group of 4
        for g, lab in enumerate(flat_labels[r]):
            mid = axes[r, g * 4 + 1]
            mid.annotate(
                lab,
                xy=(1.05, -0.06),
                xycoords="axes fraction",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="top",
                color="#333333",
            )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10, wspace=0.06, hspace=0.28)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print("Saved", path)


def save_dog_panels(row_pils, true_name, en_w):
    """Keep legacy single panels for the first (dog) row."""
    names = ["1_clean.png", "2_pure_e.png", "3_e_plus_l.png", "4_pure_l.png"]
    titles = ["Clean", "Pure E (EN+EN)", "E + L (EN+ZH)", "Pure L (ZH+ZH)"]
    subs = [
        f"true = {true_name}  ·  CIFAR-10 balanced n=1000",
        f"stickers = '{en_w}' + '{en_w}'",
        f"stickers = '{en_w}' + ZH class name",
        "stickers = ZH class name x2",
    ]
    for pil, name, title, sub in zip(row_pils, names, titles, subs):
        fig, ax = plt.subplots(figsize=(4.2, 4.8))
        ax.imshow(upscale(pil))
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.5, -0.04, sub,
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333",
        )
        ax.axis("off")
        fig.tight_layout()
        out = os.path.join(HERE, name)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved", out)


def main():
    assert platform.system() == "Windows" or os.path.isfile(_LAT)

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

    picks = pick_rows(true, target, attack_pos)
    font_en = _get_font(_LAT)
    font_zh = _get_font(_CJK)

    grid = []
    row_labels = []
    pick_records = []

    for i in picks:
        t, g = int(true[i]), int(target[i])
        true_name = CLASSES["en"][t]
        en_w, zh_w = CLASSES["en"][g], CLASSES["zh"][g]
        xy_en = attack_pos["en"][i]
        xy_l = attack_pos["l"][i]
        raw = rows[i][image_key]
        row_pils = render_row(raw, en_w, zh_w, xy_en, xy_l, font_en, font_zh)
        grid.append(row_pils)
        row_labels.append(f"{true_name} → {en_w}")
        pick_records.append({
            "i": int(i),
            "true": true_name,
            "target": en_w,
            "xy_en": list(map(int, xy_en)),
            "xy_l": list(map(int, xy_l)),
        })
        print(
            f"Row i={i} true={true_name} target={en_w} "
            f"xy_en={xy_en} xy_l={xy_l}"
        )

    save_dog_panels(grid[0], pick_records[0]["true"], pick_records[0]["target"])
    save_composite(os.path.join(HERE, "dataset_figure.png"), grid, row_labels)

    picks_path = os.path.join(HERE, "picks.json")
    with open(picks_path, "w", encoding="utf-8") as f:
        json.dump({"columns": COL_TITLES, "rows": pick_records}, f, indent=2)
    print("Wrote", picks_path)
    print("Done.")


if __name__ == "__main__":
    main()
