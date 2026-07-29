"""Attack-geometry ablation: font size and number of boxes.

Sweeps under frozen gated cc_bbox_black (EN∩ZH):
  font  ∈ {12, 24, 40}  with NUM_BOXES=2 (frozen attack_pos)
  boxes ∈ {1, 2, 3}     with FONT_SIZE=24

Arms per cell: never / always_en_cap_zh / gated_en_cap_zh (black fill).
Gate trained per cell on clean vs that cell's Attn-last features.

CUDA required. Smoke: --n 100. Final: --n 1000.
  python run_ablation.py --sweep font --n 100
  python run_ablation.py --sweep boxes --n 100
  python run_ablation.py --sweep both --n 1000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
IMG_SAMPLES = HERE.parent / "image_samples"
EN_HELPERS = HERE.parent / "en_neglect_vs_blur"
AD = HERE.parent / "attack_detector"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(IMG_SAMPLES))
sys.path.insert(0, str(EN_HELPERS))
sys.path.insert(0, str(AD))

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

from _common.protocol import (  # noqa: E402
    CLASSES,
    DISPLAY_SIZE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    classify_batch,
    load_protocol_data,
    write_summary,
)
from attack_placement import (  # noqa: E402
    BH_EXTRA,
    PAD,
    _font_paths,
    _load_font,
    clamp_xy,
    _rects_overlap,
)
from helpers import (  # noqa: E402
    apply_mask,
    build_cc_bbox_mask,
    classify_and_attn_en,
    classify_and_attn_zh,
    rects_to_mask,
)
from make_phase_ab_viz import build_feature_matrix, train_gate  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
THR = 0.95
DILATE = 3
# Default top_k for font sweep / clean masks (protocol dual-box). Box sweep
# overrides per cell: top_k = num_boxes so capacity matches the threat.
TOP_K = 2
FILL = "black"
FONT_SIZES = (12, 24, 40)
BOX_COUNTS = (1, 2, 3)
GALLERY_IDS = (0, 1, 2, 3)
ARMS = ("never", "always_en_cap_zh", "gated_en_cap_zh")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fonts_at(size: int) -> dict:
    cjk, lat, _ko = _font_paths()
    return {
        "en": _load_font(lat, size),
        "zh": _load_font(cjk, size),
    }


def _draw_one_box(draw, word, font, xy):
    bb = draw.textbbox((0, 0), word, font=font)
    bw = (bb[2] - bb[0]) + 2 * PAD
    bh = (bb[3] - bb[1]) + PAD + BH_EXTRA
    rx, ry = clamp_xy(xy, bw, bh, DISPLAY_SIZE)
    draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
    draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill="black", font=font)
    return (rx, ry, rx + bw, ry + bh)


def _third_xy(img_idx: int, bw: int, bh: int, placed: list) -> tuple[int, int]:
    """Seeded non-overlapping top-left for box 3 (not in frozen attack_pos)."""
    rng = random.Random(int(img_idx) * 3 + 2)
    x_hi = max(0, DISPLAY_SIZE - bw)
    y_hi = max(0, DISPLAY_SIZE - bh)
    rx = ry = 0
    for _ in range(64):
        rx = rng.randint(0, x_hi) if x_hi > 0 else 0
        ry = rng.randint(0, y_hi) if y_hi > 0 else 0
        rect = (rx, ry, rx + bw, ry + bh)
        if all(not _rects_overlap(rect, p) for p in placed):
            return rx, ry
    return rx, ry


def build_attack(data, fonts: dict, num_boxes: int):
    """Return attacked PILs + GT rect lists. num_boxes in {1,2,3}."""
    out, all_rects = [], []
    for i in range(data["n"]):
        t = int(data["target"][i])
        en_w, zh_w = CLASSES["en"][t], CLASSES["zh"][t]
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        img = data["clean_224"][i].copy()
        draw = ImageDraw.Draw(img)
        boxes = []
        boxes.append(_draw_one_box(draw, en_w, fonts["en"], xy0))
        if num_boxes >= 2:
            boxes.append(_draw_one_box(draw, zh_w, fonts["zh"], xy1))
        if num_boxes >= 3:
            bb = draw.textbbox((0, 0), en_w, font=fonts["en"])
            bw = (bb[2] - bb[0]) + 2 * PAD
            bh = (bb[3] - bb[1]) + PAD + BH_EXTRA
            xy2 = _third_xy(i, bw, bh, boxes)
            boxes.append(_draw_one_box(draw, en_w, fonts["en"], xy2))
        out.append(img)
        all_rects.append(boxes)
    return out, all_rects


def compute_cams(en, zh, en_txt, zh_txt, images, label=""):
    cams_en, cams_zh = [], []
    t0 = time.time()
    n = len(images)
    for i, img in enumerate(images):
        _, c_en = classify_and_attn_en(en, en_txt, img, device=DEVICE)
        _, c_zh = classify_and_attn_zh(zh, zh_txt, img, device=DEVICE)
        cams_en.append(c_en)
        cams_zh.append(c_zh)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(
                f"  cams {label} {i+1}/{n}  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return cams_en, cams_zh


def build_masks(cams_en, cams_zh, top_k: int = TOP_K):
    return [
        build_cc_bbox_mask(ce, cz, threshold=THR, dilate=DILATE, top_k=top_k)
        for ce, cz in zip(cams_en, cams_zh)
    ]


def mean_coverage(masks) -> float:
    if not masks:
        return 0.0
    return float(np.mean([m.mean() for m in masks]))


def classify_always(model, text_emb, images, masks):
    defended = [apply_mask(img, m, fill=FILL) for img, m in zip(images, masks)]
    return classify_batch(model, defended, text_emb), mean_coverage(masks)


def classify_gated(model, text_emb, images, masks, gate_flags):
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    out = np.zeros(n, dtype=np.int64)
    def_idx = np.where(gate_flags)[0]
    pass_idx = np.where(~gate_flags)[0]
    if len(pass_idx):
        out[pass_idx] = classify_batch(
            model, [images[i] for i in pass_idx], text_emb
        )
    if len(def_idx):
        def_imgs = [images[i] for i in def_idx]
        def_masks = [masks[i] for i in def_idx]
        defended = [
            apply_mask(img, m, fill=FILL) for img, m in zip(def_imgs, def_masks)
        ]
        out[def_idx] = classify_batch(model, defended, text_emb)
    return out, float(len(def_idx) / max(n, 1))


def train_mode_gate(cams_clean_en, cams_clean_zh, cams_atk_en, cams_atk_zh, n):
    clean_pack = {"cams_en": cams_clean_en, "cams_l": cams_clean_zh}
    atk_pack = {"cams_en": cams_atk_en, "cams_l": cams_atk_zh}
    X, y, img_ids, _names = build_feature_matrix(clean_pack, atk_pack, n)
    gate = train_gate(X, y, img_ids, n)
    return {
        "gate_clean": gate["all_pred"][:n].astype(bool),
        "gate_attacked": gate["all_pred"][n:].astype(bool),
        "threshold": float(gate["threshold"]),
        "primary": gate["primary_name"],
        "fire_clean": float(gate["fire_clean"]),
        "fire_attacked": float(gate["fire_attacked"]),
        "val_auc": float(gate["val_auc"]),
        "test_auc": float(gate["test_auc"]),
    }


def score_arm(preds_en, preds_zh, true, target):
    en_acc, en_asr = acc_asr(preds_en, true, target)
    zh_acc, zh_asr = acc_asr(preds_zh, true, target)
    return {
        "en_acc": en_acc,
        "en_asr": en_asr,
        "zh_acc": zh_acc,
        "zh_asr": zh_asr,
    }


def score_clean_delta(cln_preds_en, cln_preds_zh, true, never_clean_en, never_clean_zh, target):
    cln_en_acc, _ = acc_asr(cln_preds_en, true, target)
    cln_zh_acc, _ = acc_asr(cln_preds_zh, true, target)
    return {
        "clean_en_acc": cln_en_acc,
        "clean_zh_acc": cln_zh_acc,
        "clean_delta_en": cln_en_acc - never_clean_en,
        "clean_delta_zh": cln_zh_acc - never_clean_zh,
    }


def eval_cell(
    label,
    imgs,
    rects,
    en,
    zh,
    en_txt,
    zh_txt,
    clean_imgs,
    cams_clean_en,
    cams_clean_zh,
    masks_clean,
    true,
    target,
    never_clean_en,
    never_clean_zh,
    top_k: int = TOP_K,
):
    print(f"\n--- cell={label} top_k={top_k} ---", flush=True)
    cams_atk_en, cams_atk_zh = compute_cams(
        en, zh, en_txt, zh_txt, imgs, label=label
    )
    masks = build_masks(cams_atk_en, cams_atk_zh, top_k=top_k)
    # Rebuild clean masks at the same top_k so Clean Δ matches attack capacity.
    masks_clean_k = build_masks(cams_clean_en, cams_clean_zh, top_k=top_k)
    gate = train_mode_gate(
        cams_clean_en, cams_clean_zh, cams_atk_en, cams_atk_zh, len(imgs)
    )
    print(
        f"  gate primary={gate['primary']} thr={gate['threshold']:.4f} "
        f"fire_atk={gate['fire_attacked']:.3f} fire_clean={gate['fire_clean']:.3f}",
        flush=True,
    )

    arms = {}
    never_en = classify_batch(en, imgs, en_txt)
    never_zh = classify_batch(zh, imgs, zh_txt)
    arms["never"] = {
        **score_arm(never_en, never_zh, true, target),
        "clean_en_acc": never_clean_en,
        "clean_zh_acc": never_clean_zh,
        "clean_delta_en": 0.0,
        "clean_delta_zh": 0.0,
        "mean_mask_coverage": 0.0,
        "mean_gt_coverage": float(np.mean([rects_to_mask(r).mean() for r in rects])),
    }

    atk_en, cov = classify_always(en, en_txt, imgs, masks)
    atk_zh, _ = classify_always(zh, zh_txt, imgs, masks)
    cln_en, _ = classify_always(en, en_txt, clean_imgs, masks_clean_k)
    cln_zh, _ = classify_always(zh, zh_txt, clean_imgs, masks_clean_k)
    arms["always_en_cap_zh"] = {
        **score_arm(atk_en, atk_zh, true, target),
        **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, target),
        "mean_mask_coverage": cov,
    }

    atk_en, def_frac = classify_gated(
        en, en_txt, imgs, masks, gate["gate_attacked"]
    )
    atk_zh, _ = classify_gated(
        zh, zh_txt, imgs, masks, gate["gate_attacked"]
    )
    cln_en, def_frac_cln = classify_gated(
        en, en_txt, clean_imgs, masks_clean_k, gate["gate_clean"]
    )
    cln_zh, _ = classify_gated(
        zh, zh_txt, clean_imgs, masks_clean_k, gate["gate_clean"]
    )
    arms["gated_en_cap_zh"] = {
        **score_arm(atk_en, atk_zh, true, target),
        **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, target),
        "mean_mask_coverage": mean_coverage(masks),
        "defended_frac_atk": def_frac,
        "defended_frac_clean": def_frac_cln,
        "fire_attacked": gate["fire_attacked"],
        "fire_clean": gate["fire_clean"],
    }

    for arm_name in ARMS:
        a = arms[arm_name]
        print(
            f"  {arm_name:18} EN {fmt_pct(a['en_acc'])}/{fmt_pct(a['en_asr'])} "
            f"ZH {fmt_pct(a['zh_acc'])}/{fmt_pct(a['zh_asr'])} "
            f"dClnEN={100 * a.get('clean_delta_en', 0):+.1f}pp",
            flush=True,
        )

    return {
        "arms": arms,
        "detector": {
            "primary": gate["primary"],
            "threshold": gate["threshold"],
            "fire_attacked": gate["fire_attacked"],
            "fire_clean": gate["fire_clean"],
            "val_auc": gate["val_auc"],
            "test_auc": gate["test_auc"],
        },
        "top_k": int(top_k),
        "images": imgs,
        "masks": masks,
    }


def save_gallery(cells: dict, out_name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    keys = list(cells.keys())
    n_show = min(len(GALLERY_IDS), len(cells[keys[0]]["images"]))
    ids = list(GALLERY_IDS[:n_show])
    ncols = len(keys) * 2
    fig, axes = plt.subplots(len(ids), ncols, figsize=(3.0 * ncols, 3.0 * len(ids)))
    if len(ids) == 1:
        axes = np.array([axes])
    for r, idx in enumerate(ids):
        for c, key in enumerate(keys):
            img = cells[key]["images"][idx]
            m = cells[key]["masks"][idx]
            black = apply_mask(img, m, fill=FILL)
            for col, (pil, title) in enumerate(
                ((img, f"{key} raw#{idx}"), (black, f"{key} black#{idx}"))
            ):
                ax = axes[r, 2 * c + col]
                ax.imshow(np.array(pil.convert("RGB")))
                ax.set_title(title, fontsize=8)
                ax.axis("off")
    fig.tight_layout()
    path = FIGURES / out_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


def strip_images(summary_cells: dict) -> dict:
    """Drop PIL/mask blobs before JSON write."""
    out = {}
    for k, v in summary_cells.items():
        out[k] = {
            "arms": v["arms"],
            "detector": v["detector"],
            "top_k": v.get("top_k", TOP_K),
        }
    return out


def run(n: int, sweep: str, save_gallery_flag: bool = True):
    data = load_protocol_data(n=n)
    true = data["true"]
    target = data["target"]
    clean_imgs = data["clean_224"]

    print("Loading EN + ZH CLIP…", flush=True)
    en = EnCLIP()
    zh = ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    print("Clean never baseline…", flush=True)
    never_clean_en_preds = classify_batch(en, clean_imgs, en_txt)
    never_clean_zh_preds = classify_batch(zh, clean_imgs, zh_txt)
    never_clean_en, _ = acc_asr(never_clean_en_preds, true, target)
    never_clean_zh, _ = acc_asr(never_clean_zh_preds, true, target)

    print("Clean Attn-last cams…", flush=True)
    cams_clean_en, cams_clean_zh = compute_cams(
        en, zh, en_txt, zh_txt, clean_imgs, label="clean"
    )
    masks_clean = build_masks(cams_clean_en, cams_clean_zh)

    base = {
        "n": int(data["n"]),
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "fill": FILL,
        "partner": "zh",
        "attack": "multi",
        "arms_list": list(ARMS),
        "clean_never": {"en_acc": never_clean_en, "zh_acc": never_clean_zh},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)

    if sweep in ("font", "both"):
        cells = {}
        for fs in FONT_SIZES:
            fonts = fonts_at(fs)
            imgs, rects = build_attack(data, fonts, num_boxes=2)
            label = f"font{fs}"
            cell = eval_cell(
                label,
                imgs,
                rects,
                en,
                zh,
                en_txt,
                zh_txt,
                clean_imgs,
                cams_clean_en,
                cams_clean_zh,
                masks_clean,
                true,
                target,
                never_clean_en,
                never_clean_zh,
            )
            cells[label] = cell
        if save_gallery_flag:
            save_gallery(cells, f"gallery_font_n{data['n']}.png")
        summary = {
            **base,
            "sweep": "font",
            "font_sizes": list(FONT_SIZES),
            "num_boxes": 2,
            "cells": strip_images(cells),
            "note": "Font-size sweep at frozen dual-box attack_pos; protocol default=24.",
        }
        write_summary(RESULTS / f"font_n{data['n']}.json", summary)

    if sweep in ("boxes", "both"):
        cells = {}
        fonts = fonts_at(24)
        for nb in BOX_COUNTS:
            imgs, rects = build_attack(data, fonts, num_boxes=nb)
            label = f"boxes{nb}"
            # Match mask capacity to the number of stickers (1→1, 2→2, 3→3).
            cell = eval_cell(
                label,
                imgs,
                rects,
                en,
                zh,
                en_txt,
                zh_txt,
                clean_imgs,
                cams_clean_en,
                cams_clean_zh,
                masks_clean,
                true,
                target,
                never_clean_en,
                never_clean_zh,
                top_k=nb,
            )
            cells[label] = cell
        if save_gallery_flag:
            save_gallery(cells, f"gallery_boxes_n{data['n']}.png")
        summary = {
            **base,
            "sweep": "boxes",
            "font_size": 24,
            "box_counts": list(BOX_COUNTS),
            "top_k_policy": "top_k = num_boxes (capacity matched to threat)",
            "cells": strip_images(cells),
            "note": (
                "Box-count sweep at FONT_SIZE=24. boxes=1 uses frozen EN anchor only; "
                "boxes=2 is protocol dual EN+ZH; boxes=3 adds seeded third EN box. "
                "Mask top_k equals num_boxes so three stickers are not capacity-capped."
            ),
        }
        write_summary(RESULTS / f"boxes_n{data['n']}.json", summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="subset size (100 smoke / 1000 final)")
    ap.add_argument(
        "--sweep",
        choices=("font", "boxes", "both"),
        default="both",
        help="which geometry sweep to run",
    )
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    run(n=args.n, sweep=args.sweep, save_gallery_flag=not args.no_gallery)


if __name__ == "__main__":
    main()
