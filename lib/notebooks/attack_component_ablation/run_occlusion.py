"""Content-mode gated black occlusion: full vs white_only vs text_only.

Protocol: font 24, dual-box attack_pos, EN∩ZH cc_bbox_black, thr ≥ 0.95.
Arms per mode: never / always_en_cap_zh / gated_en_cap_zh.
Gate trained per mode on clean vs that mode's Attn-last features.

CUDA required. Smoke: --n 100. Final: --n 1000.
  python run_occlusion.py --n 100
  python run_occlusion.py --n 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
IMG_SAMPLES = HERE.parent / "image_samples"
EN_HELPERS = HERE.parent / "en_neglect_vs_blur"
AD = HERE.parent / "attack_detector"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(IMG_SAMPLES))
sys.path.insert(0, str(EN_HELPERS))
sys.path.insert(0, str(AD))
sys.path.insert(0, str(HERE))

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

from _common.protocol import (  # noqa: E402
    CLASSES,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    classify_batch,
    load_protocol_data,
    write_summary,
)
from helpers import (  # noqa: E402
    apply_mask,
    build_cc_bbox_mask,
    classify_and_attn_en,
    classify_and_attn_zh,
    rects_to_mask,
)
from make_phase_ab_viz import build_feature_matrix, train_gate  # noqa: E402
from run_ablation import (  # noqa: E402
    DRAW_MODES,
    MODES,
    _fonts,
    build_mode_attack,
)

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
THR = 0.95
DILATE = 3
TOP_K = 2
FILL = "black"
GALLERY_IDS = (0, 1, 2, 3)
ARMS = ("never", "always_en_cap_zh", "gated_en_cap_zh")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


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


def score_clean_delta(
    cln_preds_en, cln_preds_zh, true, never_clean_en, never_clean_zh, target
):
    cln_en_acc, _ = acc_asr(cln_preds_en, true, target)
    cln_zh_acc, _ = acc_asr(cln_preds_zh, true, target)
    return {
        "clean_en_acc": cln_en_acc,
        "clean_zh_acc": cln_zh_acc,
        "clean_delta_en": cln_en_acc - never_clean_en,
        "clean_delta_zh": cln_zh_acc - never_clean_zh,
    }


def eval_mode(
    mode,
    imgs,
    rects,
    en,
    zh,
    en_txt,
    zh_txt,
    clean_imgs,
    cams_clean_en,
    cams_clean_zh,
    true,
    target,
    never_clean_en,
    never_clean_zh,
):
    print(f"\n--- mode={mode} ---", flush=True)
    cams_atk_en, cams_atk_zh = compute_cams(
        en, zh, en_txt, zh_txt, imgs, label=mode
    )
    masks = build_masks(cams_atk_en, cams_atk_zh)
    masks_clean = build_masks(cams_clean_en, cams_clean_zh)
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
        "mean_gt_coverage": float(
            np.mean([rects_to_mask(r).mean() for r in rects])
        ),
    }

    atk_en, cov = classify_always(en, en_txt, imgs, masks)
    atk_zh, _ = classify_always(zh, zh_txt, imgs, masks)
    cln_en, _ = classify_always(en, en_txt, clean_imgs, masks_clean)
    cln_zh, _ = classify_always(zh, zh_txt, clean_imgs, masks_clean)
    arms["always_en_cap_zh"] = {
        **score_arm(atk_en, atk_zh, true, target),
        **score_clean_delta(
            cln_en, cln_zh, true, never_clean_en, never_clean_zh, target
        ),
        "mean_mask_coverage": cov,
    }

    atk_en, def_frac = classify_gated(
        en, en_txt, imgs, masks, gate["gate_attacked"]
    )
    atk_zh, _ = classify_gated(
        zh, zh_txt, imgs, masks, gate["gate_attacked"]
    )
    cln_en, def_frac_cln = classify_gated(
        en, en_txt, clean_imgs, masks_clean, gate["gate_clean"]
    )
    cln_zh, _ = classify_gated(
        zh, zh_txt, clean_imgs, masks_clean, gate["gate_clean"]
    )
    arms["gated_en_cap_zh"] = {
        **score_arm(atk_en, atk_zh, true, target),
        **score_clean_delta(
            cln_en, cln_zh, true, never_clean_en, never_clean_zh, target
        ),
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
    out = {}
    for k, v in summary_cells.items():
        out[k] = {"arms": v["arms"], "detector": v["detector"]}
    return out


def run(n: int, save_gallery_flag: bool = True):
    data = load_protocol_data(n=n)
    true = data["true"]
    target = data["target"]
    clean_imgs = data["clean_224"]
    fonts = _fonts()

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

    cells = {}
    for mode in MODES:
        assert mode in DRAW_MODES
        imgs, rects = build_mode_attack(data, mode, fonts)
        cells[mode] = eval_mode(
            mode,
            imgs,
            rects,
            en,
            zh,
            en_txt,
            zh_txt,
            clean_imgs,
            cams_clean_en,
            cams_clean_zh,
            true,
            target,
            never_clean_en,
            never_clean_zh,
        )

    if save_gallery_flag:
        save_gallery(cells, f"gallery_content_occlusion_n{n}.png")

    summary = {
        "method": "content_occlusion_cc_bbox_black",
        "n": int(data["n"]),
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "fill": FILL,
        "partner": "zh",
        "attack": "multi",
        "modes": list(MODES),
        "arms_list": list(ARMS),
        "clean_never": {"en_acc": never_clean_en, "zh_acc": never_clean_zh},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cells": strip_images(cells),
        "note": (
            "Gated black recovery on full / white_only / text_only dual-box modes. "
            "Gate trained per-mode on clean vs mode-attacked Attn-last features."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"content_occlusion_n{n}.json"
    write_summary(out, summary)
    print("Wrote", out)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--no-gallery", action="store_true")
    args = p.parse_args()
    run(n=args.n, save_gallery_flag=not args.no_gallery)


if __name__ == "__main__":
    main()
