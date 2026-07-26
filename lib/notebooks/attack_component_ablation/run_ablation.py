"""Attack-component attention ablation: full vs white_only vs text_only.

Measures Attn-last localization (inbox focus, IoU, detect@IoU, peak-in-box)
and undefended accuracy/ASR for EN, ZH, and EN∩ZH mask localization.

CUDA required. Smoke: --n 100. Final: --n 1000.
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
from PIL import Image

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
IMG_SAMPLES = HERE.parent / "image_samples"
EN_HELPERS = HERE.parent / "en_neglect_vs_blur"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(IMG_SAMPLES))
sys.path.insert(0, str(EN_HELPERS))

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
    DRAW_MODES,
    FONT_SIZE,
    PAD,
    _font_paths,
    _load_font,
    draw_dual_box_at,
)
from helpers import (  # noqa: E402
    align_cam,
    build_cc_bbox_mask,
    cam_to_mask,
    classify_and_attn_en,
    classify_and_attn_zh,
    filter_mask_components,
    n_cam_intersection,
    rects_to_mask,
)

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
MODES = list(DRAW_MODES)  # full, white_only, text_only
GALLERY_IDS = (0, 1, 2, 3)
THR = 0.95
DILATE = 3
TOP_K = 2


def _fonts():
    cjk, lat, ko = _font_paths()
    return {
        "en": _load_font(lat, FONT_SIZE),
        "zh": _load_font(cjk, FONT_SIZE),
    }


def build_mode_attack(data, mode: str, fonts: dict):
    """Return attacked PIL list + per-image GT rects [(x0,y0,x1,y1), ...]."""
    out, all_rects = [], []
    for i in range(data["n"]):
        t = int(data["target"][i])
        en_w, zh_w = CLASSES["en"][t], CLASSES["zh"][t]
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        img, boxes = draw_dual_box_at(
            data["clean_224"][i],
            en_w,
            fonts["en"],
            zh_w,
            fonts["zh"],
            xy0,
            xy1,
            already_224=True,
            pad=PAD,
            mode=mode,
            return_boxes=True,
        )
        out.append(img)
        all_rects.append(boxes)
    return out, all_rects


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def inbox_focus_ratio(saliency: np.ndarray, gt_mask: np.ndarray) -> float:
    s = np.asarray(saliency, dtype=np.float64)
    m = gt_mask.astype(bool)
    if not m.any() or s.sum() <= 0:
        return 0.0
    mass_frac = float(s[m].sum() / (s.sum() + 1e-9))
    area_frac = float(m.mean() + 1e-9)
    return mass_frac / area_frac


def peak_in_box(saliency: np.ndarray, gt_mask: np.ndarray) -> bool:
    s = np.asarray(saliency)
    if s.size == 0 or not gt_mask.any():
        return False
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return bool(gt_mask[y, x])


def mask_from_single_cam(cam, threshold=THR, dilate=DILATE, top_k=TOP_K):
    s = align_cam(cam)
    mask = cam_to_mask(s, threshold=threshold, dilate=dilate)
    return filter_mask_components(mask, top_k=top_k, bbox_snap=True)


def overlay_cam(pil_img, cam, alpha=0.45):
    """Jet colormap blend for gallery panels."""
    s = align_cam(cam)
    cmap = plt.get_cmap("jet")
    heat = (cmap(s)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(pil_img.convert("RGB"))
    blend = (base.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(
        np.uint8
    )
    return Image.fromarray(blend)


def localization_stats(cams, gt_masks, pred_masks):
    inbox, ious, peak, det01, det03 = [], [], [], [], []
    for cam, gt, pred in zip(cams, gt_masks, pred_masks):
        s = align_cam(cam)
        inbox.append(inbox_focus_ratio(s, gt))
        ious.append(iou(pred, gt))
        peak.append(peak_in_box(s, gt))
        iv = iou(pred, gt)
        det01.append(iv >= 0.1)
        det03.append(iv >= 0.3)
    return {
        "inbox_focus_mean": float(np.mean(inbox)),
        "iou_mean": float(np.mean(ious)),
        "peak_in_box": float(np.mean(peak)),
        "detect_iou_ge_0.1": float(np.mean(det01)),
        "detect_iou_ge_0.3": float(np.mean(det03)),
    }


def compute_cams(en, zh, en_txt, zh_txt, images, label=""):
    cams_en, cams_zh = [], []
    preds_en, preds_zh = [], []
    t0 = time.time()
    n = len(images)
    for i, img in enumerate(images):
        p_en, c_en = classify_and_attn_en(en, en_txt, img, device=DEVICE)
        p_zh, c_zh = classify_and_attn_zh(zh, zh_txt, img, device=DEVICE)
        preds_en.append(p_en)
        preds_zh.append(p_zh)
        cams_en.append(c_en)
        cams_zh.append(c_zh)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(
                f"  cams {label} {i+1}/{n}  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return (
        np.array(preds_en),
        np.array(preds_zh),
        cams_en,
        cams_zh,
    )


def save_gallery(mode_images, mode_cams_en, mode_cams_zh, modes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(len(GALLERY_IDS), len(mode_images[modes[0]]))
    ids = list(GALLERY_IDS[:n_show])
    fig, axes = plt.subplots(len(ids), len(modes) * 2, figsize=(4 * len(modes) * 2, 4 * len(ids)))
    if len(ids) == 1:
        axes = np.array([axes])
    for r, idx in enumerate(ids):
        for c, mode in enumerate(modes):
            img = mode_images[mode][idx]
            cam_en = mode_cams_en[mode][idx]
            cam_zh = mode_cams_zh[mode][idx]
            axes[r, 2 * c].imshow(overlay_cam(img, cam_en))
            axes[r, 2 * c].set_title(f"{mode} EN#{idx}")
            axes[r, 2 * c].axis("off")
            axes[r, 2 * c + 1].imshow(overlay_cam(img, cam_zh))
            axes[r, 2 * c + 1].set_title(f"{mode} ZH#{idx}")
            axes[r, 2 * c + 1].axis("off")
    fig.tight_layout()
    path = out_dir / "gallery_attn_overlay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def print_table(summary: dict):
    print("\n=== Attack-component ablation ===")
    header = (
        f"{'mode':12} {'map':10} {'acc':>7} {'ASR':>7} "
        f"{'inbox':>7} {'IoU':>7} {'peak':>7} {'det@.1':>7} {'det@.3':>7}"
    )
    print(header)
    for mode in MODES:
        block = summary["modes"][mode]
        for map_name in ("en", "zh", "en_cap_zh"):
            loc = block["localization"][map_name]
            if map_name == "en":
                acc, asr = block["en_acc"], block["en_asr"]
            elif map_name == "zh":
                acc, asr = block["zh_acc"], block["zh_asr"]
            else:
                acc = asr = float("nan")
            print(
                f"{mode:12} {map_name:10} "
                f"{fmt_pct(acc) if acc == acc else '   n/a':>7} "
                f"{fmt_pct(asr) if asr == asr else '   n/a':>7} "
                f"{loc['inbox_focus_mean']:7.2f} "
                f"{loc['iou_mean']:7.3f} "
                f"{fmt_pct(loc['peak_in_box']):>7} "
                f"{fmt_pct(loc['detect_iou_ge_0.1']):>7} "
                f"{fmt_pct(loc['detect_iou_ge_0.3']):>7}"
            )


def run(n: int, save_gallery_flag: bool = True):
    fonts = _fonts()
    data = load_protocol_data(n=n)
    true, target = data["true"], data["target"]

    print("Loading EN + ZH CLIP…", flush=True)
    en = EnCLIP()
    zh = ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    # Clean baseline
    print("Clean baseline…", flush=True)
    clean_en = classify_batch(en, data["clean_224"], en_txt)
    clean_zh = classify_batch(zh, data["clean_224"], zh_txt)
    clean_en_acc, _ = acc_asr(clean_en, true, target)
    clean_zh_acc, _ = acc_asr(clean_zh, true, target)

    mode_images = {}
    mode_rects = {}
    mode_cams_en = {}
    mode_cams_zh = {}
    summary = {
        "n": int(data["n"]),
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "partner": "zh",
        "attack": "multi",
        "clean": {"en_acc": clean_en_acc, "zh_acc": clean_zh_acc},
        "modes": {},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    for mode in MODES:
        print(f"\n--- mode={mode} ---", flush=True)
        imgs, rects = build_mode_attack(data, mode, fonts)
        mode_images[mode] = imgs
        mode_rects[mode] = rects
        gt_masks = [rects_to_mask(r) for r in rects]

        preds_en, preds_zh, cams_en, cams_zh = compute_cams(
            en, zh, en_txt, zh_txt, imgs, label=mode
        )
        mode_cams_en[mode] = cams_en
        mode_cams_zh[mode] = cams_zh

        en_acc, en_asr = acc_asr(preds_en, true, target)
        zh_acc, zh_asr = acc_asr(preds_zh, true, target)

        masks_en = [mask_from_single_cam(c) for c in cams_en]
        masks_zh = [mask_from_single_cam(c) for c in cams_zh]
        masks_inter = [
            build_cc_bbox_mask(ce, cz, threshold=THR, dilate=DILATE, top_k=TOP_K)
            for ce, cz in zip(cams_en, cams_zh)
        ]
        # Also score thr-mask IoU using aligned CAM before cc_bbox for reference:
        # primary localization uses production cc_bbox masks as specified.
        loc = {
            "en": localization_stats(cams_en, gt_masks, masks_en),
            "zh": localization_stats(cams_zh, gt_masks, masks_zh),
            "en_cap_zh": localization_stats(
                [n_cam_intersection(ce, cz) for ce, cz in zip(cams_en, cams_zh)],
                gt_masks,
                masks_inter,
            ),
        }

        summary["modes"][mode] = {
            "en_acc": en_acc,
            "en_asr": en_asr,
            "zh_acc": zh_acc,
            "zh_asr": zh_asr,
            "localization": loc,
            "mean_gt_coverage": float(np.mean([m.mean() for m in gt_masks])),
        }
        print(
            f"  EN acc={fmt_pct(en_acc)} ASR={fmt_pct(en_asr)} | "
            f"ZH acc={fmt_pct(zh_acc)} ASR={fmt_pct(zh_asr)}",
            flush=True,
        )
        print(
            f"  EN&ZH IoU={loc['en_cap_zh']['iou_mean']:.3f} "
            f"inbox={loc['en_cap_zh']['inbox_focus_mean']:.2f} "
            f"det@.1={fmt_pct(loc['en_cap_zh']['detect_iou_ge_0.1'])}",
            flush=True,
        )

    if save_gallery_flag:
        save_gallery(mode_images, mode_cams_en, mode_cams_zh, MODES, FIGURES)

    print_table(summary)
    out_path = RESULTS / f"summary_n{data['n']}.json"
    write_summary(out_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="subset size (100 smoke / 1000 final)")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    run(n=args.n, save_gallery_flag=not args.no_gallery)


if __name__ == "__main__":
    main()
