"""Animal-sticker vs text dual-box ablation.

Modes:
  all_sticker — both slots: proportional CIFAR-10 animal paste (no white pad)
  mixed       — slot0 animal, slot1 EN class-name text (same wrong class)
  all_text    — production EN+ZH typographic dual box (protocol target)

Measures Attn-last localization + undefended acc/ASR.
CUDA required. Smoke: --n 100. Final: --n 1000.
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
from datasets import load_dataset
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
    EnCLIP,
    ZhCLIP,
    acc_asr,
    classify_batch,
    load_protocol_data,
    write_summary,
)
from attack_placement import (  # noqa: E402
    ANIMAL_CLASS_IDS,
    ANIMAL_STICKER_SIDE,
    FONT_SIZE,
    PAD,
    _font_paths,
    _load_font,
    draw_dual_box_at,
    draw_dual_content_at,
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
MODES = ("all_sticker", "mixed", "all_text")
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


def load_animal_pools():
    """CIFAR-10 train animal PILs keyed by class id (disjoint from test hosts)."""
    hf = load_dataset("uoft-cs/cifar10", split="train")
    label_key = "label" if "label" in hf.column_names else "labels"
    image_key = "img" if "img" in hf.column_names else "image"
    pools: dict[int, list[Image.Image]] = {c: [] for c in ANIMAL_CLASS_IDS}
    labels = hf[label_key]
    images = hf[image_key]
    for lab, im in zip(labels, images):
        c = int(lab)
        if c in pools:
            pools[c].append(im.convert("RGB"))
    for c, lst in pools.items():
        assert len(lst) > 10, f"animal class {c} has only {len(lst)} train images"
        print(f"  animal pool class={c} ({CLASSES['en'][c]}): {len(lst)}", flush=True)
    return pools


def animal_targets(true: np.ndarray) -> np.ndarray:
    """Per-image wrong animal class ≠ true, seeded by index."""
    out = []
    for i, t in enumerate(true):
        t = int(t)
        choices = [c for c in ANIMAL_CLASS_IDS if c != t]
        out.append(random.Random(i).choice(choices))
    return np.array(out, dtype=np.int64)


def sample_two_animals(pools, class_id: int, img_idx: int):
    rng = random.Random(10_000 + img_idx * 17 + class_id)
    pool = pools[class_id]
    if len(pool) >= 2:
        i0, i1 = rng.sample(range(len(pool)), 2)
    else:
        i0 = i1 = 0
    return pool[i0].copy(), pool[i1].copy()


def build_mode_attack(data, mode: str, fonts: dict, pools, animal_tgt):
    out, all_rects, targets = [], [], []
    for i in range(data["n"]):
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        if mode == "all_text":
            t = int(data["target"][i])
            en_w, zh_w = CLASSES["en"][t], CLASSES["zh"][t]
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
                mode="full",
                return_boxes=True,
            )
            targets.append(t)
        elif mode == "all_sticker":
            t = int(animal_tgt[i])
            a0, a1 = sample_two_animals(pools, t, i)
            img, boxes = draw_dual_content_at(
                data["clean_224"][i],
                ("animal", a0),
                ("animal", a1),
                xy0,
                xy1,
                animal_side=ANIMAL_STICKER_SIDE,
                already_224=True,
                pad=PAD,
                return_boxes=True,
            )
            targets.append(t)
        elif mode == "mixed":
            t = int(animal_tgt[i])
            a0, _ = sample_two_animals(pools, t, i)
            word = CLASSES["en"][t]
            img, boxes = draw_dual_content_at(
                data["clean_224"][i],
                ("animal", a0),
                ("text", word, fonts["en"]),
                xy0,
                xy1,
                animal_side=ANIMAL_STICKER_SIDE,
                already_224=True,
                pad=PAD,
                return_boxes=True,
            )
            targets.append(t)
        else:
            raise ValueError(mode)
        out.append(img)
        all_rects.append(boxes)
    return out, all_rects, np.array(targets, dtype=np.int64)


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
    return np.array(preds_en), np.array(preds_zh), cams_en, cams_zh


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
    print("\n=== Animal-sticker ablation ===")
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
    true = data["true"]

    print("Loading CIFAR-10 train animal pools…", flush=True)
    pools = load_animal_pools()
    animal_tgt = animal_targets(true)

    print("Loading EN + ZH CLIP…", flush=True)
    en = EnCLIP()
    zh = ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    print("Clean baseline…", flush=True)
    clean_en = classify_batch(en, data["clean_224"], en_txt)
    clean_zh = classify_batch(zh, data["clean_224"], zh_txt)
    clean_en_acc, _ = acc_asr(clean_en, true, data["target"])
    clean_zh_acc, _ = acc_asr(clean_zh, true, data["target"])

    mode_images = {}
    mode_cams_en = {}
    mode_cams_zh = {}
    summary = {
        "n": int(data["n"]),
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "partner": "zh",
        "modes_list": list(MODES),
        "animal_sticker_side": ANIMAL_STICKER_SIDE,
        "animal_classes": list(ANIMAL_CLASS_IDS),
        "clean": {"en_acc": clean_en_acc, "zh_acc": clean_zh_acc},
        "modes": {},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "note": (
            "all_sticker: both slots proportional CIFAR animal paste (no white pad); "
            "mixed: animal + EN text same wrong animal class; "
            "all_text: EN+ZH typographic with protocol target."
        ),
    }

    for mode in MODES:
        print(f"\n--- mode={mode} ---", flush=True)
        imgs, rects, tgt = build_mode_attack(data, mode, fonts, pools, animal_tgt)
        mode_images[mode] = imgs
        gt_masks = [rects_to_mask(r) for r in rects]

        preds_en, preds_zh, cams_en, cams_zh = compute_cams(
            en, zh, en_txt, zh_txt, imgs, label=mode
        )
        mode_cams_en[mode] = cams_en
        mode_cams_zh[mode] = cams_zh

        en_acc, en_asr = acc_asr(preds_en, true, tgt)
        zh_acc, zh_asr = acc_asr(preds_zh, true, tgt)

        masks_en = [mask_from_single_cam(c) for c in cams_en]
        masks_zh = [mask_from_single_cam(c) for c in cams_zh]
        masks_inter = [
            build_cc_bbox_mask(ce, cz, threshold=THR, dilate=DILATE, top_k=TOP_K)
            for ce, cz in zip(cams_en, cams_zh)
        ]
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
            "target_kind": "protocol" if mode == "all_text" else "animal",
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
    RESULTS.mkdir(parents=True, exist_ok=True)
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
