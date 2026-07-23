"""OCR + Gaussian blur baseline on PROTOCOL dual-box CIFAR.

Detect text with EasyOCR → blur boxes → reclassify EN (+ ZH for mean).
Gate ladder: n=16 → n=100 → n=1000.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.protocol import (  # noqa: E402
    BLUR_RADIUS,
    CLASSES,
    DEVICE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    blur_regions,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    progress_log,
    write_summary,
)

RESULTS = Path(__file__).resolve().parent / "results"
# Cost: OCR (~1) + EN reclass + ZH reclass ≈ 3 (proxy; OCR is external)
COST_PROXY = 3


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return inter / (area_a + area_b - inter + 1e-9)


def detect_boxes(reader, pil_img):
    """Return list of (x0,y0,x1,y1) from EasyOCR."""
    arr = np.asarray(pil_img)
    # detail=1 → (bbox, text, conf); bbox = 4 corners
    results = reader.readtext(arr)
    boxes = []
    for item in results:
        bbox = item[0]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes


def sticker_hit_rate(gt_rects, det_boxes, thr=0.1):
    """Fraction of ground-truth stickers that overlap some detection (IoU>=thr)."""
    hits = 0
    total = 0
    for gts, dets in zip(gt_rects, det_boxes):
        for g in gts:
            total += 1
            if any(iou(g, d) >= thr for d in dets):
                hits += 1
    return hits / max(total, 1), hits, total


def run_eval(n: int, status: str):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, gt_rects = build_multi_attack(data)
    true, target = data["true"], data["target"]

    print("Loading EasyOCR (en+ch_sim)...", flush=True)
    reader = easyocr.Reader(["en", "ch_sim"], gpu=True)
    print("Loading CLIP models...", flush=True)
    en = EnCLIP()
    zh = ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    # Baselines
    clean_en = classify_batch(en, data["clean_224"], en_txt)
    clean_zh = classify_batch(zh, data["clean_224"], zh_txt)
    atk_en = classify_batch(en, attacked, en_txt)
    atk_zh = classify_batch(zh, attacked, zh_txt)
    clean_acc_en, clean_acc_zh = float((clean_en == true).mean()), float((clean_zh == true).mean())
    atk_acc_en, atk_asr_en = acc_asr(atk_en, true, target)
    atk_acc_zh, atk_asr_zh = acc_asr(atk_zh, true, target)
    print(
        f"Vanilla clean EN/ZH={100*clean_acc_en:.1f}/{100*clean_acc_zh:.1f}%  "
        f"atk EN/ZH={100*atk_acc_en:.1f}/{100*atk_acc_zh:.1f}%"
    )

    # Defend attacked
    det_lists, defended = [], []
    n_det_imgs = 0
    t0 = time.time()
    running = 0
    for i, img in enumerate(attacked):
        boxes = detect_boxes(reader, img)
        det_lists.append(boxes)
        if boxes:
            n_det_imgs += 1
        defended.append(blur_regions(img, boxes, radius=BLUR_RADIUS))
        # progress on detect rate
        if (i + 1) % 50 == 0 or (i + 1) == len(attacked):
            print(
                f"  OCR {i+1}/{len(attacked)}  imgs_with_box={n_det_imgs}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    hit_rate, hits, total_stickers = sticker_hit_rate(gt_rects, det_lists)
    detect_img_rate = n_det_imgs / len(attacked)
    print(
        f"OCR detect_img_rate={100*detect_img_rate:.1f}%  "
        f"sticker_hit_rate={100*hit_rate:.1f}% ({hits}/{total_stickers})"
    )
    if status == "sanity" and n_det_imgs == 0:
        raise RuntimeError("Gate A fail: OCR found zero boxes on attacked images")

    def_en = classify_batch(en, defended, en_txt)
    def_zh = classify_batch(zh, defended, zh_txt)
    def_acc_en, def_asr_en = acc_asr(def_en, true, target)
    def_acc_zh, def_asr_zh = acc_asr(def_zh, true, target)
    mean_acc = 0.5 * (def_acc_en + def_acc_zh)

    # Clean degradation
    clean_defended = []
    for img in data["clean_224"]:
        boxes = detect_boxes(reader, img)
        clean_defended.append(blur_regions(img, boxes, radius=BLUR_RADIUS))
    c_en = classify_batch(en, clean_defended, en_txt)
    c_zh = classify_batch(zh, clean_defended, zh_txt)
    d_en = float((c_en == true).mean()) - clean_acc_en
    d_zh = float((c_zh == true).mean()) - clean_acc_zh
    mean_delta = 0.5 * (d_en + d_zh)

    print(
        f"OCR+blur mean={100*mean_acc:.1f}%  EN/ZH={100*def_acc_en:.1f}/{100*def_acc_zh:.1f}%  "
        f"CleanΔ={100*mean_delta:.1f}pp"
    )

    payload = {
        "method": "ocr_blur",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_zh_multi",
        "ocr": "easyocr_en_ch_sim",
        "blur_radius": BLUR_RADIUS,
        "inference_cost": COST_PROXY,
        "detect_img_rate": detect_img_rate,
        "sticker_hit_rate": hit_rate,
        "sticker_hits": hits,
        "sticker_total": total_stickers,
        "defense": {
            "en": {
                "acc": def_acc_en,
                "asr": def_asr_en,
                "baseline_acc": atk_acc_en,
                "baseline_asr": atk_asr_en,
            },
            "zh": {
                "acc": def_acc_zh,
                "asr": def_asr_zh,
                "baseline_acc": atk_acc_zh,
                "baseline_asr": atk_asr_zh,
            },
        },
        "clean_degradation": {
            "en": {
                "baseline_acc": clean_acc_en,
                "masked_acc": float((c_en == true).mean()),
                "delta_acc": d_en,
            },
            "zh": {
                "baseline_acc": clean_acc_zh,
                "masked_acc": float((c_zh == true).mean()),
                "delta_acc": d_zh,
            },
        },
        "defense_acc_mean": mean_acc,
        "clean_delta_mean": mean_delta,
        "notes": "EasyOCR detect → Gaussian blur; scored EN+ZH on multi attack.",
    }
    write_summary(RESULTS / f"comparison_summary_{status}_n{data['n']}.json", payload)
    write_summary(RESULTS / "comparison_summary.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    args = ap.parse_args()
    run_eval(args.n, args.status)


if __name__ == "__main__":
    main()
