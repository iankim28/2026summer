"""OCR + Gaussian blur on PROTOCOL dual-box CIFAR; score EN/ZH/KO/JA.

Same frozen EN+ZH multi attack as other baselines. EasyOCR langs: en, ch_sim, ko, ja.
Gate ladder: n=16 → n=100 → n=1000.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, ChineseCLIPModel, ChineseCLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.protocol import (  # noqa: E402
    BLUR_RADIUS,
    DEVICE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    blur_regions,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    write_summary,
)

RESULTS = Path(__file__).resolve().parent / "results"
COST_PROXY = 3

ALL_LANGS = ("en", "zh", "ko", "ja")
CLASSES = {
    "en": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
    "ko": ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"],
    "ja": ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"],
}
TMPL = {
    "en": "a photo of a {}.",
    "zh": "一张{}的照片。",
    "ko": "{}의 사진.",
    "ja": "{}の写真。",
}


def _clip_feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


class KoCLIP:
    def __init__(self):
        self.m = (
            AutoModel.from_pretrained(
                "Bingsu/clip-vit-base-patch32-ko", attn_implementation="eager"
            )
            .to(DEVICE)
            .eval()
        )
        self.p = AutoProcessor.from_pretrained("Bingsu/clip-vit-base-patch32-ko")

    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(
            text=[TMPL["ko"].format(w) for w in words],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t["input_ids"], attention_mask=t["attention_mask"]
        )
        return F.normalize(_clip_feat(out), dim=-1)


class JaCLIP:
    def __init__(self):
        mid = "hf-hub:llm-jp/llm-jp-clip-vit-base-patch16"
        self.m, _, self.pp = open_clip.create_model_and_transforms(mid)
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer(mid)

    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL["ja"].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)


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


def detect_boxes(readers, pil_img):
    """Merge boxes from multiple EasyOCR readers (script packs are incompatible)."""
    if not isinstance(readers, (list, tuple)):
        readers = [readers]
    arr = np.asarray(pil_img)
    boxes = []
    for reader in readers:
        for item in reader.readtext(arr):
            bbox = item[0]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes


def sticker_hit_rate(gt_rects, det_boxes, thr=0.1):
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

    print("Loading EasyOCR readers (en+ch_sim, ko, ja)...", flush=True)
    # EasyOCR: ch_sim only pairs with en; KO/JA need separate readers.
    # verbose=False avoids Unicode progress-bar crashes on Windows cp1252 consoles.
    readers = [
        easyocr.Reader(["en", "ch_sim"], gpu=True, verbose=False),
        easyocr.Reader(["ko"], gpu=True, verbose=False),
        easyocr.Reader(["ja"], gpu=True, verbose=False),
    ]
    print("Loading CLIP models EN/ZH/KO/JA...", flush=True)
    models = {
        "en": EnCLIP(),
        "zh": ZhCLIP(),
        "ko": KoCLIP(),
        "ja": JaCLIP(),
    }
    text = {lang: models[lang].embed_texts(CLASSES[lang]) for lang in ALL_LANGS}

    clean_acc = {}
    atk_acc = {}
    atk_asr = {}
    for lang in ALL_LANGS:
        cpred = classify_batch(models[lang], data["clean_224"], text[lang])
        apred = classify_batch(models[lang], attacked, text[lang])
        clean_acc[lang] = float((cpred == true).mean())
        atk_acc[lang], atk_asr[lang] = acc_asr(apred, true, target)
    print(
        "Vanilla clean "
        + " ".join(f"{L.upper()}={100*clean_acc[L]:.1f}%" for L in ALL_LANGS)
    )
    print(
        "Vanilla atk "
        + " ".join(f"{L.upper()}={100*atk_acc[L]:.1f}%" for L in ALL_LANGS)
    )

    det_lists, defended = [], []
    n_det_imgs = 0
    t0 = time.time()
    for i, img in enumerate(attacked):
        boxes = detect_boxes(readers, img)
        det_lists.append(boxes)
        if boxes:
            n_det_imgs += 1
        defended.append(blur_regions(img, boxes, radius=BLUR_RADIUS))
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

    def_acc = {}
    def_asr = {}
    for lang in ALL_LANGS:
        dpred = classify_batch(models[lang], defended, text[lang])
        def_acc[lang], def_asr[lang] = acc_asr(dpred, true, target)

    clean_defended = []
    for img in data["clean_224"]:
        boxes = detect_boxes(readers, img)
        clean_defended.append(blur_regions(img, boxes, radius=BLUR_RADIUS))

    clean_deg = {}
    for lang in ALL_LANGS:
        cpred = classify_batch(models[lang], clean_defended, text[lang])
        masked = float((cpred == true).mean())
        delta = masked - clean_acc[lang]
        clean_deg[lang] = {
            "baseline_acc": clean_acc[lang],
            "masked_acc": masked,
            "delta_acc": delta,
        }

    mean_atk = float(np.mean([def_acc[L] for L in ALL_LANGS]))
    mean_mixed = float(
        np.mean(
            [
                0.5 * def_acc[L] + 0.5 * clean_deg[L]["masked_acc"]
                for L in ALL_LANGS
            ]
        )
    )
    mean_delta = float(np.mean([clean_deg[L]["delta_acc"] for L in ALL_LANGS]))

    print(
        f"OCR+blur 4-lang mean_atk={100*mean_atk:.1f}% mean_mixed={100*mean_mixed:.1f}%  "
        + " ".join(f"{L.upper()}={100*def_acc[L]:.1f}%" for L in ALL_LANGS)
        + f"  CleanΔEN={100*clean_deg['en']['delta_acc']:.1f}pp"
    )

    payload = {
        "method": "ocr_blur_4lang",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_zh_ko_ja_multi",
        "ocr": "easyocr_en_ch_sim_ko_ja",
        "blur_radius": BLUR_RADIUS,
        "inference_cost": COST_PROXY,
        "detect_img_rate": detect_img_rate,
        "sticker_hit_rate": hit_rate,
        "sticker_hits": hits,
        "sticker_total": total_stickers,
        "defense": {
            lang: {
                "acc": def_acc[lang],
                "asr": def_asr[lang],
                "baseline_acc": atk_acc[lang],
                "baseline_asr": atk_asr[lang],
                "mixed_2000": 0.5 * def_acc[lang] + 0.5 * clean_deg[lang]["masked_acc"],
            }
            for lang in ALL_LANGS
        },
        "clean_degradation": clean_deg,
        "defense_acc_mean": mean_atk,
        "mixed_2000_mean": mean_mixed,
        "clean_delta_mean": mean_delta,
        "clean_delta_en": clean_deg["en"]["delta_acc"],
        "notes": "EasyOCR detect → Gaussian blur; scored EN+ZH+KO+JA on EN+ZH multi attack.",
    }
    write_summary(RESULTS / f"comparison_summary_4lang_{status}_n{data['n']}.json", payload)
    write_summary(RESULTS / "comparison_summary_4lang.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    args = ap.parse_args()
    run_eval(args.n, args.status)


if __name__ == "__main__":
    main()
