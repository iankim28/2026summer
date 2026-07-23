"""Shared dual-box eval helpers for paper baselines (PROTOCOL.md)."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

DISPLAY_SIZE = 224
FONT_SIZE = 24
PAD = 8
BLUR_RADIUS = 12
PROGRESS_EVERY = 50

CLASSES = {
    "en": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}
TMPL = {
    "en": "a photo of a {}.",
    "zh": "一张{}的照片。",
}

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE.parent.parent / "image_samples" / "CIFAR10_BALANCED_1000_SAMPLE.json"


def _font_paths():
    if platform.system() == "Windows":
        wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        return os.path.join(wf, "msyh.ttc"), os.path.join(wf, "arial.ttf")
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    lat = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return cjk, lat


_CJK_FONT, _LAT_FONT = _font_paths()
_FONT_CACHE: dict = {}


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


def load_protocol_data(n: int | None = None):
    """Load clean 224 images, labels, targets, frozen attack_pos. Optionally truncate to first n."""
    hf = load_dataset("uoft-cs/cifar10", split="test")
    label_key = "label" if "label" in hf.column_names else "labels"
    image_key = "img" if "img" in hf.column_names else "image"
    with open(_SAMPLE, encoding="utf-8") as f:
        saved = json.load(f)
    idx = saved["idx"]
    attack_pos = saved["attack_pos"]
    rows = hf.select(idx)
    true = np.array(rows[label_key])
    assert len(idx) == 1000 and np.array_equal(true, np.array(saved["true"]))
    rng = random.Random(0)
    target = np.array(
        [rng.choice([c for c in range(10) if c != int(true[k])]) for k in range(len(idx))]
    )
    clean_224 = [
        im.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
        for im in rows[image_key]
    ]
    if n is not None and n < len(clean_224):
        # Prefer tune subset (first 10 per class) for n=100; for n=16 take first 16 of tune
        tune = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
        if n <= 100:
            sel = tune[:n]
        else:
            sel = np.arange(n)
        clean_224 = [clean_224[i] for i in sel]
        true = true[sel]
        target = target[sel]
        # Remap attack_pos rows
        attack_pos = {
            **attack_pos,
            "en": [attack_pos["en"][int(i)] for i in sel],
            "l": [attack_pos["l"][int(i)] for i in sel],
            "_sel": [int(i) for i in sel],
        }
        print(f"Subset n={n} (sel first {n} of tune/all)")
    else:
        print(f"Full n={len(clean_224)}")
    return {
        "clean_224": clean_224,
        "true": true,
        "target": target,
        "attack_pos": attack_pos,
        "n": len(clean_224),
    }


def draw_dual_box(img, word0, lang0, word1, lang1, xy0, xy1):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_LAT_FONT if lang == "en" else _CJK_FONT)
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill="black", font=font)
    return img


def box_rects(word0, lang0, word1, lang1, xy0, xy1):
    """Return list of (x0,y0,x1,y1) for the two sticker boxes (for OCR miss-rate / mining)."""
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rects = []
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_LAT_FONT if lang == "en" else _CJK_FONT)
        bb = tmp.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        rects.append((rx, ry, rx + bw, ry + bh))
    return rects


def build_multi_attack(data):
    """EN+ZH dual-box multi attack (Option B multi with L=zh)."""
    out, rects = [], []
    for i in range(data["n"]):
        t = int(data["target"][i])
        en_w, zh_w = CLASSES["en"][t], CLASSES["zh"][t]
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        out.append(draw_dual_box(data["clean_224"][i], en_w, "en", zh_w, "zh", xy0, xy1))
        rects.append(box_rects(en_w, "en", zh_w, "zh", xy0, xy1))
    return out, rects


def blur_regions(pil_img, boxes, radius=BLUR_RADIUS):
    """Gaussian-blur axis-aligned boxes (x0,y0,x1,y1) in-place copy."""
    if not boxes:
        return pil_img.copy()
    arr = np.array(pil_img)
    blurred = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius)))
    out = arr.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
    return Image.fromarray(out)


class EnCLIP:
    def __init__(self):
        self.m, _, self.pp = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words, template=None):
        tmpl = template or TMPL["en"]
        t = self.tok([tmpl.format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)


class ZhCLIP:
    def __init__(self):
        self.m = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", attn_implementation="eager"
        ).to(DEVICE).eval()
        self.p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        out = self.m.get_image_features(pixel_values=pv)
        feat = out if torch.is_tensor(out) else out.pooler_output
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(
            text=[TMPL["zh"].format(w) for w in words],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t["input_ids"],
            attention_mask=t["attention_mask"],
            token_type_ids=t.get("token_type_ids"),
        )
        feat = out if torch.is_tensor(out) else out.pooler_output
        return F.normalize(feat, dim=-1)


def classify_batch(model, imgs, text_emb, batch_size=64):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i : i + batch_size])
        preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def acc_asr(preds, true, target):
    preds = np.asarray(preds)
    return float((preds == true).mean()), float((preds == target).mean())


def write_summary(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", path)


def progress_log(i, n, running_correct, t0):
    if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == n:
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (n - i - 1) / max(rate, 1e-6)
        acc = running_correct / (i + 1)
        print(
            f"  {i+1}/{n}  running_acc={100*acc:.1f}%  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            flush=True,
        )
