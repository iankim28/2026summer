"""Fixed dual-box typographic attack placement for the balanced 1000-sample.

Positions are frozen in CIFAR10_BALANCED_1000_SAMPLE.json under `attack_pos`.
The seeded RNG here is only used once when generating those coordinates.
"""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

DISPLAY_SIZE = 224
NUM_BOXES = 2
FONT_SIZE = 24
PAD = 8
BH_EXTRA = 12

SAMPLE_NAME = "CIFAR10_BALANCED_1000_SAMPLE.json"
DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent / SAMPLE_NAME

CLASSES = {
    "en": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
    "ko": ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"],
    "ja": ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"],
}


def _font_paths():
    if platform.system() == "Windows":
        wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        cjk = os.path.join(wf, "msyh.ttc")
        lat = os.path.join(wf, "arial.ttf")
        ko = os.path.join(wf, "malgun.ttf")
        if not os.path.isfile(ko):
            ko = cjk
        return cjk, lat, ko
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    lat = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.isfile(cjk):
        cjk = "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"
    return cjk, lat, cjk


def _load_font(path: str | None, size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _font_for_lang(lang: str, cjk: str, lat: str, ko: str):
    if lang == "en":
        return _load_font(lat)
    if lang == "ko":
        return _load_font(ko)
    return _load_font(cjk)


def measure_box(word: str, font) -> tuple[int, int]:
    """Return (bw, bh) for a white text box around `word`."""
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = tmp.textbbox((0, 0), word, font=font)
    bw = (bb[2] - bb[0]) + 2 * PAD
    bh = (bb[3] - bb[1]) + PAD + BH_EXTRA
    return int(bw), int(bh)


def reference_box_size(
    classes: dict[str, list[str]] | None = None,
    font_size: int = FONT_SIZE,
) -> tuple[int, int]:
    """Max box size over all class words / fonts used in the stack."""
    classes = classes or CLASSES
    cjk, lat, ko = _font_paths()
    max_bw = max_bh = 0
    for lang, words in classes.items():
        font = _font_for_lang(lang, cjk, lat, ko)
        # re-load at requested size
        if lang == "en":
            font = _load_font(lat, font_size)
        elif lang == "ko":
            font = _load_font(ko, font_size)
        else:
            font = _load_font(cjk, font_size)
        for w in words:
            bw, bh = measure_box(w, font)
            max_bw = max(max_bw, bw)
            max_bh = max(max_bh, bh)
    return max_bw, max_bh


def _rects_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _random_nonoverlapping_rect(rng_, bw, bh, placed, display_size=DISPLAY_SIZE):
    x_hi = max(0, display_size - bw)
    y_hi = max(0, display_size - bh)
    rx = ry = 0
    for _ in range(64):
        rx = rng_.randint(0, x_hi) if x_hi > 0 else 0
        ry = rng_.randint(0, y_hi) if y_hi > 0 else 0
        rect = (rx, ry, rx + bw, ry + bh)
        if all(not _rects_overlap(rect, p) for p in placed):
            return rect
    return (rx, ry, rx + bw, ry + bh)


def generate_attack_pos(
    n_images: int = 1000,
    ref_bw: int | None = None,
    ref_bh: int | None = None,
    display_size: int = DISPLAY_SIZE,
    num_boxes: int = NUM_BOXES,
) -> dict:
    """Bake EN/L top-left anchors with the historical seed formula."""
    if ref_bw is None or ref_bh is None:
        computed_bw, computed_bh = reference_box_size()
        ref_bw = ref_bw if ref_bw is not None else computed_bw
        ref_bh = ref_bh if ref_bh is not None else computed_bh

    en_xy: list[list[int]] = []
    l_xy: list[list[int]] = []
    for i in range(n_images):
        placed = []
        for box_i in range(num_boxes):
            rng_ = random.Random(int(i) * num_boxes + box_i)
            rect = _random_nonoverlapping_rect(rng_, ref_bw, ref_bh, placed, display_size)
            placed.append(rect)
            xy = [int(rect[0]), int(rect[1])]
            if box_i == 0:
                en_xy.append(xy)
            else:
                l_xy.append(xy)

    cjk, lat, ko = _font_paths()
    return {
        "display_size": display_size,
        "font_size": FONT_SIZE,
        "pad": PAD,
        "bh_extra": BH_EXTRA,
        "num_boxes": num_boxes,
        "ref_bw": int(ref_bw),
        "ref_bh": int(ref_bh),
        "note": (
            "Per-image top-left anchors for dual-box typographic attacks. "
            "en = slot 0, l = slot 1. Generated once with seed "
            "Random(img_idx * num_boxes + box_i) and fixed reference box size "
            "so every defense shares the same geometry."
        ),
        "fonts": {"lat": lat, "cjk": cjk, "ko": ko},
        "en": en_xy,
        "l": l_xy,
    }


def clamp_xy(xy, bw, bh, display_size=DISPLAY_SIZE) -> tuple[int, int]:
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, display_size - bw)))
    y = max(0, min(y, max(0, display_size - bh)))
    return x, y


def load_sample(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_SAMPLE_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_attack_pos(sample: dict) -> dict:
    if "attack_pos" not in sample:
        raise KeyError(
            "Sample JSON missing 'attack_pos'. Run bake_attack_pos.py to generate it."
        )
    return sample["attack_pos"]


def xy_for_image(attack_pos: dict, img_idx: int) -> tuple[Sequence[int], Sequence[int]]:
    return attack_pos["en"][img_idx], attack_pos["l"][img_idx]


def draw_dual_box_at(
    img,
    word0,
    font0,
    word1,
    font1,
    xy0,
    xy1,
    already_224: bool = False,
    display_size: int = DISPLAY_SIZE,
    pad: int = PAD,
):
    """Draw two text boxes at saved top-left anchors (no RNG)."""
    if not already_224:
        img = img.convert("RGB").resize((display_size, display_size), Image.BICUBIC)
    else:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    for word, font, xy in ((word0, font0, xy0), (word1, font1, xy1)):
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * pad
        bh = (bb[3] - bb[1]) + pad + BH_EXTRA
        rx, ry = clamp_xy(xy, bw, bh, display_size)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text((rx + pad - bb[0], ry + pad - bb[1]), word, fill="black", font=font)
    return img


def bake_into_sample(
    sample_path: str | Path | None = None,
    n_images: int | None = None,
) -> dict:
    """Generate attack_pos and write it into the balanced sample JSON."""
    sample_path = Path(sample_path) if sample_path else DEFAULT_SAMPLE_PATH
    sample = load_sample(sample_path)
    n = n_images if n_images is not None else int(sample.get("n_images", len(sample["idx"])))
    attack_pos = generate_attack_pos(n_images=n)
    if len(attack_pos["en"]) != len(sample["idx"]) or len(attack_pos["l"]) != len(sample["idx"]):
        raise ValueError(
            f"Generated {len(attack_pos['en'])} positions but sample has {len(sample['idx'])} images"
        )
    sample["attack_pos"] = attack_pos
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return attack_pos


if __name__ == "__main__":
    pos = bake_into_sample()
    print(
        f"Wrote attack_pos: n={len(pos['en'])} ref=({pos['ref_bw']}x{pos['ref_bh']}) "
        f"-> {DEFAULT_SAMPLE_PATH}"
    )
