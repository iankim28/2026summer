"""Train Japanese CLIP Defense-Prefix on CIFAR-10 train (never uses eval sample)."""
from __future__ import annotations

import argparse
import os
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from ja_dp_encode import (
    class_prompts,
    encode_images,
    encode_text_vanilla,
    encode_text_with_dp,
    load_ja_clip,
    make_dp_embedding,
    preprocess_images,
)

# Match PROTOCOL eval stickers (EN+ZH), not native JA — ZH retune lesson.
ZH_CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

DISPLAY = 224
PAD = 8
FONT_SIZE = 24
OUT_DIR = Path(__file__).resolve().parent / "results"
TOKEN_OUT = OUT_DIR / "dp_cifar10_ja_vit-b16.pt"

EN_CLASSES = [
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
]


def _font(path: str):
    try:
        return ImageFont.truetype(path, FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def _cjk_font_path():
    if platform.system() == "Windows":
        return os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc")
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if not os.path.isfile(fp):
        fp = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    return fp


def _lat_font_path():
    if platform.system() == "Windows":
        return os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


FONT_CJK = _font(_cjk_font_path())
FONT_LAT = _font(_lat_font_path())


def draw_typo_ja(img: Image.Image, wrong: int, rng: random.Random) -> Image.Image:
    """EN+ZH dual-box attack matching PROTOCOL eval (build_multi_attack)."""
    img = img.convert("RGB").resize((DISPLAY, DISPLAY), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    placed = []
    for word, font in ((EN_CLASSES[wrong], FONT_LAT), (ZH_CLASSES[wrong], FONT_CJK)):
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        for _try in range(32):
            x = rng.randint(0, max(0, DISPLAY - bw))
            y = rng.randint(0, max(0, DISPLAY - bh))
            rect = (x, y, x + bw, y + bh)
            if all(
                not (rect[0] < p[2] and rect[2] > p[0] and rect[1] < p[3] and rect[3] > p[1])
                for p in placed
            ):
                break
        placed.append(rect)
        draw.rectangle(rect, fill="white")
        draw.text((x + PAD - bb[0], y + PAD - bb[1]), word, fill="black", font=font)
    return img


class CifarTypoTrainJa(Dataset):
    def __init__(self, seed=0, max_n=None):
        self.hf = load_dataset("uoft-cs/cifar10", split="train")
        self.label_key = "label" if "label" in self.hf.column_names else "labels"
        self.image_key = "img" if "img" in self.hf.column_names else "image"
        self.seed = seed
        self.n = len(self.hf) if max_n is None else min(max_n, len(self.hf))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.hf[i]
        img, label = row[self.image_key], int(row[self.label_key])
        rng = random.Random(self.seed * 1_000_003 + i)
        wrong = rng.choice([c for c in range(10) if c != label])
        typo = draw_typo_ja(img, wrong, rng)
        clean = img.convert("RGB").resize((DISPLAY, DISPLAY), Image.BICUBIC)
        return clean, typo, label


def collate_pil(batch):
    cleans, typos, labels = zip(*batch)
    return list(cleans), list(typos), torch.tensor(labels, dtype=torch.long)


def train(epochs=10, batch_size=64, lr=0.002, max_n=20000, gamma=3.0, seta=1.0):
    assert torch.cuda.is_available()
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess, tokenizer = load_ja_clip(device)
    emb = make_dp_embedding(model, device)
    optimizer = optim.SGD(emb.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=5e-5)

    ds = CifarTypoTrainJa(seed=0, max_n=max_n)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_pil,
    )
    print(
        f"JA DP train samples={len(ds)} batch={batch_size} epochs={epochs} "
        f"device={device} attack=EN+ZH_multi (eval-matched)",
        flush=True,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_kd = nn.KLDivLoss(reduction="batchmean")
    prompts_dp = class_prompts(dp=True)
    prompts_orig = class_prompts(dp=False)

    t0 = time.time()
    for ep in range(epochs):
        running, n_steps = 0.0, 0
        for cleans, typos, labels in loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                clean_pv = preprocess_images(preprocess, cleans, device)
                typo_pv = preprocess_images(preprocess, typos, device)
                clean_f = encode_images(model, clean_pv)
                typo_f = encode_images(model, typo_pv)
                orig_txt = encode_text_vanilla(model, tokenizer, prompts_orig, device)
            pref_txt = encode_text_with_dp(
                model, tokenizer, prompts_dp, emb.weight[0], device
            )
            scale = model.logit_scale.exp().float()
            logits_typo = scale * typo_f @ pref_txt.t()
            logits_reg = scale * clean_f @ pref_txt.t()
            logits_orig = scale * clean_f @ orig_txt.t()
            loss1 = loss_ce(logits_typo, labels)
            loss2 = loss_kd(
                F.log_softmax(logits_reg, dim=-1), logits_orig.softmax(dim=-1)
            )
            loss = seta * loss1 + gamma * loss2
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_steps += 1
            if n_steps % 50 == 0:
                print(
                    f"  ep{ep+1} step{n_steps}/{len(loader)} "
                    f"loss={loss.item():.3f} (ce={loss1.item():.3f} kd={loss2.item():.3f})",
                    flush=True,
                )
        sched.step()
        print(
            f"Epoch {ep+1}/{epochs} mean_loss={running/max(n_steps,1):.4f} "
            f"lr={sched.get_last_lr()[0]:.6f} elapsed={time.time()-t0:.0f}s",
            flush=True,
        )
        torch.save(emb.weight.detach().cpu(), OUT_DIR / f"dp_cifar10_ja_ep{ep+1}.pt")
        torch.cuda.empty_cache()

    torch.save(emb.weight.detach().cpu(), TOKEN_OUT)
    print("Saved", TOKEN_OUT, flush=True)
    return TOKEN_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--max_n", type=int, default=20000)
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_n=args.max_n)


if __name__ == "__main__":
    main()
