"""Train ChineseCLIP Defense-Prefix on CIFAR-10 train (never uses eval sample)."""
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

from zh_dp_encode import (
    ZH_CLASSES,
    class_prompts,
    encode_images,
    encode_text_vanilla,
    encode_text_with_dp,
    load_zh_clip,
    make_dp_embedding,
    preprocess_images,
)

DISPLAY = 224
PAD = 8
FONT_SIZE = 24
OUT_DIR = Path(__file__).resolve().parent / "results"
TOKEN_OUT = OUT_DIR / "dp_cifar10_zh_vit-b16.pt"


def _cjk_font():
    if platform.system() == "Windows":
        fp = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc")
    else:
        fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if not os.path.isfile(fp):
            fp = "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"
    try:
        return ImageFont.truetype(fp, FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


FONT = _cjk_font()


def draw_typo_zh(img: Image.Image, word: str, rng: random.Random) -> Image.Image:
    """Dual-box same ZH wrong-class word (mirrors EN train_cifar_dp.draw_typo)."""
    img = img.convert("RGB").resize((DISPLAY, DISPLAY), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    placed = []
    for _ in range(2):
        bb = draw.textbbox((0, 0), word, font=FONT)
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
        draw.text((x + PAD - bb[0], y + PAD - bb[1]), word, fill="black", font=FONT)
    return img


class CifarTypoTrainZh(Dataset):
    def __init__(self, processor, device: str, seed=0, max_n=None):
        self.hf = load_dataset("uoft-cs/cifar10", split="train")
        self.label_key = "label" if "label" in self.hf.column_names else "labels"
        self.image_key = "img" if "img" in self.hf.column_names else "image"
        self.processor = processor
        self.device = device
        self.seed = seed
        self.n = len(self.hf) if max_n is None else min(max_n, len(self.hf))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.hf[i]
        img, label = row[self.image_key], int(row[self.label_key])
        rng = random.Random(self.seed * 1_000_003 + i)
        wrong = rng.choice([c for c in range(10) if c != label])
        typo = draw_typo_zh(img, ZH_CLASSES[wrong], rng)
        clean = img.convert("RGB").resize((DISPLAY, DISPLAY), Image.BICUBIC)
        # Return PILs; collate in train loop (processor needs lists)
        return clean, typo, label


def collate_pil(batch):
    cleans, typos, labels = zip(*batch)
    return list(cleans), list(typos), torch.tensor(labels, dtype=torch.long)


def train(epochs=10, batch_size=128, lr=0.002, max_n=20000, gamma=3.0, seta=1.0):
    assert torch.cuda.is_available()
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = load_zh_clip(device)
    emb = make_dp_embedding(model, device)
    optimizer = optim.SGD(emb.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=5e-5)

    ds = CifarTypoTrainZh(processor, device, seed=0, max_n=max_n)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_pil,
    )
    print(
        f"ZH DP train samples={len(ds)} batch={batch_size} epochs={epochs} "
        f"device={device} font={FONT}",
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
                clean_pv = preprocess_images(processor, cleans, device)
                typo_pv = preprocess_images(processor, typos, device)
                clean_f = encode_images(model, clean_pv)
                typo_f = encode_images(model, typo_pv)
                orig_txt = encode_text_vanilla(model, processor, prompts_orig, device)
            pref_txt = encode_text_with_dp(
                model, processor, prompts_dp, emb.weight[0], device
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
        torch.save(emb.weight.detach().cpu(), OUT_DIR / f"dp_cifar10_zh_ep{ep+1}.pt")

    torch.save(emb.weight.detach().cpu(), TOKEN_OUT)
    print("Saved", TOKEN_OUT, flush=True)
    return TOKEN_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--max_n", type=int, default=20000)
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_n=args.max_n)


if __name__ == "__main__":
    main()
