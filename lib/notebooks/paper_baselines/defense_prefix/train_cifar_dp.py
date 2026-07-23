"""Train Defense-Prefix on CIFAR-10 train (HF datasets; never uses eval sample)."""
from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor" / "Defense-Prefix"
sys.path.insert(0, str(VENDOR))
from utils.non_nv import encode_text_with_learnt_tokens  # noqa: E402

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
TMPL = "a photo of a {}."
DISPLAY = 224
PAD = 8
FONT_SIZE = 24
OUT_DIR = Path(__file__).resolve().parent / "results"
TOKEN_OUT = OUT_DIR / "dp_cifar10_vit-b32.pt"


def _lat_font():
    if platform.system() == "Windows":
        fp = os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts", "arial.ttf")
    else:
        fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(fp, FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


FONT = _lat_font()


def draw_typo(img: Image.Image, word: str, rng: random.Random) -> Image.Image:
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


class CifarTypoTrain(Dataset):
    def __init__(self, preprocess, seed=0, max_n=None):
        self.hf = load_dataset("uoft-cs/cifar10", split="train")
        self.label_key = "label" if "label" in self.hf.column_names else "labels"
        self.image_key = "img" if "img" in self.hf.column_names else "image"
        self.preprocess = preprocess
        self.seed = seed
        self.n = len(self.hf) if max_n is None else min(max_n, len(self.hf))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.hf[i]
        img, label = row[self.image_key], int(row[self.label_key])
        rng = random.Random(self.seed * 1_000_003 + i)
        wrong = rng.choice([c for c in range(10) if c != label])
        typo = draw_typo(img, CLASSES[wrong], rng)
        clean = img.convert("RGB").resize((DISPLAY, DISPLAY), Image.BICUBIC)
        return self.preprocess(clean), self.preprocess(typo), label


def train(epochs=10, batch_size=128, lr=0.002, max_n=20000, gamma=3.0, seta=1.0):
    assert torch.cuda.is_available()
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    func_type = type(model.encode_text)
    model.encode_text_with_learnt_tokens = func_type(encode_text_with_learnt_tokens, model)

    dim = model.token_embedding.weight.shape[1]
    prefix = torch.empty(1, dim, dtype=model.dtype, device=device)
    nn.init.normal_(prefix, std=0.02)
    emb = nn.Embedding.from_pretrained(prefix, freeze=False)

    asterix = clip.tokenize(["*"]).to(device)[0][1]
    optimizer = optim.SGD(emb.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=5e-5)

    ds = CifarTypoTrain(preprocess, seed=0, max_n=max_n)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Train samples={len(ds)} batch={batch_size} epochs={epochs} device={device}")

    loss_ce = nn.CrossEntropyLoss()
    loss_kd = nn.KLDivLoss(reduction="batchmean")
    tok_prefix = clip.tokenize([TMPL.format(f"* {c}") for c in CLASSES]).to(device)
    tok_orig = clip.tokenize([TMPL.format(c) for c in CLASSES]).to(device)

    t0 = time.time()
    for ep in range(epochs):
        running, n_steps = 0.0, 0
        for clean, typo, labels in loader:
            clean, typo, labels = clean.to(device), typo.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                clean_f = F.normalize(model.encode_image(clean).float(), dim=-1)
                typo_f = F.normalize(model.encode_image(typo).float(), dim=-1)
                orig_txt = F.normalize(model.encode_text(tok_orig).float(), dim=-1)
            pref_txt = model.encode_text_with_learnt_tokens(
                tok_prefix, asterix, emb, is_emb=True
            )
            pref_txt = F.normalize(pref_txt.float(), dim=-1)
            scale = model.logit_scale.exp().float()
            logits_typo = scale * typo_f @ pref_txt.t()
            logits_reg = scale * clean_f @ pref_txt.t()
            logits_orig = scale * clean_f @ orig_txt.t()
            loss1 = loss_ce(logits_typo, labels)
            loss2 = loss_kd(F.log_softmax(logits_reg, dim=-1), logits_orig.softmax(dim=-1))
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
        torch.save(emb.weight.detach().cpu(), OUT_DIR / f"dp_cifar10_ep{ep+1}.pt")

    torch.save(emb.weight.detach().cpu(), TOKEN_OUT)
    print("Saved", TOKEN_OUT)
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
