"""Run EN/ZH typographic comparison on the shared CIFAR-10 1000-image subset."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

LANGS = ["en", "zh"]
CLASSES = {
    "en": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}
TMPL = {"en": "a photo of a {}.", "zh": "一张{}的照片。"}


def classify(model, imgs, words):
    imf = model.embed_images(imgs)
    tf = model.embed_texts(words)
    return (imf @ tf.t()).argmax(-1).cpu().numpy()


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


class EnCLIP:
    lang = "en"

    def __init__(self):
        self.m, _, self.pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL["en"].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)


class ZhCLIP:
    lang = "zh"

    def __init__(self):
        self.m = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(DEVICE).eval()
        self.p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL["zh"].format(w) for w in words], padding=True, return_tensors="pt").to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t["input_ids"],
            attention_mask=t["attention_mask"],
            token_type_ids=t.get("token_type_ids"),
        )
        return F.normalize(_clip_feat(out), dim=-1)


def _font_paths():
    if platform.system() == "Windows":
        win = r"C:\Windows\Fonts"
        cjk = os.path.join(win, "msgothic.ttc")
        latin = os.path.join(win, "arial.ttf")
        return (cjk if os.path.exists(cjk) else None, latin if os.path.exists(latin) else None)
    for d in ["/usr/share/fonts", "/Library/Fonts", os.path.expanduser("~/.fonts")]:
        for f in ["NotoSansCJK-Regular.ttc", "NotoSans-Regular.ttf"]:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return p, p
    return None, None


_CJK_FONT, _LAT_FONT = _font_paths()


def draw_word(img, word, where="bottom"):
    img = img.copy()
    w, h = img.size
    font_size = max(4, h // 24)
    try:
        fp = _CJK_FONT if any(ord(c) > 127 for c in word) else _LAT_FONT
        font = ImageFont.truetype(fp, font_size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), word, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 1
    x = (w - tw) // 2
    y = h - th - pad * 2 if where == "bottom" else (h - th) // 2
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill="white")
    draw.text((x, y), word, fill="black", font=font)
    return img


def load_shared_subset():
    hf = load_dataset("uoft-cs/cifar10", split="test")
    label_key = "label" if "label" in hf.column_names else "labels"
    image_key = "img" if "img" in hf.column_names else "image"
    indices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_samples", "CIFAR10_4LANG_1000_SAMPLE.json")
    if not os.path.exists(indices_path):
        rng = random.Random(0)
        idx = rng.sample(range(len(hf)), 1000)
        rows = hf.select(idx)
        true = np.array(rows[label_key])
        target = [int(rng.choice([c for c in range(10) if c != true[k]])) for k in range(len(idx))]
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump(
                {"seed": 0, "n_images": 1000, "idx": idx, "target": target, "true": true.tolist()},
                f,
            )
    else:
        with open(indices_path, encoding="utf-8") as f:
            saved = json.load(f)
        idx = saved["idx"]
        target = np.array(saved["target"])
        rows = hf.select(idx)
        true = np.array(rows[label_key])
        assert len(idx) == 1000
        assert np.array_equal(true, np.array(saved["true"]))
    clean = [im.convert("RGB") for im in rows[image_key]]
    return idx, target, true, clean


def disagreement_score(preds_mat):
    return np.array([len(set(preds_mat[:, i])) for i in range(preds_mat.shape[1])])


def roc_auc(pos_scores, neg_scores):
    gt = (pos_scores[:, None] > neg_scores[None, :]).astype(float)
    tie = (pos_scores[:, None] == neg_scores[None, :]).astype(float)
    return float(np.mean(gt + 0.5 * tie))


def main():
    idx, target, true, clean = load_shared_subset()
    print(f"Loaded {len(clean)} images (shared CIFAR-10 subset)")
    for c, name in enumerate(CLASSES["en"]):
        print(f"  {name:10s}: {(true == c).sum():3d}")

    models = {}
    for lang, cls in {"en": EnCLIP, "zh": ZhCLIP}.items():
        t0 = time.time()
        print(f"Loading {lang}...", end=" ", flush=True)
        models[lang] = cls()
        print(f"{time.time() - t0:.1f}s")

    clean_acc = {}
    clean_preds = {}
    print("\nClean baseline accuracy:")
    for lang in LANGS:
        p = classify(models[lang], clean, CLASSES[lang])
        clean_preds[lang] = p
        clean_acc[lang] = float((p == true).mean())
        print(f"  model_{lang}: {100 * clean_acc[lang]:.1f}%")

    preds_attacked = {}
    for attack_lang in LANGS:
        t0 = time.time()
        print(f"\nAttack language: {attack_lang}")
        attacked = [
            draw_word(clean[k], CLASSES[attack_lang][target[k]], where="bottom") for k in range(len(idx))
        ]
        preds_attacked[attack_lang] = {}
        for model_lang in LANGS:
            p = classify(models[model_lang], attacked, CLASSES[model_lang])
            preds_attacked[attack_lang][model_lang] = p
            acc = (p == true).mean()
            asr = (p == target).mean()
            print(f"  model_{model_lang}: acc={100 * acc:.1f}%  ASR={100 * asr:.1f}%")
        print(f"  [{time.time() - t0:.1f}s]")

    n = len(LANGS)
    acc_matrix = np.zeros((n, n))
    asr_matrix = np.zeros((n, n))
    for i, al in enumerate(LANGS):
        for j, ml in enumerate(LANGS):
            p = preds_attacked[al][ml]
            acc_matrix[i, j] = (p == true).mean()
            asr_matrix[i, j] = (p == target).mean()

    clean_row = np.array([clean_acc[ml] for ml in LANGS])
    table_acc = np.vstack([clean_row, acc_matrix])
    row_labels = ["clean (no attack)"] + [f"attack_{l}" for l in LANGS]
    col_labels = [f"model_{l}" for l in LANGS]

    print("\nAccuracy matrix (%):")
    print(f'  {"":22s}', "  ".join(f"{c:>12s}" for c in col_labels))
    for r, row in zip(row_labels, table_acc):
        print(f"  {r:22s}", "  ".join(f"{100 * v:11.1f}%" for v in row))

    table_asr = np.vstack([np.zeros(n), asr_matrix])
    print("\nAttack Success Rate matrix (%):")
    print(f'  {"":22s}', "  ".join(f"{c:>12s}" for c in col_labels))
    for r, row in zip(row_labels, table_asr):
        print(f"  {r:22s}", "  ".join(f"{100 * v:11.1f}%" for v in row))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, mat, title in zip(
        axes,
        [acc_matrix, asr_matrix],
        ["Accuracy under attack (%)", "Attack Success Rate (%)"],
    ):
        im = ax.imshow(mat * 100, vmin=0, vmax=100, cmap="RdYlGn" if "Accuracy" in title else "RdYlGn_r")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"model_{l}" for l in LANGS])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"attack_{l}" for l in LANGS])
        ax.set_title(title, fontsize=11)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{100 * mat[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if mat[i, j] < 0.3 or mat[i, j] > 0.7 else "black",
                )
        plt.colorbar(im, ax=ax, format="%.0f%%")
    plt.tight_layout()
    plt.savefig("results/accuracy_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved -> results/accuracy_matrix.png")

    preds_clean_mat = np.stack([clean_preds[l] for l in LANGS])
    scores_clean = disagreement_score(preds_clean_mat)
    agree_clean = float((scores_clean == 1).mean())

    print("\n=== Disagreement detector — EN vs ZH ===\n")
    print(f"Clean all-agree rate: {100 * agree_clean:.1f}%\n")
    print(f"{'Attack':>8}  {'All-agree':>10}  {'AUC':>8}")
    print("-" * 32)

    detector_results = {}
    for attack_lang in LANGS:
        preds_atk_mat = np.stack([preds_attacked[attack_lang][l] for l in LANGS])
        scores_attacked = disagreement_score(preds_atk_mat)
        auc_val = roc_auc(scores_attacked, scores_clean)
        agree_atk = float((scores_attacked == 1).mean())
        print(f"  {attack_lang.upper():>4}    {100 * agree_atk:6.1f}%    {auc_val:.4f}")
        detector_results[attack_lang] = {
            "agree_attacked": agree_atk,
            "auc": auc_val,
            "score_dist_attacked": {str(s): int((scores_attacked == s).sum()) for s in range(1, 3)},
        }

    print("\nScore distribution for clean images:")
    for s in range(1, 3):
        n_s = (scores_clean == s).sum()
        print(f"  score={s}: {n_s:4d} ({100 * n_s / len(scores_clean):.0f}%)")

    results = {
        "langs": LANGS,
        "classes": CLASSES,
        "n_images": len(idx),
        "sample_idx": idx,
        "clean_acc": {l: float(clean_acc[l]) for l in LANGS},
        "acc_matrix": acc_matrix.tolist(),
        "asr_matrix": asr_matrix.tolist(),
        "detector": {
            "agree_clean": agree_clean,
            "score_dist_clean": {str(s): int((scores_clean == s).sum()) for s in range(1, 3)},
            "by_attack_lang": detector_results,
        },
    }
    with open("results/confusion_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved -> results/confusion_results.json")


if __name__ == "__main__":
    main()
