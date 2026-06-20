"""Cross-lingual TYPOGRAPHIC attack on multilingual CLIP (high-school-friendly: no gradients).

We write a misleading class word ONTO the image (in some language) and ask whether CLIP's
prediction flips to that written word -- and whether this works ACROSS languages:
does writing "dog" in English fool the model when we classify with Korean labels, etc.

Outputs an attack-success matrix over (written-text language) x (prompt language).
"""
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
from mclip_lib import (load_model, build_text_embeddings, encode_image,
                       logits_for, get_logit_scale, LANGS, TRANSLATIONS, STL10_CLASSES)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def draw_word(pil_img, word, font_size=34):
    """Resize to 224 and stamp `word` in a high-contrast band near the bottom (classic typo attack)."""
    img = pil_img.convert("RGB").resize((224, 224), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    bb = d.textbbox((0, 0), word, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    x = (224 - w) // 2
    y = 224 - h - 16
    # white box + black text = maximally legible
    d.rectangle([x - 8, y - 8, x + w + 8, y + h + 12], fill=(255, 255, 255))
    d.text((x - bb[0], y - bb[1]), word, fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--font_size", type=int, default=34)
    ap.add_argument("--save_examples", type=int, default=6)
    args = ap.parse_args()
    device = "cuda"
    rng = random.Random(0)

    model, tokenizer, _, mean, std = load_model(device)
    classes = STL10_CLASSES
    txt = build_text_embeddings(model, tokenizer, classes, device)
    ls = get_logit_scale(model)

    ds = torchvision.datasets.STL10("data", split="test", download=False)  # raw PIL
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:args.n]

    to_tensor = torchvision.transforms.ToTensor()

    def classify_batch(pil_list, prompt_lang):
        xs = torch.stack([to_tensor(im) for im in pil_list]).to(device)
        with torch.no_grad():
            feats = encode_image(model, xs, mean, std)
            return logits_for(feats, txt[prompt_lang], ls).argmax(-1).cpu().numpy()

    # ---- clean accuracy (sanity) ----
    clean_imgs = [ds[i][0].convert("RGB").resize((224, 224), Image.BICUBIC) for i in idx]
    true = np.array([ds[i][1] for i in idx])
    clean_pred = {l: classify_batch(clean_imgs, l) for l in LANGS}
    print("Clean accuracy per prompt-language:")
    for l in LANGS:
        print(f"  {l}: {100*(clean_pred[l]==true).mean():.1f}%")

    # ---- pick a target (wrong) class per image, fixed across conditions ----
    target = np.array([rng.choice([c for c in range(len(classes)) if c != true[k]])
                       for k in range(len(idx))])

    # ---- typographic attack matrix: written-language x prompt-language ----
    print("\nAttack-success rate ASR = P(prediction == WRITTEN target class):")
    header = "text\\prompt |" + "".join(f"{l:>7}" for l in LANGS)
    print(header)
    asr = {}
    flip = {}  # also track 'fooled' = pred != true
    for text_lang in LANGS:
        # build attacked images: write target class name in text_lang
        att_imgs = [draw_word(ds[idx[k]][0], TRANSLATIONS[classes[target[k]]][text_lang],
                              args.font_size) for k in range(len(idx))]
        row = []
        for prompt_lang in LANGS:
            pred = classify_batch(att_imgs, prompt_lang)
            a = (pred == target).mean()
            f = (pred != true).mean()
            asr[(text_lang, prompt_lang)] = float(a)
            flip[(text_lang, prompt_lang)] = float(f)
            row.append(a)
        print(f"{text_lang:>10} |" + "".join(f"{100*v:6.1f}%" for v in row))

    # diagonal (same lang) vs off-diagonal (cross-lingual) summary
    same = np.mean([asr[(l, l)] for l in LANGS])
    cross = np.mean([asr[(a, b)] for a in LANGS for b in LANGS if a != b])
    print(f"\nMean ASR same-language (text==prompt): {100*same:.1f}%")
    print(f"Mean ASR cross-language (text!=prompt): {100*cross:.1f}%")
    best_attacker = max(LANGS, key=lambda t: np.mean([asr[(t, p)] for p in LANGS]))
    print(f"Strongest attacker language (highest mean ASR across prompts): {best_attacker}")

    # save a few example images for the writeup
    for j in range(args.save_examples):
        k = j
        im = draw_word(ds[idx[k]][0], TRANSLATIONS[classes[target[k]]]["en"], args.font_size)
        im.save(f"results/typo_example_{j}_true-{classes[true[k]]}_wrote-{classes[target[k]]}.png")

    out = {"clean_acc": {l: float((clean_pred[l]==true).mean()) for l in LANGS},
           "asr": {f"{a}->{b}": asr[(a, b)] for a in LANGS for b in LANGS},
           "same_lang_asr": float(same), "cross_lang_asr": float(cross),
           "strongest_attacker": best_attacker, "n": len(idx)}
    with open("results/typographic.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/typographic.json")


if __name__ == "__main__":
    main()
