"""Deeper analysis of the (naive) multilingual VOTING ensemble on shared-encoder M-CLIP.

- more samples (default n=2000, vs 200-300 before) -> tight standard errors
- typographic attack written in EACH language (not just English)
- both hard majority-vote and soft (mean-softmax) ensembles
- compares ensemble vs best/mean single language (does voting buy anything?)
"""
import argparse, json, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
from mclip_lib import (load_model, build_text_embeddings, encode_image, logits_for,
                       get_logit_scale, LANGS, TRANSLATIONS, STL10_CLASSES)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # covers Latin + CJK

def draw_word(pil, word, size=40):
    img = pil.convert("RGB").resize((224, 224), Image.BICUBIC); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size); bb = d.textbbox((0, 0), word, font=f)
    w = bb[2]-bb[0]
    if w > 208:                                   # shrink long words to fit width
        f = ImageFont.truetype(FONT, max(14, int(size*204/w))); bb = d.textbbox((0,0), word, font=f); w = bb[2]-bb[0]
    h = bb[3]-bb[1]; x = (224-w)//2; y = 224-h-16
    d.rectangle([x-8, y-8, x+w+8, y+h+12], fill=(255,255,255))
    d.text((x-bb[0], y-bb[1]), word, fill=(0,0,0), font=f)
    return img

def se(p, n):  # binomial standard error in %
    return 100*np.sqrt(max(p*(1-p), 1e-9)/n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--font_size", type=int, default=40)
    args = ap.parse_args()
    device = "cuda"; rng = random.Random(0)

    model, tok, _, mean, std = load_model(device)
    classes = STL10_CLASSES
    TXT = build_text_embeddings(model, tok, classes, device)   # dict lang -> [C,D]
    ls = get_logit_scale(model)

    ds = torchvision.datasets.STL10("data", split="test", download=False)
    idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[:args.n]
    N = len(idx)
    true = np.array([ds[i][1] for i in idx])
    target = np.array([rng.choice([c for c in range(10) if c != true[k]]) for k in range(N)])
    tt = torchvision.transforms.ToTensor()

    @torch.no_grad()
    def softmaxes(pil_list):
        """return dict lang -> [N,C] softmax probs, batched."""
        outs = {l: [] for l in LANGS}
        for s in range(0, len(pil_list), args.bs):
            batch = [im.convert("RGB").resize((224, 224), Image.BICUBIC) for im in pil_list[s:s+args.bs]]
            xs = torch.stack([tt(im) for im in batch]).to(device)
            f = encode_image(model, xs, mean, std)
            for l in LANGS:
                outs[l].append(F.softmax(logits_for(f, TXT[l], ls), -1).cpu().numpy())
        return {l: np.concatenate(outs[l]) for l in LANGS}

    def metrics(probs, tag):
        preds = {l: probs[l].argmax(1) for l in LANGS}
        # ensembles
        soft = np.mean([probs[l] for l in LANGS], 0).argmax(1)
        P = np.stack([preds[l] for l in LANGS])                      # [L,N]
        vote = np.array([np.bincount(P[:, i], minlength=10).argmax() for i in range(N)])
        agree = (P == P[0:1]).all(0).mean()
        per_acc = {l: (preds[l] == true).mean() for l in LANGS}
        per_asr = {l: (preds[l] == target).mean() for l in LANGS}
        row = {
            "vote_acc": (vote == true).mean(), "vote_asr": (vote == target).mean(),
            "soft_acc": (soft == true).mean(), "soft_asr": (soft == target).mean(),
            "agree": float(agree),
            "best_single_acc": float(max(per_acc.values())),
            "mean_single_acc": float(np.mean(list(per_acc.values()))),
            "per_lang_acc": {l: float(per_acc[l]) for l in LANGS},
            "per_lang_asr": {l: float(per_asr[l]) for l in LANGS},
        }
        return row

    results = {"n": N, "rows": {}}
    clean = metrics(softmaxes([ds[i][0] for i in idx]), "clean")
    results["rows"]["clean"] = clean

    print(f"=== Voting-ensemble deeper analysis (STL-10, shared M-CLIP, n={N}) ===")
    print(f"(standard error on a rate near 50% is ~{se(0.5,N):.1f}%; near 95% ~{se(0.95,N):.1f}%)\n")
    print(f"{'written':>9} | {'VOTE acc':>14} | {'VOTE ASR':>14} | {'soft acc':>9} | {'agree':>6} | {'best-1 acc':>10}")
    def fmt(p): return f"{100*p:5.1f}±{se(p,N):.1f}%"
    print(f"{'(clean)':>9} | {fmt(clean['vote_acc']):>14} | {'   -        ':>14} | {100*clean['soft_acc']:7.1f}% | {100*clean['agree']:5.1f}% | {100*clean['best_single_acc']:9.1f}%")

    for wl in LANGS:
        att = [draw_word(ds[idx[k]][0], TRANSLATIONS[classes[target[k]]][wl], args.font_size) for k in range(N)]
        r = metrics(softmaxes(att), wl)
        results["rows"][wl] = r
        print(f"{wl:>9} | {fmt(r['vote_acc']):>14} | {fmt(r['vote_asr']):>14} | {100*r['soft_acc']:7.1f}% | {100*r['agree']:5.1f}% | {100*r['best_single_acc']:9.1f}%")

    # does voting buy anything over a single language?
    print("\nEnsemble vs single language (accuracy under each written-language attack):")
    print(f"{'written':>9} | {'vote':>6} | {'mean-1':>6} | {'best-1':>6} | vote - best_single")
    for wl in ["clean"]+LANGS:
        r = results["rows"][wl]
        print(f"{wl:>9} | {100*r['vote_acc']:5.1f}% | {100*r['mean_single_acc']:5.1f}% | {100*r['best_single_acc']:5.1f}% | {100*(r['vote_acc']-r['best_single_acc']):+5.1f}%")

    # per-language ASR breakdown for the strongest (English) written attack
    print("\nPer-query-language ASR under ENGLISH-written attack (who gets fooled):")
    for l in LANGS:
        p = results["rows"]["en"]["per_lang_asr"][l]
        print(f"  {l}: {100*p:5.1f}±{se(p,N):.1f}%")

    json.dump(results, open("results/ensemble_analysis.json", "w"), indent=2, ensure_ascii=False)
    print("\nsaved results/ensemble_analysis.json")

if __name__ == "__main__":
    main()
