"""Core experiment: single-language (English) attack, measure transfer to all langs.

Tests H1: does an English-only attack transfer only PARTIALLY to other languages?
Saves per-language softmax probs on clean & adv images for downstream defenses.
"""
import sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from mclip_lib import (load_model, build_text_embeddings, encode_image,
                       logits_for, get_logit_scale, LANGS)
from data_utils import get_loader
from attacks import fgsm, pgd

EPS_GRID = [2, 4, 8, 16]  # /255


@torch.no_grad()
def per_lang_probs(model, x, mean, std, txt, ls):
    feats = encode_image(model, x, mean, std)
    out = {}
    for l in LANGS:
        out[l] = F.softmax(logits_for(feats, txt[l], ls), dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--attack", default="pgd", choices=["pgd", "fgsm"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--attacked", default="en", help="comma-sep langs the attack targets")
    ap.add_argument("--eps_grid", default="", help="comma-sep eps (in /255) overriding default")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    attacked = args.attacked.split(",")
    global EPS_GRID
    if args.eps_grid:
        EPS_GRID = [float(e) for e in args.eps_grid.split(",")]

    device = "cuda"
    model, tokenizer, _, mean, std = load_model(device)
    loader, classes = get_loader(args.dataset, n=args.n, batch_size=args.bs)
    txt = build_text_embeddings(model, tokenizer, classes, device)
    ls = get_logit_scale(model)

    # storage: probs[eps][lang] -> [N, C]; clean probs too
    N = args.n
    summary = {"dataset": args.dataset, "attack": args.attack, "steps": args.steps,
               "attacked": attacked, "n": N, "eps_grid": EPS_GRID, "results": {}}

    # collect clean first
    all_labels = []
    clean_probs = {l: [] for l in LANGS}
    adv_probs = {e: {l: [] for l in LANGS} for e in EPS_GRID}

    t0 = time.time()
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        all_labels.append(y.cpu().numpy())
        cp = per_lang_probs(model, x, mean, std, txt, ls)
        for l in LANGS:
            clean_probs[l].append(cp[l].cpu().numpy())
        for e in EPS_GRID:
            eps = e / 255.0
            if args.attack == "pgd":
                x_adv = pgd(model, x, y, mean, std, txt, ls, eps,
                            steps=args.steps, attacked_langs=attacked)
            else:
                x_adv = fgsm(model, x, y, mean, std, txt, ls, eps, attacked_langs=attacked)
            ap_ = per_lang_probs(model, x_adv, mean, std, txt, ls)
            for l in LANGS:
                adv_probs[e][l].append(ap_[l].cpu().numpy())
    print(f"attack/eval time: {time.time()-t0:.1f}s")

    labels = np.concatenate(all_labels)
    clean_probs = {l: np.concatenate(clean_probs[l]) for l in LANGS}
    adv_probs = {e: {l: np.concatenate(adv_probs[e][l]) for l in LANGS} for e in EPS_GRID}

    def acc(probs, l):
        return (probs[l].argmax(1) == labels).mean()

    def ens_acc(probs):
        avg = np.mean([probs[l] for l in LANGS], axis=0)
        return (avg.argmax(1) == labels).mean()

    def agreement(probs):
        preds = np.stack([probs[l].argmax(1) for l in LANGS], 0)  # [L,N]
        return (preds == preds[0:1]).all(0).mean()

    # clean
    clean_acc = {l: float(acc(clean_probs, l)) for l in LANGS}
    summary["clean"] = {"acc": clean_acc, "ens": float(ens_acc(clean_probs)),
                        "agreement": float(agreement(clean_probs))}

    print(f"\n=== {args.dataset} | attack={args.attack} steps={args.steps} | attacked={attacked} ===")
    hdr = "eps  " + "  ".join(f"{l:>6}" for l in LANGS) + "   ens    agree"
    print("CLEAN " + "  ".join(f"{clean_acc[l]*100:6.1f}" for l in LANGS) +
          f"  {ens_acc(clean_probs)*100:6.1f}  {agreement(clean_probs)*100:6.1f}")
    print(hdr)
    for e in EPS_GRID:
        ra = {l: float(acc(adv_probs[e], l)) for l in LANGS}
        ea = float(ens_acc(adv_probs[e]))
        ag = float(agreement(adv_probs[e]))
        summary["results"][e] = {"acc": ra, "ens": ea, "agreement": ag}
        print(f"{e:>3}  " + "  ".join(f"{ra[l]*100:6.1f}" for l in LANGS) +
              f"  {ea*100:6.1f}  {ag*100:6.1f}")

    # transfer fraction: (clean_acc_L - robust_acc_L) for non-attacked L,
    # normalized by the drop on the attacked language(s).
    print("\nTransfer fraction (acc drop on lang / acc drop on attacked-lang avg):")
    for e in EPS_GRID:
        ra = summary["results"][e]["acc"]
        att_drop = np.mean([clean_acc[l] - ra[l] for l in attacked]) + 1e-9
        tf = {l: float((clean_acc[l] - ra[l]) / att_drop) for l in LANGS if l not in attacked}
        summary["results"][e]["transfer_fraction"] = tf
        print(f"  eps={e}: " + "  ".join(f"{l}={tf[l]:.2f}" for l in tf))

    tag = f"_{args.tag}" if args.tag else ""
    np.savez(f"results/probs_{args.dataset}_{args.attack}{tag}.npz",
             labels=labels,
             **{f"clean_{l}": clean_probs[l] for l in LANGS},
             **{f"adv{e}_{l}": adv_probs[e][l] for e in EPS_GRID for l in LANGS})
    with open(f"results/transfer_{args.dataset}_{args.attack}{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsaved results/transfer_{args.dataset}_{args.attack}{tag}.json")


if __name__ == "__main__":
    main()
