"""Consensus-purification denoiser (proposal's MAIN defense) + adaptive evaluation.

Denoiser: small residual DnCNN-style CNN, x_hat = clip(x + f(x)).
Training (self-supervised): generate language-specific (English-PGD) adversarial
examples, train the denoiser so the PURIFIED image's per-language predictions all
match the CLEAN consensus pseudo-label (argmax of clean ensemble; no human labels),
plus an L2 fidelity term to the clean image to prevent collapse.

Evaluation:
  * clean (through denoiser)
  * non-adaptive: PGD attacks the classifier, then purify+classify
  * adaptive: PGD attacks the FULL pipeline classify(purify(x)) end-to-end
    (denoiser is differentiable -> exact white-box adaptive attack, not BPDA)
"""
import argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Subset
from mclip_lib import (load_model, build_text_embeddings, encode_image,
                       logits_for, get_logit_scale, LANGS)

DATA_ROOT = "/ssd4tb/etc/adversarial/data"
_pixel_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(224), T.ToTensor()])


class Denoiser(nn.Module):
    def __init__(self, depth=8, ch=64):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 3, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x + self.body(x), 0, 1)


def ensemble_logits(model, x_pixel, mean, std, txt, ls):
    feats = encode_image(model, x_pixel, mean, std)
    probs = torch.stack([F.softmax(logits_for(feats, txt[l], ls), -1) for l in LANGS], 0)
    return probs.mean(0)  # ensemble prob


def per_lang_ce(model, x_pixel, mean, std, txt, ls, y):
    feats = encode_image(model, x_pixel, mean, std)
    loss = 0.0
    for l in LANGS:
        loss = loss + F.cross_entropy(logits_for(feats, txt[l], ls), y)
    return loss


def pgd_classifier(model, x, y, mean, std, txt, ls, eps, steps, attacked_langs):
    """Standard PGD on the classifier (non-adaptive wrt denoiser)."""
    x0 = x.clone().detach()
    alpha = 2.5 * eps / steps
    x_adv = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        feats = encode_image(model, x_adv, mean, std)
        loss = sum(F.cross_entropy(logits_for(feats, txt[l], ls), y) for l in attacked_langs)
        g = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = torch.min(torch.max(x_adv + alpha * g.sign(), x0 - eps), x0 + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv


def pgd_adaptive(model, denoiser, x, y, mean, std, txt, ls, eps, steps):
    """Adaptive PGD: attack the full defended pipeline classify(purify(x))."""
    x0 = x.clone().detach()
    alpha = 2.5 * eps / steps
    x_adv = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        x_hat = denoiser(x_adv)
        # attack the ensemble loss of the PURIFIED image (true label -> maximize)
        feats = encode_image(model, x_hat, mean, std)
        loss = sum(F.cross_entropy(logits_for(feats, txt[l], ls), y) for l in LANGS)
        g = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = torch.min(torch.max(x_adv + alpha * g.sign(), x0 - eps), x0 + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv


def get_split(name, split, n, seed=0):
    if name == "stl10":
        ds = torchvision.datasets.STL10(DATA_ROOT, split=split, download=False, transform=_pixel_tf)
    else:
        ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=(split == "train"), download=False, transform=_pixel_tf)
    if n and n < len(ds):
        idx = np.random.default_rng(seed).permutation(len(ds))[:n]
        ds = Subset(ds, idx.tolist())
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="stl10")
    ap.add_argument("--train_n", type=int, default=3000)
    ap.add_argument("--test_n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=40)
    ap.add_argument("--train_steps", type=int, default=10, help="PGD steps for training adv")
    ap.add_argument("--lam_fid", type=float, default=5.0)
    ap.add_argument("--eval_eps", default="2,4,8,16")
    ap.add_argument("--eval_steps", type=int, default=20)
    ap.add_argument("--train_langs", default="en,ko,es,fr,ja",
                    help="languages whose CE the denoiser is trained on + pseudo-label source. "
                         "'en' = single-language ablation; all-5 = consensus.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda"
    train_langs = args.train_langs.split(",")

    model, tokenizer, _, mean, std = load_model(device)
    from data_utils import get_dataset
    _, classes = get_dataset(args.dataset)
    txt = build_text_embeddings(model, tokenizer, classes, device)
    ls = get_logit_scale(model)

    train_ds = get_split(args.dataset, "train", args.train_n)
    test_ds = get_split(args.dataset, "test", args.test_n, seed=123)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=8)

    den = Denoiser().to(device)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)

    print(f"=== Training consensus-purification denoiser on {args.dataset} "
          f"(train_n={args.train_n}, epochs={args.epochs}) ===")
    train_eps_choices = [2/255, 4/255, 8/255]
    for ep in range(args.epochs):
        den.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for x, _y in train_loader:
            x = x.to(device)
            # consensus pseudo-label from CLEAN predictions over train_langs (self-supervised)
            with torch.no_grad():
                feats_c = encode_image(model, x, mean, std)
                probs_c = torch.stack([F.softmax(logits_for(feats_c, txt[l], ls), -1)
                                       for l in train_langs], 0).mean(0)
                pseudo = probs_c.argmax(-1)
            # language-specific adversarial example (English PGD) -- same for all configs
            eps = train_eps_choices[nb % len(train_eps_choices)]
            den.eval()
            x_adv = pgd_classifier(model, x, pseudo, mean, std, txt, ls, eps,
                                   args.train_steps, attacked_langs=["en"])
            den.train()
            # purify and train to restore pseudo-label across train_langs + fidelity.
            # Also denoise the CLEAN image (identity-preservation) to protect clean accuracy.
            x_hat = den(x_adv)
            x_hat_clean = den(x)
            feats = encode_image(model, x_hat, mean, std)
            feats_cl = encode_image(model, x_hat_clean, mean, std)
            ce = (sum(F.cross_entropy(logits_for(feats, txt[l], ls), pseudo) for l in train_langs)
                  + sum(F.cross_entropy(logits_for(feats_cl, txt[l], ls), pseudo) for l in train_langs)
                  ) / (2 * len(train_langs))
            fid = F.mse_loss(x_hat, x) + F.mse_loss(x_hat_clean, x)
            loss = ce + args.lam_fid * fid
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}: loss={tot/nb:.4f}  ({time.time()-t0:.1f}s)")
    tag = f"_{args.tag}" if args.tag else ""
    torch.save(den.state_dict(), f"results/denoiser_{args.dataset}{tag}.pt")

    # ---------- Evaluation ----------
    den.eval()
    eval_eps = [int(e) for e in args.eval_eps.split(",")]

    def ens_acc_of(x_pixel, y):
        with torch.no_grad():
            return (ensemble_logits(model, x_pixel, mean, std, txt, ls).argmax(-1) == y).float().sum().item()

    res = {"clean_nodenoise": 0, "clean_denoise": 0,
           "nonadaptive": {e: 0 for e in eval_eps},
           "nonadaptive_nodefense": {e: 0 for e in eval_eps},
           "adaptive": {e: 0 for e in eval_eps}}
    total = 0
    for x, y in test_loader:
        x = x.to(device); y = y.to(device); total += y.numel()
        res["clean_nodenoise"] += ens_acc_of(x, y)
        with torch.no_grad():
            res["clean_denoise"] += ens_acc_of(den(x), y)
        for e in eval_eps:
            eps = e / 255.0
            # non-adaptive attack on classifier
            x_adv = pgd_classifier(model, x, y, mean, std, txt, ls, eps, args.eval_steps, ["en"])
            res["nonadaptive_nodefense"][e] += ens_acc_of(x_adv, y)
            with torch.no_grad():
                res["nonadaptive"][e] += ens_acc_of(den(x_adv), y)
            # adaptive attack through denoiser
            x_adv2 = pgd_adaptive(model, den, x, y, mean, std, txt, ls, eps, args.eval_steps)
            with torch.no_grad():
                res["adaptive"][e] += ens_acc_of(den(x_adv2), y)

    def pct(c): return 100.0 * c / total
    print(f"\n=== Denoiser evaluation ({args.dataset}, test_n={total}) ===")
    print(f"clean, no denoiser : {pct(res['clean_nodenoise']):.1f}%")
    print(f"clean, denoised    : {pct(res['clean_denoise']):.1f}%")
    print(f"\n{'eps':>4} | {'no-defense':>10} | {'denoised(non-adapt)':>19} | {'denoised(ADAPTIVE)':>18}")
    out = {"dataset": args.dataset, "total": total,
           "clean_nodenoise": pct(res["clean_nodenoise"]),
           "clean_denoise": pct(res["clean_denoise"]), "by_eps": {}}
    for e in eval_eps:
        nd = pct(res["nonadaptive_nodefense"][e])
        na = pct(res["nonadaptive"][e])
        ad = pct(res["adaptive"][e])
        out["by_eps"][e] = {"no_defense": nd, "denoised_nonadaptive": na, "denoised_adaptive": ad}
        print(f"{e:>4} | {nd:10.1f} | {na:19.1f} | {ad:18.1f}")
    out["train_langs"] = train_langs
    with open(f"results/denoiser_{args.dataset}{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results/denoiser_{args.dataset}{tag}.json")


if __name__ == "__main__":
    main()
