"""Clean zero-shot accuracy per language + cross-language agreement on clean images."""
import sys, json, argparse
import torch
import torch.nn.functional as F
from mclip_lib import (load_model, build_text_embeddings, encode_image,
                       logits_for, get_logit_scale, LANGS)
from data_utils import get_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    device = "cuda"
    model, tokenizer, _, mean, std = load_model(device)
    loader, classes = get_loader(args.dataset, n=args.n, batch_size=args.bs)
    txt = build_text_embeddings(model, tokenizer, classes, device)
    ls = get_logit_scale(model)

    correct = {l: 0 for l in LANGS}
    total = 0
    # agreement: fraction of images where all langs predict the same class
    all_agree = 0
    preds_store = {l: [] for l in LANGS}
    labels_store = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            feats = encode_image(model, x, mean, std)
            batch_preds = {}
            for l in LANGS:
                lg = logits_for(feats, txt[l], ls)
                p = lg.argmax(-1)
                batch_preds[l] = p
                correct[l] += (p == y).sum().item()
                preds_store[l].append(p.cpu())
            stacked = torch.stack([batch_preds[l] for l in LANGS], 0)  # [L, B]
            agree = (stacked == stacked[0:1]).all(0)
            all_agree += agree.sum().item()
            labels_store.append(y.cpu())
            total += y.numel()

    acc = {l: correct[l] / total for l in LANGS}
    print(f"=== Clean zero-shot accuracy ({args.dataset}, n={total}) ===")
    for l in LANGS:
        print(f"  {l}: {acc[l]*100:.2f}%")
    print(f"  all-language agreement (clean): {all_agree/total*100:.2f}%")

    out = {"dataset": args.dataset, "n": total, "acc": acc,
           "clean_agreement": all_agree / total}
    with open(f"results/clean_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
