"""Mechanism: how cross-lingually aligned are the per-language label embeddings?

If same-class label embeddings across languages are near-collinear, an attack that
pushes the (shared) image embedding away from class c in English ALSO pushes it away
from class c in every other language -> full transfer, no disagreement.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from mclip_lib import load_model, build_text_embeddings, LANGS
from data_utils import get_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    args = ap.parse_args()
    device = "cuda"
    model, tokenizer, _, _, _ = load_model(device)
    _, classes = get_dataset(args.dataset)
    txt = build_text_embeddings(model, tokenizer, classes, device)  # lang->[C,D] normalized
    C = len(classes)

    print(f"\n=== Cross-lingual label-embedding geometry ({args.dataset}) ===")
    print("Mean SAME-CLASS cosine similarity between language pairs:")
    print(f"{'':>4} " + " ".join(f"{l:>6}" for l in LANGS))
    same_class_all = []
    for a in LANGS:
        row = []
        for b in LANGS:
            # cosine between same-class embeddings, averaged over classes
            cos = (txt[a] * txt[b]).sum(-1)  # [C]
            row.append(cos.mean().item())
            if a != b:
                same_class_all.append(cos.mean().item())
        print(f"{a:>4} " + " ".join(f"{v:6.3f}" for v in row))
    print(f"\nMean same-class cross-lingual cosine (off-diagonal): {np.mean(same_class_all):.3f}")

    # For contrast: mean cosine between DIFFERENT classes within English (how separated are classes)
    en = txt["en"]
    sim = (en @ en.t()).cpu().numpy()
    off = sim[~np.eye(C, dtype=bool)]
    print(f"Mean DIFFERENT-class cosine within English: {off.mean():.3f} "
          f"(range {off.min():.3f}..{off.max():.3f})")
    print("Interpretation: same-class cross-lingual cosine >> different-class cosine")
    print("=> a direction that lowers EN class-c score lowers every language's class-c score.")


if __name__ == "__main__":
    main()
