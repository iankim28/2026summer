"""Analyze saved per-language probs: agreement, prediction collapse, and whether a
cross-lingual DISAGREEMENT detector can separate adversarial from clean images.

This directly evaluates the proposal's 'disagreement detector' defense.
"""
import sys, json, argparse
import numpy as np
from mclip_lib import LANGS


def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, eps, 1); b = np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b), axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def disagreement_scores(probs):
    """probs: dict lang->[N,C]. Returns dict of per-image disagreement scores."""
    langs = LANGS
    preds = np.stack([probs[l].argmax(1) for l in langs], 0)  # [L,N]
    N = preds.shape[1]
    # number of unique predicted classes among languages
    n_unique = np.array([len(np.unique(preds[:, i])) for i in range(N)], dtype=float)
    # vote entropy over predicted classes
    vote_ent = np.zeros(N)
    for i in range(N):
        _, cnts = np.unique(preds[:, i], return_counts=True)
        p = cnts / cnts.sum()
        vote_ent[i] = -(p * np.log(p)).sum()
    # mean pairwise JS divergence of soft distributions
    L = len(langs)
    js_acc = np.zeros(N); npairs = 0
    for a in range(L):
        for b in range(a + 1, L):
            js_acc += js_divergence(probs[langs[a]], probs[langs[b]])
            npairs += 1
    mean_js = js_acc / npairs
    return {"n_unique": n_unique, "vote_entropy": vote_ent, "mean_js": mean_js}


def auc(neg, pos):
    """AUC that score>thresh => positive (adversarial). neg=clean scores, pos=adv scores."""
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    # (simple tie handling)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank_per_val = csum - (cnt - 1) / 2.0
    ranks = avg_rank_per_val[inv]
    n_pos = pos.shape[0]; n_neg = neg.shape[0]
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    args = ap.parse_args()
    d = np.load(args.probs)
    labels = d["labels"]
    eps_keys = sorted({k.split("_")[0][3:] for k in d.files if k.startswith("adv")},
                      key=lambda x: float(x))

    clean = {l: d[f"clean_{l}"] for l in LANGS}
    clean_sc = disagreement_scores(clean)

    print(f"\n=== Disagreement-detector analysis: {args.probs} ===")
    print("Clean agreement breakdown:")
    preds_c = np.stack([clean[l].argmax(1) for l in LANGS], 0)
    print(f"  all-5-agree: {(preds_c==preds_c[0:1]).all(0).mean()*100:.1f}%  "
          f"mean unique classes/img: {clean_sc['n_unique'].mean():.2f}")

    print(f"\n{'eps':>5} | {'all-agree%':>10} | {'uniq/img':>8} | "
          f"AUC(n_uniq) AUC(vote_ent) AUC(mean_js)  [<.5 = detector FAILS]")
    rows = {}
    for e in eps_keys:
        adv = {l: d[f"adv{e}_{l}"] for l in LANGS}
        preds_a = np.stack([adv[l].argmax(1) for l in LANGS], 0)
        allagree = (preds_a == preds_a[0:1]).all(0).mean()
        adv_sc = disagreement_scores(adv)
        a1 = auc(clean_sc["n_unique"], adv_sc["n_unique"])
        a2 = auc(clean_sc["vote_entropy"], adv_sc["vote_entropy"])
        a3 = auc(clean_sc["mean_js"], adv_sc["mean_js"])
        rows[e] = {"all_agree": float(allagree), "uniq_per_img": float(adv_sc["n_unique"].mean()),
                   "auc_n_unique": float(a1), "auc_vote_entropy": float(a2), "auc_mean_js": float(a3)}
        print(f"{e:>5} | {allagree*100:10.1f} | {adv_sc['n_unique'].mean():8.2f} | "
              f"   {a1:.3f}      {a2:.3f}        {a3:.3f}")

    # prediction collapse: how concentrated are adversarial predictions?
    print("\nAdversarial prediction collapse (English, largest eps):")
    e = eps_keys[-1]
    ap_preds = d[f"adv{e}_en"].argmax(1)
    vals, cnts = np.unique(ap_preds, return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:5]
    print(f"  eps={e}: top predicted classes (count): " +
          ", ".join(f"cls{int(v)}:{int(c)}" for c, v in top))

    outpath = args.probs.replace("probs_", "detector_").replace(".npz", ".json")
    with open(outpath, "w") as f:
        json.dump({"clean_all_agree": float((preds_c==preds_c[0:1]).all(0).mean()),
                   "by_eps": rows}, f, indent=2)
    print(f"\nsaved {outpath}")


if __name__ == "__main__":
    main()
