"""SamplingTAR-style circuit ablation on PROTOCOL dual-box CIFAR.

Port of SamplingTAR (Liu et al.): mine heads by CLS→text-patch attribution
overlap, select via z-threshold on score distribution, ablate with
fix_attn_head_list (alpha=1). EN ViT-B/32 openai. No SAE (direct attn score).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.attn_ablate import (  # noqa: E402
    cls_to_patch_attn,
    fix_cls_attn_heads,
    heads_to_layer_spec,
    patch_mask_from_rects,
)
from _common.protocol import (  # noqa: E402
    CLASSES,
    DEVICE,
    EnCLIP,
    acc_asr,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    progress_log,
    write_summary,
)

RESULTS = Path(__file__).resolve().parent / "results"
COST_PROXY = 2
N_LAYERS, N_HEADS, GRID = 12, 12, 7


@torch.no_grad()
def mine_head_scores(en: EnCLIP, attacked, rects, max_images=50):
    """Attribution-like score: mean CLS attention mass on sticker patches."""
    scores = {(L, H): [] for L in range(N_LAYERS) for H in range(N_HEADS)}
    n = min(max_images, len(attacked))
    for i in range(n):
        x = en.pp(attacked[i]).unsqueeze(0).to(DEVICE)
        mask = patch_mask_from_rects(rects[i], grid=GRID)
        if not mask.any():
            continue
        for layer in range(N_LAYERS):
            attn = cls_to_patch_attn(en.m.visual, x, layer)[0].float().cpu().numpy()
            for h in range(N_HEADS):
                scores[(layer, h)].append(float(attn[h, mask].sum()))
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    vals = np.array(list(means.values()))
    mu, sigma = float(vals.mean()), float(vals.std() + 1e-8)
    print(f"Mined on {n} imgs; score mean={mu:.4f} std={sigma:.4f}")
    return means, mu, sigma


def select_by_z(means, mu, sigma, z=1.0):
    """Keep heads with score >= mu + z*sigma (SamplingTAR z-threshold)."""
    thr = mu + z * sigma
    heads = [(L, H) for (L, H), s in means.items() if s >= thr]
    heads.sort(key=lambda lh: -means[lh])
    print(f"z={z} thr={thr:.4f} -> {len(heads)} heads: {heads[:12]}...")
    return heads, thr


@torch.no_grad()
def classify_with_heads(en, imgs, text_emb, heads, batch_size=32):
    spec = heads_to_layer_spec(heads)
    preds = []
    with fix_cls_attn_heads(en.m.visual, spec, alpha=1.0):
        for i in range(0, len(imgs), batch_size):
            imf = en.embed_images(imgs[i : i + batch_size])
            preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def tune_z(en, text_emb, clean, attacked, true, means, mu, sigma, probe_n=40):
    """Pick z in {0.5,1,1.5,2} maximizing attack acc subject to clean drop <= 5pp."""
    clean_p, atk_p, true_p = clean[:probe_n], attacked[:probe_n], true[:probe_n]
    base_clean = float((classify_batch(en, clean_p, text_emb) == true_p).mean())
    best = None  # (score, a_acc, z, heads, thr)  score = a_acc - drop
    for z in [2.0, 1.5, 1.0, 0.5]:  # prefer fewer heads
        heads, thr = select_by_z(means, mu, sigma, z=z)
        if not heads:
            continue
        heads = heads[:16]
        c_acc = float((classify_with_heads(en, clean_p, text_emb, heads) == true_p).mean())
        a_acc = float((classify_with_heads(en, atk_p, text_emb, heads) == true_p).mean())
        drop = base_clean - c_acc
        print(f"  tune z={z}: heads={len(heads)} atk={100*a_acc:.1f}% clean_drop={100*drop:.1f}pp")
        score = a_acc - max(drop, 0.0)
        feasible = drop <= 0.05
        if best is None:
            best = (score, a_acc, z, heads, thr, feasible)
        elif feasible and not best[5]:
            best = (score, a_acc, z, heads, thr, True)
        elif feasible == best[5] and (a_acc > best[1] or (a_acc == best[1] and z > best[2])):
            best = (score, a_acc, z, heads, thr, feasible)
    if best is None:
        heads, thr = select_by_z(means, mu, sigma, z=2.0)
        return heads[:8], 2.0, thr
    return best[3], best[2], best[4]


def run_eval(n: int, status: str):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, rects = build_multi_attack(data)
    true, target = data["true"], data["target"]

    en = EnCLIP()
    text_emb = en.embed_texts(CLASSES["en"])

    mine_n = min(50, data["n"])
    means, mu, sigma = mine_head_scores(en, attacked, rects, max_images=mine_n)
    if status == "sanity" and max(means.values()) <= 0:
        raise RuntimeError("Gate A fail: head mining produced zero scores")

    heads, z_used, thr = tune_z(
        en, text_emb, data["clean_224"], attacked, true, means, mu, sigma,
        probe_n=min(40, data["n"]),
    )
    if not heads:
        # top-5 by score
        heads = [k for k, _ in sorted(means.items(), key=lambda kv: -kv[1])[:5]]
        print("Fallback top-5 heads", heads)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "selected_heads.json", "w", encoding="utf-8") as f:
        json.dump({"heads": heads, "z": z_used, "thr": thr, "n_mine": mine_n}, f, indent=2)

    clean_p = classify_batch(en, data["clean_224"], text_emb)
    atk_p = classify_batch(en, attacked, text_emb)
    clean_acc = float((clean_p == true).mean())
    atk_acc, atk_asr = acc_asr(atk_p, true, target)
    print(f"Vanilla clean={100*clean_acc:.1f}% atk={100*atk_acc:.1f}% ASR={100*atk_asr:.1f}%")

    t0 = time.time()
    def_preds = classify_with_heads(en, attacked, text_emb, heads)
    progress_log(len(def_preds) - 1, len(def_preds), int((def_preds == true).sum()), t0)
    def_acc, def_asr = acc_asr(def_preds, true, target)
    changed = int((def_preds != atk_p).sum())
    print(f"Hook fired: preds_changed={changed}/{len(def_preds)}")
    if status == "sanity" and changed == 0 and len(heads) > 0:
        raise RuntimeError("Gate A fail: SamplingTAR hook did not change predictions")

    clean_def = classify_with_heads(en, data["clean_224"], text_emb, heads)
    clean_def_acc = float((clean_def == true).mean())
    clean_delta = clean_def_acc - clean_acc
    print(
        f"SamplingTAR EN acc={100*def_acc:.1f}% Clean_delta={100*clean_delta:.1f}pp "
        f"heads={len(heads)} z={z_used}"
    )

    payload = {
        "method": "sampling_tar",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_only",
        "inference_cost": COST_PROXY,
        "n_heads_ablated": len(heads),
        "heads": heads,
        "z": z_used,
        "score_threshold": thr,
        "defense": {
            "en": {
                "acc": def_acc,
                "asr": def_asr,
                "baseline_acc": atk_acc,
                "baseline_asr": atk_asr,
            }
        },
        "clean_degradation": {
            "en": {
                "baseline_acc": clean_acc,
                "masked_acc": clean_def_acc,
                "delta_acc": clean_delta,
            }
        },
        "defense_acc_en": def_acc,
        "clean_delta_en": clean_delta,
        "preds_changed_vs_vanilla": changed,
        "notes": "SamplingTAR-style: attn-to-sticker mining + z-threshold; fix_attn alpha=1. No SAE.",
    }
    write_summary(RESULTS / f"comparison_summary_{status}_n{data['n']}.json", payload)
    write_summary(RESULTS / "comparison_summary.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    args = ap.parse_args()
    run_eval(args.n, args.status)


if __name__ == "__main__":
    main()
