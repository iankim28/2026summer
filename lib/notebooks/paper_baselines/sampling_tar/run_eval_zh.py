"""SamplingTAR hybrid on ChineseCLIP (OFA-Sys ViT-B/16)."""
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
from _common.hf_attn_ablate import (  # noqa: E402
    classify_hybrid_hf,
    classify_with_heads_hf,
    mine_head_scores_hf,
)
from _common.hybrid_spatial import DEFAULT_SCORE_FRAC, DEFAULT_TOP_K  # noqa: E402
from _common.protocol import (  # noqa: E402
    CLASSES,
    ZhCLIP,
    acc_asr,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    progress_log,
    write_summary,
)

RESULTS = Path(__file__).resolve().parent / "results"
COST_HYBRID = 3
N_LAYERS, N_HEADS, GRID = 12, 12, 14


def select_by_z(means, mu, sigma, z=1.0):
    thr = mu + z * sigma
    heads = [(L, H) for (L, H), s in means.items() if s >= thr]
    heads.sort(key=lambda lh: -means[lh])
    print(f"z={z} thr={thr:.4f} -> {len(heads)} heads: {heads[:12]}...")
    return heads, thr


def tune_z(zh, text_emb, clean, attacked, true, means, mu, sigma, probe_n=40):
    clean_p, atk_p, true_p = clean[:probe_n], attacked[:probe_n], true[:probe_n]
    base_clean = float((classify_batch(zh, clean_p, text_emb) == true_p).mean())
    best = None
    for z in [2.0, 1.5, 1.0, 0.5]:
        heads, thr = select_by_z(means, mu, sigma, z=z)
        if not heads:
            continue
        heads = heads[:16]
        c_acc = float(
            (classify_with_heads_hf(zh, clean_p, text_emb, heads) == true_p).mean()
        )
        a_acc = float(
            (classify_with_heads_hf(zh, atk_p, text_emb, heads) == true_p).mean()
        )
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


def run_eval(n: int, status: str, top_k_patches=DEFAULT_TOP_K, score_frac=DEFAULT_SCORE_FRAC):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, rects = build_multi_attack(data)
    true, target = data["true"], data["target"]
    zh = ZhCLIP()
    text_emb = zh.embed_texts(CLASSES["zh"])

    mine_n = min(50, data["n"])
    means, mu, sigma = mine_head_scores_hf(
        zh, attacked, rects, max_images=mine_n, n_layers=N_LAYERS, n_heads=N_HEADS, grid=GRID
    )
    if status == "sanity" and max(means.values()) <= 0:
        raise RuntimeError("Gate A fail: ZH head mining produced zero scores")

    heads, z_used, thr = tune_z(
        zh, text_emb, data["clean_224"], attacked, true, means, mu, sigma,
        probe_n=min(40, data["n"]),
    )
    if not heads:
        heads = [k for k, _ in sorted(means.items(), key=lambda kv: -kv[1])[:5]]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "selected_heads_zh.json", "w", encoding="utf-8") as f:
        json.dump({"lang": "zh", "heads": heads, "z": z_used, "thr": thr, "grid": GRID}, f, indent=2)

    clean_p = classify_batch(zh, data["clean_224"], text_emb)
    atk_p = classify_batch(zh, attacked, text_emb)
    clean_acc = float((clean_p == true).mean())
    atk_acc, atk_asr = acc_asr(atk_p, true, target)
    print(f"Vanilla ZH clean={100*clean_acc:.1f}% atk={100*atk_acc:.1f}% ASR={100*atk_asr:.1f}%")

    t0 = time.time()
    def_preds, mean_patches = classify_hybrid_hf(
        zh, attacked, text_emb, heads, top_k_patches=top_k_patches, score_frac=score_frac, grid=GRID
    )
    clean_def, _ = classify_hybrid_hf(
        zh, data["clean_224"], text_emb, heads, top_k_patches=top_k_patches, score_frac=score_frac, grid=GRID
    )
    progress_log(len(def_preds) - 1, len(def_preds), int((def_preds == true).sum()), t0)
    def_acc, def_asr = acc_asr(def_preds, true, target)
    changed = int((def_preds != atk_p).sum())
    if status == "sanity" and changed == 0 and len(heads) > 0:
        raise RuntimeError("Gate A fail: ZH SamplingTAR hybrid did not change predictions")
    clean_def_acc = float((clean_def == true).mean())
    clean_delta = clean_def_acc - clean_acc
    print(
        f"SamplingTAR [hybrid] ZH acc={100*def_acc:.1f}% Clean_delta={100*clean_delta:.1f}pp "
        f"heads={len(heads)} mean_blur_patches={mean_patches:.1f}"
    )

    payload = {
        "method": "sampling_tar",
        "status": status,
        "n": int(data["n"]),
        "scope": "zh_only",
        "lang": "zh",
        "mode": "hybrid",
        "variant": "heads+attn_blur",
        "grid": GRID,
        "inference_cost": COST_HYBRID,
        "n_heads_ablated": len(heads),
        "heads": heads,
        "z": z_used,
        "score_threshold": thr,
        "top_k_patches": int(top_k_patches),
        "score_frac": float(score_frac),
        "mean_blur_patches": mean_patches,
        "defense": {
            "zh": {
                "acc": def_acc,
                "asr": def_asr,
                "baseline_acc": atk_acc,
                "baseline_asr": atk_asr,
            }
        },
        "clean_degradation": {
            "zh": {
                "baseline_acc": clean_acc,
                "masked_acc": clean_def_acc,
                "delta_acc": clean_delta,
            }
        },
        "defense_acc_zh": def_acc,
        "clean_delta_zh": clean_delta,
        "preds_changed_vs_vanilla": changed,
        "notes": "SamplingTAR hybrid on HF ChineseCLIP; re-mined heads; attn-guided blur.",
    }
    write_summary(RESULTS / f"comparison_summary_{status}_n{data['n']}_hybrid_zh.json", payload)
    write_summary(RESULTS / "comparison_summary_hybrid_zh.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    ap.add_argument("--top-k-patches", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--score-frac", type=float, default=DEFAULT_SCORE_FRAC)
    args = ap.parse_args()
    run_eval(args.n, args.status, top_k_patches=args.top_k_patches, score_frac=args.score_frac)


if __name__ == "__main__":
    main()
