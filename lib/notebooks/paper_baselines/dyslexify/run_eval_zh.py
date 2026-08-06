"""Dyslexify hybrid on ChineseCLIP (OFA-Sys ViT-B/16)."""
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
    typographic_scores_hf,
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


def greedy_select(zh, text_emb, clean, attacked, true, ranked,
                  eps=0.001, stop_at_delta=0.05, max_heads=24, probe_n=40):
    clean_p = clean[:probe_n]
    atk_p = attacked[:probe_n]
    true_p = true[:probe_n]
    base_clean = float((classify_batch(zh, clean_p, text_emb) == true_p).mean())
    base_atk = float((classify_batch(zh, atk_p, text_emb) == true_p).mean())
    print(f"Select probe n={probe_n} base_clean={100*base_clean:.1f}% base_atk={100*base_atk:.1f}%")

    selected = []
    cur_atk = base_atk
    skips = 0
    for score, L, H in ranked:
        cand = selected + [(L, H)]
        c_acc = float((classify_with_heads_hf(zh, clean_p, text_emb, cand) == true_p).mean())
        a_acc = float((classify_with_heads_hf(zh, atk_p, text_emb, cand) == true_p).mean())
        gain = a_acc - cur_atk
        clean_drop = base_clean - c_acc
        if gain < eps or clean_drop > stop_at_delta:
            skips += 1
            if skips >= 15:
                break
            continue
        skips = 0
        selected.append((L, H))
        cur_atk = a_acc
        print(
            f"  greedy keep L{L}H{H} score={score:.3f} atk={100*a_acc:.1f}% "
            f"drop={100*clean_drop:.1f}pp",
            flush=True,
        )
        if len(selected) >= max_heads:
            break

    best = (cur_atk if selected else base_atk, list(selected))
    K = min(12, len(ranked))
    for k in range(1, K + 1):
        cand = [(L, H) for _, L, H in ranked[:k]]
        c_acc = float((classify_with_heads_hf(zh, clean_p, text_emb, cand) == true_p).mean())
        a_acc = float((classify_with_heads_hf(zh, atk_p, text_emb, cand) == true_p).mean())
        drop = base_clean - c_acc
        print(f"  prefix k={k}: atk={100*a_acc:.1f}% clean_drop={100*drop:.1f}pp", flush=True)
        if drop <= stop_at_delta and a_acc >= best[0]:
            best = (a_acc, cand)

    selected = best[1]
    if not selected:
        selected = [(ranked[0][1], ranked[0][2])]
        print("Fallback: single top head", selected)
    print(f"Selected {len(selected)} heads: {selected} (probe_atk={100*best[0]:.1f}%)")
    return selected


def run_eval(n: int, status: str, top_k_patches=DEFAULT_TOP_K, score_frac=DEFAULT_SCORE_FRAC):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, rects = build_multi_attack(data)
    true, target = data["true"], data["target"]
    zh = ZhCLIP()
    text_emb = zh.embed_texts(CLASSES["zh"])

    mine_n = min(50, data["n"])
    ranked, _ = typographic_scores_hf(
        zh, attacked, rects, max_images=mine_n, n_layers=N_LAYERS, n_heads=N_HEADS, grid=GRID
    )
    if status == "sanity" and ranked[0][0] <= 0:
        raise RuntimeError("Gate A fail: ZH typographic scores all zero")

    heads = greedy_select(
        zh, text_emb, data["clean_224"], attacked, true, ranked,
        probe_n=min(40, data["n"]),
        max_heads=16 if n <= 100 else 24,
    )
    if status == "sanity" and len(heads) == 0:
        heads = [(L, H) for _, L, H in ranked[:3]]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "selected_heads_zh.json", "w", encoding="utf-8") as f:
        json.dump({"lang": "zh", "heads": heads, "n_mine": mine_n, "grid": GRID}, f, indent=2)

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
        raise RuntimeError("Gate A fail: ZH Dyslexify hybrid did not change predictions")
    clean_def_acc = float((clean_def == true).mean())
    clean_delta = clean_def_acc - clean_acc
    print(
        f"Dyslexify [hybrid] ZH acc={100*def_acc:.1f}% Clean_delta={100*clean_delta:.1f}pp "
        f"heads={len(heads)} mean_blur_patches={mean_patches:.1f}"
    )

    payload = {
        "method": "dyslexify",
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
        "notes": "Dyslexify hybrid on HF ChineseCLIP; re-mined heads; attn-guided blur.",
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
