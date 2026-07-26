"""Dyslexify-style typographic head ablation on PROTOCOL dual-box CIFAR.

Port of Dyslexify (Hufe et al.): mine heads by typographic attention score on
sticker patches, greedy-select by attack-acc gain with clean-acc budget, ablate
CLS→spatial attention (alpha=1) at inference. EN ViT-B/32 openai only.

Modes:
  heads  — original head ablation only
  hybrid — attn-guided Gaussian blur of top typo-attn patches + head ablation
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
from _common.hybrid_spatial import (  # noqa: E402
    DEFAULT_SCORE_FRAC,
    DEFAULT_TOP_K,
    classify_hybrid,
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
COST_HEADS = 2
COST_HYBRID = 3
N_LAYERS, N_HEADS, GRID = 12, 12, 7


@torch.no_grad()
def typographic_scores(en: EnCLIP, attacked, rects, max_images=50):
    """Mean CLS→sticker-patch attention / total spatial attn, per (layer, head)."""
    scores = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
    counts = 0
    n = min(max_images, len(attacked))
    for i in range(n):
        x = en.pp(attacked[i]).unsqueeze(0).to(DEVICE)
        mask = patch_mask_from_rects(rects[i], grid=GRID)
        if not mask.any():
            continue
        for layer in range(N_LAYERS):
            attn = cls_to_patch_attn(en.m.visual, x, layer)  # (1, H, P)
            a = attn[0].float().cpu().numpy()  # (H, P)
            total = a.sum(axis=1) + 1e-8
            typo = a[:, mask].sum(axis=1)
            scores[layer] += typo / total
        counts += 1
    scores /= max(counts, 1)
    ranked = []
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            ranked.append((float(scores[L, H]), L, H))
    ranked.sort(reverse=True)
    print(f"Typo scores from {counts} images; top5={ranked[:5]}")
    return ranked, scores


@torch.no_grad()
def classify_with_heads(en, imgs, text_emb, heads, batch_size=32):
    spec = heads_to_layer_spec(heads)
    preds = []
    with fix_cls_attn_heads(en.m.visual, spec, alpha=1.0):
        for i in range(0, len(imgs), batch_size):
            imf = en.embed_images(imgs[i : i + batch_size])
            preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def greedy_select(en, text_emb, clean, attacked, true, ranked,
                  eps=0.001, stop_at_delta=0.05, max_heads=24, probe_n=40):
    """Select a prefix of typo-ranked heads maximizing attack acc under clean budget."""
    clean_p = clean[:probe_n]
    atk_p = attacked[:probe_n]
    true_p = true[:probe_n]
    base_clean = float((classify_batch(en, clean_p, text_emb) == true_p).mean())
    base_atk = float((classify_batch(en, atk_p, text_emb) == true_p).mean())
    print(f"Select probe n={probe_n} base_clean={100*base_clean:.1f}% base_atk={100*base_atk:.1f}%")

    selected = []
    cur_atk = base_atk
    skips = 0
    for score, L, H in ranked:
        cand = selected + [(L, H)]
        c_acc = float((classify_with_heads(en, clean_p, text_emb, cand) == true_p).mean())
        a_acc = float((classify_with_heads(en, atk_p, text_emb, cand) == true_p).mean())
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
        c_acc = float((classify_with_heads(en, clean_p, text_emb, cand) == true_p).mean())
        a_acc = float((classify_with_heads(en, atk_p, text_emb, cand) == true_p).mean())
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


def run_eval(
    n: int,
    status: str,
    mode: str = "hybrid",
    top_k_patches: int = DEFAULT_TOP_K,
    score_frac: float = DEFAULT_SCORE_FRAC,
):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, rects = build_multi_attack(data)
    true, target = data["true"], data["target"]

    en = EnCLIP()
    text_emb = en.embed_texts(CLASSES["en"])

    mine_n = min(50, data["n"])
    ranked, score_mat = typographic_scores(en, attacked, rects, max_images=mine_n)
    if status == "sanity" and ranked[0][0] <= 0:
        raise RuntimeError("Gate A fail: typographic scores all zero")

    heads = greedy_select(
        en, text_emb, data["clean_224"], attacked, true, ranked,
        probe_n=min(40, data["n"]),
        max_heads=16 if n <= 100 else 24,
    )
    if status == "sanity" and len(heads) == 0:
        heads = [(L, H) for _, L, H in ranked[:3]]
        print("Sanity fallback: using top-3 scored heads", heads)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "selected_heads.json", "w", encoding="utf-8") as f:
        json.dump({"heads": heads, "n_mine": mine_n, "mode": mode}, f, indent=2)

    clean_p = classify_batch(en, data["clean_224"], text_emb)
    atk_p = classify_batch(en, attacked, text_emb)
    clean_acc = float((clean_p == true).mean())
    atk_acc, atk_asr = acc_asr(atk_p, true, target)
    print(f"Vanilla clean={100*clean_acc:.1f}% atk={100*atk_acc:.1f}% ASR={100*atk_asr:.1f}%")

    t0 = time.time()
    mean_patches = None
    if mode == "hybrid":
        def_preds, mean_patches = classify_hybrid(
            en,
            attacked,
            text_emb,
            heads,
            top_k_patches=top_k_patches,
            score_frac=score_frac,
        )
        clean_def, _ = classify_hybrid(
            en,
            data["clean_224"],
            text_emb,
            heads,
            top_k_patches=top_k_patches,
            score_frac=score_frac,
        )
        cost = COST_HYBRID
        notes = (
            f"Dyslexify hybrid: typo-attn heads + attn-guided blur "
            f"(top_k={top_k_patches}, score_frac={score_frac}) + CLS attn redirect."
        )
        variant = "heads+attn_blur"
    else:
        def_preds = classify_with_heads(en, attacked, text_emb, heads, batch_size=32)
        clean_def = classify_with_heads(en, data["clean_224"], text_emb, heads)
        cost = COST_HEADS
        notes = "Dyslexify-style: typo-attn score + greedy select; CLS attn redirect (alpha=1)."
        variant = "heads"

    for i in range(0, len(def_preds), 50):
        end = min(i + 50, len(def_preds))
        running = int((def_preds[:end] == true[:end]).sum())
        progress_log(end - 1, len(def_preds), running, t0)

    def_acc, def_asr = acc_asr(def_preds, true, target)
    changed = int((def_preds != atk_p).sum())
    print(f"Hook/hybrid fired: preds_changed={changed}/{len(def_preds)}")
    if status == "sanity" and changed == 0 and len(heads) > 0:
        raise RuntimeError("Gate A fail: ablation/hybrid did not change predictions")

    clean_def_acc = float((clean_def == true).mean())
    clean_delta = clean_def_acc - clean_acc
    print(
        f"Dyslexify [{mode}] EN acc={100*def_acc:.1f}% Clean_delta={100*clean_delta:.1f}pp "
        f"heads={len(heads)}"
        + (f" mean_blur_patches={mean_patches:.1f}" if mean_patches is not None else "")
    )

    payload = {
        "method": "dyslexify",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_only",
        "mode": mode,
        "variant": variant,
        "inference_cost": cost,
        "n_heads_ablated": len(heads),
        "heads": heads,
        "top_k_patches": int(top_k_patches) if mode == "hybrid" else None,
        "score_frac": float(score_frac) if mode == "hybrid" else None,
        "mean_blur_patches": mean_patches,
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
        "notes": notes,
    }
    suffix = f"_{mode}" if mode == "hybrid" else ""
    write_summary(RESULTS / f"comparison_summary_{status}_n{data['n']}{suffix}.json", payload)
    if mode == "hybrid":
        write_summary(RESULTS / "comparison_summary_hybrid.json", payload)
    else:
        write_summary(RESULTS / "comparison_summary.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    ap.add_argument("--mode", choices=["heads", "hybrid"], default="hybrid")
    ap.add_argument("--top-k-patches", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--score-frac", type=float, default=DEFAULT_SCORE_FRAC)
    args = ap.parse_args()
    run_eval(
        args.n,
        args.status,
        mode=args.mode,
        top_k_patches=args.top_k_patches,
        score_frac=args.score_frac,
    )


if __name__ == "__main__":
    main()
