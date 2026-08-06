"""Mask-aggregation chase: beat OCR+blur 4-lang means without OCR.

Reuses four_way_occlusion models/cams/gate; varies how EN/ZH/KO/JA Attn-last
maps combine before cc_bbox_black.

CUDA required.
  Smoke: python run_agg_chase.py --n 100
  Final: python run_agg_chase.py --n 1000
  Subset of arms: python run_agg_chase.py --n 100 --arms en_cap_max,pair_union
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import shared pipeline pieces from run_eval
from run_eval import (  # noqa: E402
    ALL_LANGS,
    DILATE,
    FILL,
    RESULTS,
    THR,
    TOP_K,
    JaCLIP,
    KoCLIP,
    ZhCLIP,
    _tag_en,
    align_cam,
    apply_mask,
    build_en_zh_multi,
    cam_to_mask,
    classify_always,
    classify_gated,
    compute_cams,
    filter_mask_components,
    fmt_pct,
    mixed2000,
    n_cam_intersection,
    partner_mean_cams,
    score_langs,
    train_mode_gate,
)
from _common.protocol import EnCLIP, classify_batch, load_protocol_data, write_summary  # noqa: E402

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))


def build_cc_bbox_from_saliency(saliency, threshold=THR, dilate=DILATE, top_k=TOP_K):
    mask = cam_to_mask(saliency, threshold, dilate=dilate)
    return filter_mask_components(mask, top_k=top_k, bbox_snap=True)


def mask_intersect4(cam_en, cam_zh, cam_ko, cam_ja, **kw):
    inter = n_cam_intersection(cam_en, cam_zh, cam_ko, cam_ja)
    return build_cc_bbox_from_saliency(inter, **kw)


def mask_en_cap_max(cam_en, cam_zh, cam_ko, cam_ja, **kw):
    en = align_cam(cam_en)
    partner = np.maximum.reduce(
        [align_cam(cam_zh), align_cam(cam_ko), align_cam(cam_ja)]
    )
    return build_cc_bbox_from_saliency(np.minimum(en, partner), **kw)


def mask_en_cap_mean(cam_en, cam_zh, cam_ko, cam_ja, **kw):
    en = align_cam(cam_en)
    partner = (
        align_cam(cam_zh) + align_cam(cam_ko) + align_cam(cam_ja)
    ) / 3.0
    return build_cc_bbox_from_saliency(np.minimum(en, partner), **kw)


def mask_en_cap_maj(cam_en, cam_zh, cam_ko, cam_ja, threshold=THR, dilate=DILATE, top_k=TOP_K):
    """EN cc_bbox AND majority(ZH,KO,JA binary at thr); empty maj falls back to EN."""
    from run_eval import dilate_mask

    en = align_cam(cam_en)
    partners = [align_cam(c) for c in (cam_zh, cam_ko, cam_ja)]
    bins = []
    for p in partners:
        thr_v = np.percentile(p, threshold * 100)
        bins.append(p >= thr_v)
    maj = (bins[0].astype(np.int8) + bins[1].astype(np.int8) + bins[2].astype(np.int8)) >= 2
    en_mask = build_cc_bbox_from_saliency(en, threshold=threshold, dilate=dilate, top_k=top_k)
    if not maj.any():
        return en_mask
    maj_d = dilate_mask(maj, iterations=max(dilate, 1))
    out = en_mask & maj_d
    if not out.any():
        return en_mask
    return filter_mask_components(out, top_k=top_k, bbox_snap=True)


def _pair_mask(cam_en, cam_l, **kw):
    inter = n_cam_intersection(cam_en, cam_l)
    return build_cc_bbox_from_saliency(inter, **kw)


def mask_pair_union(cam_en, cam_zh, cam_ko, cam_ja, **kw):
    m = _pair_mask(cam_en, cam_zh, **kw)
    m = m | _pair_mask(cam_en, cam_ko, **kw)
    m = m | _pair_mask(cam_en, cam_ja, **kw)
    return m


MASK_BUILDERS = {
    "intersect4": mask_intersect4,
    "en_cap_max": mask_en_cap_max,
    "en_cap_mean": mask_en_cap_mean,
    "en_cap_maj": mask_en_cap_maj,
    "pair_union": mask_pair_union,
}

MASK_NOTES = {
    "intersect4": "EN&ZH&KO&JA Attn-last cc_bbox (baseline)",
    "en_cap_max": "EN & max(ZH,KO,JA) Attn-last cc_bbox",
    "en_cap_mean": "EN & mean(ZH,KO,JA) Attn-last cc_bbox",
    "en_cap_maj": "EN & majority(ZH,KO,JA>=2/3 after thr) Attn-last cc_bbox",
    "pair_union": "bbox-union of EN&ZH | EN&KO | EN&JA",
}


def build_masks_for_arm(cams, arm: str, dilate=DILATE, top_k=TOP_K):
    builder = MASK_BUILDERS[arm]
    kw = dict(threshold=THR, dilate=dilate, top_k=top_k)
    return [
        builder(cams["en"][i], cams["zh"][i], cams["ko"][i], cams["ja"][i], **kw)
        for i in range(len(cams["en"]))
    ]


def score_arm(
    models,
    text_embs,
    clean_imgs,
    atk_imgs,
    masks_clean,
    masks_atk,
    gate,
    never_clean_acc,
    true,
    target,
):
    arms = {}

    atk_preds, cov = classify_always(models, text_embs, atk_imgs, masks_atk)
    cln_preds, _ = classify_always(models, text_embs, clean_imgs, masks_clean)
    atk_scores = score_langs(atk_preds, true, target)
    cln_acc = {lang: float((cln_preds[lang] == true).mean()) for lang in ALL_LANGS}
    arms["always"] = {
        **{f"{lang}_acc": atk_scores[lang]["acc"] for lang in ALL_LANGS},
        **{f"{lang}_asr": atk_scores[lang]["asr"] for lang in ALL_LANGS},
        "mean_atk_acc": atk_scores["mean_acc"],
        **{f"clean_{lang}_acc": cln_acc[lang] for lang in ALL_LANGS},
        "mean_clean_acc": float(np.mean(list(cln_acc.values()))),
        **{
            f"clean_delta_{lang}": cln_acc[lang] - never_clean_acc[lang]
            for lang in ALL_LANGS
        },
        "mean_mask_coverage": cov,
        **{
            f"{lang}_mixed_2000": mixed2000(atk_scores[lang]["acc"], cln_acc[lang])
            for lang in ALL_LANGS
        },
        "mean_mixed_2000": mixed2000(
            atk_scores["mean_acc"], float(np.mean(list(cln_acc.values())))
        ),
    }

    atk_preds, def_frac = classify_gated(
        models, text_embs, atk_imgs, masks_atk, gate["gate_attacked"]
    )
    cln_preds, def_frac_cln = classify_gated(
        models, text_embs, clean_imgs, masks_clean, gate["gate_clean"]
    )
    atk_scores = score_langs(atk_preds, true, target)
    cln_acc = {lang: float((cln_preds[lang] == true).mean()) for lang in ALL_LANGS}
    arms["gated"] = {
        **{f"{lang}_acc": atk_scores[lang]["acc"] for lang in ALL_LANGS},
        **{f"{lang}_asr": atk_scores[lang]["asr"] for lang in ALL_LANGS},
        "mean_atk_acc": atk_scores["mean_acc"],
        **{f"clean_{lang}_acc": cln_acc[lang] for lang in ALL_LANGS},
        "mean_clean_acc": float(np.mean(list(cln_acc.values()))),
        **{
            f"clean_delta_{lang}": cln_acc[lang] - never_clean_acc[lang]
            for lang in ALL_LANGS
        },
        "mean_mask_coverage": float(np.mean([m.mean() for m in masks_atk])),
        "defended_frac_atk": def_frac,
        "defended_frac_clean": def_frac_cln,
        "fire_attacked": gate["fire_attacked"],
        "fire_clean": gate["fire_clean"],
        **{
            f"{lang}_mixed_2000": mixed2000(atk_scores[lang]["acc"], cln_acc[lang])
            for lang in ALL_LANGS
        },
        "mean_mixed_2000": mixed2000(
            atk_scores["mean_acc"], float(np.mean(list(cln_acc.values())))
        ),
    }
    return arms


def run(n: int, arm_names: list[str], dilate: int = DILATE, top_k: int = TOP_K):
    from run_eval import CLASSES  # local

    data = load_protocol_data(n=n)
    true = data["true"]
    target = data["target"]
    clean_imgs = data["clean_224"]
    atk_imgs = build_en_zh_multi(data)

    print("Loading EN/ZH/KO/JA…", flush=True)
    models = {
        "en": _tag_en(EnCLIP()),
        "zh": ZhCLIP(),
        "ko": KoCLIP(),
        "ja": JaCLIP(),
    }
    text_embs = {lang: models[lang].embed_texts(CLASSES[lang]) for lang in ALL_LANGS}

    print("Never baselines…", flush=True)
    never_clean = {
        lang: classify_batch(models[lang], clean_imgs, text_embs[lang]) for lang in ALL_LANGS
    }
    never_atk = {
        lang: classify_batch(models[lang], atk_imgs, text_embs[lang]) for lang in ALL_LANGS
    }
    never_clean_acc = {lang: float((never_clean[lang] == true).mean()) for lang in ALL_LANGS}
    never_atk_scores = score_langs(never_atk, true, target)
    never_block = {
        **{f"{lang}_acc": never_atk_scores[lang]["acc"] for lang in ALL_LANGS},
        **{f"{lang}_asr": never_atk_scores[lang]["asr"] for lang in ALL_LANGS},
        "mean_atk_acc": never_atk_scores["mean_acc"],
        **{f"clean_{lang}_acc": never_clean_acc[lang] for lang in ALL_LANGS},
        "mean_clean_acc": float(np.mean(list(never_clean_acc.values()))),
        "mean_mask_coverage": 0.0,
        **{
            f"{lang}_mixed_2000": mixed2000(
                never_atk_scores[lang]["acc"], never_clean_acc[lang]
            )
            for lang in ALL_LANGS
        },
        "mean_mixed_2000": mixed2000(
            never_atk_scores["mean_acc"], float(np.mean(list(never_clean_acc.values())))
        ),
    }

    print("Attn-last cams (clean)…", flush=True)
    cams_clean = compute_cams(models, text_embs, clean_imgs, label="clean")
    print("Attn-last cams (attacked)…", flush=True)
    cams_atk = compute_cams(models, text_embs, atk_imgs, label="atk")

    gate = train_mode_gate(cams_clean, cams_atk, data["n"])
    print(
        f"Gate primary={gate['primary']} thr={gate['threshold']:.4f} "
        f"fire_atk={gate['fire_attacked']:.3f} fire_clean={gate['fire_clean']:.3f}",
        flush=True,
    )

    by_arm = {}
    for arm in arm_names:
        print(f"\n=== Arm: {arm} ({MASK_NOTES[arm]}) dilate={dilate} top_k={top_k} ===", flush=True)
        t0 = time.time()
        masks_clean = build_masks_for_arm(cams_clean, arm, dilate=dilate, top_k=top_k)
        masks_atk = build_masks_for_arm(cams_atk, arm, dilate=dilate, top_k=top_k)
        arm_scores = score_arm(
            models,
            text_embs,
            clean_imgs,
            atk_imgs,
            masks_clean,
            masks_atk,
            gate,
            never_clean_acc,
            true,
            target,
        )
        arm_scores["never"] = never_block
        by_arm[arm] = {
            "mask_note": MASK_NOTES[arm],
            "dilate": dilate,
            "top_k": top_k,
            "arms": arm_scores,
        }
        g = arm_scores["gated"]
        print(
            f"  gated mean_atk={fmt_pct(g['mean_atk_acc'])} "
            f"EN={fmt_pct(g['en_acc'])} ZH={fmt_pct(g['zh_acc'])} "
            f"KO={fmt_pct(g['ko_acc'])} JA={fmt_pct(g['ja_acc'])} "
            f"MIXED={fmt_pct(g['mean_mixed_2000'])} "
            f"CleanΔEN={100*g['clean_delta_en']:+.1f}pp "
            f"cov={g['mean_mask_coverage']:.3f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    # Rank by gated mean MIXED
    ranking = sorted(
        (
            (
                arm,
                by_arm[arm]["arms"]["gated"]["mean_mixed_2000"],
                by_arm[arm]["arms"]["gated"]["mean_atk_acc"],
            )
            for arm in arm_names
        ),
        key=lambda t: (t[1], t[2]),
        reverse=True,
    )
    print("\n=== Ranking (gated mean MIXED) ===", flush=True)
    for arm, mx, atk in ranking:
        print(f"  {arm:14s} MIXED={fmt_pct(mx)}  atk={fmt_pct(atk)}", flush=True)

    summary = {
        "method": "four_way_mask_aggregation_chase",
        "n": int(data["n"]),
        "attack": "multi_en_zh",
        "threshold": THR,
        "dilate": dilate,
        "top_k": top_k,
        "fill": FILL,
        "ocr_target": {"mean_atk": 0.788, "mean_mixed": 0.840},
        "gate_note": (
            "Gate trained on EN Attn-last vs mean(ZH,KO,JA) Attn-last features; "
            "same gate for all mask arms."
        ),
        "detector": {
            "primary": gate["primary"],
            "threshold": gate["threshold"],
            "fire_attacked": gate["fire_attacked"],
            "fire_clean": gate["fire_clean"],
            "val_auc": gate["val_auc"],
            "test_auc": gate["test_auc"],
        },
        "clean_never": never_clean_acc,
        "arms_evaluated": arm_names,
        "ranking_gated_mixed": [
            {"arm": a, "mean_mixed_2000": m, "mean_atk_acc": t} for a, m, t in ranking
        ],
        "by_arm": by_arm,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"d{dilate}_k{top_k}"
    out_path = RESULTS / f"agg_chase_n{data['n']}_{tag}.json"
    write_summary(out_path, summary)
    print("Saved", out_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument(
        "--arms",
        type=str,
        default=",".join(MASK_BUILDERS.keys()),
        help="comma-separated arm names",
    )
    ap.add_argument("--dilate", type=int, default=DILATE)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    args = ap.parse_args()
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in names:
        if a not in MASK_BUILDERS:
            raise SystemExit(f"Unknown arm {a}; choose from {list(MASK_BUILDERS)}")
    run(n=args.n, arm_names=names, dilate=args.dilate, top_k=args.top_k)


if __name__ == "__main__":
    main()
