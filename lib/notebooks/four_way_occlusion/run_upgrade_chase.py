"""Phase 2/3 chase: KO (and optional JA) ViT-L/14 + adaptive occlusion.

CUDA required.
  Smoke: python run_upgrade_chase.py --n 100 --ko-l14
  Final: python run_upgrade_chase.py --n 1000 --ko-l14 --adaptive
  Both L/14: python run_upgrade_chase.py --n 1000 --ko-l14 --ja-l14 --adaptive
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper_baselines"))

from run_eval import (  # noqa: E402
    ALL_LANGS,
    CLASSES,
    DILATE,
    FILL,
    RESULTS,
    THR,
    TMPL,
    TOP_K,
    DEVICE,
    JaCLIP,
    KoCLIP,
    ZhCLIP,
    _clip_feat,
    _tag_en,
    apply_mask,
    build_en_zh_multi,
    classify_and_attn,
    classify_batch,
    classify_gated,
    compute_cams,
    fmt_pct,
    mixed2000,
    score_langs,
    train_mode_gate,
)
from run_agg_chase import (  # noqa: E402
    MASK_BUILDERS,
    MASK_NOTES,
    build_masks_for_arm,
    score_arm,
)
from _common.protocol import EnCLIP, load_protocol_data, write_summary  # noqa: E402
from run_eval import dilate_mask  # noqa: E402

assert torch.cuda.is_available(), "CUDA required"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

KO_L14 = "Bingsu/clip-vit-large-patch14-ko"
JA_L14 = "hf-hub:llm-jp/llm-jp-clip-vit-large-patch14"


class KoCLIP_L14:
    lang = "ko"
    backend = "hf"
    model_id = KO_L14

    def __init__(self):
        self.m = (
            AutoModel.from_pretrained(KO_L14, attn_implementation="eager")
            .to(DEVICE)
            .eval()
        )
        self.p = AutoProcessor.from_pretrained(KO_L14)

    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(
            text=[TMPL["ko"].format(w) for w in words],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t["input_ids"], attention_mask=t["attention_mask"]
        )
        return F.normalize(_clip_feat(out), dim=-1)


class JaCLIP_L14:
    lang = "ja"
    backend = "open_clip"
    model_id = JA_L14

    def __init__(self):
        mid = JA_L14
        self.m, _, self.pp = open_clip.create_model_and_transforms(mid)
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer(mid)

    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL["ja"].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)


def expand_low_coverage_masks(masks, cams, arm, cov_tau=0.06, dilate_hi=5, top_k_hi=3):
    """If mask coverage < tau, rebuild with stronger dilate/top_k."""
    out = []
    n_exp = 0
    for i, m in enumerate(masks):
        if float(m.mean()) < cov_tau:
            m2 = MASK_BUILDERS[arm](
                cams["en"][i],
                cams["zh"][i],
                cams["ko"][i],
                cams["ja"][i],
                threshold=THR,
                dilate=dilate_hi,
                top_k=top_k_hi,
            )
            out.append(m2)
            n_exp += 1
        else:
            out.append(m)
    return out, n_exp


def classify_gated_second_pass(
    models, text_embs, images, masks, gate_flags, true_unused, target
):
    """Gated black fill; if pred still equals attack target, dilate mask once and retry."""
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    out = {lang: np.zeros(n, dtype=np.int64) for lang in ALL_LANGS}
    n_second = 0
    for i in range(n):
        if not gate_flags[i]:
            for lang in ALL_LANGS:
                pred, _ = classify_and_attn(models[lang], text_embs[lang], images[i])
                out[lang][i] = pred
            continue
        m = masks[i]
        defended = apply_mask(images[i], m, fill=FILL)
        preds = {}
        for lang in ALL_LANGS:
            pred, _ = classify_and_attn(models[lang], text_embs[lang], defended)
            preds[lang] = pred
            out[lang][i] = pred
        # Second pass if EN (primary threat) still predicts target
        if int(preds["en"]) == int(target[i]):
            m2 = dilate_mask(m.astype(bool), iterations=2)
            defended2 = apply_mask(images[i], m2, fill=FILL)
            n_second += 1
            for lang in ALL_LANGS:
                pred, _ = classify_and_attn(models[lang], text_embs[lang], defended2)
                out[lang][i] = pred
    return out, float(gate_flags.mean()), n_second


def score_gated_only(
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
    second_pass=False,
):
    if second_pass:
        atk_preds, def_frac, n_sec_atk = classify_gated_second_pass(
            models, text_embs, atk_imgs, masks_atk, gate["gate_attacked"], true, target
        )
        cln_preds, def_frac_cln, n_sec_cln = classify_gated_second_pass(
            models, text_embs, clean_imgs, masks_clean, gate["gate_clean"], true, target
        )
    else:
        atk_preds, def_frac = classify_gated(
            models, text_embs, atk_imgs, masks_atk, gate["gate_attacked"]
        )
        cln_preds, def_frac_cln = classify_gated(
            models, text_embs, clean_imgs, masks_clean, gate["gate_clean"]
        )
        n_sec_atk = n_sec_cln = 0

    atk_scores = score_langs(atk_preds, true, target)
    cln_acc = {lang: float((cln_preds[lang] == true).mean()) for lang in ALL_LANGS}
    gated = {
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
        "second_pass_atk": int(n_sec_atk),
        "second_pass_clean": int(n_sec_cln),
        **{
            f"{lang}_mixed_2000": mixed2000(atk_scores[lang]["acc"], cln_acc[lang])
            for lang in ALL_LANGS
        },
        "mean_mixed_2000": mixed2000(
            atk_scores["mean_acc"], float(np.mean(list(cln_acc.values())))
        ),
    }
    return gated


def run_pairwise_en_ko(models, text_embs, clean_imgs, atk_imgs, cams_clean, cams_atk, gate, never_clean_acc, true, target):
    """Production-style EN&KO pairwise masks scored on EN+KO only (and report 4-lang with same mask)."""
    from run_agg_chase import build_cc_bbox_from_saliency
    from run_eval import n_cam_intersection

    def pair_masks(cams):
        return [
            build_cc_bbox_from_saliency(
                n_cam_intersection(cams["en"][i], cams["ko"][i]),
                threshold=THR,
                dilate=DILATE,
                top_k=TOP_K,
            )
            for i in range(len(cams["en"]))
        ]

    mc = pair_masks(cams_clean)
    ma = pair_masks(cams_atk)
    return score_gated_only(
        models, text_embs, clean_imgs, atk_imgs, mc, ma, gate, never_clean_acc, true, target
    )


def run(
    n: int,
    ko_l14: bool,
    ja_l14: bool,
    arms: list[str],
    adaptive: bool,
    pairwise_en_ko: bool,
):
    data = load_protocol_data(n=n)
    true = data["true"]
    target = data["target"]
    clean_imgs = data["clean_224"]
    atk_imgs = build_en_zh_multi(data)

    ko_cls = KoCLIP_L14 if ko_l14 else KoCLIP
    ja_cls = JaCLIP_L14 if ja_l14 else JaCLIP
    print(
        f"Loading models (KO={'L14' if ko_l14 else 'B32'}, JA={'L14' if ja_l14 else 'B16'})…",
        flush=True,
    )
    models = {
        "en": _tag_en(EnCLIP()),
        "zh": ZhCLIP(),
        "ko": ko_cls(),
        "ja": ja_cls(),
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

    variants = {}
    for arm in arms:
        print(f"\n=== {arm}: standard ===", flush=True)
        masks_clean = build_masks_for_arm(cams_clean, arm)
        masks_atk = build_masks_for_arm(cams_atk, arm)
        # full always+gated via score_arm
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
        g = arm_scores["gated"]
        print(
            f"  gated mean_atk={fmt_pct(g['mean_atk_acc'])} "
            f"EN={fmt_pct(g['en_acc'])} ZH={fmt_pct(g['zh_acc'])} "
            f"KO={fmt_pct(g['ko_acc'])} JA={fmt_pct(g['ja_acc'])} "
            f"MIXED={fmt_pct(g['mean_mixed_2000'])} "
            f"CleanΔEN={100*g['clean_delta_en']:+.1f}pp",
            flush=True,
        )
        variants[f"{arm}"] = {"mask_note": MASK_NOTES[arm], "arms": arm_scores}

        if adaptive:
            print(f"=== {arm}: low-coverage expand ===", flush=True)
            mc2, n_exp_c = expand_low_coverage_masks(masks_clean, cams_clean, arm)
            ma2, n_exp_a = expand_low_coverage_masks(masks_atk, cams_atk, arm)
            g2 = score_gated_only(
                models,
                text_embs,
                clean_imgs,
                atk_imgs,
                mc2,
                ma2,
                gate,
                never_clean_acc,
                true,
                target,
                second_pass=False,
            )
            print(
                f"  gated mean_atk={fmt_pct(g2['mean_atk_acc'])} "
                f"MIXED={fmt_pct(g2['mean_mixed_2000'])} "
                f"expanded atk/clean={n_exp_a}/{n_exp_c} "
                f"CleanΔEN={100*g2['clean_delta_en']:+.1f}pp",
                flush=True,
            )
            variants[f"{arm}+cov_expand"] = {
                "mask_note": MASK_NOTES[arm] + " + low-cov expand",
                "n_expanded_atk": n_exp_a,
                "n_expanded_clean": n_exp_c,
                "arms": {"gated": g2},
            }

            print(f"=== {arm}: target-persistence second pass ===", flush=True)
            g3 = score_gated_only(
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
                second_pass=True,
            )
            print(
                f"  gated mean_atk={fmt_pct(g3['mean_atk_acc'])} "
                f"MIXED={fmt_pct(g3['mean_mixed_2000'])} "
                f"2nd_pass atk/clean={g3['second_pass_atk']}/{g3['second_pass_clean']} "
                f"CleanΔEN={100*g3['clean_delta_en']:+.1f}pp",
                flush=True,
            )
            variants[f"{arm}+second_pass"] = {
                "mask_note": MASK_NOTES[arm] + " + target-persist 2nd pass",
                "arms": {"gated": g3},
            }

            print(f"=== {arm}: cov_expand + second_pass ===", flush=True)
            g4 = score_gated_only(
                models,
                text_embs,
                clean_imgs,
                atk_imgs,
                mc2,
                ma2,
                gate,
                never_clean_acc,
                true,
                target,
                second_pass=True,
            )
            print(
                f"  gated mean_atk={fmt_pct(g4['mean_atk_acc'])} "
                f"MIXED={fmt_pct(g4['mean_mixed_2000'])} "
                f"CleanΔEN={100*g4['clean_delta_en']:+.1f}pp",
                flush=True,
            )
            variants[f"{arm}+cov_expand+second_pass"] = {
                "mask_note": MASK_NOTES[arm] + " + cov expand + 2nd pass",
                "arms": {"gated": g4},
            }

    if pairwise_en_ko:
        print("\n=== pairwise EN&KO (4-lang score with EN&KO masks) ===", flush=True)
        gpk = run_pairwise_en_ko(
            models,
            text_embs,
            clean_imgs,
            atk_imgs,
            cams_clean,
            cams_atk,
            gate,
            never_clean_acc,
            true,
            target,
        )
        print(
            f"  gated mean_atk={fmt_pct(gpk['mean_atk_acc'])} "
            f"EN={fmt_pct(gpk['en_acc'])} KO={fmt_pct(gpk['ko_acc'])} "
            f"MIXED={fmt_pct(gpk['mean_mixed_2000'])}",
            flush=True,
        )
        variants["pairwise_en_ko"] = {
            "mask_note": "EN&KO pairwise cc_bbox, scored on all 4 langs",
            "arms": {"gated": gpk},
        }

    ranking = sorted(
        (
            (
                name,
                variants[name]["arms"]["gated"]["mean_mixed_2000"],
                variants[name]["arms"]["gated"]["mean_atk_acc"],
            )
            for name in variants
        ),
        key=lambda t: (t[1], t[2]),
        reverse=True,
    )
    print("\n=== Ranking (gated mean MIXED) ===", flush=True)
    for name, mx, atk in ranking:
        beat = "BEATS OCR" if mx > 0.840 else ("~OCR" if mx >= 0.838 else "")
        print(
            f"  {name:40s} MIXED={fmt_pct(mx)}  atk={fmt_pct(atk)}  {beat}",
            flush=True,
        )

    summary = {
        "method": "upgrade_adaptive_chase",
        "n": int(data["n"]),
        "ko_model": KO_L14 if ko_l14 else "Bingsu/clip-vit-base-patch32-ko",
        "ja_model": JA_L14 if ja_l14 else "llm-jp/llm-jp-clip-vit-base-patch16",
        "zh_model": "OFA-Sys/chinese-clip-vit-base-patch16",
        "en_model": "openai ViT-B-32",
        "attack": "multi_en_zh",
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "fill": FILL,
        "ocr_target": {"mean_atk": 0.788, "mean_mixed": 0.840},
        "detector": {
            "primary": gate["primary"],
            "threshold": gate["threshold"],
            "fire_attacked": gate["fire_attacked"],
            "fire_clean": gate["fire_clean"],
            "val_auc": gate["val_auc"],
            "test_auc": gate["test_auc"],
        },
        "clean_never": never_clean_acc,
        "never_atk_mean": never_atk_scores["mean_acc"],
        "ranking_gated_mixed": [
            {"variant": a, "mean_mixed_2000": m, "mean_atk_acc": t} for a, m, t in ranking
        ],
        "variants": variants,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = ("koL14" if ko_l14 else "koB32") + ("_jaL14" if ja_l14 else "_jaB16")
    if adaptive:
        tag += "_adapt"
    out_path = RESULTS / f"upgrade_chase_n{data['n']}_{tag}.json"
    write_summary(out_path, summary)
    print("Saved", out_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ko-l14", action="store_true")
    ap.add_argument("--ja-l14", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--pairwise-en-ko", action="store_true", default=True)
    ap.add_argument("--no-pairwise-en-ko", action="store_true")
    ap.add_argument(
        "--arms",
        type=str,
        default="intersect4,en_cap_mean",
        help="comma-separated mask arms",
    )
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in MASK_BUILDERS:
            raise SystemExit(f"Unknown arm {a}")
    run(
        n=args.n,
        ko_l14=args.ko_l14,
        ja_l14=args.ja_l14,
        arms=arms,
        adaptive=args.adaptive,
        pairwise_en_ko=not args.no_pairwise_en_ko,
    )


if __name__ == "__main__":
    main()
