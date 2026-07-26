"""Beat DP EN MIXED2000 81.65% with occlusion only (NO Defense-Prefix).

Arms (all gated with Phase-C ZH detector):
  1. cc_bbox + black
  2. OCR∪cc_bbox + black
  3. OCR∪cc_bbox + multi-fill ensemble (conf-drop of pre-def EN top class;
     tie-break EN–ZH agree, then max EN top-1 prob)
  4. Same as 3 but include raw image as 4th hypothesis (auto if 3 < bar)

CUDA required. Smoke: --n 16. Final: --n 1000.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NB = HERE.parent
PB = NB / "paper_baselines"
AD = NB / "attack_detector"
NEG = NB / "en_neglect_vs_blur"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(NEG))

from _common.protocol import (  # noqa: E402
    CLASSES,
    DEVICE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    build_multi_attack,
    load_protocol_data,
    write_summary,
)
from helpers import (  # noqa: E402
    apply_mask,
    build_cc_bbox_mask,
    classify_and_attn_en,
    classify_and_attn_zh,
    rects_to_mask,
)

RESULTS = HERE / "results"
DP_BAR = 0.8165
GATED_BLACK_REF = 0.7935
CACHE_NPZ = AD / "results" / "zh" / "multi" / "cache" / "attn_en_zh_clean_multi.npz"
FILLS = ("black", "blur", "mean")


def mixed2000(atk, clean):
    return 0.5 * float(atk) + 0.5 * float(clean)


def fmt(x):
    return f"{100 * float(x):.2f}%"


def load_gate_flags(n_images=1000):
    sys.path.insert(0, str(AD))
    from make_phase_ab_viz import build_feature_matrix, load_cache, train_gate  # noqa: E402

    packed = load_cache("zh", AD / "results" / "zh" / "multi")
    X, y, img_ids, _ = build_feature_matrix(packed["clean"], packed["atk"], n_images)
    gate = train_gate(X, y, img_ids, n_images)
    return {
        "gate_clean": gate["all_pred"][:n_images].astype(bool),
        "gate_attacked": gate["all_pred"][n_images:].astype(bool),
        "threshold": float(gate["threshold"]),
        "primary": gate["primary_name"],
        "fire_clean": float(gate["fire_clean"]),
        "fire_attacked": float(gate["fire_attacked"]),
    }


def load_cached_cams(n=1000):
    z = np.load(CACHE_NPZ, allow_pickle=False)
    return {
        "clean_en": z["clean_cams_en"][:n],
        "clean_l": z["clean_cams_l"][:n] if "clean_cams_l" in z.files else z["clean_cams_zh"][:n],
        "atk_en": z["atk_cams_en"][:n],
        "atk_l": z["atk_cams_l"][:n] if "atk_cams_l" in z.files else z["atk_cams_zh"][:n],
    }


def build_masks_from_cams(cams_en, cams_l, threshold=0.95):
    return [build_cc_bbox_mask(ce, cl, threshold=threshold) for ce, cl in zip(cams_en, cams_l)]


def compute_cams_live(en, zh, en_txt, zh_txt, images, label=""):
    cams_en, cams_l = [], []
    t0 = time.time()
    for i, img in enumerate(images):
        _, ce = classify_and_attn_en(en, en_txt, img, device=DEVICE)
        _, cl = classify_and_attn_zh(zh, zh_txt, img, device=DEVICE)
        cams_en.append(ce)
        cams_l.append(cl)
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  cams {label} {i+1}/{len(images)} {time.time()-t0:.0f}s", flush=True)
    return cams_en, cams_l


def detect_ocr_boxes(reader, pil_img):
    results = reader.readtext(np.asarray(pil_img))
    boxes = []
    for item in results:
        bbox = item[0]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes


def union_masks(attn_masks, ocr_masks):
    return [a | o for a, o in zip(attn_masks, ocr_masks)]


@torch.no_grad()
def probs_batch(model, text_emb, imgs, batch_size=64):
    """Return (preds [N], probs [N, C]) softmax over cosine sims."""
    all_preds, all_probs = [], []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i : i + batch_size])
        logits = imf @ text_emb.t()
        prob = F.softmax(logits.float() * 100.0, dim=-1)  # CLIP-style temp ~100
        pred = logits.argmax(-1)
        all_preds.append(pred.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


def pick_ensemble(
    en,
    zh,
    en_txt,
    zh_txt,
    img,
    mask,
    fills=FILLS,
    include_raw=False,
):
    """Conf-drop of pre-def EN top class; tie-break EN-ZH agree; then max EN p1."""
    raw_pred, raw_prob = probs_batch(en, en_txt, [img])
    c0 = int(raw_pred[0])
    p0 = float(raw_prob[0, c0])

    candidates = []
    if include_raw:
        candidates.append(("raw", img))
    for fill in fills:
        candidates.append((fill, apply_mask(img, mask, fill=fill)))

    best = None  # (score_tuple, pred_en, name)
    for name, cand in candidates:
        en_pred, en_prob = probs_batch(en, en_txt, [cand])
        zh_pred, _ = probs_batch(zh, zh_txt, [cand])
        en_p = int(en_pred[0])
        zh_p = int(zh_pred[0])
        drop = p0 - float(en_prob[0, c0])
        agree = int(en_p == zh_p)
        top_p = float(en_prob[0, en_p])
        # Maximize drop, then agree, then top_p
        key = (drop, agree, top_p)
        if best is None or key > best[0]:
            best = (key, en_p, name)
    return best[1], best[2], float(best[0][0])


def run_gated_arm(
    name,
    en,
    zh,
    en_txt,
    zh_txt,
    images,
    masks,
    gate_flags,
    mode,
    true,
    target,
    never_clean_acc,
):
    """mode: 'black' | 'ensemble' | 'ensemble_raw'."""
    n = len(images)
    preds = np.zeros(n, dtype=np.int64)
    n_def = 0
    pick_counts = {}
    t0 = time.time()

    # Batch pass-through
    pass_idx = np.where(~np.asarray(gate_flags, dtype=bool))[0]
    def_idx = np.where(np.asarray(gate_flags, dtype=bool))[0]
    if len(pass_idx):
        p_pass, _ = probs_batch(en, en_txt, [images[i] for i in pass_idx])
        preds[pass_idx] = p_pass

    include_raw = mode == "ensemble_raw"
    use_ensemble = mode in ("ensemble", "ensemble_raw")

    for k, i in enumerate(def_idx):
        n_def += 1
        if use_ensemble:
            pred, picked, _drop = pick_ensemble(
                en,
                zh,
                en_txt,
                zh_txt,
                images[i],
                masks[i],
                fills=FILLS,
                include_raw=include_raw,
            )
            preds[i] = pred
            pick_counts[picked] = pick_counts.get(picked, 0) + 1
        else:
            img = apply_mask(images[i], masks[i], fill="black")
            pred, _ = probs_batch(en, en_txt, [img])
            preds[i] = int(pred[0])
            pick_counts["black"] = pick_counts.get("black", 0) + 1
        if (k + 1) % 50 == 0 or (k + 1) == len(def_idx):
            print(
                f"    {name} defended {k+1}/{len(def_idx)} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    atk_acc, asr = acc_asr(preds, true, target)
    # This helper is used for both halves; caller passes matching true/target.
    # For clean half, asr vs attack target is meaningless but harmless.
    return {
        "preds": preds,
        "atk_or_clean_acc": float(atk_acc),
        "asr": float(asr),
        "defend_frac": float(n_def / max(n, 1)),
        "pick_counts": pick_counts,
        "elapsed_s": time.time() - t0,
    }


def eval_arm_both_halves(
    name,
    en,
    zh,
    en_txt,
    zh_txt,
    attacked,
    clean_imgs,
    atk_masks,
    clean_masks,
    g_atk,
    g_cln,
    mode,
    true,
    target,
    never_clean_acc,
):
    print(f"\n=== {name} mode={mode} ===")
    atk_out = run_gated_arm(
        name + "/atk",
        en,
        zh,
        en_txt,
        zh_txt,
        attacked,
        atk_masks,
        g_atk,
        mode,
        true,
        target,
        never_clean_acc,
    )
    cln_out = run_gated_arm(
        name + "/cln",
        en,
        zh,
        en_txt,
        zh_txt,
        clean_imgs,
        clean_masks,
        g_cln,
        mode,
        true,
        target,
        never_clean_acc,
    )
    atk_acc = float((atk_out["preds"] == true).mean())
    cln_acc = float((cln_out["preds"] == true).mean())
    asr = float((atk_out["preds"] == target).mean())
    mix = mixed2000(atk_acc, cln_acc)
    arm = {
        "name": name,
        "mode": mode,
        "atk_acc": atk_acc,
        "asr": asr,
        "clean_acc": cln_acc,
        "clean_delta": cln_acc - never_clean_acc,
        "mixed_2000": mix,
        "beats_dp": mix > DP_BAR,
        "defend_frac_attacked": atk_out["defend_frac"],
        "defend_frac_clean": cln_out["defend_frac"],
        "pick_counts_atk": atk_out["pick_counts"],
        "pick_counts_clean": cln_out["pick_counts"],
        "elapsed_s": atk_out["elapsed_s"] + cln_out["elapsed_s"],
    }
    print(
        f"  {name}: atk={fmt(atk_acc)} clean={fmt(cln_acc)} MIXED={fmt(mix)} "
        f"{'BEATS DP' if arm['beats_dp'] else 'below DP'} "
        f"picks_atk={atk_out['pick_counts']}"
    )
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--status", default="sanity")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print("Device:", DEVICE, torch.cuda.get_device_name(0))
    RESULTS.mkdir(parents=True, exist_ok=True)

    n_arg = args.n if args.n < 1000 else None
    data = load_protocol_data(n=n_arg)
    n = data["n"]
    attacked, _gt = build_multi_attack(data)
    true, target = data["true"], data["target"]
    clean_imgs = data["clean_224"]

    print("Loading EN/ZH...")
    en, zh = EnCLIP(), ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    # Masks from attn
    if n == 1000 and CACHE_NPZ.exists():
        print("Loading cached cams...")
        cams = load_cached_cams(1000)
        atk_attn = build_masks_from_cams(cams["atk_en"], cams["atk_l"], args.threshold)
        cln_attn = build_masks_from_cams(cams["clean_en"], cams["clean_l"], args.threshold)
    else:
        print("Computing cams live...")
        ae, al = compute_cams_live(en, zh, en_txt, zh_txt, attacked, "atk")
        ce, cl = compute_cams_live(en, zh, en_txt, zh_txt, clean_imgs, "clean")
        atk_attn = build_masks_from_cams(ae, al, args.threshold)
        cln_attn = build_masks_from_cams(ce, cl, args.threshold)

    # OCR masks
    if args.skip_ocr:
        atk_ocr = [np.zeros((224, 224), dtype=bool) for _ in range(n)]
        cln_ocr = [np.zeros((224, 224), dtype=bool) for _ in range(n)]
        print("OCR skipped")
    else:
        import easyocr

        print("Running EasyOCR...")
        reader = easyocr.Reader(["en", "ch_sim"], gpu=True)
        atk_ocr, cln_ocr = [], []
        t0 = time.time()
        for i in range(n):
            atk_ocr.append(rects_to_mask(detect_ocr_boxes(reader, attacked[i])))
            cln_ocr.append(rects_to_mask(detect_ocr_boxes(reader, clean_imgs[i])))
            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"  OCR {i+1}/{n} elapsed={time.time()-t0:.0f}s", flush=True)

    atk_union = union_masks(atk_attn, atk_ocr)
    cln_union = union_masks(cln_attn, cln_ocr)

    # Gates
    if n != 1000:
        print("WARN: gates from full-1000 detector; using first-n / sel remap carefully")
    gates = load_gate_flags(1000)
    # For subset: protocol sel stored in attack_pos
    sel = data["attack_pos"].get("_sel")
    if sel is not None:
        g_atk = gates["gate_attacked"][np.array(sel)]
        g_cln = gates["gate_clean"][np.array(sel)]
    else:
        g_atk = gates["gate_attacked"][:n]
        g_cln = gates["gate_clean"][:n]
    print(
        f"Gate {gates['primary']} thr={gates['threshold']:.4f} "
        f"fire_atk={gates['fire_attacked']:.3f} fire_cln={gates['fire_clean']:.3f}"
    )

    never_atk, _ = probs_batch(en, en_txt, attacked)
    never_cln, _ = probs_batch(en, en_txt, clean_imgs)
    never_atk_acc = float((never_atk == true).mean())
    never_cln_acc = float((never_cln == true).mean())
    print(
        f"Never: atk={fmt(never_atk_acc)} clean={fmt(never_cln_acc)} "
        f"MIXED={fmt(mixed2000(never_atk_acc, never_cln_acc))}"
    )

    arms = []
    # Arm A: gated cc_bbox black (ref re-run)
    arms.append(
        eval_arm_both_halves(
            "gated_cc_bbox_black",
            en,
            zh,
            en_txt,
            zh_txt,
            attacked,
            clean_imgs,
            atk_attn,
            cln_attn,
            g_atk,
            g_cln,
            "black",
            true,
            target,
            never_cln_acc,
        )
    )
    # Arm 1: gated OCR∪cc_bbox black
    arms.append(
        eval_arm_both_halves(
            "gated_union_black",
            en,
            zh,
            en_txt,
            zh_txt,
            attacked,
            clean_imgs,
            atk_union,
            cln_union,
            g_atk,
            g_cln,
            "black",
            true,
            target,
            never_cln_acc,
        )
    )
    # Arm 2: gated union ensemble
    arms.append(
        eval_arm_both_halves(
            "gated_union_ensemble",
            en,
            zh,
            en_txt,
            zh_txt,
            attacked,
            clean_imgs,
            atk_union,
            cln_union,
            g_atk,
            g_cln,
            "ensemble",
            true,
            target,
            never_cln_acc,
        )
    )
    # Also ensemble on cc_bbox-only for ablation
    arms.append(
        eval_arm_both_halves(
            "gated_cc_bbox_ensemble",
            en,
            zh,
            en_txt,
            zh_txt,
            attacked,
            clean_imgs,
            atk_attn,
            cln_attn,
            g_atk,
            g_cln,
            "ensemble",
            true,
            target,
            never_cln_acc,
        )
    )

    best = max(arms, key=lambda a: a["mixed_2000"])
    # Arm 3: include raw if best still below bar
    if best["mixed_2000"] <= DP_BAR:
        print("\nBest so far below DP — running ensemble_raw (Arm 3)...")
        arms.append(
            eval_arm_both_halves(
                "gated_union_ensemble_raw",
                en,
                zh,
                en_txt,
                zh_txt,
                attacked,
                clean_imgs,
                atk_union,
                cln_union,
                g_atk,
                g_cln,
                "ensemble_raw",
                true,
                target,
                never_cln_acc,
            )
        )
        best = max(arms, key=lambda a: a["mixed_2000"])

    summary = {
        "status": args.status,
        "n": n,
        "dp_bar": DP_BAR,
        "gated_black_ref": GATED_BLACK_REF,
        "constraint": "occlusion_only_no_DP",
        "never": {
            "atk_acc": never_atk_acc,
            "clean_acc": never_cln_acc,
            "mixed_2000": mixed2000(never_atk_acc, never_cln_acc),
        },
        "detector": {
            "primary": gates["primary"],
            "threshold": gates["threshold"],
            "fire_attacked": gates["fire_attacked"],
            "fire_clean": gates["fire_clean"],
        },
        "arms": [{k: v for k, v in a.items() if k != "preds"} for a in arms],
        "best": {k: v for k, v in best.items() if k != "preds"},
        "beat_dp": bool(best["mixed_2000"] > DP_BAR),
        "ceiling_note": (
            f"Need EN atk >~77.4% at clean~85.9% for MIXED>81.65%. "
            f"Best atk={fmt(best['atk_acc'])} clean={fmt(best['clean_acc'])} "
            f"MIXED={fmt(best['mixed_2000'])}."
        ),
    }
    write_summary(RESULTS / f"summary_n{n}.json", summary)

    print("\n=== OCCLUSION-ONLY LEADERBOARD (EN MIXED2000) ===")
    print(f"  {'DP bar':40s} MIXED={fmt(DP_BAR)}")
    print(f"  {'gated black ref (prior)':40s} MIXED={fmt(GATED_BLACK_REF)}")
    for a in sorted(arms, key=lambda x: -x["mixed_2000"]):
        flag = "BEATS DP" if a["beats_dp"] else "below"
        print(
            f"  {a['name']:40s} MIXED={fmt(a['mixed_2000'])} "
            f"atk={fmt(a['atk_acc'])} clean={fmt(a['clean_acc'])} {flag}"
        )
    print(
        f"\nBest: {best['name']} MIXED={fmt(best['mixed_2000'])} — "
        f"{'BEATS' if summary['beat_dp'] else 'does NOT beat'} DP 81.65% (occlusion only)"
    )
    print(summary["ceiling_note"])
    print("Done.")


if __name__ == "__main__":
    main()
