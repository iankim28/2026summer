"""EN neglect vs Gaussian blur — chase MIXED2000 EN > 81.65% (Defense-Prefix bar).

Stages:
  oracle   — GT sticker boxes + blur/mean/black/neglect (atk ceiling)
  always   — EN∩ZH cc_bbox + fills, always-on (atk + clean → MIXED2000)
  gated    — Phase-C detector gates + fills
  escalate — patch_thr ablations, OCR+neglect, DP+neglect hybrid
  all      — run everything in order

CUDA required. Smoke: --n 16 --stage oracle. Final: --n 1000 --stage all.
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
PB = HERE.parent / "paper_baselines"
AD = HERE.parent / "attack_detector"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(HERE))

from _common.protocol import (  # noqa: E402
    CLASSES,
    DEVICE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    progress_log,
    write_summary,
)
from helpers import (  # noqa: E402
    BLUR_RADIUS,
    apply_mask,
    build_cc_bbox_mask,
    classify_and_attn_en,
    classify_and_attn_zh,
    classify_neglect_batch,
    rects_to_mask,
)

RESULTS = HERE / "results"
DP_BAR = 0.8165  # Defense-Prefix EN MIXED2000 always
CACHE_NPZ = AD / "results" / "zh" / "multi" / "cache" / "attn_en_zh_clean_multi.npz"
DP_TOKEN = PB / "defense_prefix" / "results" / "dp_cifar10_vit-b32.pt"


def mixed2000(atk, clean):
    return 0.5 * float(atk) + 0.5 * float(clean)


def fmt(x):
    return f"{100 * x:.2f}%"


def load_gate_flags(n_images=1000):
    """Recompute Phase-C ZH multi gates from attn cache (no CLIP)."""
    sys.path.insert(0, str(AD))
    from make_phase_ab_viz import (  # noqa: E402
        build_feature_matrix,
        load_cache,
        train_gate,
    )

    packed = load_cache("zh", AD / "results" / "zh" / "multi")
    X, y, img_ids, _names = build_feature_matrix(packed["clean"], packed["atk"], n_images)
    gate = train_gate(X, y, img_ids, n_images)
    return {
        "gate_clean": gate["all_pred"][:n_images].astype(bool),
        "gate_attacked": gate["all_pred"][n_images:].astype(bool),
        "threshold": gate["threshold"],
        "primary": gate["primary_name"],
        "fire_clean": gate["fire_clean"],
        "fire_attacked": gate["fire_attacked"],
    }


def load_cached_cams(n=1000):
    z = np.load(CACHE_NPZ, allow_pickle=False)
    return {
        "clean_en": z["clean_cams_en"][:n],
        "clean_l": z["clean_cams_l"][:n] if "clean_cams_l" in z.files else z["clean_cams_zh"][:n],
        "atk_en": z["atk_cams_en"][:n],
        "atk_l": z["atk_cams_l"][:n] if "atk_cams_l" in z.files else z["atk_cams_zh"][:n],
    }


def compute_cams_live(en, zh, en_txt, zh_txt, images, label=""):
    cams_en, cams_l = [], []
    t0 = time.time()
    for i, img in enumerate(images):
        _, c_en = classify_and_attn_en(en, en_txt, img, device=DEVICE)
        _, c_l = classify_and_attn_zh(zh, zh_txt, img, device=DEVICE)
        cams_en.append(c_en)
        cams_l.append(c_l)
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(
                f"  cams {label} {i+1}/{len(images)}  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return cams_en, cams_l


def build_masks_from_cams(cams_en, cams_l, threshold=0.95):
    return [
        build_cc_bbox_mask(ce, cl, threshold=threshold) for ce, cl in zip(cams_en, cams_l)
    ]


def classify_fill_arm(en, en_txt, images, masks, fill, patch_thr=0.5):
    """Classify with image fill or ViT neglect. Returns preds, extra_stats."""
    if fill == "neglect":
        t0 = time.time()
        preds, n_patches = classify_neglect_batch(
            en, en_txt, images, masks, patch_thr=patch_thr, device=DEVICE
        )
        print(
            f"      neglect classify n={len(images)} in {time.time()-t0:.0f}s "
            f"mean_patches={n_patches.mean():.2f}",
            flush=True,
        )
        return preds, {"mean_neglected_patches": float(n_patches.mean())}
    defended = [apply_mask(img, m, fill=fill) for img, m in zip(images, masks)]
    preds = classify_batch(en, defended, en_txt)
    cov = float(np.mean([m.mean() for m in masks])) if masks else 0.0
    return preds, {"mean_mask_coverage": cov}


def classify_gated(en, en_txt, images, masks, gate_flags, fill, patch_thr=0.5):
    """Pass-through when gate False; apply fill/neglect when True (batched)."""
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    out_preds = np.zeros(n, dtype=np.int64)
    def_idx = np.where(gate_flags)[0]
    pass_idx = np.where(~gate_flags)[0]
    if len(pass_idx):
        pass_imgs = [images[i] for i in pass_idx]
        out_preds[pass_idx] = classify_batch(en, pass_imgs, en_txt)
    if len(def_idx):
        def_imgs = [images[i] for i in def_idx]
        def_masks = [masks[i] for i in def_idx]
        if fill == "neglect":
            preds, _ = classify_neglect_batch(
                en, en_txt, def_imgs, def_masks, patch_thr=patch_thr, device=DEVICE
            )
        else:
            defended = [apply_mask(img, m, fill=fill) for img, m in zip(def_imgs, def_masks)]
            preds = classify_batch(en, defended, en_txt)
        out_preds[def_idx] = preds
    return out_preds, float(len(def_idx) / max(n, 1))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_oracle(en, en_txt, data, attacked, gt_rects, fills, patch_thr=0.5):
    true, target = data["true"], data["target"]
    n = data["n"]
    print(f"\n=== ORACLE (GT boxes) n={n} ===")
    never_preds = classify_batch(en, attacked, en_txt)
    never_acc, never_asr = acc_asr(never_preds, true, target)
    out = {
        "n": n,
        "never": {"atk_acc": never_acc, "asr": never_asr},
        "arms": {},
    }
    print(f"  never: atk={fmt(never_acc)} ASR={fmt(never_asr)}")
    masks = [rects_to_mask(r) for r in gt_rects]
    for fill in fills:
        t0 = time.time()
        preds, stats = classify_fill_arm(
            en, en_txt, attacked, masks, fill, patch_thr=patch_thr
        )
        acc, asr = acc_asr(preds, true, target)
        out["arms"][fill] = {
            "atk_acc": acc,
            "asr": asr,
            "elapsed_s": time.time() - t0,
            **stats,
        }
        print(f"  oracle_{fill}: atk={fmt(acc)} ASR={fmt(asr)}  {stats}")
    # Ceiling check for plan escalation
    neg = out["arms"].get("neglect", {}).get("atk_acc", 0)
    out["ceiling_note"] = (
        f"oracle neglect EN atk={fmt(neg)}; "
        f"{'BELOW' if neg < 0.774 else 'ABOVE'} ~77.4% needed for MIXED>81.65% at clean=85.9%"
    )
    print(" ", out["ceiling_note"])
    write_summary(RESULTS / f"oracle_n{n}.json", out)
    return out


def run_always(en, zh, en_txt, zh_txt, data, attacked, fills, patch_thr=0.5, threshold=0.95):
    true, target = data["true"], data["target"]
    n = data["n"]
    print(f"\n=== ALWAYS-ON cc_bbox fills n={n} thr={threshold} ===")

    if n == 1000 and CACHE_NPZ.exists():
        print("  Loading cached Attn-last cams...")
        cams = load_cached_cams(1000)
        atk_masks = build_masks_from_cams(cams["atk_en"], cams["atk_l"], threshold)
        clean_masks = build_masks_from_cams(cams["clean_en"], cams["clean_l"], threshold)
    else:
        print("  Computing Attn-last cams live...")
        atk_ce, atk_cl = compute_cams_live(en, zh, en_txt, zh_txt, attacked, "atk")
        cln_ce, cln_cl = compute_cams_live(en, zh, en_txt, zh_txt, data["clean_224"], "clean")
        atk_masks = build_masks_from_cams(atk_ce, atk_cl, threshold)
        clean_masks = build_masks_from_cams(cln_ce, cln_cl, threshold)

    never_atk = classify_batch(en, attacked, en_txt)
    never_cln = classify_batch(en, data["clean_224"], en_txt)
    never_atk_acc, never_asr = acc_asr(never_atk, true, target)
    never_cln_acc = float((never_cln == true).mean())

    out = {
        "n": n,
        "threshold": threshold,
        "patch_thr": patch_thr,
        "blur_radius": BLUR_RADIUS,
        "never": {
            "atk_acc": never_atk_acc,
            "asr": never_asr,
            "clean_acc": never_cln_acc,
            "mixed_2000": mixed2000(never_atk_acc, never_cln_acc),
        },
        "arms": {},
        "dp_bar": DP_BAR,
    }
    print(
        f"  never: atk={fmt(never_atk_acc)} clean={fmt(never_cln_acc)} "
        f"MIXED={fmt(out['never']['mixed_2000'])}"
    )

    for fill in fills:
        print(f"  arm={fill} ...")
        t0 = time.time()
        atk_preds, atk_stats = classify_fill_arm(
            en, en_txt, attacked, atk_masks, fill, patch_thr=patch_thr
        )
        cln_preds, cln_stats = classify_fill_arm(
            en, en_txt, data["clean_224"], clean_masks, fill, patch_thr=patch_thr
        )
        atk_acc, asr = acc_asr(atk_preds, true, target)
        cln_acc = float((cln_preds == true).mean())
        mix = mixed2000(atk_acc, cln_acc)
        clean_delta = cln_acc - never_cln_acc
        arm = {
            "atk_acc": atk_acc,
            "asr": asr,
            "clean_acc": cln_acc,
            "clean_delta": clean_delta,
            "mixed_2000": mix,
            "beats_dp": mix > DP_BAR,
            "elapsed_s": time.time() - t0,
            **{f"atk_{k}": v for k, v in atk_stats.items()},
            **{f"clean_{k}": v for k, v in cln_stats.items()},
        }
        out["arms"][fill] = arm
        print(
            f"    {fill}: atk={fmt(atk_acc)} clean={fmt(cln_acc)} "
            f"MIXED={fmt(mix)} dClean={100*clean_delta:+.2f}pp "
            f"{'BEATS DP' if arm['beats_dp'] else 'below DP'}  ({arm['elapsed_s']:.0f}s)"
        )

    write_summary(RESULTS / f"always_n{n}.json", out)
    return out, atk_masks, clean_masks


def run_gated(
    en,
    en_txt,
    data,
    attacked,
    atk_masks,
    clean_masks,
    fills,
    patch_thr=0.5,
):
    true, target = data["true"], data["target"]
    n = data["n"]
    print(f"\n=== GATED fills n={n} ===")
    if n != 1000:
        print("  WARN: gates trained on full 1000; subset uses first-n of gate vectors")
    gates = load_gate_flags(1000)
    g_atk = gates["gate_attacked"][:n]
    g_cln = gates["gate_clean"][:n]
    print(
        f"  detector={gates['primary']} thr={gates['threshold']:.4f} "
        f"fire_atk={gates['fire_attacked']:.3f} fire_clean={gates['fire_clean']:.3f}"
    )

    never_cln = classify_batch(en, data["clean_224"], en_txt)
    never_cln_acc = float((never_cln == true).mean())

    out = {
        "n": n,
        "detector": {
            "primary": gates["primary"],
            "threshold": gates["threshold"],
            "fire_attacked": gates["fire_attacked"],
            "fire_clean": gates["fire_clean"],
        },
        "arms": {},
        "dp_bar": DP_BAR,
    }
    for fill in fills:
        print(f"  gated arm={fill} ...")
        atk_preds, def_frac_atk = classify_gated(
            en, en_txt, attacked, atk_masks, g_atk, fill, patch_thr=patch_thr
        )
        cln_preds, def_frac_cln = classify_gated(
            en, en_txt, data["clean_224"], clean_masks, g_cln, fill, patch_thr=patch_thr
        )
        atk_acc, asr = acc_asr(atk_preds, true, target)
        cln_acc = float((cln_preds == true).mean())
        mix = mixed2000(atk_acc, cln_acc)
        arm = {
            "atk_acc": atk_acc,
            "asr": asr,
            "clean_acc": cln_acc,
            "clean_delta": cln_acc - never_cln_acc,
            "mixed_2000": mix,
            "beats_dp": mix > DP_BAR,
            "defend_frac_attacked": def_frac_atk,
            "defend_frac_clean": def_frac_cln,
        }
        out["arms"][fill] = arm
        print(
            f"    gated_{fill}: atk={fmt(atk_acc)} clean={fmt(cln_acc)} "
            f"MIXED={fmt(mix)} {'BEATS DP' if arm['beats_dp'] else 'below DP'}"
        )
    write_summary(RESULTS / f"gated_n{n}.json", out)
    return out


def run_escalate(en, zh, en_txt, zh_txt, data, attacked, gt_rects, atk_masks, clean_masks):
    """patch_thr ablations → OCR+neglect → DP+neglect."""
    true, target = data["true"], data["target"]
    n = data["n"]
    print(f"\n=== ESCALATE n={n} ===")
    out = {"n": n, "dp_bar": DP_BAR, "patch_thr": {}, "ocr_neglect": {}, "dp_neglect": {}}

    # 1) patch_thr on always-on neglect
    for pt in (0.25, 0.5, 0.75):
        print(f"  patch_thr={pt} neglect always...")
        atk_preds, _ = classify_fill_arm(en, en_txt, attacked, atk_masks, "neglect", patch_thr=pt)
        cln_preds, _ = classify_fill_arm(
            en, en_txt, data["clean_224"], clean_masks, "neglect", patch_thr=pt
        )
        atk_acc, asr = acc_asr(atk_preds, true, target)
        cln_acc = float((cln_preds == true).mean())
        mix = mixed2000(atk_acc, cln_acc)
        out["patch_thr"][str(pt)] = {
            "atk_acc": atk_acc,
            "asr": asr,
            "clean_acc": cln_acc,
            "mixed_2000": mix,
            "beats_dp": mix > DP_BAR,
        }
        print(f"    thr={pt}: atk={fmt(atk_acc)} clean={fmt(cln_acc)} MIXED={fmt(mix)}")

    # 2) OCR boxes + neglect
    try:
        import easyocr

        print("  OCR+neglect...")
        reader = easyocr.Reader(["en", "ch_sim"], gpu=True)
        ocr_masks_atk = []
        for i, img in enumerate(attacked):
            det = reader.readtext(np.array(img))
            boxes = []
            for item in det:
                box = item[0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            ocr_masks_atk.append(rects_to_mask(boxes) if boxes else np.zeros((224, 224), bool))
            if (i + 1) % 50 == 0:
                print(f"    OCR atk {i+1}/{n}", flush=True)
        ocr_masks_cln = []
        for i, img in enumerate(data["clean_224"]):
            det = reader.readtext(np.array(img))
            boxes = []
            for item in det:
                box = item[0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            ocr_masks_cln.append(rects_to_mask(boxes) if boxes else np.zeros((224, 224), bool))
            if (i + 1) % 50 == 0:
                print(f"    OCR clean {i+1}/{n}", flush=True)
        atk_preds, _ = classify_fill_arm(
            en, en_txt, attacked, ocr_masks_atk, "neglect", patch_thr=0.5
        )
        cln_preds, _ = classify_fill_arm(
            en, en_txt, data["clean_224"], ocr_masks_cln, "neglect", patch_thr=0.5
        )
        atk_acc, asr = acc_asr(atk_preds, true, target)
        cln_acc = float((cln_preds == true).mean())
        mix = mixed2000(atk_acc, cln_acc)
        out["ocr_neglect"] = {
            "atk_acc": atk_acc,
            "asr": asr,
            "clean_acc": cln_acc,
            "mixed_2000": mix,
            "beats_dp": mix > DP_BAR,
        }
        print(f"    OCR+neglect: atk={fmt(atk_acc)} clean={fmt(cln_acc)} MIXED={fmt(mix)}")
    except Exception as e:
        out["ocr_neglect"] = {"error": str(e)}
        print(f"  OCR+neglect failed: {e}")

    # 3) DP + neglect hybrid (spatial neglect then DP text prompts)
    if DP_TOKEN.exists():
        print("  DP+neglect hybrid...")
        try:
            import clip
            import torch.nn.functional as F

            vendor = PB / "_vendor" / "Defense-Prefix"
            sys.path.insert(0, str(vendor))
            from utils.non_nv import encode_text_with_learnt_tokens

            model, preprocess = clip.load("ViT-B/32", device=DEVICE)
            model.eval()
            func_type = type(model.encode_text)
            model.encode_text_with_learnt_tokens = func_type(
                encode_text_with_learnt_tokens, model
            )
            prefix = torch.load(DP_TOKEN, map_location=DEVICE, weights_only=False)
            if isinstance(prefix, torch.nn.Parameter):
                prefix = prefix.data
            prefix = prefix.to(DEVICE).to(model.dtype)
            text_prefix = torch.cat(
                [clip.tokenize(f"a photo of a * {c}.") for c in CLASSES["en"]]
            ).to(DEVICE)
            asterix = clip.tokenize(["*"]).to(DEVICE)[0][1]
            dp_txt = model.encode_text_with_learnt_tokens(
                text_prefix, asterix, prefix.unsqueeze(0), is_emb=False
            )
            dp_txt = F.normalize(dp_txt.float(), dim=-1)

            # Image-space black neglect (compatible with openai clip preprocess),
            # then DP classify — pairs DP text with hard spatial remove.
            # Also try ViT-neglect via open_clip en + DP is awkward across libs;
            # use black fill as spatial arm for hybrid (true neglect of pixels).
            def _dp_classify(imgs):
                preds = []
                for i in range(0, len(imgs), 64):
                    batch = imgs[i : i + 64]
                    x = torch.stack([preprocess(im) for im in batch]).to(DEVICE)
                    imf = F.normalize(model.encode_image(x).float(), dim=-1)
                    preds.append((imf @ dp_txt.t()).argmax(-1).cpu().numpy())
                return np.concatenate(preds)

            for spatial in ("black", "blur", "neglect_openclip"):
                if spatial == "neglect_openclip":
                    # open_clip neglect features vs DP text (same ViT-B/32 space? NO —
                    # openai clip vs open_clip openai weights should match, but text
                    # towers differ in tokenization path). Safer: black/blur + DP text.
                    continue
                atk_imgs = [
                    apply_mask(attacked[i], atk_masks[i], fill=spatial) for i in range(n)
                ]
                cln_imgs = [
                    apply_mask(data["clean_224"][i], clean_masks[i], fill=spatial)
                    for i in range(n)
                ]
                atk_preds = _dp_classify(atk_imgs)
                cln_preds = _dp_classify(cln_imgs)
                atk_acc, asr = acc_asr(atk_preds, true, target)
                cln_acc = float((cln_preds == true).mean())
                mix = mixed2000(atk_acc, cln_acc)
                out["dp_neglect"][spatial] = {
                    "atk_acc": atk_acc,
                    "asr": asr,
                    "clean_acc": cln_acc,
                    "mixed_2000": mix,
                    "beats_dp": mix > DP_BAR,
                }
                print(
                    f"    DP+{spatial}: atk={fmt(atk_acc)} clean={fmt(cln_acc)} "
                    f"MIXED={fmt(mix)} {'BEATS DP' if mix > DP_BAR else 'below DP'}"
                )

            # Also DP-only on raw images (sanity / hybrid upper with DP always)
            atk_preds = _dp_classify(attacked)
            cln_preds = _dp_classify(data["clean_224"])
            atk_acc, asr = acc_asr(atk_preds, true, target)
            cln_acc = float((cln_preds == true).mean())
            mix = mixed2000(atk_acc, cln_acc)
            out["dp_neglect"]["dp_only"] = {
                "atk_acc": atk_acc,
                "asr": asr,
                "clean_acc": cln_acc,
                "mixed_2000": mix,
                "beats_dp": mix > DP_BAR,
            }
            print(f"    DP-only: atk={fmt(atk_acc)} clean={fmt(cln_acc)} MIXED={fmt(mix)}")

            # True hybrid: open_clip ViT neglect image feats + vanilla EN text is already
            # covered. For DP text + neglect patches: encode neglected images with openai
            # clip by zeroing pixel patches then DP classify.
            from helpers import mask_to_patch_flags

            def _zero_patches_pil(pil_img, mask, patch_thr=0.5):
                arr = np.array(pil_img.convert("RGB"))
                flags, _ = mask_to_patch_flags(mask, patch_thr=patch_thr)
                for p in np.where(flags)[0]:
                    r, c = divmod(int(p), 7)
                    arr[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = 0
                return Image.fromarray(arr)

            from PIL import Image

            atk_imgs = [
                _zero_patches_pil(attacked[i], atk_masks[i]) for i in range(n)
            ]
            cln_imgs = [
                _zero_patches_pil(data["clean_224"][i], clean_masks[i]) for i in range(n)
            ]
            atk_preds = _dp_classify(atk_imgs)
            cln_preds = _dp_classify(cln_imgs)
            atk_acc, asr = acc_asr(atk_preds, true, target)
            cln_acc = float((cln_preds == true).mean())
            mix = mixed2000(atk_acc, cln_acc)
            out["dp_neglect"]["patch_zero_pixels"] = {
                "atk_acc": atk_acc,
                "asr": asr,
                "clean_acc": cln_acc,
                "mixed_2000": mix,
                "beats_dp": mix > DP_BAR,
            }
            print(
                f"    DP+patch_zero: atk={fmt(atk_acc)} clean={fmt(cln_acc)} "
                f"MIXED={fmt(mix)} {'BEATS DP' if mix > DP_BAR else 'below DP'}"
            )
        except Exception as e:
            out["dp_neglect"]["error"] = str(e)
            print(f"  DP+neglect failed: {e}")
    else:
        out["dp_neglect"] = {"error": f"missing token {DP_TOKEN}"}
        print(f"  skip DP hybrid — missing {DP_TOKEN}")

    # Best across escalation
    candidates = []
    for k, v in out["patch_thr"].items():
        candidates.append((f"patch_thr={k}", v.get("mixed_2000", 0), v))
    if "mixed_2000" in out["ocr_neglect"]:
        candidates.append(("ocr_neglect", out["ocr_neglect"]["mixed_2000"], out["ocr_neglect"]))
    for k, v in out["dp_neglect"].items():
        if isinstance(v, dict) and "mixed_2000" in v:
            candidates.append((f"dp+{k}", v["mixed_2000"], v))
    if candidates:
        best_name, best_mix, best = max(candidates, key=lambda t: t[1])
        out["best"] = {"name": best_name, "mixed_2000": best_mix, **best}
        print(f"  BEST escalate: {best_name} MIXED={fmt(best_mix)}")
    write_summary(RESULTS / f"escalate_n{n}.json", out)
    return out


def summarize_leaderboard(oracle, always, gated, escalate):
    rows = []
    rows.append(
        {
            "method": "Defense-Prefix (bar)",
            "policy": "always",
            "mixed_2000": DP_BAR,
            "atk": 0.738,
            "clean": 0.895,
        }
    )
    if always:
        for fill, arm in always.get("arms", {}).items():
            rows.append(
                {
                    "method": f"cc_bbox_{fill}",
                    "policy": "always",
                    "mixed_2000": arm["mixed_2000"],
                    "atk": arm["atk_acc"],
                    "clean": arm["clean_acc"],
                    "beats_dp": arm["beats_dp"],
                }
            )
    if gated:
        for fill, arm in gated.get("arms", {}).items():
            rows.append(
                {
                    "method": f"cc_bbox_{fill}",
                    "policy": "gated",
                    "mixed_2000": arm["mixed_2000"],
                    "atk": arm["atk_acc"],
                    "clean": arm["clean_acc"],
                    "beats_dp": arm["beats_dp"],
                }
            )
    if escalate and escalate.get("best"):
        b = escalate["best"]
        rows.append(
            {
                "method": b["name"],
                "policy": "escalate",
                "mixed_2000": b["mixed_2000"],
                "atk": b.get("atk_acc"),
                "clean": b.get("clean_acc"),
                "beats_dp": b.get("beats_dp"),
            }
        )
    best = max(rows[1:], key=lambda r: r["mixed_2000"] or 0) if len(rows) > 1 else rows[0]
    summary = {
        "dp_bar": DP_BAR,
        "rows": rows,
        "best_ours": best,
        "beat_dp": bool(best.get("beats_dp") or (best.get("mixed_2000") or 0) > DP_BAR),
        "oracle_neglect_atk": (oracle or {}).get("arms", {}).get("neglect", {}).get("atk_acc"),
    }
    write_summary(RESULTS / "leaderboard.json", summary)
    print("\n=== LEADERBOARD (EN MIXED2000) ===")
    for r in sorted(rows, key=lambda x: -(x["mixed_2000"] or 0)):
        print(
            f"  {r['policy']:8s} {r['method']:28s} "
            f"MIXED={fmt(r['mixed_2000'] or 0)} "
            f"atk={fmt(r['atk'] or 0)} clean={fmt(r['clean'] or 0)}"
        )
    print(
        f"\nBest ours: {best['method']} @ {fmt(best.get('mixed_2000') or 0)} "
        f"— {'BEATS' if summary['beat_dp'] else 'does NOT beat'} DP 81.65%"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument(
        "--stage",
        choices=["oracle", "always", "gated", "escalate", "all"],
        default="all",
    )
    ap.add_argument("--status", default="sanity")
    ap.add_argument("--patch-thr", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument(
        "--fills",
        default="blur,mean,black,neglect",
        help="comma-separated fill arms",
    )
    args = ap.parse_args()
    fills = [f.strip() for f in args.fills.split(",") if f.strip()]

    assert torch.cuda.is_available(), "CUDA required"
    print("Device:", DEVICE, torch.cuda.get_device_name(0))
    RESULTS.mkdir(parents=True, exist_ok=True)

    n = args.n if args.n < 1000 else None
    data = load_protocol_data(n=n)
    attacked, gt_rects = build_multi_attack(data)

    print("Loading EN/ZH CLIP...")
    en, zh = EnCLIP(), ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    meta = {
        "status": args.status,
        "n": data["n"],
        "stage": args.stage,
        "fills": fills,
        "patch_thr": args.patch_thr,
        "threshold": args.threshold,
        "device": str(torch.cuda.get_device_name(0)),
    }
    write_summary(RESULTS / f"run_meta_n{data['n']}.json", meta)

    oracle = always = gated = escalate = None
    atk_masks = clean_masks = None

    if args.stage in ("oracle", "all"):
        oracle = run_oracle(
            en, en_txt, data, attacked, gt_rects, fills, patch_thr=args.patch_thr
        )

    if args.stage in ("always", "gated", "escalate", "all"):
        always, atk_masks, clean_masks = run_always(
            en,
            zh,
            en_txt,
            zh_txt,
            data,
            attacked,
            fills,
            patch_thr=args.patch_thr,
            threshold=args.threshold,
        )

    if args.stage in ("gated", "all"):
        assert atk_masks is not None
        gated = run_gated(
            en,
            en_txt,
            data,
            attacked,
            atk_masks,
            clean_masks,
            fills,
            patch_thr=args.patch_thr,
        )

    if args.stage in ("escalate", "all"):
        assert atk_masks is not None
        escalate = run_escalate(
            en, zh, en_txt, zh_txt, data, attacked, gt_rects, atk_masks, clean_masks
        )

    summarize_leaderboard(oracle, always, gated, escalate)
    print("\nDone.")


if __name__ == "__main__":
    main()
