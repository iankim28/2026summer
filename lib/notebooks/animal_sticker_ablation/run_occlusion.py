"""Animal-sticker black occlusion recovery.

Modes (same as run_ablation.py):
  all_sticker / mixed / all_text

Arms per mode:
  never              — undefended
  always_en_cap_zh   — EN∩ZH cc_bbox black, always-on
  always_en_only     — EN-only cc_bbox black, always-on
  gated_en_cap_zh    — EN∩ZH black gated by Phase-C detector
  gated_en_only      — EN-only black gated by Phase-C detector
  oracle_gt          — GT sticker union black (ceiling)

Gate is trained fresh on clean vs this mode's Attn-last features
(not typographic-cached gates). CUDA required.
Smoke: --n 100. Final: --n 1000.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
IMG_SAMPLES = HERE.parent / "image_samples"
EN_HELPERS = HERE.parent / "en_neglect_vs_blur"
AD = HERE.parent / "attack_detector"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(IMG_SAMPLES))
sys.path.insert(0, str(EN_HELPERS))
sys.path.insert(0, str(AD))
sys.path.insert(0, str(HERE))

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

from _common.protocol import (  # noqa: E402
    CLASSES,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    classify_batch,
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
from make_phase_ab_viz import build_feature_matrix, train_gate  # noqa: E402
from run_ablation import (  # noqa: E402
    DILATE,
    GALLERY_IDS,
    MODES,
    THR,
    TOP_K,
    _fonts,
    animal_targets,
    build_mode_attack,
    load_animal_pools,
    mask_from_single_cam,
)

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FILL = "black"
ARMS = (
    "never",
    "always_en_cap_zh",
    "always_en_only",
    "gated_en_cap_zh",
    "gated_en_only",
    "oracle_gt",
)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def compute_cams(en, zh, en_txt, zh_txt, images, label=""):
    cams_en, cams_zh = [], []
    t0 = time.time()
    n = len(images)
    for i, img in enumerate(images):
        _, c_en = classify_and_attn_en(en, en_txt, img, device=DEVICE)
        _, c_zh = classify_and_attn_zh(zh, zh_txt, img, device=DEVICE)
        cams_en.append(c_en)
        cams_zh.append(c_zh)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(
                f"  cams {label} {i+1}/{n}  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return cams_en, cams_zh


def build_masks_en_cap_zh(cams_en, cams_zh):
    return [
        build_cc_bbox_mask(ce, cz, threshold=THR, dilate=DILATE, top_k=TOP_K)
        for ce, cz in zip(cams_en, cams_zh)
    ]


def build_masks_en_only(cams_en):
    return [mask_from_single_cam(c) for c in cams_en]


def mean_coverage(masks) -> float:
    if not masks:
        return 0.0
    return float(np.mean([m.mean() for m in masks]))


def classify_always(model, text_emb, images, masks):
    defended = [apply_mask(img, m, fill=FILL) for img, m in zip(images, masks)]
    preds = classify_batch(model, defended, text_emb)
    return preds, mean_coverage(masks)


def classify_gated(model, text_emb, images, masks, gate_flags):
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    out = np.zeros(n, dtype=np.int64)
    def_idx = np.where(gate_flags)[0]
    pass_idx = np.where(~gate_flags)[0]
    if len(pass_idx):
        out[pass_idx] = classify_batch(
            model, [images[i] for i in pass_idx], text_emb
        )
    if len(def_idx):
        def_imgs = [images[i] for i in def_idx]
        def_masks = [masks[i] for i in def_idx]
        defended = [apply_mask(img, m, fill=FILL) for img, m in zip(def_imgs, def_masks)]
        out[def_idx] = classify_batch(model, defended, text_emb)
    return out, float(len(def_idx) / max(n, 1))


def train_mode_gate(cams_clean_en, cams_clean_zh, cams_atk_en, cams_atk_zh, n):
    clean_pack = {"cams_en": cams_clean_en, "cams_l": cams_clean_zh}
    atk_pack = {"cams_en": cams_atk_en, "cams_l": cams_atk_zh}
    X, y, img_ids, _names = build_feature_matrix(clean_pack, atk_pack, n)
    gate = train_gate(X, y, img_ids, n)
    return {
        "gate_clean": gate["all_pred"][:n].astype(bool),
        "gate_attacked": gate["all_pred"][n:].astype(bool),
        "threshold": float(gate["threshold"]),
        "primary": gate["primary_name"],
        "fire_clean": float(gate["fire_clean"]),
        "fire_attacked": float(gate["fire_attacked"]),
        "val_auc": float(gate["val_auc"]),
        "test_auc": float(gate["test_auc"]),
    }


def score_arm(preds_en, preds_zh, true, target, never_clean_en, never_clean_zh, **extra):
    en_acc, en_asr = acc_asr(preds_en, true, target)
    zh_acc, zh_asr = acc_asr(preds_zh, true, target)
    return {
        "en_acc": en_acc,
        "en_asr": en_asr,
        "zh_acc": zh_acc,
        "zh_asr": zh_asr,
        **extra,
    }


def score_clean_delta(cln_preds_en, cln_preds_zh, true, never_clean_en, never_clean_zh, target):
    # target unused for clean Δ but kept for signature symmetry with acc_asr callers
    cln_en_acc, _ = acc_asr(cln_preds_en, true, target)
    cln_zh_acc, _ = acc_asr(cln_preds_zh, true, target)
    return {
        "clean_en_acc": cln_en_acc,
        "clean_zh_acc": cln_zh_acc,
        "clean_delta_en": cln_en_acc - never_clean_en,
        "clean_delta_zh": cln_zh_acc - never_clean_zh,
    }


def save_gallery(mode_images, mode_masks_inter, mode_masks_en, modes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(len(GALLERY_IDS), len(mode_images[modes[0]]))
    ids = list(GALLERY_IDS[:n_show])
    # columns: per mode → raw | EN∩ZH black | EN-only black
    ncols = len(modes) * 3
    fig, axes = plt.subplots(len(ids), ncols, figsize=(3.2 * ncols, 3.2 * len(ids)))
    if len(ids) == 1:
        axes = np.array([axes])
    for r, idx in enumerate(ids):
        for c, mode in enumerate(modes):
            img = mode_images[mode][idx]
            m_inter = mode_masks_inter[mode][idx]
            m_en = mode_masks_en[mode][idx]
            black_inter = apply_mask(img, m_inter, fill=FILL)
            black_en = apply_mask(img, m_en, fill=FILL)
            for col, (pil, title) in enumerate(
                (
                    (img, f"{mode} raw#{idx}"),
                    (black_inter, f"{mode} EN∩ZH#{idx}"),
                    (black_en, f"{mode} EN-only#{idx}"),
                )
            ):
                ax = axes[r, 3 * c + col]
                ax.imshow(np.array(pil.convert("RGB")))
                ax.set_title(title, fontsize=8)
                ax.axis("off")
    fig.tight_layout()
    path = out_dir / "gallery_occlusion.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


def print_table(summary: dict):
    print("\n=== Animal-sticker occlusion ===")
    header = (
        f"{'mode':12} {'arm':18} "
        f"{'EN_acc':>7} {'EN_ASR':>7} {'ZH_acc':>7} {'ZH_ASR':>7} "
        f"{'dClnEN':>8} {'dClnZH':>8} {'cov':>6}"
    )
    print(header)
    for mode in MODES:
        block = summary["modes"][mode]
        for arm in ARMS:
            a = block["arms"][arm]
            d_en = a.get("clean_delta_en", float("nan"))
            d_zh = a.get("clean_delta_zh", float("nan"))
            cov = a.get("mean_mask_coverage", a.get("defended_frac", float("nan")))
            print(
                f"{mode:12} {arm:18} "
                f"{fmt_pct(a['en_acc']):>7} {fmt_pct(a['en_asr']):>7} "
                f"{fmt_pct(a['zh_acc']):>7} {fmt_pct(a['zh_asr']):>7} "
                f"{(f'{100*d_en:+.1f}pp' if d_en == d_en else 'n/a'):>8} "
                f"{(f'{100*d_zh:+.1f}pp' if d_zh == d_zh else 'n/a'):>8} "
                f"{(f'{100*cov:.1f}%' if cov == cov else 'n/a'):>6}"
            )


def run(n: int, save_gallery_flag: bool = True):
    fonts = _fonts()
    data = load_protocol_data(n=n)
    true = data["true"]
    clean_imgs = data["clean_224"]

    print("Loading CIFAR-10 train animal pools…", flush=True)
    pools = load_animal_pools()
    animal_tgt = animal_targets(true)

    print("Loading EN + ZH CLIP…", flush=True)
    en = EnCLIP()
    zh = ZhCLIP()
    en_txt = en.embed_texts(CLASSES["en"])
    zh_txt = zh.embed_texts(CLASSES["zh"])

    print("Clean never baseline…", flush=True)
    never_clean_en_preds = classify_batch(en, clean_imgs, en_txt)
    never_clean_zh_preds = classify_batch(zh, clean_imgs, zh_txt)
    # dummy target for clean-only acc
    dummy_tgt = data["target"]
    never_clean_en, _ = acc_asr(never_clean_en_preds, true, dummy_tgt)
    never_clean_zh, _ = acc_asr(never_clean_zh_preds, true, dummy_tgt)

    print("Clean Attn-last cams…", flush=True)
    cams_clean_en, cams_clean_zh = compute_cams(
        en, zh, en_txt, zh_txt, clean_imgs, label="clean"
    )
    masks_clean_inter = build_masks_en_cap_zh(cams_clean_en, cams_clean_zh)
    masks_clean_en = build_masks_en_only(cams_clean_en)

    summary = {
        "n": int(data["n"]),
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "fill": FILL,
        "partner": "zh",
        "modes_list": list(MODES),
        "arms_list": list(ARMS),
        "clean_never": {"en_acc": never_clean_en, "zh_acc": never_clean_zh},
        "modes": {},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "note": (
            "Black occlusion recovery on animal/mixed/text modes. "
            "Gate trained per-mode on clean vs mode-attacked Attn-last features."
        ),
    }

    mode_images = {}
    mode_masks_inter = {}
    mode_masks_en = {}

    for mode in MODES:
        print(f"\n--- mode={mode} ---", flush=True)
        imgs, rects, tgt = build_mode_attack(data, mode, fonts, pools, animal_tgt)
        mode_images[mode] = imgs
        gt_masks = [rects_to_mask(r) for r in rects]

        print(f"  attack cams…", flush=True)
        cams_atk_en, cams_atk_zh = compute_cams(
            en, zh, en_txt, zh_txt, imgs, label=mode
        )

        masks_inter = build_masks_en_cap_zh(cams_atk_en, cams_atk_zh)
        masks_en = build_masks_en_only(cams_atk_en)
        mode_masks_inter[mode] = masks_inter
        mode_masks_en[mode] = masks_en

        print(f"  train Phase-C gate on {mode}…", flush=True)
        gate = train_mode_gate(
            cams_clean_en, cams_clean_zh, cams_atk_en, cams_atk_zh, data["n"]
        )
        print(
            f"  gate primary={gate['primary']} thr={gate['threshold']:.4f} "
            f"fire_atk={gate['fire_attacked']:.3f} fire_clean={gate['fire_clean']:.3f} "
            f"val_auc={gate['val_auc']:.3f}",
            flush=True,
        )

        # never (undefended)
        never_en = classify_batch(en, imgs, en_txt)
        never_zh = classify_batch(zh, imgs, zh_txt)
        arms = {}
        arms["never"] = {
            **score_arm(never_en, never_zh, true, tgt, never_clean_en, never_clean_zh),
            **{
                "clean_en_acc": never_clean_en,
                "clean_zh_acc": never_clean_zh,
                "clean_delta_en": 0.0,
                "clean_delta_zh": 0.0,
                "mean_mask_coverage": 0.0,
            },
        }

        # always EN∩ZH
        atk_en, cov = classify_always(en, en_txt, imgs, masks_inter)
        atk_zh, _ = classify_always(zh, zh_txt, imgs, masks_inter)
        cln_en, _ = classify_always(en, en_txt, clean_imgs, masks_clean_inter)
        cln_zh, _ = classify_always(zh, zh_txt, clean_imgs, masks_clean_inter)
        arms["always_en_cap_zh"] = {
            **score_arm(atk_en, atk_zh, true, tgt, never_clean_en, never_clean_zh),
            **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, tgt),
            "mean_mask_coverage": cov,
            "mean_clean_mask_coverage": mean_coverage(masks_clean_inter),
        }

        # always EN-only
        atk_en, cov = classify_always(en, en_txt, imgs, masks_en)
        atk_zh, _ = classify_always(zh, zh_txt, imgs, masks_en)
        cln_en, _ = classify_always(en, en_txt, clean_imgs, masks_clean_en)
        cln_zh, _ = classify_always(zh, zh_txt, clean_imgs, masks_clean_en)
        arms["always_en_only"] = {
            **score_arm(atk_en, atk_zh, true, tgt, never_clean_en, never_clean_zh),
            **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, tgt),
            "mean_mask_coverage": cov,
            "mean_clean_mask_coverage": mean_coverage(masks_clean_en),
        }

        # gated EN∩ZH
        atk_en, def_frac = classify_gated(
            en, en_txt, imgs, masks_inter, gate["gate_attacked"]
        )
        atk_zh, _ = classify_gated(
            zh, zh_txt, imgs, masks_inter, gate["gate_attacked"]
        )
        cln_en, def_frac_cln = classify_gated(
            en, en_txt, clean_imgs, masks_clean_inter, gate["gate_clean"]
        )
        cln_zh, _ = classify_gated(
            zh, zh_txt, clean_imgs, masks_clean_inter, gate["gate_clean"]
        )
        arms["gated_en_cap_zh"] = {
            **score_arm(atk_en, atk_zh, true, tgt, never_clean_en, never_clean_zh),
            **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, tgt),
            "mean_mask_coverage": mean_coverage(masks_inter),
            "defended_frac_atk": def_frac,
            "defended_frac_clean": def_frac_cln,
            "fire_attacked": gate["fire_attacked"],
            "fire_clean": gate["fire_clean"],
        }

        # gated EN-only
        atk_en, def_frac = classify_gated(
            en, en_txt, imgs, masks_en, gate["gate_attacked"]
        )
        atk_zh, _ = classify_gated(
            zh, zh_txt, imgs, masks_en, gate["gate_attacked"]
        )
        cln_en, def_frac_cln = classify_gated(
            en, en_txt, clean_imgs, masks_clean_en, gate["gate_clean"]
        )
        cln_zh, _ = classify_gated(
            zh, zh_txt, clean_imgs, masks_clean_en, gate["gate_clean"]
        )
        arms["gated_en_only"] = {
            **score_arm(atk_en, atk_zh, true, tgt, never_clean_en, never_clean_zh),
            **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, tgt),
            "mean_mask_coverage": mean_coverage(masks_en),
            "defended_frac_atk": def_frac,
            "defended_frac_clean": def_frac_cln,
            "fire_attacked": gate["fire_attacked"],
            "fire_clean": gate["fire_clean"],
        }

        # oracle GT
        atk_en, cov = classify_always(en, en_txt, imgs, gt_masks)
        atk_zh, _ = classify_always(zh, zh_txt, imgs, gt_masks)
        # clean oracle: no stickers → empty masks (pass-through)
        empty = [np.zeros((224, 224), dtype=bool) for _ in range(data["n"])]
        cln_en, _ = classify_always(en, en_txt, clean_imgs, empty)
        cln_zh, _ = classify_always(zh, zh_txt, clean_imgs, empty)
        arms["oracle_gt"] = {
            **score_arm(atk_en, atk_zh, true, tgt, never_clean_en, never_clean_zh),
            **score_clean_delta(cln_en, cln_zh, true, never_clean_en, never_clean_zh, tgt),
            "mean_mask_coverage": cov,
            "mean_gt_coverage": cov,
        }

        summary["modes"][mode] = {
            "target_kind": "protocol" if mode == "all_text" else "animal",
            "mean_gt_coverage": float(np.mean([m.mean() for m in gt_masks])),
            "detector": {
                "primary": gate["primary"],
                "threshold": gate["threshold"],
                "fire_attacked": gate["fire_attacked"],
                "fire_clean": gate["fire_clean"],
                "val_auc": gate["val_auc"],
                "test_auc": gate["test_auc"],
            },
            "arms": arms,
        }

        for arm_name in ("never", "always_en_cap_zh", "always_en_only", "oracle_gt"):
            a = arms[arm_name]
            print(
                f"  {arm_name:18} EN {fmt_pct(a['en_acc'])}/{fmt_pct(a['en_asr'])} "
                f"ZH {fmt_pct(a['zh_acc'])}/{fmt_pct(a['zh_asr'])} "
                f"dClnEN={100*a['clean_delta_en']:+.1f}pp",
                flush=True,
            )

    if save_gallery_flag:
        save_gallery(mode_images, mode_masks_inter, mode_masks_en, MODES, FIGURES)

    print_table(summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"occlusion_n{data['n']}.json"
    write_summary(out_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="subset size (100 smoke / 1000 final)")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    run(n=args.n, save_gallery_flag=not args.no_gallery)


if __name__ == "__main__":
    main()
