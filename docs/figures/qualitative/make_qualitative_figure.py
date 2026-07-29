"""Make Qualitative results figures for the paper.

Outputs (this directory):
  qualitative_figure.png       — success-only 2x5 recoveries (ZH/KO/JA round-robin)
  qualitative_failures.png     — failure-only 2x3 (one residual fail per partner)
  picks.json                   — chosen indices / preds for reproducibility

Gated cc_bbox_black on dual-box E+L attacks. CUDA required for full scan.

Notes:
  - Partial sticker occlusion (e.g. old JA frog example) is NOT intentional:
    Attn-last EN∩L cc_bbox can miss a box while class still recovers. Success
    picks now require both stickers to be well covered by the mask.
  - Percentages under panels are CLIP top-1 softmax classification probabilities.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PF = REPO / "lib" / "notebooks" / "partner_fill_ablation"
PB = REPO / "lib" / "notebooks" / "paper_baselines"
AD = REPO / "lib" / "notebooks" / "attack_detector"

sys.path.insert(0, str(PB))
sys.path.insert(0, str(PF))

from _common.protocol import DEVICE, EnCLIP, load_protocol_data  # noqa: E402
from helpers import apply_mask, build_cc_bbox_mask  # noqa: E402
from run_eval import (  # noqa: E402
    CLASSES,
    MODEL_CLS,
    build_partner_multi_attack,
    load_cached_cams,
    load_gate_flags,
)

UPSCALE = 3
DISPLAY_SIZE = 224
THRESHOLD = 0.95
BATCH = 64
PAD = 8
BH_EXTRA = 12
FONT_SIZE = 24
PARTNERS = ["zh", "ko", "ja"]
LANG_TAG = {"zh": "ZH", "ko": "KO", "ja": "JA"}
MIN_STICKER_COVER = 0.70  # both stickers must be mostly blacked for success picks
# Round-robin partners so all three appear; length = number of recovery columns.
SUCCESS_PARTNER_ORDER = ["zh", "ko", "ja", "zh", "ko"]
N_SUCCESS = len(SUCCESS_PARTNER_ORDER)

COLOR_OK = "#1B7A3D"
COLOR_BAD = "#B00020"
COLOR_MUTED = "#555555"


def require_cuda():
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is required for qualitative CLIP scoring, but "
            f"torch.cuda.is_available() is False (torch={torch.__version__}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Device: cuda ({torch.__version__})")


def dog_ship_index(true, target):
    prefer_t = CLASSES["en"].index("dog")
    prefer_g = CLASSES["en"].index("ship")
    for i in range(len(true)):
        if int(true[i]) == prefer_t and int(target[i]) == prefer_g:
            return int(i)
    return None


def skip_success_indices(true, target):
    """Skip intro dog<-ship and any other dog-true recoveries (avoid dog in this figure)."""
    dog = CLASSES["en"].index("dog")
    skip = set()
    ds = dog_ship_index(true, target)
    if ds is not None:
        skip.add(ds)
    for i in range(len(true)):
        if int(true[i]) == dog:
            skip.add(int(i))
    return skip


def _font_paths():
    if platform.system() == "Windows":
        wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        cjk = os.path.join(wf, "msyh.ttc")
        lat = os.path.join(wf, "arial.ttf")
        ko = os.path.join(wf, "malgun.ttf")
        if not os.path.isfile(ko):
            ko = cjk
        return cjk, lat, ko
    return (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )


_CJK, _LAT, _KO = _font_paths()
_FONT_CACHE = {}


def _get_font(fp, size=FONT_SIZE):
    key = (fp, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _font_for_lang(lang):
    if lang == "en":
        return _get_font(_LAT)
    if lang == "ko":
        return _get_font(_KO)
    return _get_font(_CJK)


def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y


def measure_sticker_boxes(en_w, l_w, L, xy_en, xy_l):
    """GT white-box rects matching production dual-box draw."""
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    boxes = []
    for word, lang, xy in [(en_w, "en", xy_en), (l_w, L, xy_l)]:
        font = _font_for_lang(lang)
        bb = tmp.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + BH_EXTRA
        rx, ry = _clamp_xy(xy, bw, bh)
        boxes.append((rx, ry, rx + bw, ry + bh))
    return boxes


def box_cover_frac(mask, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(DISPLAY_SIZE, x1), min(DISPLAY_SIZE, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = mask[y0:y1, x0:x1]
    return float(region.mean()) if region.size else 0.0


def sticker_cover_score(mask, boxes):
    covers = [box_cover_frac(mask, b) for b in boxes]
    return float(min(covers)), covers


def visual_black_cover(ours_pil, boxes, dark_thr=40):
    """Fraction of each sticker rect that is near-black on the defended image."""
    arr = np.asarray(ours_pil.convert("RGB"))
    covers = []
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(DISPLAY_SIZE, x1), min(DISPLAY_SIZE, y1)
        if x1 <= x0 or y1 <= y0:
            covers.append(0.0)
            continue
        region = arr[y0:y1, x0:x1]
        covers.append(float((region.mean(axis=2) < dark_thr).mean()))
    return float(min(covers)), covers


@torch.no_grad()
def classify_with_probs(model, imgs, text_emb, batch_size=BATCH):
    preds, confs = [], []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i : i + batch_size])
        logits = 100.0 * (imf @ text_emb.t())
        prob = F.softmax(logits.float(), dim=-1)
        conf, pred = prob.max(dim=-1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


@torch.no_grad()
def classify_batch_preds(model, imgs, text_emb, batch_size=BATCH):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i : i + batch_size])
        preds.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def load_cache_atk_preds(L, n=1000):
    path = AD / "results" / L / "multi" / "cache" / f"attn_en_{L}_clean_multi.npz"
    z = np.load(path, allow_pickle=False)
    en = z["atk_preds_en"][:n].astype(np.int64)
    for key in (f"atk_preds_{L}", "atk_preds_l", "atk_preds_zh"):
        if key in z.files:
            return en, z[key][:n].astype(np.int64)
    raise KeyError(f"no L atk preds in {path}; files={list(z.files)}")


def build_gated_images(attacked, masks, gate_flags):
    out = []
    for img, m, g in zip(attacked, masks, gate_flags):
        out.append(apply_mask(img, m, fill="black") if g else img)
    return out


def scan_partner(L, data, en, en_txt, skip_idx):
    print(f"\n=== Scanning partner {L.upper()} ===")
    partner = MODEL_CLS[L]()
    l_txt = partner.embed_texts(CLASSES[L]).detach()

    attacked = build_partner_multi_attack(data, L)
    cams = load_cached_cams(L, 1000)
    masks = [
        build_cc_bbox_mask(cams["atk_en"][i], cams["atk_l"][i], threshold=THRESHOLD)
        for i in range(data["n"])
    ]
    gates = load_gate_flags(L, 1000)
    g_atk = gates["gate_attacked"][: data["n"]].astype(bool)

    atk_en, atk_l = load_cache_atk_preds(L, data["n"])
    gated_imgs = build_gated_images(attacked, masks, g_atk)

    print("  classifying gated images (EN + L)...")
    ours_en = classify_batch_preds(en, gated_imgs, en_txt)
    ours_l = classify_batch_preds(partner, gated_imgs, l_txt)

    true = data["true"]
    target = data["target"]
    skip_set = skip_idx if isinstance(skip_idx, (set, frozenset)) else (
        {skip_idx} if skip_idx is not None else set()
    )

    success, fail_applied, fail_miss = [], [], []
    for i in range(data["n"]):
        if i in skip_set:
            continue
        t, g = int(true[i]), int(target[i])
        en_w, l_w = CLASSES["en"][g], CLASSES[L][g]
        boxes = measure_sticker_boxes(
            en_w, l_w, L, data["attack_pos"]["en"][i], data["attack_pos"]["l"][i],
        )
        mask_cover, _ = sticker_cover_score(masks[i], boxes)
        vis_cover, covers = visual_black_cover(gated_imgs[i], boxes)
        # Prefer visual blackness of sticker rects (matches what readers see)
        cover_min = vis_cover

        fooled_en = int(atk_en[i]) != t
        fooled_l = int(atk_l[i]) != t
        fooled = fooled_en  # EN fooled (primary)
        recovered = int(ours_en[i]) == t
        both = recovered and int(ours_l[i]) == t
        both_fooled = fooled_en and fooled_l
        rec = {
            "i": int(i),
            "both": bool(both),
            "both_fooled": bool(both_fooled),
            "true": t,
            "target": g,
            "atk_en": int(atk_en[i]),
            "atk_l": int(atk_l[i]),
            "ours_en": int(ours_en[i]),
            "ours_l": int(ours_l[i]),
            "gate": bool(g_atk[i]),
            "cover_min": cover_min,
            "mask_cover": float(mask_cover),
            "covers": covers,
            "fooled": bool(fooled),
        }
        if fooled_en and recovered and bool(g_atk[i]):
            success.append(rec)
        if bool(g_atk[i]) and not recovered:
            fail_applied.append(rec)
        if (not bool(g_atk[i])) and fooled:
            fail_miss.append(rec)

    n_cov = sum(1 for s in success if s["cover_min"] >= MIN_STICKER_COVER)
    print(
        f"  success={len(success)} (both={sum(1 for s in success if s['both'])}, "
        f"full_cover>={MIN_STICKER_COVER}: {n_cov}) "
        f"fail_applied={len(fail_applied)} fail_miss={len(fail_miss)}"
    )

    del partner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "L": L,
        "attacked": attacked,
        "gated": gated_imgs,
        "masks": masks,
        "gate": g_atk,
        "success": success,
        "fail_applied": fail_applied,
        "fail_miss": fail_miss,
    }


def pick_success(scan, used_true, used_target, used_i):
    """Prefer both-fooled + both-recovered + both stickers visually blacked."""
    pool = sorted(
        scan["success"],
        key=lambda s: (
            not s.get("both_fooled", False),
            not s["both"],
            -(s["cover_min"]),
            s["i"],
        ),
    )
    for require_fooled_both in (True, False):
        for require_cover in (True, False):
            for uniq_both in (True, False):
                for s in pool:
                    if s["i"] in used_i:
                        continue
                    if require_fooled_both and not s.get("both_fooled", False):
                        continue
                    if require_cover and s["cover_min"] < MIN_STICKER_COVER:
                        continue
                    if uniq_both and (s["true"] in used_true or s["target"] in used_target):
                        continue
                    if (not uniq_both) and s["true"] in used_true:
                        continue
                    return s
    for s in pool:
        if s["i"] not in used_i:
            return s
    return None


def pick_failures(scans, used_i, n_per_partner=1):
    """One gate-on residual fail per partner; diversify true classes."""
    picks = []
    used_true = set()
    for L in PARTNERS:
        pool = [
            s for s in scans[L]["fail_applied"]
            if s["i"] not in used_i and s.get("fooled", True)
        ]
        # Prefer unused true class, then higher sticker black-cover (occlusion applied)
        pool.sort(key=lambda s: (s["true"] in used_true, -s.get("cover_min", 0), s["i"]))
        if not pool:
            pool = [s for s in scans[L]["fail_applied"] if s["i"] not in used_i]
            pool.sort(key=lambda s: s["i"])
        if not pool:
            raise RuntimeError(f"not enough failures for {L}")
        s = pool[0]
        picks.append((L, s))
        used_i.add(s["i"])
        used_true.add(s["true"])
    return picks


def upscale(pil):
    w = DISPLAY_SIZE * UPSCALE
    return pil.resize((w, w), Image.NEAREST)


def _pred_color(pred, true):
    return COLOR_OK if int(pred) == int(true) else COLOR_BAD


def score_pair(en, en_txt, partner, l_txt, atk_img, ours_img):
    (ae, ac), (oe, oc) = (
        classify_with_probs(en, [atk_img], en_txt),
        classify_with_probs(en, [ours_img], en_txt),
    )
    (al, alc), (ol, olc) = (
        classify_with_probs(partner, [atk_img], l_txt),
        classify_with_probs(partner, [ours_img], l_txt),
    )
    return {
        "atk": {"en": int(ae[0]), "en_p": float(ac[0]), "l": int(al[0]), "l_p": float(alc[0])},
        "ours": {"en": int(oe[0]), "en_p": float(oc[0]), "l": int(ol[0]), "l_p": float(olc[0])},
    }


def draw_cell(ax, pil, en_pred, en_p, l_tag, l_pred, l_p, true):
    ax.imshow(upscale(pil))
    ax.axis("off")
    en_name = CLASSES["en"][int(en_pred)]
    l_name = CLASSES["en"][int(l_pred)]
    ax.text(
        0.5, -0.05,
        f"EN  {en_name}  {100.0 * en_p:.0f}%",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, fontweight="bold", color=_pred_color(en_pred, true),
        clip_on=False,
    )
    ax.text(
        0.5, -0.18,
        f"{l_tag}  {l_name}  {100.0 * l_p:.0f}%",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9.5, color=_pred_color(l_pred, true),
        clip_on=False,
    )


def save_grid(path, columns, title, footnote):
    """2 x N Attacked/Ours grid with breathing room."""
    n = len(columns)
    fig = plt.figure(figsize=(min(3.2 * n + 1.0, 18.5), 7.8))
    gs = fig.add_gridspec(
        2, n,
        left=0.06, right=0.99,
        top=0.86, bottom=0.12,
        wspace=0.22 if n <= 3 else 0.16,
        hspace=0.50,
    )
    axes = np.empty((2, n), dtype=object)
    for r in range(2):
        for c in range(n):
            axes[r, c] = fig.add_subplot(gs[r, c])

    axes[0, 0].annotate(
        "Attacked", xy=(-0.28, 0.5), xycoords="axes fraction",
        fontsize=13, fontweight="bold", rotation=90, va="center", ha="center",
        color=COLOR_MUTED, clip_on=False,
    )
    axes[1, 0].annotate(
        "Ours", xy=(-0.28, 0.5), xycoords="axes fraction",
        fontsize=13, fontweight="bold", rotation=90, va="center", ha="center",
        color=COLOR_MUTED, clip_on=False,
    )

    for c, col in enumerate(columns):
        tag = LANG_TAG[col["L"]]
        true_name = CLASSES["en"][col["true"]]
        tgt_name = CLASSES["en"][col["target"]]
        header = f"{true_name} → {tgt_name}  ·  EN ∩ {tag}"
        hdr_fs = 10 if n <= 3 else 8.5
        axes[0, c].annotate(
            header, xy=(0.5, 1.12), xycoords="axes fraction",
            fontsize=hdr_fs, fontweight="bold", ha="center", va="bottom",
            color="#1a1a1a", clip_on=False,
        )
        s = col["scores"]
        draw_cell(
            axes[0, c], col["atk_img"],
            s["atk"]["en"], s["atk"]["en_p"], tag, s["atk"]["l"], s["atk"]["l_p"],
            col["true"],
        )
        draw_cell(
            axes[1, c], col["ours_img"],
            s["ours"]["en"], s["ours"]["en_p"], tag, s["ours"]["l"], s["ours"]["l_p"],
            col["true"],
        )

    fig.text(
        0.54, 0.965, title,
        ha="center", va="top", fontsize=12, fontweight="bold", color="#222222",
    )
    fig.text(
        0.54, 0.035, footnote,
        ha="center", va="bottom", fontsize=8.5, color="#444444",
        style="italic",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)
    print("Saved", path)


def score_columns(pairs, scans, en, en_txt):
    columns, records = [], []
    for kind, L, s in pairs:
        print(f"  scoring {kind} {L} i={s['i']} cover_min={s.get('cover_min', float('nan')):.2f}")
        partner = MODEL_CLS[L]()
        l_txt = partner.embed_texts(CLASSES[L]).detach()
        atk_img = scans[L]["attacked"][s["i"]]
        ours_img = scans[L]["gated"][s["i"]]
        scores = score_pair(en, en_txt, partner, l_txt, atk_img, ours_img)
        col = {
            "L": L,
            "kind": kind,
            "true": int(s["true"]),
            "target": int(s["target"]),
            "atk_img": atk_img,
            "ours_img": ours_img,
            "scores": scores,
            "gate": bool(scans[L]["gate"][s["i"]]),
            "i": int(s["i"]),
            "cover_min": float(s.get("cover_min", -1)),
        }
        columns.append(col)
        records.append({
            "kind": kind,
            "L": L,
            "i": int(s["i"]),
            "true": CLASSES["en"][int(s["true"])],
            "target": CLASSES["en"][int(s["target"])],
            "gate": col["gate"],
            "cover_min": col["cover_min"],
            "covers": s.get("covers"),
            "scores": scores,
        })
        del partner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return columns, records


def build_columns_from_records(records, data, en, en_txt):
    """Rebuild images + scores for a list of pick records."""
    columns = []
    by_L = {}
    for order, rec in enumerate(records):
        by_L.setdefault(rec["L"], []).append((order, rec))

    tmp = {}
    for L, items in by_L.items():
        print(f"  rebuild L={L} for {len(items)} pick(s)...")
        partner = MODEL_CLS[L]()
        l_txt = partner.embed_texts(CLASSES[L]).detach()
        cams = load_cached_cams(L, 1000)
        gates = load_gate_flags(L, 1000)
        g_atk = gates["gate_attacked"]
        attacked_all = build_partner_multi_attack(data, L)
        for order, rec in items:
            i = int(rec["i"])
            mask = build_cc_bbox_mask(
                cams["atk_en"][i], cams["atk_l"][i], threshold=THRESHOLD,
            )
            atk_img = attacked_all[i]
            ours_img = apply_mask(atk_img, mask, fill="black") if g_atk[i] else atk_img
            scores = score_pair(en, en_txt, partner, l_txt, atk_img, ours_img)
            tmp[order] = {
                "L": L,
                "kind": rec["kind"],
                "true": CLASSES["en"].index(rec["true"]),
                "target": CLASSES["en"].index(rec["target"]),
                "atk_img": atk_img,
                "ours_img": ours_img,
                "scores": scores,
                "gate": bool(g_atk[i]),
                "i": i,
                "cover_min": float(rec.get("cover_min", -1)),
            }
            rec["scores"] = scores
            rec["gate"] = bool(g_atk[i])
        del partner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for order in range(len(records)):
        columns.append(tmp[order])
    return columns


def run_full_scan(data, en, en_txt):
    skip = skip_success_indices(data["true"], data["target"])
    print(f"Skip dog-true / dog<-ship indices: n={len(skip)}")

    scans = {}
    for L in PARTNERS:
        scans[L] = scan_partner(L, data, en, en_txt, skip)

    used_true, used_target, used_i = set(), set(), set()
    success_pairs = []
    for L in SUCCESS_PARTNER_ORDER:
        s = pick_success(scans[L], used_true, used_target, used_i)
        if s is None:
            raise RuntimeError(f"no success pick for {L}")
        used_true.add(s["true"])
        used_target.add(s["target"])
        used_i.add(s["i"])
        success_pairs.append(("success", L, s))
        print(
            f"  pick success {L.upper()} i={s['i']} "
            f"{CLASSES['en'][s['true']]}->{CLASSES['en'][s['target']]} "
            f"both={s['both']} cover_min={s['cover_min']:.2f}"
        )
    assert len(success_pairs) == N_SUCCESS

    fail_pairs = [("fail", L, s) for L, s in pick_failures(scans, used_i, n_per_partner=1)]
    for kind, L, s in fail_pairs:
        print(
            f"  pick fail {L.upper()} i={s['i']} "
            f"{CLASSES['en'][s['true']]}->{CLASSES['en'][s['target']]} "
            f"gate={s['gate']}"
        )

    success_cols, success_recs = score_columns(success_pairs, scans, en, en_txt)
    fail_cols, fail_recs = score_columns(fail_pairs, scans, en, en_txt)

    picks = {
        "skip_dog_true_indices_n": len(skip),
        "threshold": THRESHOLD,
        "attack": "multi",
        "fill": "black",
        "min_sticker_cover": MIN_STICKER_COVER,
        "note": (
            "Percentages are CLIP top-1 softmax classification probabilities "
            "(temperature 100). Success picks require both stickers >= "
            f"{MIN_STICKER_COVER:.0%} covered by cc_bbox_black."
        ),
        "success": success_recs,
        "failures": fail_recs,
        # backward-compat alias
        "columns": success_recs,
    }
    return success_cols, fail_cols, picks


FOOTNOTE = (
    "% = CLIP top-1 softmax classification probability. "
    "Green = matches true label; red = incorrect."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerender", action="store_true",
        help="Rebuild figures from existing picks.json (skip full scan).",
    )
    args = parser.parse_args()

    require_cuda()
    assert DEVICE == "cuda", DEVICE

    out_ok = HERE / "qualitative_figure.png"
    out_fail = HERE / "qualitative_failures.png"
    out_json = HERE / "picks.json"

    data = load_protocol_data(1000)
    print("Loading EN CLIP...")
    en = EnCLIP()
    en_txt = en.embed_texts(CLASSES["en"]).detach()

    if args.rerender:
        if not out_json.is_file():
            print("ERROR: picks.json missing; run without --rerender first.", file=sys.stderr)
            sys.exit(1)
        with open(out_json, encoding="utf-8") as f:
            picks = json.load(f)
        success_recs = picks.get("success") or [
            c for c in picks.get("columns", []) if c.get("kind") != "fail"
        ]
        fail_recs = picks.get("failures") or [
            c for c in picks.get("columns", []) if c.get("kind") == "fail"
        ]
        print("Rerender success columns...")
        success_cols = build_columns_from_records(success_recs, data, en, en_txt)
        print("Rerender failure columns...")
        fail_cols = build_columns_from_records(fail_recs, data, en, en_txt)
        picks["success"] = success_recs
        picks["failures"] = fail_recs
        picks["columns"] = success_recs
    else:
        success_cols, fail_cols, picks = run_full_scan(data, en, en_txt)

    save_grid(
        out_ok, success_cols,
        title="Qualitative results — gated cc_bbox_black (recoveries)",
        footnote=FOOTNOTE,
    )
    save_grid(
        out_fail, fail_cols,
        title="Qualitative failure cases — gated cc_bbox_black",
        footnote=FOOTNOTE + " Residual fails after detector-gated black fill.",
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(picks, f, indent=2)
    print("Wrote", out_json)
    print("Done.")


if __name__ == "__main__":
    main()
