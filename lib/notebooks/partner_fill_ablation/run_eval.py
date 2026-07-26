"""Partner gated fill ablation — blur / mean / black / neglect for ZH, KO, JA.

Follows EN lead in en_neglect_vs_blur (gated arms only). Reuses Phase-C
Attn-last caches + detector gates from attack_detector. Scores EN and L under
the same EN∩L cc_bbox masks; reports L MIXED2000 and bilingual EN+L MIXED2000.

CUDA required. Smoke: --n 16 --partner zh. Final: --n 1000 --partner all.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, ChineseCLIPModel, ChineseCLIPProcessor

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
AD = HERE.parent / "attack_detector"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(HERE))

from _common.protocol import (  # noqa: E402
    DEVICE,
    EnCLIP,
    acc_asr,
    classify_batch,
    load_protocol_data,
    write_summary,
)
from helpers import (  # noqa: E402
    BACKBONE,
    BLUR_RADIUS,
    apply_mask,
    build_cc_bbox_mask,
    classify_neglect_batch,
)

RESULTS = HERE / "results"
FILLS = ["neglect", "blur", "mean", "black"]
PARTNERS = ["zh", "ko", "ja"]

CLASSES = {
    "en": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
    "ko": ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"],
    "ja": ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"],
}
TMPL = {
    "en": "a photo of a {}.",
    "zh": "一张{}的照片。",
    "ko": "{}의 사진.",
    "ja": "{}の写真。",
}
DISPLAY_SIZE = 224
FONT_SIZE = 24
PAD = 8


def mixed2000(atk, clean):
    return 0.5 * float(atk) + 0.5 * float(clean)


def fmt(x):
    return f"{100 * x:.2f}%"


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


class ZhCLIP:
    lang = "zh"

    def __init__(self):
        self.m = (
            ChineseCLIPModel.from_pretrained(
                "OFA-Sys/chinese-clip-vit-base-patch16", attn_implementation="eager"
            )
            .to(DEVICE)
            .eval()
        )
        self.p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(
            text=[TMPL["zh"].format(w) for w in words],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t["input_ids"],
            attention_mask=t["attention_mask"],
            token_type_ids=t.get("token_type_ids"),
        )
        return F.normalize(_clip_feat(out), dim=-1)


class KoCLIP:
    lang = "ko"

    def __init__(self):
        self.m = (
            AutoModel.from_pretrained(
                "Bingsu/clip-vit-base-patch32-ko", attn_implementation="eager"
            )
            .to(DEVICE)
            .eval()
        )
        self.p = AutoProcessor.from_pretrained("Bingsu/clip-vit-base-patch32-ko")

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


class JaCLIP:
    lang = "ja"

    def __init__(self):
        mid = "hf-hub:llm-jp/llm-jp-clip-vit-base-patch16"
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


MODEL_CLS = {"zh": ZhCLIP, "ko": KoCLIP, "ja": JaCLIP}


def _font_paths():
    if platform.system() == "Windows":
        wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        cjk = os.path.join(wf, "msyh.ttc")
        lat = os.path.join(wf, "arial.ttf")
        ko = os.path.join(wf, "malgun.ttf")
        if not os.path.isfile(ko):
            ko = cjk
        return cjk, lat, ko
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    lat = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return cjk, lat, cjk


_CJK_FONT, _LAT_FONT, _KO_FONT = _font_paths()
_FONT_CACHE: dict = {}


def _font_for_lang(lang):
    if lang == "en":
        return _LAT_FONT
    if lang == "ko":
        return _KO_FONT
    return _CJK_FONT


def _get_font(fp, size=FONT_SIZE):
    key = (fp, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y


def draw_dual_box(img, word0, lang0, word1, lang1, xy0, xy1):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_font_for_lang(lang))
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill="black", font=font)
    return img


def build_partner_multi_attack(data, L):
    """EN+L dual-box multi attack with partner-native labels/fonts."""
    out = []
    for i in range(data["n"]):
        t = int(data["target"][i])
        en_w, l_w = CLASSES["en"][t], CLASSES[L][t]
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        # attack_pos rows may be remapped lists under subset; protocol keeps en/l lists
        out.append(draw_dual_box(data["clean_224"][i], en_w, "en", l_w, L, xy0, xy1))
    return out


def load_gate_flags(L, n_images=1000):
    """Recompute Phase-C gates from attn cache (no CLIP)."""
    sys.path.insert(0, str(AD))
    from make_phase_ab_viz import (  # noqa: E402
        build_feature_matrix,
        load_cache,
        train_gate,
    )

    packed = load_cache(L, AD / "results" / L / "multi")
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


def load_cached_cams(L, n=1000):
    path = AD / "results" / L / "multi" / "cache" / f"attn_en_{L}_clean_multi.npz"
    z = np.load(path, allow_pickle=False)

    def _cams_l(prefix):
        for key in (f"{prefix}_cams_l", f"{prefix}_cams_{L}", f"{prefix}_cams_zh"):
            if key in z.files:
                return z[key][:n]
        raise KeyError(f"no L cams in {path} for prefix={prefix}; files={z.files}")

    return {
        "clean_en": z["clean_cams_en"][:n],
        "clean_l": _cams_l("clean"),
        "atk_en": z["atk_cams_en"][:n],
        "atk_l": _cams_l("atk"),
    }


def build_masks_from_cams(cams_en, cams_l, threshold=0.95):
    return [build_cc_bbox_mask(ce, cl, threshold=threshold) for ce, cl in zip(cams_en, cams_l)]


def _subset_sel(data):
    """Indices used by load_protocol_data subset (tune-first for n<=100)."""
    if "_sel" in data["attack_pos"]:
        return np.array(data["attack_pos"]["_sel"], dtype=np.int64)
    return np.arange(data["n"], dtype=np.int64)


def classify_fill_one(model, lang, text_emb, images, masks, fill, patch_thr=0.5):
    if fill == "neglect":
        preds, n_patches = classify_neglect_batch(
            model, lang, text_emb, images, masks, patch_thr=patch_thr, device=DEVICE
        )
        return preds, {"mean_neglected_patches": float(n_patches.mean()) if len(n_patches) else 0.0}
    defended = [apply_mask(img, m, fill=fill) for img, m in zip(images, masks)]
    preds = classify_batch(model, defended, text_emb)
    cov = float(np.mean([m.mean() for m in masks])) if masks else 0.0
    return preds, {"mean_mask_coverage": cov}


def classify_gated_pair(
    en,
    en_txt,
    partner,
    L,
    l_txt,
    images,
    masks,
    gate_flags,
    fill,
    patch_thr=0.5,
):
    """Gated classify for EN and L. Image fills share defended PIL; neglect is per-model."""
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    en_preds = np.zeros(n, dtype=np.int64)
    l_preds = np.zeros(n, dtype=np.int64)
    def_idx = np.where(gate_flags)[0]
    pass_idx = np.where(~gate_flags)[0]
    if len(pass_idx):
        pass_imgs = [images[i] for i in pass_idx]
        en_preds[pass_idx] = classify_batch(en, pass_imgs, en_txt)
        l_preds[pass_idx] = classify_batch(partner, pass_imgs, l_txt)
    if len(def_idx):
        def_imgs = [images[i] for i in def_idx]
        def_masks = [masks[i] for i in def_idx]
        if fill == "neglect":
            en_preds[def_idx], _ = classify_neglect_batch(
                en, "en", en_txt, def_imgs, def_masks, patch_thr=patch_thr, device=DEVICE
            )
            l_preds[def_idx], _ = classify_neglect_batch(
                partner, L, l_txt, def_imgs, def_masks, patch_thr=patch_thr, device=DEVICE
            )
        else:
            defended = [apply_mask(img, m, fill=fill) for img, m in zip(def_imgs, def_masks)]
            en_preds[def_idx] = classify_batch(en, defended, en_txt)
            l_preds[def_idx] = classify_batch(partner, defended, l_txt)
    return en_preds, l_preds, float(len(def_idx) / max(n, 1))


def run_partner(L, n, fills, patch_thr=0.5, threshold=0.95, status="final"):
    print(f"\n######## PARTNER {L.upper()} n={n} status={status} ########")
    t_all = time.time()
    data = load_protocol_data(n)
    attacked = build_partner_multi_attack(data, L)
    sel = _subset_sel(data)

    print("  Loading models...")
    en = EnCLIP()
    partner = MODEL_CLS[L]()
    en_txt = en.embed_texts(CLASSES["en"]).detach()
    l_txt = partner.embed_texts(CLASSES[L]).detach()

    # Caches are full-1000 ordered; subset by protocol sel
    print("  Loading cached cams + building cc_bbox masks...")
    cams = load_cached_cams(L, 1000)
    atk_masks = [
        build_cc_bbox_mask(cams["atk_en"][i], cams["atk_l"][i], threshold=threshold) for i in sel
    ]
    clean_masks = [
        build_cc_bbox_mask(cams["clean_en"][i], cams["clean_l"][i], threshold=threshold)
        for i in sel
    ]

    gates = load_gate_flags(L, 1000)
    g_atk = gates["gate_attacked"][sel]
    g_cln = gates["gate_clean"][sel]
    print(
        f"  detector={gates['primary']} thr={gates['threshold']:.4f} "
        f"fire_atk={gates['fire_attacked']:.3f} fire_clean={gates['fire_clean']:.3f}"
    )

    true, target = data["true"], data["target"]

    # never baselines
    never_en_atk = classify_batch(en, attacked, en_txt)
    never_l_atk = classify_batch(partner, attacked, l_txt)
    never_en_cln = classify_batch(en, data["clean_224"], en_txt)
    never_l_cln = classify_batch(partner, data["clean_224"], l_txt)
    never = {
        "en": {
            "atk_acc": float((never_en_atk == true).mean()),
            "asr": float((never_en_atk == target).mean()),
            "clean_acc": float((never_en_cln == true).mean()),
        },
        L: {
            "atk_acc": float((never_l_atk == true).mean()),
            "asr": float((never_l_atk == target).mean()),
            "clean_acc": float((never_l_cln == true).mean()),
        },
    }
    never["en"]["mixed_2000"] = mixed2000(never["en"]["atk_acc"], never["en"]["clean_acc"])
    never[L]["mixed_2000"] = mixed2000(never[L]["atk_acc"], never[L]["clean_acc"])
    never_atk_mean = 0.5 * (never["en"]["atk_acc"] + never[L]["atk_acc"])
    never_cln_mean = 0.5 * (never["en"]["clean_acc"] + never[L]["clean_acc"])
    never["bilingual_mixed_2000"] = mixed2000(never_atk_mean, never_cln_mean)
    print(
        f"  never: EN atk={fmt(never['en']['atk_acc'])} {L.upper()} atk={fmt(never[L]['atk_acc'])} "
        f"EN+{L.upper()} MIXED={fmt(never['bilingual_mixed_2000'])}"
    )

    out = {
        "L": L,
        "n": n,
        "status": status,
        "threshold": threshold,
        "patch_thr": patch_thr,
        "blur_radius": BLUR_RADIUS,
        "fills": fills,
        "backbone": {k: BACKBONE[k] for k in ("en", L)},
        "detector": {
            "primary": gates["primary"],
            "threshold": gates["threshold"],
            "fire_attacked": gates["fire_attacked"],
            "fire_clean": gates["fire_clean"],
        },
        "never": never,
        "arms": {},
        "note": (
            "gated Attn-last EN∩L cc_bbox; fills=neglect|blur|mean|black; "
            "bilingual_mixed_2000 = 0.5*mean(EN,L atk)+0.5*mean(EN,L clean_policy)"
        ),
    }

    for fill in fills:
        print(f"  gated arm={fill} ...", flush=True)
        t0 = time.time()
        en_atk, l_atk, def_frac_atk = classify_gated_pair(
            en, en_txt, partner, L, l_txt, attacked, atk_masks, g_atk, fill, patch_thr
        )
        en_cln, l_cln, def_frac_cln = classify_gated_pair(
            en,
            en_txt,
            partner,
            L,
            l_txt,
            data["clean_224"],
            clean_masks,
            g_cln,
            fill,
            patch_thr,
        )
        en_atk_acc, en_asr = acc_asr(en_atk, true, target)
        l_atk_acc, l_asr = acc_asr(l_atk, true, target)
        en_cln_acc = float((en_cln == true).mean())
        l_cln_acc = float((l_cln == true).mean())
        l_mixed = mixed2000(l_atk_acc, l_cln_acc)
        en_mixed = mixed2000(en_atk_acc, en_cln_acc)
        atk_mean = 0.5 * (en_atk_acc + l_atk_acc)
        cln_mean = 0.5 * (en_cln_acc + l_cln_acc)
        bi_mixed = mixed2000(atk_mean, cln_mean)
        arm = {
            "en": {
                "atk_acc": en_atk_acc,
                "asr": en_asr,
                "clean_acc": en_cln_acc,
                "clean_delta": en_cln_acc - never["en"]["clean_acc"],
                "mixed_2000": en_mixed,
            },
            L: {
                "atk_acc": l_atk_acc,
                "asr": l_asr,
                "clean_acc": l_cln_acc,
                "clean_delta": l_cln_acc - never[L]["clean_acc"],
                "mixed_2000": l_mixed,
            },
            "atk_mean": atk_mean,
            "clean_policy_mean": cln_mean,
            "bilingual_mixed_2000": bi_mixed,
            "defend_frac_attacked": def_frac_atk,
            "defend_frac_clean": def_frac_cln,
            "elapsed_s": time.time() - t0,
        }
        out["arms"][fill] = arm
        print(
            f"    gated_{fill}: {L.upper()} atk={fmt(l_atk_acc)} clean={fmt(l_cln_acc)} "
            f"L_MIXED={fmt(l_mixed)} | EN+{L.upper()} MIXED={fmt(bi_mixed)} "
            f"({arm['elapsed_s']:.0f}s)",
            flush=True,
        )

    # ranking by bilingual MIXED then L MIXED
    ranking = sorted(
        fills,
        key=lambda f: (
            out["arms"][f]["bilingual_mixed_2000"],
            out["arms"][f][L]["mixed_2000"],
        ),
        reverse=True,
    )
    out["ranking_bilingual_mixed"] = ranking
    out["winner"] = ranking[0]
    out["elapsed_s"] = time.time() - t_all

    out_path = RESULTS / L / f"gated_n{n}.json"
    write_summary(out_path, out)
    return out


def build_leaderboard(partner_outs):
    rows = []
    for out in partner_outs:
        L = out["L"]
        for fill, arm in out["arms"].items():
            rows.append(
                {
                    "L": L,
                    "fill": fill,
                    "policy": "gated",
                    "l_atk": arm[L]["atk_acc"],
                    "l_clean": arm[L]["clean_acc"],
                    "l_mixed_2000": arm[L]["mixed_2000"],
                    "en_atk": arm["en"]["atk_acc"],
                    "en_clean": arm["en"]["clean_acc"],
                    "en_mixed_2000": arm["en"]["mixed_2000"],
                    "bilingual_mixed_2000": arm["bilingual_mixed_2000"],
                    "defend_frac_attacked": arm["defend_frac_attacked"],
                    "defend_frac_clean": arm["defend_frac_clean"],
                }
            )
    # blur baseline bilingual from attack_detector for reference
    blur_ref = {}
    mix_path = AD / "results" / "mixed_2000_summary.json"
    if mix_path.exists():
        payload = json.loads(mix_path.read_text(encoding="utf-8"))
        for p in payload.get("partners", []):
            g = p["policies"]["gated"]
            blur_ref[p["L"]] = g["mixed_2000_mean_acc"]
    board = {
        "definition": (
            "Partner gated fill ranking (neglect/blur/mean/black). "
            "bilingual_mixed_2000 = 0.5*mean(EN,L atk)+0.5*mean(EN,L clean_policy). "
            "neglect = ViT patch-token zero (ignore)."
        ),
        "rows": rows,
        "winners": {o["L"]: o["winner"] for o in partner_outs},
        "phase_c_blur_bilingual_mixed_ref": blur_ref,
        "tables": {},
    }
    for out in partner_outs:
        L = out["L"]
        board["tables"][L] = [
            {
                "fill": fill,
                "l_atk": out["arms"][fill][L]["atk_acc"],
                "l_clean": out["arms"][fill][L]["clean_acc"],
                "l_mixed_2000": out["arms"][fill][L]["mixed_2000"],
                "bilingual_mixed_2000": out["arms"][fill]["bilingual_mixed_2000"],
            }
            for fill in FILLS
            if fill in out["arms"]
        ]
    write_summary(RESULTS / "leaderboard.json", board)
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--partner", default="all", help="zh|ko|ja|all")
    ap.add_argument("--fills", default=",".join(FILLS))
    ap.add_argument("--patch-thr", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--status", default="final")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — refuse CPU long runs")
    print("Device:", DEVICE, torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    fills = [f.strip() for f in args.fills.split(",") if f.strip()]
    for f in fills:
        if f not in FILLS:
            raise SystemExit(f"unknown fill={f}; allowed={FILLS}")

    if args.partner == "all":
        partners = PARTNERS
    else:
        if args.partner not in PARTNERS:
            raise SystemExit(f"unknown partner={args.partner}")
        partners = [args.partner]

    outs = []
    for L in partners:
        outs.append(
            run_partner(
                L,
                n=args.n,
                fills=fills,
                patch_thr=args.patch_thr,
                threshold=args.threshold,
                status=args.status,
            )
        )
    board = build_leaderboard(outs)
    print("\n=== LEADERBOARD (bilingual MIXED2000) ===")
    for L, table in board["tables"].items():
        print(f"  {L.upper()}: winner={board['winners'][L]}")
        for row in table:
            print(
                f"    {row['fill']:8s}  L_atk={fmt(row['l_atk'])}  "
                f"L_clean={fmt(row['l_clean'])}  L_MIXED={fmt(row['l_mixed_2000'])}  "
                f"EN+L_MIXED={fmt(row['bilingual_mixed_2000'])}"
            )


if __name__ == "__main__":
    main()
