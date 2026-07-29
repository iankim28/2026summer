"""True 4-way occlusion: EN ∩ ZH ∩ KO ∩ JA Attn-last → cc_bbox_black.

Protocol dual-box EN+ZH multi attack on CIFAR10_BALANCED_1000_SAMPLE.
Mask from intersection of all four Attn-last maps (thr=0.95, dilate=3, top_k=2).
Scores EN/ZH/KO/JA under never / always / gated (gate trained on EN vs
partner-mean cams using the Phase-C feature recipe).

CUDA required.
  Smoke: python run_eval.py --n 16
  Final: python run_eval.py --n 1000
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
from scipy import ndimage
from transformers import AutoModel, AutoProcessor, ChineseCLIPModel, ChineseCLIPProcessor

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "paper_baselines"
AD = HERE.parent / "attack_detector"
EN_HELPERS = HERE.parent / "en_neglect_vs_blur"
sys.path.insert(0, str(PB))
sys.path.insert(0, str(AD))
sys.path.insert(0, str(EN_HELPERS))

assert torch.cuda.is_available(), "CUDA required — refuse CPU long runs"
DEVICE = "cuda"
print("Device:", DEVICE, torch.cuda.get_device_name(0))

from _common.protocol import (  # noqa: E402
    DISPLAY_SIZE,
    EnCLIP,
    acc_asr,
    classify_batch,
    load_protocol_data,
    write_summary,
)
from make_phase_ab_viz import build_feature_matrix, train_gate  # noqa: E402

RESULTS = HERE / "results"
ALL_LANGS = ("en", "zh", "ko", "ja")
THR = 0.95
DILATE = 3
TOP_K = 2
FILL = "black"
FONT_SIZE = 24
PAD = 8

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


def mixed2000(atk, clean):
    return 0.5 * float(atk) + 0.5 * float(clean)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


class ZhCLIP:
    lang = "zh"
    backend = "hf"

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
    backend = "hf"

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
    backend = "open_clip"

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


# Patch EnCLIP with backend tag for attn
def _tag_en(en):
    en.backend = "open_clip"
    en.lang = "en"
    return en


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
    font_map = {"en": _LAT_FONT, "zh": _CJK_FONT, "ko": _KO_FONT, "ja": _CJK_FONT}
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(font_map[lang])
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill="black", font=font)
    return img


def build_en_zh_multi(data):
    """Protocol dual-box EN+ZH multi attack."""
    out = []
    for i in range(data["n"]):
        t = int(data["target"][i])
        en_w, zh_w = CLASSES["en"][t], CLASSES["zh"][t]
        xy0 = data["attack_pos"]["en"][i]
        xy1 = data["attack_pos"]["l"][i]
        out.append(draw_dual_box(data["clean_224"][i], en_w, "en", zh_w, "zh", xy0, xy1))
    return out


def align_cam(cam, size=DISPLAY_SIZE):
    cam = np.asarray(cam, dtype=np.float64)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    mx = cam.max()
    if mx > 0:
        cam = cam / mx
    return (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
        )
        / 255.0
    )


def n_cam_intersection(*cams):
    return np.minimum.reduce([align_cam(c) for c in cams])


def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad[:-2, :-2]
            | pad[:-2, 1:-1]
            | pad[:-2, 2:]
            | pad[1:-1, :-2]
            | pad[1:-1, 1:-1]
            | pad[1:-1, 2:]
            | pad[2:, :-2]
            | pad[2:, 1:-1]
            | pad[2:, 2:]
        )
    return m


def cam_to_mask(saliency, threshold=0.95, dilate=3):
    thr = np.percentile(saliency, threshold * 100)
    mask = saliency >= thr
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask


def filter_mask_components(mask, top_k=2, bbox_snap=False):
    labeled, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = set(np.argsort(sizes)[::-1][:top_k] + 1)
    out = np.zeros_like(mask, dtype=bool)
    for i in keep:
        comp = labeled == i
        if bbox_snap:
            ys, xs = np.where(comp)
            out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        else:
            out |= comp
    return out


def build_cc_bbox_4way(cam_en, cam_zh, cam_ko, cam_ja, threshold=THR, dilate=DILATE, top_k=TOP_K):
    inter = n_cam_intersection(cam_en, cam_zh, cam_ko, cam_ja)
    mask = cam_to_mask(inter, threshold, dilate=dilate)
    return filter_mask_components(mask, top_k=top_k, bbox_snap=True)


def apply_mask(pil_img, mask, fill="black"):
    arr = np.array(pil_img.convert("RGB"))
    m = mask.astype(bool)
    if m.shape != arr.shape[:2]:
        m = (
            np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(arr.shape[1::-1], Image.NEAREST)
            )
            > 127
        )
    out = arr.copy()
    if not m.any():
        return Image.fromarray(out)
    if fill == "black":
        out[m] = 0
    else:
        raise ValueError(fill)
    return Image.fromarray(out.astype(np.uint8))


def _make_openclip_hook(collector):
    def hook(module, inputs, output):
        q_in = inputs[0]
        if getattr(module, "batch_first", False):
            B, L, D = q_in.shape
        else:
            L, B, D = q_in.shape
            q_in = q_in.transpose(0, 1).contiguous()
        n_head = module.num_heads
        hd = D // n_head
        with torch.no_grad():
            qkv = F.linear(q_in, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * (hd**-0.5)
            attn = attn.softmax(dim=-1)
        collector.append(attn[0].detach().cpu())

    return hook


def _norm_cam(cam):
    cam = np.maximum(cam if isinstance(cam, np.ndarray) else cam.cpu().numpy(), 0)
    cam -= cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def _build_attn_cam(all_attns):
    a = all_attns[-1]
    cls_row = a.mean(0)[0, 1:]
    n = int(round(cls_row.shape[0] ** 0.5))
    return _norm_cam(cls_row.reshape(n, n).numpy())


@torch.no_grad()
def classify_and_attn(model, text_emb, pil_img):
    backend = getattr(model, "backend", "open_clip")
    if backend == "open_clip":
        x = model.pp(pil_img).unsqueeze(0).to(DEVICE)
        collector = []
        handles = [
            rb.attn.register_forward_hook(_make_openclip_hook(collector))
            for rb in model.m.visual.transformer.resblocks
        ]
        feat = model.m.visual(x)
        imf = F.normalize(feat, dim=-1)
        pred = int((imf @ text_emb.t()).squeeze().argmax().item())
        for h in handles:
            h.remove()
        return pred, _build_attn_cam(collector)

    pv = model.p(images=[pil_img], return_tensors="pt").pixel_values.to(DEVICE)
    vis_out = model.m.vision_model(pixel_values=pv, output_attentions=True)
    if hasattr(model.m, "visual_projection"):
        proj = model.m.visual_projection(vis_out.pooler_output)
    else:
        proj = vis_out.pooler_output
    imf = F.normalize(proj, dim=-1)
    pred = int((imf @ text_emb.t()).squeeze().argmax().item())
    attns = [a[0].cpu() for a in vis_out.attentions]
    return pred, _build_attn_cam(attns)


def compute_cams(models, text_embs, images, label=""):
    cams = {lang: [] for lang in ALL_LANGS}
    t0 = time.time()
    n = len(images)
    for i, img in enumerate(images):
        for lang in ALL_LANGS:
            _, cam = classify_and_attn(models[lang], text_embs[lang], img)
            cams[lang].append(cam)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(
                f"  cams {label} {i+1}/{n}  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return cams


def build_masks(cams):
    return [
        build_cc_bbox_4way(cams["en"][i], cams["zh"][i], cams["ko"][i], cams["ja"][i])
        for i in range(len(cams["en"]))
    ]


def partner_mean_cams(cams):
    """Mean of ZH/KO/JA maps — used as cams_l for the Phase-C gate features."""
    out = []
    for i in range(len(cams["zh"])):
        out.append(
            (align_cam(cams["zh"][i]) + align_cam(cams["ko"][i]) + align_cam(cams["ja"][i])) / 3.0
        )
    return out


def train_mode_gate(cams_clean, cams_atk, n):
    clean_pack = {"cams_en": cams_clean["en"], "cams_l": partner_mean_cams(cams_clean)}
    atk_pack = {"cams_en": cams_atk["en"], "cams_l": partner_mean_cams(cams_atk)}
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


def classify_always(models, text_embs, images, masks):
    defended = [apply_mask(img, m, fill=FILL) for img, m in zip(images, masks)]
    preds = {lang: classify_batch(models[lang], defended, text_embs[lang]) for lang in ALL_LANGS}
    cov = float(np.mean([m.mean() for m in masks])) if masks else 0.0
    return preds, cov


def classify_gated(models, text_embs, images, masks, gate_flags):
    gate_flags = np.asarray(gate_flags, dtype=bool)
    n = len(images)
    out = {lang: np.zeros(n, dtype=np.int64) for lang in ALL_LANGS}
    def_idx = np.where(gate_flags)[0]
    pass_idx = np.where(~gate_flags)[0]
    if len(pass_idx):
        pass_imgs = [images[i] for i in pass_idx]
        for lang in ALL_LANGS:
            out[lang][pass_idx] = classify_batch(models[lang], pass_imgs, text_embs[lang])
    if len(def_idx):
        def_imgs = [images[i] for i in def_idx]
        def_masks = [masks[i] for i in def_idx]
        defended = [apply_mask(img, m, fill=FILL) for img, m in zip(def_imgs, def_masks)]
        for lang in ALL_LANGS:
            out[lang][def_idx] = classify_batch(models[lang], defended, text_embs[lang])
    return out, float(len(def_idx) / max(n, 1))


def score_langs(preds, true, target):
    out = {}
    for lang in ALL_LANGS:
        acc, asr = acc_asr(preds[lang], true, target)
        out[lang] = {"acc": acc, "asr": asr}
    out["mean_acc"] = float(np.mean([out[l]["acc"] for l in ALL_LANGS]))
    return out


def run(n: int):
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

    print("Never baselines (clean + attacked)…", flush=True)
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
    masks_clean = build_masks(cams_clean)

    print("Attn-last cams (attacked)…", flush=True)
    cams_atk = compute_cams(models, text_embs, atk_imgs, label="atk")
    masks_atk = build_masks(cams_atk)

    gate = train_mode_gate(cams_clean, cams_atk, data["n"])
    print(
        f"Gate primary={gate['primary']} thr={gate['threshold']:.4f} "
        f"fire_atk={gate['fire_attacked']:.3f} fire_clean={gate['fire_clean']:.3f}",
        flush=True,
    )

    arms = {}
    arms["never"] = {
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

    for arm_name, a in arms.items():
        print(
            f"  {arm_name:8s} mean_atk={fmt_pct(a['mean_atk_acc'])} "
            f"EN={fmt_pct(a['en_acc'])} ZH={fmt_pct(a['zh_acc'])} "
            f"KO={fmt_pct(a['ko_acc'])} JA={fmt_pct(a['ja_acc'])} "
            f"MIXED={fmt_pct(a['mean_mixed_2000'])}",
            flush=True,
        )

    summary = {
        "method": "four_way_cc_bbox_black",
        "n": int(data["n"]),
        "attack": "multi_en_zh",
        "mask": "EN∩ZH∩KO∩JA Attn-last cc_bbox",
        "threshold": THR,
        "dilate": DILATE,
        "top_k": TOP_K,
        "fill": FILL,
        "gate_note": (
            "Gate trained on EN Attn-last vs mean(ZH,KO,JA) Attn-last features "
            "(Phase-C extract_pair_features recipe); masks are true 4-way intersection."
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
        "arms": arms,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"four_way_n{data['n']}.json"
    write_summary(out_path, summary)
    print("Saved", out_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="subset size (16 smoke / 1000 final)")
    args = ap.parse_args()
    run(n=args.n)


if __name__ == "__main__":
    main()
