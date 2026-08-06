"""4x4 grid occlusion winner C_2p_confdrop_blur on PROTOCOL dual-box CIFAR.

Search uses EN+ZH conf-drop (published recipe). Score all of EN/ZH/KO/JA.
Clean arm applies the same 2-patch greedy search on clean images.

Gate ladder: n=16 → n=100 → n=1000.
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
from PIL import Image, ImageFilter
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.protocol import (  # noqa: E402
    DEVICE,
    EnCLIP,
    ZhCLIP,
    acc_asr,
    build_multi_attack,
    classify_batch,
    load_protocol_data,
    write_summary,
)

RESULTS = Path(__file__).resolve().parent / "results"
COST_PROXY = 62
AGREE_BONUS = 0.05
BLUR_RADIUS = 12
DISPLAY_SIZE = 224
SEARCH_LANGS = ("en", "zh")
ALL_LANGS = ("en", "zh", "ko", "ja")
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


def _clip_feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


class KoCLIP:
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


def make_grid_patches(n, size=DISPLAY_SIZE):
    ph = pw = size // n
    return [
        (c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)
        for r in range(n)
        for c in range(n)
    ]


PATCHES_4 = make_grid_patches(4)


def occlude_blur(arr, rects, radius=BLUR_RADIUS):
    out = arr.copy()
    blurred = np.array(
        Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius))
    )
    for x0, y0, x1, y1 in rects:
        out[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
    return out


@torch.no_grad()
def classify_sims(model, imgs, words, text_emb=None, batch_size=64):
    all_sims = []
    tf = text_emb if text_emb is not None else model.embed_texts(words)
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i : i + batch_size])
        all_sims.append((imf @ tf.t()).cpu().numpy())
    return np.concatenate(all_sims, axis=0)


def score_confdrop(models, text, candidates, img_i, top, conf):
    scores = np.zeros(len(candidates))
    preds = {}
    for ml in SEARCH_LANGS:
        sims = classify_sims(models[ml], candidates, CLASSES[ml], text_emb=text[ml])
        drop = conf[ml][img_i] - sims[:, top[ml][img_i]]
        scores += drop
        preds[ml] = sims.argmax(1)
    scores /= len(SEARCH_LANGS)
    agree = (preds["en"] == preds["zh"]).astype(np.float64)
    scores += AGREE_BONUS * agree
    return scores


def run_1patch(models, text, imgs, top, conf, label=""):
    result, best, n_fwd = [], [], 0
    for j, img in enumerate(imgs):
        arr = np.array(img.convert("RGB"))
        cands = [Image.fromarray(occlude_blur(arr, [r])) for r in PATCHES_4]
        scores = score_confdrop(models, text, cands, j, top, conf)
        bi = int(scores.argmax())
        best.append(bi)
        result.append(cands[bi])
        n_fwd += len(PATCHES_4) * 2
        if (j + 1) % 25 == 0 or (j + 1) == len(imgs):
            print(f"  {label} 1p {j+1}/{len(imgs)}", flush=True)
    return result, best, n_fwd / len(imgs)


def run_2patch_greedy(models, text, imgs, first_idxs, top, conf, label=""):
    result, seconds, n_fwd = [], [], 0
    for j, (img, fp) in enumerate(zip(imgs, first_idxs)):
        arr = np.array(img.convert("RGB"))
        arr1 = occlude_blur(arr, [PATCHES_4[fp]])
        remain = [i for i in range(len(PATCHES_4)) if i != fp]
        cands = [Image.fromarray(occlude_blur(arr1, [PATCHES_4[pi]])) for pi in remain]
        scores = score_confdrop(models, text, cands, j, top, conf)
        bi = int(scores.argmax())
        seconds.append(remain[bi])
        result.append(cands[bi])
        n_fwd += len(remain) * 2
        if (j + 1) % 25 == 0 or (j + 1) == len(imgs):
            print(f"  {label} 2p {j+1}/{len(imgs)}", flush=True)
    return result, seconds, n_fwd / len(imgs)


def defend_set(models, text, imgs, label):
    sims = {
        ml: classify_sims(models[ml], imgs, CLASSES[ml], text_emb=text[ml])
        for ml in SEARCH_LANGS
    }
    top = {ml: sims[ml].argmax(1) for ml in SEARCH_LANGS}
    conf = {ml: sims[ml].max(1) for ml in SEARCH_LANGS}
    t0 = time.time()
    _, best, cost1 = run_1patch(models, text, imgs, top, conf, label=label)
    defended, _, cost2 = run_2patch_greedy(
        models, text, imgs, best, top, conf, label=label
    )
    return defended, cost1 + cost2, time.time() - t0


def run_eval(n: int, status: str):
    assert torch.cuda.is_available()
    data = load_protocol_data(n=n)
    attacked, _ = build_multi_attack(data)
    true, target = data["true"], data["target"]

    print("Loading CLIP models EN/ZH/KO/JA...", flush=True)
    models = {
        "en": EnCLIP(),
        "zh": ZhCLIP(),
        "ko": KoCLIP(),
        "ja": JaCLIP(),
    }
    text = {lang: models[lang].embed_texts(CLASSES[lang]) for lang in ALL_LANGS}

    clean_acc = {}
    atk_acc = {}
    atk_asr = {}
    for lang in ALL_LANGS:
        cpred = classify_batch(models[lang], data["clean_224"], text[lang])
        apred = classify_batch(models[lang], attacked, text[lang])
        clean_acc[lang] = float((cpred == true).mean())
        atk_acc[lang], atk_asr[lang] = acc_asr(apred, true, target)
    print(
        "Vanilla clean "
        + " ".join(f"{L.upper()}={100*clean_acc[L]:.1f}%" for L in ALL_LANGS)
    )
    print(
        "Vanilla atk "
        + " ".join(f"{L.upper()}={100*atk_acc[L]:.1f}%" for L in ALL_LANGS)
    )

    print("=== Attacked: C_2p_confdrop_blur ===", flush=True)
    defended, cost, elapsed = defend_set(models, text, attacked, label="atk")
    print(f"Attack search cost~{cost:.0f}/img in {elapsed:.1f}s", flush=True)

    def_acc = {}
    def_asr = {}
    for lang in ALL_LANGS:
        dpred = classify_batch(models[lang], defended, text[lang])
        def_acc[lang], def_asr[lang] = acc_asr(dpred, true, target)

    print("=== Clean arm: C_2p_confdrop_blur ===", flush=True)
    clean_defended, _, clean_elapsed = defend_set(
        models, text, data["clean_224"], label="clean"
    )
    print(f"Clean search done in {clean_elapsed:.1f}s", flush=True)

    clean_deg = {}
    for lang in ALL_LANGS:
        cpred = classify_batch(models[lang], clean_defended, text[lang])
        masked = float((cpred == true).mean())
        clean_deg[lang] = {
            "baseline_acc": clean_acc[lang],
            "masked_acc": masked,
            "delta_acc": masked - clean_acc[lang],
        }

    mean_atk = float(np.mean([def_acc[L] for L in ALL_LANGS]))
    mean_mixed = float(
        np.mean(
            [0.5 * def_acc[L] + 0.5 * clean_deg[L]["masked_acc"] for L in ALL_LANGS]
        )
    )
    print(
        f"Grid 4-lang mean_atk={100*mean_atk:.1f}% mean_mixed={100*mean_mixed:.1f}%  "
        + " ".join(f"{L.upper()}={100*def_acc[L]:.1f}%" for L in ALL_LANGS)
        + f"  CleanΔEN={100*clean_deg['en']['delta_acc']:.1f}pp"
    )

    if status == "sanity" and def_acc["en"] <= atk_acc["en"] + 0.05:
        raise RuntimeError("Gate A fail: grid defense did not improve EN attacked acc")

    payload = {
        "method": "grid_occlusion_C_2p_confdrop_blur",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_zh_ko_ja_multi",
        "search_langs": list(SEARCH_LANGS),
        "score_langs": list(ALL_LANGS),
        "agree_bonus": AGREE_BONUS,
        "blur_radius": BLUR_RADIUS,
        "inference_cost": COST_PROXY,
        "measured_cost_per_img": float(cost),
        "defense": {
            lang: {
                "acc": def_acc[lang],
                "asr": def_asr[lang],
                "baseline_acc": atk_acc[lang],
                "baseline_asr": atk_asr[lang],
                "mixed_2000": 0.5 * def_acc[lang]
                + 0.5 * clean_deg[lang]["masked_acc"],
            }
            for lang in ALL_LANGS
        },
        "clean_degradation": clean_deg,
        "defense_acc_mean": mean_atk,
        "mixed_2000_mean": mean_mixed,
        "clean_delta_en": clean_deg["en"]["delta_acc"],
        "notes": (
            "4x4 greedy 2-patch conf-drop + blur; search on EN+ZH; "
            "KO/JA score-only on chosen occlusions; clean arm same search."
        ),
    }
    write_summary(
        RESULTS / f"comparison_summary_4lang_{status}_n{data['n']}.json", payload
    )
    write_summary(RESULTS / "comparison_summary_4lang.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    args = ap.parse_args()
    run_eval(args.n, args.status)


if __name__ == "__main__":
    main()
