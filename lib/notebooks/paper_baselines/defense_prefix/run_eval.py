"""Defense-Prefix baseline on PROTOCOL dual-box CIFAR (pretrained EN token).

Gate ladder: n=16 sanity → n=100 smoke → n=1000 final.
Uses openai CLIP ViT-B/32 + published dp_vit-b32.pt (Azuma & Matsui 2023).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.protocol import (  # noqa: E402
    CLASSES,
    DEVICE,
    build_multi_attack,
    load_protocol_data,
    progress_log,
    write_summary,
)

VENDOR = ROOT / "_vendor" / "Defense-Prefix"
sys.path.insert(0, str(VENDOR))
from utils.non_nv import encode_text_with_learnt_tokens  # noqa: E402

DEFAULT_TOKEN = VENDOR / "learned_token" / "dp_vit-b32.pt"
CIFAR_TOKEN = Path(__file__).resolve().parent / "results" / "dp_cifar10_vit-b32.pt"
RESULTS = Path(__file__).resolve().parent / "results"
COST_PROXY = 2  # image fwd + text (cached) — same order as no_defense


def load_dp_model(token_path: Path):
    assert torch.cuda.is_available(), "CUDA required"
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    func_type = type(model.encode_text)
    model.encode_text_with_learnt_tokens = func_type(encode_text_with_learnt_tokens, model)
    prefix = torch.load(token_path, map_location=DEVICE, weights_only=False)
    if isinstance(prefix, torch.nn.Parameter):
        prefix = prefix.data
    prefix = prefix.to(DEVICE).to(model.dtype)
    print("Loaded DP token", tuple(prefix.shape), "from", token_path)
    return model, preprocess, prefix


@torch.no_grad()
def build_text_features(model, prefix_tokens):
    """Vanilla and DP-prefixed text features for CIFAR-10 EN classes."""
    classes = CLASSES["en"]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in classes]).to(DEVICE)
    text_prefix = torch.cat(
        [clip.tokenize(f"a photo of a * {c}.") for c in classes]
    ).to(DEVICE)
    asterix = clip.tokenize(["*"]).to(DEVICE)[0][1]

    vanilla = model.encode_text(text_inputs)
    vanilla = F.normalize(vanilla.float(), dim=-1)

    # Gate A check: DP features must differ from vanilla
    dp = model.encode_text_with_learnt_tokens(
        text_prefix, asterix, prefix_tokens.unsqueeze(0), is_emb=False
    )
    dp = F.normalize(dp.float(), dim=-1)
    cos = (vanilla * dp).sum(-1).mean().item()
    print(f"Mean cosine(vanilla, DP text feats) = {cos:.4f} (must be < 1)")
    if cos > 0.9999:
        raise RuntimeError("DP token had no effect on text features")
    return vanilla, dp


@torch.no_grad()
def classify_clip(model, preprocess, imgs, text_feat, batch_size=64):
    preds = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        x = torch.stack([preprocess(im) for im in batch]).to(DEVICE)
        imf = model.encode_image(x)
        imf = F.normalize(imf.float(), dim=-1)
        preds.append((imf @ text_feat.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def run_eval(n: int, status: str, token_path: Path):
    data = load_protocol_data(n=n)
    model, preprocess, prefix = load_dp_model(token_path)
    vanilla_txt, dp_txt = build_text_features(model, prefix)

    attacked, _ = build_multi_attack(data)
    true, target = data["true"], data["target"]

    # Baselines
    clean_vanilla = classify_clip(model, preprocess, data["clean_224"], vanilla_txt)
    atk_vanilla = classify_clip(model, preprocess, attacked, vanilla_txt)
    clean_acc = float((clean_vanilla == true).mean())
    atk_acc = float((atk_vanilla == true).mean())
    atk_asr = float((atk_vanilla == target).mean())
    print(f"Vanilla clean={100*clean_acc:.1f}%  attack_acc={100*atk_acc:.1f}%  ASR={100*atk_asr:.1f}%")

    # Defense on attacked + clean (for Clean Δ)
    t0 = time.time()
    def_preds = []
    running = 0
    for i in range(len(attacked)):
        pred = classify_clip(model, preprocess, [attacked[i]], dp_txt)[0]
        def_preds.append(pred)
        running += int(pred == true[i])
        progress_log(i, len(attacked), running, t0)
    def_preds = np.array(def_preds)
    def_acc = float((def_preds == true).mean())
    def_asr = float((def_preds == target).mean())

    clean_dp = classify_clip(model, preprocess, data["clean_224"], dp_txt)
    clean_dp_acc = float((clean_dp == true).mean())
    clean_delta = clean_dp_acc - clean_acc

    changed = int((def_preds != atk_vanilla).sum())
    print(
        f"DP defended_acc={100*def_acc:.1f}%  ASR={100*def_asr:.1f}%  "
        f"Clean_delta={100*clean_delta:.1f}pp  "
        f"preds_changed_vs_vanilla={changed}/{len(def_preds)}"
    )
    if status == "sanity" and changed == 0 and abs(def_acc - atk_acc) < 1e-9:
        raise RuntimeError("Gate A fail: DP predictions identical to no-defense")

    payload = {
        "method": "defense_prefix",
        "status": status,
        "n": int(data["n"]),
        "scope": "en_only",
        "token": str(token_path),
        "inference_cost": COST_PROXY,
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
                "masked_acc": clean_dp_acc,
                "delta_acc": clean_delta,
            }
        },
        "defense_acc_en": def_acc,
        "clean_delta_en": clean_delta,
        "preds_changed_vs_vanilla": changed,
        "notes": "Pretrained ImageNet-100 DP token; EN prompts only; multi EN+ZH stickers.",
    }
    out = RESULTS / f"comparison_summary_{status}_n{data['n']}.json"
    write_summary(out, payload)
    write_summary(RESULTS / "comparison_summary.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="Path to DP token .pt (default: CIFAR-trained if present else pretrained)",
    )
    args = ap.parse_args()
    if args.token:
        token = Path(args.token)
    elif CIFAR_TOKEN.is_file():
        token = CIFAR_TOKEN
    else:
        token = DEFAULT_TOKEN
    run_eval(args.n, args.status, token)


if __name__ == "__main__":
    main()
