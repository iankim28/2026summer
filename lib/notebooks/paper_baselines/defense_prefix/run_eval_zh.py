"""ChineseCLIP Defense-Prefix eval on PROTOCOL dual-box CIFAR (ZH model + ZH DP token).

Gate ladder: n=16 sanity → n=100 smoke → n=1000 final.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _common.protocol import (  # noqa: E402
    DEVICE,
    build_multi_attack,
    load_protocol_data,
    progress_log,
    write_summary,
)

from zh_dp_encode import (  # noqa: E402
    ZH_MODEL_ID,
    class_prompts,
    encode_images,
    encode_text_vanilla,
    encode_text_with_dp,
    load_zh_clip,
    preprocess_images,
)

RESULTS = Path(__file__).resolve().parent / "results"
CIFAR_TOKEN_ZH = RESULTS / "dp_cifar10_zh_vit-b16.pt"
COST_PROXY = 2


def load_dp_vec(token_path: Path, device: str = DEVICE) -> torch.Tensor:
    prefix = torch.load(token_path, map_location=device, weights_only=False)
    if isinstance(prefix, torch.nn.Parameter):
        prefix = prefix.data
    if isinstance(prefix, torch.Tensor) and prefix.ndim == 2:
        prefix = prefix.reshape(-1)
    return prefix.to(device).float()


@torch.no_grad()
def classify(model, processor, imgs, text_feat, batch_size=64):
    preds = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        pv = preprocess_images(processor, batch, DEVICE)
        imf = encode_images(model, pv)
        preds.append((imf @ text_feat.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def run_eval(n: int, status: str, token_path: Path):
    assert torch.cuda.is_available(), "CUDA required"
    data = load_protocol_data(n=n)
    model, processor = load_zh_clip(DEVICE)
    dp_vec = load_dp_vec(token_path, DEVICE)

    prompts_v = class_prompts(dp=False)
    prompts_dp = class_prompts(dp=True)
    vanilla_txt = encode_text_vanilla(model, processor, prompts_v, DEVICE)
    dp_txt = encode_text_with_dp(model, processor, prompts_dp, dp_vec, DEVICE)
    cos = float((vanilla_txt * dp_txt).sum(-1).mean().item())
    print(f"Mean cosine(vanilla, DP ZH text feats) = {cos:.4f} (must be < 1)")
    if cos > 0.9999:
        raise RuntimeError("ZH DP token had no effect on text features")

    attacked, _ = build_multi_attack(data)
    true, target = data["true"], data["target"]

    clean_vanilla = classify(model, processor, data["clean_224"], vanilla_txt)
    atk_vanilla = classify(model, processor, attacked, vanilla_txt)
    clean_acc = float((clean_vanilla == true).mean())
    atk_acc = float((atk_vanilla == true).mean())
    atk_asr = float((atk_vanilla == target).mean())
    print(
        f"Vanilla ZH clean={100*clean_acc:.1f}%  attack_acc={100*atk_acc:.1f}%  "
        f"ASR={100*atk_asr:.1f}%"
    )

    t0 = time.time()
    def_preds = []
    running = 0
    for i in range(len(attacked)):
        pred = classify(model, processor, [attacked[i]], dp_txt)[0]
        def_preds.append(pred)
        running += int(pred == true[i])
        progress_log(i, len(attacked), running, t0)
    def_preds = np.array(def_preds)
    def_acc = float((def_preds == true).mean())
    def_asr = float((def_preds == target).mean())

    clean_dp = classify(model, processor, data["clean_224"], dp_txt)
    clean_dp_acc = float((clean_dp == true).mean())
    clean_delta = clean_dp_acc - clean_acc

    changed = int((def_preds != atk_vanilla).sum())
    print(
        f"ZH DP defended_acc={100*def_acc:.1f}%  ASR={100*def_asr:.1f}%  "
        f"Clean_delta={100*clean_delta:.1f}pp  "
        f"preds_changed_vs_vanilla={changed}/{len(def_preds)}"
    )
    if status == "sanity" and changed == 0 and abs(def_acc - atk_acc) < 1e-9:
        raise RuntimeError("Gate A fail: ZH DP predictions identical to no-defense")

    payload = {
        "method": "defense_prefix",
        "status": status,
        "n": int(data["n"]),
        "scope": "zh_only",
        "lang": "zh",
        "model": ZH_MODEL_ID,
        "token": str(token_path),
        "inference_cost": COST_PROXY,
        "defense": {
            "zh": {
                "acc": def_acc,
                "asr": def_asr,
                "baseline_acc": atk_acc,
                "baseline_asr": atk_asr,
            }
        },
        "clean_degradation": {
            "zh": {
                "baseline_acc": clean_acc,
                "masked_acc": clean_dp_acc,
                "delta_acc": clean_delta,
            }
        },
        "defense_acc_zh": def_acc,
        "clean_delta_zh": clean_delta,
        "preds_changed_vs_vanilla": changed,
        "mean_cosine_vanilla_dp_text": cos,
        "notes": (
            "CIFAR-trained ChineseCLIP DP token; ZH prompts "
            "(一张*{}的照片。); multi EN+ZH stickers; eval sample unused in train."
        ),
    }
    out = RESULTS / f"comparison_summary_{status}_n{data['n']}_zh.json"
    write_summary(out, payload)
    write_summary(RESULTS / "comparison_summary_zh.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--status", choices=["sanity", "smoke", "final"], required=True)
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="Path to ZH DP token .pt (default: results/dp_cifar10_zh_vit-b16.pt)",
    )
    args = ap.parse_args()
    token = Path(args.token) if args.token else CIFAR_TOKEN_ZH
    if not token.is_file():
        raise FileNotFoundError(f"ZH DP token not found: {token}")
    run_eval(args.n, args.status, token)


if __name__ == "__main__":
    main()
