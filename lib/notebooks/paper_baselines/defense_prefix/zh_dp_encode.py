"""ChineseCLIP Defense-Prefix: splice a learned embedding at the `*` token."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

ZH_MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
ZH_CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
# Mirror EN `a photo of a * {c}.` — placeholder `*` before the class name.
ZH_TMPL = "一张{}的照片。"
ZH_TMPL_DP = "一张*{}的照片。"
STAR_TOKEN = "*"


def load_zh_clip(device: str = "cuda"):
    assert torch.cuda.is_available(), "CUDA required"
    model = ChineseCLIPModel.from_pretrained(
        ZH_MODEL_ID, attn_implementation="eager"
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = ChineseCLIPProcessor.from_pretrained(ZH_MODEL_ID)
    return model, processor


def star_token_id(processor: ChineseCLIPProcessor) -> int:
    tid = processor.tokenizer.convert_tokens_to_ids(STAR_TOKEN)
    if tid is None or tid == processor.tokenizer.unk_token_id:
        raise RuntimeError(f"Tokenizer has no usable '{STAR_TOKEN}' token")
    return int(tid)


def make_dp_embedding(model: ChineseCLIPModel, device: str = "cuda") -> nn.Embedding:
    dim = model.text_model.embeddings.word_embeddings.weight.shape[1]
    prefix = torch.empty(1, dim, device=device)
    nn.init.normal_(prefix, std=0.02)
    return nn.Embedding.from_pretrained(prefix, freeze=False)


def preprocess_images(processor: ChineseCLIPProcessor, imgs, device: str = "cuda"):
    """PIL list or tensor batch → pixel_values on device."""
    pv = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
    return pv


@torch.no_grad()
def encode_images(model: ChineseCLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    out = model.get_image_features(pixel_values=pixel_values)
    feat = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(feat.float(), dim=-1)


def encode_text_vanilla(
    model: ChineseCLIPModel,
    processor: ChineseCLIPProcessor,
    prompts: list[str],
    device: str = "cuda",
) -> torch.Tensor:
    tok = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    out = model.get_text_features(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        token_type_ids=tok.get("token_type_ids"),
    )
    feat = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(feat.float(), dim=-1)


def encode_text_with_dp(
    model: ChineseCLIPModel,
    processor: ChineseCLIPProcessor,
    prompts_dp: list[str],
    dp_vec: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode DP prompts; replace every `*` token embedding with `dp_vec` (1, D) or (D,)."""
    tok = processor(text=prompts_dp, padding=True, return_tensors="pt").to(device)
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    token_type_ids = tok.get("token_type_ids")
    star_id = star_token_id(processor)

    word = model.text_model.embeddings.word_embeddings(input_ids)
    dp = dp_vec.reshape(-1).to(device=word.device, dtype=word.dtype)
    if dp.numel() != word.shape[-1]:
        raise ValueError(f"dp dim {dp.numel()} != word emb {word.shape[-1]}")
    mask = input_ids == star_id
    if not mask.any():
        raise RuntimeError("No '*' token found in DP prompts — check ZH_TMPL_DP")
    # Expand dp across all star positions
    word = word.clone()
    word[mask] = dp

    text_outputs = model.text_model(
        inputs_embeds=word,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    pooled = text_outputs.last_hidden_state[:, 0, :]
    feat = model.text_projection(pooled)
    return F.normalize(feat.float(), dim=-1)


def class_prompts(dp: bool = False) -> list[str]:
    tmpl = ZH_TMPL_DP if dp else ZH_TMPL
    return [tmpl.format(c) for c in ZH_CLASSES]
