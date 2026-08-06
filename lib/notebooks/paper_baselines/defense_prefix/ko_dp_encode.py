"""Korean CLIP Defense-Prefix: splice a learned embedding at the `*` token."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from transformers.masking_utils import create_causal_mask

KO_MODEL_ID = "Bingsu/clip-vit-base-patch32-ko"
KO_CLASSES = [
    "비행기",
    "자동차",
    "새",
    "고양이",
    "사슴",
    "개",
    "개구리",
    "말",
    "배",
    "트럭",
]
KO_TMPL = "{}의 사진."
KO_TMPL_DP = "*{}의 사진."
# CLIP BPE encodes standalone * as *</w> (id 252), not bare '*'.
STAR_TOKEN = "*</w>"


def _feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


def load_ko_clip(device: str = "cuda"):
    assert torch.cuda.is_available(), "CUDA required"
    model = (
        AutoModel.from_pretrained(KO_MODEL_ID, attn_implementation="eager")
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(KO_MODEL_ID)
    return model, processor


def star_token_id(processor) -> int:
    tid = processor.tokenizer.convert_tokens_to_ids(STAR_TOKEN)
    if tid is None or tid == processor.tokenizer.unk_token_id:
        raise RuntimeError(f"Tokenizer has no usable '{STAR_TOKEN}' token")
    return int(tid)


def make_dp_embedding(model, device: str = "cuda") -> nn.Embedding:
    dim = model.text_model.embeddings.token_embedding.weight.shape[1]
    prefix = torch.empty(1, dim, device=device)
    nn.init.normal_(prefix, std=0.02)
    return nn.Embedding.from_pretrained(prefix, freeze=False)


def preprocess_images(processor, imgs, device: str = "cuda"):
    return processor(images=imgs, return_tensors="pt").pixel_values.to(device)


@torch.no_grad()
def encode_images(model, pixel_values: torch.Tensor) -> torch.Tensor:
    out = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(_feat(out).float(), dim=-1)


def encode_text_vanilla(model, processor, prompts: list[str], device: str = "cuda"):
    tok = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    out = model.get_text_features(
        input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
    )
    return F.normalize(_feat(out).float(), dim=-1)


def encode_text_with_dp(
    model, processor, prompts_dp: list[str], dp_vec: torch.Tensor, device: str = "cuda"
):
    tok = processor(text=prompts_dp, padding=True, return_tensors="pt").to(device)
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    star_id = star_token_id(processor)

    hidden = model.text_model.embeddings(input_ids=input_ids)
    dp = dp_vec.reshape(-1).to(device=hidden.device, dtype=hidden.dtype)
    if dp.numel() != hidden.shape[-1]:
        raise ValueError(f"dp dim {dp.numel()} != word emb {hidden.shape[-1]}")
    mask = input_ids == star_id
    if not mask.any():
        raise RuntimeError("No '*' token found in KO DP prompts — check KO_TMPL_DP")
    hidden = hidden.clone()
    hidden[mask] = dp

    causal = create_causal_mask(
        config=model.text_model.config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    enc = model.text_model.encoder(
        inputs_embeds=hidden, attention_mask=causal, is_causal=True
    )
    last = model.text_model.final_layer_norm(enc.last_hidden_state)
    eos = input_ids.to(dtype=torch.int, device=last.device).argmax(dim=-1)
    pooled = last[torch.arange(last.shape[0], device=last.device), eos]
    feat = model.text_projection(pooled)
    return F.normalize(feat.float(), dim=-1)


def class_prompts(dp: bool = False) -> list[str]:
    tmpl = KO_TMPL_DP if dp else KO_TMPL
    return [tmpl.format(c) for c in KO_CLASSES]
