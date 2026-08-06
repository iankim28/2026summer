"""Japanese CLIP (llm-jp) Defense-Prefix: splice a learned embedding at `*`."""
from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

JA_MODEL_ID = "hf-hub:llm-jp/llm-jp-clip-vit-base-patch16"
JA_CLASSES = [
    "飛行機",
    "自動車",
    "鳥",
    "猫",
    "鹿",
    "犬",
    "カエル",
    "馬",
    "船",
    "トラック",
]
JA_TMPL = "{}の写真。"
JA_TMPL_DP = "*{}の写真。"


def load_ja_clip(device: str = "cuda"):
    assert torch.cuda.is_available(), "CUDA required"
    model, _, preprocess = open_clip.create_model_and_transforms(JA_MODEL_ID)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(JA_MODEL_ID)
    return model, preprocess, tokenizer


def star_token_id(tokenizer) -> int:
    """Find the token id used for `*` in DP prompts."""
    ids = tokenizer(["*X"])[0].tolist()
    # Drop BOS/EOS/PAD; keep first content token (should be *).
    pad = 4
    bos, eos = 5, 6
    content = [i for i in ids if i not in (pad, bos, eos, 0)]
    if not content:
        raise RuntimeError("Could not locate '*' token id in JA tokenizer")
    return int(content[0])


def make_dp_embedding(model, device: str = "cuda") -> nn.Embedding:
    dim = model.text.transformer.embeddings.word_embeddings.weight.shape[1]
    prefix = torch.empty(1, dim, device=device)
    nn.init.normal_(prefix, std=0.02)
    return nn.Embedding.from_pretrained(prefix, freeze=False)


def preprocess_images(preprocess, imgs, device: str = "cuda"):
    return torch.stack([preprocess(im) for im in imgs]).to(device)


@torch.no_grad()
def encode_images(model, pixel_values: torch.Tensor) -> torch.Tensor:
    return F.normalize(model.encode_image(pixel_values).float(), dim=-1)


def encode_text_vanilla(model, tokenizer, prompts: list[str], device: str = "cuda"):
    x = tokenizer(prompts).to(device)
    return F.normalize(model.encode_text(x).float(), dim=-1)


def encode_text_with_dp(
    model, tokenizer, prompts_dp: list[str], dp_vec: torch.Tensor, device: str = "cuda"
):
    x = tokenizer(prompts_dp).to(device)
    pad = model.text.config.pad_token_id
    attn = (x != pad).long()
    star_id = star_token_id(tokenizer)
    star_mask = x == star_id
    if not star_mask.any():
        raise RuntimeError("No '*' token found in JA DP prompts — check JA_TMPL_DP")

    dp = dp_vec.reshape(-1).to(device=device)
    # Build embeddings once (no monkeypatch — avoids long-run hangs).
    emb_mod = model.text.transformer.embeddings
    hidden = emb_mod(input_ids=x)
    hidden = hidden.clone()
    hidden[star_mask] = dp.to(dtype=hidden.dtype)
    out = model.text.transformer(inputs_embeds=hidden, attention_mask=attn)
    # Pool at star token (MeanPooler dilutes DP to ~1/6 of the mean).
    star_pos = star_mask.float().argmax(dim=-1)
    pooled = out.last_hidden_state[
        torch.arange(x.shape[0], device=device), star_pos
    ]
    projected = model.text.proj(pooled)
    return F.normalize(projected.float(), dim=-1)


def class_prompts(dp: bool = False) -> list[str]:
    tmpl = JA_TMPL_DP if dp else JA_TMPL
    return [tmpl.format(c) for c in JA_CLASSES]
