"""Partner CLIP wrappers for multi-lang baseline evals (KO / JA)."""
from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from .protocol import DEVICE

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
    "ko": ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"],
    "ja": ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"],
}
TMPL = {
    "en": "a photo of a {}.",
    "ko": "{}의 사진.",
    "ja": "{}の写真。",
}


def _clip_feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


class KoCLIP:
    """HF Bingsu KO CLIP; visual is transformers ViT (not open_clip MHA)."""

    lang = "ko"
    grid = 7
    n_layers = 12
    n_heads = 12

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
        return F.normalize(
            _clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1
        )

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

    def preprocess_tensor(self, pil_img):
        return self.p(images=[pil_img], return_tensors="pt").pixel_values[0]


class JaCLIP:
    """llm-jp open_clip ViT-B/16; same MHA hook surface as EN, grid=14."""

    lang = "ja"
    grid = 14
    n_layers = 12
    n_heads = 12

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
