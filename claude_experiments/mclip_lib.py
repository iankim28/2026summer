"""
Core library for multilingual CLIP adversarial transfer experiments.

Model: open_clip 'xlm-roberta-base-ViT-B-32' (laion5b) -- a multilingual CLIP
with a SHARED ViT-B/32 image encoder and an XLM-RoBERTa text tower (100 langs).

Design notes
------------
* Adversarial attacks operate in *pixel space* [0,1] at the encoder's input
  resolution (224x224). CLIP normalization is folded into the forward pass so
  an L-inf eps/255 ball is a genuine ball on the encoder input.
* Text label embeddings are precomputed per language and cached (no grad).
* The image is encoded ONCE; the single image embedding is scored against every
  language's label embeddings -- exactly the shared-encoder setup the proposal
  relies on.
"""
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip

MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"

# ---- class names + translations -------------------------------------------------
# STL-10 classes (order matches torchvision STL10 labels 0..9)
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]
# CIFAR-10 classes (order matches torchvision CIFAR10 labels 0..9)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# Per-language translations, indexed by the English class name.
TRANSLATIONS = {
    "airplane":   {"en": "airplane",   "ko": "비행기",   "es": "avión",      "fr": "avion",      "ja": "飛行機"},
    "automobile": {"en": "automobile", "ko": "자동차",   "es": "automóvil",  "fr": "automobile", "ja": "自動車"},
    "car":        {"en": "car",        "ko": "자동차",   "es": "coche",      "fr": "voiture",    "ja": "車"},
    "bird":       {"en": "bird",       "ko": "새",       "es": "pájaro",     "fr": "oiseau",     "ja": "鳥"},
    "cat":        {"en": "cat",        "ko": "고양이",   "es": "gato",       "fr": "chat",       "ja": "猫"},
    "deer":       {"en": "deer",       "ko": "사슴",     "es": "ciervo",     "fr": "cerf",       "ja": "鹿"},
    "dog":        {"en": "dog",        "ko": "개",       "es": "perro",      "fr": "chien",      "ja": "犬"},
    "frog":       {"en": "frog",       "ko": "개구리",   "es": "rana",       "fr": "grenouille", "ja": "カエル"},
    "horse":      {"en": "horse",      "ko": "말",       "es": "caballo",    "fr": "cheval",     "ja": "馬"},
    "monkey":     {"en": "monkey",     "ko": "원숭이",   "es": "mono",       "fr": "singe",      "ja": "猿"},
    "ship":       {"en": "ship",       "ko": "배",       "es": "barco",      "fr": "bateau",     "ja": "船"},
    "truck":      {"en": "truck",      "ko": "트럭",     "es": "camión",     "fr": "camion",     "ja": "トラック"},
}

# "a photo of a {}" per language (single canonical template).
TEMPLATES = {
    "en": "a photo of a {}.",
    "ko": "{}의 사진.",
    "es": "una foto de un {}.",
    "fr": "une photo d'un {}.",
    "ja": "{}の写真。",
}

LANGS = ["en", "ko", "es", "fr", "ja"]


def load_model(device="cuda"):
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Extract normalization stats from the val preprocessing transform.
    norm = [t for t in preprocess_val.transforms if isinstance(t, T.Normalize)][0]
    mean = torch.tensor(norm.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=device).view(1, 3, 1, 1)
    # Pixel-space preprocessing (NO normalize): resize -> centercrop -> [0,1] tensor
    n_px = 224
    pixel_preprocess = T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        T.ToTensor(),  # -> [0,1], CHW
    ])
    return model, tokenizer, pixel_preprocess, mean, std


def normalize_pixels(x, mean, std):
    return (x - mean) / std


@torch.no_grad()
def build_text_embeddings(model, tokenizer, classnames, device="cuda"):
    """Return dict lang -> [num_classes, dim] L2-normalized text embeddings."""
    out = {}
    for lang in LANGS:
        tmpl = TEMPLATES[lang]
        prompts = [tmpl.format(TRANSLATIONS[c][lang]) for c in classnames]
        toks = tokenizer(prompts).to(device)
        feats = model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        out[lang] = feats
    return out


def encode_image(model, x_pixel, mean, std):
    """x_pixel in [0,1], shape [B,3,224,224]. Returns L2-normalized features (with grad)."""
    x = normalize_pixels(x_pixel, mean, std)
    feats = model.encode_image(x)
    feats = F.normalize(feats, dim=-1)
    return feats


def logits_for(img_feats, txt_feats, logit_scale):
    return logit_scale * img_feats @ txt_feats.t()


def get_logit_scale(model):
    return model.logit_scale.exp().detach()
