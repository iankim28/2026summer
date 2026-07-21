"""
Build all notebooks for the EN/ZH multi-vs-unilingual typographic attack study.

Run: python _build_notebooks.py

Generates:
  multilingual/attack_comparison.ipynb
  multilingual/cam_defense.ipynb
  multilingual/grid_defense.ipynb
  unilingual/attack_comparison.ipynb
  unilingual/cam_defense.ipynb
  unilingual/grid_defense.ipynb
  cost_vs_performance.ipynb

Experiment overview
-------------------
Two attack setups, each placing 2 random typographic boxes per 224x224 CIFAR-10 image:

  Multilingual: Box 0 = EN attack word, Box 1 = ZH attack word
  Unilingual:   Box 0 = EN attack word, Box 1 = EN attack word (same word repeated)

Three defence strategies are tested:
  cam_2mod   - intersect GradCAMs from 2 (model, text-language) combos
  cam_4mod   - intersect GradCAMs from all 4 cross-language combos (multilingual only)
  grid_1patch - greedy 1-patch 4x4 grid occlusion (black-box)
  grid_2patch - greedy 2-patch 4x4 grid occlusion (black-box)

Inference cost (forward passes / image, estimated):
  no_defense   2   (1 EN classify + 1 ZH classify)
  cam_2mod     6   (2 GradCAM backprop + 2 re-classify)
  cam_4mod    10   (4 GradCAM backprop + 2 re-classify)
  grid_1patch 32   (16 patches x 2 models)
  grid_2patch 62   (31 patch evals x 2 models, greedy)
"""
import json, uuid, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
MULTI = HERE / "multilingual"
UNI   = HERE / "unilingual"

for d in [
    MULTI, UNI,
    MULTI / "results" / "attack",
    MULTI / "results" / "cam_2mod" / "cache",
    MULTI / "results" / "cam_4mod" / "cache",
    MULTI / "results" / "grid_1patch",
    MULTI / "results" / "grid_2patch",
    UNI / "results" / "attack",
    UNI / "results" / "cam_2mod" / "cache",
    UNI / "results" / "grid_1patch",
    UNI / "results" / "grid_2patch",
]:
    d.mkdir(parents=True, exist_ok=True)


# ─── notebook helpers ──────────────────────────────────────────────────────────
def md(text):
    return {
        "cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
        "outputs": [], "execution_count": None,
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def write_nb(path, cells):
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}  ({len(cells)} cells)")


# ─── shared code blocks ────────────────────────────────────────────────────────

INSTALL_CODE = """\
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'open_clip_torch', 'transformers', 'datasets',
                'matplotlib', 'Pillow'], check=False)
"""

SHARED_IMPORTS = """\
import importlib, sys, os, platform, random, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

LANGS = ['en', 'zh']
CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
}
TMPL = {'en': 'a photo of a {}.', 'zh': '一张{}的照片。'}
"""

MODEL_CODE = """\
def classify(model, imgs, words, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i:i + batch_size])
        tf  = model.embed_texts(words)
        preds.append((imf @ tf.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)

def classify_conf(model, imgs, words, batch_size=128):
    \"\"\"Returns (predictions, max-cosine-sim confidences).\"\"\"
    preds, confs = [], []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i:i + batch_size])
        tf  = model.embed_texts(words)
        sims = imf @ tf.t()
        preds.append(sims.argmax(-1).cpu().numpy())
        confs.append(sims.max(-1).values.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)

def _clip_feat(out):
    if torch.is_tensor(out): return out
    if getattr(out, 'pooler_output', None) is not None: return out.pooler_output
    raise TypeError(type(out))

class EnCLIP:
    lang = 'en'
    def __init__(self):
        self.m, _, self.pp = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer('ViT-B-32')
    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL['en'].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)

class ZhCLIP:
    lang = 'zh'
    def __init__(self):
        self.m = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16').to(DEVICE).eval()
        self.p = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors='pt').pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL['zh'].format(w) for w in words], padding=True, return_tensors='pt').to(DEVICE)
        out = self.m.get_text_features(input_ids=t['input_ids'], attention_mask=t['attention_mask'],
                                        token_type_ids=t.get('token_type_ids'))
        return F.normalize(_clip_feat(out), dim=-1)

MODEL_CLS = {'en': EnCLIP, 'zh': ZhCLIP}
print('Model classes defined:', list(MODEL_CLS.keys()))
"""

# ── font + drawing helpers shared by all attack code ──────────────────────────
FONT_SHARED_CODE = """\
DISPLAY_SIZE = 224
NUM_BOXES    = 2
FONT_SIZE    = 24
PAD          = 8
_FONT_CACHE  = {}

def _font_paths():
    if platform.system() == 'Windows':
        winfonts = os.path.join(os.environ.get('WINDIR', r'C:\\Windows'), 'Fonts')
        cjk   = os.path.join(winfonts, 'msyh.ttc')
        latin = os.path.join(winfonts, 'arial.ttf')
        if not os.path.exists(latin): latin = cjk
        return (cjk if os.path.exists(cjk) else None,
                latin if os.path.exists(latin) else None)
    for d in ['/usr/share/fonts', '/Library/Fonts', os.path.expanduser('~/.fonts')]:
        for f in ['NotoSansCJK-Regular.ttc', 'NotoSans-Regular.ttf']:
            p = os.path.join(d, f)
            if os.path.exists(p): return p, p
    return None, None

_CJK_FONT, _LAT_FONT = _font_paths()

def _font_for(word):
    return _CJK_FONT if any(ord(c) > 127 for c in word) else _LAT_FONT

def _get_font(fp, size=FONT_SIZE):
    key = (fp or '__default__', size)
    if key not in _FONT_CACHE:
        try:    _FONT_CACHE[key] = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        except: _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

def _random_nonoverlapping_rect(rng, box_w, box_h, placed):
    x_hi = max(0, DISPLAY_SIZE - box_w)
    y_hi = max(0, DISPLAY_SIZE - box_h)
    rect_x, rect_y = 0, 0
    for _ in range(64):
        rect_x = rng.randint(0, x_hi) if x_hi > 0 else 0
        rect_y = rng.randint(0, y_hi) if y_hi > 0 else 0
        rect = (rect_x, rect_y, rect_x + box_w, rect_y + box_h)
        if all(not _rects_overlap(rect, p) for p in placed): return rect
    return (rect_x, rect_y, rect_x + box_w, rect_y + box_h)

def _draw_text_box(draw, word, rect, font):
    rx, ry, rx2, ry2 = rect
    bb = draw.textbbox((0, 0), word, font=font)
    draw.rectangle([rx, ry, rx2, ry2], fill='white')
    draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill='black', font=font)

print('Font paths:', _CJK_FONT, _LAT_FONT)
"""

# ── multilingual attack: Box-0 = EN word, Box-1 = ZH word ─────────────────────
MULTI_ATTACK_CODE = """\
def draw_multilingual_attack(img, en_word, zh_word, img_idx, already_224=False):
    \"\"\"Place English text at box-0 position, Chinese text at box-1 position.\"\"\"
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    placed = []
    for box_i, (word, fp) in enumerate([(en_word, _LAT_FONT), (zh_word, _CJK_FONT)]):
        font = _get_font(fp)
        bb   = draw.textbbox((0, 0), word, font=font)
        bw   = (bb[2] - bb[0]) + 2 * PAD
        bh   = (bb[3] - bb[1]) + PAD + 12
        rng  = random.Random(int(img_idx) * NUM_BOXES + box_i)
        rect = _random_nonoverlapping_rect(rng, bw, bh, placed)
        placed.append(rect)
        _draw_text_box(draw, word, rect, font)
    return img

def build_multilingual_attacked_images(base_imgs, img_indices, n_workers=None):
    \"\"\"Two boxes per image: EN attack word at box-0, ZH attack word at box-1.\"\"\"
    n_workers = n_workers or min(8, os.cpu_count() or 4)
    tasks = [(im, int(k)) for im, k in zip(base_imgs, img_indices)]
    def _one(args):
        im, img_idx = args
        return draw_multilingual_attack(im,
                                        CLASSES['en'][target[img_idx]],
                                        CLASSES['zh'][target[img_idx]],
                                        img_idx, already_224=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one, tasks))

print(f'Multilingual attack helper ready: {NUM_BOXES} boxes @ size {FONT_SIZE} (box-0=EN, box-1=ZH)')
"""

# ── unilingual attack: Box-0 = EN word, Box-1 = EN word (same) ────────────────
UNI_ATTACK_CODE = """\
def draw_word(img, word, img_idx, already_224=False):
    \"\"\"Place the same word in NUM_BOXES non-overlapping positions (all English).\"\"\"
    fp = _font_for(word)
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    font  = _get_font(fp)
    draw  = ImageDraw.Draw(img)
    bb    = draw.textbbox((0, 0), word, font=font)
    bw    = (bb[2] - bb[0]) + 2 * PAD
    bh    = (bb[3] - bb[1]) + PAD + 12
    placed = []
    for box_i in range(NUM_BOXES):
        rng  = random.Random(int(img_idx) * NUM_BOXES + box_i)
        rect = _random_nonoverlapping_rect(rng, bw, bh, placed)
        placed.append(rect)
        _draw_text_box(draw, word, rect, font)
    return img

def build_attacked_images(base_imgs, img_indices, attack_lang, n_workers=None):
    \"\"\"Unilingual attack: same word in both boxes.\"\"\"
    words     = [CLASSES[attack_lang][target[int(k)]] for k in img_indices]
    n_workers = n_workers or min(8, os.cpu_count() or 4)
    def _one(args):
        im, word, img_idx = args
        return draw_word(im, word, img_idx=int(img_idx), already_224=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one, zip(base_imgs, words, img_indices)))

print(f'Unilingual attack helper ready: {NUM_BOXES} boxes @ size {FONT_SIZE} (both EN)')
"""

DATA_LOAD_CODE = """\
hf = load_dataset('uoft-cs/cifar10', split='test')
label_key = 'label' if 'label' in hf.column_names else 'labels'
image_key = 'img'   if 'img'   in hf.column_names else 'image'

_sample_path = '../../../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json'
with open(_sample_path, encoding='utf-8') as f:
    _saved = json.load(f)

idx  = _saved['idx']
rows = hf.select(idx)
true = np.array(rows[label_key])
assert len(idx) == 1000
assert np.array_equal(true, np.array(_saved['true']))
assert all((true == c).sum() == 100 for c in range(10))

rng    = random.Random(0)
target = np.array([rng.choice([c for c in range(10) if c != int(true[k])])
                   for k in range(len(idx))])

clean     = [im.convert('RGB') for im in rows[image_key]]
print('Upscaling clean images to 224px...', end=' ', flush=True)
t0 = time.time()
clean_224 = [im.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC) for im in clean]
print(f'{time.time()-t0:.1f}s')

all_idx  = np.arange(len(clean))
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
print(f'Loaded {len(clean)} images; tune subset = {len(tune_idx)}')
"""

LOAD_MODELS_CODE = """\
models = {}
for lang, cls in MODEL_CLS.items():
    t0 = time.time()
    print(f'Loading {lang}...', end=' ', flush=True)
    models[lang] = cls()
    print(f'{time.time()-t0:.1f}s')

# Standard text embeddings: each model with its own language
TEXT_EMB = {lang: models[lang].embed_texts(CLASSES[lang]).detach() for lang in LANGS}

clean_preds = {lang: classify(models[lang], clean_224, CLASSES[lang]) for lang in LANGS}
clean_acc   = {lang: float((clean_preds[lang] == true).mean()) for lang in LANGS}
print('Clean acc:', {k: f'{100*v:.1f}%' for k, v in clean_acc.items()})
"""

GRADCAM_STANDARD_CODE = """\
def _norm_cam(cam):
    cam = cam.relu() if isinstance(cam, torch.Tensor) else np.maximum(cam, 0)
    cam = cam.detach().cpu().numpy() if isinstance(cam, torch.Tensor) else cam
    cam = cam - cam.min()
    mx  = cam.max()
    return cam / mx if mx > 0 else cam

def _cam_from_conv(act, grad):
    w = grad.mean(dim=(2, 3), keepdim=True)
    return _norm_cam((w * act).sum(dim=1).squeeze(0))

def gradcam_en(pil_img, target_idx):
    wrapper = models['en']; acts = {}
    def hook(_m, _i, out): out.retain_grad(); acts['v'] = out
    handle = wrapper.m.visual.conv1.register_forward_hook(hook)
    x        = wrapper.pp(pil_img).unsqueeze(0).to(DEVICE)
    feat     = wrapper.m.visual(x)
    img_feat = F.normalize(feat, dim=-1)
    score    = (img_feat @ TEXT_EMB['en'][target_idx:target_idx+1].T).squeeze()
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam

def gradcam_zh(pil_img, target_idx):
    wrapper = models['zh']; acts = {}
    patch  = wrapper.m.vision_model.embeddings.patch_embedding
    def hook(_m, _i, out): out.retain_grad(); acts['v'] = out
    handle = patch.register_forward_hook(hook)
    pv     = wrapper.p(images=[pil_img], return_tensors='pt').pixel_values.to(DEVICE)
    out    = wrapper.m.get_image_features(pixel_values=pv)
    img_feat = F.normalize(_clip_feat(out), dim=-1)
    score  = (img_feat @ TEXT_EMB['zh'][target_idx:target_idx+1].T).squeeze()
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam

GRADCAM_FN = {'en': gradcam_en, 'zh': gradcam_zh}
print('Standard GradCAM helpers ready.')
"""

# ── extended text embeddings for 4-mod and unilingual cam defense ─────────────
CROSS_TEXT_EMBS_CODE = """\
# Cross-language text embeddings:
#   TEXT_EMBS[('en','zh')] = EN model's text encoder applied to ZH class names
#   TEXT_EMBS[('zh','en')] = ZH model's text encoder applied to EN class names
# These are "cross-lingual probes" - each model sees the other language's vocabulary.

with torch.no_grad():
    # EN model encoding ZH class names (EN tokenizer processes CJK glyphs as UNK/subwords)
    _t_zh_via_en = models['en'].tok(
        [TMPL['en'].format(w) for w in CLASSES['zh']]
    ).to(DEVICE)
    TEXT_EMB_EN_ZH = F.normalize(models['en'].m.encode_text(_t_zh_via_en), dim=-1).detach()

    # ZH model encoding EN class names
    _t_en_via_zh = models['zh'].p(
        text=[TMPL['zh'].format(w) for w in CLASSES['en']],
        padding=True, return_tensors='pt',
    ).to(DEVICE)
    _out_en_via_zh = models['zh'].m.get_text_features(
        input_ids=_t_en_via_zh['input_ids'],
        attention_mask=_t_en_via_zh['attention_mask'],
        token_type_ids=_t_en_via_zh.get('token_type_ids'),
    )
    TEXT_EMB_ZH_EN = F.normalize(_clip_feat(_out_en_via_zh), dim=-1).detach()

# Unified lookup: (model_lang, text_lang) -> text_embedding
TEXT_EMBS = {
    ('en', 'en'): TEXT_EMB['en'],     # standard EN
    ('en', 'zh'): TEXT_EMB_EN_ZH,    # EN model, ZH class names
    ('zh', 'en'): TEXT_EMB_ZH_EN,    # ZH model, EN class names
    ('zh', 'zh'): TEXT_EMB['zh'],     # standard ZH
}
print('Cross-language text embeddings computed.')
"""

GRADCAM_GENERAL_CODE = """\
def gradcam_en_with_emb(pil_img, text_emb, target_idx=None):
    \"\"\"GradCAM using EN model, scored against text_emb[target_idx].\"\"\"
    wrapper = models['en']; acts = {}
    def hook(_m, _i, out): out.retain_grad(); acts['v'] = out
    handle   = wrapper.m.visual.conv1.register_forward_hook(hook)
    x        = wrapper.pp(pil_img).unsqueeze(0).to(DEVICE)
    feat     = wrapper.m.visual(x)
    img_feat = F.normalize(feat, dim=-1)
    sims     = (img_feat @ text_emb.T).squeeze(0)
    if target_idx is None:
        target_idx = int(sims.detach().argmax().item())
    score = sims[target_idx]
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam, target_idx

def gradcam_zh_with_emb(pil_img, text_emb, target_idx=None):
    \"\"\"GradCAM using ZH model, scored against text_emb[target_idx].\"\"\"
    wrapper = models['zh']; acts = {}
    patch   = wrapper.m.vision_model.embeddings.patch_embedding
    def hook(_m, _i, out): out.retain_grad(); acts['v'] = out
    handle   = patch.register_forward_hook(hook)
    pv       = wrapper.p(images=[pil_img], return_tensors='pt').pixel_values.to(DEVICE)
    out      = wrapper.m.get_image_features(pixel_values=pv)
    img_feat = F.normalize(_clip_feat(out), dim=-1)
    sims     = (img_feat @ text_emb.T).squeeze(0)
    if target_idx is None:
        target_idx = int(sims.detach().argmax().item())
    score = sims[target_idx]
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam, target_idx

def get_n_cams(pil_img, combos):
    \"\"\"Compute CAMs for all (model_lang, text_lang) combos. Returns list of cams.\"\"\"
    cams = []
    for ml, tl in combos:
        emb = TEXT_EMBS[(ml, tl)]
        fn  = gradcam_en_with_emb if ml == 'en' else gradcam_zh_with_emb
        cam, _ = fn(pil_img, emb)
        cams.append(cam)
    return cams

print('Generalised GradCAM helpers ready.')
"""

# ── CAM masking helpers ────────────────────────────────────────────────────────
CAM_MASKING_CODE = """\
def align_cam(cam, size=DISPLAY_SIZE):
    return np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ) / 255.0

def n_cam_intersection(*cams):
    \"\"\"Elementwise min of N CAMs after resizing to DISPLAY_SIZE.\"\"\"
    return np.minimum.reduce([align_cam(c) for c in cams])

def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        m = (pad[:-2,:-2]|pad[:-2,1:-1]|pad[:-2,2:]|
             pad[1:-1,:-2]|pad[1:-1,1:-1]|pad[1:-1,2:]|
             pad[2:,:-2]  |pad[2:,1:-1]  |pad[2:,2:])
    return m

def cam_to_mask(saliency, threshold=0.85, dilate=3):
    thr  = np.percentile(saliency, threshold * 100)
    mask = saliency >= thr
    if dilate > 0: mask = dilate_mask(mask, iterations=dilate)
    return mask

def apply_mask(pil_img, mask, fill='mean_nonmask'):
    pil_img = pil_img.convert('RGB')
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    if mask.shape != (h, w):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 127
    out = arr.copy()
    m   = mask.astype(bool)
    if fill == 'mean_nonmask':
        bg = ~m
        mean_color = arr[bg].mean(0) if bg.any() else arr.reshape(-1, 3).mean(0)
        out[m] = mean_color
    return Image.fromarray(out.astype(np.uint8))

def overlay_cam(pil_img, cam, alpha=0.50):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    heat  = cm.jet(np.array(cam_img) / 255.0)[:, :, :3]
    base  = np.array(pil_img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE))).astype(np.float32) / 255.0
    blend = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((blend * 255).astype(np.uint8))

def mask_overlay(pil_img, mask, alpha=0.45):
    arr = np.array(pil_img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE))).astype(np.float32)
    red = np.zeros_like(arr); red[:, :, 0] = 255
    m   = mask.astype(np.float32)[..., None]
    return Image.fromarray((arr * (1 - alpha * m) + red * (alpha * m)).astype(np.uint8))

print('CAM masking helpers ready.')
"""

# ── 4-CAM cache ────────────────────────────────────────────────────────────────
CAM_CACHE_4MOD_CODE = """\
COMBOS_2MOD = [('en','en'), ('zh','zh')]
COMBOS_4MOD = [('en','en'), ('en','zh'), ('zh','en'), ('zh','zh')]

def _cache_path_4mod(condition, n):
    return os.path.join(CACHE_DIR_4MOD, f'cams4_{condition}_n{n}.npz')

def compute_and_cache_cams(condition, indices, combos=COMBOS_4MOD):
    n     = len(indices)
    nmod  = len(combos)
    label = '_'.join(f'{m}{t}' for m, t in combos)
    cfile = os.path.join(CACHE_DIR_4MOD, f'cams_{label}_{condition}_n{n}.npz')
    keys  = [f'cam_{m}_{t}' for m, t in combos]

    if os.path.exists(cfile):
        data = np.load(cfile, allow_pickle=True)
        print(f'Loaded cache {os.path.basename(cfile)}')
        return {k: data[k] for k in keys}, np.array(data['indices'])

    # Build attacked images for this condition
    if condition == 'multi_attack':
        imgs = [attacked_imgs[i] for i in indices]
    elif condition == 'uni_attack':
        imgs = [attacked_imgs[i] for i in indices]
    elif condition == 'clean':
        imgs = [clean_224[i] for i in indices]
    else:
        raise ValueError(f'Unknown condition: {condition}')

    cam_lists = {k: [] for k in keys}
    t0 = time.time()
    for j, img in enumerate(imgs):
        for (ml, tl), k in zip(combos, keys):
            emb = TEXT_EMBS[(ml, tl)]
            fn  = gradcam_en_with_emb if ml == 'en' else gradcam_zh_with_emb
            cam, _ = fn(img, emb)
            cam_lists[k].append(cam)
        if (j + 1) % 50 == 0:
            print(f'  CAM {j+1}/{n} [{time.time()-t0:.1f}s]')

    data_to_save = {k: np.stack(cam_lists[k]) for k in keys}
    data_to_save['indices'] = np.array(indices)
    np.savez(cfile, **data_to_save)
    print(f'Saved cache {os.path.basename(cfile)} [{time.time()-t0:.1f}s]')
    return {k: data_to_save[k] for k in keys}, np.array(indices)

print('4-CAM cache helpers ready.')
"""

# ── threshold sweep ────────────────────────────────────────────────────────────
THRESHOLD_SWEEP_CODE = """\
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

def eval_masked(masked_imgs, indices, label=''):
    t   = true[indices]
    tgt = target[indices]
    out = {}
    for ml in LANGS:
        p = classify(models[ml], masked_imgs, CLASSES[ml])
        out[ml] = {'acc': float((p == t).mean()), 'asr': float((p == tgt).mean())}
    return out

def sweep_threshold(imgs, cam_data_by_cond, combos, tune_idx, setup_label):
    \"\"\"Sweep thresholds on tune_idx and return best threshold + sweep rows.\"\"\"
    sweep_rows = []
    for cond, (cam_data, _) in cam_data_by_cond.items():
        cond_imgs = [imgs[i] for i in tune_idx]
        base_acc  = {ml: float((preds_attacked[ml][tune_idx] == true[tune_idx]).mean())
                     for ml in LANGS}
        for thr in THRESHOLDS:
            sal_list = []
            for j in range(len(tune_idx)):
                cams_j = [cam_data[f'cam_{m}_{t}'][j] for m, t in combos]
                sal_list.append(n_cam_intersection(*cams_j))
            masks   = [cam_to_mask(s, threshold=thr, dilate=3) for s in sal_list]
            masked  = [apply_mask(img, m) for img, m in zip(cond_imgs, masks)]
            res     = eval_masked(masked, tune_idx)
            cov     = float(np.mean([m.mean() for m in masks]))
            for ml in LANGS:
                sweep_rows.append({
                    'setup': setup_label, 'condition': cond, 'combos': str(combos),
                    'model': ml, 'threshold': thr, 'coverage': cov,
                    'baseline_acc': base_acc[ml],
                    'masked_acc':   res[ml]['acc'],
                    'masked_asr':   res[ml]['asr'],
                })

    # Select best threshold: maximise masked_acc for primary condition, EN model
    cond0 = list(cam_data_by_cond.keys())[0]
    rows_primary = [r for r in sweep_rows if r['condition'] == cond0 and r['model'] == 'en']
    best_row = max(rows_primary, key=lambda r: r['masked_acc'])
    BEST_THR = best_row['threshold']
    print(f'Best threshold ({setup_label}): {BEST_THR}  '
          f'baseline={100*best_row["baseline_acc"]:.1f}% -> masked={100*best_row["masked_acc"]:.1f}%')
    return BEST_THR, sweep_rows

print('Threshold sweep helpers ready.')
"""

# ── full CAM evaluation ────────────────────────────────────────────────────────
FULL_CAM_EVAL_CODE = """\
def full_cam_eval(attacked_imgs, cam_data, combos, best_thr, all_idx, setup_label, n_mods):
    \"\"\"Evaluate CAM defence on full 1000-image set.\"\"\"
    results = {'setup': setup_label, 'method': f'cam_{n_mods}mod',
               'attack': ATTACK_LABEL, 'n_images': len(all_idx),
               'best_threshold': best_thr, 'combos': str(combos),
               'inference_cost': 2 + 2 * n_mods,
               'clean_acc': clean_acc,
               'baseline_acc': {ml: float((preds_attacked[ml] == true).mean()) for ml in LANGS},
               'baseline_asr': {ml: float((preds_attacked[ml] == target).mean()) for ml in LANGS},
               'defense': {}}

    sal_list = []
    for j in range(len(all_idx)):
        cams_j = [cam_data[f'cam_{m}_{t}'][j] for m, t in combos]
        sal_list.append(n_cam_intersection(*cams_j))

    masks       = [cam_to_mask(s, threshold=best_thr, dilate=3) for s in sal_list]
    masked_imgs = [apply_mask(img, m) for img, m in zip(attacked_imgs, masks)]

    for ml in LANGS:
        base_p = preds_attacked[ml]
        new_p  = classify(models[ml], masked_imgs, CLASSES[ml])
        wrong      = base_p != true
        recovered  = wrong & (new_p == true)
        results['defense'][ml] = {
            'acc':           float((new_p == true).mean()),
            'asr':           float((new_p == target).mean()),
            'recovery_rate': float(recovered.sum() / wrong.sum()) if wrong.any() else 0.0,
        }

    coverage = float(np.mean([m.mean() for m in masks]))
    results['coverage_mean'] = coverage

    # Clean-image degradation
    clean_sal   = []
    clean_cd, _ = compute_and_cache_cams('clean', all_idx, combos=combos)
    for j in range(len(all_idx)):
        cams_j = [clean_cd[f'cam_{m}_{t}'][j] for m, t in combos]
        clean_sal.append(n_cam_intersection(*cams_j))
    clean_masks   = [cam_to_mask(s, threshold=best_thr, dilate=3) for s in clean_sal]
    clean_masked  = [apply_mask(img, m) for img, m in zip(clean_224, clean_masks)]
    results['clean_degradation'] = {}
    for ml in LANGS:
        cp  = classify(models[ml], clean_masked, CLASSES[ml])
        results['clean_degradation'][ml] = {
            'baseline_acc': clean_acc[ml],
            'masked_acc':   float((cp == true).mean()),
            'delta_acc':    float((cp == true).mean()) - clean_acc[ml],
        }

    # Summary mean (average over EN and ZH)
    results['defense_acc_mean'] = float(np.mean([results['defense'][ml]['acc'] for ml in LANGS]))
    results['defense_asr_mean'] = float(np.mean([results['defense'][ml]['asr'] for ml in LANGS]))
    return results, masked_imgs, masks
"""

# ── grid defence ───────────────────────────────────────────────────────────────
GRID_HELPERS_CODE = """\
GRID_ROWS = GRID_COLS = 4
PATCH_H   = PATCH_W   = DISPLAY_SIZE // GRID_ROWS   # 56 px
PATCHES   = [
    (c * PATCH_W, r * PATCH_H, (c + 1) * PATCH_W, (r + 1) * PATCH_H)
    for r in range(GRID_ROWS) for c in range(GRID_COLS)
]   # 16 non-overlapping patches

def occlude_rect(arr, rect, fill_color):
    out = arr.copy()
    x0, y0, x1, y1 = rect
    out[y0:y1, x0:x1] = fill_color
    return out

def score_candidates(candidates):
    \"\"\"Return per-candidate score = mean max-cosine-sim across EN and ZH models.\"\"\"
    scores = np.zeros(len(candidates))
    for ml in LANGS:
        _, confs = classify_conf(models[ml], candidates, CLASSES[ml])
        scores += confs
    return scores / len(LANGS)

print(f'Grid defence: {GRID_ROWS}x{GRID_COLS} grid = {len(PATCHES)} patches of {PATCH_H}x{PATCH_W}px')
"""

GRID_RUN_CODE = """\
def run_grid_1patch(imgs):
    \"\"\"Find and apply the single best patch occlusion for each image.\"\"\"
    result_imgs, best_patches = [], []
    t0 = time.time()
    for j, img in enumerate(imgs):
        arr        = np.array(img.convert('RGB'))
        fill_color = arr.reshape(-1, 3).mean(0).astype(np.uint8)
        candidates = [Image.fromarray(occlude_rect(arr, rect, fill_color)) for rect in PATCHES]
        scores     = score_candidates(candidates)
        best_pi    = int(scores.argmax())
        best_patches.append(best_pi)
        result_imgs.append(candidates[best_pi])
        if (j + 1) % 100 == 0:
            print(f'  Grid-1patch {j+1}/{len(imgs)} [{time.time()-t0:.1f}s]')
    return result_imgs, best_patches

def run_grid_2patch_greedy(imgs, first_patches):
    \"\"\"Given 1st-patch results, greedily find the best 2nd patch.\"\"\"
    result_imgs, second_patches = [], []
    t0 = time.time()
    for j, (img, fp) in enumerate(zip(imgs, first_patches)):
        arr        = np.array(img.convert('RGB'))
        fill_color = arr.reshape(-1, 3).mean(0).astype(np.uint8)
        arr1       = occlude_rect(arr, PATCHES[fp], fill_color)
        remain     = [i for i in range(len(PATCHES)) if i != fp]
        candidates = [Image.fromarray(occlude_rect(arr1, PATCHES[pi], fill_color)) for pi in remain]
        scores     = score_candidates(candidates)
        best_local = int(scores.argmax())
        second_pi  = remain[best_local]
        second_patches.append(second_pi)
        result_imgs.append(candidates[best_local])
        if (j + 1) % 100 == 0:
            print(f'  Grid-2patch {j+1}/{len(imgs)} [{time.time()-t0:.1f}s]')
    return result_imgs, second_patches

print('Grid defence run helpers ready.')
"""

GRID_EVAL_CODE = """\
def eval_grid(defended_imgs, all_idx, method_label, inference_cost):
    t    = true[all_idx]
    tgt  = target[all_idx]
    out  = {'setup': SETUP_LABEL, 'method': method_label, 'attack': ATTACK_LABEL,
            'n_images': len(all_idx), 'inference_cost': inference_cost,
            'clean_acc': clean_acc,
            'baseline_acc': {ml: float((preds_attacked[ml] == true).mean()) for ml in LANGS},
            'baseline_asr': {ml: float((preds_attacked[ml] == target).mean()) for ml in LANGS},
            'defense': {}}
    for ml in LANGS:
        base_p = preds_attacked[ml]
        new_p  = classify(models[ml], defended_imgs, CLASSES[ml])
        wrong  = base_p != t
        out['defense'][ml] = {
            'acc':           float((new_p == t).mean()),
            'asr':           float((new_p == tgt).mean()),
            'recovery_rate': float((wrong & (new_p == t)).sum() / wrong.sum()) if wrong.any() else 0.0,
        }
    out['defense_acc_mean'] = float(np.mean([out['defense'][ml]['acc'] for ml in LANGS]))
    out['defense_asr_mean'] = float(np.mean([out['defense'][ml]['asr'] for ml in LANGS]))
    return out

print('Grid evaluation helper ready.')
"""

COST_PERF_NB_CODE = """\
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath('cost_vs_performance.ipynb')) if '__file__' not in dir() else os.path.dirname(os.path.abspath(__file__))

# Collect all result JSONs
patterns = [
    os.path.join('multilingual', 'results', '**', '*.json'),
    os.path.join('unilingual',   'results', '**', '*.json'),
]
all_results = []
for pat in patterns:
    for path in glob.glob(pat, recursive=True):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        all_results.append(d)
        print(f'Loaded {path}: method={d.get("method")}, setup={d.get("setup")}')
print(f'Total result files: {len(all_results)}')
"""

COST_PERF_PLOT_CODE = """\
# Inference cost table (forward passes per image)
COST_TABLE = {
    'no_defense':   2,
    'cam_2mod':     6,
    'cam_4mod':    10,
    'grid_1patch': 32,
    'grid_2patch': 62,
}

METHOD_ORDER = ['no_defense', 'cam_2mod', 'cam_4mod', 'grid_1patch', 'grid_2patch']
SETUP_COLORS = {'multilingual': 'C0', 'unilingual': 'C1'}
METHOD_MARKERS = {
    'no_defense':   'x',
    'cam_2mod':     'o',
    'cam_4mod':     's',
    'grid_1patch':  '^',
    'grid_2patch':  'D',
}

def get_perf(d):
    \"\"\"Mean post-defence accuracy across EN and ZH models.\"\"\"
    if d.get('method') == 'no_defense':
        acc = d.get('attacked_acc', d.get('defense_acc', {}))
    else:
        acc = d.get('defense', d.get('defense_acc', {}))
        if isinstance(acc, dict) and 'en' not in acc:
            # cam defence stores nested dict under 'defense'
            acc_vals = [v['acc'] for v in acc.values() if isinstance(v, dict) and 'acc' in v]
            return float(np.mean(acc_vals)) * 100 if acc_vals else None
    if isinstance(acc, dict):
        vals = []
        for v in acc.values():
            if isinstance(v, (int, float)):
                vals.append(v)
            elif isinstance(v, dict) and 'acc' in v:
                vals.append(v['acc'])
        if vals: return float(np.mean(vals)) * 100
    return None

fig, ax = plt.subplots(figsize=(8, 5))

for setup, color in SETUP_COLORS.items():
    xs, ys, labels = [], [], []
    for d in all_results:
        if d.get('setup') != setup: continue
        method = d.get('method', 'unknown')
        cost   = COST_TABLE.get(method)
        perf   = get_perf(d)
        if cost is None or perf is None: continue
        xs.append(cost); ys.append(perf); labels.append(method)
    if not xs: continue
    # Sort by cost
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; labels = [labels[i] for i in order]
    ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=1.2)
    for x, y, lbl in zip(xs, ys):
        marker = METHOD_MARKERS.get(lbl, 'o')
        ax.scatter(x, y, color=color, marker=marker, s=80, zorder=5,
                   label=f'{setup} / {lbl}')
        ax.annotate(lbl.replace('_', ' '), (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5)

# Baseline (no-defense) reference line
baseline_results = [d for d in all_results if d.get('method') == 'no_defense']
if baseline_results:
    baseline_perfs = [p for d in baseline_results for p in [get_perf(d)] if p is not None]
    if baseline_perfs:
        ax.axhline(float(np.mean(baseline_perfs)), color='grey', linestyle=':', alpha=0.7, label='no-defense baseline')

ax.set_xlabel('Inference cost (forward passes per image)', fontsize=11)
ax.set_ylabel('Mean post-defence accuracy (%)', fontsize=11)
ax.set_title('Inference Cost vs. Defence Performance\\n(multilingual vs unilingual typographic attack)', fontsize=12)
ax.grid(True, alpha=0.3)

# Legend: one entry per (setup, method) combination, no duplicates
handles, lbls = ax.get_legend_handles_labels()
seen = set(); unique = [(h, l) for h, l in zip(handles, lbls) if l not in seen and not seen.add(l)]
ax.legend([h for h, _ in unique], [l for _, l in unique], fontsize=8, loc='lower right', ncol=2)

plt.tight_layout()
plt.savefig('cost_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved -> cost_vs_performance.png')
"""


# ══════════════════════════════════════════════════════════════════════════════
# Notebook builders
# ══════════════════════════════════════════════════════════════════════════════

def _attack_comparison_cells(setup, results_dir, attack_build_code, attack_label,
                              attack_description):
    """Shared structure for multilingual/attack_comparison and unilingual/attack_comparison."""
    cells = [
        md(f"""\
# {attack_label} Typographic Attack — Classification Performance

{attack_description}

Evaluates classification accuracy and attack success rate for **both English and Chinese CLIP models**
on CIFAR-10 (balanced 1000-image sample, 100/class).

Results saved to `{results_dir}/`.
"""),
        code(INSTALL_CODE),
        code(SHARED_IMPORTS + f"\nRESULTS_DIR = '{results_dir}'\nos.makedirs(RESULTS_DIR, exist_ok=True)\n"
             + f"SETUP_LABEL  = '{setup}'\nATTACK_LABEL = '{attack_label.lower().replace(' ','_')}'"),
        md("## 1. Model definitions"),
        code(MODEL_CODE),
        md("## 2. Attack helpers"),
        code(FONT_SHARED_CODE),
        code(attack_build_code),
        md("## 3. Dataset"),
        code(DATA_LOAD_CODE),
        md("## 4. Load models"),
        code(LOAD_MODELS_CODE),
        md("## 5. Build attacked images & classify"),
        code(f"""\
print('Building {attack_label} attacked images...')
t0 = time.time()
attacked_imgs = {'build_multilingual_attacked_images(clean_224, all_idx)' if setup == 'multilingual' else "build_attacked_images(clean_224, all_idx, 'en')"}
print(f'Done in {{time.time()-t0:.1f}}s')

preds_attacked = {{}}
for ml in LANGS:
    preds_attacked[ml] = classify(models[ml], attacked_imgs, CLASSES[ml])

acc_atk  = {{ml: float((preds_attacked[ml] == true).mean())   for ml in LANGS}}
asr_atk  = {{ml: float((preds_attacked[ml] == target).mean()) for ml in LANGS}}

print('\\nAttack results:')
print(f'  Clean acc    EN={{100*clean_acc["en"]:.1f}}%  ZH={{100*clean_acc["zh"]:.1f}}%')
print(f'  Attacked acc EN={{100*acc_atk["en"]:.1f}}%  ZH={{100*acc_atk["zh"]:.1f}}%')
print(f'  ASR          EN={{100*asr_atk["en"]:.1f}}%  ZH={{100*asr_atk["zh"]:.1f}}%')
"""),
        md("## 6. Per-class accuracy breakdown"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
class_names = CLASSES['en']
x = np.arange(len(class_names))
for ax, ml in zip(axes, LANGS):
    clean_pc  = np.array([(clean_preds[ml][true == c] == c).mean() for c in range(10)])
    attack_pc = np.array([(preds_attacked[ml][true == c] == c).mean() for c in range(10)])
    ax.bar(x - 0.2, clean_pc  * 100, 0.4, label='clean',   color='steelblue')
    ax.bar(x + 0.2, attack_pc * 100, 0.4, label='attacked', color='tomato')
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=8)
    ax.set_title(f'{ml.upper()} model — per-class accuracy')
    ax.set_ylabel('%'); ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Clean vs attacked per-class accuracy', fontsize=12)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved -> per_class_accuracy.png')
"""),
        md("## 7. GradCAM heatmaps (sample)"),
        code(GRADCAM_STANDARD_CODE),
        code("""\
TEXT_EMB  # already defined in load-models cell

# Visualise GradCAM on 5 samples (one per row: attacked, EN cam, ZH cam)
sample_pos = [int(np.where(true == c)[0][0]) for c in range(5)]
fig, axes  = plt.subplots(5, 3, figsize=(9, 17))
col_titles = ['Attacked', 'EN GradCAM', 'ZH GradCAM']
for ax, t_title in zip(axes[0], col_titles):
    ax.set_title(t_title, fontsize=10, fontweight='bold')

for row_i, pos in enumerate(sample_pos):
    img   = attacked_imgs[pos]
    pred_en = int(preds_attacked['en'][pos])
    pred_zh = int(preds_attacked['zh'][pos])
    cam_en  = gradcam_en(img, pred_en)
    cam_zh  = gradcam_zh(img, pred_zh)
    for col_i, panel in enumerate([img, overlay_cam(img, cam_en), overlay_cam(img, cam_zh)]):
        ax = axes[row_i, col_i]
        ax.imshow(panel); ax.axis('off')
    axes[row_i, 0].set_ylabel(CLASSES['en'][true[pos]], fontsize=8, rotation=0, labelpad=36, va='center')

plt.suptitle('GradCAM on attacked images', fontsize=12)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/gradcam_heatmaps.png', dpi=140, bbox_inches='tight')
plt.close()
print('Saved -> gradcam_heatmaps.png')
"""),
        md("## 8. Sample visualisation"),
        code(f"""\
sample_idx = [int(np.where(true == c)[0][0]) for c in range(5)]
fig, axes  = plt.subplots(5, 2, figsize=(7, 17))
for row_i, pos in enumerate(sample_idx):
    for col_i, img in enumerate([clean_224[pos], attacked_imgs[pos]]):
        axes[row_i, col_i].imshow(img); axes[row_i, col_i].axis('off')
        axes[row_i, col_i].set_title(['Clean', 'Attacked'][col_i], fontsize=9)
    axes[row_i, 0].set_ylabel(CLASSES['en'][true[pos]], fontsize=8, rotation=0, labelpad=36, va='center')
plt.suptitle('{attack_label} sample visualisation', fontsize=11)
plt.tight_layout()
plt.savefig(f'{{RESULTS_DIR}}/sample_viz.png', dpi=140, bbox_inches='tight')
plt.close()
print('Saved -> sample_viz.png')
"""),
        md("## 9. Save results JSON"),
        code(f"""\
results = {{
    'setup':          SETUP_LABEL,
    'method':         'no_defense',
    'attack':         ATTACK_LABEL,
    'sample':         'CIFAR10_BALANCED_1000_SAMPLE',
    'n_images':       len(all_idx),
    'inference_cost': 2,
    'clean_acc':      clean_acc,
    'attacked_acc':   acc_atk,
    'attacked_asr':   asr_atk,
    'attacked_acc_mean': float(np.mean(list(acc_atk.values()))),
}}
out_path = f'{{RESULTS_DIR}}/confusion_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Saved -> {{out_path}}')
"""),
    ]
    return cells


def _cam_defense_cells_multi():
    """multilingual/cam_defense.ipynb — 2-mod and 4-mod."""
    cells = [
        md("""\
# CAM Intersection Defence — Multilingual Attack (2-mod & 4-mod)

Defends against a **multilingual typographic attack** (EN text at box-0, ZH text at box-1)
using GradCAM intersection masking.

Two defence variants:
- **2-mod**: intersect GradCAM(EN model, EN text) ∩ GradCAM(ZH model, ZH text)
- **4-mod**: intersect all 4 cross-language combos (EN-EN ∩ EN-ZH ∩ ZH-EN ∩ ZH-ZH)

Results saved to `results/cam_2mod/` and `results/cam_4mod/`.
"""),
        code(INSTALL_CODE),
        code(SHARED_IMPORTS + """\

RESULTS_DIR_2MOD = 'results/cam_2mod'
RESULTS_DIR_4MOD = 'results/cam_4mod'
CACHE_DIR_4MOD   = os.path.join(RESULTS_DIR_4MOD, 'cache')
CACHE_DIR_2MOD   = os.path.join(RESULTS_DIR_2MOD, 'cache')
for d in [RESULTS_DIR_2MOD, RESULTS_DIR_4MOD, CACHE_DIR_4MOD, CACHE_DIR_2MOD]:
    os.makedirs(d, exist_ok=True)

SETUP_LABEL  = 'multilingual'
ATTACK_LABEL = 'multilingual'
"""),
        md("## 1. Model definitions"),
        code(MODEL_CODE),
        md("## 2. Attack helpers"),
        code(FONT_SHARED_CODE),
        code(MULTI_ATTACK_CODE),
        md("## 3. Dataset"),
        code(DATA_LOAD_CODE),
        md("## 4. Load models + build multilingual attack"),
        code(LOAD_MODELS_CODE),
        code("""\
print('Building multilingual attacked images...')
t0 = time.time()
attacked_imgs = build_multilingual_attacked_images(clean_224, all_idx)
print(f'Done in {time.time()-t0:.1f}s')

preds_attacked_2d = {}   # preds_attacked_2d[ml] for EN/ZH model on multilingual attack
for ml in LANGS:
    preds_attacked_2d[ml] = classify(models[ml], attacked_imgs, CLASSES[ml])
baseline_acc = {ml: float((preds_attacked_2d[ml] == true).mean()) for ml in LANGS}
baseline_asr = {ml: float((preds_attacked_2d[ml] == target).mean()) for ml in LANGS}
print('Baseline acc:', {k: f'{100*v:.1f}%' for k, v in baseline_acc.items()})
print('Baseline ASR:', {k: f'{100*v:.1f}%' for k, v in baseline_asr.items()})
"""),
        md("""\
## 5. GradCAM + masking helpers

EN ViT-B/32 and ZH ViT-B/16 produce CAMs at different patch resolutions (~7×7 vs ~14×14).
Both are resized to 224×224 before intersection.
"""),
        code(GRADCAM_STANDARD_CODE),
        code(CAM_MASKING_CODE),
        md("## 6. Cross-language text embeddings (for 4-mod)"),
        code(CROSS_TEXT_EMBS_CODE),
        code(GRADCAM_GENERAL_CODE),
        md("## 7. 4-CAM cache"),
        code(CAM_CACHE_4MOD_CODE),
        md("## 8. Threshold sweep on 100-image tune subset"),
        code("""\
# Alias for the cache to work with both 2-mod and 4-mod
cam_2mod_cache = {}
cam_4mod_cache = {}
for cond, label in [('multi_attack', 'multilingual attack'), ('clean', 'clean')]:
    imgs_cond = attacked_imgs if cond == 'multi_attack' else clean_224
    print(f'Computing 4-mod CAMs for {label} (tune subset)...')
    cd4, _ = compute_and_cache_cams(cond, tune_idx, combos=COMBOS_4MOD)
    cam_4mod_cache[cond] = cd4
    # 2-mod uses only the 'en_en' and 'zh_zh' cams
    cam_2mod_cache[cond] = {k: cd4[k] for k in ['cam_en_en', 'cam_zh_zh']}
"""),
        code("""\
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

def sweep_one(cam_data, combos, imgs_tune, label):
    rows = []
    base = {ml: float((preds_attacked_2d[ml][tune_idx] == true[tune_idx]).mean()) for ml in LANGS}
    for thr in THRESHOLDS:
        sal_list = []
        for j in range(len(tune_idx)):
            cams_j = [cam_data[f'cam_{m}_{t}'][j] for m, t in combos]
            sal_list.append(n_cam_intersection(*cams_j))
        masks  = [cam_to_mask(s, thr, dilate=3) for s in sal_list]
        masked = [apply_mask(imgs_tune[j], masks[j]) for j in range(len(tune_idx))]
        for ml in LANGS:
            p = classify(models[ml], masked, CLASSES[ml])
            t_sub = true[tune_idx]; tgt_sub = target[tune_idx]
            rows.append({
                'variant': label, 'model': ml, 'threshold': thr,
                'baseline_acc': base[ml],
                'masked_acc':   float((p == t_sub).mean()),
                'masked_asr':   float((p == tgt_sub).mean()),
                'coverage':     float(np.mean([m.mean() for m in masks])),
            })
    best = max([r for r in rows if r['model'] == 'en'], key=lambda r: r['masked_acc'])
    print(f'[{label}] best thr={best["threshold"]}  '
          f'acc {100*best["baseline_acc"]:.1f}% -> {100*best["masked_acc"]:.1f}%')
    return rows, best['threshold']

imgs_tune = [attacked_imgs[i] for i in tune_idx]
rows_2mod, BEST_THR_2MOD = sweep_one(cam_2mod_cache['multi_attack'], COMBOS_2MOD, imgs_tune, '2-mod')
rows_4mod, BEST_THR_4MOD = sweep_one(cam_4mod_cache['multi_attack'], COMBOS_4MOD, imgs_tune, '4-mod')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (rows, label) in zip(axes, [(rows_2mod, '2-mod'), (rows_4mod, '4-mod')]):
    for ml, color in zip(LANGS, ['C0', 'C1']):
        xs  = THRESHOLDS
        ys  = [r['masked_acc'] for r in rows if r['model'] == ml]
        asr = [r['masked_asr'] for r in rows if r['model'] == ml]
        ax.plot(xs, [100*y for y in ys],  'o-', color=color, label=f'{ml} acc')
        ax.plot(xs, [100*y for y in asr], 's--', color=color, alpha=0.6, label=f'{ml} ASR')
    ax.set_title(f'{label} threshold sweep'); ax.set_xlabel('Percentile threshold')
    ax.set_ylabel('%'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Threshold sweep — multilingual attack (100-image tune subset)', fontsize=11)
plt.tight_layout()
plt.savefig('results/cam_2mod/threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved -> threshold_sweep.png')
"""),
        md("## 9. Full evaluation on 1000 images"),
        code("""\
print('Computing full 4-mod CAM cache (1000 images)...')
cd4_full, _ = compute_and_cache_cams('multi_attack', all_idx, combos=COMBOS_4MOD)
cd4_clean_full, _ = compute_and_cache_cams('clean', all_idx, combos=COMBOS_4MOD)
cd2_full  = {k: cd4_full[k]  for k in ['cam_en_en', 'cam_zh_zh']}
cd2_clean = {k: cd4_clean_full[k] for k in ['cam_en_en', 'cam_zh_zh']}
"""),
        code("""\
def run_cam_eval(cam_data, cam_data_clean, combos, best_thr, n_mods, results_dir):
    \"\"\"Full CAM defence evaluation.\"\"\"
    sal_list = [n_cam_intersection(*[cam_data[f'cam_{m}_{t}'][j] for m, t in combos])
                for j in range(len(all_idx))]
    masks        = [cam_to_mask(s, best_thr, dilate=3) for s in sal_list]
    masked_imgs  = [apply_mask(attacked_imgs[j], masks[j]) for j in range(len(all_idx))]

    defense_res  = {}
    for ml in LANGS:
        base_p = preds_attacked_2d[ml]
        new_p  = classify(models[ml], masked_imgs, CLASSES[ml])
        wrong  = base_p != true
        defense_res[ml] = {
            'acc':           float((new_p == true).mean()),
            'asr':           float((new_p == target).mean()),
            'recovery_rate': float((wrong & (new_p == true)).sum() / wrong.sum()) if wrong.any() else 0.0,
            'baseline_acc':  float((base_p == true).mean()),
            'baseline_asr':  float((base_p == target).mean()),
        }

    # Clean degradation
    clean_sal    = [n_cam_intersection(*[cam_data_clean[f'cam_{m}_{t}'][j] for m, t in combos])
                    for j in range(len(all_idx))]
    clean_masks  = [cam_to_mask(s, best_thr, dilate=3) for s in clean_sal]
    clean_masked = [apply_mask(clean_224[j], clean_masks[j]) for j in range(len(all_idx))]
    clean_deg    = {}
    for ml in LANGS:
        cp = classify(models[ml], clean_masked, CLASSES[ml])
        clean_deg[ml] = {
            'baseline_acc': clean_acc[ml],
            'masked_acc':   float((cp == true).mean()),
            'delta_acc':    float((cp == true).mean()) - clean_acc[ml],
        }

    coverage = float(np.mean([m.mean() for m in masks]))
    results  = {
        'setup':             'multilingual',
        'method':            f'cam_{n_mods}mod',
        'attack':            'multilingual',
        'n_images':          len(all_idx),
        'best_threshold':    best_thr,
        'combos':            str(combos),
        'inference_cost':    2 + 2 * n_mods,
        'clean_acc':         clean_acc,
        'baseline_acc':      {ml: defense_res[ml]['baseline_acc'] for ml in LANGS},
        'baseline_asr':      {ml: defense_res[ml]['baseline_asr'] for ml in LANGS},
        'defense':           defense_res,
        'clean_degradation': clean_deg,
        'coverage_mean':     coverage,
        'defense_acc_mean':  float(np.mean([defense_res[ml]['acc']  for ml in LANGS])),
        'defense_asr_mean':  float(np.mean([defense_res[ml]['asr']  for ml in LANGS])),
    }
    out_path = f'{results_dir}/confusion_results_cam_defense.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved -> {out_path}')
    return results, masked_imgs, masks

print('Running 2-mod full eval...')
res_2mod, masked_2mod, masks_2mod = run_cam_eval(
    cd2_full, cd2_clean, COMBOS_2MOD, BEST_THR_2MOD, 2, RESULTS_DIR_2MOD)
print('Running 4-mod full eval...')
res_4mod, masked_4mod, masks_4mod = run_cam_eval(
    cd4_full, cd4_clean_full, COMBOS_4MOD, BEST_THR_4MOD, 4, RESULTS_DIR_4MOD)
"""),
        md("## 10. Visual diagnostics"),
        code("""\
def plot_delta(results, label, save_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    vals = np.array([[results['defense'][ml]['acc'] - results['defense'][ml]['baseline_acc']]
                      for ml in LANGS]) * 100
    im = ax.imshow(vals, vmin=-20, vmax=60, cmap='RdYlGn')
    ax.set_xticks([0]); ax.set_xticklabels(['multilingual attack'])
    ax.set_yticks(range(len(LANGS))); ax.set_yticklabels([f'model_{l}' for l in LANGS])
    ax.set_title(f'Accuracy delta after {label} (pp)')
    for i, ml in enumerate(LANGS):
        ax.text(0, i, f'{vals[i,0]:+.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, format=lambda x, _: f'{x:+.0f}pp')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {save_path}')

plot_delta(res_2mod, '2-mod CAM masking', f'{RESULTS_DIR_2MOD}/accuracy_delta_matrix.png')
plot_delta(res_4mod, '4-mod CAM masking', f'{RESULTS_DIR_4MOD}/accuracy_delta_matrix.png')

# Mask examples: 5 images x 7 columns for 2-mod
cam_select = [(c, int(np.where(true == c)[0][0])) for c in range(5)]
fig, axes  = plt.subplots(5, 6, figsize=(14, 18))
col_titles = ['Attacked', 'EN-EN CAM', 'ZH-ZH CAM', 'Intersection', 'Mask', 'Masked']
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=9, fontweight='bold')
for row_i, (c, pos) in enumerate(cam_select):
    img  = attacked_imgs[pos]
    ce   = align_cam(cd2_full['cam_en_en'][pos])
    cz   = align_cam(cd2_full['cam_zh_zh'][pos])
    inter = n_cam_intersection(cd2_full['cam_en_en'][pos], cd2_full['cam_zh_zh'][pos])
    mask = masks_2mod[pos]
    panels = [img, overlay_cam(img, ce), overlay_cam(img, cz),
              overlay_cam(img, inter), mask_overlay(img, mask), masked_2mod[pos]]
    for col_i, panel in enumerate(panels):
        ax = axes[row_i, col_i]
        ax.imshow(panel if not isinstance(panel, np.ndarray) else panel)
        ax.axis('off')
    axes[row_i, 0].set_ylabel(CLASSES['en'][c], fontsize=8, rotation=0, labelpad=36, va='center')
plt.suptitle('2-mod CAM defence examples (multilingual attack)', fontsize=11, y=1.002)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR_2MOD}/mask_examples.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved -> {RESULTS_DIR_2MOD}/mask_examples.png')
"""),
        md("## 11. Summary"),
        code("""\
print('\\n=== Multilingual CAM Defence Summary ===')
for label, res in [('2-mod', res_2mod), ('4-mod', res_4mod)]:
    print(f'\\n{label}:')
    for ml in LANGS:
        d = res['defense'][ml]
        print(f'  model_{ml}: {100*d["baseline_acc"]:.1f}% -> {100*d["acc"]:.1f}%  '
              f'ASR {100*d["baseline_asr"]:.1f}% -> {100*d["asr"]:.1f}%  '
              f'recovery={100*d["recovery_rate"]:.1f}%')
"""),
    ]
    return cells


def _cam_defense_cells_uni():
    """unilingual/cam_defense.ipynb — 2-mod (EN-EN ∩ ZH-EN)."""
    cells = [
        md("""\
# CAM Intersection Defence — Unilingual Attack (2-mod)

Defends against a **unilingual (English-only) typographic attack** (EN text at both boxes)
using GradCAM intersection masking.

**2-mod**: intersect GradCAM(EN model, EN text) ∩ GradCAM(ZH model, EN-text-via-ZH-encoder)

The ZH model is probed with English class-name text to find where it attends even when the
attack language is foreign to it.

Results saved to `results/cam_2mod/`.
"""),
        code(INSTALL_CODE),
        code(SHARED_IMPORTS + """\

RESULTS_DIR  = 'results/cam_2mod'
CACHE_DIR    = os.path.join(RESULTS_DIR, 'cache')
CACHE_DIR_4MOD = CACHE_DIR   # reuse naming convention
os.makedirs(CACHE_DIR, exist_ok=True)

SETUP_LABEL  = 'unilingual'
ATTACK_LABEL = 'unilingual'
"""),
        md("## 1. Model definitions"),
        code(MODEL_CODE),
        md("## 2. Attack helpers"),
        code(FONT_SHARED_CODE),
        code(UNI_ATTACK_CODE),
        md("## 3. Dataset"),
        code(DATA_LOAD_CODE),
        md("## 4. Load models + build unilingual attack"),
        code(LOAD_MODELS_CODE),
        code("""\
print('Building unilingual EN attacked images...')
t0 = time.time()
attacked_imgs = build_attacked_images(clean_224, all_idx, 'en')
print(f'Done in {time.time()-t0:.1f}s')

preds_attacked = {ml: classify(models[ml], attacked_imgs, CLASSES[ml]) for ml in LANGS}
baseline_acc   = {ml: float((preds_attacked[ml] == true).mean())   for ml in LANGS}
baseline_asr   = {ml: float((preds_attacked[ml] == target).mean()) for ml in LANGS}
print('Baseline acc:', {k: f'{100*v:.1f}%' for k, v in baseline_acc.items()})
print('Baseline ASR:', {k: f'{100*v:.1f}%' for k, v in baseline_asr.items()})
"""),
        md("""\
## 5. GradCAM helpers + cross text embedding

For unilingual 2-mod:
1. GradCAM(EN model, EN text)          — standard
2. GradCAM(ZH model, EN-via-ZH text)   — ZH model probed with English class names
"""),
        code(GRADCAM_STANDARD_CODE),
        code(CAM_MASKING_CODE),
        code(CROSS_TEXT_EMBS_CODE),
        code(GRADCAM_GENERAL_CODE),
        md("## 6. CAM cache"),
        code("""\
# Unilingual 2-mod uses combos: (EN model, EN text) and (ZH model, EN text)
COMBOS_UNI_2MOD = [('en', 'en'), ('zh', 'en')]

def compute_and_cache_cams(condition, indices, combos=COMBOS_UNI_2MOD):
    n     = len(indices)
    label = '_'.join(f'{m}{t}' for m, t in combos)
    cfile = os.path.join(CACHE_DIR, f'cams_{label}_{condition}_n{n}.npz')
    keys  = [f'cam_{m}_{t}' for m, t in combos]

    if os.path.exists(cfile):
        data = np.load(cfile, allow_pickle=True)
        print(f'Loaded cache {os.path.basename(cfile)}')
        return {k: data[k] for k in keys}, np.array(data['indices'])

    imgs = attacked_imgs if condition == 'uni_attack' else clean_224
    imgs = [imgs[i] for i in indices]

    cam_lists = {k: [] for k in keys}
    t0 = time.time()
    for j, img in enumerate(imgs):
        for (ml, tl), k in zip(combos, keys):
            emb      = TEXT_EMBS[(ml, tl)]
            fn       = gradcam_en_with_emb if ml == 'en' else gradcam_zh_with_emb
            cam, _   = fn(img, emb)
            cam_lists[k].append(cam)
        if (j + 1) % 50 == 0:
            print(f'  CAM {j+1}/{n} [{time.time()-t0:.1f}s]')

    data_save = {k: np.stack(cam_lists[k]) for k in keys}
    data_save['indices'] = np.array(indices)
    np.savez(cfile, **data_save)
    print(f'Saved cache {os.path.basename(cfile)} [{time.time()-t0:.1f}s]')
    return {k: data_save[k] for k in keys}, np.array(indices)

print('CAM cache helper ready (unilingual 2-mod).')
"""),
        md("## 7. Threshold sweep"),
        code("""\
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
print('Computing tune-set CAMs...')
cd_tune, _ = compute_and_cache_cams('uni_attack', tune_idx)

sweep_rows = []
imgs_tune  = [attacked_imgs[i] for i in tune_idx]
base = {ml: float((preds_attacked[ml][tune_idx] == true[tune_idx]).mean()) for ml in LANGS}
for thr in THRESHOLDS:
    sal_list = [n_cam_intersection(*[cd_tune[f'cam_{m}_{t}'][j] for m, t in COMBOS_UNI_2MOD])
                for j in range(len(tune_idx))]
    masks  = [cam_to_mask(s, thr, 3) for s in sal_list]
    masked = [apply_mask(imgs_tune[j], masks[j]) for j in range(len(tune_idx))]
    for ml in LANGS:
        p = classify(models[ml], masked, CLASSES[ml])
        sweep_rows.append({
            'model': ml, 'threshold': thr,
            'baseline_acc': base[ml],
            'masked_acc':   float((p == true[tune_idx]).mean()),
            'masked_asr':   float((p == target[tune_idx]).mean()),
        })

en_rows    = [r for r in sweep_rows if r['model'] == 'en']
best_row   = max(en_rows, key=lambda r: r['masked_acc'])
BEST_THR   = best_row['threshold']
print(f'Best threshold: {BEST_THR}  '
      f'{100*best_row["baseline_acc"]:.1f}% -> {100*best_row["masked_acc"]:.1f}%')
"""),
        md("## 8. Full evaluation (1000 images)"),
        code("""\
print('Computing full CAM cache...')
cd_full, _       = compute_and_cache_cams('uni_attack', all_idx)
cd_clean_full, _ = compute_and_cache_cams('clean',      all_idx)

def make_sal_list(cam_data, combos, n):
    return [n_cam_intersection(*[cam_data[f'cam_{m}_{t}'][j] for m, t in combos])
            for j in range(n)]

sal_list     = make_sal_list(cd_full, COMBOS_UNI_2MOD, len(all_idx))
masks        = [cam_to_mask(s, BEST_THR, 3) for s in sal_list]
masked_imgs  = [apply_mask(attacked_imgs[j], masks[j]) for j in range(len(all_idx))]

defense_res  = {}
for ml in LANGS:
    base_p = preds_attacked[ml]
    new_p  = classify(models[ml], masked_imgs, CLASSES[ml])
    wrong  = base_p != true
    defense_res[ml] = {
        'acc':           float((new_p == true).mean()),
        'asr':           float((new_p == target).mean()),
        'recovery_rate': float((wrong & (new_p == true)).sum() / wrong.sum()) if wrong.any() else 0.0,
        'baseline_acc':  float((base_p == true).mean()),
        'baseline_asr':  float((base_p == target).mean()),
    }

# Clean degradation
sal_clean   = make_sal_list(cd_clean_full, COMBOS_UNI_2MOD, len(all_idx))
masks_clean = [cam_to_mask(s, BEST_THR, 3) for s in sal_clean]
m_clean     = [apply_mask(clean_224[j], masks_clean[j]) for j in range(len(all_idx))]
clean_deg   = {}
for ml in LANGS:
    cp = classify(models[ml], m_clean, CLASSES[ml])
    clean_deg[ml] = {
        'baseline_acc': clean_acc[ml],
        'masked_acc':   float((cp == true).mean()),
        'delta_acc':    float((cp == true).mean()) - clean_acc[ml],
    }

results = {
    'setup':             'unilingual',
    'method':            'cam_2mod',
    'attack':            'unilingual',
    'n_images':          len(all_idx),
    'best_threshold':    BEST_THR,
    'combos':            str(COMBOS_UNI_2MOD),
    'inference_cost':    6,
    'clean_acc':         clean_acc,
    'baseline_acc':      {ml: defense_res[ml]['baseline_acc'] for ml in LANGS},
    'baseline_asr':      {ml: defense_res[ml]['baseline_asr'] for ml in LANGS},
    'defense':           defense_res,
    'clean_degradation': clean_deg,
    'coverage_mean':     float(np.mean([m.mean() for m in masks])),
    'defense_acc_mean':  float(np.mean([defense_res[ml]['acc'] for ml in LANGS])),
    'defense_asr_mean':  float(np.mean([defense_res[ml]['asr'] for ml in LANGS])),
}
out_path = f'{RESULTS_DIR}/confusion_results_cam_defense.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Saved -> {out_path}')
print('\\nDefence summary:')
for ml in LANGS:
    d = defense_res[ml]
    print(f'  model_{ml}: {100*d["baseline_acc"]:.1f}% -> {100*d["acc"]:.1f}%  '
          f'ASR {100*d["baseline_asr"]:.1f}% -> {100*d["asr"]:.1f}%')
"""),
        md("## 9. Visual diagnostics"),
        code("""\
fig, ax = plt.subplots(figsize=(4, 3))
vals = np.array([[defense_res[ml]['acc'] - defense_res[ml]['baseline_acc']]
                  for ml in LANGS]) * 100
im = ax.imshow(vals, vmin=-20, vmax=60, cmap='RdYlGn')
ax.set_xticks([0]); ax.set_xticklabels(['unilingual EN attack'])
ax.set_yticks(range(len(LANGS))); ax.set_yticklabels([f'model_{l}' for l in LANGS])
ax.set_title('Accuracy delta after 2-mod CAM masking (pp)')
for i, ml in enumerate(LANGS):
    ax.text(0, i, f'{vals[i,0]:+.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, format=lambda x, _: f'{x:+.0f}pp')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/accuracy_delta_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved -> accuracy_delta_matrix.png')
"""),
    ]
    return cells


def _grid_defense_cells(setup, attack_build_code, attack_label, attack_key):
    """Shared grid defence notebook (works for both multilingual and unilingual)."""
    build_call = ("build_multilingual_attacked_images(clean_224, all_idx)"
                  if setup == 'multilingual'
                  else "build_attacked_images(clean_224, all_idx, 'en')")
    cells = [
        md(f"""\
# Grid Search Occlusion Defence — {attack_label} Attack

Defends against {attack_label.lower()} typographic attack using **4×4 grid occlusion**:
- **1-patch**: try all 16 patches, keep the occlusion with highest mean model confidence.
- **2-patch (greedy)**: fix best 1st patch, try remaining 15 for the 2nd best patch.

No GradCAM — purely black-box. Works by finding which patch, when removed, makes both models
most confident in their prediction.

Results saved to `results/grid_1patch/` and `results/grid_2patch/`.
"""),
        code(INSTALL_CODE),
        code(SHARED_IMPORTS + f"""\

RESULTS_DIR_1P = 'results/grid_1patch'
RESULTS_DIR_2P = 'results/grid_2patch'
os.makedirs(RESULTS_DIR_1P, exist_ok=True)
os.makedirs(RESULTS_DIR_2P, exist_ok=True)

SETUP_LABEL  = '{setup}'
ATTACK_LABEL = '{attack_key}'
"""),
        md("## 1. Model definitions"),
        code(MODEL_CODE),
        md("## 2. Attack helpers"),
        code(FONT_SHARED_CODE),
        code(attack_build_code),
        md("## 3. Dataset"),
        code(DATA_LOAD_CODE),
        md("## 4. Load models + build attacked images"),
        code(LOAD_MODELS_CODE),
        code(f"""\
print('Building attacked images...')
t0 = time.time()
attacked_imgs = {build_call}
print(f'Done in {{time.time()-t0:.1f}}s')

preds_attacked = {{ml: classify(models[ml], attacked_imgs, CLASSES[ml]) for ml in LANGS}}
baseline_acc   = {{ml: float((preds_attacked[ml] == true).mean())   for ml in LANGS}}
baseline_asr   = {{ml: float((preds_attacked[ml] == target).mean()) for ml in LANGS}}
print('Baseline acc:', {{k: f'{{100*v:.1f}}%' for k, v in baseline_acc.items()}})
"""),
        md("""\
## 5. Grid defence helpers

**Scoring criterion**: after occluding a patch, take the maximum cosine-similarity (across 10
CIFAR classes) for both EN and ZH models and average them.  A higher score means both models
are more "confident" in some class prediction, which tends to happen when adversarial text
has been occluded.
"""),
        code(GRID_HELPERS_CODE),
        code(GRID_RUN_CODE),
        md("## 6. Run 1-patch defence"),
        code("""\
print('Running 1-patch grid defence...')
t0 = time.time()
defended_1p, best_patches_1p = run_grid_1patch(attacked_imgs)
print(f'Done in {time.time()-t0:.1f}s')

preds_1p = {ml: classify(models[ml], defended_1p, CLASSES[ml]) for ml in LANGS}
acc_1p   = {ml: float((preds_1p[ml] == true).mean())   for ml in LANGS}
asr_1p   = {ml: float((preds_1p[ml] == target).mean()) for ml in LANGS}
print('1-patch defence acc:', {k: f'{100*v:.1f}%' for k, v in acc_1p.items()})
print('1-patch defence ASR:', {k: f'{100*v:.1f}%' for k, v in asr_1p.items()})
"""),
        md("## 7. Run 2-patch defence (greedy)"),
        code("""\
print('Running 2-patch greedy grid defence...')
t0 = time.time()
defended_2p, second_patches = run_grid_2patch_greedy(attacked_imgs, best_patches_1p)
print(f'Done in {time.time()-t0:.1f}s')

preds_2p = {ml: classify(models[ml], defended_2p, CLASSES[ml]) for ml in LANGS}
acc_2p   = {ml: float((preds_2p[ml] == true).mean())   for ml in LANGS}
asr_2p   = {ml: float((preds_2p[ml] == target).mean()) for ml in LANGS}
print('2-patch defence acc:', {k: f'{100*v:.1f}%' for k, v in acc_2p.items()})
print('2-patch defence ASR:', {k: f'{100*v:.1f}%' for k, v in asr_2p.items()})
"""),
        md("## 8. Visual diagnostics"),
        code("""\
# Show which patch was selected for each class (use first 5 classes)
sample_pos = [int(np.where(true == c)[0][0]) for c in range(5)]
fig, axes  = plt.subplots(5, 4, figsize=(13, 17))
col_titles = ['Attacked', '1-patch occluded', '2-patch occluded', 'Patch grid']
for ax, t_title in zip(axes[0], col_titles):
    ax.set_title(t_title, fontsize=9, fontweight='bold')

for row_i, pos in enumerate(sample_pos):
    arr  = np.array(attacked_imgs[pos].convert('RGB'))
    fill = arr.reshape(-1, 3).mean(0).astype(np.uint8)

    # Grid overlay showing selected patches
    grid_img = attacked_imgs[pos].copy()
    grid_arr = np.array(grid_img)
    # Draw grid lines
    for r in range(1, GRID_ROWS):
        grid_arr[r * PATCH_H - 1:r * PATCH_H + 1, :] = [255, 0, 0]
    for c in range(1, GRID_COLS):
        grid_arr[:, c * PATCH_W - 1:c * PATCH_W + 1] = [255, 0, 0]
    # Highlight selected patches
    x0, y0, x1, y1 = PATCHES[best_patches_1p[pos]]
    grid_arr[y0:y0+3, x0:x1] = [0, 255, 0]
    grid_arr[y1-3:y1, x0:x1] = [0, 255, 0]
    grid_arr[y0:y1, x0:x0+3] = [0, 255, 0]
    grid_arr[y0:y1, x1-3:x1] = [0, 255, 0]
    x0, y0, x1, y1 = PATCHES[second_patches[pos]]
    grid_arr[y0:y0+3, x0:x1] = [0, 0, 255]
    grid_arr[y1-3:y1, x0:x1] = [0, 0, 255]
    grid_arr[y0:y1, x0:x0+3] = [0, 0, 255]
    grid_arr[y0:y1, x1-3:x1] = [0, 0, 255]

    panels = [attacked_imgs[pos], defended_1p[pos], defended_2p[pos], Image.fromarray(grid_arr)]
    for col_i, panel in enumerate(panels):
        axes[row_i, col_i].imshow(panel); axes[row_i, col_i].axis('off')
    axes[row_i, 0].set_ylabel(CLASSES['en'][true[pos]], fontsize=8, rotation=0, labelpad=36, va='center')

plt.suptitle(f'Grid defence examples — {SETUP_LABEL} attack', fontsize=11, y=1.001)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR_1P}/grid_defence_examples.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved -> grid_defence_examples.png')
"""),
        md("## 9. Save results JSON"),
        code(f"""\
def save_grid_results(preds_def, acc_def, asr_def, method, cost, results_dir):
    defense_res = {{}}
    for ml in LANGS:
        base_p = preds_attacked[ml]
        new_p  = preds_def[ml]
        wrong  = base_p != true
        defense_res[ml] = {{
            'acc':           float((new_p == true).mean()),
            'asr':           float((new_p == target).mean()),
            'recovery_rate': float((wrong & (new_p == true)).sum() / wrong.sum()) if wrong.any() else 0.0,
            'baseline_acc':  float((base_p == true).mean()),
            'baseline_asr':  float((base_p == target).mean()),
        }}
    results = {{
        'setup':            SETUP_LABEL,
        'method':           method,
        'attack':           ATTACK_LABEL,
        'n_images':         len(all_idx),
        'inference_cost':   cost,
        'clean_acc':        clean_acc,
        'baseline_acc':     {{ml: defense_res[ml]['baseline_acc'] for ml in LANGS}},
        'baseline_asr':     {{ml: defense_res[ml]['baseline_asr'] for ml in LANGS}},
        'defense':          defense_res,
        'defense_acc_mean': float(np.mean([defense_res[ml]['acc'] for ml in LANGS])),
        'defense_asr_mean': float(np.mean([defense_res[ml]['asr'] for ml in LANGS])),
    }}
    out_path = f'{{results_dir}}/confusion_results_grid.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved -> {{out_path}}')
    return results

res_1p = save_grid_results(preds_1p, acc_1p, asr_1p, 'grid_1patch', 32, RESULTS_DIR_1P)
res_2p = save_grid_results(preds_2p, acc_2p, asr_2p, 'grid_2patch', 62, RESULTS_DIR_2P)

print('\\n=== Grid Defence Summary ===')
for label, res in [('1-patch', res_1p), ('2-patch', res_2p)]:
    print(f'{{label}}:')
    for ml in LANGS:
        d = res['defense'][ml]
        print(f'  model_{{ml}}: {{100*d["baseline_acc"]:.1f}}% -> {{100*d["acc"]:.1f}}%  '
              f'ASR {{100*d["baseline_asr"]:.1f}}% -> {{100*d["asr"]:.1f}}%')
"""),
    ]
    return cells


# ─── main builders ─────────────────────────────────────────────────────────────

def build_multi_attack():
    cells = _attack_comparison_cells(
        setup='multilingual',
        results_dir='results/attack',
        attack_build_code=MULTI_ATTACK_CODE,
        attack_label='Multilingual',
        attack_description=(
            "Each image has **two typographic text boxes**: "
            "Box-0 contains the English attack word; Box-1 contains the Chinese (ZH) attack word. "
            "Both aim at the same incorrect target class."
        ),
    )
    write_nb(MULTI / "attack_comparison.ipynb", cells)


def build_multi_cam():
    cells = _cam_defense_cells_multi()
    write_nb(MULTI / "cam_defense.ipynb", cells)


def build_multi_grid():
    cells = _grid_defense_cells(
        setup='multilingual',
        attack_build_code=MULTI_ATTACK_CODE,
        attack_label='Multilingual',
        attack_key='multilingual',
    )
    write_nb(MULTI / "grid_defense.ipynb", cells)


def build_uni_attack():
    cells = _attack_comparison_cells(
        setup='unilingual',
        results_dir='results/attack',
        attack_build_code=UNI_ATTACK_CODE,
        attack_label='Unilingual',
        attack_description=(
            "Each image has **two typographic text boxes**, both containing the same "
            "English attack word (same target class). Only English text is present."
        ),
    )
    # Patch the attack-build call in cell 14 (baseline eval) for unilingual
    for cell in cells:
        if isinstance(cell.get('source'), list):
            src = ''.join(cell['source'])
            if 'build_multilingual_attacked_images' in src:
                new_src = src.replace(
                    'build_multilingual_attacked_images',
                    "lambda b, i, n=None: build_attacked_images(b, i, 'en', n)",
                )
                cell['source'] = new_src.splitlines(True)
    write_nb(UNI / "attack_comparison.ipynb", cells)


def build_uni_cam():
    cells = _cam_defense_cells_uni()
    write_nb(UNI / "cam_defense.ipynb", cells)


def build_uni_grid():
    cells = _grid_defense_cells(
        setup='unilingual',
        attack_build_code=UNI_ATTACK_CODE,
        attack_label='Unilingual',
        attack_key='unilingual',
    )
    write_nb(UNI / "grid_defense.ipynb", cells)


def build_cost_vs_performance():
    cells = [
        md("""\
# Inference Cost vs. Defence Performance

Aggregates all experiment results and plots **inference cost** (forward passes per image)
against **mean post-defence accuracy** for both multilingual and unilingual setups.

## Inference cost table
| Method        | Forward passes / image |
|---------------|------------------------|
| no_defense    | 2                      |
| cam_2mod      | 6                      |
| cam_4mod      | 10                     |
| grid_1patch   | 32                     |
| grid_2patch   | 62                     |
"""),
        code(COST_PERF_NB_CODE),
        code(COST_PERF_PLOT_CODE),
    ]
    write_nb(HERE / "cost_vs_performance.ipynb", cells)


if __name__ == "__main__":
    build_multi_attack()
    build_multi_cam()
    build_multi_grid()
    build_uni_attack()
    build_uni_cam()
    build_uni_grid()
    build_cost_vs_performance()
    print("\nAll notebooks generated.")
