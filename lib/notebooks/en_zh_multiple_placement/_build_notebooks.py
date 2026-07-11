"""Generate random_placement_comparison.ipynb and cam_intersection_defense.ipynb."""
import json
import textwrap
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BALANCED = HERE.parent / "en_zh_typographic" / "balanced_typographic_comparison.ipynb"

RANDOM_ATTACK_CODE = textwrap.dedent("""
DISPLAY_SIZE = 224
FONT_SIZE_SINGLE = 40
NUM_BOXES = 2
# 40/sqrt(2)≈28 ignores fixed padding; size 24 keeps dual total area ~match single box.
FONT_SIZE = 24
PAD = 8
_FONT_CACHE = {}

def _font_paths():
    if platform.system() == 'Windows':
        winfonts = os.path.join(os.environ.get('WINDIR', r'C:\\Windows'), 'Fonts')
        cjk   = os.path.join(winfonts, 'msyh.ttc')
        latin = os.path.join(winfonts, 'arial.ttf')
        if not os.path.exists(latin):
            latin = cjk
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
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
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
        if all(not _rects_overlap(rect, p) for p in placed):
            return rect
    return (rect_x, rect_y, rect_x + box_w, rect_y + box_h)

def _draw_text_box(draw, word, rect, font):
    rect_x, rect_y, rect_x2, rect_y2 = rect
    bb = draw.textbbox((0, 0), word, font=font)
    x = rect_x + PAD
    y = rect_y + PAD
    draw.rectangle([rect_x, rect_y, rect_x2, rect_y2], fill='white')
    draw.text((x - bb[0], y - bb[1]), word, fill='black', font=font)

def draw_word(img, word, img_idx, already_224=False):
    fp = _font_for(word)
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    font = _get_font(fp)
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), word, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    box_w = tw + 2 * PAD
    box_h = th + PAD + 12
    placed = []
    for box_i in range(NUM_BOXES):
        rng = random.Random(int(img_idx) * NUM_BOXES + box_i)
        rect = _random_nonoverlapping_rect(rng, box_w, box_h, placed)
        placed.append(rect)
        _draw_text_box(draw, word, rect, font)
    return img

def build_attacked_images(base_imgs, img_indices, attack_lang, n_workers=None):
    words = [CLASSES[attack_lang][target[k]] for k in img_indices]
    n_workers = n_workers or min(8, os.cpu_count() or 4)

    def _one(args):
        im, word, img_idx = args
        return draw_word(im, word, img_idx=img_idx, already_224=True)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one, zip(base_imgs, words, img_indices)))

print('Font paths:', _CJK_FONT, _LAT_FONT)
print(f'Overlay: {NUM_BOXES} boxes @ size {FONT_SIZE} ({DISPLAY_SIZE}px), placement=random (seed per img_idx+box)')
""").strip()


def md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def cell_source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = textwrap.dedent(text).strip().splitlines(True)


def build_baseline_notebook():
    nb = json.loads(BALANCED.read_text(encoding="utf-8"))
    cells = nb["cells"]

    set_source(cells[0], """
# EN vs ZH Typographic Attacks — Random Dual Placement

Same experiment as `en_zh_typographic/balanced_typographic_comparison.ipynb` but **two** text boxes are placed at
**random non-overlapping positions** across the 224×224 canvas (deterministic per image/box via `random.Random(img_idx * 2 + box_i)`).

Uses `CIFAR10_BALANCED_1000_SAMPLE.json` — 100 images per class. Overlay: **2 boxes @ font size 24** (≈ same total area as one size-40 box).

Results saved to `results/`.
""")

    set_source(cells[2], cell_source(cells[2]).replace(
        "RESULTS_DIR = 'results/balanced'",
        "RESULTS_DIR = 'results'",
    ))

    set_source(cells[5], "## 2. Typographic attack helper (dual box, random placement)")
    set_source(cells[6], RANDOM_ATTACK_CODE)

    cells.insert(9, md("## 2b. Dual-box preview (sanity check before model load)"))
    cells.insert(10, code("""
preview_idx = [int(np.where(true == c)[0][0]) for c in range(4)]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle(f'Dual-box attack preview ({NUM_BOXES} boxes @ size {FONT_SIZE})', fontsize=12)
for col, pos in enumerate(preview_idx):
    word_en = CLASSES['en'][target[pos]]
    word_zh = CLASSES['zh'][target[pos]]
    axes[0, col].imshow(draw_word(clean_224[pos], word_en, img_idx=pos, already_224=True))
    axes[0, col].set_title(f"EN: {word_en}", fontsize=9)
    axes[0, col].axis('off')
    axes[1, col].imshow(draw_word(clean_224[pos], word_zh, img_idx=pos + 1000, already_224=True))
    axes[1, col].set_title(f"ZH: {word_zh}", fontsize=9)
    axes[1, col].axis('off')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/dual_box_preview.png', dpi=130, bbox_inches='tight')
plt.show()
print(f'Saved -> {RESULTS_DIR}/dual_box_preview.png')
"""))

    set_source(cells[16], """
all_idx = np.arange(len(clean))
attacked_by_lang = {}
preds_attacked = {}
for attack_lang in LANGS:
    t0 = time.time()
    print(f'\\nAttack language: {attack_lang}')
    attacked_by_lang[attack_lang] = build_attacked_images(clean_224, all_idx, attack_lang)
    preds_attacked[attack_lang] = {}
    for model_lang in LANGS:
        p   = classify(models[model_lang], attacked_by_lang[attack_lang], CLASSES[model_lang])
        preds_attacked[attack_lang][model_lang] = p
        acc = (p == true).mean()
        asr = (p == target).mean()
        print(f'  model_{model_lang}: acc={100*acc:.1f}%  ASR={100*asr:.1f}%')
    print(f'  [{time.time()-t0:.1f}s]')
print('\\nAttack loop complete.')
""")

    gradcam_src = cell_source(cells[22])
    gradcam_src = gradcam_src.replace(
        "Large-font overlay fills 224×224 frame (size 40 text); balanced sample",
        "Dual random placement (2 boxes @ size 24 ≈ one size-40 area); balanced sample",
    )
    set_source(cells[20], gradcam_src)

    viz_src = cell_source(cells[24])
    viz_src = viz_src.replace(
        "balanced sample (100 per class), large font (size 40 @ 224px)",
        "balanced sample (100 per class), dual random placement (2 boxes @ size 24)",
    )
    set_source(cells[22], viz_src)

    save_src = cell_source(cells[28])
    save_src = save_src.replace(
        "'sample':     'CIFAR10_BALANCED_1000_SAMPLE',",
        "'sample':     'CIFAR10_BALANCED_1000_SAMPLE',\n    'placement':  'random_dual',\n    'overlay':    {'font_size': 24, 'num_boxes': 2, 'display_size': 224, 'single_box_equiv_font_size': 40},",
    )
    set_source(cells[26], save_src)

    for cell in cells:
        cell.pop("execution_count", None)
        cell["outputs"] = []

    out = HERE / "random_placement_comparison.ipynb"
    nb["cells"] = cells
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(cells)} cells)")


def build_defense_notebook():
    cells = []

    cells.append(md("""
# CAM Intersection Masking Defense — Random Dual Placement

Intersect EN and ZH GradCAM saliency on typographic-attacked CIFAR-10 images with **two random text boxes** per image,
mask high-saliency regions, re-classify, and measure accuracy recovery.

Uses the balanced 1000-image sample (`../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`) with
dual overlays (2 boxes @ size 24 ≈ one size-40 area). Results saved to `results/cam_defense/`.

Fast re-run (uses CAM cache): `python run_cam_defense_fast.py`
"""))

    cells.append(code("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'open_clip_torch', 'transformers', 'datasets',
                'matplotlib', 'Pillow'], check=False)
"""))

    cells.append(code("""
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

RESULTS_DIR = 'results/cam_defense'
CACHE_DIR = os.path.join(RESULTS_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
"""))

    cells.append(md("## 1. Model definitions"))

    cells.append(code("""
def classify(model, imgs, words, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i:i + batch_size])
        tf = model.embed_texts(words)
        preds.append((imf @ tf.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)

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
"""))

    cells.append(md("## 2. Typographic attack helper (dual box, random placement)"))
    cells.append(code(RANDOM_ATTACK_CODE))

    cells.append(md("## 3. Load balanced dataset"))
    cells.append(code("""
hf = load_dataset('uoft-cs/cifar10', split='test')
label_key = 'label' if 'label' in hf.column_names else 'labels'
image_key = 'img'   if 'img'   in hf.column_names else 'image'

_indices_path = '../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json'
with open(_indices_path, encoding='utf-8') as f:
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

clean = [im.convert('RGB') for im in rows[image_key]]
clean_224 = [im.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC) for im in clean]
all_idx = np.arange(len(clean))
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
print(f'Loaded {len(clean)} images; tune subset = {len(tune_idx)}')
"""))

    cells.append(md("## 4. Load models + baseline attack loop"))
    cells.append(code("""
models = {}
for lang, cls in MODEL_CLS.items():
    t0 = time.time()
    print(f'Loading {lang}...', end=' ', flush=True)
    models[lang] = cls()
    print(f'{time.time()-t0:.1f}s')

TEXT_EMB = {lang: models[lang].embed_texts(CLASSES[lang]).detach() for lang in LANGS}

clean_preds = {lang: classify(models[lang], clean, CLASSES[lang]) for lang in LANGS}
clean_acc = {lang: float((clean_preds[lang] == true).mean()) for lang in LANGS}
print('Clean acc:', {k: f'{100*v:.1f}%' for k, v in clean_acc.items()})

attacked_by_lang = {}
preds_attacked = {}
for attack_lang in LANGS:
    print(f'Building {attack_lang} attack...', end=' ', flush=True)
    t0 = time.time()
    attacked_by_lang[attack_lang] = build_attacked_images(clean_224, all_idx, attack_lang)
    preds_attacked[attack_lang] = {
        ml: classify(models[ml], attacked_by_lang[attack_lang], CLASSES[ml])
        for ml in LANGS
    }
    print(f'{time.time()-t0:.1f}s')

n = len(LANGS)
acc_matrix = np.zeros((n, n))
asr_matrix = np.zeros((n, n))
for i, al in enumerate(LANGS):
    for j, ml in enumerate(LANGS):
        p = preds_attacked[al][ml]
        acc_matrix[i, j] = (p == true).mean()
        asr_matrix[i, j] = (p == target).mean()

print('Baseline acc matrix (%):')
for i, al in enumerate(LANGS):
    print(f'  attack_{al}:', '  '.join(f'model_{ml}={100*acc_matrix[i,j]:.1f}%' for j, ml in enumerate(LANGS)))
"""))

    cells.append(md("""
## 5. GradCAM + intersection masking

EN ViT-B/32 and ZH ViT-B/16 produce CAMs at different patch resolutions (~7x7 vs ~14x14).
Both are resized to 224x224 before intersection.
"""))

    cells.append(code("""
def _norm_cam(cam):
    cam = cam.relu() if isinstance(cam, torch.Tensor) else np.maximum(cam, 0)
    cam = cam.detach().cpu().numpy() if isinstance(cam, torch.Tensor) else cam
    cam = cam - cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam

def _cam_from_conv(act, grad):
    w = grad.mean(dim=(2, 3), keepdim=True)
    return _norm_cam((w * act).sum(dim=1).squeeze(0))

def gradcam_en(pil_img, target_idx):
    wrapper = models['en']; acts = {}
    def hook(_m, _i, out):
        out.retain_grad(); acts['v'] = out
    handle = wrapper.m.visual.conv1.register_forward_hook(hook)
    x = wrapper.pp(pil_img).unsqueeze(0).to(DEVICE)
    feat = wrapper.m.visual(x)
    img_feat = F.normalize(feat, dim=-1)
    score = (img_feat @ TEXT_EMB['en'][target_idx:target_idx+1].T).squeeze()
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam

def gradcam_zh(pil_img, target_idx):
    wrapper = models['zh']; acts = {}
    patch = wrapper.m.vision_model.embeddings.patch_embedding
    def hook(_m, _i, out):
        out.retain_grad(); acts['v'] = out
    handle = patch.register_forward_hook(hook)
    pv = wrapper.p(images=[pil_img], return_tensors='pt').pixel_values.to(DEVICE)
    out = wrapper.m.get_image_features(pixel_values=pv)
    img_feat = F.normalize(_clip_feat(out), dim=-1)
    score = (img_feat @ TEXT_EMB['zh'][target_idx:target_idx+1].T).squeeze()
    wrapper.m.zero_grad(); score.backward()
    cam = _cam_from_conv(acts['v'].detach(), acts['v'].grad)
    handle.remove(); return cam

GRADCAM_FN = {'en': gradcam_en, 'zh': gradcam_zh}

def get_raw_cam(model_lang, pil_img, target_idx):
    return GRADCAM_FN[model_lang](pil_img, int(target_idx))

def align_cams(cam_en, cam_zh, size=DISPLAY_SIZE):
    def _resize(cam):
        return np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    return _resize(cam_en), _resize(cam_zh)

def intersection_map(cam_en, cam_zh, mode='min'):
    a, b = align_cams(cam_en, cam_zh)
    return np.minimum(a, b) if mode == 'min' else (a * b)

def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        m = (
            pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] |
            pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
            pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:]
        )
    return m

def cam_to_mask(saliency, threshold=0.85, dilate=3):
    sal = saliency.copy()
    thr = np.percentile(sal, threshold * 100)
    mask = sal >= thr
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask

def apply_mask(pil_img, mask, fill='mean_nonmask'):
    pil_img = pil_img.convert('RGB')
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    m = mask.astype(bool)
    out = arr.copy()
    if fill == 'mean_nonmask':
        bg = ~m
        mean_color = arr[bg].mean(axis=0) if bg.any() else arr.reshape(-1, 3).mean(axis=0)
        out[m] = mean_color
    return Image.fromarray(out.astype(np.uint8))

def overlay_cam(pil_img, cam, alpha=0.50):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    heat = cm.jet(np.array(cam_img) / 255.0)[:, :, :3]
    base = np.array(pil_img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE))).astype(np.float32) / 255.0
    blended = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

def mask_overlay(pil_img, mask, alpha=0.45):
    arr = np.array(pil_img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE))).astype(np.float32)
    red = np.zeros_like(arr); red[:, :, 0] = 255
    m = mask.astype(np.float32)[..., None]
    blended = arr * (1 - alpha * m) + red * (alpha * m)
    return Image.fromarray(blended.astype(np.uint8))

print('GradCAM + masking helpers ready.')
"""))

    cells.append(md("## 6. CAM cache + mask builder"))
    cells.append(code("""
def _cache_path(name):
    return os.path.join(CACHE_DIR, name)

def compute_and_cache_cams(condition='en_attack', indices=None):
    cache_file = _cache_path(f'cams_{condition}_dual{NUM_BOXES}_fs{FONT_SIZE}_n{len(indices)}.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        print(f'Loaded cache {cache_file}')
        return data['cam_en'], data['cam_zh'], data['indices']

    if condition == 'clean':
        imgs = [clean_224[i] for i in indices]
        pred_en = clean_preds['en'][indices]
        pred_zh = clean_preds['zh'][indices]
    else:
        atk = condition.replace('_attack', '')
        imgs = [attacked_by_lang[atk][i] for i in indices]
        pred_en = preds_attacked[atk]['en'][indices]
        pred_zh = preds_attacked[atk]['zh'][indices]

    cam_en_list, cam_zh_list = [], []
    t0 = time.time()
    for j, (img, pe, pz) in enumerate(zip(imgs, pred_en, pred_zh)):
        cam_en_list.append(get_raw_cam('en', img, pe))
        cam_zh_list.append(get_raw_cam('zh', img, pz))
        if (j + 1) % 50 == 0:
            print(f'  CAM {j+1}/{len(imgs)} [{time.time()-t0:.1f}s]')
    cam_en = np.stack(cam_en_list)
    cam_zh = np.stack(cam_zh_list)
    np.savez(cache_file, cam_en=cam_en, cam_zh=cam_zh, indices=np.array(indices))
    print(f'Saved cache {cache_file} [{time.time()-t0:.1f}s]')
    return cam_en, cam_zh, np.array(indices)

def build_masks_from_cams(cam_en, cam_zh, config):
    masks, coverages = [], []
    for ce, cz in zip(cam_en, cam_zh):
        if config.get('strategy') == 'en_only':
            sal, _ = align_cams(ce, ce)
        elif config.get('strategy') == 'zh_only':
            _, sal = align_cams(cz, cz)
        else:
            sal = intersection_map(ce, cz, mode=config.get('mode', 'min'))
        mask = cam_to_mask(sal, threshold=config.get('threshold', 0.85),
                           dilate=config.get('dilate', 3))
        masks.append(mask)
        coverages.append(mask.mean())
    return masks, np.array(coverages)

def eval_config(imgs, cam_en, cam_zh, config, indices):
    masks, coverages = build_masks_from_cams(cam_en, cam_zh, config)
    masked = [apply_mask(img, m) for img, m in zip(imgs, masks)]
    preds = {ml: classify(models[ml], masked, CLASSES[ml]) for ml in LANGS}
    t = true[indices]
    tgt = target[indices]
    out = {'coverage_mean': float(coverages.mean()), 'models': {}}
    for ml in LANGS:
        p = preds[ml]
        out['models'][ml] = {
            'acc': float((p == t).mean()),
            'asr': float((p == tgt).mean()),
        }
    return out, masked, masks, preds

print('Cache helpers ready.')
"""))

    cells.append(md("## 7. Threshold sweep on 100-image subset"))
    cells.append(code("""
cam_cache = {}
for cond in ['en_attack', 'zh_attack', 'clean']:
    cam_cache[cond] = compute_and_cache_cams(cond, tune_idx)

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
sweep_rows = []

for cond in ['en_attack', 'zh_attack']:
    atk = cond.replace('_attack', '')
    imgs = [attacked_by_lang[atk][i] for i in tune_idx]
    ce, cz, _ = cam_cache[cond]
    base = {ml: float((preds_attacked[atk][ml][tune_idx] == true[tune_idx]).mean()) for ml in LANGS}
    for thr in THRESHOLDS:
        cfg = {'strategy': 'intersection', 'mode': 'min', 'threshold': thr, 'dilate': 3}
        res, _, _, _ = eval_config(imgs, ce, cz, cfg, tune_idx)
        for ml in LANGS:
            sweep_rows.append({
                'condition': cond, 'model': ml, 'threshold': thr,
                'baseline_acc': base[ml], 'masked_acc': res['models'][ml]['acc'],
                'masked_asr': res['models'][ml]['asr'],
                'coverage': res['coverage_mean'],
            })

en_rows = [r for r in sweep_rows if r['condition'] == 'en_attack' and r['model'] == 'en']
best_row = max(en_rows, key=lambda r: r['masked_acc'])
BEST_THRESHOLD = best_row['threshold']
print(f'Best threshold on tune subset (EN attack, EN model): {BEST_THRESHOLD}')
print(f"  baseline={100*best_row['baseline_acc']:.1f}% -> masked={100*best_row['masked_acc']:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cond in zip(axes, ['en_attack', 'zh_attack']):
    for ml, color in zip(LANGS, ['C0', 'C1']):
        xs, ys, ys_asr = [], [], []
        for thr in THRESHOLDS:
            row = next(r for r in sweep_rows if r['condition']==cond and r['model']==ml and r['threshold']==thr)
            xs.append(thr); ys.append(row['masked_acc']); ys_asr.append(row['masked_asr'])
        ax.plot(xs, [100*y for y in ys], 'o-', color=color, label=f'model_{ml} acc')
        ax.plot(xs, [100*y for y in ys_asr], 's--', color=color, alpha=0.6, label=f'model_{ml} ASR')
    ax.set_title(cond.replace('_', ' ').title())
    ax.set_xlabel('Percentile threshold'); ax.set_ylabel('%')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Threshold sweep (100-image tune subset)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved -> {RESULTS_DIR}/threshold_sweep.png')
"""))

    cells.append(md("## 8. Full evaluation + ablations (1000 images)"))
    cells.append(code("""
full_cache = {}
for cond in ['en_attack', 'zh_attack', 'clean']:
    full_cache[cond] = compute_and_cache_cams(cond, all_idx)

DEFAULT_CFG = {'strategy': 'intersection', 'mode': 'min', 'threshold': BEST_THRESHOLD, 'dilate': 3}

ABLATIONS = {
    'intersection_min': DEFAULT_CFG,
    'intersection_product': {**DEFAULT_CFG, 'mode': 'product'},
    'en_only': {**DEFAULT_CFG, 'strategy': 'en_only'},
    'zh_only': {**DEFAULT_CFG, 'strategy': 'zh_only'},
}

results = {
    'sample': 'CIFAR10_BALANCED_1000_SAMPLE',
    'placement': 'random_dual',
    'overlay': {'font_size': 24, 'num_boxes': 2, 'display_size': 224, 'single_box_equiv_font_size': 40},
    'best_threshold_tune': BEST_THRESHOLD,
    'default_config': DEFAULT_CFG,
    'baseline': {
        'clean_acc': clean_acc,
        'acc_matrix': acc_matrix.tolist(),
        'asr_matrix': asr_matrix.tolist(),
    },
    'defense': {},
    'ablations': {},
    'clean_degradation': {},
    'threshold_sweep': sweep_rows,
}

for cond in ['en_attack', 'zh_attack']:
    atk = cond.replace('_attack', '')
    imgs = attacked_by_lang[atk]
    ce, cz, idxs = full_cache[cond]
    base_preds = {ml: preds_attacked[atk][ml] for ml in LANGS}

    res, masked_imgs, masks, new_preds = eval_config(imgs, ce, cz, DEFAULT_CFG, all_idx)
    cell = {}
    for ml in LANGS:
        base_p = base_preds[ml]
        new_p = new_preds[ml]
        wrong = base_p != true
        recovered = wrong & (new_p == true)
        cell[ml] = {
            **res['models'][ml],
            'recovery_rate': float(recovered.sum() / wrong.sum()) if wrong.any() else 0.0,
            'baseline_acc': float((base_p == true).mean()),
            'baseline_asr': float((base_p == target).mean()),
        }
    results['defense'][cond] = {'coverage_mean': res['coverage_mean'], 'models': cell}

ce, cz, _ = full_cache['clean']
clean_res, _, _, _ = eval_config(clean_224, ce, cz, DEFAULT_CFG, all_idx)
results['clean_degradation'] = {
    'coverage_mean': clean_res['coverage_mean'],
    'models': {
        ml: {
            'baseline_acc': clean_acc[ml],
            'masked_acc': clean_res['models'][ml]['acc'],
            'delta_acc': clean_res['models'][ml]['acc'] - clean_acc[ml],
        }
        for ml in LANGS
    }
}

atk_imgs = attacked_by_lang['en']
ce, cz, _ = full_cache['en_attack']
for name, cfg in ABLATIONS.items():
    res, _, _, _ = eval_config(atk_imgs, ce, cz, cfg, all_idx)
    results['ablations'][name] = {
        'coverage_mean': res['coverage_mean'],
        'models': res['models'],
    }

print('Full evaluation complete.')
print('Defense EN attack:')
for ml in LANGS:
    d = results['defense']['en_attack']['models'][ml]
    print(f"  model_{ml}: {100*d['baseline_acc']:.1f}% -> {100*d['acc']:.1f}%  ASR {100*d['baseline_asr']:.1f}% -> {100*d['asr']:.1f}%  recovery={100*d['recovery_rate']:.1f}%")
"""))

    cells.append(md("## 9. Visual diagnostics"))
    cells.append(code("""
delta = np.zeros((n, n))
for i, al in enumerate(LANGS):
    cond = f'{al}_attack'
    for j, ml in enumerate(LANGS):
        d = results['defense'][cond]['models'][ml]
        delta[i, j] = d['acc'] - d['baseline_acc']

fig, ax = plt.subplots(figsize=(4.5, 3.5))
im = ax.imshow(delta * 100, vmin=-20, vmax=60, cmap='RdYlGn')
ax.set_xticks(range(n)); ax.set_xticklabels([f'model_{l}' for l in LANGS])
ax.set_yticks(range(n)); ax.set_yticklabels([f'attack_{l}' for l in LANGS])
ax.set_title('Accuracy delta after CAM masking (pp)')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{delta[i,j]*100:+.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, format=lambda x, _: f'{x:+.0f}pp')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/accuracy_delta_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

ce, cz, _ = full_cache['en_attack']
cam_select = [(c, int(np.where(true == c)[0][0])) for c in range(10)]
fig, axes = plt.subplots(10, 7, figsize=(16, 34))
col_titles = ['Attacked', 'EN CAM', 'ZH CAM', 'Intersect', 'Mask', 'Masked', 'Pred change']
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=9, fontweight='bold')

for row_i, (c, pos) in enumerate(cam_select):
    img = attacked_by_lang['en'][pos]
    ce1, cz1 = ce[pos], cz[pos]
    pe, pz = int(preds_attacked['en']['en'][pos]), int(preds_attacked['en']['zh'][pos])
    a, b = align_cams(ce1, cz1)
    inter = intersection_map(ce1, cz1, mode='min')
    mask = cam_to_mask(inter, threshold=BEST_THRESHOLD, dilate=3)
    masked = apply_mask(img, mask)
    new_en = classify(models['en'], [masked], CLASSES['en'])[0]
    new_zh = classify(models['zh'], [masked], CLASSES['zh'])[0]
    panels = [img, overlay_cam(img, a), overlay_cam(img, b), overlay_cam(img, inter),
              mask_overlay(img, mask), masked]
    for col_i, panel in enumerate(panels):
        ax = axes[row_i, col_i]
        ax.imshow(panel); ax.axis('off')
        if col_i == 0:
            ax.set_ylabel(CLASSES['en'][c], fontsize=8, rotation=0, labelpad=36, va='center')
    ax = axes[row_i, 6]
    ax.axis('off')
    txt = (
        f"EN: {CLASSES['en'][pe][:6]}->{CLASSES['en'][new_en][:6]}\\n"
        f"ZH: {CLASSES['en'][pz][:6]}->{CLASSES['en'][new_zh][:6]}\\n"
        f"true: {CLASSES['en'][true[pos]]}"
    )
    color = 'green' if new_en == true[pos] or new_zh == true[pos] else 'red'
    ax.text(0.05, 0.5, txt, transform=ax.transAxes, fontsize=7, va='center', color=color)

plt.suptitle('CAM intersection defense examples (EN attack, dual random placement)', fontsize=12, y=1.002)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/mask_examples.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved -> {RESULTS_DIR}/accuracy_delta_matrix.png')
print(f'Saved -> {RESULTS_DIR}/mask_examples.png')
"""))

    cells.append(md("""
## Interpretation notes

- **Random dual placement:** two non-overlapping text boxes per image; CAM intersection must localize both without a bottom-band prior.
- **EN attack:** both models may still co-attend the text region when fooled, but saliency may be more diffuse than bottom-fixed attacks.
- **ZH attack:** the EN model may not attend strongly to Chinese glyphs; ZH-only masking may outperform intersection.
- **Clean images:** both models attend to the object; intersection may mask salient object regions — report clean degradation.
"""))

    cells.append(md("## 10. Save results JSON"))
    cells.append(code("""
out_path = f'{RESULTS_DIR}/confusion_results_cam_defense.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Saved -> {out_path}')
"""))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    out = HERE / "cam_intersection_defense.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build_baseline_notebook()
    build_defense_notebook()
