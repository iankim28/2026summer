"""Fast completion runner — uses existing CAM caches, no GradCAM recompute."""
import importlib
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
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
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LANGS = ['en', 'zh']
CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
}
TMPL = {'en': 'a photo of a {}.', 'zh': '一张{}的照片。'}
DISPLAY_SIZE = 224
FONT_SIZE = 40
RESULTS_DIR = 'results'
CACHE_DIR = os.path.join(RESULTS_DIR, 'cache')
os.makedirs(RESULTS_DIR, exist_ok=True)


def classify(model, imgs, words, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i:i + batch_size])
        tf = model.embed_texts(words)
        preds.append((imf @ tf.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, 'pooler_output', None) is not None:
        return out.pooler_output
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
        out = self.m.get_text_features(
            input_ids=t['input_ids'], attention_mask=t['attention_mask'],
            token_type_ids=t.get('token_type_ids'))
        return F.normalize(_clip_feat(out), dim=-1)


_FONT_CACHE = {}

def _font_paths():
    if platform.system() == 'Windows':
        winfonts = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        cjk = os.path.join(winfonts, 'msyh.ttc')
        latin = os.path.join(winfonts, 'arial.ttf')
        if not os.path.exists(latin):
            latin = cjk
        return (cjk if os.path.exists(cjk) else None,
                latin if os.path.exists(latin) else None)
    for d in ['/usr/share/fonts', '/Library/Fonts', os.path.expanduser('~/.fonts')]:
        for f in ['NotoSansCJK-Regular.ttc', 'NotoSans-Regular.ttf']:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return p, p
    return None, None

_CJK_FONT, _LAT_FONT = _font_paths()

def _font_for(word):
    return _CJK_FONT if any(ord(c) > 127 for c in word) else _LAT_FONT

def _get_font(fp):
    key = fp or '__default__'
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, FONT_SIZE) if fp else ImageFont.load_default()
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def draw_word(img, word, where='bottom', already_224=False):
    fp = _font_for(word)
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    font = _get_font(fp)
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), word, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (DISPLAY_SIZE - tw) // 2
    y = (DISPLAY_SIZE - th - 16) if where == 'bottom' else (DISPLAY_SIZE - th) // 2
    draw.rectangle([x - 8, y - 8, x + tw + 8, y + th + 12], fill='white')
    draw.text((x - bb[0], y - bb[1]), word, fill='black', font=font)
    return img

def build_attacked_images(base_imgs, attack_lang, target, n_workers=None):
    words = [CLASSES[attack_lang][target[k]] for k in range(len(base_imgs))]
    n_workers = n_workers or min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(lambda pair: draw_word(pair[0], pair[1], already_224=True),
                             zip(base_imgs, words)))


def align_cams(cam_en, cam_zh, size=DISPLAY_SIZE):
    def _resize(cam):
        return np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    return _resize(cam_en), _resize(cam_zh)

def intersection_map(cam_en, cam_zh, mode='min'):
    a, b = align_cams(cam_en, cam_zh)
    return np.minimum(a, b) if mode == 'min' else (a * b)

def bottom_band_mask(size=DISPLAY_SIZE, fraction=0.25):
    mask = np.zeros((size, size), dtype=bool)
    mask[int(size * (1 - fraction)):, :] = True
    return mask

BOTTOM_PRIOR = bottom_band_mask()

def cam_to_mask(saliency, method='percentile', threshold=0.85, dilate=3, spatial_prior=None):
    if method == 'oracle_bottom':
        mask = BOTTOM_PRIOR.copy()
    else:
        sal = saliency.copy()
        if spatial_prior is not None:
            sal = sal * spatial_prior.astype(np.float32)
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
    red = np.zeros_like(arr)
    red[:, :, 0] = 255
    m = mask.astype(np.float32)[..., None]
    blended = arr * (1 - alpha * m) + red * (alpha * m)
    return Image.fromarray(blended.astype(np.uint8))

def load_cam_cache(condition, n):
    path = os.path.join(CACHE_DIR, f'cams_{condition}_n{n}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing cache: {path}')
    data = np.load(path)
    return data['cam_en'], data['cam_zh']

def build_masks_from_cams(cam_en, cam_zh, config):
    masks, coverages = [], []
    for ce, cz in zip(cam_en, cam_zh):
        if config.get('strategy') == 'en_only':
            sal, _ = align_cams(ce, ce)
        elif config.get('strategy') == 'zh_only':
            _, sal = align_cams(cz, cz)
        elif config.get('strategy') == 'oracle_bottom':
            mask = cam_to_mask(None, method='oracle_bottom', dilate=0)
            masks.append(mask)
            coverages.append(mask.mean())
            continue
        else:
            sal = intersection_map(ce, cz, mode=config.get('mode', 'min'))
        prior = BOTTOM_PRIOR if config.get('use_bottom_prior') else None
        mask = cam_to_mask(sal, threshold=config.get('threshold', 0.85),
                           dilate=config.get('dilate', 3), spatial_prior=prior)
        masks.append(mask)
        coverages.append(mask.mean())
    return masks, np.array(coverages)

def eval_config(imgs, cam_en, cam_zh, config, true, target):
    masks, coverages = build_masks_from_cams(cam_en, cam_zh, config)
    masked = [apply_mask(img, m) for img, m in zip(imgs, masks)]
    preds = {ml: classify(models[ml], masked, CLASSES[ml]) for ml in LANGS}
    out = {'coverage_mean': float(coverages.mean()), 'models': {}}
    for ml in LANGS:
        p = preds[ml]
        out['models'][ml] = {'acc': float((p == true).mean()), 'asr': float((p == target).mean())}
    return out, masked, masks, preds


def main():
    global models
    t0 = time.time()
    print('Device:', DEVICE)

    hf = load_dataset('uoft-cs/cifar10', split='test')
    label_key = 'label' if 'label' in hf.column_names else 'labels'
    image_key = 'img' if 'img' in hf.column_names else 'image'
    with open('../../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json', encoding='utf-8') as f:
        saved = json.load(f)
    rows = hf.select(saved['idx'])
    true = np.array(rows[label_key])
    rng = random.Random(0)
    target = np.array([rng.choice([c for c in range(10) if c != int(true[k])]) for k in range(len(true))])
    clean = [im.convert('RGB') for im in rows[image_key]]
    clean_224 = [im.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC) for im in clean]
    tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])

    models = {lang: cls() for lang, cls in {'en': EnCLIP, 'zh': ZhCLIP}.items()}

    clean_preds = {lang: classify(models[lang], clean, CLASSES[lang]) for lang in LANGS}
    clean_acc = {lang: float((clean_preds[lang] == true).mean()) for lang in LANGS}

    attacked_by_lang = {}
    preds_attacked = {}
    for attack_lang in LANGS:
        attacked_by_lang[attack_lang] = build_attacked_images(clean_224, attack_lang, target)
        preds_attacked[attack_lang] = {
            ml: classify(models[ml], attacked_by_lang[attack_lang], CLASSES[ml]) for ml in LANGS
        }

    n = len(LANGS)
    acc_matrix = np.zeros((n, n))
    asr_matrix = np.zeros((n, n))
    for i, al in enumerate(LANGS):
        for j, ml in enumerate(LANGS):
            p = preds_attacked[al][ml]
            acc_matrix[i, j] = (p == true).mean()
            asr_matrix[i, j] = (p == target).mean()

    print('Baseline EN attack EN model acc:', f'{100*acc_matrix[0,0]:.1f}%')

    # Threshold sweep from cached tune CAMs (instant)
    THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
    sweep_rows = []
    for cond in ['en_attack', 'zh_attack']:
        atk = cond.replace('_attack', '')
        imgs = [attacked_by_lang[atk][i] for i in tune_idx]
        ce, cz = load_cam_cache(cond, len(tune_idx))
        t_sub, tgt_sub = true[tune_idx], target[tune_idx]
        base = {ml: float((preds_attacked[atk][ml][tune_idx] == t_sub).mean()) for ml in LANGS}
        for thr in THRESHOLDS:
            cfg = {'strategy': 'intersection', 'mode': 'min', 'threshold': thr, 'dilate': 3}
            res, _, _, _ = eval_config(imgs, ce, cz, cfg, t_sub, tgt_sub)
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
    print(f'Best threshold: {BEST_THRESHOLD} ({100*best_row["baseline_acc"]:.1f}% -> {100*best_row["masked_acc"]:.1f}%)')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cond in zip(axes, ['en_attack', 'zh_attack']):
        for ml, color in zip(LANGS, ['C0', 'C1']):
            xs, ys, ys_asr = [], [], []
            for thr in THRESHOLDS:
                row = next(r for r in sweep_rows if r['condition'] == cond and r['model'] == ml and r['threshold'] == thr)
                xs.append(thr)
                ys.append(row['masked_acc'])
                ys_asr.append(row['masked_asr'])
            ax.plot(xs, [100 * y for y in ys], 'o-', color=color, label=f'model_{ml} acc')
            ax.plot(xs, [100 * y for y in ys_asr], 's--', color=color, alpha=0.6, label=f'model_{ml} ASR')
        ax.set_title(cond.replace('_', ' ').title())
        ax.set_xlabel('Percentile threshold')
        ax.set_ylabel('%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Threshold sweep (100-image tune subset)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/threshold_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    DEFAULT_CFG = {'strategy': 'intersection', 'mode': 'min', 'threshold': BEST_THRESHOLD, 'dilate': 3}
    ABLATIONS = {
        'intersection_min': DEFAULT_CFG,
        'intersection_product': {**DEFAULT_CFG, 'mode': 'product'},
        'en_only': {**DEFAULT_CFG, 'strategy': 'en_only'},
        'zh_only': {**DEFAULT_CFG, 'strategy': 'zh_only'},
        'bottom_prior': {**DEFAULT_CFG, 'use_bottom_prior': True},
        'oracle_bottom': {'strategy': 'oracle_bottom', 'threshold': 0, 'dilate': 0},
    }

    results = {
        'sample': 'CIFAR10_BALANCED_1000_SAMPLE',
        'best_threshold_tune': BEST_THRESHOLD,
        'default_config': DEFAULT_CFG,
        'baseline': {'clean_acc': clean_acc, 'acc_matrix': acc_matrix.tolist(), 'asr_matrix': asr_matrix.tolist()},
        'defense': {},
        'ablations': {},
        'clean_degradation': {},
        'threshold_sweep': sweep_rows,
    }

    full_cache = {}
    for cond in ['en_attack', 'zh_attack', 'clean']:
        full_cache[cond] = load_cam_cache(cond, len(clean))

    for cond in ['en_attack', 'zh_attack']:
        atk = cond.replace('_attack', '')
        imgs = attacked_by_lang[atk]
        ce, cz = full_cache[cond]
        base_preds = {ml: preds_attacked[atk][ml] for ml in LANGS}
        res, _, _, new_preds = eval_config(imgs, ce, cz, DEFAULT_CFG, true, target)
        cell = {}
        for ml in LANGS:
            base_p, new_p = base_preds[ml], new_preds[ml]
            wrong = base_p != true
            recovered = wrong & (new_p == true)
            cell[ml] = {
                **res['models'][ml],
                'recovery_rate': float(recovered.sum() / wrong.sum()) if wrong.any() else 0.0,
                'baseline_acc': float((base_p == true).mean()),
                'baseline_asr': float((base_p == target).mean()),
            }
        results['defense'][cond] = {'coverage_mean': res['coverage_mean'], 'models': cell}

    ce, cz = full_cache['clean']
    clean_res, _, _, _ = eval_config(clean_224, ce, cz, DEFAULT_CFG, true, target)
    results['clean_degradation'] = {
        'coverage_mean': clean_res['coverage_mean'],
        'models': {
            ml: {
                'baseline_acc': clean_acc[ml],
                'masked_acc': clean_res['models'][ml]['acc'],
                'delta_acc': clean_res['models'][ml]['acc'] - clean_acc[ml],
            }
            for ml in LANGS
        },
    }

    atk_imgs = attacked_by_lang['en']
    ce, cz = full_cache['en_attack']
    for name, cfg in ABLATIONS.items():
        res, _, _, _ = eval_config(atk_imgs, ce, cz, cfg, true, target)
        results['ablations'][name] = {'coverage_mean': res['coverage_mean'], 'models': res['models']}

    print('Defense EN attack:')
    for ml in LANGS:
        d = results['defense']['en_attack']['models'][ml]
        print(f"  model_{ml}: {100*d['baseline_acc']:.1f}% -> {100*d['acc']:.1f}%  recovery={100*d['recovery_rate']:.1f}%")

    delta = np.zeros((n, n))
    for i, al in enumerate(LANGS):
        for j, ml in enumerate(LANGS):
            d = results['defense'][f'{al}_attack']['models'][ml]
            delta[i, j] = d['acc'] - d['baseline_acc']

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(delta * 100, vmin=-20, vmax=60, cmap='RdYlGn')
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'model_{l}' for l in LANGS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'attack_{l}' for l in LANGS])
    ax.set_title('Accuracy delta after CAM masking (pp)')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{delta[i,j]*100:+.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, format=lambda x, _: f'{x:+.0f}pp')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/accuracy_delta_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    ce, cz = full_cache['en_attack']
    cam_select = [(c, int(np.where(true == c)[0][0])) for c in range(10)]
    fig, axes = plt.subplots(10, 7, figsize=(16, 34))
    col_titles = ['Attacked', 'EN CAM', 'ZH CAM', 'Intersect', 'Mask', 'Masked', 'Pred change']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight='bold')

    for row_i, (c, pos) in enumerate(cam_select):
        img = attacked_by_lang['en'][pos]
        ce1, cz1 = ce[pos], cz[pos]
        a, b = align_cams(ce1, cz1)
        inter = intersection_map(ce1, cz1, mode='min')
        mask = cam_to_mask(inter, threshold=BEST_THRESHOLD, dilate=3)
        masked = apply_mask(img, mask)
        new_en = classify(models['en'], [masked], CLASSES['en'])[0]
        new_zh = classify(models['zh'], [masked], CLASSES['zh'])[0]
        pe = int(preds_attacked['en']['en'][pos])
        pz = int(preds_attacked['en']['zh'][pos])
        panels = [img, overlay_cam(img, a), overlay_cam(img, b), overlay_cam(img, inter),
                  mask_overlay(img, mask), masked]
        for col_i, panel in enumerate(panels):
            ax = axes[row_i, col_i]
            ax.imshow(panel)
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(CLASSES['en'][c], fontsize=8, rotation=0, labelpad=36, va='center')
        ax = axes[row_i, 6]
        ax.axis('off')
        txt = (
            f"EN: {CLASSES['en'][pe][:6]}->{CLASSES['en'][new_en][:6]}\n"
            f"ZH: {CLASSES['en'][pz][:6]}->{CLASSES['en'][new_zh][:6]}\n"
            f"true: {CLASSES['en'][true[pos]]}"
        )
        color = 'green' if new_en == true[pos] or new_zh == true[pos] else 'red'
        ax.text(0.05, 0.5, txt, transform=ax.transAxes, fontsize=7, va='center', color=color)

    plt.suptitle('CAM intersection defense examples (EN attack)', fontsize=12, y=1.002)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/mask_examples.png', dpi=120, bbox_inches='tight')
    plt.close()

    out_path = f'{RESULTS_DIR}/confusion_results_cam_defense.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'Saved -> {out_path}')
    print(f'Done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
