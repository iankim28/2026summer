"""Regenerate dual-box preview + sample_viz.png without running the full notebook."""
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
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LANGS = ['en', 'zh']
CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
}
TMPL = {'en': 'a photo of a {}.', 'zh': '一张{}的照片。'}
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DISPLAY_SIZE = 224
FONT_SIZE_SINGLE = 40
NUM_BOXES = 2
FONT_SIZE = 24
PAD = 8
_FONT_CACHE = {}


def classify(model, imgs, words):
    imf = model.embed_images(imgs)
    tf = model.embed_texts(words)
    return (imf @ tf.t()).argmax(-1).cpu().numpy()


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


def build_attacked_images(base_imgs, img_indices, attack_lang, target, n_workers=None):
    words = [CLASSES[attack_lang][target[k]] for k in img_indices]
    n_workers = n_workers or min(8, os.cpu_count() or 4)

    def _one(args):
        im, word, img_idx = args
        return draw_word(im, word, img_idx=img_idx, already_224=True)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one, zip(base_imgs, words, img_indices)))


def pred_labels_2(ax, preds_by_model, array_idx, true_class, y=-0.10):
    for j, ml in enumerate(LANGS):
        pred = int(preds_by_model[ml][array_idx])
        ok = (pred == true_class)
        col = '#1a7a1a' if ok else '#cc1111'
        sym = '\u2713' if ok else '\u2717'
        ax.text((j + 0.5) / len(LANGS), y,
                f"{ml.upper()}: {CLASSES['en'][pred][:4]}{sym}",
                transform=ax.transAxes, fontsize=7, color=col,
                ha='center', va='top', clip_on=False)


def main():
    t0 = time.time()
    print('Device:', DEVICE)

    hf = load_dataset('uoft-cs/cifar10', split='test')
    label_key = 'label' if 'label' in hf.column_names else 'labels'
    image_key = 'img' if 'img' in hf.column_names else 'image'
    with open('../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json', encoding='utf-8') as f:
        saved = json.load(f)
    rows = hf.select(saved['idx'])
    true = np.array(rows[label_key])
    rng = random.Random(0)
    target = np.array([rng.choice([c for c in range(10) if c != int(true[k])]) for k in range(len(true))])
    clean = [im.convert('RGB') for im in rows[image_key]]
    clean_224 = [im.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC) for im in clean]
    all_idx = np.arange(len(clean))

    preview_idx = [int(np.where(true == c)[0][0]) for c in range(4)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f'Dual-box attack preview ({NUM_BOXES} boxes @ size {FONT_SIZE})', fontsize=12)
    for col, pos in enumerate(preview_idx):
        word_en = CLASSES['en'][target[pos]]
        word_zh = CLASSES['zh'][target[pos]]
        axes[0, col].imshow(draw_word(clean_224[pos], word_en, img_idx=pos, already_224=True))
        axes[0, col].set_title(f'EN: {word_en}', fontsize=9)
        axes[0, col].axis('off')
        axes[1, col].imshow(draw_word(clean_224[pos], word_zh, img_idx=pos + 1000, already_224=True))
        axes[1, col].set_title(f'ZH: {word_zh}', fontsize=9)
        axes[1, col].axis('off')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/dual_box_preview.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {RESULTS_DIR}/dual_box_preview.png')

    models = {lang: cls() for lang, cls in {'en': EnCLIP, 'zh': ZhCLIP}.items()}
    clean_preds = {lang: classify(models[lang], clean, CLASSES[lang]) for lang in LANGS}

    attacked_by_lang = {}
    preds_attacked = {}
    for attack_lang in LANGS:
        print(f'Building {attack_lang} dual-box attack...')
        attacked_by_lang[attack_lang] = build_attacked_images(clean_224, all_idx, attack_lang, target)
        preds_attacked[attack_lang] = {
            ml: classify(models[ml], attacked_by_lang[attack_lang], CLASSES[ml]) for ml in LANGS
        }

    select = [(c, int(np.where(true == c)[0][0])) for c in range(10)]
    fig, axes = plt.subplots(10, 3, figsize=(10, 36))
    fig.suptitle(
        'EN vs ZH models — clean, EN-attacked, ZH-attacked\n'
        'balanced sample (100 per class), dual random placement (2 boxes @ size 24)   green = correct   red = fooled',
        fontsize=12, y=1.005)
    for ax, t in zip(axes[0], ['Clean', 'EN attack', 'ZH attack']):
        ax.set_title(t, fontsize=11, fontweight='bold', pad=6)
    for row_i, (c, pos) in enumerate(select):
        true_class = int(true[pos])
        images = [clean_224[pos], attacked_by_lang['en'][pos], attacked_by_lang['zh'][pos]]
        sources = [
            {ml: clean_preds[ml] for ml in LANGS},
            {ml: preds_attacked['en'][ml] for ml in LANGS},
            {ml: preds_attacked['zh'][ml] for ml in LANGS},
        ]
        for col_i, (img, src) in enumerate(zip(images, sources)):
            ax = axes[row_i, col_i]
            ax.imshow(img, interpolation='nearest')
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(CLASSES['en'][c], fontsize=9, rotation=0, labelpad=50, va='center')
            pred_labels_2(ax, src, pos, true_class)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/sample_viz.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {RESULTS_DIR}/sample_viz.png')
    print(f'Done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
