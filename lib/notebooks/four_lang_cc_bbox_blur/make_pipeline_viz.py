"""Generate qualitative pipeline panels for cc_bbox_blur.

Outputs (under results/):
  - pipeline_steps.png   — one example, every intermediate stage
  - pipeline_examples.png — several examples, key stages side-by-side
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys

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
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs('results', exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DISPLAY_SIZE = 224
NUM_BOXES = 2
FONT_SIZE = 24
PAD = 8
BLUR_RADIUS = 12
THRESHOLD = 0.95
DILATE = 3
N_EXAMPLES = 5
EXAMPLE_SEED = 0

CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
}
TMPL = {'en': 'a photo of a {}.', 'zh': '一张{}的照片。'}


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, 'pooler_output', None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


class EnCLIP:
    def __init__(self):
        self.m, _, self.pp = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer('ViT-B-32')

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL['en'].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)


class ZhCLIP:
    def __init__(self):
        self.m = ChineseCLIPModel.from_pretrained(
            'OFA-Sys/chinese-clip-vit-base-patch16',
            attn_implementation='eager').to(DEVICE).eval()
        self.p = ChineseCLIPProcessor.from_pretrained(
            'OFA-Sys/chinese-clip-vit-base-patch16')

    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL['zh'].format(w) for w in words], padding=True,
                   return_tensors='pt').to(DEVICE)
        out = self.m.get_text_features(
            input_ids=t['input_ids'],
            attention_mask=t['attention_mask'],
            token_type_ids=t.get('token_type_ids'))
        return F.normalize(_clip_feat(out), dim=-1)


def _font_paths():
    if platform.system() == 'Windows':
        wf = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        return os.path.join(wf, 'msyh.ttc'), os.path.join(wf, 'arial.ttf')
    # WSL / Linux
    cjk_candidates = [
        '/mnt/c/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    lat_candidates = [
        '/mnt/c/Windows/Fonts/arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    cjk = next((p for p in cjk_candidates if os.path.exists(p)), None)
    lat = next((p for p in lat_candidates if os.path.exists(p)), None)
    return cjk, lat


_FONT_CACHE = {}
_CJK_FONT, _LAT_FONT = _font_paths()
attack_pos = None


def _get_font(fp, size=FONT_SIZE):
    key = (fp or '__default__', size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = (
                ImageFont.truetype(fp, size) if fp else ImageFont.load_default())
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y


def draw_multilingual_attack(img, en_word, zh_word, img_idx):
    img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    xy0 = attack_pos['en'][int(img_idx)]
    xy1 = attack_pos['l'][int(img_idx)]
    for (word, fp), xy in zip([(en_word, _LAT_FONT), (zh_word, _CJK_FONT)], [xy0, xy1]):
        font = _get_font(fp)
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill='white')
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill='black', font=font)
    return img


def _norm_cam(cam):
    cam = np.maximum(cam if isinstance(cam, np.ndarray) else cam.cpu().numpy(), 0)
    cam -= cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def align_cam(cam, size=DISPLAY_SIZE):
    return np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0


def _make_en_hook(collector):
    def hook(module, inputs, output):
        q_in = inputs[0]
        if getattr(module, 'batch_first', False):
            B, L, D = q_in.shape
        else:
            L, B, D = q_in.shape
            q_in = q_in.transpose(0, 1).contiguous()
        n_head = module.num_heads
        hd = D // n_head
        with torch.no_grad():
            qkv = F.linear(q_in, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.softmax(dim=-1)
        collector.append(attn[0].detach().cpu())
    return hook


def _build_attn_cam(all_attns):
    a = all_attns[-1]
    cls_row = a.mean(0)[0, 1:]
    n = int(round(cls_row.shape[0] ** 0.5))
    return _norm_cam(cls_row.reshape(n, n).numpy())


def classify_and_attn_en(wrapper, text_emb, pil_img):
    x = wrapper.pp(pil_img).unsqueeze(0).to(DEVICE)
    collector = []
    handles = [rb.attn.register_forward_hook(_make_en_hook(collector))
               for rb in wrapper.m.visual.transformer.resblocks]
    with torch.no_grad():
        feat = wrapper.m.visual(x)
        imf = F.normalize(feat, dim=-1)
        pred = int((imf @ text_emb.T).squeeze().argmax().item())
    for h in handles:
        h.remove()
    return pred, _build_attn_cam(collector)


def classify_and_attn_zh(wrapper, text_emb, pil_img):
    pv = wrapper.p(images=[pil_img], return_tensors='pt').pixel_values.to(DEVICE)
    with torch.no_grad():
        vis_out = wrapper.m.vision_model(pixel_values=pv, output_attentions=True)
        proj_out = wrapper.m.visual_projection(vis_out.pooler_output)
        imf = F.normalize(proj_out, dim=-1)
        pred = int((imf @ text_emb.T).squeeze().argmax().item())
    attns = [a[0].cpu() for a in vis_out.attentions]
    return pred, _build_attn_cam(attns)


def n_cam_intersection(*cams):
    return np.minimum.reduce([align_cam(c) for c in cams])


def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        m = (pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] |
             pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
             pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:])
    return m


def cam_to_mask(saliency, threshold=0.85, dilate=3):
    thr = np.percentile(saliency, threshold * 100)
    mask = saliency >= thr
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask


def filter_mask_components(mask, top_k=2, bbox_snap=False):
    labeled, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = set(np.argsort(sizes)[::-1][:top_k] + 1)
    out = np.zeros_like(mask, dtype=bool)
    for i in keep:
        comp = labeled == i
        if bbox_snap:
            ys, xs = np.where(comp)
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        else:
            out |= comp
    return out


def apply_mask(pil_img, mask, fill='blur'):
    arr = np.array(pil_img.convert('RGB'))
    m = mask.astype(bool)
    if mask.shape != arr.shape[:2]:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            arr.shape[1::-1], Image.NEAREST)) > 127
    out = arr.copy()
    if fill == 'blur':
        blurred = np.array(Image.fromarray(arr).filter(
            ImageFilter.GaussianBlur(radius=BLUR_RADIUS)))
        out[m] = blurred[m]
    else:
        mean = arr[~m].mean(0) if (~m).any() else arr.reshape(-1, 3).mean(0)
        out[m] = mean
    return Image.fromarray(out.astype(np.uint8))


def overlay_heatmap(pil_img, cam, alpha=0.45):
    base = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
    heat = cm.jet(align_cam(cam))[:, :, :3]
    mix = (1 - alpha) * base + alpha * heat
    return Image.fromarray((mix * 255).astype(np.uint8))


def overlay_mask(pil_img, mask, color=(255, 40, 40), alpha=0.55):
    arr = np.array(pil_img.convert('RGB')).astype(np.float32)
    m = mask.astype(bool)
    tint = np.zeros_like(arr)
    tint[:] = color
    arr[m] = (1 - alpha) * arr[m] + alpha * tint[m]
    return Image.fromarray(arr.astype(np.uint8))


def pipeline_stages(img, cam_en, cam_zh):
    inter = n_cam_intersection(cam_en, cam_zh)
    raw = cam_to_mask(inter, THRESHOLD, dilate=DILATE)
    cc_only = filter_mask_components(raw, top_k=2, bbox_snap=False)
    cc_bbox = filter_mask_components(raw, top_k=2, bbox_snap=True)
    defended = apply_mask(img, cc_bbox, fill='blur')
    mean_fill = apply_mask(img, cc_bbox, fill='mean')
    return {
        'attacked': img,
        'attn_en': overlay_heatmap(img, cam_en),
        'attn_zh': overlay_heatmap(img, cam_zh),
        'intersection': overlay_heatmap(img, inter),
        'raw_mask': overlay_mask(img, raw),
        'cc_only': overlay_mask(img, cc_only),
        'cc_bbox': overlay_mask(img, cc_bbox),
        'mean_fill': mean_fill,
        'cc_bbox_blur': defended,
        'raw_mask_bool': raw,
        'cc_bbox_bool': cc_bbox,
    }


def main():
    print('Device:', DEVICE)
    print('Fonts:', _CJK_FONT, _LAT_FONT)

    print('Loading models...')
    en = EnCLIP()
    zh = ZhCLIP()
    text_en = en.embed_texts(CLASSES['en']).detach()
    text_zh = zh.embed_texts(CLASSES['zh']).detach()

    hf = load_dataset('uoft-cs/cifar10', split='test')
    label_key = 'label' if 'label' in hf.column_names else 'labels'
    image_key = 'img' if 'img' in hf.column_names else 'image'
    sample_path = os.path.join(HERE, '..', 'image_samples',
                               'CIFAR10_BALANCED_1000_SAMPLE.json')
    with open(sample_path, encoding='utf-8') as f:
        saved = json.load(f)
    idx = saved['idx']
    global attack_pos
    attack_pos = saved['attack_pos']
    rows = hf.select(idx)
    true = np.array(rows[label_key])
    rng = random.Random(0)
    target = np.array([
        rng.choice([c for c in range(10) if c != int(true[k])])
        for k in range(len(idx))
    ])

    # Prefer successfully attacked examples that look clear after defense.
    pick = []
    scan = list(range(len(idx)))
    random.Random(EXAMPLE_SEED).shuffle(scan)
    for i in scan:
        if len(pick) >= N_EXAMPLES:
            break
        atk = draw_multilingual_attack(
            rows[i][image_key], CLASSES['en'][target[i]],
            CLASSES['zh'][target[i]], i)
        pred_en, cam_en = classify_and_attn_en(en, text_en, atk)
        pred_zh, cam_zh = classify_and_attn_zh(zh, text_zh, atk)
        # Prefer cases where attack flipped at least one model.
        if pred_en != true[i] or pred_zh != true[i]:
            stages = pipeline_stages(atk, cam_en, cam_zh)
            pick.append((i, true[i], target[i], stages, pred_en, pred_zh))
            print(f'  picked i={i} true={CLASSES["en"][true[i]]} '
                  f'tgt={CLASSES["en"][target[i]]} '
                  f'preds=({CLASSES["en"][pred_en]}, {CLASSES["zh"][pred_zh]})')

    if len(pick) < N_EXAMPLES:
        print('Warning: fewer attacked examples than requested; padding.', file=sys.stderr)
        for i in scan:
            if len(pick) >= N_EXAMPLES:
                break
            if any(p[0] == i for p in pick):
                continue
            atk = draw_multilingual_attack(
                rows[i][image_key], CLASSES['en'][target[i]],
                CLASSES['zh'][target[i]], i)
            pred_en, cam_en = classify_and_attn_en(en, text_en, atk)
            pred_zh, cam_zh = classify_and_attn_zh(zh, text_zh, atk)
            stages = pipeline_stages(atk, cam_en, cam_zh)
            pick.append((i, true[i], target[i], stages, pred_en, pred_zh))

    # --- Full pipeline for first example ---
    step_keys = [
        ('attacked', '1. Attacked\n(EN+ZH stickers)'),
        ('attn_en', '2. Attn-last EN'),
        ('attn_zh', '3. Attn-last ZH'),
        ('intersection', '4. Intersection\nEN ∩ ZH'),
        ('raw_mask', '5. Threshold +\nDilate'),
        ('cc_only', '6. Top-2 CC'),
        ('cc_bbox', '7. BBox snap\n(cc_bbox)'),
        ('cc_bbox_blur', '8. Blur fill\n(cc_bbox_blur)'),
    ]
    i0, t0, tgt0, stages0, _, _ = pick[0]
    fig, axes = plt.subplots(1, len(step_keys), figsize=(2.2 * len(step_keys), 2.8))
    for ax, (key, title) in zip(axes, step_keys):
        ax.imshow(stages0[key])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    fig.suptitle(
        f'cc_bbox_blur pipeline — true={CLASSES["en"][t0]}, '
        f'attack=EN+ZH "{CLASSES["en"][tgt0]}"',
        fontsize=11)
    fig.tight_layout()
    out1 = 'results/pipeline_steps.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out1)

    # --- Multi-example grid (paper-friendly) ---
    grid_keys = [
        ('attacked', 'Attacked'),
        ('attn_en', 'Attn EN'),
        ('attn_zh', 'Attn ZH'),
        ('intersection', 'EN ∩ ZH'),
        ('raw_mask', 'Raw mask'),
        ('cc_bbox', 'CC+bbox'),
        ('cc_bbox_blur', 'Blur fill'),
    ]
    fig, axes = plt.subplots(
        len(pick), len(grid_keys),
        figsize=(2.0 * len(grid_keys), 2.15 * len(pick)))
    if len(pick) == 1:
        axes = np.array([axes])
    for r, (i, t, tgt, stages, pe, pz) in enumerate(pick):
        for c, (key, title) in enumerate(grid_keys):
            ax = axes[r, c]
            ax.imshow(stages[key])
            ax.axis('off')
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(
                    f'{CLASSES["en"][t]}\n→ EN+ZH {CLASSES["en"][tgt]}',
                    fontsize=7, rotation=0, labelpad=36, va='center')
    fig.suptitle(
        'cc_bbox_blur stages on multilingual dual-box attacks (Attn-last)',
        fontsize=12)
    fig.tight_layout()
    out2 = 'results/pipeline_examples.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out2)

    # Compact method figure: mean fill vs blur for one example
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, key, title in zip(
        axes,
        ['attacked', 'cc_bbox', 'mean_fill', 'cc_bbox_blur'],
        ['Attacked', 'Mask (CC+bbox)', 'Mean fill', 'Blur fill (ours)'],
    ):
        ax.imshow(stages0[key])
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle('Fill choice after cc_bbox shaping', fontsize=11)
    fig.tight_layout()
    out3 = 'results/pipeline_fill_compare.png'
    fig.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out3)
    print('Done.')


if __name__ == '__main__':
    main()
