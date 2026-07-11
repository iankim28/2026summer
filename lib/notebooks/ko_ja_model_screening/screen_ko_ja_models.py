"""Screen Korean / Japanese CLIP models on balanced CIFAR-10 clean accuracy.

Compares candidates against EN (OpenAI ViT-B/32) and ZH (Chinese CLIP) baselines
on the same 1000-image balanced subset used by balanced_typographic_comparison.ipynb.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 64

CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
    'ko': ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭'],
    'ja': ['飛行機', '自動車', '鳥', '猫', '鹿', '犬', 'カエル', '馬', '船', 'トラック'],
}
TMPL = {
    'en': 'a photo of a {}.',
    'zh': '一张{}的照片。',
    'ko': '{}의 사진.',
    'ja': '{}の写真。',
}


def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, 'pooler_output', None) is not None:
        return out.pooler_output
    raise TypeError(type(out))


@dataclass
class ModelSpec:
    key: str
    lang: str
    label: str
    hf_id: str
    loader: str  # open_clip | hf_clip | hf_ja_custom
    notes: str = ''
    current: bool = False


CANDIDATES = [
    # Baselines
    ModelSpec('en_openai', 'en', 'OpenAI ViT-B/32', 'openai', 'open_clip',
              notes='EN baseline', current=False),
    ModelSpec('zh_chinese', 'zh', 'Chinese CLIP ViT-B/16', 'OFA-Sys/chinese-clip-vit-base-patch16',
              'hf_clip', notes='ZH baseline', current=False),
    # Korean
    ModelSpec('ko_bingsu_b32', 'ko', 'Bingsu ViT-B/32', 'Bingsu/clip-vit-base-patch32-ko',
              'hf_clip', notes='Current KO model in 4-lang notebook', current=True),
    ModelSpec('ko_bingsu_l14', 'ko', 'Bingsu ViT-L/14', 'Bingsu/clip-vit-large-patch14-ko',
              'hf_clip', notes='Larger Bingsu distillation model'),
    # Japanese
    ModelSpec('ja_llmjp_b16', 'ja', 'llm-jp ViT-B/16', 'llm-jp/llm-jp-clip-vit-base-patch16',
              'open_clip', notes='Current JA model in 4-lang notebook', current=True),
    ModelSpec('ja_llmjp_l14', 'ja', 'llm-jp ViT-L/14', 'llm-jp/llm-jp-clip-vit-large-patch14',
              'open_clip', notes='Larger llm-jp; 96.4% CIFAR-10 in model card'),
    ModelSpec('ja_ly_v2', 'ja', 'LY clip-japanese-base-v2', 'line-corporation/clip-japanese-base-v2',
              'hf_ja_custom', notes='~2B pairs + distillation; strong Recruit/WAON scores'),
    ModelSpec('ja_stable_l16', 'ja', 'Stability JA ViT-L/16', 'stabilityai/japanese-stable-clip-vit-l-16',
              'hf_ja_custom', notes='Gated HF repo; 97.6% CIFAR-10 in llm-jp table'),
]


class ModelWrapper:
    lang: str

    def embed_images(self, imgs):
        raise NotImplementedError

    def embed_texts(self, words):
        raise NotImplementedError


def load_wrapper(spec: ModelSpec) -> ModelWrapper:
    if spec.loader == 'open_clip':
        import open_clip

        hub = f'hf-hub:{spec.hf_id}' if spec.key != 'en_openai' else 'ViT-B-32'
        pretrained = spec.hf_id if spec.key == 'en_openai' else None
        if spec.key == 'en_openai':
            m, _, pp = open_clip.create_model_and_transforms(hub, pretrained=pretrained)
            tok = open_clip.get_tokenizer(hub)
        else:
            m, _, pp = open_clip.create_model_and_transforms(hub)
            tok = open_clip.get_tokenizer(hub)
        m = m.to(DEVICE).eval()

        class W(ModelWrapper):
            lang = spec.lang
            def embed_images(self, imgs):
                x = torch.stack([pp(im) for im in imgs]).to(DEVICE)
                with torch.no_grad():
                    return F.normalize(m.encode_image(x), dim=-1)
            def embed_texts(self, words):
                t = tok([TMPL[spec.lang].format(w) for w in words]).to(DEVICE)
                with torch.no_grad():
                    return F.normalize(m.encode_text(t), dim=-1)
        return W()

    if spec.loader == 'hf_clip':
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(spec.hf_id).to(DEVICE).eval()
        proc = AutoProcessor.from_pretrained(spec.hf_id)

        class W(ModelWrapper):
            lang = spec.lang
            def embed_images(self, imgs):
                pv = proc(images=imgs, return_tensors='pt').pixel_values.to(DEVICE)
                with torch.no_grad():
                    out = model.get_image_features(pixel_values=pv)
                    return F.normalize(_clip_feat(out), dim=-1)
            def embed_texts(self, words):
                t = proc(text=[TMPL[spec.lang].format(w) for w in words],
                         padding=True, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model.get_text_features(
                        input_ids=t['input_ids'], attention_mask=t['attention_mask'])
                    return F.normalize(_clip_feat(out), dim=-1)
        return W()

    if spec.loader == 'hf_ja_custom':
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model = AutoModel.from_pretrained(spec.hf_id, trust_remote_code=True).to(DEVICE).eval()
        if spec.key == 'ja_stable_l16':
            proc = AutoProcessor.from_pretrained(spec.hf_id)
            tok = None
        elif spec.key == 'ja_ly_v2':
            proc = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
        else:
            proc = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)

        class W(ModelWrapper):
            lang = spec.lang
            def embed_images(self, imgs):
                pv = proc(images=imgs, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    if hasattr(model, 'get_image_features'):
                        out = model.get_image_features(**pv)
                    else:
                        out = model.encode_image(pv['pixel_values'])
                    return F.normalize(_clip_feat(out) if not torch.is_tensor(out) else out, dim=-1)
            def embed_texts(self, words):
                texts = [TMPL[spec.lang].format(w) for w in words]
                with torch.no_grad():
                    if tok is not None:
                        t = tok(texts).to(DEVICE)
                        out = model.get_text_features(**t)
                    else:
                        t = proc(text=texts, return_tensors='pt', padding=True).to(DEVICE)
                        out = model.get_text_features(**t)
                    feat = out if torch.is_tensor(out) else _clip_feat(out)
                    return F.normalize(feat, dim=-1)
        return W()

    raise ValueError(spec.loader)


def classify(model: ModelWrapper, imgs, words):
    words_list = list(words)
    text_emb = model.embed_texts(words_list)
    chunks = []
    for i in range(0, len(imgs), BATCH):
        imf = model.embed_images(imgs[i:i + BATCH])
        chunks.append((imf @ text_emb.t()).argmax(-1).cpu().numpy())
    return np.concatenate(chunks)


def load_balanced_sample():
    path = ROOT / 'image_samples' / 'CIFAR10_BALANCED_1000_SAMPLE.json'
    with open(path, encoding='utf-8') as f:
        saved = json.load(f)
    hf = load_dataset('uoft-cs/cifar10', split='test')
    label_key = 'label' if 'label' in hf.column_names else 'labels'
    image_key = 'img' if 'img' in hf.column_names else 'image'
    rows = hf.select(saved['idx'])
    true = np.array(rows[label_key])
    clean = [im.convert('RGB') for im in rows[image_key]]
    assert all((true == c).sum() == 100 for c in range(10))
    return clean, true


def per_class_acc(preds, true):
    return {c: float((preds[true == c] == c).mean()) for c in range(10)}


def main():
    print(f'Device: {DEVICE}')
    clean, true = load_balanced_sample()
    print(f'Loaded {len(clean)} balanced CIFAR-10 images\n')

    results = []
    for spec in CANDIDATES:
        row = {
            'key': spec.key,
            'lang': spec.lang,
            'label': spec.label,
            'hf_id': spec.hf_id,
            'notes': spec.notes,
            'current': spec.current,
        }
        print(f'--- {spec.label} ({spec.hf_id}) ---')
        t0 = time.time()
        try:
            model = load_wrapper(spec)
            preds = classify(model, clean, CLASSES[spec.lang])
            acc = float((preds == true).mean())
            pc = per_class_acc(preds, true)
            row.update({
                'status': 'ok',
                'clean_acc': acc,
                'per_class': pc,
                'load_eval_s': round(time.time() - t0, 1),
            })
            print(f'  clean acc: {100*acc:.1f}%  ({row["load_eval_s"]}s)')
        except Exception as e:
            row.update({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'load_eval_s': round(time.time() - t0, 1),
            })
            print(f'  FAILED: {e}')
        results.append(row)
        # free GPU memory between large models
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    out = {
        'sample': 'CIFAR10_BALANCED_1000_SAMPLE',
        'n_images': len(clean),
        'device': DEVICE,
        'classes_en': CLASSES['en'],
        'models': results,
    }
    out_path = RESULTS_DIR / 'screening_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 72)
    print('CLEAN ACCURACY — balanced 1000-image CIFAR-10 (100 per class)')
    print('=' * 72)
    hdr = f'{"Model":<28} {"Lang":>4} {"Acc":>7}  {"vs ZH":>7}  Notes'
    print(hdr)
    print('-' * len(hdr))

    zh_acc = next((r['clean_acc'] for r in results if r['key'] == 'zh_chinese' and r['status'] == 'ok'), None)
    for r in results:
        if r['status'] != 'ok':
            print(f'{r["label"]:<28} {r["lang"]:>4} {"ERROR":>7}  {"":>7}  {r.get("error", "")[:40]}')
            continue
        delta = ''
        if zh_acc is not None and r['lang'] in ('ko', 'ja'):
            delta = f'{100*(r["clean_acc"] - zh_acc):+6.1f}pp'
        cur = ' [current]' if r.get('current') else ''
        print(f'{r["label"]:<28} {r["lang"]:>4} {100*r["clean_acc"]:6.1f}%  {delta:>7}  {r["notes"][:35]}{cur}')

    print(f'\nSaved -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
