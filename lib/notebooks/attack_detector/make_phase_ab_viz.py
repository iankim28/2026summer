"""Phase A/B gated-occlusion visualization (no CLIP forward if caches exist).

Writes under results/{zh,ko,ja}/multi/:
  - phase_ab_process.png   — clean vs attacked process strip + gate decision
  - phase_b_roc_cm.png     — ROC + confusion at recall≥0.99 threshold
  - phase_ab_viz_summary.json
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

DISPLAY_SIZE = 224
NUM_BOXES = 2
FONT_SIZE = 24
PAD = 8
BLUR_RADIUS = 12
PARTNER_LANGS = ['zh', 'ko', 'ja']
ATTACK = 'multi'
DEFENSE_THR = 0.95
SPLIT_SEED = 0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ATTACK_RECALL_TARGET = 0.99
EXAMPLE_SEED = 0

CLASSES = {
    'en': ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck'],
    'zh': ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'],
    'ko': ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭'],
    'ja': ['飛行機', '自動車', '鳥', '猫', '鹿', '犬', 'カエル', '馬', '船', 'トラック'],
}
LANG_LABEL = {'zh': 'ZH', 'ko': 'KO', 'ja': 'JA'}


# ---------------------------------------------------------------------------
# Fonts / attack render (mirrors _cells/06_data.py)
# ---------------------------------------------------------------------------

def _font_paths():
    if platform.system() == 'Windows':
        wf = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        cjk = os.path.join(wf, 'msyh.ttc')
        lat = os.path.join(wf, 'arial.ttf')
        ko = os.path.join(wf, 'malgun.ttf')
        if not os.path.isfile(ko):
            ko = cjk
        return cjk, lat, ko
    cjk = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    lat = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if not os.path.isfile(cjk):
        cjk = '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf'
    return cjk, lat, cjk


_CJK_FONT, _LAT_FONT, _KO_FONT = _font_paths()
_FONT_CACHE = {}


def _font_for_lang(lang):
    if lang == 'en':
        return _LAT_FONT
    if lang == 'ko':
        return _KO_FONT
    return _CJK_FONT


def _get_font(fp, size=FONT_SIZE):
    key = (fp or '__default__', size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y


def draw_dual_box(img, word0, lang0, word1, lang1, img_idx, attack_pos):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    xy0 = attack_pos['en'][int(img_idx)]
    xy1 = attack_pos['l'][int(img_idx)]
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_font_for_lang(lang))
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill='white')
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill='black', font=font)
    return img


# ---------------------------------------------------------------------------
# Mask / features (mirrors _cells/08, 10, 14)
# ---------------------------------------------------------------------------

def align_cam(cam, size=DISPLAY_SIZE):
    return np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ) / 255.0


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


def build_cc_bbox_blur_mask(cam_en, cam_l, threshold=0.95, dilate=3, top_k=2):
    inter = n_cam_intersection(cam_en, cam_l)
    mask = cam_to_mask(inter, threshold, dilate=dilate)
    return filter_mask_components(mask, top_k=top_k, bbox_snap=True)


def _entropy(p):
    p = p.ravel().astype(np.float64)
    p = p - p.min()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())


def _topk_mass(p, frac):
    flat = p.ravel().astype(np.float64)
    flat = flat - flat.min()
    s = flat.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(round(len(flat) * frac)))
    top = np.partition(flat, -k)[-k:]
    return float(top.sum() / s)


def _gini(p):
    flat = np.sort(p.ravel().astype(np.float64))
    n = len(flat)
    if n == 0 or flat.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * flat).sum()) / (n * flat.sum()) - (n + 1) / n)


def _spatial_kurtosis(p):
    flat = p.ravel().astype(np.float64)
    mu = flat.mean()
    sig = flat.std()
    if sig < 1e-12:
        return 0.0
    z = (flat - mu) / sig
    return float((z ** 4).mean() - 3.0)


def _map_features(cam, prefix):
    a = align_cam(cam)
    mx = float(a.max())
    mn = float(a.mean()) + 1e-12
    cov95 = float((a >= np.percentile(a, 95)).mean())
    _, ncc = ndimage.label(a >= np.percentile(a, 95))
    return {
        f'{prefix}_entropy': _entropy(a),
        f'{prefix}_topk05': _topk_mass(a, 0.05),
        f'{prefix}_topk10': _topk_mass(a, 0.10),
        f'{prefix}_max_over_mean': mx / mn,
        f'{prefix}_gini': _gini(a),
        f'{prefix}_kurtosis': _spatial_kurtosis(a),
        f'{prefix}_cov95': cov95,
        f'{prefix}_ncc95': float(ncc),
    }


def extract_pair_features(cam_en, cam_l):
    feats = {}
    feats.update(_map_features(cam_en, 'en'))
    feats.update(_map_features(cam_l, 'l'))
    inter = n_cam_intersection(cam_en, cam_l)
    feats.update(_map_features(inter, 'inter'))
    ae, al = align_cam(cam_en), align_cam(cam_l)
    feats['en_l_corr'] = (
        float(np.corrcoef(ae.ravel(), al.ravel())[0, 1])
        if ae.std() > 0 and al.std() > 0 else 0.0
    )
    hot_e = ae >= np.percentile(ae, 95)
    hot_l = al >= np.percentile(al, 95)
    union = (hot_e | hot_l).sum()
    feats['en_l_iou95'] = float((hot_e & hot_l).sum() / union) if union > 0 else 0.0
    return feats


def build_feature_matrix(clean_cams, atk_cams, n_images):
    rows_feat, y_labels, img_ids = [], [], []
    for i in range(n_images):
        rows_feat.append(extract_pair_features(clean_cams['cams_en'][i], clean_cams['cams_l'][i]))
        y_labels.append(0)
        img_ids.append(i)
    for i in range(n_images):
        rows_feat.append(extract_pair_features(atk_cams['cams_en'][i], atk_cams['cams_l'][i]))
        y_labels.append(1)
        img_ids.append(i)
    names = list(rows_feat[0].keys())
    X = np.array([[r[k] for k in names] for r in rows_feat], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_labels, dtype=np.int64)
    img_ids = np.array(img_ids, dtype=np.int64)
    return X, y, img_ids, names


# ---------------------------------------------------------------------------
# Cache + gate
# ---------------------------------------------------------------------------

class _ScaledModel:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict_proba(self, X_):
        return self.model.predict_proba(self.scaler.transform(X_))

    def predict(self, X_):
        return self.model.predict(self.scaler.transform(X_))


def load_cache(L, out_dir):
    cache_path = Path(out_dir) / 'cache' / f'attn_en_{L}_clean_multi.npz'
    if not cache_path.exists():
        print(f'ERROR: Attn-last cache missing: {cache_path}', file=sys.stderr)
        print('Baking would require CUDA + full notebook run. Stop and ask before baking.',
              file=sys.stderr)
        sys.exit(1)
    z = np.load(cache_path, allow_pickle=False)

    def _get(prefix, new, old):
        if new in z.files:
            return z[new]
        return z[old]

    return {
        'clean': {
            'cams_en': z['clean_cams_en'],
            'cams_l': _get('clean', 'clean_cams_l', 'clean_cams_zh'),
            'preds_en': z['clean_preds_en'],
            'preds_l': _get('clean', 'clean_preds_l', 'clean_preds_zh'),
        },
        'atk': {
            'cams_en': z['atk_cams_en'],
            'cams_l': _get('atk', 'atk_cams_l', 'atk_cams_zh'),
            'preds_en': z['atk_preds_en'],
            'preds_l': _get('atk', 'atk_preds_l', 'atk_preds_zh'),
        },
    }


def train_gate(X, y, img_ids, n_images):
    rng_split = np.random.RandomState(SPLIT_SEED)
    perm = rng_split.permutation(n_images)
    n_train = int(round(TRAIN_FRAC * n_images))
    n_val = int(round(VAL_FRAC * n_images))
    train_imgs = set(perm[:n_train].tolist())
    val_imgs = set(perm[n_train:n_train + n_val].tolist())
    test_imgs = set(perm[n_train + n_val:].tolist())
    train_m = np.array([i in train_imgs for i in img_ids])
    val_m = np.array([i in val_imgs for i in img_ids])
    test_m = np.array([i in test_imgs for i in img_ids])

    X_tr, y_tr = X[train_m], y[train_m]
    X_va, y_va = X[val_m], y[val_m]
    X_te, y_te = X[test_m], y[test_m]

    logit = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SPLIT_SEED)),
    ])
    logit.fit(X_tr, y_tr)

    _svm_scaler = StandardScaler().fit(X_tr)
    svm_inner = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=5000, random_state=SPLIT_SEED), cv=3)
    svm_inner.fit(_svm_scaler.transform(X_tr), y_tr)
    svm = _ScaledModel(_svm_scaler, svm_inner)

    def _eval(model, X_, y_):
        proba = model.predict_proba(X_)[:, 1]
        auc = float(roc_auc_score(y_, proba))
        return {'auc': auc, 'proba': proba}

    logit_va, logit_te = _eval(logit, X_va, y_va), _eval(logit, X_te, y_te)
    svm_va, svm_te = _eval(svm, X_va, y_va), _eval(svm, X_te, y_te)
    if logit_va['auc'] >= svm_va['auc']:
        primary_name, primary = 'logistic', logit
        primary_va, primary_te = logit_va, logit_te
    else:
        primary_name, primary = 'linear_svm', svm
        primary_va, primary_te = svm_va, svm_te

    proba_va = primary_va['proba']
    _, tpr, thr_curve = roc_curve(y_va, proba_va)
    candidates = [(float(th), float(rec)) for th, rec in zip(thr_curve, tpr) if not np.isnan(th)]
    ok = [c for c in candidates if c[1] >= ATTACK_RECALL_TARGET]
    if ok:
        best_thr = max(ok, key=lambda c: c[0])[0]
    else:
        best_f1, best_thr = -1.0, 0.5
        for th in np.linspace(0.05, 0.95, 37):
            pred = (proba_va >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_va, pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(th)

    all_proba = primary.predict_proba(X)[:, 1]
    all_pred = (all_proba >= best_thr).astype(int)

    fpr_te, tpr_te, _ = roc_curve(y_te, primary_te['proba'])
    fpr_va, tpr_va, _ = roc_curve(y_va, proba_va)
    pred_te = (primary_te['proba'] >= best_thr).astype(int)
    cm_te = confusion_matrix(y_te, pred_te)

    return {
        'primary_name': primary_name,
        'threshold': float(best_thr),
        'val_auc': float(primary_va['auc']),
        'test_auc': float(primary_te['auc']),
        'all_proba': all_proba,
        'all_pred': all_pred,
        'fpr_te': fpr_te, 'tpr_te': tpr_te,
        'fpr_va': fpr_va, 'tpr_va': tpr_va,
        'cm_te': cm_te,
        'fire_clean': float(all_pred[:n_images].mean()),
        'fire_attacked': float(all_pred[n_images:].mean()),
        'test_fire_clean': float(pred_te[y_te == 0].mean()) if (y_te == 0).any() else 0.0,
        'test_fire_attacked': float(pred_te[y_te == 1].mean()) if (y_te == 1).any() else 0.0,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def cam_rgb(cam):
    a = align_cam(cam)
    colored = cm.jet(a)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def overlay_cam(pil_img, cam, alpha=0.45):
    base = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
    heat = cam_rgb(cam).astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def score_panel(score, fired, thr, size=224):
    """Small card: gate probability + fire/skip."""
    color = (228, 87, 86) if fired else (76, 120, 168)
    bg = (255, 255, 255)
    img = Image.new('RGB', (size, size), bg)
    draw = ImageDraw.Draw(img)
    margin = 14
    draw.rectangle([margin, margin, size - margin, size - margin],
                   outline=color, width=3, fill=(
                       int(0.85 * 255 + 0.15 * color[0]),
                       int(0.85 * 255 + 0.15 * color[1]),
                       int(0.85 * 255 + 0.15 * color[2]),
                   ))
    font_big = _get_font(_LAT_FONT, 36)
    font_sm = _get_font(_LAT_FONT, 14)
    font_xs = _get_font(_LAT_FONT, 12)
    decision = 'FIRE -> blur' if fired else 'SKIP (pass)'
    for text, font, y in [
        (f'{score:.3f}', font_big, size * 0.32),
        (decision, font_sm, size * 0.55),
        (f'thr={thr:.3f}', font_xs, size * 0.72),
    ]:
        bb = draw.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        draw.text(((size - tw) / 2, y), text, fill=color, font=font)
    return np.array(img)


def pick_examples(gate, n_images):
    """One clean (prefer skip) + one attacked (prefer fire), fixed seed."""
    rng = random.Random(EXAMPLE_SEED)
    clean_order = list(range(n_images))
    atk_order = list(range(n_images))
    rng.shuffle(clean_order)
    rng.shuffle(atk_order)

    clean_idx = None
    for i in clean_order:
        if not gate['all_pred'][i]:
            clean_idx = i
            break
    if clean_idx is None:
        clean_idx = clean_order[0]

    atk_idx = None
    for i in atk_order:
        if gate['all_pred'][n_images + i]:
            atk_idx = i
            break
    if atk_idx is None:
        atk_idx = atk_order[0]
    return clean_idx, atk_idx


def plot_process(L, out_dir, clean_img, atk_img, clean_cams, atk_cams,
                 clean_i, atk_i, gate, n_images):
    rows = []
    # (label, pil_or_rgb, cam_en, cam_l, row_offset_in_gate, is_attacked)
    specs = [
        ('clean', clean_img, clean_cams, clean_i, 0),
        ('attacked', atk_img, atk_cams, atk_i, n_images),
    ]
    col_titles = [
        'Image', 'Attn EN', 'Attn L', 'EN&L inter', 'Gate score', 'Gated result',
    ]
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.2))
    for r, (tag, img, cams, idx, offset) in enumerate(specs):
        cam_en = cams['cams_en'][idx]
        cam_l = cams['cams_l'][idx]
        inter = n_cam_intersection(cam_en, cam_l)
        score = float(gate['all_proba'][offset + idx])
        fired = bool(gate['all_pred'][offset + idx])
        if fired:
            mask = build_cc_bbox_blur_mask(cam_en, cam_l, threshold=DEFENSE_THR)
            result = apply_mask(img, mask, fill='blur')
        else:
            result = img

        panels = [
            np.array(img.convert('RGB')),
            overlay_cam(img, cam_en),
            overlay_cam(img, cam_l),
            overlay_cam(img, inter),
            score_panel(score, fired, gate['threshold']),
            np.array(result.convert('RGB')),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(tag, fontsize=11, rotation=0, labelpad=36, va='center')
        rows.append({
            'tag': tag, 'idx': int(idx), 'score': score, 'fired': fired,
        })

    fig.suptitle(
        f'Phase A/B process — EN&{LANG_LABEL[L]} / {ATTACK}  '
        f'(primary={gate["primary_name"]}, thr={gate["threshold"]:.3f})',
        fontsize=12)
    fig.tight_layout()
    path = Path(out_dir) / 'phase_ab_process.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', path)
    return rows


def plot_roc_cm(L, out_dir, gate):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(gate['fpr_va'], gate['tpr_va'], label=f'val AUC={gate["val_auc"]:.3f}', color='#4C78A8')
    ax.plot(gate['fpr_te'], gate['tpr_te'], label=f'test AUC={gate["test_auc"]:.3f}', color='#E45756')
    # Mark operating point on test curve approximately via closest TPR at thr
    ax.axhline(ATTACK_RECALL_TARGET, color='#999', ls='--', lw=0.8,
               label=f'recall target {ATTACK_RECALL_TARGET}')
    ax.plot([0, 1], [0, 1], color='#ccc', lw=0.8)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC EN&{LANG_LABEL[L]} ({gate["primary_name"]})')
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    cm_ = gate['cm_te']
    im = ax.imshow(cm_, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['pred clean', 'pred attack'])
    ax.set_yticklabels(['true clean', 'true attack'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm_[i, j])), ha='center', va='center',
                    color='white' if cm_[i, j] > cm_.max() / 2 else 'black', fontsize=12)
    ax.set_title(f'Test confusion @ thr={gate["threshold"]:.3f}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Phase B detector — EN&{LANG_LABEL[L]} / {ATTACK}  '
        f'(test fire clean={100*gate["test_fire_clean"]:.1f}% / '
        f'atk={100*gate["test_fire_attacked"]:.1f}%)',
        fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / 'phase_b_roc_cm.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', path)


def load_existing_captions(out_dir):
    phase_a = {}
    metrics = {}
    pa = Path(out_dir) / 'phase_a_summary.json'
    dm = Path(out_dir) / 'detector_metrics.json'
    if pa.exists():
        with open(pa, encoding='utf-8') as f:
            phase_a = json.load(f)
    if dm.exists():
        with open(dm, encoding='utf-8') as f:
            metrics = json.load(f)
    return phase_a, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Phase A/B viz (cache + sklearn; no CLIP bake)')
    sample_path = HERE / '..' / 'image_samples' / 'CIFAR10_BALANCED_1000_SAMPLE.json'
    with open(sample_path, encoding='utf-8') as f:
        saved = json.load(f)
    idx = saved['idx']
    attack_pos = saved['attack_pos']

    hf = load_dataset('uoft-cs/cifar10', split='test')
    label_key = 'label' if 'label' in hf.column_names else 'labels'
    image_key = 'img' if 'img' in hf.column_names else 'image'
    rows = hf.select(idx)
    true = np.array(rows[label_key])
    assert len(idx) == 1000 and np.array_equal(true, np.array(saved['true']))

    rng = random.Random(0)
    target = np.array([
        rng.choice([c for c in range(10) if c != int(true[k])])
        for k in range(len(idx))
    ])
    clean_224 = [
        im.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
        for im in rows[image_key]
    ]
    n_images = len(clean_224)
    print(f'Loaded {n_images} images')

    all_summaries = {}
    for L in PARTNER_LANGS:
        out_dir = Path('results') / L / ATTACK
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n===== EN&{L.upper()} =====')
        packed = load_cache(L, out_dir)
        clean_cams, atk_cams = packed['clean'], packed['atk']
        X, y, img_ids, feature_names = build_feature_matrix(clean_cams, atk_cams, n_images)
        print(f'Features: {X.shape}')
        gate = train_gate(X, y, img_ids, n_images)
        print(f'  primary={gate["primary_name"]} thr={gate["threshold"]:.4f} '
              f'test_auc={gate["test_auc"]:.3f} '
              f'fire clean/atk={100*gate["fire_clean"]:.1f}%/{100*gate["fire_attacked"]:.1f}%')

        clean_i, atk_i = pick_examples(gate, n_images)
        atk_img = draw_dual_box(
            clean_224[atk_i],
            CLASSES['en'][int(target[atk_i])], 'en',
            CLASSES[L][int(target[atk_i])], L,
            atk_i, attack_pos,
        )
        clean_img = clean_224[clean_i]
        print(f'  examples: clean_i={clean_i} atk_i={atk_i} '
              f'tgt={CLASSES["en"][int(target[atk_i])]}')

        examples = plot_process(
            L, out_dir, clean_img, atk_img, clean_cams, atk_cams,
            clean_i, atk_i, gate, n_images,
        )
        plot_roc_cm(L, out_dir, gate)

        phase_a, metrics = load_existing_captions(out_dir)
        summary = {
            'L': L,
            'attack': ATTACK,
            'primary': gate['primary_name'],
            'threshold': gate['threshold'],
            'val_auc': gate['val_auc'],
            'test_auc': gate['test_auc'],
            'fire_rate_clean_full': gate['fire_clean'],
            'fire_rate_attacked_full': gate['fire_attacked'],
            'test_fire_clean': gate['test_fire_clean'],
            'test_fire_attacked': gate['test_fire_attacked'],
            'examples': examples,
            'phase_a_existing': phase_a,
            'detector_metrics_existing': {
                'primary': metrics.get('primary'),
                'threshold': metrics.get('threshold'),
                'test_at_threshold': metrics.get('test_at_threshold'),
                'phase_a_pca_nn_cent_acc': metrics.get('phase_a_pca_nn_cent_acc'),
            },
            'outputs': {
                'phase_ab_process': str(out_dir / 'phase_ab_process.png'),
                'phase_b_roc_cm': str(out_dir / 'phase_b_roc_cm.png'),
                'pca_features': str(out_dir / 'pca_features.png'),
                'tsne_features': str(out_dir / 'tsne_features.png'),
                'feature_importance': str(out_dir / 'feature_importance.png'),
            },
            'n_features': len(feature_names),
        }
        sum_path = out_dir / 'phase_ab_viz_summary.json'
        with open(sum_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print('Saved', sum_path)
        all_summaries[L] = summary

    roll = Path('results') / 'phase_ab_viz_rollups.json'
    with open(roll, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)
    print('\nSaved', roll)
    print('Done.')


if __name__ == '__main__':
    main()
