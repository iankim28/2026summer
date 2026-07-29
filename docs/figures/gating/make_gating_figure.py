"""Make gating decision-boundary figure for the paper (Figure 5).

Outputs (this directory):
  gating_figure.png — PCA feature space (left) + clean/attacked process strips
    (right), with arrows from cluster examples to gate FIRE / SKIP outcomes.
    Production fill = black. Partner L = ZH (same recipe for KO / JA).

Uses frozen ZH Attn-last cache + sklearn (CPU only; no CLIP bake).
Layout inspired by the JHSS process+PCA composite style.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CACHE_DIR = REPO / "lib" / "notebooks" / "attack_detector" / "results" / "zh" / "multi"
CACHE_PATH = CACHE_DIR / "cache" / "attn_en_zh_clean_multi.npz"
METRICS_PATH = CACHE_DIR / "detector_metrics.json"
SAMPLE_PATH = REPO / "lib" / "notebooks" / "image_samples" / "CIFAR10_BALANCED_1000_SAMPLE.json"

DISPLAY_SIZE = 224
FONT_SIZE = 24
PAD = 8
DEFENSE_THR = 0.95
DILATE = 3
TOP_K = 2
SPLIT_SEED = 0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ATTACK_RECALL_TARGET = 0.99
EXAMPLE_SEED = 0
L = "zh"

CLASSES = {
    "en": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "zh": ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
}

C_CLEAN = "#4C78A8"
C_ATK = "#E45756"
C_SKIP = "#2E7D32"
C_FIRE = "#B00020"
C_TAU = "#333333"
C_NOTE = "#555555"


# ---------------------------------------------------------------------------
# Fonts / attack render
# ---------------------------------------------------------------------------

def _font_paths():
    if platform.system() == "Windows":
        wf = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        return os.path.join(wf, "msyh.ttc"), os.path.join(wf, "arial.ttf")
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    lat = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.isfile(cjk):
        cjk = "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"
    return cjk, lat


_CJK_FONT, _LAT_FONT = _font_paths()
_FONT_CACHE = {}


def _get_font(fp, size=FONT_SIZE):
    key = (fp or "__default__", size)
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
    xy0 = attack_pos["en"][int(img_idx)]
    xy1 = attack_pos["l"][int(img_idx)]
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_LAT_FONT if lang == "en" else _CJK_FONT)
        bb = draw.textbbox((0, 0), word, font=font)
        bw = (bb[2] - bb[0]) + 2 * PAD
        bh = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill="white")
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill="black", font=font)
    return img


# ---------------------------------------------------------------------------
# Mask / features (mirrors attack_detector)
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
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:]
            | pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:]
            | pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:]
        )
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


def apply_black(pil_img, mask):
    arr = np.array(pil_img.convert("RGB"))
    m = mask.astype(bool)
    if mask.shape != arr.shape[:2]:
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize(
                arr.shape[1::-1], Image.NEAREST)
        ) > 127
    out = arr.copy()
    out[m] = 0
    return Image.fromarray(out.astype(np.uint8))


def build_cc_bbox_mask(cam_en, cam_l, threshold=DEFENSE_THR, dilate=DILATE, top_k=TOP_K):
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
        f"{prefix}_entropy": _entropy(a),
        f"{prefix}_topk05": _topk_mass(a, 0.05),
        f"{prefix}_topk10": _topk_mass(a, 0.10),
        f"{prefix}_max_over_mean": mx / mn,
        f"{prefix}_gini": _gini(a),
        f"{prefix}_kurtosis": _spatial_kurtosis(a),
        f"{prefix}_cov95": cov95,
        f"{prefix}_ncc95": float(ncc),
    }


def extract_pair_features(cam_en, cam_l):
    feats = {}
    feats.update(_map_features(cam_en, "en"))
    feats.update(_map_features(cam_l, "l"))
    inter = n_cam_intersection(cam_en, cam_l)
    feats.update(_map_features(inter, "inter"))
    ae, al = align_cam(cam_en), align_cam(cam_l)
    feats["en_l_corr"] = (
        float(np.corrcoef(ae.ravel(), al.ravel())[0, 1])
        if ae.std() > 0 and al.std() > 0 else 0.0
    )
    hot_e = ae >= np.percentile(ae, 95)
    hot_l = al >= np.percentile(al, 95)
    union = (hot_e | hot_l).sum()
    feats["en_l_iou95"] = float((hot_e & hot_l).sum() / union) if union > 0 else 0.0
    return feats


def build_feature_matrix(clean_cams, atk_cams, n_images):
    rows_feat, y_labels, img_ids = [], [], []
    for i in range(n_images):
        rows_feat.append(extract_pair_features(clean_cams["cams_en"][i], clean_cams["cams_l"][i]))
        y_labels.append(0)
        img_ids.append(i)
    for i in range(n_images):
        rows_feat.append(extract_pair_features(atk_cams["cams_en"][i], atk_cams["cams_l"][i]))
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


def load_cache():
    if not CACHE_PATH.exists():
        print(f"ERROR: Attn-last cache missing: {CACHE_PATH}", file=sys.stderr)
        print(
            "Baking would require CUDA + full notebook run. Stop and ask before baking.",
            file=sys.stderr,
        )
        sys.exit(1)
    z = np.load(CACHE_PATH, allow_pickle=False)

    def _get(new, old):
        return z[new] if new in z.files else z[old]

    return {
        "clean": {
            "cams_en": z["clean_cams_en"],
            "cams_l": _get("clean_cams_l", "clean_cams_zh"),
        },
        "atk": {
            "cams_en": z["atk_cams_en"],
            "cams_l": _get("atk_cams_l", "atk_cams_zh"),
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
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SPLIT_SEED)),
    ])
    logit.fit(X_tr, y_tr)

    _svm_scaler = StandardScaler().fit(X_tr)
    svm_inner = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=SPLIT_SEED), cv=3)
    svm_inner.fit(_svm_scaler.transform(X_tr), y_tr)
    svm = _ScaledModel(_svm_scaler, svm_inner)

    def _eval(model, X_, y_):
        proba = model.predict_proba(X_)[:, 1]
        return {"auc": float(roc_auc_score(y_, proba)), "proba": proba}

    logit_va, logit_te = _eval(logit, X_va, y_va), _eval(logit, X_te, y_te)
    svm_va, svm_te = _eval(svm, X_va, y_va), _eval(svm, X_te, y_te)
    if logit_va["auc"] >= svm_va["auc"]:
        primary_name, primary = "logistic", logit
        primary_va, primary_te = logit_va, logit_te
    else:
        primary_name, primary = "linear_svm", svm
        primary_va, primary_te = svm_va, svm_te

    proba_va = primary_va["proba"]
    fpr_va, tpr_va, thr_curve = roc_curve(y_va, proba_va)
    candidates = [(float(th), float(rec)) for th, rec in zip(thr_curve, tpr_va) if not np.isnan(th)]
    ok = [c for c in candidates if c[1] >= ATTACK_RECALL_TARGET]
    if ok:
        best_thr = max(ok, key=lambda c: c[0])[0]
    else:
        best_f1, best_thr = -1.0, 0.5
        for th in np.linspace(0.05, 0.95, 37):
            pred = (proba_va >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_va, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(th)

    # Operating-point FPR on val at chosen threshold
    op_fpr = float(((proba_va >= best_thr) & (y_va == 0)).sum() / max(1, (y_va == 0).sum()))
    op_tpr = float(((proba_va >= best_thr) & (y_va == 1)).sum() / max(1, (y_va == 1).sum()))

    all_proba = primary.predict_proba(X)[:, 1]
    all_pred = (all_proba >= best_thr).astype(int)
    fpr_te, tpr_te, _ = roc_curve(y_te, primary_te["proba"])

    return {
        "primary_name": primary_name,
        "primary": primary,
        "threshold": float(best_thr),
        "val_auc": float(primary_va["auc"]),
        "test_auc": float(primary_te["auc"]),
        "all_proba": all_proba,
        "all_pred": all_pred,
        "train_m": train_m,
        "fpr_te": fpr_te,
        "tpr_te": tpr_te,
        "fpr_va": fpr_va,
        "tpr_va": tpr_va,
        "op_fpr": op_fpr,
        "op_tpr": op_tpr,
        "fire_clean": float(all_pred[:n_images].mean()),
        "fire_attacked": float(all_pred[n_images:].mean()),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def overlay_cam(pil_img, cam, alpha=0.45):
    base = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    heat = cm.jet(align_cam(cam))[:, :, :3]
    out = (1 - alpha) * base + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def pick_examples(gate, n_images, true, target):
    """Prefer dog<-ship (matches other paper figures) when gate agrees."""
    prefer = None
    for i in range(n_images):
        if CLASSES["en"][int(true[i])] == "dog" and CLASSES["en"][int(target[i])] == "ship":
            prefer = i
            break

    if prefer is not None and (not gate["all_pred"][prefer]) and gate["all_pred"][n_images + prefer]:
        return prefer, prefer

    # Fallback: prefer train-split clean-skip / atk-fire
    train_imgs = {int(i) for i in range(n_images) if gate["train_m"][i]}
    rng = random.Random(EXAMPLE_SEED)
    clean_order = [i for i in range(n_images) if i in train_imgs] or list(range(n_images))
    atk_order = list(clean_order)
    rng.shuffle(clean_order)
    rng.shuffle(atk_order)

    clean_idx = next((i for i in clean_order if not gate["all_pred"][i]), clean_order[0])
    atk_idx = next(
        (i for i in atk_order if gate["all_pred"][n_images + i]), atk_order[0])
    return clean_idx, atk_idx


def _imshow_off(ax, arr):
    ax.imshow(arr)
    ax.axis("off")


def score_panel(score, fired, thr, size=224):
    """Gate probability card: FIRE -> black / SKIP (pass)."""
    color = (228, 87, 86) if fired else (76, 120, 168)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 14
    fill = (
        int(0.85 * 255 + 0.15 * color[0]),
        int(0.85 * 255 + 0.15 * color[1]),
        int(0.85 * 255 + 0.15 * color[2]),
    )
    draw.rectangle(
        [margin, margin, size - margin, size - margin],
        outline=color, width=3, fill=fill,
    )
    font_big = _get_font(_LAT_FONT, 36)
    font_sm = _get_font(_LAT_FONT, 14)
    font_xs = _get_font(_LAT_FONT, 12)
    decision = "FIRE -> black" if fired else "SKIP (pass)"
    for text, font, y in [
        (f"{score:.3f}", font_big, size * 0.30),
        (decision, font_sm, size * 0.54),
        (f"thr={thr:.3f}", font_xs, size * 0.72),
    ]:
        bb = draw.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        draw.text(((size - tw) / 2, y), text, fill=color, font=font)
    return np.array(img)


def _row_panels(img, cam_en, cam_l, score, fired, thr):
    inter = n_cam_intersection(cam_en, cam_l)
    if fired:
        result = apply_black(img, build_cc_bbox_mask(cam_en, cam_l))
    else:
        result = img
    return [
        np.array(img.convert("RGB")),
        overlay_cam(img, cam_en),
        overlay_cam(img, cam_l),
        overlay_cam(img, inter),
        score_panel(score, fired, thr),
        np.array(result.convert("RGB")),
    ]


def make_figure(
    clean_224, atk_224, clean_cams, atk_cams, gate, X, y, n_images,
    clean_i, atk_i, out_path,
):
    """PCA (left) + clean/attacked process strips (right), JHSS-style."""
    thr = gate["threshold"]
    train_m = gate["train_m"]
    model = gate["primary"]

    # PCA fitted on the training split only (same images used to train the gate)
    scaler_pca = StandardScaler().fit(X[train_m])
    pca = PCA(n_components=2, random_state=SPLIT_SEED)
    Z_train = pca.fit_transform(scaler_pca.transform(X[train_m]))
    y_train = y[train_m]
    Z_all = pca.transform(scaler_pca.transform(X))
    z_clean = Z_all[clean_i]
    z_atk = Z_all[n_images + atk_i]

    score_c = float(gate["all_proba"][clean_i])
    score_a = float(gate["all_proba"][n_images + atk_i])
    fired_c = bool(gate["all_pred"][clean_i])
    fired_a = bool(gate["all_pred"][n_images + atk_i])

    clean_panels = _row_panels(
        clean_224[clean_i],
        clean_cams["cams_en"][clean_i],
        clean_cams["cams_l"][clean_i],
        score_c, fired_c, thr,
    )
    atk_panels = _row_panels(
        atk_224[atk_i],
        atk_cams["cams_en"][atk_i],
        atk_cams["cams_l"][atk_i],
        score_a, fired_a, thr,
    )

    fig = plt.figure(figsize=(16.5, 6.8), facecolor="white")
    # Left PCA | right process (titles top/bottom outside the bordered image rows)
    outer = GridSpec(
        1, 2, figure=fig,
        left=0.04, right=0.99, bottom=0.07, top=0.90,
        wspace=0.10, width_ratios=[1.05, 2.35],
    )

    # ---- Left: PCA (train) + decision boundary at τ ----
    ax_pca = fig.add_subplot(outer[0, 0])
    ax_pca.scatter(
        Z_train[y_train == 0, 0], Z_train[y_train == 0, 1], s=12, alpha=0.40,
        c=C_CLEAN, label="clean (train)", rasterized=True, edgecolors="none",
    )
    ax_pca.scatter(
        Z_train[y_train == 1, 0], Z_train[y_train == 1, 1], s=12, alpha=0.40,
        c=C_ATK, label="attacked (train)", rasterized=True, edgecolors="none",
    )

    # Contour of P(attack)=τ in the PCA plane via inverse map → gate model
    x_min, x_max = Z_train[:, 0].min(), Z_train[:, 0].max()
    y_min, y_max = Z_train[:, 1].min(), Z_train[:, 1].max()
    pad_x = 0.08 * (x_max - x_min + 1e-6)
    pad_y = 0.08 * (y_max - y_min + 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 200),
        np.linspace(y_min - pad_y, y_max + pad_y, 200),
    )
    grid_z = np.column_stack([xx.ravel(), yy.ravel()])
    X_grid = scaler_pca.inverse_transform(pca.inverse_transform(grid_z))
    proba_grid = model.predict_proba(X_grid)[:, 1].reshape(xx.shape)
    ax_pca.contour(
        xx, yy, proba_grid, levels=[thr], colors=[C_TAU],
        linewidths=2.2, zorder=4,
    )
    ax_pca.contourf(
        xx, yy, proba_grid, levels=[0.0, thr, 1.0],
        colors=[C_CLEAN, C_ATK], alpha=0.07, zorder=0,
    )
    ax_pca.plot([], [], color=C_TAU, lw=2.2, label=f"decision boundary (τ={thr:.3f})")

    ax_pca.scatter(
        [z_clean[0]], [z_clean[1]], s=90, c=C_CLEAN, edgecolors="black",
        linewidths=1.0, zorder=5, marker="o",
    )
    ax_pca.scatter(
        [z_atk[0]], [z_atk[1]], s=90, c=C_ATK, edgecolors="black",
        linewidths=1.0, zorder=5, marker="o",
    )
    ax_pca.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_pca.set_ylim(y_min - pad_y, y_max + pad_y)
    ax_pca.set_xlabel("PC1", fontsize=10)
    ax_pca.set_ylabel("PC2", fontsize=10)
    ax_pca.set_title(
        f"(a) Feature space — EN&ZH (train)",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax_pca.legend(frameon=False, fontsize=11, loc="lower right")
    ax_pca.tick_params(labelsize=8)
    ax_pca.text(
        0.02, 0.98,
        f"{gate['primary_name']}, τ={thr:.3f}\n"
        "PCA fit on train split (70%)\n"
        "boundary = P(attack)=τ in 26-D\n"
        "(projected via PCA inverse)",
        transform=ax_pca.transAxes, fontsize=7.5, color=C_NOTE,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#CCC", lw=0.6, alpha=0.95),
        zorder=7,
    )

    # ---- Right: title row / clean / attacked / title row ----
    right = outer[0, 1].subgridspec(
        4, 6, wspace=0.06, hspace=0.04,
        height_ratios=[0.10, 1.0, 1.0, 0.10],
    )
    col_titles = [
        "Image", "Attn EN", "Attn L", "EN∩L inter", "Gate score", "Gated result",
    ]
    row_specs = [
        (1, "clean", C_CLEAN, clean_panels),
        (2, "attacked", C_ATK, atk_panels),
    ]

    # Column titles at top and bottom — larger font, snug against image rows
    for title_row in (0, 3):
        # top titles sit on the bottom edge of their cell; bottom titles on the top edge
        y, va = (0.0, "bottom") if title_row == 0 else (1.0, "top")
        for c, name in enumerate(col_titles):
            ax_t = fig.add_subplot(right[title_row, c])
            ax_t.set_axis_off()
            ax_t.text(
                0.5, y, name, transform=ax_t.transAxes,
                ha="center", va=va, fontsize=12, color="#333",
            )

    first_axes = []
    all_row_axes = []
    for r, tag, color, panels in row_specs:
        row_axes = []
        for c, panel in enumerate(panels):
            ax = fig.add_subplot(right[r, c])
            _imshow_off(ax, panel)
            row_axes.append(ax)
            if c == 0:
                ax.set_ylabel(
                    tag, fontsize=11, rotation=0, labelpad=40, va="center",
                    color=color, fontweight="bold",
                )
        first_axes.append(row_axes[0])
        all_row_axes.append((row_axes, color))

    # Outer row border hugging the image axes only (no title overlap)
    fig.canvas.draw()
    for row_axes, color in all_row_axes:
        bb0 = row_axes[0].get_position()
        bb1 = row_axes[-1].get_position()
        pad = 0.0015  # slight inset/outset around the actual image boxes
        frame = FancyBboxPatch(
            (bb0.x0 - pad, bb0.y0 - pad),
            (bb1.x1 - bb0.x0) + 2 * pad,
            (bb0.y1 - bb0.y0) + 2 * pad,
            boxstyle="square,pad=0",
            fill=False, edgecolor=color, linewidth=2.2,
            transform=fig.transFigure, clip_on=False, zorder=4,
        )
        fig.add_artist(frame)

    fig.text(
        0.70, 0.965,
        "(b) Gate decision — score ≥ τ → black occlude, else pass-through",
        fontsize=12, fontweight="bold", ha="center", va="top",
    )

    # Arrows from PCA example points to process rows (figure coords)
    fig.canvas.draw()
    for z_pt, ax_dst, color in [
        (z_clean, first_axes[0], C_CLEAN),
        (z_atk, first_axes[1], C_ATK),
    ]:
        disp = ax_pca.transData.transform(z_pt)
        inv = fig.transFigure.inverted()
        x0, y0 = inv.transform(disp)
        bb = ax_dst.get_position()
        x1, y1 = bb.x0 - 0.008, (bb.y0 + bb.y1) / 2
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=2.0, color=color,
            connectionstyle="arc3,rad=0.05",
            clip_on=False, zorder=6,
        )
        fig.add_artist(arr)

    fig.text(
        0.5, 0.012,
        f"EN&ZH multi  ·  primary={gate['primary_name']}  ·  "
        f"τ={thr:.3f} (val recall ≥ {ATTACK_RECALL_TARGET:g})  ·  "
        f"fire clean={100 * gate['fire_clean']:.1f}% / "
        f"atk={100 * gate['fire_attacked']:.1f}%  ·  same recipe for KO / JA",
        fontsize=8, color=C_NOTE, ha="center", va="bottom",
    )

    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Gating decision-boundary figure (ZH cache + sklearn; no CLIP bake)")
    if not CACHE_PATH.exists():
        print(f"ERROR: missing cache {CACHE_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    idx = saved["idx"]
    attack_pos = saved["attack_pos"]

    hf = load_dataset("uoft-cs/cifar10", split="test")
    label_key = "label" if "label" in hf.column_names else "labels"
    image_key = "img" if "img" in hf.column_names else "image"
    rows = hf.select(idx)
    true = np.array(rows[label_key])
    assert len(idx) == 1000 and np.array_equal(true, np.array(saved["true"]))

    rng = random.Random(0)
    target = np.array([
        rng.choice([c for c in range(10) if c != int(true[k])])
        for k in range(len(idx))
    ])
    clean_224 = [
        im.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
        for im in rows[image_key]
    ]
    n_images = len(clean_224)
    atk_224 = [
        draw_dual_box(
            clean_224[i],
            CLASSES["en"][int(target[i])], "en",
            CLASSES["zh"][int(target[i])], "zh",
            i, attack_pos,
        )
        for i in range(n_images)
    ]
    print(f"Loaded {n_images} images")

    packed = load_cache()
    clean_cams, atk_cams = packed["clean"], packed["atk"]
    print("Building 26-D features...")
    X, y, img_ids, feature_names = build_feature_matrix(clean_cams, atk_cams, n_images)
    print(f"  X={X.shape}  features={len(feature_names)}")

    gate = train_gate(X, y, img_ids, n_images)
    print(
        f"  primary={gate['primary_name']} thr={gate['threshold']:.4f} "
        f"test_auc={gate['test_auc']:.3f} "
        f"fire clean/atk={100 * gate['fire_clean']:.1f}%/{100 * gate['fire_attacked']:.1f}%"
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            frozen = json.load(f)
        frozen_thr = float(frozen.get("threshold", float("nan")))
        print(f"  frozen detector_metrics thr={frozen_thr:.4f} "
              f"(delta={abs(gate['threshold'] - frozen_thr):.4f})")

    clean_i, atk_i = pick_examples(gate, n_images, true, target)
    print(
        f"  examples: clean_i={clean_i} score={gate['all_proba'][clean_i]:.3f} "
        f"atk_i={atk_i} score={gate['all_proba'][n_images + atk_i]:.3f}"
    )

    out_path = HERE / "gating_figure.png"
    make_figure(
        clean_224, atk_224, clean_cams, atk_cams, gate, X, y, n_images,
        clean_i, atk_i, out_path,
    )

    summary = {
        "L": "zh",
        "primary": gate["primary_name"],
        "threshold": gate["threshold"],
        "val_auc": gate["val_auc"],
        "test_auc": gate["test_auc"],
        "fire_clean": gate["fire_clean"],
        "fire_attacked": gate["fire_attacked"],
        "clean_i": int(clean_i),
        "atk_i": int(atk_i),
        "n_features": len(feature_names),
        "output": str(out_path),
    }
    summary_path = HERE / "gating_figure_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved", summary_path)


if __name__ == "__main__":
    main()
