"""Make attack-detector pipeline figure (PCA → SVM → decision).

Outputs (this directory):
  detector_pipeline.png — train-set stack → PCA feature view → SVM
    decision boundary → test-image decision. Matches the hand sketch
    layout; uses the same ZH Attn-last cache + calibrated linear SVM
    as make_gating_figure.py (τ ≈ 0.473).

CPU / sklearn only — no CLIP bake.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reuse gate training / feature / attack-render helpers
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_gating_figure as gf  # noqa: E402

OUT_PATH = HERE / "detector_pipeline.png"
SUMMARY_PATH = HERE / "detector_pipeline_summary.json"

C_CLEAN = gf.C_CLEAN
C_ATK = gf.C_ATK
C_TAU = gf.C_TAU
C_NOTE = gf.C_NOTE
C_ARROW = "#444444"
C_LABEL = "#555555"

N_STACK = 4
STACK_OFFSET = 14
STACK_SIZE = 100


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def make_image_stack(images, size=STACK_SIZE, offset=STACK_OFFSET, border_color=(80, 80, 80)):
    """Overlapping diagonal stack of RGB images (train-set icon)."""
    imgs = [
        im.convert("RGB").resize((size, size), Image.BICUBIC) for im in images
    ]
    n = len(imgs)
    canvas_w = size + (n - 1) * offset + 4
    canvas_h = size + (n - 1) * offset + 4
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    bc = (*border_color, 230)
    for i, im in enumerate(imgs):
        x = i * offset
        y = i * offset
        framed = Image.new("RGBA", (size + 2, size + 2), (255, 255, 255, 255))
        framed.paste(im, (1, 1))
        border = Image.new("RGBA", (size + 2, size + 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(border)
        draw.rectangle([0, 0, size + 1, size + 1], outline=bc, width=2)
        canvas.paste(framed, (x, y), framed)
        canvas.alpha_composite(border, (x, y))
    return canvas.convert("RGB")


def make_dual_stream_panel(clean_imgs, atk_imgs, size=STACK_SIZE, offset=STACK_OFFSET):
    """Two labeled stacks — clean (top) and attacked (bottom)."""
    stack_c = make_image_stack(clean_imgs, size=size, offset=offset,
                               border_color=_hex_rgb(C_CLEAN))
    stack_a = make_image_stack(atk_imgs, size=size, offset=offset,
                               border_color=_hex_rgb(C_ATK))

    gap = 18
    label_h = 22
    pad = 6
    col_w = max(stack_c.size[0], stack_a.size[0])
    canvas_w = col_w + 2 * pad
    canvas_h = (
        label_h + stack_c.size[1] + gap + label_h + stack_a.size[1] + 2 * pad
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"),
            16,
        ) if os.name == "nt" else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    y = pad
    for label, color, stack in [
        ("clean", _hex_rgb(C_CLEAN), stack_c),
        ("attacked", _hex_rgb(C_ATK), stack_a),
    ]:
        bb = draw.textbbox((0, 0), label, font=font)
        tw = bb[2] - bb[0]
        draw.text(((canvas_w - tw) / 2, y), label, fill=color, font=font)
        y += label_h
        x = (canvas_w - stack.size[0]) // 2
        canvas.paste(stack, (x, y))
        y += stack.size[1] + gap

    return canvas


def _class_axis(Z, y, cls, length_scale=0.55):
    """Short line through class cloud along its first 2-D principal direction."""
    pts = Z[y == cls]
    if len(pts) < 3:
        return None
    mu = pts.mean(axis=0)
    centered = pts - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    # Span a fraction of the class spread
    proj = centered @ direction
    half = length_scale * (proj.max() - proj.min()) / 2
    return mu - half * direction, mu + half * direction


def _pca_limits(Z, pad_frac=0.10):
    x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
    y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
    pad_x = pad_frac * (x_max - x_min + 1e-6)
    pad_y = pad_frac * (y_max - y_min + 1e-6)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _draw_scatter(ax, Z, y, s=11, alpha=0.42):
    ax.scatter(
        Z[y == 0, 0], Z[y == 0, 1], s=s, alpha=alpha,
        c=C_CLEAN, label="clean", rasterized=True, edgecolors="none", zorder=2,
    )
    ax.scatter(
        Z[y == 1, 0], Z[y == 1, 1], s=s, alpha=alpha,
        c=C_ATK, label="attacked", rasterized=True, edgecolors="none", zorder=2,
    )


def _style_axes(ax, xlim, ylim, xlabel=True, ylabel=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel("PC1", fontsize=9, labelpad=3)
    else:
        ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel("PC2", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=7.5, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")
        spine.set_linewidth(0.7)


def _arrow_between(fig, xy_a, xy_b, label):
    """Full horizontal arrow; label floats above the shaft (never overlaps axes)."""
    x0, y0 = xy_a
    x1, y1 = xy_b
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)

    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.55, color=C_ARROW,
        clip_on=False, zorder=8,
    ))
    # Label sits just above the shaft (still under panel titles)
    fig.text(
        mx, my + 0.018, label, ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#222222", zorder=10,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="#E5E5E5", lw=0.5, alpha=1.0),
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(clean_224, atk_224, gate, X, y, n_images, test_i, out_path):
    thr = gate["threshold"]
    train_m = gate["train_m"]
    model = gate["primary"]

    scaler_pca = StandardScaler().fit(X[train_m])
    pca = PCA(n_components=2, random_state=gf.SPLIT_SEED)
    Z_train = pca.fit_transform(scaler_pca.transform(X[train_m]))
    y_train = y[train_m]
    Z_all = pca.transform(scaler_pca.transform(X))
    z_test = Z_all[test_i]
    test_proba = float(gate["all_proba"][test_i])
    test_pred = bool(gate["all_pred"][test_i])
    decision = "attacked" if test_pred else "clean"
    decision_color = C_ATK if test_pred else C_CLEAN

    # Two train streams: clean stack + attacked stack (same image ids)
    train_img_ids = sorted({int(i) for i in range(n_images) if train_m[i]})
    rng = random.Random(0)
    pick = rng.sample(train_img_ids, k=min(N_STACK, len(train_img_ids)))
    clean_stack_imgs = [clean_224[i] for i in pick]
    atk_stack_imgs = [atk_224[i] for i in pick]
    stack = make_dual_stream_panel(clean_stack_imgs, atk_stack_imgs)

    x0, x1, y0, y1 = _pca_limits(Z_train)
    xlim, ylim = (x0, x1), (y0, y1)

    # Decision surface on PCA plane (same recipe as gating figure)
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 220),
        np.linspace(ylim[0], ylim[1], 220),
    )
    grid_z = np.column_stack([xx.ravel(), yy.ravel()])
    X_grid = scaler_pca.inverse_transform(pca.inverse_transform(grid_z))
    proba_grid = model.predict_proba(X_grid)[:, 1].reshape(xx.shape)

    # ---- Layout ----
    # Wider gaps between columns so PCA/SVM arrow labels sit in clear space
    fig = plt.figure(figsize=(15.4, 7.6), facecolor="white")
    gs = GridSpec(
        2, 3, figure=fig,
        left=0.045, right=0.98, bottom=0.07, top=0.90,
        wspace=0.48, hspace=0.28,
        height_ratios=[1.45, 1.15],
        width_ratios=[0.85, 1.20, 1.25],
    )

    # (1) Train set — two streams (clean / attacked)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_train.imshow(np.array(stack))
    ax_train.set_axis_off()
    ax_train.set_title("Train set", fontsize=12, color=C_LABEL, pad=8)
    ax_train.set_aspect("equal")
    arr = np.array(stack)
    h, w = arr.shape[:2]
    pad_frac = 0.04
    ax_train.set_xlim(-pad_frac * w, w * (1 + pad_frac))
    ax_train.set_ylim(h * (1 + pad_frac), -pad_frac * h)

    # (2) PCA view (no boundary) — hide y-label; shared with SVM via title
    ax_pca = fig.add_subplot(gs[0, 1])
    _draw_scatter(ax_pca, Z_train, y_train)
    for cls, color in [(0, C_CLEAN), (1, C_ATK)]:
        ends = _class_axis(Z_train, y_train, cls)
        if ends is not None:
            (a, b) = ends
            ax_pca.plot(
                [a[0], b[0]], [a[1], b[1]],
                color=color, lw=1.1, alpha=0.45, zorder=1,
            )
    _style_axes(ax_pca, xlim, ylim, ylabel=True)
    ax_pca.set_title("Feature space (PCA view)", fontsize=11, color="#333333", pad=8)
    leg_pca = ax_pca.legend(
        frameon=True, fontsize=9, loc="upper right",
        handletextpad=0.3, markerscale=1.3, framealpha=0.92,
        edgecolor="#E0E0E0", fancybox=True, borderpad=0.35,
    )
    leg_pca.get_frame().set_linewidth(0.6)

    # (3) SVM decision view
    ax_svm = fig.add_subplot(gs[0, 2])
    ax_svm.contourf(
        xx, yy, proba_grid, levels=[0.0, thr, 1.0],
        colors=[C_CLEAN, C_ATK], alpha=0.11, zorder=0,
    )
    ax_svm.contour(
        xx, yy, proba_grid, levels=[thr], colors=[C_TAU],
        linewidths=2.0, zorder=3,
    )
    _draw_scatter(ax_svm, Z_train, y_train)
    # Region labels in open corners (away from dense clouds + legend)
    region_bbox = dict(
        boxstyle="round,pad=0.20", facecolor="white",
        edgecolor="none", alpha=0.78,
    )
    ax_svm.text(
        0.06, 0.92, "clean", transform=ax_svm.transAxes,
        color=C_CLEAN, fontsize=14, fontweight="bold",
        ha="left", va="top", alpha=0.92, zorder=4, bbox=region_bbox,
    )
    ax_svm.text(
        0.94, 0.08, "attacked", transform=ax_svm.transAxes,
        color=C_ATK, fontsize=14, fontweight="bold",
        ha="right", va="bottom", alpha=0.92, zorder=4, bbox=region_bbox,
    )
    # Highlight test point
    ax_svm.scatter(
        [z_test[0]], [z_test[1]], s=120, c=decision_color,
        edgecolors="black", linewidths=1.3, zorder=6, marker="o",
    )
    _style_axes(ax_svm, xlim, ylim, ylabel=False)
    ax_svm.set_title(
        f"Decision boundary  ·  {gate['primary_name']}, τ={thr:.3f}",
        fontsize=11, color="#333333", pad=8,
    )
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CLEAN,
               markersize=6.5, label="clean (train)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ATK,
               markersize=6.5, label="attacked (train)"),
        Line2D([0], [0], color=C_TAU, lw=2.0, label="decision boundary"),
    ]
    leg_svm = ax_svm.legend(
        handles=legend_elems, frameon=True, fontsize=8,
        loc="upper right", handletextpad=0.35, framealpha=0.92,
        edgecolor="#E0E0E0", fancybox=True, borderpad=0.35,
    )
    leg_svm.get_frame().set_linewidth(0.6)

    # (4) Test image under the SVM column; decision text to its right
    ax_test = fig.add_subplot(gs[1, 2])
    ax_test.set_anchor("W")
    test_img = clean_224[test_i] if not test_pred else atk_224[test_i]
    ax_test.imshow(np.array(test_img.convert("RGB")))
    ax_test.set_axis_off()

    for r, c in [(1, 0), (1, 1)]:
        ax_empty = fig.add_subplot(gs[r, c])
        ax_empty.set_axis_off()

    fig.canvas.draw()

    # Compact test-image square; leave room for decision text
    pos = ax_test.get_position()
    # ~2.0" display square; leave room on the right for the decision label
    side_in = 2.05
    img_w = side_in / fig.get_figwidth()
    img_h = side_in / fig.get_figheight()
    if img_w > pos.width * 0.65:
        scale = (pos.width * 0.65) / img_w
        img_w *= scale
        img_h *= scale
    if img_h > pos.height * 0.95:
        scale = (pos.height * 0.95) / img_h
        img_w *= scale
        img_h *= scale
    ax_test.set_position([
        pos.x0 + 0.005,
        pos.y0 + 0.45 * (pos.height - img_h),
        img_w,
        img_h,
    ])
    p_test_img = ax_test.get_position()
    fig.text(
        p_test_img.x0 + 0.5 * p_test_img.width,
        p_test_img.y0 - 0.016,
        "Test image",
        ha="center", va="top", fontsize=12, color=C_LABEL,
    )
    fig.text(
        p_test_img.x1 + 0.028,
        p_test_img.y0 + 0.5 * p_test_img.height,
        f"Decision: {decision}",
        ha="left", va="center", fontsize=18, fontweight="bold",
        color=decision_color,
    )

    # Arrows: train → PCA → SVM — mid-gap, labels above (clear of PC2)
    fig.canvas.draw()
    p_train = ax_train.get_position()
    p_pca = ax_pca.get_position()
    p_svm = ax_svm.get_position()
    p_test = ax_test.get_position()

    # High shafts (just under panel titles) so they never cross the PC2 label
    y_arrow = p_pca.y1 - 0.04
    inset = 0.014
    _arrow_between(
        fig,
        (p_train.x1 + inset, y_arrow),
        (p_pca.x0 - inset, y_arrow),
        "PCA",
    )
    _arrow_between(
        fig,
        (p_pca.x1 + inset, y_arrow),
        (p_svm.x0 - inset, y_arrow),
        "SVM",
    )

    # Caption under PCA panel — clear of PC1 / ticks
    fig.text(
        0.5 * (p_pca.x0 + p_pca.x1), p_pca.y0 - 0.062,
        "26-D Attn-last shape features  →  2-D PCA view",
        ha="center", va="top", fontsize=8.5, color=C_NOTE, style="italic",
    )

    # Arrow from test image up to highlighted point (arc away from tick labels)
    disp = ax_svm.transData.transform(z_test)
    inv = fig.transFigure.inverted()
    x_pt, y_pt = inv.transform(disp)
    x_src = p_test.x0 + 0.15 * p_test.width
    y_src = p_test.y1 + 0.008
    arr = FancyArrowPatch(
        (x_src, y_src), (x_pt - 0.008, y_pt),
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.35, color="#888888",
        connectionstyle="arc3,rad=-0.25",
        clip_on=False, zorder=8,
    )
    fig.add_artist(arr)

    fig.text(
        0.5, 0.965,
        "Attack detector — Attn-last shape features  ·  EN&ZH (same recipe for KO / JA)",
        fontsize=13, fontweight="bold", ha="center", va="top", color="#222222",
    )
    fig.text(
        0.5, 0.018,
        f"Train-split PCA view of 26-D features  ·  calibrated {gate['primary_name']}  ·  "
        f"τ={thr:.3f} (val attack recall ≥ {gf.ATTACK_RECALL_TARGET:g})  ·  "
        f"test P(attack)={test_proba:.3f} → {decision}",
        fontsize=8, color=C_NOTE, ha="center", va="bottom",
    )

    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print("Saved", out_path)
    return {
        "test_i": int(test_i),
        "test_proba": test_proba,
        "decision": decision,
        "threshold": thr,
        "primary": gate["primary_name"],
        "pca_var_ratio": pca.explained_variance_ratio_.tolist(),
    }


def main():
    print("Attack-detector pipeline figure (ZH cache + sklearn; no CLIP bake)")
    if not gf.CACHE_PATH.exists():
        print(f"ERROR: missing cache {gf.CACHE_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(gf.SAMPLE_PATH, encoding="utf-8") as f:
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
        im.convert("RGB").resize((gf.DISPLAY_SIZE, gf.DISPLAY_SIZE), Image.BICUBIC)
        for im in rows[image_key]
    ]
    n_images = len(clean_224)
    atk_224 = [
        gf.draw_dual_box(
            clean_224[i],
            gf.CLASSES["en"][int(target[i])], "en",
            gf.CLASSES["zh"][int(target[i])], "zh",
            i, attack_pos,
        )
        for i in range(n_images)
    ]
    print(f"Loaded {n_images} images")

    packed = gf.load_cache()
    clean_cams, atk_cams = packed["clean"], packed["atk"]
    print("Building 26-D features...")
    X, y, img_ids, feature_names = gf.build_feature_matrix(clean_cams, atk_cams, n_images)
    print(f"  X={X.shape}  features={len(feature_names)}")

    gate = gf.train_gate(X, y, img_ids, n_images)
    print(
        f"  primary={gate['primary_name']} thr={gate['threshold']:.4f} "
        f"test_auc={gate['test_auc']:.3f}"
    )

    # Prefer the same clean example as the gating figure (clean skip)
    clean_i, _ = gf.pick_examples(gate, n_images, true, target)
    test_i = clean_i
    print(
        f"  test image i={test_i}  P(attack)={gate['all_proba'][test_i]:.3f}  "
        f"pred={'attacked' if gate['all_pred'][test_i] else 'clean'}"
    )

    meta = make_figure(
        clean_224, atk_224, gate, X, y, n_images, test_i, OUT_PATH,
    )
    summary = {
        "L": "zh",
        "n_features": len(feature_names),
        "fire_clean": gate["fire_clean"],
        "fire_attacked": gate["fire_attacked"],
        "output": str(OUT_PATH),
        **meta,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved", SUMMARY_PATH)


if __name__ == "__main__":
    main()
