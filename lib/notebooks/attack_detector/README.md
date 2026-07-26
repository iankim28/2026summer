# attack_detector

**Research question:** Can Attn-last heatmap patterns separate typographic-attacked
images from clean ones, and does gating occlusion on that detector improve Clean Δ?

**Production fill (frozen):** solid **black** (`cc_bbox_black`) for EN + ZH/KO/JA — see [`../PROTOCOL.md`](../PROTOCOL.md). Phase-C JSONs in this folder historically logged **blur**; quote bilingual MIXED from [`results/mixed_2000_black_summary.json`](results/mixed_2000_black_summary.json) / [`../partner_fill_ablation/`](../partner_fill_ablation/).

**Status:** Step 1 (EN∩ZH) + Step 2 (EN∩KO, EN∩JA) complete for `multi`. Attack-recall target **0.99** — all three partners pass success bar (atk drop <1 pp). See `results/comparison_summary.json`.

## Idea

Typographic stickers produce **spiky** attention; clean images look **spread**. Learn a
detector from heatmap features, then occlude **only when attacked** is predicted
(production fill = black; Phase C here used blur).

| Step | Partners | Folder |
|------|----------|--------|
| 1 | EN ∩ ZH | `results/zh/multi/` |
| 2 | EN ∩ KO, EN ∩ JA | `results/ko/multi/`, `results/ja/multi/` |

Phases per partner: **A** PCA/t-SNE → **B** logistic/SVM → **C** gated defense.

## Protocol

- **Models:** EN OpenAI ViT-B/32, ZH Chinese-CLIP B/16, KO `Bingsu/...`, JA `llm-jp/...`
- **Geometry:** frozen `attack_pos` from [`../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`](../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json)
- **Attack:** `multi` (EN + L dual box); score EN + L
- **Defense (this notebook’s Phase C logs):** EN ∩ L Attn-last → `cc_bbox_blur` (thr ≥ 0.95), gated by detector
- **Production defense (quote going forward):** same gate + masks with **black** fill — [`../partner_fill_ablation/`](../partner_fill_ablation/)
- Shared conventions: [`../PROTOCOL.md`](../PROTOCOL.md)

## Notebook

[`attack_detector.ipynb`](attack_detector.ipynb)

```bash
python _build_notebook.py
python run_all.py
```

## Phase A/B visualization

Standalone (uses baked Attn-last cache + sklearn; no CLIP bake):

```bash
python make_phase_ab_viz.py
```

Writes process strips + ROC/confusion into `results/{L}/multi/` (same folder as Phase A/B/C). Diary: `docs/research_diary.md` entry **2026-07-25**.

## Outputs

| Path | Contents |
|------|----------|
| `results/{L}/multi/cache/*.npz` | Baked Attn-last maps |
| `results/{L}/multi/pca_features.png` / `tsne_features.png` | Phase A |
| `results/{L}/multi/phase_a_summary.json` | Phase A metrics |
| `results/{L}/multi/detector_metrics.json` / `feature_importance.png` | Phase B |
| `results/{L}/multi/phase_ab_process.png` | Phase A/B process strip (clean vs attacked + gate) |
| `results/{L}/multi/phase_b_roc_cm.png` | Phase B ROC + test confusion |
| `results/{L}/multi/phase_ab_viz_summary.json` | Viz captions / example indices |
| `results/{L}/multi/gated_comparison.json` / `gated_comparison.png` | Phase C |
| `results/comparison_summary.json` | Roll-up across partners |
| `results/phase_ab_viz_rollups.json` | Cross-partner viz summary |
