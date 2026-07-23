# attack_detector

**Research question:** Can Attn-last heatmap patterns separate typographic-attacked
images from clean ones, and does gating `cc_bbox_blur` on that detector improve Clean Δ?

**Status:** Step 1 (EN∩ZH) + Step 2 (EN∩KO, EN∩JA) complete for `multi`. Attack-recall target **0.99** — all three partners pass success bar (atk drop <1 pp). See `results/comparison_summary.json`.

## Idea

Typographic stickers produce **spiky** attention; clean images look **spread**. Learn a
detector from heatmap features, then blur **only when attacked** is predicted.

| Step | Partners | Folder |
|------|----------|--------|
| 1 | EN ∩ ZH | `results/zh/multi/` |
| 2 | EN ∩ KO, EN ∩ JA | `results/ko/multi/`, `results/ja/multi/` |

Phases per partner: **A** PCA/t-SNE → **B** logistic/SVM → **C** gated defense.

## Protocol

- **Models:** EN OpenAI ViT-B/32, ZH Chinese-CLIP B/16, KO `Bingsu/...`, JA `llm-jp/...`
- **Geometry:** frozen `attack_pos` from [`../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`](../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json)
- **Attack:** `multi` (EN + L dual box); score EN + L
- **Defense:** EN ∩ L Attn-last → `cc_bbox_blur` (thr ≥ 0.95), gated by detector
- Shared conventions: [`../PROTOCOL.md`](../PROTOCOL.md)

## Notebook

[`attack_detector.ipynb`](attack_detector.ipynb)

```bash
python _build_notebook.py
python run_all.py
```

## Outputs

| Path | Contents |
|------|----------|
| `results/{L}/multi/cache/*.npz` | Baked Attn-last maps |
| `results/{L}/multi/pca_features.png` / `tsne_features.png` | Phase A |
| `results/{L}/multi/detector_metrics.json` | Phase B |
| `results/{L}/multi/gated_comparison.json` | Phase C |
| `results/comparison_summary.json` | Roll-up across partners |
