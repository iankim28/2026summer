# four_lang_cc_bbox_blur

**Research question:** Does the EN/ZH winner **`cc_bbox_blur`** transfer to KO and JA
under the same dual-box typographic geometry?

**Status:** Initial trial complete (n=1000). See `results/comparison_summary.json` and diary 2026-07-18/19.

## Protocol

- **Models:** EN OpenAI ViT-B/32, ZH Chinese-CLIP B/16, KO `Bingsu/clip-vit-base-patch32-ko`,
  JA `llm-jp/llm-jp-clip-vit-base-patch16`
- **Geometry:** `NUM_BOXES=2`, `FONT_SIZE=24`, frozen `attack_pos` from
  `../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json` (see [`../PROTOCOL.md`](../PROTOCOL.md))
- **Defense:** EN ∩ L Attn-last → percentile mask → dilate → top-2 CC + bbox snap → blur fill
  (`BLUR_RADIUS=12`). Cost = 4 passes / image.
- **Sample sizes:** tune threshold on n=100 (10/class), then **enforce thr ≥ 0.95** for full n=1000.
  Logs both `threshold_free` (raw tune) and `threshold` (floored).

## Option B matrix

For each partner language `L ∈ {zh, ko, ja}`:

| Attack | Boxes | Score |
|--------|-------|-------|
| `uni_en` | EN + EN | EN + L |
| `uni_l` | L + L | EN + L |
| `multi` | EN + L | EN + L |

## Notebook

[`four_lang_cc_bbox_blur.ipynb`](four_lang_cc_bbox_blur.ipynb)

Cell sources live in [`_cells/`](_cells/); regenerate with:

```bash
python _build_notebook.py
```


## Outputs

| Path | Contents |
|------|----------|
| `results/{L}/{attack}/confusion_results.json` | Per-cell metrics |
| `results/tune_best_cfg.json` | Best threshold per (L, attack) |
| `results/comparison_summary.json` | Rolled-up table |
| `results/final_comparison.png` | Bar chart |
| `results/pipeline_steps.png` | Qualitative: 8-stage `cc_bbox_blur` walkthrough (one example) |
| `results/pipeline_examples.png` | Qualitative: 5 examples × key stages |
| `results/pipeline_fill_compare.png` | Mean fill vs blur fill after CC+bbox |

Regenerate pipeline figures with [`make_pipeline_viz.py`](make_pipeline_viz.py).
