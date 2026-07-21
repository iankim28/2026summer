# ko_ja_clean_damage

**Research question:** Can threshold selection and mask geometry reduce KO/JA
clean-image damage under EN∩L `cc_bbox_blur` without giving up much defended
accuracy?

**Status:** Full n=1000 run complete. See `results/winners.json` and diary 2026-07-19.

## Why

The 4-lang trial ([`../four_lang_cc_bbox_blur/`](../four_lang_cc_bbox_blur/)) showed:

| Partner | Clean Δ | Notes |
|---------|--------:|-------|
| ZH | −1.5pp | thr always 0.95 |
| KO / JA | −11 to −23pp | EN∩L less precise; `uni_en` tune picks thr=0.90 |

ZH is left alone. This notebook only ablates `L ∈ {ko, ja}`.

## Protocol

- **Models:** EN OpenAI ViT-B/32, KO `Bingsu/clip-vit-base-patch32-ko`,
  JA `llm-jp/llm-jp-clip-vit-base-patch16`
- **Geometry:** `NUM_BOXES=2`, `FONT_SIZE=24`, frozen `attack_pos` from
  `../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json` (see [`../PROTOCOL.md`](../PROTOCOL.md))
- **Defense core:** EN ∩ L Attn-last → percentile mask → dilate → top-2 CC
  (+ optional bbox snap) → blur fill (`BLUR_RADIUS=12`)
- **Speed:** Attn-last CAMs cached once per image; variants only rebuild masks
- **Sample sizes:** tune on n=100 (10/class), then full n=1000

## Attack matrix

| Attack | Boxes | Score |
|--------|-------|-------|
| `uni_en` | EN + EN | EN + L |
| `uni_l` | L + L | EN + L |
| `multi` | EN + L | EN + L |

## Variants

| Variant | Threshold | dilate | bbox_snap |
|---------|-----------|-------:|:---------:|
| `baseline` | max tune EN attacked acc over `{0.75…0.95}` | 3 | True |
| `thr_floor_095` | fixed 0.95 | 3 | True |
| `pareto_tune` | max `en_atk_acc + 0.5 * mean_clean_delta` on tune | 3 | True |
| `tight_dilate` | pareto thr | 1 | True |
| `no_bbox` | pareto thr | 3 | False |

If `thr_floor_095` tune coverage still exceeds 12%, non-baseline variants also get
`max_coverage=0.12` (raise percentile until coverage ≤ cap).

**Winner rule (per cell):** among variants with mean defended acc within 3pp of
baseline, pick the one with Clean Δ closest to 0; else best Clean Δ overall.

## Notebook

[`ko_ja_clean_damage.ipynb`](ko_ja_clean_damage.ipynb)

Cell sources live in [`_cells/`](_cells/); regenerate with:

```bash
python _build_notebook.py
```

## Outputs

| Path | Contents |
|------|----------|
| `results/{L}/{attack}/{variant}.json` | Per-variant metrics |
| `results/tune_best_cfg.json` | Tuned thresholds / cfgs |
| `results/comparison_summary.json` | All cells × variants |
| `results/winners.json` | Best variant per cell |
| `results/final_comparison.png` | Clean Δ + mean acc bars |

## Success criteria

- `uni_en` Clean Δ for KO and JA **≥ −12pp** without mean def dropping >~3pp vs baseline
- `multi` Clean Δ improved vs −11pp if a geometry variant wins; else residual is heatmap quality
