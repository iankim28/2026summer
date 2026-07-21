# attention_defense / unilingual

**Research question:** Does attention-based saliency still beat GradCAM for the
2-model CAM-intersection defense when the attack is **sole-language dual-box**
(both boxes EN, or both boxes ZH) rather than mixed EN+ZH multilingual?

Companion to the multilingual experiment one level up
([`../attention_defense_test.ipynb`](../attention_defense_test.ipynb)).

## Attack

| Variant | Box 0 | Box 1 |
|---------|-------|-------|
| EN sole | EN target word | EN target word |
| ZH sole | ZH target word | ZH target word |

Geometry matches the rest of the suite: `NUM_BOXES=2`, `FONT_SIZE=24`, `PAD=8`,
non-overlapping random placement seeded by `img_idx * NUM_BOXES + box_i`.

## Protocol

1. Load balanced 1000-image CIFAR-10 sample (`../../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`).
2. Build EN-only and ZH-only dual-box attacked images.
3. Compare GradCAM (cost 6) vs Attn-last / Attn-rollout (cost 4) feeding the same
   EN∩ZH intersection mask.
4. **Tune on 100** (10/class) per attack lang → pick best threshold by EN masked acc.
5. **Full 1000** only if the tune looks promising (manual gate — run §8 when ready).

## Notebook

[`attention_defense_test.ipynb`](attention_defense_test.ipynb)

## Results (1000 images)

Multilingual row included for comparison (from parent folder). Full write-up in
`docs/research_diary.md` (2026-07-16 unilingual entry).

| Attack | Method | Cost | Mean acc | Best thr | Clean drop (EN / ZH) |
|---|---|---:|---:|---:|---|
| Multilingual (EN+ZH) | Attn-last | 4 | **72.6%** | 0.95 | −8.8pp / −2.6pp |
| Unilingual EN+EN | Attn-last | 4 | **67.6%** | 0.95 | −8.8pp / −2.6pp |
| Unilingual EN+EN | GradCAM | 6 | 28.7% | 0.80 | −44.2pp / −36.2pp |
| Unilingual ZH+ZH | Attn-last | 4 | **62.5%** | 0.85 | −29.6pp / −22.7pp |
| Unilingual ZH+ZH | GradCAM | 6 | 50.9% | 0.95 | −17.5pp / −10.4pp |

Attn-last still wins on both sole-language attacks; the margin over GradCAM is
large on EN+EN and smaller on ZH+ZH.

## Outputs

- `results/heatmap_comparison.png` — GradCAM vs attention on EN dual-box examples
- `results/occlusion_comparison.png`
- `results/threshold_sweep_comparison.png` — tune sweeps per attack lang × variant
- `results/final_comparison.png` — summary vs placement GradCAM dual-box baseline
- `results/confusion_results_{gradcam,attn_last,attn_rollout}_{en,zh}.json`
