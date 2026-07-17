# attention_defense

**Research question:** Can ViT self-attention weights replace GradCAM as the saliency signal
in the CAM-intersection defense against multilingual typographic attacks — dropping the
inference cost from 6 to 4 forward passes per image while matching or beating accuracy?

Spun out of `en_zh_multi_uni_attack/` (`_test_attention_defense/`) into its own folder once the
full 1000-image results confirmed attention-based saliency is a clear win, not just a cheaper
alternative.

## Notebook

[`attention_defense_test.ipynb`](attention_defense_test.ipynb) — loads the same balanced
1000-image CIFAR-10 sample (`../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`) and multilingual
typographic attack used throughout `en_zh_multi_uni_attack/`, then compares three saliency
methods feeding the same 2-model CAM-intersection masking defense:

| Method | Passes/image | Mechanism |
|---|---:|---|
| GradCAM | 6 | forward + backward per model; gradient-weighted first-layer activation |
| Attn-last | 4 | CLS→patch attention row from the final transformer block, heads averaged |
| Attn-rollout | 4 | Abnar & Zuidema (2020) rollout — `∏(0.5·A + 0.5·I)` across all 12 layers |

## Results (1000 images, multilingual attack)

| Method | Cost | EN acc | ZH acc | Mean acc | Coverage | Best thr | Clean-acc drop (EN/ZH) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GradCAM | 6 | 32.0% | 34.3% | 33.1% | 26.6% | 0.85 | −35.4pp / −26.5pp |
| Attn-rollout | 4 | 56.3% | 69.6% | 62.9% | 21.4% | 0.85 | −25.6pp / −16.8pp |
| **Attn-last** | **4** | **68.7%** | **76.5%** | **72.6%** | **7.7%** | 0.95 | **−8.8pp / −2.6pp** |

`Attn-last` is a Pareto win over GradCAM: cheaper, more accurate, and far less damage to clean
images. See `results/heatmap_comparison.png` for a visual side-by-side and
`results/final_comparison.png` / `results/threshold_sweep_comparison.png` for the aggregate
charts. Full narrative and plain-language explanation in `docs/research_diary.md`
(2026-07-13 "Attention-based saliency" entry and the 2026-07-16 plain-language recap).

## Outputs

- `results/heatmap_comparison.png` — GradCAM vs Attn-last vs Attn-rollout on 5 example images
- `results/occlusion_comparison.png` — occlusion-sensitivity maps for reference
- `results/threshold_sweep_comparison.png` — accuracy/ASR vs mask threshold, per method
- `results/final_comparison.png` — summary bar chart, all methods vs production GradCAM baseline
- `results/confusion_results_{gradcam,attn_last,attn_rollout}.json` — full per-method metrics
