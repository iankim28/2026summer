# cc_bbox_blur

**Question:** Does combining the two winning ablations from
[`../heatmap_improvements.ipynb`](../heatmap_improvements.ipynb) beat either alone?

| Variant | Mask | Fill |
|---------|------|------|
| `attn_last_baseline` | intersection | mean |
| `blur_fill` | intersection | blur |
| `cc_bbox` | top-2 CC + bbox snap | mean |
| **`cc_bbox_blur`** | top-2 CC + bbox snap | **blur** |

Notebook: [`cc_bbox_blur.ipynb`](cc_bbox_blur.ipynb)

## Results (multilingual n=1000)

| Variant | Mean acc | Clean Δ (mean) |
|---------|--------:|---------------:|
| attn_last_baseline | 72.6% | −5.7pp |
| blur_fill | 73.4% | −1.6pp |
| cc_bbox | 74.9% | −2.7pp |
| **cc_bbox_blur** | **74.9%** | **−1.5pp** |

Combo matches best attacked accuracy and best clean tradeoff. See `results/final_comparison.png`.

Qualitative pipeline figures (how the method works) live under
[`../../four_lang_cc_bbox_blur/results/pipeline_*.png`](../../four_lang_cc_bbox_blur/results/).
