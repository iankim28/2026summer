# heatmap_defense_improvements

**Status:** Done for now (2026-07-18). Winner: **`cc_bbox_blur`** — see [`cc_bbox_blur/`](cc_bbox_blur/).
Further heatmap ablations paused; next experiment is the 4-lang transfer trial in
[`../four_lang_cc_bbox_blur/`](../four_lang_cc_bbox_blur/).

**Research question:** Can we close the remaining gap between Attn-last CAM-intersection
defense (~72.6% mean acc on multilingual attack) and clean accuracy (85.9% EN / 91.4% ZH)
with better gating, masking, fill, and saliency choices — without raising inference cost much?

Frozen baseline lives in [`../attention_defense/`](../attention_defense/) (do not edit).
This folder is the ablation / improvement sandbox (concluded).

## Layout

| Path | Ideas | Notes |
|------|-------|-------|
| [`heatmap_improvements.ipynb`](heatmap_improvements.ipynb) | 1–4, 7 | Gate, union masks, blur-fill, CC/bbox cleanup, peaked heads |
| [`cc_bbox_blur/`](cc_bbox_blur/) | combo | Best two ablations together: CC+bbox snap + blur-fill; pipeline figs in [`../four_lang_cc_bbox_blur/results/pipeline_*.png`](../four_lang_cc_bbox_blur/results/) |
| [`vit16_en/`](vit16_en/) | 5 | EN CLIP ViT-B/16 instead of B/32 |
| [`attn_confdrop_hybrid/`](attn_confdrop_hybrid/) | 6 | Attn shortlist + conf-drop scoring |

## Ideas (short)

1. **Gated masking** — only mask when saliency looks attack-like (peakiness) or models disagree
2. **Union masks** — threshold EN/ZH separately, OR instead of intersection
3. **Blur-fill** — blur masked pixels instead of mean color
4. **CC / bbox** — keep top-2 connected components; optional rectangle snap
5. **EN ViT-B/16** — finer patch grid for English CLIP
6. **Attn + conf-drop hybrid** — shortlist cells by attention, pick with conf-drop
7. **Peaked heads** — average only lowest-entropy attention heads

## Protocol

Same as attention defense: balanced 1000 CIFAR-10, multilingual dual-box attack,
tune thresholds on 100 (10/class), full 1000 if promising. Cost counted in forward/backward
passes per image.

## Outputs

Per-notebook `results/` with `confusion_results_*.json` and comparison charts.
