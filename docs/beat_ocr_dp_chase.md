# Chase: beat OCR+blur (and DP) on 4-lang means

**Status:** chase complete — winning Ours arm beats OCR+blur and Defense-Prefix on 4-lang mean atk / mean MIXED.  
**Protocol:** frozen dual-box CIFAR-10 n=1000, attack = EN+ZH `multi`, thr ≥ 0.95, dilate=3, top_k=2, fill=black, CUDA.  
**Do not merge into paper tables** — this file is a separate chase log. Frozen exports (`tables_index.md`, `paper_tables_final.md`, `4_lang_table.md`) are untouched.

---

## Winning Ours recipe

| Piece | Choice |
|-------|--------|
| Mask | **`en_cap_mean`**: EN ∩ mean(ZH, KO, JA) Attn-last → cc_bbox → black |
| Gate | Phase-C features on EN vs mean(ZH,KO,JA); fire atk **100%** / clean **0.5%** |
| EN | OpenAI ViT-B/32 |
| ZH | `OFA-Sys/chinese-clip-vit-base-patch16` |
| KO | **`Bingsu/clip-vit-large-patch14-ko`** (upgrade from B/32) |
| JA | **`llm-jp/llm-jp-clip-vit-large-patch14`** (upgrade from B/16) |
| Cost | ~8 image forwards for saliency (4 models × clean/atk path) + classify |

Adaptive add-ons (low-coverage expand, target-persistence second pass) were tried; they did **not** meaningfully beat plain `en_cap_mean` (same 87.1% MIXED within rounding). Production chase pick = plain gated `en_cap_mean` + L/14 partners.

---

## Main 4-lang comparison

Language cells = **atk acc / mixed acc** (%). MIXED2000 = 0.5 × atk + 0.5 × clean-policy on the same 1000 indices. Means average EN+ZH+KO+JA.

| Method | EN acc / mixed | ZH acc / mixed | KO acc / mixed | JA acc / mixed | Mean atk | Mean MIXED | Clean Δ EN | Cost |
|--------|---------------:|---------------:|---------------:|---------------:|---------:|-----------:|-----------:|-----:|
| OCR + blur | 72.8 / 79.1 | 74.7 / 82.7 | 80.2 / 84.5 | 87.6 / 89.7 | **78.8%** | **84.0%** | −0.6 pp | 3 |
| Defense-Prefix | 73.8 / 81.7 | 81.4 / 86.2 | 69.1 / 79.4 | 84.8 / 89.2 | **77.3%** | **84.1%** | +0.5 pp | 2 |
| **Ours (en_cap_mean, KO/JA L/14)** | **72.5 / 79.1** | **76.0 / 83.7** | **91.4 / 94.0** | **86.4 / 91.7** | **81.6%** | **87.1%** | **−0.2 pp** | **~8** |
| Ours (4-way ∩, B32/B16 baseline) | 72.0 / 78.7 | 76.4 / 83.8 | 76.5 / 82.9 | 84.5 / 88.3 | 77.4% | 83.4% | −0.6 pp | ~8 |

**Verdict:** winning Ours **+2.8 pp mean atk** and **+3.1 pp mean MIXED** vs OCR+blur; **+4.3 / +3.0 pp** vs Defense-Prefix. Clean Δ EN stays near zero (−0.2 pp).

---

## Per-lang deltas (Ours winner − baseline)

| Lang | Ours − OCR (atk) | Ours − OCR (MIXED) | Ours − DP (atk) | Ours − DP (MIXED) |
|------|-----------------:|-------------------:|----------------:|------------------:|
| EN | −0.3 | 0.0 | −1.3 | −2.6 |
| ZH | +1.3 | +1.0 | −5.4 | −2.5 |
| KO | **+11.2** | **+9.5** | **+22.3** | **+14.6** |
| JA | −1.2 | +2.0 | +1.6 | +2.5 |
| **Mean** | **+2.8** | **+3.1** | **+4.3** | **+3.0** |

Win is driven by the KO (and JA) partner upgrades: KO atk 76.5% → **91.4%** under the same gated occlusion family. EN remains the hardest cell and still trails DP on EN MIXED (−2.6 pp).

---

## What we tried (short)

### Phase 1 — Mask aggregation (current B32/B16 partners)

Source: [`lib/notebooks/four_way_occlusion/results/agg_chase_n1000_d3_k2.json`](../lib/notebooks/four_way_occlusion/results/agg_chase_n1000_d3_k2.json)

| Arm | Mean atk | Mean MIXED |
|-----|---------:|-----------:|
| intersect4 (baseline) | 77.4% | **83.4%** |
| en_cap_mean | 76.4% | 83.0% |
| pair_union | 75.2% | 82.3% |
| en_cap_maj | 73.8% | 81.8% |
| en_cap_max | 73.9% | 81.7% |

Softer aggregation alone did **not** beat OCR (still ≤ 83.4% MIXED).

### Phase 2 — KO / JA ViT-L/14

Source: [`lib/notebooks/four_way_occlusion/results/upgrade_chase_n1000_koL14_jaL14_adapt.json`](../lib/notebooks/four_way_occlusion/results/upgrade_chase_n1000_koL14_jaL14_adapt.json)

| Variant | Mean atk | Mean MIXED |
|---------|---------:|-----------:|
| **en_cap_mean** | **81.6%** | **87.1%** |
| intersect4 | 81.2% | 86.9% |
| pairwise EN∩KO (scored 4-lang) | 79.8% | 86.2% |

### Phase 3 — Adaptive occlusion

On L/14 models, `+cov_expand` / `+second_pass` matched or rounded to the same MIXED as plain `en_cap_mean` (87.1%). Not needed for the OCR win.

---

## Baseline sources

| Method | JSON |
|--------|------|
| OCR + blur 4-lang | [`ocr_blur/results/comparison_summary_4lang_final_n1000.json`](../lib/notebooks/paper_baselines/ocr_blur/results/comparison_summary_4lang_final_n1000.json) |
| DP EN+ZH | [`defense_prefix/results/comparison_summary_final_n1000_en_zh.json`](../lib/notebooks/paper_baselines/defense_prefix/results/comparison_summary_final_n1000_en_zh.json) |
| DP KO n=1000 | [`defense_prefix/results/comparison_summary_final_n1000_ko.json`](../lib/notebooks/paper_baselines/defense_prefix/results/comparison_summary_final_n1000_ko.json) |
| DP JA n=1000 | [`defense_prefix/results/comparison_summary_final_n1000_ja.json`](../lib/notebooks/paper_baselines/defense_prefix/results/comparison_summary_final_n1000_ja.json) |
| Ours 4-way baseline | [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json) |
| Ours winner | [`four_way_occlusion/results/upgrade_chase_n1000_koL14_jaL14_adapt.json`](../lib/notebooks/four_way_occlusion/results/upgrade_chase_n1000_koL14_jaL14_adapt.json) |

DP KO MIXED = 0.5×69.1% + 0.5×89.7% = **79.4%**. DP JA MIXED = 0.5×84.8% + 0.5×93.6% = **89.2%**. DP 4-lang mean MIXED = **84.1%**.

---

## Runners

```bash
cd lib/notebooks/four_way_occlusion
python run_agg_chase.py --n 1000
python run_upgrade_chase.py --n 1000 --ko-l14 --ja-l14 --adaptive --arms intersect4,en_cap_mean
```
