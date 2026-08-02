# Tables index

**Protocol:** frozen dual-box CIFAR-10 n=1000, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**Canonical detail:** [`4_lang_table.md`](4_lang_table.md), [`ablation_study.md`](ablation_study.md).  
**Numbers:** one decimal place throughout.  
**Bolding:** Table 1 — Ours rows + Raw CLIP floor (mean atk / Scope MIXED). Ablations — production / chosen setting only (not every column max).

---

## Table 1 — Main: baselines vs ours (all languages)

| Method | EN atk | ZH atk | KO | JA | Mean atk | EN MIXED | ZH MIXED | Scope MIXED | Cost |
|--------|-------:|-------:|:--:|:--:|---------:|---------:|---------:|------------:|-----:|
| Raw CLIP (no defense) | 4.5% | 6.4% | 11.6% | 6.0% | **7.1%** | 45.2% | 48.9% | **48.5%** (4-lang) | 1 |
| OCR + blur | 72.8% | 74.7% | — | — | 73.8% | 79.1% | 82.7% | 80.9% (EN+ZH) | 3 |
| Defense-Prefix | 73.8% | 44.5% | — | — | 59.2% | 81.7% | 68.2% | 74.9% (EN+ZH) | 2 |
| SamplingTAR hybrid | 67.3% | — | — | — | 67.3% | 72.5% | — | 72.5% (EN) | 3 |
| Dyslexify hybrid | 66.9% | — | — | — | 66.9% | 72.4% | — | 72.4% (EN) | 3 |
| **Ours (EN∩ZH black)** | **72.9%** | **76.5%** | — | — | **74.7%** | **79.4%** | **84.0%** | **81.7%** bilingual | **4** |
| **Ours (EN∩KO black)** | **65.6%** | — | **73.1%** | — | **69.4%** | **75.5%** | — | **78.4%** bilingual | **4** |
| **Ours (EN∩JA black)** | **68.9%** | — | — | **82.8%** | **75.9%** | **77.4%** | — | **82.5%** bilingual | **4** |

Raw CLIP from [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json) `arms.never`.

#### Table 1 (slim) — paper-body candidate

Same rows; fewer columns for print width. Mean atk / EN MIXED / ZH MIXED live in the full table above (or caption).

| Method | EN | ZH | KO | JA | Scope MIXED | Cost |
|--------|---:|---:|:--:|:--:|------------:|-----:|
| Raw CLIP (no defense) | 4.5% | 6.4% | 11.6% | 6.0% | **48.5%** (4-lang) | 1 |
| OCR + blur | 72.8% | 74.7% | — | — | 80.9% (EN+ZH) | 3 |
| Defense-Prefix | 73.8% | 44.5% | — | — | 74.9% (EN+ZH) | 2 |
| SamplingTAR hybrid | 67.3% | — | — | — | 72.5% (EN) | 3 |
| Dyslexify hybrid | 66.9% | — | — | — | 72.4% (EN) | 3 |
| **Ours (EN∩ZH black)** | **72.9%** | **76.5%** | — | — | **81.7%** bilingual | **4** |
| **Ours (EN∩KO black)** | **65.6%** | — | **73.1%** | — | **78.4%** bilingual | **4** |
| **Ours (EN∩JA black)** | **68.9%** | — | — | **82.8%** | **82.5%** bilingual | **4** |

---

## Ablations

### Table 2 — Attack details

Gated = EN∩ZH `cc_bbox_black`. Text vs white gated from [`content_occlusion_n1000.json`](../lib/notebooks/attack_component_ablation/results/content_occlusion_n1000.json).  
Bold = production setting (font **24**, content **full**, boxes **2**) and that row’s gated acc.

| Ablation | Setting | EN acc (no def) | EN acc (gated) | ZH acc (gated) |
|----------|---------|----------------:|---------------:|---------------:|
| Font size | 12 | 9.8% | 76.0% | 83.9% |
| Font size | **24** (production) | 4.5% | **72.9%** | **76.5%** |
| Font size | 40 | 3.0% | 58.3% | 56.3% |
| Text vs white bg | white_only (pad, no letters) | 75.5% | 70.9% | 75.0% |
| Text vs white bg | text_only (letters, no pad) | 9.6% | 75.8% | 82.7% |
| Text vs white bg | **full** (white pad + letters) | 4.5% | **72.9%** | **76.5%** |
| Number of boxes | 1 (`top_k=1`) | 5.7% | 78.2% | 82.9% |
| Number of boxes | **2** (production, `top_k=2`) | 4.5% | **72.9%** | **76.5%** |
| Number of boxes | 3 (`top_k=3`) | 2.4% | 45.9% | 56.1% |

---

### Table 3 — Defense on sticker / text / hybrid (attack)

Gated `cc_bbox_black`, n=1000. Raw CLIP = never-defend arm. Source: [`occlusion_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json).  
Bold = production arm **Ours EN∩ZH** (not EN-only, even when EN-only is higher on animals).

| Mode | Raw CLIP EN | Ours EN∩ZH | Ours EN-only | EN ASR (EN∩ZH) | Clean Δ EN (EN∩ZH) |
|------|------------:|-----------:|-------------:|---------------:|-------------------:|
| all_text (typographic) | 4.5% | **72.9%** | 68.0% | 3.7% | −0.1 pp |
| mixed (hybrid) | 1.3% | **43.5%** | 40.2% | 27.3% | −0.1 pp |
| all_sticker (visual / animal) | 11.0% | **20.6%** | 32.9% | 54.2% | −1.8 pp |

---

### Table 4 — Occlusion algorithm (method)

Bold = production choices: **EN∩ZH (2)**, **gated**, **black**. (4-way is an ablation variant, not bolded as the default recipe.)

| Ablation | Setting | EN atk | Other atk / mean | Clean Δ EN | Score |
|----------|---------|-------:|-----------------:|-----------:|------:|
| # language models | EN-only (text) | 68.0% | — | −0.1 pp | — |
| # language models | **EN∩ZH (2)** | **72.9%** | ZH **76.5%** | −0.1 pp | bilingual MIXED **81.7%** |
| # language models | EN∩ZH∩KO∩JA (4) gated | 72.0% | mean 77.4% | −0.6 pp | mean MIXED 83.4% |
| Gate | always-on (ZH / KO / JA) | — | — | −1.5 / −11.3 / −11.5 pp | MIXED 80.6 / 73.2 / 76.8% |
| Gate | **gated** (ZH / KO / JA) | — | — | **0.0 / −0.2 / 0.0 pp** | MIXED **81.3 / 78.5 / 82.5%** |
| Fill | neglect | 60.5% | — | ≈0 | EN MIXED 73.2% |
| Fill | blur | 69.9% | — | ≈0 | EN MIXED 77.9% |
| Fill | mean | 70.4% | — | ≈0 | EN MIXED 78.2% |
| Fill | **black** | **72.9%** | — | ≈0 | EN MIXED **79.4%** |

Gate rows use Phase-C blur-log MIXED (design history). Fill and #langs rows use production black.

#### Table 4 (single, clearer Score) — paper-body candidate

Still **one** table. Drop “Other atk / mean”; make Score always a MIXED % with the kind named in the Setting / note so slash-cells stay only where partners are listed together.  
Bold = same production rule as above.

| Ablation | Setting | EN atk | Clean Δ EN | MIXED |
|----------|---------|-------:|-----------:|------:|
| # language models | EN-only (text) | 68.0% | −0.1 pp | — |
| # language models | **EN∩ZH (2)** | **72.9%** | −0.1 pp | **81.7%** bilingual |
| # language models | EN∩ZH∩KO∩JA (4) gated | 72.0% | −0.6 pp | 83.4% mean (4-lang) |
| Gate | always-on ZH / KO / JA | — | −1.5 / −11.3 / −11.5 pp | 80.6 / 73.2 / 76.8% (blur-log) |
| Gate | **gated** ZH / KO / JA | — | **0.0 / −0.2 / 0.0 pp** | **81.3 / 78.5 / 82.5%** (blur-log) |
| Fill | neglect | 60.5% | ≈0 | 73.2% EN |
| Fill | blur | 69.9% | ≈0 | 77.9% EN |
| Fill | mean | 70.4% | ≈0 | 78.2% EN |
| Fill | **black** | **72.9%** | ≈0 | **79.4%** EN |
