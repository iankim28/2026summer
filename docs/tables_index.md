# Tables index

**Protocol:** frozen dual-box CIFAR-10 n=1000, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**Paper export:** [`paper_tables_final.md`](paper_tables_final.md).  
**Canonical detail:** [`4_lang_table.md`](4_lang_table.md), [`ablation_study.md`](ablation_study.md).  
**Numbers:** one decimal place throughout.  
**Bolding:** Table 1 — Ours rows + Raw CLIP floor (mean acc (attacked) / mean mixed acc). Ablations — production / chosen setting only.

**Scoped means:** Mean acc (attacked) / mean mixed acc average only languages scored in that row (4-lang when filled; 2-lang for each Ours EN∩L row).

**Reading the tables.** Language codes: EN / ZH / KO / JA = English / Chinese / Korean / Japanese. **acc** = top-1 accuracy (%). Unless noted, accuracy columns are measured **under attack**. Language cells in Table 1 are **acc / mixed acc**. **mixed acc** (MIXED2000) = 0.5 × attacked acc + 0.5 × clean-policy acc on the same 1000 images. **Clean Δ EN** = Δ (delta / change) in English clean (unattacked) accuracy under the defense, in percentage points (pp). **ASR** = attack success rate. **Cost** = passes per image. Em dash (—) = not evaluated.

---

## Table 1 — Main: baselines vs ours (all languages)

| Method | EN acc / mixed acc | ZH acc / mixed acc | KO acc / mixed acc | JA acc / mixed acc | Mean acc (attacked) | Mean mixed acc | Clean Δ EN | Cost |
|--------|-------------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------:|-----------:|-----:|
| No defense (Raw CLIP) | 4.5 / 45.2 | 6.4 / 48.9 | 11.6 / 50.6 | 6.0 / 49.3 | **7.1%** | **48.5%** | 0.0 pp | 1 |
| 4×4 grid occlusion | 47.8 / 65.3 | 49.2 / 66.8 | 52.8 / 65.8 | 55.1 / 70.8 | 51.2% | 67.2% | −3.2 pp | 62 |
| OCR + blur | 72.8 / 79.1 | 74.7 / 82.7 | 80.2 / 84.5 | 87.6 / 89.7 | **78.8%** | **84.0%** | −0.6 pp | 3 |
| Defense-Prefix | 73.8 / 81.7 | 81.4 / 86.2 | 69.1 / 79.4 | 84.8 / 89.2 | 77.3% | 84.1% | +0.5 pp | 2 |
| SamplingTAR + blur | 67.3 / 72.5 | 68.9 / 79.3 | 59.0 / 62.9 | 51.5 / 72.1 | 61.7% | 71.7% | −8.3 pp | 3 |
| Dyslexify + blur | 66.9 / 72.4 | 68.2 / 79.6 | 62.9 / 67.1 | 43.2 / 67.0 | 60.3% | 71.5% | −8.1 pp | 3 |
| **Ours (EN∩ZH)** | **72.9 / 79.4** | **76.5 / 84.0** | — | — | **74.7%** | **81.7%** | **−0.1 pp** | **4** |
| **Ours (EN∩KO)** | **65.6 / 75.5** | — | **73.1 / 81.3** | — | **69.4%** | **78.4%** | **−0.6 pp** | **4** |
| **Ours (EN∩JA)** | **68.9 / 77.4** | — | — | **82.8 / 87.7** | **75.9%** | **82.5%** | **0.0 pp** | **4** |

All baseline cells are n=1000 finals. Sources: [`grid_occlusion/results/comparison_summary_4lang_final_n1000.json`](../lib/notebooks/paper_baselines/grid_occlusion/results/comparison_summary_4lang_final_n1000.json); DP KO/JA `comparison_summary_final_n1000_{ko,ja}.json`; hybrid ZH/KO/JA `comparison_summary_final_n1000_hybrid_{zh,ko,ja}.json`; MIXED from [`mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json).

---

## Ablations

### Table 2 — Attack details

Gated = EN∩ZH `cc_bbox_black`. Text vs white gated from [`content_occlusion_n1000.json`](../lib/notebooks/attack_component_ablation/results/content_occlusion_n1000.json).  
Bold = production setting (font **24**, content **full**, boxes **2**) and that row’s gated acc.

| Ablation | Setting | EN acc (no defense) | EN acc (gated) | ZH acc (gated) |
|----------|---------|--------------------:|---------------:|---------------:|
| Font size | 12 | 9.8% | 76.0% | 83.9% |
| Font size | **24** | 4.5% | **72.9%** | **76.5%** |
| Font size | 40 | 3.0% | 58.3% | 56.3% |
| Sticker type | white pad only | 75.5% | 70.9% | 75.0% |
| Sticker type | letters only | 9.6% | 75.8% | 82.7% |
| Sticker type | **full sticker** | 4.5% | **72.9%** | **76.5%** |
| Number of boxes | 1 | 5.7% | 78.2% | 82.9% |
| Number of boxes | **2** | 4.5% | **72.9%** | **76.5%** |
| Number of boxes | 3 | 2.4% | 45.9% | 56.1% |

---

### Table 3 — Defense on sticker / text / hybrid (attack)

Gated `cc_bbox_black`, n=1000. Raw CLIP = no-defense arm. Source: [`occlusion_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json).  
Bold = production arm **EN∩ZH** (not EN-only, even when EN-only is higher on animals).

| Overlay | EN acc (no defense) | EN∩ZH acc | EN-only acc | ASR (EN∩ZH) | Clean Δ EN (EN∩ZH) |
|---------|--------------------:|----------:|------------:|------------:|-------------------:|
| All text (typographic) | 4.5% | **72.9%** | 68.0% | 3.7% | −0.1 pp |
| Text + animal (hybrid) | 1.3% | **43.5%** | 40.2% | 27.3% | −0.1 pp |
| Animal only | 11.0% | **20.6%** | 32.9% | 54.2% | −1.8 pp |

---

### Table 4 — Occlusion algorithm (method)

Bold = production choices: **EN∩ZH**, **gated**, **black**. Gate rows report English mixed acc (Phase-C blur) for always on vs gated.

| Ablation | Setting | EN acc | Clean Δ EN | Mixed acc |
|----------|---------|-------:|-----------:|----------:|
| Language used | English only | 68.0% | −0.1 pp | — |
| Language used | **EN∩ZH** | **72.9%** | **−0.1 pp** | **81.7%** bilingual |
| Language used | EN∩ZH∩KO∩JA | 72.0% | −0.6 pp | 83.4% mean (4-lang) |
| Gate | always on (ZH / KO / JA) | — | −2.2 / −12.1 / −13.6 pp | EN mixed acc 76.8 / 69.6 / 70.5% |
| Gate | **gated** (ZH / KO / JA) | — | **0.0 / −0.3 / 0.0 pp** | EN mixed acc **77.9 / 75.3 / 77.2%** |
| Fill type | no fill | 60.5% | ≈0.0 pp | 73.2% EN |
| Fill type | blur | 69.9% | ≈0.0 pp | 77.9% EN |
| Fill type | mean color | 70.4% | ≈0.0 pp | 78.2% EN |
| Fill type | **black** | **72.9%** | ≈0.0 pp | **79.4%** EN |

Gate rows use Phase-C blur-log EN mixed acc. Fill and language-count rows use production black.
