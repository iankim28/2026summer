# Tables index

**Protocol:** frozen dual-box CIFAR-10 n=1000, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**Canonical detail:** [`4_lang_table.md`](4_lang_table.md), [`ablation_study.md`](ablation_study.md).

---

## Table 1 — Main: baselines vs ours (all languages)

| Method | EN atk | ZH atk | KO | JA | Mean atk | EN MIXED | ZH MIXED | Scope MIXED | Cost |
|--------|-------:|-------:|:--:|:--:|---------:|---------:|---------:|------------:|-----:|
| OCR + blur | 72.8% | 74.7% | — | — | **73.8%** | **79.05%** | **82.70%** | **80.88%** (EN+ZH) | 3 |
| Defense-Prefix | **73.8%** | 44.5% | — | — | **59.2%** | **81.65%** | **68.15%** | **74.90%** (EN+ZH) | 2 |
| SamplingTAR hybrid | 67.3% | — | — | — | 67.3% | 72.45% | — | 72.45% (EN) | 3 |
| Dyslexify hybrid | 66.9% | — | — | — | 66.9% | 72.35% | — | 72.35% (EN) | 3 |
| **Ours (EN∩ZH black)** | **72.9%** | **76.5%** | — | — | **74.7%** | **79.35%** | **83.95%** | **81.65%** bilingual | 4 |
| **Ours (EN∩KO black)** | 65.6% | — | **73.1%** | — | **69.4%** | 75.45% | — | **78.35%** bilingual | 4 |
| **Ours (EN∩JA black)** | 68.9% | — | — | **82.8%** | **75.9%** | 77.40% | — | **82.53%** bilingual | 4 |

---

## Ablations

### Table 2 — Attack details

#### Font size

| Font | never EN (acc/ASR) | gated EN | gated ZH | Clean Δ EN |
|------|-------------------:|---------:|---------:|-----------:|
| 12 | 9.8% / 89.9% | **76.0%** | **83.9%** | +0.0 pp |
| **24** (production) | 4.5% / 95.3% | **72.9%** | **76.5%** | −0.1 pp |
| 40 | 3.0% / 96.9% | 58.3% | 56.3% | −0.4 pp |

#### Text only vs text + white background

| Mode | EN ASR | ZH ASR | EN∩ZH IoU | EN∩ZH det@0.1 | Undefended EN acc |
|------|-------:|-------:|----------:|--------------:|------------------:|
| white_only (pad, no letters) | 2.3% | 2.1% | 0.083 | 41.1% | **75.5%** |
| text_only (letters, no pad) | **89.8%** | **84.5%** | **0.669** | **100%** | 9.6% |
| full (white pad + letters) | **95.3%** | **93.6%** | **0.691** | **100%** | 4.5% |

#### Number of boxes

| Boxes | top_k | never EN (acc/ASR) | gated EN | gated ZH | Clean Δ EN |
|------:|------:|-------------------:|---------:|---------:|-----------:|
| 1 | 1 | 5.7% / 94.1% | **78.2%** | **82.9%** | −0.1 pp |
| **2** (production) | 2 | 4.5% / 95.3% | **72.9%** | **76.5%** | −0.1 pp |
| 3 | 3 | 2.4% / 97.5% | **45.9%** | **56.1%** | −0.8 pp |

---

### Table 3 — Defense on sticker / text / hybrid (attack)

Gated `cc_bbox_black`, n=1000.

| Mode | Arm | EN acc | EN ASR | Clean Δ EN |
|------|-----|-------:|-------:|-----------:|
| all_text (typographic) | EN∩ZH | **72.9%** | 3.7% | −0.1 pp |
| all_text | EN-only | 68.0% | 3.6% | −0.1 pp |
| mixed (hybrid) | EN∩ZH | **43.5%** | 27.3% | −0.1 pp |
| mixed | EN-only | 40.2% | 27.4% | −0.1 pp |
| all_sticker (visual / animal) | EN∩ZH | 20.6% | 54.2% | −1.8 pp |
| all_sticker | EN-only | **32.9%** | **28.5%** | −6.1 pp |

---

### Table 4 — Occlusion algorithm (method)

#### Single-language model occlusion

| Mode | Arm | EN acc | EN ASR | Clean Δ EN |
|------|-----|-------:|-------:|-----------:|
| all_text | gated EN-only | 68.0% | 3.6% | −0.1 pp |
| all_text | gated EN∩ZH (2-lang ref) | **72.9%** | 3.7% | −0.1 pp |
| all_sticker | gated EN-only | **32.9%** | **28.5%** | −6.1 pp |
| all_sticker | gated EN∩ZH (2-lang ref) | 20.6% | 54.2% | −1.8 pp |

#### Four-language model occlusion

True EN ∩ ZH ∩ KO ∩ JA Attn-last → gated `cc_bbox_black`.

| Arm | EN atk | ZH atk | KO atk | JA atk | Mean atk | Mean MIXED | Clean Δ (EN/ZH/KO/JA) |
|-----|-------:|-------:|-------:|-------:|---------:|-----------:|----------------------:|
| never | 4.5% | 6.4% | 11.6% | 6.0% | 7.1% | 48.5% | — |
| always | 72.1% | 76.5% | 76.6% | 84.6% | **77.5%** | 73.0% | −21.7 / −23.6 / −20.5 / −19.3 pp |
| **gated** | **72.0%** | **76.4%** | **76.5%** | **84.5%** | **77.4%** | **83.40%** | −0.6 / −0.3 / −0.3 / −0.4 pp |

Pairwise EN∩L bilingual MIXED (production 2-lang): ZH **81.65%** / KO **78.35%** / JA **82.53%** (mean **80.84%**).

#### Gate on vs off

| Partner | Always Clean Δ | Gated Clean Δ | Always MIXED | Gated MIXED | Gated−always |
|---------|---------------:|--------------:|-------------:|------------:|-------------:|
| ZH | −1.45 pp | **0.00 pp** | 80.60% | 81.28% | +0.68 |
| KO | −11.25 pp | **−0.20 pp** | 73.20% | 78.50% | +5.30 |
| JA | −11.50 pp | **0.00 pp** | 76.80% | 82.45% | +5.65 |

#### Fill type

| Fill | EN atk | EN clean | EN MIXED2000 |
|------|-------:|---------:|-------------:|
| neglect | 60.5% | 85.9% | 73.20% |
| blur | 69.9% | 85.9% | 77.90% |
| mean | 70.4% | 85.9% | 78.15% |
| **black** | **72.9%** | **85.8%** | **79.35%** |

Partner bilingual MIXED (gated): ZH/JA win **black** (81.65% / 82.53%); KO blur +0.15pp (ablation only — production stays black for all langs).
