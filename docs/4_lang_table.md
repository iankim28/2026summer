# 4-lang main results table

> One-page index of all tables: [`tables_index.md`](tables_index.md).

**Protocol:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**MIXED2000:** `0.5 * attacked_acc + 0.5 * clean_policy_acc` (equal weight on the same 1000 indices once clean + once attacked).  
**Bilingual MIXED2000 (partner L):** `0.5 * mean(EN,L atk) + 0.5 * mean(EN,L clean_policy)`.  
**Per-lang MIXED (baselines):** `0.5 * lang_atk + 0.5 * lang_clean_policy` from [`mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json) `per_lang`.

Sources: [`baseline_comparison.md`](baseline_comparison.md), [`partner_fill_ablation/results/leaderboard.json`](../lib/notebooks/partner_fill_ablation/results/leaderboard.json), [`en_neglect_vs_blur/results/gated_n1000.json`](../lib/notebooks/en_neglect_vs_blur/results/gated_n1000.json), [`paper_baselines/results/mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json), [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json) (`arms.never` = Raw CLIP).

---

## Table 1 — Main: language coverage (baselines + ours)

Primary comparison. Prefer **per-lang MIXED** when a language is evaluated; Em dashes (—) = not evaluated under the current baseline ports.

| Method | EN atk | ZH atk | KO | JA | Mean atk | EN MIXED | ZH MIXED | Scope MIXED | Cost |
|--------|-------:|-------:|:--:|:--:|---------:|---------:|---------:|------------:|-----:|
| Raw CLIP (no defense) | 4.5% | 6.4% | 11.6% | 6.0% | **7.1%** | 45.2% | 48.9% | **48.5%** (4-lang) | 1 |
| OCR + blur | 72.8% | 74.7% | — | — | **73.8%** | **79.05%** | **82.70%** | **80.88%** (EN+ZH) | 3 |
| Defense-Prefix | **73.8%** | 44.5% | — | — | **59.2%** | **81.65%** | **68.15%** | **74.90%** (EN+ZH) | 2 |
| SamplingTAR hybrid | 67.3% | — | — | — | 67.3% | 72.45% | — | 72.45% (EN) | 3 |
| Dyslexify hybrid | 66.9% | — | — | — | 66.9% | 72.35% | — | 72.35% (EN) | 3 |
| **Ours (EN∩ZH black)** | **72.9%** | **76.5%** | — | — | **74.7%** | **79.35%** | **83.95%** | **81.65%** bilingual | 4 |
| **Ours (EN∩KO black)** | 65.6% | — | **73.1%** | — | **69.4%** | 75.45% | — | **78.35%** bilingual | 4 |
| **Ours (EN∩JA black)** | 68.9% | — | — | **82.8%** | **75.9%** | 77.40% | — | **82.53%** bilingual | 4 |

**How baseline MIXED is computed**

- OCR EN MIXED = \(0.5 \times 72.8\% + 0.5 \times 85.3\% = \mathbf{79.05\%}\); ZH MIXED = \(0.5 \times 74.7\% + 0.5 \times 90.7\% = \mathbf{82.70\%}\); EN+ZH Scope MIXED = **80.88%**.
- DP EN MIXED = \(0.5 \times 73.8\% + 0.5 \times 89.5\% = \mathbf{81.65\%}\); ZH MIXED = \(0.5 \times 44.5\% + 0.5 \times 91.8\% = \mathbf{68.15\%}\); EN+ZH Scope MIXED = **74.90%**.
- Clean Δ: OCR EN/ZH −0.6 / −0.7 pp; DP EN/ZH +0.5 / +0.4 pp; OCR sticker hit 90.3%.

**Reading the table**

- Raw CLIP collapses under dual-box `multi` (mean atk **7.1%**; Scope MIXED **48.5%**) — the undefended floor for all comparisons.
- OCR ZH MIXED (**82.70%**) is competitive with ours ZH (**83.95%**); bilingual Scope MIXED is nearly tied (**80.88%** vs ours **81.65%**).
- Defense-Prefix wins **EN MIXED** (**81.65%** vs our EN **79.35%**) but **collapses on ZH MIXED** (**68.15%**), dragging EN+ZH Scope MIXED to **74.90%**.
- Ours leads once partner languages are in scope (partner mean bilingual MIXED **80.84%** vs DP EN+ZH **74.90%**).

---

## Table 2 — Ours detail (EN∩L pairings)

| Pairing | EN atk | L atk | Mean atk | EN MIXED | L MIXED | Bilingual MIXED | Gate fire (atk / clean) |
|---------|-------:|------:|---------:|---------:|--------:|----------------:|-------------------------:|
| EN∩ZH | **72.9%** | **76.5%** | **74.7%** | **79.35%** | **83.95%** | **81.65%** | 99.8% / 0.4% |
| EN∩KO | 65.6% | 73.1% | 69.4% | 75.45% | 81.25% | **78.35%** | 99.4% / 2.5% |
| EN∩JA | 68.9% | 82.8% | 75.9% | 77.40% | 87.65% | **82.53%** | 99.8% / 0.3% |
| **Mean (3 partners)** | — | — | **73.3%** | — | — | **80.84%** | — |

EN clean under gated black (ZH pairing): **85.8%** (Clean Δ **−0.1 pp**). Partner L clean floors ≈ ZH **91.4%**, KO **89.4–89.5%**, JA **92.5%**.

---

## Table 2b — Four-way occlusion (EN ∩ ZH ∩ KO ∩ JA)

True 4-language Attn-last intersection → gated `cc_bbox_black`. Attack = protocol EN+ZH `multi`; score all four models. Gate uses EN vs mean(ZH,KO,JA) features; masks are 4-way. Source: [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json).

| Arm | EN atk | ZH atk | KO atk | JA atk | Mean atk | EN MIXED | ZH MIXED | KO MIXED | JA MIXED | Mean MIXED | Clean Δ (EN/ZH/KO/JA) |
|-----|-------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|---------:|-----------:|----------------------:|
| never | 4.5% | 6.4% | 11.6% | 6.0% | 7.1% | 45.2% | 48.9% | 50.6% | 49.3% | 48.5% | — |
| always | 72.1% | 76.5% | 76.6% | 84.6% | **77.5%** | 68.15% | 72.15% | 72.85% | 78.90% | 73.0% | −21.7 / −23.6 / −20.5 / −19.3 pp |
| **gated** | **72.0%** | **76.4%** | **76.5%** | **84.5%** | **77.4%** | **78.65%** | **83.75%** | **82.90%** | **88.30%** | **83.40%** | −0.6 / −0.3 / −0.3 / −0.4 pp |

Gate fire: atk **99.8%** / clean **2.1%**. Cost ≈ 8 image fwds for saliency (4 models × clean/atk path) vs 4 for pairwise EN∩L.

**Reading:** Gated 4-way mean MIXED (**83.40%**) exceeds pairwise partner-mean bilingual MIXED (**80.84%**). Always-on destroys clean accuracy (~−20 pp); the gate is required. Production pairwise EN∩L remains the default (lower cost); 4-way is the all-language occlusion variant.

---

## Table 3 — Method averages (supporting)

One row per method. Scope differs — always read with Table 1.

| Method | Scope | Atk acc (avg) | Clean Δ (avg) | MIXED2000 (avg) | Cost |
|--------|-------|--------------:|--------------:|----------------:|-----:|
| **Gated `cc_bbox_black` (ours)** | EN∩ZH / EN∩KO / EN∩JA | **73.3%** | **≈ 0.0 / −0.2 / 0.0 pp** | **80.84%** | 4 |
| OCR + blur | EN+ZH | 73.8% | −0.7 pp | 80.88% | 3 |
| Defense-Prefix | EN+ZH | 59.2% | +0.5 pp | 74.90% | 2 |
| SamplingTAR hybrid | EN only | 67.3% | −8.3 pp | 72.45% | 3 |
| Dyslexify hybrid | EN only | 66.9% | −8.1 pp | 72.35% | 3 |

---

## Quote set (paper / meeting)

| Claim | Number |
|-------|-------:|
| Raw CLIP mean atk / Scope MIXED (4-lang) | **7.1% / 48.5%** |
| Ours avg bilingual MIXED (ZH/KO/JA black) | **80.84%** |
| Ours EN MIXED (gated black) | **79.35%** |
| Ours ZH MIXED (gated black) | **83.95%** |
| Ours EN atk / clean (gated black) | **72.9% / 85.8%** |
| Partner bilingual MIXED black (ZH / KO / JA) | **81.65 / 78.35 / 82.53%** |
| OCR EN / ZH MIXED | **79.05% / 82.70%** |
| OCR EN+ZH MIXED | **80.88%** |
| DP EN / ZH MIXED | **81.65% / 68.15%** |
| DP EN+ZH MIXED / mean atk | **74.90% / 59.2%** |
| Four-way gated mean MIXED / mean atk | **83.40% / 77.4%** |
| Dyslexify / SamplingTAR hybrid EN MIXED | **72.35% / 72.45%** |

---

## Notes / honesty

1. Table 3 averages **are not identical scopes** — Dyslexify/SamplingTAR are EN-only; OCR/DP are EN+ZH; ours averages three EN∩L pairings. Prefer Table 1 for language coverage.
2. Do **not** claim ours beats DP on EN MIXED2000 (**79.35% < 81.65%**, −2.30 pp).
3. Do claim ours wins when ZH is included against DP (**ZH MIXED 83.95% vs 68.15%**; EN+ZH Scope **81.65%** bilingual vs DP **74.90%**).
4. Heads-only Dyslexify / SamplingTAR (20.0% / 11.6% EN) are negatives; hybrids are the reportable peer numbers above.
5. KO bilingual black is **−0.15 pp** vs blur; protocol still freezes **black for all langs** (ablation only).
