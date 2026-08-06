# 4-lang main results table

> One-page index of all tables: [`tables_index.md`](tables_index.md).  
> Paper export (scoped Table 1 / cleaned Table 4): [`paper_tables_final.md`](paper_tables_final.md).

**Protocol:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**MIXED2000:** `0.5 * attacked_acc + 0.5 * clean_policy_acc` (equal weight on the same 1000 indices once clean + once attacked).  
**Bilingual MIXED2000 (partner L):** `0.5 * mean(EN,L atk) + 0.5 * mean(EN,L clean_policy)`.  
**Per-lang MIXED (baselines):** `0.5 * lang_atk + 0.5 * lang_clean_policy` from [`mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json) `per_lang`.

Sources: [`baseline_comparison.md`](baseline_comparison.md), [`partner_fill_ablation/results/leaderboard.json`](../lib/notebooks/partner_fill_ablation/results/leaderboard.json), [`en_neglect_vs_blur/results/gated_n1000.json`](../lib/notebooks/en_neglect_vs_blur/results/gated_n1000.json), [`paper_baselines/results/mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json), [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json) (`arms.never` = Raw CLIP).

---

## Table 1 — Main: language coverage (baselines + ours)

Primary comparison (scoped layout matches [`paper_tables_final.md`](paper_tables_final.md)). Em dashes (—) = not evaluated / intentionally skipped.

| Method | EN acc / mixed acc | ZH acc / mixed acc | KO acc / mixed acc | JA acc / mixed acc | Mean acc (attacked) | Mean mixed acc | Clean Δ EN | Cost |
|--------|-------------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------:|-----------:|-----:|
| No defense (Raw CLIP) | 4.5 / 45.2 | 6.4 / 48.9 | 11.6 / 50.6 | 6.0 / 49.3 | **7.1%** | **48.5%** | 0.0 pp | 1 |
| 4×4 grid occlusion (naive) | 47.8 / 65.3 | 49.2 / 66.8 | 52.8 / 65.8 | 55.1 / 70.8 | 51.2% | 67.2% | −3.2 pp | 62 |
| OCR + blur (spatial upper bound) | 72.8 / 79.1 | 74.7 / 82.7 | 80.2 / 84.5 | 87.6 / 89.7 | **78.8%** | **84.0%** | −0.6 pp | 3 |
| Defense-Prefix | 73.8 / 81.7 | 81.4 / 86.2 | 69.1 / 79.4 | 84.8 / 89.2 | 77.3% | 84.1% | +0.5 pp | 2 |
| SamplingTAR + blur | 67.3 / 72.5 | 68.9 / 79.3 | 59.0 / 62.9 | 51.5 / 72.1 | 61.7% | 71.7% | −8.3 pp | 3 |
| Dyslexify + blur | 66.9 / 72.4 | 68.2 / 79.6 | 62.9 / 67.1 | 43.2 / 67.0 | 60.3% | 71.5% | −8.1 pp | 3 |
| **Ours (EN∩ZH)** | **72.9 / 79.4** | **76.5 / 84.0** | — | — | **74.7%** | **81.7%** | **−0.1 pp** | 4 |
| **Ours (EN∩KO)** | **65.6 / 75.5** | — | **73.1 / 81.3** | — | **69.4%** | **78.4%** | **−0.6 pp** | 4 |
| **Ours (EN∩JA)** | **68.9 / 77.4** | — | — | **82.8 / 87.7** | **75.9%** | **82.5%** | **0.0 pp** | 4 |

**How baseline MIXED is computed**

- OCR EN MIXED = \(0.5 \times 72.8\% + 0.5 \times 85.3\% = \mathbf{79.05\%}\); ZH = **82.70%**; KO = **84.50%**; JA = **89.65%**; 4-lang mean MIXED = **83.97%**.
- DP EN MIXED = \(0.5 \times 73.8\% + 0.5 \times 89.5\% = \mathbf{81.65\%}\); ZH = **86.20%**; KO = **79.40%**; JA = **89.20%**; 4-lang mean MIXED = **84.11%**.
- Grid / SamplingTAR / Dyslexify hybrids: per-lang MIXED from n=1000 finals in [`mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json).
- Clean Δ EN: OCR **−0.6 pp**; DP **+0.5 pp**; grid **−3.2 pp**; OCR sticker hit 90.3%. ZH DP retuned with EN+ZH multi train (20 ep).

**Reading the table**

- No defense (Raw CLIP) collapses under dual-box `multi` (mean acc (attacked) **7.1%**; mean mixed acc **48.5%**) — the no-defense floor for all comparisons.
- **Naive floor:** 4×4 grid reaches **51.2%** mean acc (attacked) / **67.2%** mean mixed at cost **62**.
- **Spatial upper bound:** OCR+blur 4-lang mean mixed acc (**84.0%**); ours bilingual EN∩ZH (**81.7%**) is close without OCR. OCR ZH mixed acc (**82.7%**) is competitive with ours ZH (**84.0%**).
- Defense-Prefix reaches ZH acc **81.4%** / mixed **86.2%** and 4-lang mean mixed **84.1%**; EN mixed acc still **81.7%** vs our EN **79.4%**.
- Ours remains competitive on bilingual scope with near-zero Clean Δ EN on ZH/JA pairings.

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
| 4×4 grid occlusion (naive) | EN+ZH+KO+JA | 51.2% | −3.2 pp EN | 67.2% | 62 |
| **Gated `cc_bbox_black` (ours)** | EN∩ZH / EN∩KO / EN∩JA | **73.3%** | **≈ 0.0 / −0.2 / 0.0 pp** | **80.84%** | 4 |
| OCR + blur (spatial upper bound) | EN+ZH+KO+JA | 78.8% | −0.7 pp | 84.0% | 3 |
| Defense-Prefix | EN+ZH+KO+JA | 77.3% | +0.5 pp EN | 84.1% | 2 |
| SamplingTAR hybrid | EN+ZH+KO+JA | 61.7% | −8.3 pp EN | 71.7% | 3 |
| Dyslexify hybrid | EN+ZH+KO+JA | 60.3% | −8.1 pp EN | 71.5% | 3 |

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
| OCR EN / ZH / KO / JA MIXED | **79.05% / 82.70% / 84.50% / 89.65%** |
| OCR 4-lang mean MIXED / mean atk | **83.97% / 78.83%** |
| DP EN / ZH / KO / JA MIXED | **81.65% / 86.20% / 79.40% / 89.20%** |
| DP 4-lang mean MIXED / mean atk | **84.11% / 77.28%** |
| Four-way gated mean MIXED / mean atk | **83.40% / 77.4%** |
| Dyslexify / SamplingTAR hybrid EN MIXED | **72.35% / 72.45%** |
| Dyslexify / SamplingTAR 4-lang mean atk | **60.3% / 61.7%** |
| Grid 4-lang mean atk / mean MIXED | **51.2% / 67.2%** |

---

## Notes / honesty

1. Table 3 averages **are not identical scopes** — grid / OCR / DP / head hybrids are 4-lang n=1000 finals; ours averages three EN∩L pairings. Prefer Table 1 for language coverage.
2. Do **not** claim ours beats DP on EN MIXED2000 (**79.35% < 81.65%**, −2.30 pp).
3. Do claim ours wins bilingual clean-cost tradeoff vs always-on and remains competitive with OCR without an external recognizer.
4. Heads-only Dyslexify / SamplingTAR (20.0% / 11.6% EN) are negatives; hybrids are the reportable peer numbers above.
5. KO bilingual black is **−0.15 pp** vs blur; protocol still freezes **black for all langs** (ablation only).
6. Defense-Prefix ZH retuned 2026-08-03 (EN+ZH multi train, 20 ep) → ZH atk **81.4%**. Hybrid ZH/KO/JA finals: SamplingTAR **68.9/59.0/51.5**, Dyslexify **68.2/62.9/43.2**.
