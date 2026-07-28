# 4-lang main results table

**Protocol:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.  
**MIXED2000:** `0.5 * attacked_acc + 0.5 * clean_policy_acc` (equal weight on the same 1000 indices once clean + once attacked).  
**Bilingual MIXED2000 (partner L):** `0.5 * mean(EN,L atk) + 0.5 * mean(EN,L clean_policy)`.

Sources: [`baseline_comparison.md`](baseline_comparison.md), [`partner_fill_ablation/results/leaderboard.json`](../lib/notebooks/partner_fill_ablation/results/leaderboard.json), [`en_neglect_vs_blur/results/gated_n1000.json`](../lib/notebooks/en_neglect_vs_blur/results/gated_n1000.json), [`paper_baselines/results/mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json).

---

## Table 1 — Main (averages): ours vs 4 baselines

One row per method. **Avg score** is the fairest available mean for that method (see Scope). Prefer **MIXED2000** as the joint headline; attacked accuracy and Clean Δ are supporting columns.

| Method | Scope | Atk acc (avg) | Clean Δ (avg) | MIXED2000 (avg) | Cost |
|--------|-------|--------------:|--------------:|----------------:|-----:|
| **Gated `cc_bbox_black` (ours)** | EN∩ZH / EN∩KO / EN∩JA | **73.3%** | **≈ 0.0 / −0.2 / 0.0 pp** | **80.84%** | 4 |
| OCR + blur | EN+ZH | 73.8% | −0.7 pp | 80.88% | 3 |
| Defense-Prefix | EN+ZH | 59.2% | +0.5 pp | 74.90% | 2 |
| SamplingTAR hybrid | EN only | 67.3% | −8.3 pp | 72.45% | 3 |
| Dyslexify hybrid | EN only | 66.9% | −8.1 pp | 72.35% | 3 |

**How ours averages are computed**

- Partner bilingual MIXED (black): ZH **81.65%**, KO **78.35%**, JA **82.53%** → mean **80.84%**.
- Partner mean attacked acc (black): ZH **74.7%**, KO **69.4%**, JA **75.9%** → mean **73.3%**.
- Clean Δ: Phase-C gated figures for ZH / KO / JA (`multi`).

**How baseline averages are computed**

- OCR + blur: mean of EN+ZH defended acc (**72.8 / 74.7 → 73.8%**); MIXED2000 from EN+ZH always policy (**80.88%**).
- Defense-Prefix: mean of EN+ZH defended acc (**73.8 / 44.5 → 59.2%**); MIXED2000 EN+ZH (**74.90%**).
- Dyslexify / SamplingTAR hybrids: EN-only MIXED2000 / atk / Clean Δ (no KO/JA ports).

**Reading the table**

- Ours leads on **cross-language** average MIXED (**80.84%**) among methods that cover partners.
- OCR is the closest spatial peer on EN+ZH MIXED (**80.88%** ≈ ours ZH bilingual **81.65%**), but has no KO/JA row yet.
- Defense-Prefix wins **EN-only** MIXED (**81.65%** vs our EN **79.35%**) but collapses once ZH is averaged in (**74.90%** / mean atk **59.2%**).
- Head+blur hybrids clear ~50% but trail gated black on EN MIXED (~72.4% vs **79.35%**).

---

## Table 2 — Language detail (EN / ZH / KO / JA)

Per-language (or per EN∩L pairing) numbers. Use this when Table 1 is too coarse.

### 2a. Ours — gated `cc_bbox_black`

| Pairing | EN atk | L atk | Mean atk | EN MIXED | L MIXED | Bilingual MIXED | Gate fire (atk / clean) |
|---------|-------:|------:|---------:|---------:|--------:|----------------:|-------------------------:|
| EN∩ZH | **72.9%** | **76.5%** | **74.7%** | **79.35%** | **83.95%** | **81.65%** | 99.8% / 0.4% |
| EN∩KO | 65.6% | 73.1% | 69.4% | 75.45% | 81.25% | **78.35%** | 99.4% / 2.5% |
| EN∩JA | 68.9% | 82.8% | 75.9% | 77.40% | 87.65% | **82.53%** | 99.8% / 0.3% |
| **Mean (3 partners)** | — | — | **73.3%** | — | — | **80.84%** | — |

EN clean under gated black (ZH pairing): **85.8%** (Clean Δ **−0.1 pp**). Partner L clean floors ≈ ZH **91.4%**, KO **89.4–89.5%**, JA **92.5%**.

### 2b. Baselines — language coverage

| Method | EN atk | ZH atk | KO | JA | Mean (langs present) | EN MIXED | Notes |
|--------|-------:|-------:|:--:|:--:|---------------------:|---------:|-------|
| OCR + blur | 72.8% | 74.7% | — | — | **73.8%** | ~80.9% (EN+ZH) | Clean Δ EN/ZH −0.6 / −0.7 pp; sticker hit 90.3% |
| Defense-Prefix | **73.8%** | 44.5% | — | — | **59.2%** | **81.65%** EN | ZH ASR 52.5%; Clean Δ EN/ZH +0.5 / +0.4 pp |
| SamplingTAR hybrid | 67.3% | — | — | — | 67.3% | 72.45% | Clean Δ −8.3 pp |
| Dyslexify hybrid | 66.9% | — | — | — | 66.9% | 72.35% | Clean Δ −8.1 pp |
| **Ours (EN∩ZH black)** | **72.9%** | **76.5%** | — | — | **74.7%** | **79.35%** | Bilingual MIXED **81.65%** |
| **Ours (EN∩KO black)** | 65.6% | — | **73.1%** | — | **69.4%** | 75.45% | Bilingual MIXED **78.35%** |
| **Ours (EN∩JA black)** | 68.9% | — | — | **82.8%** | **75.9%** | 77.40% | Bilingual MIXED **82.53%** |

Em dashes (—) = not evaluated for that language under the current baseline ports.

---

## Quote set (paper / meeting)

| Claim | Number |
|-------|-------:|
| Ours avg bilingual MIXED (ZH/KO/JA black) | **80.84%** |
| Ours EN MIXED (gated black) | **79.35%** |
| Ours EN atk / clean (gated black) | **72.9% / 85.8%** |
| Partner bilingual MIXED black (ZH / KO / JA) | **81.65 / 78.35 / 82.53%** |
| OCR EN+ZH MIXED | **80.88%** |
| DP EN MIXED / EN+ZH mean atk | **81.65% / 59.2%** |
| Dyslexify / SamplingTAR hybrid EN MIXED | **72.35% / 72.45%** |

---

## Notes / honesty

1. Table 1 averages **are not identical scopes** — Dyslexify/SamplingTAR are EN-only; OCR/DP are EN+ZH; ours averages three EN∩L pairings. Always show Scope.
2. Do **not** claim ours beats DP on EN MIXED2000 (**79.35% < 81.65%**, −2.30 pp).
3. Do claim ours wins when languages are averaged with a spatial defense (DP EN+ZH mean **59.2%** / MIXED **74.90%** vs ours partner mean MIXED **80.84%**).
4. Heads-only Dyslexify / SamplingTAR (20.0% / 11.6% EN) are negatives; hybrids are the reportable peer numbers above.
5. KO bilingual black is **−0.15 pp** vs blur; protocol still freezes **black for all langs** (ablation only).
