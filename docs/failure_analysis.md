# Failure Analysis — Gated `cc_bbox_black`

**Defense (current):** heatmap-shape detector → (if fire) Attn-last EN ∩ L → CC → bbox snap → **solid black fill** → reclassify  
**Policy:** **gated only** — always-on / never-defend are retired for reporting  
**Protocol:** frozen dual-box CIFAR-10 (`attack_pos`), n=1000, thr ≥ 0.95, attack = `multi`  
**Last updated:** 2026-07-25

**Primary EN numbers** (`en_neglect_vs_blur` / `en_occlusion_beat_dp`, gated black):

| Metric | Value |
| --- | ---: |
| EN attacked acc | **72.9%** |
| EN clean acc | **85.8%** |
| EN Clean Δ | **−0.1pp** |
| EN MIXED2000 | **79.35%** |
| EN ASR (after defense) | 3.7% |
| Gate fire (atk / clean) | 99.8% / 0.4% |

> The “~79%” EN result is **MIXED2000 = 79.35%** for gated Attn `cc_bbox` + black (not attacked-only acc). Attacked-only EN is **72.9%**.

---

## Bottom line

- **Strong:** hard dual-box attacks; near-perfect detect-when-to-occlude; Clean Δ ≈ 0 under the gate; black fill beats blur/mean/neglect on EN.
- **Weak:** residual gap from defended-atk to clean (~13pp EN); still below Defense-Prefix’s EN MIXED2000 bar (**81.65%**); Pure E / Pure L under the gate not fully characterized; partner bilingual black≈blur (KO prefers blur by 0.15pp).
- Quote EN headline as **79.35% MIXED2000** (gated black), not older blur / always-on figures.

| Headline | Number |
| --- | ---: |
| EN MIXED2000 (gated black) | **79.35%** |
| EN atk / clean (gated black) | **72.9% / 85.8%** |
| vs gated blur (EN MIXED2000) | **+1.45pp** |
| vs DP EN MIXED2000 bar | **−2.30pp** |
| Residual EN (clean − atk) | **~13pp** |

---

## Current stack (what “ours” means)

1. **Detect** — linear SVM on 26 Attn-last heatmap scalars (clean vs attacked).
2. **If fire** — build EN ∩ L Attn-last mask → thr ≥ 0.95 → dilate → top-2 CC → bbox snap.
3. **Occlude** — solid **black** inside the mask (not Gaussian blur).
4. **Classify** — EN (and partner L when reporting bilingual means).

Fill ranking on EN MIXED2000 (gated, same masks):

| Fill | EN atk | EN clean | EN MIXED2000 |
| --- | ---: | ---: | ---: |
| neglect (ViT token zero) | 60.5% | 85.9% | 73.20% |
| blur | 69.9% | 85.9% | 77.90% |
| mean | 70.4% | 85.9% | 78.15% |
| **black** | **72.9%** | **85.8%** | **79.35%** |

- Prefer **black**. Blur is a weaker ablation.
- OCR∪Attn black can edge to **79.70%** MIXED2000; production localization stays Attn heatmap (no OCR).

---

## Where it does well

### 1. Hard dual-box attacks (`multi`)

- Undefended EN under multi collapses to **4.5%**.
- Gated black recovers EN to **72.9%** (+68.4pp) with ASR **3.7%**.
- Clean images stay near vanilla (**85.8%**, Clean Δ **−0.1pp**) because the gate almost never fires on clean (0.4%).

**Partner gated means** (Phase C detector + `cc_bbox`; fill in that log was blur — localization/gate story; EN fill now black):

| Partner | Gated atk mean | Gated clean mean | MIXED2000 | Clean Δ |
| --- | ---: | ---: | ---: | ---: |
| ZH | 73.9% | 88.7% | **81.3%** | 0.0pp |
| KO | 69.5% | 87.6% | **78.5%** | −0.2pp |
| JA | 75.7% | 89.2% | **82.5%** | 0.0pp |

**Why this works**

- Dual-box / English stickers make Attn-last maps spiky; EN ∩ L localizes stickers.
- The detector turns occlusion on only when heatmap shape says “attack.”
- Black wipes glyph evidence harder than soft blur.

### 2. Gate reliability (`multi`)

| Partner | Test AUC | Attack recall | Clean false-occlude (n=1000) |
| --- | ---: | ---: | ---: |
| ZH | 1.000 | 100% | 0.4% |
| JA | 1.000 | 100% | 0.3% |
| KO | 0.999 | 100% | **2.5%** |

- ZH/JA: essentially free Clean Δ under the gate.
- KO: slightly higher clean fire rate to keep 100% attack recall — still Clean Δ **−0.2pp**.

### 3. Beats weaker fills and weak baselines (EN)

- Gated black **79.35%** MIXED2000 ≫ gated neglect (**73.20%**) and gated blur (**77.90%**).
- Far above head-ablation baselines (Dyslexify / SamplingTAR heads-only ~12–20% EN atk).
- Competitive with, but not above, Defense-Prefix’s EN MIXED2000 **81.65%**.

---

## Where it suffers

### 1. Residual gap under attack (~13pp EN)

- Gated black: EN atk **72.9%** vs clean **~85.9%** → about **−13pp** left on the table.
- Partner gated means still sit ~10–18pp below their clean means.
- Better fill (black vs blur) recovered **+3.0pp** EN atk; it did **not** close the residual.
- Oracle GT boxes + black only reach **74.6%** EN atk — localization/coverage, not just fill, still limit recovery.

### 2. Below the Defense-Prefix EN MIXED2000 bar

| System | EN atk | EN clean | EN MIXED2000 |
| --- | ---: | ---: | ---: |
| **Gated `cc_bbox_black` (ours)** | **72.9%** | **85.8%** | **79.35%** |
| Defense-Prefix (EN bar) | 73.8% | 89.5% | **81.65%** |
| Gap | — | — | **−2.30pp** |

- To clear 81.65% at clean ≈ 85.9% needs EN atk ≳ **77.4%**.
- Best occlusion-only atk seen: **73.6%** (OCR∪Attn black) / **72.9%** (Attn black) — still short.
- Optional stronger hybrid (DP text + spatial patch-zero) reaches **86.65%** EN MIXED2000; that is a different system than pure gated black.

### 3. English half still harder than the partner

- Under gated ZH multi (blur-fill Phase C log): EN atk **69.9%** vs ZH atk **77.9%**.
- With black, EN rises to **72.9%**, but Latin stickers remain the dominant threat across models.
- Mean (EN+L) scores look better than EN-only because the partner recovers more.

### 4. Detector soft spot on KO

- KO clean false-occlude **2.5%** (test **10/150**) vs ZH/JA ~0.3–0.4% (test 0/150).
- Still acceptable for Clean Δ ≈ 0, but KO is the noisiest gate.

### 5. Open / under-measured under the gate

- **Pure E / Pure L gating** not yet the reporting standard — detector tuned and quoted on **`multi`**.
- Risk: if the gate fires on weak native-only (Pure L) stickers for KO/JA, black occlusion can erase object evidence and hurt accuracy (historically −5 to −6pp when occlusion was forced). Needs a gated Pure-L measurement.
- **Partner fill ranking (done):** gated neglect/blur/mean/black for ZH/KO/JA — bilingual MIXED black **81.65% / 78.35% / 82.53%**; winners ZH/JA=black, KO=blur (−0.15pp). See [`partner_fill_ablation/results/`](../lib/notebooks/partner_fill_ablation/results/).
- **Adaptive placement / higher-res scenes** not tested — frozen dual-box CIFAR-10 only.

---

## Failure taxonomy (gated `cc_bbox_black`)

| Mode | Symptom | Severity | Status |
| --- | --- | --- | --- |
| Residual attack gap | EN atk 72.9% vs clean ~85.9% (~−13pp) | High | Open |
| Below DP EN MIXED2000 | 79.35% vs 81.65% (−2.30pp) | Medium | Occlusion-only ceiling |
| EN harder than partner | EN lags L after defense | Medium | Open |
| KO gate false positives | 2.5% clean fire | Low | Acceptable for recall |
| Pure E / Pure L under gate | Not fully reported | Unknown | Future measurement |
| Adaptive / out-of-protocol attacks | Frozen dual-box only | Unknown | Paper limitation |

---

## Metric definitions

```text
MIXED2000 = 0.5 * attacked_acc + 0.5 * clean_policy_acc
```

- Same 1000 CIFAR indices, once clean + once multi-attacked (equal weight).
- For EN-only rows: both halves are English CLIP.
- For partner rows: each half is mean(EN, L).
- Clean Δ = clean_policy_acc − vanilla clean (gated ≈ 0 when the detector stays off).

---

## What to quote / what not to claim

**Quote**

- EN gated black: **72.9%** atk / **85.8%** clean / **79.35%** MIXED2000.
- Fill ranking: black > mean > blur > neglect (gated EN).
- Partner bilingual MIXED black: ZH **81.65%**, KO **78.35%**, JA **82.53%** (KO blur still +0.15pp).
- Gate Clean Δ ≈ 0 on `multi` (ZH/JA 0.0; KO −0.2).

**Do not claim**

- Do not quote always-on or blur as the current system.
- Do not say EN “got 79% attacked accuracy” — **79.35% is MIXED2000**; atk is **72.9%**.
- Do not claim occlusion-only beats Defense-Prefix on EN MIXED2000 (−2.30pp).
- Do not claim near-clean under attack — ~13pp residual remains.
- Do not over-generalize the gate beyond `multi` until Pure E / Pure L are measured.

---

## Related paths

- EN gated black leaderboard: [`lib/notebooks/en_neglect_vs_blur/results/leaderboard.json`](../lib/notebooks/en_neglect_vs_blur/results/leaderboard.json)
- EN occlusion arms: [`lib/notebooks/en_occlusion_beat_dp/results/summary_n1000.json`](../lib/notebooks/en_occlusion_beat_dp/results/summary_n1000.json)
- Partner gated MIXED2000 (Phase-C blur history): [`lib/notebooks/attack_detector/results/mixed_2000_summary.json`](../lib/notebooks/attack_detector/results/mixed_2000_summary.json)
- Partner fill ranking + black MIXED: [`lib/notebooks/partner_fill_ablation/results/`](../lib/notebooks/partner_fill_ablation/results/); [`mixed_2000_black_summary.json`](../lib/notebooks/attack_detector/results/mixed_2000_black_summary.json)
- Detector Phase C: [`lib/notebooks/attack_detector/`](../lib/notebooks/attack_detector/)
- Protocol: [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md)
- Research diary (2026-07-25 fill / occlusion entries): [`docs/research_diary.md`](research_diary.md)
