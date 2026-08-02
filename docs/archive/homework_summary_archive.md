# Archive — Homework Summary (through July 27, 2026)

**Status:** Archived. Current briefing: [`homework_summary.md`](../homework_summary.md) (**August 1**).  
**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.

This file holds all superseded homework briefings in one place:

1. **Part E** — July 27 (animal/hybrid/text occlusion; ablation appendix; font + boxes geometry)
2. **Part D** — July 25 (gated `cc_bbox_black`, fill ranking, DP chase, partner fills, hybrids)
3. **Part C** — July 20–24 (frozen protocol, detector gate, baselines, MIXED2000)
4. **Part A** — July 16–19 defense sprint (`cc_bbox_blur`, 4-lang, KO/JA clean Δ)
5. **Part B** — July 5 Assignment 1–2 (JA model swap, 4×4 attack matrix)

---

# Part E — Ablation checklist + attack geometry (July 27, 2026)

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was done **that day**.

> Current briefing: [`homework_summary.md`](../homework_summary.md).  
> Ablation appendix: [`ablation_study.md`](../ablation_study.md).  
> Main results tables: [`4_lang_table.md`](../4_lang_table.md).  
> Paper tables index: [`tables_index.md`](../tables_index.md).  
> Paper outline: [`paper_draft.md`](../paper_draft.md).  
> Protocol: [`PROTOCOL.md`](../../lib/notebooks/PROTOCOL.md).  
> Full diary for that day: [`research_diary.md`](../research_diary.md) ([animal occlusion](../research_diary.md#L2889), [ablation + geometry](../research_diary.md#L2936)).

**Shared protocol unless noted:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), 224×224, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.

---

## One-paragraph overview

That day closed the **ablation checklist**: measured production **black occlusion** on animal / mixed / text stickers; consolidated method + attack ablations into [`ablation_study.md`](../ablation_study.md); ran the two open geometry sweeps (**font 12/24/40**, **boxes 1/2/3**); and synced [`paper_draft.md`](../paper_draft.md) to [`4_lang_table.md`](../4_lang_table.md) + the new appendix. Headline: gated black stays **typographic** (text EN **72.9%**); hybrid partial (**43.5%**); animal-only weak (EN∩ZH **20.6%**, EN-only **32.9%**). Production threat model stays **font 24 / 2 boxes** — font 40 and 3 boxes stress the recipe.

---

## Checklist done that day

| Item | Kind | Status | Where |
| --- | --- | --- | --- |
| Animal / hybrid / text black occlusion | attack + method | **Done** | [`ablation_study.md`](../ablation_study.md) §A.3 / §B.2; [diary](../research_diary.md#L2889) |
| Fill type (document only) | method | **Done** | [`ablation_study.md`](../ablation_study.md) §A.1 |
| Gate on vs always-on (document only) | method | **Done** | [`ablation_study.md`](../ablation_study.md) §A.2 |
| Single-lang occlusion (document only) | method | **Done** | [`ablation_study.md`](../ablation_study.md) §A.3 |
| Four-lang main results (document only) | method | **Done** | [`4_lang_table.md`](../4_lang_table.md); [`ablation_study.md`](../ablation_study.md) §A.4 |
| Text vs white / sticker / hybrid | attack | **Done** | [`ablation_study.md`](../ablation_study.md) §B.1–B.2 |
| Font size 12 / 24 / 40 | attack | **Done** | [`ablation_study.md`](../ablation_study.md) §B.3; [diary](../research_diary.md#L2936) |
| Number of boxes 1 / 2 / 3 | attack | **Done** | [`ablation_study.md`](../ablation_study.md) §B.4; [diary](../research_diary.md#L2936) |
| Paper sync (main + ablations + geometry) | writing | **Done** | [`paper_draft.md`](../paper_draft.md) |

---

## What was done

### 1. Animal-sticker black occlusion recovery

**Question:** Does gated `cc_bbox_black` repair animal / mixed / text? Does EN-only close the bilingual miss on animals?

**Gated black (n=1000)**

| Mode | Arm | EN acc | EN ASR | Clean Δ EN |
| --- | --- | ---: | ---: | ---: |
| all_text | EN∩ZH | **72.9%** | 3.7% | −0.1 pp |
| all_text | EN-only | 68.0% | 3.6% | −0.1 pp |
| mixed | EN∩ZH | **43.5%** | 27.3% | −0.1 pp |
| mixed | EN-only | 40.2% | 27.4% | −0.1 pp |
| all_sticker | EN∩ZH | 20.6% | 54.2% | −1.8 pp |
| all_sticker | EN-only | **32.9%** | **28.5%** | −6.1 pp |

**Verdict:** Production EN∩ZH black is **typographic**. EN-only helps animals more but costs clean accuracy and still trails oracle GT black (animal EN **56.9%**) — localization is the bottleneck, not fill.

- Doc: [`ablation_study.md` §A.3 / §B.2](../ablation_study.md)  
- Diary: [`research_diary.md` § 2026-07-27 — Animal-sticker black occlusion](../research_diary.md#L2889)  
- Code / JSON / figure: [`run_occlusion.py`](../../lib/notebooks/animal_sticker_ablation/run_occlusion.py), [`occlusion_n1000.json`](../../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json), [`gallery_occlusion.png`](../../lib/notebooks/animal_sticker_ablation/figures/gallery_occlusion.png)

---

### 2. Ablation study write-up (method + attack)

New appendix source [`ablation_study.md`](../ablation_study.md) consolidates already-finished numbers (no re-run) plus that day’s geometry:

| Section | Ablation |
| --- | --- |
| A.1 | Fill type — black > mean > blur > neglect |
| A.2 | Gate on/off — KO/JA Clean Δ −11pp → ~0 |
| A.3 | Single-lang / EN-only vs EN∩L |
| A.4 | Four-lang transfer → [`4_lang_table.md`](../4_lang_table.md) |
| B.1–B.2 | Glyphs vs white pads; sticker / text / hybrid defense |
| B.3–B.4 | Font size; number of boxes |

Diary inventory: [`research_diary.md` § 2026-07-27 — Ablation study write-up + attack geometry](../research_diary.md#L2936).

---

### 3. Font size (attack geometry)

Dual-box, gated black, font ∈ {12, **24**, 40}.

| Font | never EN ASR | gated EN | gated ZH | Clean Δ EN |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 89.9% | **76.0%** | 83.9% | +0.0 pp |
| **24** | **95.3%** | **72.9%** | **76.5%** | −0.1 pp |
| 40 | 96.9% | 58.3% | 56.3% | −0.4 pp |

**Verdict:** Font **24** stays production. Font 40 drops gated EN **−14.6 pp**.

- Doc: [`ablation_study.md` §B.3](../ablation_study.md)  
- Results: [`font_n1000.json`](../../lib/notebooks/attack_geometry_ablation/results/font_n1000.json)

---

### 4. Number of boxes (attack geometry)

Font 24, boxes ∈ {1, **2**, 3}; **`top_k = num_boxes`** (capacity matched to threat).

| Boxes | never EN ASR | gated EN | gated ZH | Clean Δ EN |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 94.1% | **78.2%** | 82.9% | −0.1 pp |
| **2** | **95.3%** | **72.9%** | **76.5%** | −0.1 pp |
| 3 | 97.5% | **45.9%** | **56.1%** | −0.8 pp |

**Verdict:** Dual-box stays production. With matched `top_k=3`, three boxes recover gated EN to **45.9%** (was **8.5%** under production `top_k=2`).

- Doc: [`ablation_study.md` §B.4](../ablation_study.md)  
- Results: [`boxes_n1000.json`](../../lib/notebooks/attack_geometry_ablation/results/boxes_n1000.json)  
- Code: [`attack_geometry_ablation/run_ablation.py`](../../lib/notebooks/attack_geometry_ablation/run_ablation.py)

---

### 5. Paper draft sync

[`paper_draft.md`](../paper_draft.md) then pulled:

- Main avg + lang detail from [`4_lang_table.md`](../4_lang_table.md) (§3.5)
- Animal / hybrid / text occlusion (§3.2.2)
- Font + boxes geometry (§3.11); threat model states **font 24**
- Appendix pointer → [`ablation_study.md`](../ablation_study.md)

---

## Numbers to quote from that day

| Claim | Number |
| --- | --- |
| Gated text (all_text) EN / ZH | **72.9% / 76.5%** |
| Gated mixed EN | **43.5%** |
| Gated animal EN∩ZH / EN-only | **20.6% / 32.9%** |
| Animal oracle GT black EN | **56.9%** |
| Font 12 / **24** / 40 gated EN | **76.0% / 72.9% / 58.3%** |
| Boxes 1 / **2** / 3 gated EN | **78.2% / 72.9% / 45.9%** |
| Ours avg bilingual MIXED (ZH/KO/JA) | **80.84%** |
| Production geometry | **font 24 / NUM_BOXES=2** |

---

## One breath (July 27)

> Ablation checklist closed. Gated black recovers **typographic** dual-box (EN **72.9%**), partially hybrid (**43.5%**), poorly animal-only (**20.6%**; EN-only **32.9%** vs oracle **56.9%**). Font **24** / **2 boxes** stay production; font 40 (−14.6 pp) is a known stress case; 3 boxes with matched `top_k=3` recover gated EN **45.9%**.

---

# Part D — Gated black production freeze (July 25, 2026)

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was done **that day**.

> Current briefing: [`homework_summary.md`](../homework_summary.md) (August 1).  
> Protocol: [`PROTOCOL.md`](../../lib/notebooks/PROTOCOL.md).  
> Baselines: [`baseline_comparison.md`](../baseline_comparison.md).  
> Current stack / failure modes: [`failure_analysis.md`](../failure_analysis.md).  
> Full diary for that day: [`research_diary.md`](../research_diary.md) (2026-07-25 entries).

**Shared protocol for all tables below unless noted:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), 224×224, attack = `multi`, thr ≥ 0.95, CUDA.  
**MIXED2000:** `0.5 * attacked_acc + 0.5 * clean_policy_acc`.

---

## One-paragraph overview

That day I locked the production defense as **gated `cc_bbox_black`**: the Attn-last detector gate is a core pipeline stage (not optional), and the occlusion fill is solid **black** (blur / mean / ViT-token neglect are ablations). On English CLIP, gated black reaches **72.9% atk / 85.8% clean / 79.35% MIXED2000** — **+1.45pp** MIXED over gated blur, but still **−2.30pp** vs Defense-Prefix’s EN MIXED **81.65%**. An occlusion-only chase (incl. OCR∪Attn) could not clear that bar (best **79.70%**). Partner ZH/KO/JA got the same gated fill table; **production fill is frozen to black for all langs** — bilingual MIXED **81.65% / 78.35% / 82.53%** (ZH/KO/JA). I also finished a fair **ZH Defense-Prefix** (ZH **44.5%**, EN+ZH mean **59.2%**), Dyslexify/SamplingTAR **hybrids** (EN **66.9% / 67.3%**), detector Phase A/B visualization for ZH/KO/JA, a failure-analysis write-up, synced the paper outline to match the new stack, and wrote up **why ZH recovers higher than EN after the same occlusion** (Latin-script / training asymmetry + floors — not a ZH-favoring mask).

---

## What was done

### 1. EN fill ablations — neglect vs blur vs mean vs black

Gated Attn `cc_bbox`, EN score. Prefer **black**; still below DP EN MIXED **81.65%**.

| Fill (gated) | EN atk | EN clean | EN MIXED2000 | vs DP |
| --- | ---: | ---: | ---: | --- |
| neglect | 60.5% | 85.9% | 73.20% | below |
| blur | 69.9% | 85.9% | 77.90% | below |
| mean | 70.4% | 85.9% | 78.15% | below |
| **black** | **72.9%** | **85.8%** | **79.35%** | −2.30pp |
| Defense-Prefix (bar) | 73.8% | 89.5% | **81.65%** | — |

Full data (oracle, always-on, DP+spatial escalate): [`research_diary.md` § 2026-07-25 — EN neglect vs Gaussian blur](../research_diary.md#L2619).

---

### 2. Occlusion-only chase of DP EN MIXED 81.65% (no DP in winner)

Best attention-only vs OCR∪Attn (logged only). Did **not** beat DP.

| Arm | EN atk | ASR | EN MIXED2000 | vs DP |
| --- | ---: | ---: | ---: | --- |
| gated `cc_bbox` black (ours) | **72.9%** | 3.7% | **79.35%** | −2.30pp |
| gated OCR∪`cc_bbox` black | 73.6% | 3.1% | **79.70%** | −1.95pp |
| Defense-Prefix (bar) | 73.8% | 16.4% | **81.65%** | — |

Full data (ensembles, pick counts, ceiling math): [`research_diary.md` § 2026-07-25 — Occlusion-only chase of DP EN MIXED2000](../research_diary.md#L2685).

---

### 3. ZH Defense-Prefix (ChineseCLIP CIFAR retrain)

Fair ZH DP token; EN+ZH mean collapses vs spatial peers.

| Lang | Defended acc | ASR | Clean Δ | MIXED2000 |
| --- | ---: | ---: | ---: | ---: |
| EN (prior) | **73.8%** | 16.4% | +0.5pp | **81.65%** |
| ZH (new) | **44.5%** | 52.5% | +0.4pp | **68.15%** |
| Mean EN+ZH | **59.2%** | — | +0.45pp | **74.90%** |

Full data (train protocol, Gate A): [`research_diary.md` § 2026-07-25 — ZH Defense-Prefix](../research_diary.md#L2510).

---

### 4. Dyslexify / SamplingTAR hybrids (heads + attn-guided blur)

Occlusion, not head ablation alone, drives recovery. Still below gated black.

| Method | Mode | EN atk | Clean Δ | MIXED2000 |
| --- | --- | ---: | ---: | ---: |
| Dyslexify | heads | 20.0% | 0.0pp | 52.95% |
| Dyslexify | **hybrid** | **66.9%** | −8.1pp | **72.35%** |
| SamplingTAR | heads | 11.6% | +0.2pp | 48.85% |
| SamplingTAR | **hybrid** | **67.3%** | −8.3pp | **72.45%** |

Full data (heads, `score_frac`, smoke ladder): [`research_diary.md` § 2026-07-25 — Dyslexify / SamplingTAR hybrids](../research_diary.md#L2739).

---

### 5. Detector Phase A/B visualization (ZH / KO / JA)

Gallery: [`docs/figures/attack_detector/gallery.html`](../figures/attack_detector/gallery.html).

| Partner | PCA NN-cent. test | Test AUC | Fire clean / atk (n=1000) |
| --- | ---: | ---: | ---: |
| ZH | 99.0% | **1.000** | 0.4% / 99.8% |
| KO | 97.3% | **0.999** | 2.5% / 99.4% |
| JA | 97.7% | **1.000** | 0.3% / 99.8% |

Full data (thresholds, CMs, figure list): [`research_diary.md` § 2026-07-25 — Phase A/B gated occlusion visualization](../research_diary.md#L2562).

---

### 6. Failure analysis + paper outline sync

- [`failure_analysis.md`](../failure_analysis.md) — locked “ours” as gated `cc_bbox_black`; quote/do-not-claim list (79.35% = MIXED, not atk-only).
- [`paper_draft.md`](../paper_draft.md) — gate = core; fill = black; lead gated Clean Δ + MIXED2000 + EN gated-black.

No single diary entry for this write-up pass; numbers live in the Jul 25 diary sections above + [`failure_analysis.md`](../failure_analysis.md).

---

### 7. Why ZH > EN after occlusion, and EN headroom vs baselines

**Verdict**

1. **ZH higher post-occlusion accuracy is mostly script/training asymmetry + higher ZH floors**, not a better ZH-specific mask.
2. **Pure EN occlusion already beats every EN baseline except Defense-Prefix.** Clearing DP’s EN MIXED2000 **81.65%** without DP text tokens looks infeasible from current ceilings; **DP + spatial** is the only path that has cleared it.

**Same defense, two models** (`cc_bbox_blur`: EN∩ZH Attn-last ∩ thr≥0.95 → top-2 CC → bbox → Gaussian blur r=12; dual-box `multi`, n=1000):

| Model | Clean | Atk (no def) | After occlusion |
| --- | ---: | ---: | ---: |
| **EN** OpenAI ViT-B/32 | 85.9% | ~4.3–4.5% | **71.6%** |
| **ZH** ChineseCLIP ViT-B/16 | 91.4% | ~6.4–7.3% | **78.2%** |

Sources: [`baseline_comparison.md`](../baseline_comparison.md); diary Latin-script asymmetry (~L894–1032) + `cc_bbox_blur` results.

**Mechanism (in order of importance)**

| Factor | Claim |
| --- | --- |
| Latin universally class-relevant | Web-trained CLIPs treat English overlays as classification signal; Hangul/CJK overlays mostly do not transfer the other way |
| ChineseCLIP less sensitive to Latin even before defense | Under EN attack ZH historically ~27–33% vs EN ~5%; Chinese-caption training → Latin overlay carries less label signal |
| Higher ZH floors amplify post-defense gap | Clean 91.4% vs 85.9%; slightly higher undefended atk → same residual sticker leakage hurts EN more |
| Not primarily patch size | EN 7×7 (B/32) vs ZH 14×14 (B/16); swapping EN→ViT-B/16 did **not** help defense (`vit16_en/`) |
| Mask is shared, not ZH-favoring | Dual-box oracle: covering the **EN** sticker matters for both; ZH-only occlusion barely helps. Same mask → ZH still +6.6pp → residual Latin remains more toxic to EN |

**Can EN occlusion beat all other EN baselines?**

| EN baseline | EN atk / MIXED2000 | Ours best occlusion-only | Status |
| --- | ---: | ---: | --- |
| Dyslexify heads / hybrid | 20.0% / 66.9% (MIXED ~72.35%) | gated OCR∪`cc_bbox` black **73.6%** atk, **79.70%** MIXED | **Already beat** |
| SamplingTAR heads / hybrid | 11.6% / 67.3% (MIXED ~72.45%) | same | **Already beat** |
| OCR + blur | 72.8% EN; MIXED ~79.05% | 73.6% / **79.70%** | **Already beat** (narrowly) |
| **Defense-Prefix** | **73.8%** atk, clean 89.5%, **81.65%** MIXED | 73.6% / **79.70%** | **Not beaten** (−1.95pp MIXED) |

Paper **mean EN+ZH** view: ours always-on blur **74.9%** already beats OCR **73.8%** and DP mean **59.2%** (ZH DP collapses to 44.5%).

**Recommendation:** Treat EN-only DP as a different defense class. For the occlusion paper claim, keep emphasizing **mean EN+ZH + Clean Δ**. If the goal is literally best EN MIXED2000, ship **DP+spatial**, not more fill ablations.

---

### 8. Partner fill ablations (ZH / KO / JA) — follow EN

Same gated Attn EN∩L `cc_bbox` + Phase-C gate; fills = neglect / blur / mean / black. Score L and bilingual EN+L MIXED2000.

**Bilingual MIXED2000 (production quote):**

| Partner | neglect | blur | mean | **black** | Winner |
| --- | ---: | ---: | ---: | ---: | --- |
| ZH | 77.38% | 81.28% | 81.30% | **81.65%** | black (+0.37pp vs blur) |
| KO | 74.18% | **78.50%** | 78.23% | 78.35% | blur (+0.15pp vs black) |
| JA | 80.45% | 82.45% | 82.20% | **82.53%** | black (+0.08pp vs blur) |

**Production freeze:** gated **`cc_bbox_black` for all langs** (EN + ZH/KO/JA). Quote bilingual black rows above in future tables. Ablation note only: KO blur +0.15pp bilingual; L-only ZH/KO prefer blur; JA L-only prefers neglect — not production.

Full tables: [`research_diary.md` § 2026-07-25 — Partner fill ablation](../research_diary.md#L2758); JSON [`partner_fill_ablation/results/`](../../lib/notebooks/partner_fill_ablation/results/). Protocol: [`PROTOCOL.md`](../../lib/notebooks/PROTOCOL.md) §7.2.

---

## Numbers to quote (Jul 25)

| Claim | Number |
| --- | --- |
| EN gated black (atk / clean / MIXED) | **72.9% / 85.8% / 79.35%** |
| Gate fire (atk / clean) | **99.8% / 0.4%** |
| vs gated blur (EN MIXED) | **+1.45pp** (79.35 − 77.90) |
| vs DP EN MIXED2000 | **−2.30pp** (79.35 vs 81.65) |
| Partner bilingual MIXED **black** (ZH/KO/JA, production) | **81.65% / 78.35% / 82.53%** |
| Production fill (all langs) | gated **`cc_bbox_black`** |
| ZH DP defended / ASR / Clean Δ | **44.5% / 52.5% / +0.4pp** |
| DP EN+ZH mean | **59.2%** |
| Dyslexify / SamplingTAR hybrid EN | **66.9% / 67.3%** |

---

## One breath (Jul 25)

> Detect with Attn-last heatmap shape, localize where EN ∩ L agree, black-fill the sticker boxes, reclassify. Production fill is **black for all langs**. EN gated black is **72.9% atk / 85.8% clean / 79.35% MIXED2000** — better than blur (**+1.45pp** MIXED), still short of DP’s **81.65%** on EN alone (−2.30pp); bilingual partner black is **81.65% / 78.35% / 82.53%** (ZH/KO/JA).

---

# Part C — Frozen protocol + detector + baselines (July 20–24, 2026)

## One-paragraph overview

After locking `cc_bbox_blur` as the defense, evaluation was made fair and paper-ready. (1) **Froze attack geometry** so every method sees the same EN/L sticker positions. (2) **Restored accuracy** with thr ≥ 0.95 (ZH multi **74.0%**, Clean Δ **−1.5pp**). (3) Built an **Attn-last heatmap attack detector** — Clean Δ → **~0** for ZH/KO/JA with ≤0.45pp atk drop. (4) Ran four **published baselines** on the same protocol. (5) Jul 24: introduced **MIXED2000** so gated ≫ always is visible (esp. KO/JA).

## What was done (by topic)

### Frozen attack geometry + thr floor

- `attack_pos` baked into `CIFAR10_BALANCED_1000_SAMPLE.json`; thr = max(free, 0.95).
- ZH/KO/JA multi thr-floor def: **74.0% / 69.9% / 75.9%**; Clean Δ **−1.5 / −11.2 / −11.5**.

### Gated detector (`multi`)

| L | Always atk | Gated atk | Always Clean Δ | Gated Clean Δ |
| --- | ---: | ---: | ---: | ---: |
| zh | 74.0% | **73.9%** | −1.45 | **0.00** |
| ko | 69.9% | **69.45%** | −11.25 | **−0.20** |
| ja | 75.9% | **75.7%** | −11.50 | **0.00** |

### Baselines (n=1000, Jul 23)

| Method | Result |
| --- | --- |
| Defense-Prefix (EN then) | EN 73.8% |
| OCR + blur | 73.8% mean |
| Dyslexify / SamplingTAR heads | 20.0% / 11.6% EN |
| Ours ref (always-on blur) | 74.9% mean / −1.5pp |

### MIXED2000 (Jul 24)

| L | Always | Gated | Gated − always |
| --- | ---: | ---: | ---: |
| zh | 80.60% | **81.28%** | +0.68 |
| ko | 73.20% | **78.50%** | +5.30 |
| ja | 76.80% | **82.45%** | +5.65 |

DP EN MIXED2000 bar = **81.65%**. Full prior tables/diary: `../research_diary.md` (2026-07-20 → 2026-07-24).

---

# Part A — Defense sprint (July 16–19, 2026)

## One-paragraph overview

Over the past few days I closed the remaining experimental checklist items for the attention-based defense line. Starting from an EN/ZH “attention last layer” mask that already beat GradCAM, I (1) fixed the failed grid-search baseline with a better scoring rule, (2) ablated several heatmap refinements and settled on **`cc_bbox_blur`** as the current best defense, (3) transferred that defense to Korean and Japanese models, (4) reduced the extra clean-image damage those languages suffered, and (5) reorganized the notebook tree. Open checklist items are now writing: **start the paper** and **write up results**.

---

## Checklist status

| Item | Status | Date |
|------|--------|------|
| Focus experiments on EN + ZH models | Done | earlier |
| Improve grid search | Done | 2026-07-16 |
| Improve heatmap-based defense | Done (for now) | 2026-07-17/18 |
| 4-language transfer of `cc_bbox_blur` | Done | 2026-07-18/19 |
| Reduce KO/JA clean-image damage | Done | 2026-07-19 |
| Organize notebooks | Done | 2026-07-19 |
| Start paper | **Not started** | — |
| Write up results | **Not started** | — |

---

## What was done (by topic)

### 1. Improved grid search (sanity-check baseline)

- Old grid search failed because it **maximized post-occlusion confidence**, which often kept the wrong (attack) class.
- Switching to **confidence-drop of the pre-defense top class** raised mean accuracy from ~12% → **~48%** at the same cost (n=1000).
- Still far behind attention (~73%) and ~10× more expensive, so grid remains a **negative / reference baseline**, not the production defense.
- Extra finding: on mixed EN+ZH stickers, covering the **English** box matters much more than covering Chinese alone.

### 2. Attention vs GradCAM (confirmed across attack types)

Same EN∩ZH intersection defense, three saliency sources, n=1000:

| Attack | Best method | Mean acc after defense | vs GradCAM |
|--------|-------------|------------------------|------------|
| Multilingual (EN+ZH boxes) | Attn-last | **72.6%** | GradCAM 33.1% |
| Unilingual EN+EN | Attn-last | **67.6%** | GradCAM 28.7% |
| Unilingual ZH+ZH | Attn-last | **62.5%** | GradCAM 50.9% |

Attn-last is cheaper (cost 4 vs 6) and usually kinder to clean images. ZH-only is the main caveat (smaller gap, more clean damage).

### 3. Heatmap improvements → current defense `cc_bbox_blur`

Goal: close the gap from 72.6% toward clean accuracy (~86–91%).

Many ideas tested (gating, union masks, finer ViT, attention+grid hybrid). Most did not help. Winners:

| Defense | Mean acc (attacked) | Clean Δ | Cost |
|---------|--------------------:|--------:|-----:|
| Attn-last baseline | 72.6% | −5.7pp | 4 |
| **`cc_bbox_blur`** (keep) | **74.9%** | **−1.5pp** | 4 |

**What `cc_bbox_blur` does:** take Attn-last EN∩ZH heatmap → keep top-2 blobs → snap each to a rectangle (match sticker shape) → Gaussian-blur those regions instead of painting flat color. Same compute budget as before.

Residual gap to clean (~10–15pp) left for later; ablation round closed.

### 4. Four-language transfer (ZH / KO / JA)

Same dual-box setup with partner language L ∈ {zh, ko, ja}: English-only, native-only, and mixed EN+L attacks.

- **Attack recovery works** on hard attacks for all three partners (defended means typically mid-60s to mid-70s; ZH multi reproduces **74.9%**).
- **Problem:** KO/JA clean-image cost was much worse (−11 to −23pp) than ZH (−1.5pp).
- Native-only KO/JA attacks are already weak; English dual-box remains the universal threat.

### 5. KO/JA clean-damage reduction

Ablated threshold floors and mask geometry (dilate, bbox snap) without re-running ZH.

- Main fix: **never tune below thr = 0.95** (the old `uni_en` thr=0.90 was overshooting).
- Result: KO/JA `uni_en` Clean Δ improved from about **−18 / −23pp → −11 / −7pp**, with defended accuracy held or improved.
- Residual gap vs ZH (−1.5pp) looks like **heatmap quality of EN∩KO / EN∩JA**, not just threshold tuning.

### 6. Notebook organization

- `_archive/` — superseded work  
- `_en_zh/` — early EN/ZH GradCAM lineage  
- Top level — current stack: `attention_defense` → `heatmap_defense_improvements` → `four_lang_cc_bbox_blur` → `ko_ja_clean_damage`

---

## Numbers to quote in a meeting

| Claim | Number |
|-------|--------|
| Best EN/ZH defense so far | **74.9%** mean acc, clean Δ **−1.5pp**, cost 4 |
| Gain over Attn-last | **+2.3pp** attacked acc, much better clean Δ |
| Gain over GradCAM (multi) | **74.9%** vs **33.1%** |
| Grid search after fix (still baseline) | **~48.5%** mean @ cost 62 |
| 4-lang: defense recovers hard attacks | mid-60s–mid-70s mean for ZH/KO/JA |
| KO/JA clean damage after fix | roughly **−7 to −11pp** (was −11 to −23) |

---

## Next steps (as of 2026-07-19 — completed afterward)

> See [`homework_summary.md`](../homework_summary.md) for what was finished next: frozen `attack_pos`, thr-floor recovery, attack detector, published baselines.

1. ~~Start the paper~~ → outline in `../paper_draft.md`
2. ~~Write up results~~ → in progress via paper draft + baseline doc
3. Frozen protocol / detector / baselines — done Jul 20–23 (current summary)

---

## How to explain the method in one breath

> We build a mask from where English and partner-language CLIP *agree* they are looking (last-layer attention), reshape that mask into tight rectangles over the text stickers, blur those regions, then reclassify. It recovers most accuracy under multilingual typographic attack with almost no clean-image damage on Chinese, and transfers to Korean/Japanese with a larger but now partially mitigated clean cost.

---

## Key notebook paths

| Work | Path |
|------|------|
| Attention baseline | `lib/notebooks/attention_defense/` |
| Heatmap ablations + `cc_bbox_blur` | `lib/notebooks/heatmap_defense_improvements/` |
| 4-lang transfer | `lib/notebooks/four_lang_cc_bbox_blur/` |
| KO/JA clean damage | `lib/notebooks/ko_ja_clean_damage/` |
| Improved grid (baseline) | `lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_grid/` |
| Full diary | `../research_diary.md` (entries 2026-07-16 → 2026-07-19) |

---

# Part B — July 5 assignments (JA model + 4×4 matrix)

**Project:** Thread B — 4 separate CLIP models (EN, ZH, KO, JA) evaluated against typographic adversarial attacks on CIFAR-10.

## Experimental setup reference

### The 4 typographic attacks

All four attacks use the same method (`draw_word()` — render text on the image, no gradients). They differ only in which language the **adversarial target class name** is written in:

| Attack | What gets written on the image |
|---|---|
| **attack_en** | Target class name in English (e.g. `"truck"`) |
| **attack_zh** | Target class name in Chinese (e.g. `"卡车"`) |
| **attack_ko** | Target class name in Korean (e.g. `"트럭"`) |
| **attack_ja** | Target class name in Japanese (e.g. `"トラック"`) |

Each attack is evaluated against all four models, producing a 4×4 accuracy / ASR matrix (rows = attack language, cols = model language).

### Models used (4 per-language CLIP classifiers)

These are the **classifiers**, not separate models per attack. Every attack is run against all four:

| Language | Model | Library |
|---|---|---|
| **EN** | `ViT-B-32` (OpenAI weights) | `open_clip` |
| **ZH** | `OFA-Sys/chinese-clip-vit-base-patch16` | Hugging Face `transformers` |
| **KO** | `Bingsu/clip-vit-base-patch32-ko` | Hugging Face `transformers` |
| **JA** | `llm-jp/llm-jp-clip-vit-base-patch16` | `open_clip` via `hf-hub:` |

**Note on Japanese:** The STL-10 notebook originally used `line-corporation/clip-japanese-base` (CLYP). It was replaced in the CIFAR-10 notebook with `llm-jp/llm-jp-clip-vit-base-patch16` because CLYP scored ~14% clean accuracy; llm-jp reaches ~93%.

### Model sources (Hugging Face + GitHub)

| Lang | Hugging Face path | GitHub repo |
|---|---|---|
| EN | OpenAI weights via `open_clip` | [openai/CLIP](https://github.com/openai/CLIP), [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| ZH | [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) | [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) |
| KO | [Bingsu/clip-vit-base-patch32-ko](https://huggingface.co/Bingsu/clip-vit-base-patch32-ko) | [Bingsu/clip-vit-base-patch32-ko](https://github.com/Bingsu/clip-vit-base-patch32-ko) |
| JA (current) | [llm-jp/llm-jp-clip-vit-base-patch16](https://huggingface.co/llm-jp/llm-jp-clip-vit-base-patch16) | [llm-jp/llm-jp-clip](https://github.com/llm-jp/llm-jp-clip) |
| JA (original, broken) | [line-corporation/clip-japanese-base](https://huggingface.co/line-corporation/clip-japanese-base) | [line/CLIP-Japanese-base](https://github.com/line/CLIP-Japanese-base) |

### Local repo directories

| Path | Role |
|---|---|
| `lib/notebooks/cifar10_typographic_attack_confusion.ipynb` | CIFAR-10 4×4 experiment (llm-jp JA model) |
| `lib/notebooks/typographic_attack_confusion.ipynb` | STL-10 version |
| `PLAN_typographic_confusion_matrix.md` | Experiment plan + model table |
| `CODE_GUIDE_separate_langs_typographic.md` | Code guide for the 4 model wrappers |
| `claude_experiments/perlang_models.py` | Reusable model wrapper definitions |
| `claude_experiments/typographic_attack.py` | `draw_word` + attack matrix logic |

---

## Assignment 1 — Why is the JA model performing better than the others?

**Short answer:** The original JA model was broken and was replaced with a better one; the replacement happens to be the strongest model in the ensemble.

**What happened step by step:**

1. **Original model (CLYP / `line-corporation/clip-japanese-base`) failed.** It scored only 14–19% on CIFAR-10 — barely above chance. Diagnostics found all 10 class text embeddings were packed together (cosine similarity 0.67–0.86), so the model had no discriminative power. It was built for image–text retrieval, not zero-shot classification. No prompt template fixed it.

2. **Replacement: `llm-jp/llm-jp-clip-vit-base-patch16`.** Selected by searching for Japanese CLIP models benchmarked specifically on CIFAR-10. Developed by Japan's National Institute of Informatics; top-ranked on CIFAR-10 zero-shot accuracy tables. Uses the same `open_clip` API, so the swap was a drop-in.

3. **Final numbers (200-image run):**

| Model | Clean acc | EN attack | ZH attack | KO attack | JA attack |
|---|---|---|---|---|---|
| EN | 85.0% | 5.5% | 79.5% | 83.5% | 80.5% |
| ZH | 90.5% | 33.0% | 58.0% | 88.0% | 67.5% |
| KO | 87.0% | 14.0% | 85.5% | 86.0% | 85.0% |
| **JA** | **93.0%** | 9.5% | **93.0%** | **93.0%** | **92.5%** |

**Why JA is best:**
- **Highest clean accuracy (93%)** — larger, better-filtered training corpus.
- **Near-immune to non-Latin attacks** — Chinese characters and Korean Hangul are not classification-relevant signals in its learned representation, so ZH/KO/JA text overlays do nothing.
- **Still vulnerable to EN attack (9.5%)** — expected. JA web text mixes heavily with English, so the model learned that Latin script matters for classification. All four models share this weakness.

---

## Assignment 2 — Deeper analysis: more samples + typographic attacks in different languages

**What was changed in `lib/notebooks/cifar10_typographic_attack_confusion.ipynb`:**

1. **Sample size: 200 → 1000 images** (cell 9). Gives ~100 images per class, making per-class rates reliable (±3–4 pp confidence).

2. **New cell: per-class accuracy breakdown** (inserted after confusion matrices). For each of the 4 attack languages × 4 models, computes accuracy on each of the 10 CIFAR-10 classes individually. Saved as bar-chart PNGs: `lib/notebooks/results/cifar10_perclass_attack_{en,zh,ko,ja}.png`.

3. **Updated AUC cell: all 4 attack languages** (was EN-only). Now loops over EN/ZH/KO/JA attacks and reports disagreement-detector AUC for each.

**Updated 4×4 accuracy matrix (1000 images):**

|  | model_EN | model_ZH | model_KO | model_JA |
|---|---|---|---|---|
| Clean | 84.2% | 92.7% | 87.7% | 93.2% |
| attack_EN | 4.6% | 36.5% | 15.6% | 8.3% |
| attack_ZH | 79.2% | 58.3% | 84.5% | 90.2% |
| attack_KO | 81.9% | 89.4% | 86.1% | 91.1% |
| attack_JA | 78.2% | 69.8% | 83.5% | 89.9% |

Numbers confirm the 200-image run — no reversals. EN attack is the only universal threat.

**Per-class findings (under EN attack):**
- Most vulnerable: **dog** (0% on EN/KO/JA), **cat** (0–2% on EN/KO/JA) — short, common English words with strong visual–semantic grounding.
- Most resistant: **horse** (20% EN, 41% ZH, 26% KO, 20% JA) — harder to fully fool across all models.

**Disagreement detector AUC — all attack languages:**

| Attack | All-agree rate (attacked) | AUC |
|---|---|---|
| Clean baseline | 78.2% | — |
| EN | 59.3% | 0.588 |
| ZH | 50.4% | **0.646** |
| KO | 73.4% | 0.525 |
| JA | 58.6% | 0.604 |

Key insight: **ZH attack has the highest AUC (0.646)** even though it is the weakest attack. A weak attack that fools only one model (ZH) creates a consistent 1-vs-3 disagreement pattern, which is reliably detectable. A strong attack that fools all four produces consensus on the wrong answer and fires no alarm.

---

## Files changed

| File | Change |
|---|---|
| `lib/notebooks/cifar10_typographic_attack_confusion.ipynb` | 200→1000 samples; new per-class cell; AUC cell extended to all 4 attack langs; `clean_preds` stored |
| `lib/notebooks/results/cifar10_confusion_results.json` | Updated with 1000-image numbers; new `per_class_acc`, `per_class_asr`, `detector.by_attack_lang` keys |
| `lib/notebooks/results/cifar10_perclass_attack_en.png` | New — per-class bars under EN attack |
| `lib/notebooks/results/cifar10_perclass_attack_zh.png` | New — per-class bars under ZH attack |
| `lib/notebooks/results/cifar10_perclass_attack_ko.png` | New — per-class bars under KO attack |
| `lib/notebooks/results/cifar10_perclass_attack_ja.png` | New — per-class bars under JA attack |
| `research_goal.md` | Added Thread B section: second mermaid diagram, Q1/Q2 findings rows, Thread B details block |
| `../research_diary.md` | Multiple new entries: AUC results, JA model analysis, detector improvement directions, deeper-analysis results |
