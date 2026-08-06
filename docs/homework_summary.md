# Homework Summary — August 1, 2026

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was done since the Jul 27 briefing (Jul 29–30 table lock + paper prep).

> Older briefings: [`archive/homework_summary_archive.md`](archive/homework_summary_archive.md) (Jul 27 + earlier).  
> **Paper tables (locked):** [`paper_tables_final.md`](paper_tables_final.md); index [`tables_index.md`](tables_index.md).  
> Main / ablation detail: [`4_lang_table.md`](4_lang_table.md), [`ablation_study.md`](ablation_study.md).  
> Paper draft: [`paper_draft.md`](paper_draft.md).  
> Figures: [`paper_figures_and_notes.md`](paper_figures_and_notes.md).  
> Protocol: [`PROTOCOL.md`](../lib/notebooks/PROTOCOL.md).  
> Diary: [`research_diary.md`](research_diary.md) ([Jul 29](research_diary.md#L2983)).

**Shared protocol unless noted:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), 224×224, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.

---

## One-paragraph overview

Since Jul 27 we **locked the paper tables** and started writing. Jul 29 added ZH MIXED for OCR/DP, fixed 3-box recovery with matched `top_k`, and ran true **4-way** EN∩ZH∩KO∩JA occlusion (gated mean MIXED **83.4%**). Jul 30 filled the last table gaps: **Raw CLIP** on the main table, gated **text_only / white_only** black occlusion, Tables 1–4 reshaped in [`tables_index.md`](tables_index.md) (1-decimal, production bolding, slim Table 1 + clearer Table 4 candidates), figures marked frozen for draft, and [`paper_draft.md`](paper_draft.md) §1 expanded to first-pass prose with an abstract draft.

---

## Checklist done (Jul 29–30)

| Item | Kind | Status | Where |
| --- | --- | --- | --- |
| ZH MIXED for OCR + Defense-Prefix | results / docs | **Done** | [`4_lang_table.md`](4_lang_table.md); [`mixed_2000_summary.json`](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json) |
| 3-box re-run with `top_k = num_boxes` | attack | **Done** | gated EN **45.9%**; [`boxes_n1000.json`](../lib/notebooks/attack_geometry_ablation/results/boxes_n1000.json) |
| Four-way occlusion EN∩ZH∩KO∩JA | method | **Done** | gated mean MIXED **83.4%**; [`four_way_occlusion/`](../lib/notebooks/four_way_occlusion/) |
| Raw CLIP row on main table | tables | **Done** | mean atk **7.1%**; [`tables_index.md`](tables_index.md) Table 1 |
| Gated black on white_only / text_only / full | attack | **Done** | [`content_occlusion_n1000.json`](../lib/notebooks/attack_component_ablation/results/content_occlusion_n1000.json) |
| Paper Tables 1–4 locked + polish | writing | **Done** | [`tables_index.md`](tables_index.md) (slim T1, clearer T4, 1-dec, bold rule) |
| Figures frozen for draft | figures | **Done** | [`paper_figures_and_notes.md`](paper_figures_and_notes.md); `detector_pipeline.png` optional/out |
| Abstract + §1 first-pass prose | writing | **Done** | [`paper_draft.md`](paper_draft.md) |

---

## What was done

### 1. Main-table language coverage + ZH MIXED (Jul 29)

Promoted language-coverage as Table 1; derived OCR/DP **ZH MIXED** from existing finals (no GPU re-run).

| Method | EN MIXED | ZH MIXED | Scope MIXED |
| --- | ---: | ---: | ---: |
| OCR + blur | 79.1% | 82.7% | 80.9% (EN+ZH) |
| Defense-Prefix | 81.7% | 68.2% | 74.9% (EN+ZH) |
| **Ours EN∩ZH black** | **79.4%** | **84.0%** | **81.7%** bilingual |

DP still wins EN MIXED; collapses on ZH. Diary: [2026-07-29](research_diary.md#L2983).

---

### 2. Four-way occlusion (Jul 29)

True EN ∩ ZH ∩ KO ∩ JA Attn-last → gated `cc_bbox_black` ([`four_way_occlusion/`](../lib/notebooks/four_way_occlusion/)).

| Arm | Mean atk | Mean MIXED | Clean Δ |
| --- | ---: | ---: | --- |
| never (Raw CLIP) | 7.1% | 48.5% | — |
| always | 77.5% | 73.0% | ~−20 pp |
| **gated** | **77.4%** | **83.4%** | ≈ −0.3 to −0.6 pp |

Production default stays pairwise EN∩L (cost 4); 4-way is the all-language variant.

---

### 3. Content occlusion — text vs white gated (Jul 30)

New runner [`attack_component_ablation/run_occlusion.py`](../lib/notebooks/attack_component_ablation/run_occlusion.py); n=1000 CUDA. `full` matched production **72.9 / 76.5**.

| Mode | EN (no def) | EN gated | ZH gated |
| --- | ---: | ---: | ---: |
| white_only | 75.5% | 70.9% | 75.0% |
| text_only | 9.6% | **75.8%** | **82.7%** |
| **full** (production) | 4.5% | **72.9%** | **76.5%** |

Glyphs drive hijack and recovery; blank pads do not.

---

### 4. Paper tables locked ([`tables_index.md`](tables_index.md))

| Paper table | Content |
| --- | --- |
| **Table 1** | Raw CLIP + baselines + Ours (full + slim candidate) |
| **Table 2** | Font / text vs white / #boxes (shared columns) |
| **Table 3** | Sticker / text / hybrid with Raw CLIP |
| **Table 4** | #langs / gate / fill (original + clearer Score candidate) |

Polish: **1 decimal**; bold = Ours rows + Raw floor (T1) or **production setting** (ablations), not every column max. Recommended body set: **T1 slim + T2 + T3 + T4 clearer**.

---

### 5. Figures + writing start

- Five paper figures marked **frozen for draft** (black fill); `detector_pipeline.png` out of main set — [`paper_figures_and_notes.md`](paper_figures_and_notes.md).
- [`paper_draft.md`](paper_draft.md): abstract draft + §1 first-pass prose; §2–§3 still outline-heavy with locked table pointers.

---

## Numbers to quote in a meeting

| Claim | Number |
| --- | ---: |
| Raw CLIP mean atk / Scope MIXED | **7.1% / 48.5%** |
| Ours EN∩ZH atk / EN MIXED / bilingual MIXED | **72.9% / 79.4% / 81.7%** |
| OCR / DP Scope MIXED (EN+ZH) | **80.9% / 74.9%** |
| Four-way gated mean MIXED | **83.4%** |
| text_only / full gated EN | **75.8% / 72.9%** |
| Partner bilingual MIXED ZH / KO / JA | **81.7 / 78.4 / 82.5%** |
| Production geometry | **font 24 / 2 boxes / black / gated** |

---

## Next steps

1. Expand §2–§3 outline → full prose; draft Related Work; lock title/abstract.
2. Put **T1 slim + T2 + T3 + T4 clearer** into the paper body; full T1 in appendix if needed.
3. Optional only: animal / multi-sticker localization — not required for typographic main claim.

---

## One breath

> Paper tables are locked. Raw CLIP collapses to **7.1%** mean atk; gated black recovers EN∩ZH to **72.9%** atk / **81.7%** bilingual MIXED; 4-way gated reaches **83.4%** mean MIXED; text_only gated (**75.8%**) ≈ full — glyphs, not pads. Figures frozen; §1 prose started. Canonical tables: [`tables_index.md`](tables_index.md).

---

## Key paths

| Work | Path |
| --- | --- |
| Paper tables | [`docs/tables_index.md`](tables_index.md) |
| 4-lang / ablation detail | [`4_lang_table.md`](4_lang_table.md), [`ablation_study.md`](ablation_study.md) |
| Paper draft | [`docs/paper_draft.md`](paper_draft.md) |
| Figures checklist | [`docs/paper_figures_and_notes.md`](paper_figures_and_notes.md) |
| Content occlusion JSON | [`attack_component_ablation/results/content_occlusion_n1000.json`](../lib/notebooks/attack_component_ablation/results/content_occlusion_n1000.json) |
| Four-way JSON | [`four_way_occlusion/results/four_way_n1000.json`](../lib/notebooks/four_way_occlusion/results/four_way_n1000.json) |
| Diary (Jul 29) | [`research_diary.md`](research_diary.md#L2983) |
| Archive (Jul 27 + earlier) | [`archive/homework_summary_archive.md`](archive/homework_summary_archive.md) |
