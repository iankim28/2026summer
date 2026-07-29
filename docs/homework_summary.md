# Homework Summary — July 27, 2026

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was done **today**.

> Older briefings: [`archive/homework_summary_archive.md`](archive/homework_summary_archive.md) (Jul 25 + earlier).  
> Ablation appendix: [`ablation_study.md`](ablation_study.md).  
> Main results tables: [`4_lang_table.md`](4_lang_table.md).  
> All tables index: [`tables_index.md`](tables_index.md).  
> Paper outline: [`paper_draft.md`](paper_draft.md).  
> Protocol: [`PROTOCOL.md`](../lib/notebooks/PROTOCOL.md).  
> Full diary for today: [`research_diary.md`](research_diary.md) ([animal occlusion](research_diary.md#L2889), [ablation + geometry](research_diary.md#L2936)).

**Shared protocol unless noted:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), 224×224, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L `cc_bbox_black`.

---

## One-paragraph overview

Today closed the **ablation checklist**: measured production **black occlusion** on animal / mixed / text stickers; consolidated method + attack ablations into [`ablation_study.md`](ablation_study.md); ran the two open geometry sweeps (**font 12/24/40**, **boxes 1/2/3**); and synced [`paper_draft.md`](paper_draft.md) to [`4_lang_table.md`](4_lang_table.md) + the new appendix. Headline: gated black stays **typographic** (text EN **72.9%**); hybrid partial (**43.5%**); animal-only weak (EN∩ZH **20.6%**, EN-only **32.9%**). Production threat model stays **font 24 / 2 boxes** — font 40 and 3 boxes stress the recipe.

---

## Checklist done today

| Item | Kind | Status | Where |
| --- | --- | --- | --- |
| Animal / hybrid / text black occlusion | attack + method | **Done** | [`ablation_study.md`](ablation_study.md) §A.3 / §B.2; [diary](research_diary.md#L2889) |
| Fill type (document only) | method | **Done** | [`ablation_study.md`](ablation_study.md) §A.1 |
| Gate on vs always-on (document only) | method | **Done** | [`ablation_study.md`](ablation_study.md) §A.2 |
| Single-lang occlusion (document only) | method | **Done** | [`ablation_study.md`](ablation_study.md) §A.3 |
| Four-lang main results (document only) | method | **Done** | [`4_lang_table.md`](4_lang_table.md); [`ablation_study.md`](ablation_study.md) §A.4 |
| Text vs white / sticker / hybrid | attack | **Done** | [`ablation_study.md`](ablation_study.md) §B.1–B.2 |
| Font size 12 / 24 / 40 | attack | **Done** | [`ablation_study.md`](ablation_study.md) §B.3; [diary](research_diary.md#L2936) |
| Number of boxes 1 / 2 / 3 | attack | **Done** | [`ablation_study.md`](ablation_study.md) §B.4; [diary](research_diary.md#L2936) |
| Paper sync (main + ablations + geometry) | writing | **Done** | [`paper_draft.md`](paper_draft.md) |

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

- Doc: [`ablation_study.md` §A.3 / §B.2](ablation_study.md)  
- Diary: [`research_diary.md` § 2026-07-27 — Animal-sticker black occlusion](research_diary.md#L2889)  
- Code / JSON / figure: [`run_occlusion.py`](../lib/notebooks/animal_sticker_ablation/run_occlusion.py), [`occlusion_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json), [`gallery_occlusion.png`](../lib/notebooks/animal_sticker_ablation/figures/gallery_occlusion.png)

---

### 2. Ablation study write-up (method + attack)

New appendix source [`ablation_study.md`](ablation_study.md) consolidates already-finished numbers (no re-run) plus today’s geometry:

| Section | Ablation |
| --- | --- |
| A.1 | Fill type — black > mean > blur > neglect |
| A.2 | Gate on/off — KO/JA Clean Δ −11pp → ~0 |
| A.3 | Single-lang / EN-only vs EN∩L |
| A.4 | Four-lang transfer → [`4_lang_table.md`](4_lang_table.md) |
| B.1–B.2 | Glyphs vs white pads; sticker / text / hybrid defense |
| B.3–B.4 | Font size; number of boxes |

Diary inventory: [`research_diary.md` § 2026-07-27 — Ablation study write-up + attack geometry](research_diary.md#L2936).

---

### 3. Font size (attack geometry)

Dual-box, gated black, font ∈ {12, **24**, 40}.

| Font | never EN ASR | gated EN | gated ZH | Clean Δ EN |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 89.9% | **76.0%** | 83.9% | +0.0 pp |
| **24** | **95.3%** | **72.9%** | **76.5%** | −0.1 pp |
| 40 | 96.9% | 58.3% | 56.3% | −0.4 pp |

**Verdict:** Font **24** stays production. Font 40 drops gated EN **−14.6 pp**.

- Doc: [`ablation_study.md` §B.3](ablation_study.md)  
- Diary: [geometry section](research_diary.md#L2936)  
- Results: [`font_n1000.json`](../lib/notebooks/attack_geometry_ablation/results/font_n1000.json)

---

### 4. Number of boxes (attack geometry)

Font 24, boxes ∈ {1, **2**, 3}; **`top_k = num_boxes`** (capacity matched to threat).

| Boxes | never EN ASR | gated EN | gated ZH | Clean Δ EN |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 94.1% | **78.2%** | 82.9% | −0.1 pp |
| **2** | **95.3%** | **72.9%** | **76.5%** | −0.1 pp |
| 3 | 97.5% | **45.9%** | **56.1%** | −0.8 pp |

**Verdict:** Dual-box stays production. With matched `top_k=3`, three boxes recover gated EN to **45.9%** (was **8.5%** under production `top_k=2`).

- Doc: [`ablation_study.md` §B.4](ablation_study.md)  
- Results: [`boxes_n1000.json`](../lib/notebooks/attack_geometry_ablation/results/boxes_n1000.json)  
- Code: [`attack_geometry_ablation/run_ablation.py`](../lib/notebooks/attack_geometry_ablation/run_ablation.py)

---

### 5. Paper draft sync

[`paper_draft.md`](paper_draft.md) now pulls:

- Main avg + lang detail from [`4_lang_table.md`](4_lang_table.md) (§3.5)
- Animal / hybrid / text occlusion (§3.2.2)
- Font + boxes geometry (§3.11); threat model states **font 24**
- Appendix pointer → [`ablation_study.md`](ablation_study.md)

---

## Numbers to quote in a meeting

| Claim | Number |
| --- | --- |
| Gated text (all_text) EN / ZH | **72.9% / 76.5%** |
| Gated mixed EN | **43.5%** |
| Gated animal EN∩ZH / EN-only | **20.6% / 32.9%** |
| Animal oracle GT black EN | **56.9%** |
| Font 12 / **24** / 40 gated EN | **76.0% / 72.9% / 58.3%** |
| Boxes 1 / **2** / 3 gated EN | **78.2% / 72.9% / 45.9%** |
| Ours avg bilingual MIXED (ZH/KO/JA) | **80.84%** ([`4_lang_table.md`](4_lang_table.md)) |
| Production geometry | **font 24 / NUM_BOXES=2** |

---

## Next steps

1. Write prose (title/abstract, Related Work) from the synced outline.
2. Regenerate pipeline figures with **black** fill.
3. Optional only: stronger animal / multi-sticker localization — not required for the typographic main claim.

---

## One breath

> Ablation checklist closed. Gated black recovers **typographic** dual-box (EN **72.9%**), partially hybrid (**43.5%**), poorly animal-only (**20.6%**; EN-only **32.9%** vs oracle **56.9%**). Font **24** / **2 boxes** stay production; font 40 (−14.6 pp) is a known stress case; 3 boxes with matched `top_k=3` recover gated EN **45.9%**. Full tables live in [`ablation_study.md`](ablation_study.md) and [`4_lang_table.md`](4_lang_table.md); narrative in the [Jul 27 diary](research_diary.md#L2889).

---

## Key paths

| Work | Path |
| --- | --- |
| Ablation appendix | [`docs/ablation_study.md`](ablation_study.md) |
| Main 4-lang tables | [`docs/4_lang_table.md`](4_lang_table.md) |
| Paper outline | [`docs/paper_draft.md`](paper_draft.md) |
| Animal occlusion JSON / gallery | [`animal_sticker_ablation/results/occlusion_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json), [`gallery_occlusion.png`](../lib/notebooks/animal_sticker_ablation/figures/gallery_occlusion.png) |
| Font / boxes JSON | [`attack_geometry_ablation/results/`](../lib/notebooks/attack_geometry_ablation/results/) |
| Diary (today) | [`research_diary.md`](research_diary.md) ([#L2889](research_diary.md#L2889), [#L2936](research_diary.md#L2936)) |
| Archive (Jul 25 + earlier) | [`archive/homework_summary_archive.md`](archive/homework_summary_archive.md) |
