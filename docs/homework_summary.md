# Homework Summary — Recent Progress (July 16–19, 2026)

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was finished on the checklist, and what comes next.

> Older Assignment 1–2 notes (JA model swap, 4×4 attack matrix): [`homework_summary_assignments_jul5.md`](homework_summary_assignments_jul5.md). This file covers **defense work only**.

---

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

## Next steps (priority order)

1. **Start the paper** — outline contribution: cross-lingual attention intersection + `cc_bbox_blur` as a cheap spatial defense for typographic attacks; position vs GradCAM and occlusion search.
2. **Write up results** — consolidate the tables above (EN/ZH ablations, 4-lang transfer, KO/JA clean-Δ) into paper-ready figures and a short methods section.
3. *(Optional, later experiments)* Close residual gap to clean accuracy; improve KO/JA heatmap quality (stronger partner CLIPs or better saliency), not more threshold fiddling.

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
| Full diary | `docs/research_diary.md` (entries 2026-07-16 → 2026-07-19) |
