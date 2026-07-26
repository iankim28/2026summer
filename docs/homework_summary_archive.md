# Archive — Homework Summary (through July 24, 2026)

**Status:** Archived. Current briefing: [`homework_summary.md`](homework_summary.md) (**July 25 only**).  
**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.

This file holds all superseded homework briefings in one place:

1. **Part C** — July 20–24 (frozen protocol, detector gate, baselines, MIXED2000)
2. **Part A** — July 16–19 defense sprint (`cc_bbox_blur`, 4-lang, KO/JA clean Δ)
3. **Part B** — July 5 Assignment 1–2 (JA model swap, 4×4 attack matrix)

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

DP EN MIXED2000 bar = **81.65%**. Full prior tables/diary: `docs/research_diary.md` (2026-07-20 → 2026-07-24).

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

> See [`homework_summary.md`](homework_summary.md) for what was finished next: frozen `attack_pos`, thr-floor recovery, attack detector, published baselines.

1. ~~Start the paper~~ → outline in `docs/paper_draft.md`
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
| Full diary | `docs/research_diary.md` (entries 2026-07-16 → 2026-07-19) |

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
| `docs/PLAN_typographic_confusion_matrix.md` | Experiment plan + model table |
| `docs/CODE_GUIDE_separate_langs_typographic.md` | Code guide for the 4 model wrappers |
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
| `docs/research_goal.md` | Added Thread B section: second mermaid diagram, Q1/Q2 findings rows, Thread B details block |
| `docs/research_diary.md` | Multiple new entries: AUC results, JA model analysis, detector improvement directions, deeper-analysis results |
