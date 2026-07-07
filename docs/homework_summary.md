# Homework Summary

**Project:** Thread B — 4 separate CLIP models (EN, ZH, KO, JA) evaluated against typographic adversarial attacks on CIFAR-10.

---

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
