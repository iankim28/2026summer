# Agent Handoff Document — v2

**Project:** `d:\ian\2026summer`  
**Notebook under review:** `notebooks/updated_multilingual_consensus_colab.ipynb`  
**Handoff from:** session that resolved Q1  
**Next task:** Resolve Q2 and Q3 (see Section 6 below)

---

## 1. What this project is about

The notebook critically tests a proposed adversarial defence for multilingual CLIP. The
idea: label images in five languages simultaneously (EN, KO, ES, FR, JA); if an attack
is designed to fool English only, the other languages should stay correct and "disagree",
exposing the attack.

See `research_goal.md` for a plain-English summary. See `handoff.md` for the original
full experimental background.

---

## 2. Repository structure

```
2026summer/
├── notebooks/
│   ├── updated_multilingual_consensus_colab.ipynb   ← MAIN notebook (sections 1–11, fully run)
│   ├── multilingual_adversarial_defence_tutorial.ipynb  ← longer tutorial version
│   └── intro_to_CLIP.ipynb                          ← background reading
├── research_goal.md     ← plain-English project summary
├── handoff.md           ← original handoff (full experimental results, Sections 1–10)
├── handoff2.md          ← this file
├── research_diary.md    ← currently empty
├── requirements.txt     ← pinned deps
└── .venv/               ← Python venv (Python 3.13, CUDA, RTX 5070 Ti)
```

---

## 3. Environment

- **GPU:** NVIDIA GeForce RTX 5070 Ti  
- **Python:** 3.13 (venv at `d:\ian\2026summer\.venv\Scripts\python.exe`)  
- **Key packages:** `open_clip_torch`, `transformers 5.12.1`, `datasets`, `torch`  
- **Dataset:** CIFAR-10 via HuggingFace (`uoft-cs/cifar10`) — already cached. Do NOT
  switch to `torchvision.datasets.CIFAR10` (slow mirror).  
- **Model:** `xlm-roberta-base-ViT-B-32` on `laion5b_s13b_b90k` — already cached.

### Critical Windows/Jupyter fixes already applied

- `NUM_WORKERS = 0` on Windows (pickle errors with `num_workers > 0` in Jupyter)
- `HFCifar10` defined as a **top-level class**, not nested
- `import transformers` must come **before** `import open_clip` in the imports cell

---

## 4. Current state of the notebook

Sections 1–10 were already present and fully run before this session.  
**Section 11 was added in this session** (cells 32–38, appended at the end, original
cells untouched). It directly resolves Q1.

| Section | What it does |
|---|---|
| 1 | Setup, imports, model loading |
| 2 | CIFAR-10 loader and class labels |
| 3 | Text embedding builder (`TXT` dict) |
| 4 | Clean zero-shot accuracy (n=500) |
| 5 | H1 / Q1 transfer experiment — EN-only PGD, population-level metrics |
| 6 | Mechanism — cosine similarity proof of why transfer is inevitable |
| 7 | Disagreement detector (ROC-AUC) |
| 8 | H2 — attacker cost scaling |
| 9 | Consensus-purification denoiser (non-adaptive + adaptive) |
| 10 | Overall verdict table |
| **11** | **Q1 deep dive — per-sample retention rate, per-image grid, heatmap, text-side positive control, Q1 verdict** |

---

## 5. Q1 is resolved — full results from Section 11

**Q1:** Does an English-only image-space attack stay confined to English?  
**Answer: No.**

### Retention rate (fraction of EN-fooled images still correct in each other language)

| ε (/255) | EN-fooled (n) | KO | ES | FR | JA |
|---|---|---|---|---|---|
| 0.5 | 277/300 | 2.5% | 3.2% | 2.2% | 1.4% |
| 1 | 285/300 | 0.4% | 0.4% | 0.4% | 0.0% |
| 2 | 286/300 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4–8 | 286/300 | 0.0% | 0.0% | 0.0% | 0.0% |

Success criterion for the defence was retention > 50%. Best observed: 3.2% (ES at ε=0.5).

### Text-side positive control (confirms disagreement IS possible — just not via image-space PGD)

PGD on `TXT["en"]` only, clean images, 20 steps, eps=0.10 in embedding space:

| language | accuracy | retention |
|---|---|---|
| EN (attacked) | 11% | — |
| KO | 85% | 86.6% |
| ES | 90% | 96.3% |
| FR | 93% | 100% |
| JA | 92% | 98.8% |

The contrast (image-space: 0–3% retention; text-side: 87–100% retention) shows the
defence's mechanism works only under a text-manipulation threat model. Under the standard
image-space threat model it is architecturally unachievable.

### Why (one sentence)

Same-class cross-lingual cosine (0.914) > different-class within-English cosine (0.792),
so the gradient that fools English class-c simultaneously fools every language's class-c.

---

## 6. What to work on next

### Priority 1 — Resolve Q2 more rigorously (disagreement detector)

Section 7 already shows AUC < 0.5 for all three disagreement scores. The existing
framing reports raw AUC numbers but does not directly connect them to the Q2 question:

> *"Can disagreement be used as a warning signal?"*

Suggested additions (append new cells, do not touch existing ones):
- **Threshold sweep plot:** for the best-performing score (`n_unique`), plot precision,
  recall, and F1 vs threshold to show there is no usable operating point — not just that
  AUC < 0.5 on average.
- **Score distribution plot:** overlay histogram of disagreement scores for clean vs
  adversarial images. They should mostly overlap (or swap), making detection impossible
  to visualise directly.
- **Q2 verdict cell:** explicit markdown conclusion mirroring the Q1 verdict format.

Key variables already in memory: `dloader`, `clean_scores`, `adv_scores` (populated in
Section 7). If the kernel has been restarted, re-run Sections 1–7 first.

### Priority 2 — Resolve Q3 more rigorously (denoiser)

Section 9 shows the denoiser has three problems (see `handoff.md` §4f). The existing
framing is correct but incomplete. Suggested additions:

- **Switch dataset to STL-10** (`DATASET = "stl10"` at the top of the notebook) to fix
  the 29-point clean accuracy drop. STL-10 is 96×96 native; CIFAR-10 is 32×32 upscaled
  to 224×224 which the denoiser smooths out destructively. This is the single most
  impactful change for Q3.
- **Denoiser ceiling annotation:** add a horizontal line on the denoiser accuracy plot at
  the denoiser's clean-image ceiling (currently 63%), so the "non-adaptive recovery"
  (0% → 56.5%) is clearly labelled as returning to a degraded baseline, not to 92%.
- **Q3 verdict cell:** explicit markdown conclusion mirroring the Q1 verdict format.

### Priority 3 — Run the ablation (`RUN_ABLATION = True`)

Set `RUN_ABLATION = True` and re-run Section 9. This trains an English-only denoiser and
compares it to the all-5-language version. If they match (expected), multilingual
consensus contributes zero to the denoiser — closing the argument. ~4 minutes extra.

### Priority 4 — update research_diary.md

Currently empty. Write a brief research diary entry summarising what was found and what
remains open.

---

## 7. Key variables (after running Sections 1–11)

| Variable | Type | Description |
|---|---|---|
| `LANGS` | `list[str]` | `["en","ko","es","fr","ja"]` |
| `CLASSES` | `list[str]` | 10 CIFAR-10 class names |
| `EPS_LIST` | `list[int]` | `[0.5, 1, 2, 4, 8]` |
| `TXT` | `dict[str, Tensor]` | text embeddings per language, shape `(10, D)`, unit vectors |
| `clean_probs` | `dict[str, ndarray]` | clean softmax probs, shape `(300, 10)` |
| `adv_probs` | `dict[int, dict[str, ndarray]]` | adv softmax probs per ε |
| `labels` | `ndarray` | true class indices, shape `(300,)` |
| `LOGIT_SCALE` | `Tensor` | CLIP logit scale (scalar) |
| `encode_image(x)` | function | normalised image features from pixel tensor |
| `logits_for(f, t)` | function | `LOGIT_SCALE * f @ t.T` |
| `per_lang_probs(x)` | function | softmax probs dict for all 5 languages |
| `pgd(x, y, eps, attacked)` | function | PGD attack; `attacked=["en"]` for EN-only |
| `get_loader(n, seed)` | function | returns `(DataLoader, dataset)` for n CIFAR-10 images |
| `acc(probs, l)` | function | accuracy for language `l` |
| `agreement(probs)` | function | fraction of images where all 5 languages agree |

---

## 8. What the two notebooks are (for context)

| Notebook | Purpose |
|---|---|
| `updated_multilingual_consensus_colab.ipynb` | Critical replication — runs experiments, shows the defence fails, includes the mechanism proof and Q1 deep dive. **This is the main one.** |
| `multilingual_adversarial_defence_tutorial.ipynb` | Longer pedagogical tutorial — walks through the same pipeline, assumes the defence works, designed for learning. |
