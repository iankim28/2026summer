# Shared vs. Separate Encoders: does cross-lingual disagreement come back?

*The original proposal wanted to defend CLIP using cross-lingual **disagreement** under
attack. We showed it fails on a real multilingual CLIP because the image encoder is
**shared** (all languages agree on the wrong answer). This experiment asks the natural
follow-up the user raised: **what if the encoders are NOT shared?** Answer: the
disagreement signal comes back — strongly. All numbers below were run and verified.*

---

## The two conditions (same English typographic attack on both)

We write the English target word on an image (the strong typographic attacker) and
compare:

- **SHARED** — one multilingual CLIP (`xlm-roberta-base-ViT-B-32`); four language label
  sets (en/zh/ko/ja) share **one** image embedding. *(How real M-CLIP works.)*
- **SEPARATE** — four **independently trained** per-language CLIPs, each with its **own**
  image encoder:
  - en = OpenAI CLIP ViT-B/32
  - zh = Chinese-CLIP (`OFA-Sys/chinese-clip-vit-base-patch16`)
  - ko = `Bingsu/clip-vit-base-patch32-ko`
  - ja = `line-corporation/clip-japanese-base`

---

## Results (STL-10, 200 images, English typographic attack)

### SHARED encoder
| lang | clean | attacked | ASR (→written word) |
|---|---|---|---|
| en | 95.5 | 40.0 | 59.5 |
| zh | 95.5 | 52.5 | 47.0 |
| ko | 87.0 | 46.5 | 53.5 |
| ja | 95.5 | 46.5 | 53.5 |

- all-4-agree: **89.0% → 71.5%** under attack (stays high)
- majority-vote ensemble: 95.5% → **47.5%** (defense **fails**)
- disagreement-detector ROC-AUC: **0.588** (barely better than chance)

### SEPARATE encoders
| lang | clean | attacked | ASR (→written word) |
|---|---|---|---|
| en | 98.0 | 31.0 | 69.0 |
| zh | 96.0 | 61.0 | 37.0 |
| ko | 96.0 | 38.5 | 60.5 |
| **ja** | 98.0 | **93.0** | **5.0** |

- all-4-agree: **94.5% → 26.5%** under attack (**collapses → disagreement!**)
- majority-vote ensemble: 97.5% → **56.0%** (more robust than shared)
- disagreement-detector ROC-AUC: **0.839** (**clearly detects the attack**)

---

## The headline contrast

| | SHARED (real M-CLIP) | SEPARATE (4 independent CLIPs) |
|---|---|---|
| agreement under attack | 89% → **71.5%** (high) | 94.5% → **26.5%** (collapses) |
| disagreement-detector AUC | **0.588** (≈ chance) | **0.839** (works) |
| ensemble defense under attack | 47.5% (fails) | 56.0% (helps) |

**The disagreement signal the proposal wanted is real — but only when the encoders are
NOT shared.** With one shared image encoder, all languages move together and agree on the
wrong class (no signal). With separate encoders, each reads the image differently, they
**disagree**, and a disagreement detector becomes useful (AUC 0.84).

---

## Why separate encoders disagree (the mechanism)

The per-language encoders have **heterogeneous text-reading abilities**:

- The OpenAI (en) and Korean encoders are **heavily fooled** by English text (ASR 60–69%).
- The Chinese encoder is **moderately** fooled (37%).
- The **Japanese encoder barely reads English text at all** (ASR **5%**, accuracy stays
  93%) — it was trained on Japanese web data with little English in-image text.

So under an English typographic attack, some encoders flip to "dog" and others keep saying
the true class → they **disagree**, which is exactly the signal a detector/ensemble needs.
In the shared model there is only one encoder, so there is nothing to disagree.

**Verification (ruling out an artifact).** The Japanese model's robustness is not because
its preprocessing crops out the text: re-running with the word in the **dead center** of
the image (impossible to crop away) gives the same result — JA ASR 7.3% (acc 89.3%) while
en/ko/zh are fooled 53–79%. The robustness is genuine.

---

## What this means for the original proposal

- The proposal's intuition ("attack one language ⇒ the others disagree and catch it") is
  **architecturally correct for an ensemble of independent per-language models** — there
  the disagreement detector reaches AUC 0.84.
- It **fails for a real multilingual CLIP** because that design *shares* the image encoder
  by construction, so there is no per-language encoder to disagree.
- Trade-off: separate encoders restore the defense but cost ~4× the compute and lose the
  elegance of one model for all languages — and they still don't fully stop the attack
  (English text fools 2 of 4), they just make it **detectable**.

### A clean project framing
> *"Multilingual AI can be defended by cross-lingual disagreement — but only if the
> languages don't share an image encoder. Sharing the encoder (how real multilingual CLIP
> is built) silently removes the very disagreement the defense needs."*

This converts the earlier negative result into a constructive design principle, and the
**Japanese encoder's immunity to English text** is a memorable, mechanistic centerpiece.

---

## Reproducibility
| file | what it does |
|---|---|
| `perlang_models.py` | uniform wrappers for the 4 independent per-language CLIPs |
| `shared_vs_separate.py` | the full shared-vs-separate comparison (table above) |
| `results/shared_vs_separate.json` | raw numbers |

Models load from Hugging Face with the existing environment (transformers + open_clip);
the experiment is gradient-free and runs in a few minutes on one GPU.
