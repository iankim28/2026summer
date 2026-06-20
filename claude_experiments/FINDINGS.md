# Empirical test of "Multilingual Consensus Purification" — Findings

**Question posed:** implement the proposal and execute it to see whether the idea
is *actually possible*.

**Short answer: No.** The proposal's foundational premise (H1) is false for a
shared-encoder multilingual CLIP, and it fails in a specific, mechanistically
explained way that takes all three proposed defenses down with it.

---

## Setup (faithful to the proposal)

* **Model:** `open_clip` **`xlm-roberta-base-ViT-B-32`** (laion5b) — a frozen
  multilingual CLIP with a **shared ViT-B/32 image encoder** and an XLM-RoBERTa
  text tower (100 languages). This satisfies the proposal's "M-CLIP with a
  ViT-B/32 image encoder so all languages share one image encoder." (The result is
  also confirmed on the literal FreddeFrallan M-CLIP — see end of doc.)
* **Datasets:** STL-10 (primary, 96px) and CIFAR-10 (fallback, 32px).
* **Languages:** English (target), Korean (primary contrast), Spanish, French, Japanese.
* **Attacks:** white-box, per-image, L∞, ε ∈ {2,4,8,16}/255 (+ a fine low-ε sweep
  {0.25,0.5,1,2,4}/255). FGSM and PGD are both implemented (`attacks.py`); all
  numbers below use **PGD** (20–40 steps, random start), the stronger attack.
  Pixel-space attack at the 224² encoder input; CLIP normalization folded into the
  forward pass (a genuine L∞ ball on the input).
* **Single-language attack** = maximize cross-entropy on **English** labels only;
  the other languages then reveal transfer.

## Clean zero-shot accuracy (sanity — pipeline works)

| dataset | en | ko | es | fr | ja | ensemble | all-5-agree |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 93.7 | 84.4 | 94.2 | 94.4 | 93.6 | 94.4 | 85.1 |
| STL-10 | 96.6 | 89.1 | 96.8 | 96.3 | 96.3 | 96.5 | 89.9 |

Korean is the weakest language; even on **clean** images the 5 languages disagree
~10–15% of the time — a noise floor that already hurts a disagreement detector.

---

## Result 1 — H1 is REFUTED: a single-language attack transfers ~completely

STL-10, English-only PGD-20, robust accuracy (%):

| ε/255 | en | ko | es | fr | ja | ensemble | all-agree |
|---|---|---|---|---|---|---|---|
| clean | 96.6 | 89.1 | 96.8 | 96.3 | 96.3 | 96.5 | 89.9 |
| 2 | 0.2 | 1.0 | 0.6 | 0.5 | 0.5 | **0.4** | 93.6 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 95.5 |
| 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 94.3 |
| 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 93.7 |

CIFAR-10 is identical in character. **Transfer fraction** (drop on a non-targeted
language ÷ drop on English) ≈ **0.90–1.01** for every language at every ε.

**Low-ε sweep (CIFAR-10, PGD-40)** — the last chance for a "partial" regime:

| ε/255 | ensemble acc | transfer fraction (ko / es / fr / ja) |
|---|---|---|
| 0.25 | 14.1 | 0.84 / 0.97 / 0.99 / 0.98 |
| 0.5 | 2.7 | 0.89 / 0.99 / 1.00 / 0.99 |
| 1.0 | 0.2 | 0.90 / 1.00 / 1.01 / 1.00 |

H1 predicted *"transfer is partial, rising with ε."* Reality: transfer is
**near-complete and flat from ε=0.25/255 upward** — it never rises because it
starts saturated. Korean is the only (tiny) outlier, never a usable signal.

## Result 2 — the mechanism (why it must be so)

Cross-lingual label-embedding geometry:

* mean **same-class cross-lingual cosine** = **0.91**
* mean **different-class within-English cosine** = **0.79**

i.e. *"cat" in EN/KO/ES/FR/JA are more similar to each other (0.91) than English
"cat" is to English "dog" (0.79).* There is one shared image embedding; the
gradient that lowers the English-"cat" score points in essentially the same
direction as the gradient that lowers every other language's "cat" score. A
language-specific image attack is therefore **geometrically impossible** — fooling
one language fools all of them, and they **agree on the wrong class**.

## Result 3 — both training-free defenses fail

* **Multilingual ensemble:** robust accuracy **0.0–0.4%** at ε≥2 (table above).
  It averages five scores that have all been pushed to the same wrong class.
* **Disagreement detector:** all-language **agreement *increases* under attack**
  (85→93% CIFAR, 90→94% STL). ROC-AUC of disagreement as an adversarial-image
  detector is **below 0.5** (mean-JS: 0.22–0.37; vote-based: 0.45–0.48) on both
  datasets — **worse than random**, because adversarial images are *more*
  consensual than clean ones.

## Result 4 — H2 is REFUTED: attacker cost does not grow with #languages

The ensemble is defeated at ε≈0.5/255 by attacking **English alone**. Targeting
more languages cannot raise the attacker's required budget because one language
already suffices (see Result 1). The proposed "attacker-cost curve" is flat at the
floor. Confirmed by attacking all five languages at once (CIFAR-10), which is no
more effective than English alone:

| ε/255 | ensemble acc, attack EN only | ensemble acc, attack all 5 |
|---|---|---|
| 0.25 | 14.1 | 14.7 |
| 0.5 | 2.7 | 3.4 |
| 1.0 | 0.2 | 0.4 |
| 2.0 | 0.0 | 0.0 |

(Attacking all five even drives agreement *higher* — 98% at ε=4 — since the attack
explicitly aligns the languages on the wrong class.)

## Result 5 — the denoiser ("consensus purification", the main contribution)

A small residual DnCNN, trained self-supervised: generate English-PGD adversarial
examples, train the purified image's per-language predictions to match the clean
consensus pseudo-label + L2 fidelity. Evaluated with ensemble accuracy. The
adaptive attack is an **exact white-box attack through the differentiable denoiser**
(not BPDA): PGD on `classify(purify(x))`.

**STL-10** (primary; well-behaved):

| ε/255 | no defense | denoised, non-adaptive | denoised, ADAPTIVE |
|---|---|---|---|
| clean | 96.0 | 85.8 | — |
| 2 | 0.0 | 82.2 | **2.0** |
| 4 | 0.0 | 80.6 | **1.0** |
| 8 | 0.0 | 77.4 | **0.4** |
| 16 | 0.0 | 66.6 | **0.0** |

This matches the proposal's own H3: substantial **non-adaptive** recovery (82%),
**~total collapse under the adaptive attack (→0%)**. CIFAR-10 behaves the same
adaptively (0.0% at every ε); its non-adaptive recovery is messier (clean drops to
48% — a low-resolution/tuning artifact, fixed in the improved run).

**The crucial point:** this is *generic* adversarial purification (Athalye et al.
2018; Nie et al. 2022 — both cited by the proposal), not anything "multilingual."
The "consensus" provides no signal, because adversarial images already have high
consensus. **Ablation** — identical denoiser trained on **all-5 consensus** vs
**English only** (STL-10, with clean-preservation training):

| metric | consensus (5 langs) | English only |
|---|---|---|
| clean, denoised | 92.2 | 91.4 |
| ε=2 non-adaptive | 87.6 | 81.8 |
| ε=8 non-adaptive | 77.4 | 73.8 |
| ε=16 non-adaptive | 65.4 | 61.0 |
| ε=2 **adaptive** | **0.4** | **2.2** |
| ε=8 **adaptive** | **0.0** | **0.6** |
| ε=16 **adaptive** | **0.2** | **0.2** |

The two are nearly identical. The all-5 denoiser is ~4–6 pts better non-adaptively
— a mild regularization effect from training against five label sets, **not** the
claimed "consensus restoration." Under the adaptive attack **both collapse to ~0%.**
A single language does the same job: the "multilingual consensus" adds nothing.

---

## Verdict

The defense rests on cross-lingual **disagreement** under a language-specific
attack. With a shared image encoder and cross-lingually aligned text embeddings,
that disagreement **does not exist** — attacking one language's labels produces a
single perturbed image embedding on which all languages **agree (on the wrong
class)**. The proposal's own out-of-scope category ("language-agnostic attacks,
which preserve consensus") in fact **describes what the in-scope single-language
attack actually does.** The distinction the proposal is built on collapses
empirically.

### Confirmed on the literal FreddeFrallan M-CLIP named in the proposal
`M-CLIP/XLM-Roberta-Large-Vit-B-32` (frozen OpenAI CLIP ViT-B/32 image encoder +
XLM-R text encoder distilled to the English teacher space), STL-10, English-only PGD:

| ε/255 | en | ko | es | fr | ja | agreement |
|---|---|---|---|---|---|---|
| clean | 97.2 | 81.4 | 97.0 | 97.2 | 96.6 | 82.2 |
| 2 | 0.2 | 1.6 | 0.4 | 0.4 | 0.4 | 86.0 |
| 4 | 0.0 | 1.2 | 0.0 | 0.0 | 0.0 | 90.0 |
| 8 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 91.8 |

Identical behavior: full transfer, agreement *rising* under attack. The negative
result holds on **both** the jointly-trained open_clip model and the distillation-
based M-CLIP. (Distillation to one English teacher makes languages even more
aligned, so this was expected.)

### What *could* be salvaged
Only attacks that are language-specific on the **text side** (perturbing tokens /
prompts per language) could create genuine disagreement — but the proposal's
threat model is image-space adversarial examples, where a shared encoder forbids it.
