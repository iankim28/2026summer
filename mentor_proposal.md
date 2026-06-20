# Proposal: Dual Language-Specific Encoder Pairs for Adversarial Robustness in Multilingual CLIP

**Date:** June 19, 2026
**Notebook:** `notebooks/dual_encoder_divergence.ipynb`
**Estimated runtime:** ~80 minutes (RTX 5070 Ti)

---

## Background

A recent research proposal hypothesised that labelling an image simultaneously in five
languages (English, Korean, Spanish, French, Japanese) with a multilingual CLIP model
would provide adversarial robustness. The intuition: an attack crafted to fool the
English classifier would leave the other languages unaffected, and this cross-language
*disagreement* could be used to detect or purify adversarial examples.

We tested this hypothesis on the standard model (`xlm-roberta-base-ViT-B-32`, pretrained
on LAION-5B) using CIFAR-10 as the evaluation dataset. The core experiment is
straightforward: craft a PGD attack targeting only the English classifier and measure
what fraction of attacked images the other languages still classify correctly — the
**retention rate**. The defence requires retention above 50% to be useful. An attack
that fools all languages simultaneously (retention ~0%) means the defence cannot work.

**Q1 result (confirmed June 16):** the attack transfers completely. At every tested
budget (ε = 0.5 to 8 / 255), all five languages collapse to near-zero accuracy
simultaneously. The best observed retention was 3.2% (Spanish, ε = 0.5), far below
the 50% threshold.

**Why this happens** is geometric. In this model, a single shared ViT image encoder
maps every input to one 512-dimensional embedding. All five languages then score that
single embedding against their respective text anchors. The same-class cross-lingual
cosine similarity between text anchors is 0.914 — higher than the within-English
different-class cosine of 0.792. This means the language boundaries in embedding space
are weaker than the class boundaries. Any gradient step that moves the image embedding
away from the correct English class simultaneously moves it away from the correct class
in every other language. The attack is not language-specific; it is class-specific, and
class membership is shared across languages.

---

## What We Have Tried (Q2 Work)

The root cause being structural, we ran a series of experiments aimed at making the
image embeddings *language-specific* — so that a gradient computed against the English
text anchors would fail to transfer to Korean, Spanish, French, or Japanese.

All experiments used the same training objective: a combination of a classification
loss (keep accuracy on clean images), an orthogonalisation penalty (push
language-specific embeddings apart), and an adversarial retention loss (penalise
EN-attack fooling non-EN languages). All backbones were frozen; only small adapter
modules were trained.

**Experiment A — Rank-8 output-projection LoRA (~41K params):**
Added small per-language residual adapters at the final output of the image encoder.
Best KO retention: 8.7% at ε = 8. A small real improvement but far from 50%.

**Experiment B — Rank-64 output-projection LoRA (~327K params):**
Increased adapter capacity. Best KO retention: 29% at ε = 8. The best clean result so
far. Still 21 points below the target.

**Experiment C — Multi-layer LoRA inside the ViT (rank-16, layers 6/8/10/11):**
Injected adapters into the ViT's CLS token at intermediate transformer blocks to
influence representations earlier in the pipeline. Training collapsed — learning rate
1e-3 was too aggressive and all languages converged to ~10% accuracy (random chance).
Results uninformative.

**Experiment D — Text-tower orthogonalisation:**
Fine-tuned the shared text projection (`model.text.proj`) to reduce cross-lingual
same-class cosine from 0.914 to 0.313. Fast (3 seconds). The evaluation was confounded
by the collapsed image adapters from Experiment C.

**Experiment E — Full per-language ViT encoders:**
Created five independent deep-copies of the full 87.8M-parameter ViT, one per language,
and fine-tuned each for 5 epochs. Clean accuracy improved substantially (Korean:
83.6% → 94.6%). Adversarial retention: 0% at every budget. Five epochs were
insufficient to break the symmetry of LAION-5B pretraining — all five copies started
from identical weights and learned nearly identical representations.

**Experiment F — Text orthogonalisation in isolation (June 19):**
Tested the orthogonalised text embeddings with the base frozen image encoder. Best KO
retention: 12.5% at ε = 8. Better than collapsed Experiment C but worse than rank-64
LoRA, and well below 50%.

**Summary leaderboard (KO retention at ε = 8/255):**

| Method | KO retention |
|---|---|
| No defence (baseline) | ~86% (full transfer) |
| Rank-8 output-proj LoRA | 8.7% |
| TXT\_ORTH isolated | 12.5% |
| Rank-64 output-proj LoRA | **29.0%** ← best clean result |
| Multi-layer LoRA r16 | 0% (training collapsed) |
| Per-language ViT (5 epochs) | 0% (undertrained) |
| **Target** | **> 50%** |

---

## Why the Prior Approaches Hit a Ceiling

The experiments above share a hidden structural problem that explains the 29% ceiling.

Experiments A, B, and C trained per-language image adapters trying to push the
embeddings `z_en`, `z_ko`, etc. apart. But the text embeddings they had to align to
were **shared**: a single `model.text.proj` produced all text anchors in the same
space. This created a direct conflict between the two training objectives:

- The **image divergence loss** pushes the image embeddings apart.
- The **classification loss** pulls each image embedding toward the shared text anchor
  for its correct class — and since those anchors are nearly co-located across
  languages (cosine ~0.914, or even ~0.313 after orthogonalisation), this acts as a
  gravitational force pulling all image embeddings back together.

The adapters partially diverge, the classification loss re-centres them, and the
training reaches an equilibrium well below the 50% target. No amount of additional
LoRA capacity can escape this — the shared text anchors impose a geometric constraint
that the image adapters cannot overcome.

Experiment F confirmed this reading: even after dramatically separating the text
anchors (cosine 0.914 → 0.313), the shared image encoder still maps every input to a
single point that a single gradient step can push away from all language clusters at
once. The problem is not the *separation* of the text clusters; it is the *singularity*
of the image embedding.

---

## Proposed Approach: Dual Language-Specific Encoder Pairs

The fix is to make both the image *and* text sides language-specific simultaneously,
so the two losses reinforce each other instead of fighting.

### Architecture

Each of the five languages gets its own encoder *pair*:

```
(ViT + LoRA_en,  TextHead_en)   →   (z_en,  t_en)   EN CLIP space
(ViT + LoRA_ko,  TextHead_ko)   →   (z_ko,  t_ko)   KO CLIP space
(ViT + LoRA_es,  TextHead_es)   →   (z_es,  t_es)   ES CLIP space
(ViT + LoRA_fr,  TextHead_fr)   →   (z_fr,  t_fr)   FR CLIP space
(ViT + LoRA_ja,  TextHead_ja)   →   (z_ja,  t_ja)   JA CLIP space
```

All pairs share the same ambient R^512 space (so embeddings are geometrically
comparable) but are trained to occupy distinct regions of that space.

**Image side — per-language LoRA at all 12 ViT blocks:**
The shared ViT-B/32 backbone is frozen. Per-language rank-32 residual adapters
(`IntermediateLoRA`) are injected into the CLS token after every transformer block.
This gives the language adapters access to representations at every depth of the ViT,
not just the final output.

- Parameters: 5 languages × 12 blocks × 2 × (768 × 32) ≈ **1.18M**
- Backbone: 87.8M frozen

**Text side — per-language projection heads:**
The shared XLM-RoBERTa backbone is frozen. Instead of the single shared `model.text.proj`,
each language gets its own `nn.Linear(640, 512)` projection head that maps
XLM-RoBERTa's 640-dimensional pooled output into CLIP's 512-dimensional space.

- Parameters: 5 × (640 × 512) ≈ **1.64M**

**Total trainable: ~2.8M parameters.** The 87.8M-parameter backbone is entirely frozen.

### Loss Function

Training minimises a three-term objective over a batch of (image, label) pairs:

```
L = Σ_l  CE( f_l(x) · g_l^T, y )                        (1) classification

  + λ_div  Σ_{l ≠ l'}  cos( f_l(x), f_{l'}(x) )          (2) image divergence

  + λ_ret  Σ_{l ≠ en}  CE( f_l(x_adv) · g_l^T, y )       (3) adversarial retention

  where  x_adv = PGD( x, y ;  f_en, g_en )
```

**Term 1** keeps each language pair accurate on clean images. Each pair is trained
independently against its own text heads, so the five pairs can freely occupy different
regions of R^512.

**Term 2** is the critical new addition. It explicitly penalises the cosine similarity
between image embeddings that different language encoders produce for the same image,
directly optimising for the cross-language geometric separation we need.

**Term 3** is the adversarial retention objective: after PGD attack computed against
the English pair `(f_en, g_en)`, the other four language pairs must still classify
correctly. This is the direct training signal for the defence capability.

**Why the two key tensions are resolved:**

With per-language text heads, `g_en ≠ g_ko` and so the classification loss for each
language pulls its image encoder toward a *different* target. The natural equilibrium
of term 1 alone is already `z_en ≠ z_ko`. Term 2 then pushes this divergence further
and faster. The two losses now work in the same direction.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank | 32 | Between Exp C (16, collapsed) and Exp B (64, best result) |
| Learning rate | 1e-4 | Exp C collapsed at 1e-3; 10× reduction ensures stable training |
| λ\_div | 1.0 | Equal weight to divergence and classification |
| λ\_ret | 1.0 | Equal weight to retention loss |
| Epochs | 15 | 3× Exp C; enough for loss curves to stabilise |
| PGD steps (train) | 7 | Fast inner loop; consistent with prior experiments |
| Training images | 1,000 | Same as all prior experiments |
| Eval images | 300 | Same tloader as all prior experiments |

---

## Expected Outcomes

**If retention > 50%:** the dual language-specific encoder architecture successfully
breaks the cross-lingual attack transfer. This would be the first result in this
project to clear the defence threshold, and would directly justify scaling to a larger
model or longer training.

**If retention is 20–50%:** improvement over the 29% baseline confirms that joint
language-specific encoding is the right direction, but the 15-epoch budget or rank-32
capacity is insufficient. The natural follow-up is extending training or increasing
rank.

**If retention remains ~0%:** either (a) 15 epochs is too few for the adapters to
diverge from their shared LAION-5B initialisation — rerun with 50+ epochs on an A100,
or (b) the LoRA capacity injected at the CLS token is insufficient to alter the
internal ViT representations deeply enough — revisit full per-language ViTs
(Experiment E) with longer training.

**Clean accuracy** is expected to be preserved at ≥ 90% for all languages. If it
drops below 85%, the λ_div term is dominating and λ_div should be reduced to 0.5.

---

## Computational Plan

- **Estimated runtime:** ~80 minutes on RTX 5070 Ti (based on Experiment C timing:
  ~320s/epoch × 15 epochs; added LoRA layers and text heads add <5% overhead)
- **GPU memory:** ~2–3 GB for adapters + frozen backbone + PGD graph; well within
  the RTX 5070 Ti's 16 GB
- **Notebook:** `notebooks/dual_encoder_divergence.ipynb` (self-contained; does not
  depend on any prior session state from the existing notebook)
- **Monitoring:** loss should decrease steadily after epoch 1; if loss diverges or
  clean accuracy drops below 50% by epoch 3, stop and reduce LR to 5e-5
