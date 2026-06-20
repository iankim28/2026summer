## 2026-06-16

Ran Q1 deep dive (Section 11 of updated_multilingual_consensus_colab.ipynb).

Result: an English-only image-space PGD attack transfers completely to all other languages
(KO, ES, FR, JA). Retention rate across non-English languages was at most 3.2% (ES, ε=0.5),
far below the 50% threshold the defence requires.

Mechanism: same-class cross-lingual cosine similarity (0.914) exceeds different-class
within-English cosine (0.792), so any gradient that fools English class-c simultaneously
fools all other languages' class-c.

Text-side positive control confirmed disagreement is architecturally possible — but only
under a text-manipulation threat model (87–100% retention), not the standard image-space
threat model.

Conclusion: the multilingual consensus defence is ineffective against standard image-space
attacks. Q2 (disagreement detector) and Q3 (denoiser) remain open.

## 2026-06-17

Ran Experiments C, D, E (Sections 15–17 of updated_multilingual_consensus_colab.ipynb).

**Experiment C — Multi-layer LoRA (rank-16, injected at ViT blocks 6/8/10/11):**
Training diverged. LR=1e-3 caused the model to collapse to ~10% accuracy (random chance).
Retention 0% — results uninformative. Fix: re-run with ML_LR=1e-4.

**Experiment D — Text-tower orthogonalisation (fine-tune model.text.proj):**
Cross-lingual text cosine reduced from 0.914 → 0.313 (reduction of 0.602) in 3 seconds.
Combined C+D retention shows ~14% (KO/FR), but this is confounded by the collapsed
Section 15 image adapters. Text-side effect is untested in isolation — next step is to
evaluate base encode_image + TXT_ORTH retention.

**Experiment E — Full per-language ViT encoders (5 × deepcopy, 5 epochs, LR=1e-5):**
Clean accuracy improved dramatically (KO: 83.6%→94.6%, others +1 pp). Adversarial
retention: 0% at every ε — 5 epochs insufficient to break pretrained LAION-5B symmetry.
The architectural idea is sound but requires 10–50× more training to manifest.

**Best defence so far:** rank-64 output-projection LoRA (Section 14), up to 29% retention at ε=8.
**Most promising untested direction:** base encoder + TXT_ORTH (text orthogonalisation in isolation).

## 2026-06-19 — Meeting prep: full analysis of Q2 work

---

### TL;DR

We started by confirming that the multilingual consensus defence completely fails
because a single shared image encoder makes all five languages move together under
attack (Q1). We then ran six experiments trying to break that coupling — from
lightweight LoRA adapters up to full per-language image encoders and text
orthogonalisation — achieving at best 29% adversarial retention (rank-64 LoRA) against
a 50% target. Every experiment points to the same bottleneck: as long as the image
encoder is shared, text-side or shallow output-projection fixes give diminishing
returns. Real progress requires either much longer training of fully independent image
encoders, or a fundamentally different architecture.

---

### 1. Background and Goal

The project tests the "multilingual consensus defence": the idea that labelling an
image simultaneously in five languages (EN, KO, ES, FR, JA) with a multilingual CLIP
model provides adversarial robustness. If an English-only PGD attack only fools the
English labels while the others remain correct, the cross-language disagreement
becomes a detection signal.

**Q1 (answered 2026-06-16):** We measured whether an English-only image-space PGD
attack stays confined to English. It does not. At every tested budget (ε = 0.5–8/255),
all five languages collapse to near-zero accuracy simultaneously. The "retention rate"
— the fraction of EN-fooled images where other languages remain correct — was at most
3.2% (ES at ε = 0.5), far below the 50% threshold the defence requires to be useful.

**Why:** The same-class cross-lingual cosine similarity between text embeddings is
0.914 (e.g., the English "airplane" and Korean "비행기" embeddings point in nearly the
same direction). This exceeds the within-language different-class cosine (~0.792),
meaning the language boundaries in embedding space are weaker than the class
boundaries. Any gradient step that moves the image embedding away from class c in
English simultaneously moves it away from class c in every other language.

**Q2 goal (current work):** Can we modify the model so that the image embedding
becomes language-specific enough that a gradient computed against English text fails
to fool Korean, Spanish, French, and Japanese?

**Model:** `xlm-roberta-base-ViT-B-32` (multilingual CLIP, pretrained on LAION-5B).
Architecture: a shared ViT-B/32 image encoder (87.8M parameters) producing a 512-d
embedding, and an XLM-RoBERTa text encoder with a 640×768 → 640×512 MLP projection
(`model.text.proj`) that maps multilingual text into the same 512-d space.
**Dataset:** CIFAR-10, 300 test images for adversarial evaluation, 1000 for training.
**Attack:** L∞ PGD (20 steps, α = 2.5ε/steps), English-only target.

---

### 2. Approach: Six Experiments

The root cause is structural — the shared image encoder — so all experiments aim to
introduce language-specific divergence either in the image embedding path or in the
text embedding path.

---

#### Experiment A — Rank-8 output-projection LoRA (Section 12)

**What:** Added per-language low-rank adapter matrices (LoRA, rank 8) at the final
output projection of the image encoder. Each language gets a trainable `A ∈ R^{512×8}`
and `B ∈ R^{8×512}` that adds a residual `BAx` to the shared 512-d image embedding.
~41K trainable parameters. Trained for 5 epochs with a three-term loss:
classification loss (CE) + cross-language orthogonalisation penalty (λ=0.5, pushes
per-language image embeddings apart) + adversarial retention loss (λ=1.0, penalises
EN attack fooling non-EN languages).

**Results:**

| ε | KO retention | ES | FR | JA |
|---|---|---|---|---|
| 0.5 | 2.1% | 4.9% | 5.1% | 3.0% |
| 4   | 11.9% | 6.7% | 12.4% | 6.4% |
| 8   | 8.7% | 5.7% | 8.4% | 5.3% |

**Assessment:** Small but real improvement over 0% baseline retention. The adapters
learn but rank-8 is too low-capacity to reshape the embedding geometry meaningfully.

---

#### Experiment B — Rank-64 output-projection LoRA (Section 14)

**What:** Same architecture as A but rank 64 (8× more capacity, ~327K parameters).
Same training setup.

**Results:**

| ε | KO retention | ES | FR | JA |
|---|---|---|---|---|
| 0.5 | 0.3% | 0.3% | 0.3% | 0.3% |
| 2   | 10.5% | 9.9% | 11.1% | 7.8% |
| 4   | 24.5% | 17.0% | 23.4% | 17.5% |
| 8   | **29.0%** | 14.4% | 24.8% | 16.5% |

**Assessment:** Best result so far. At large budgets (ε=8), up to 29% of EN-fooled
images are correctly classified in KO. Still 21 points below the 50% target. The
output-projection LoRA can shift the final embedding but cannot undo the shared
spatial representations computed deep inside the ViT.

---

#### Experiment C — Multi-layer LoRA inside ViT (Section 15)

**What:** Instead of adapting the final output, we injected per-language rank-16
residual adapters into the CLS token at four intermediate transformer blocks (layers
6, 8, 10, 11 of the 12-block ViT). This gives the adapter access to the intermediate
representations before they are fully computed. ~491K trainable parameters.

**Results:** Training collapsed. Learning rate LR=1e-3 was too aggressive for
in-block injection — all languages converged to ~10% accuracy (random chance on
CIFAR-10). Retention was 0% or artifactual. Results are uninformative.

**Assessment:** The idea is sound but requires LR ≈ 1e-4. Needs a re-run.

---

#### Experiment D — Text-tower orthogonalisation (Section 16)

**What:** Instead of the image side, we fine-tuned `model.text.proj` (the MLP
projecting XLM-RoBERTa's 768-d output to CLIP's 512-d space) with a CE +
orthogonalisation loss that minimises the mean pairwise cosine similarity between
same-class text embeddings across language pairs. 20 epochs, LR=1e-4, takes 3
seconds.

**Results (text embeddings):** Cross-lingual same-class cosine dropped from 0.914
to 0.313 — a reduction of 0.602. This is a large geometric separation.

**Combined C+D retention:** ~14% (KO/FR). But this result is confounded: the
evaluation used the collapsed Experiment C image adapters (which achieve random
accuracy), so the image side was broken.

**Assessment:** Text orth is fast and dramatically reshapes the text space. Effect on
retention is untested in isolation until Experiment F (below).

---

#### Experiment E — Full per-language ViT encoders (Section 17)

**What:** Created 5 independent deep-copies of the full 87.8M-parameter ViT image
encoder (one per language, 1.76GB total). Each copy was fine-tuned independently for
5 epochs with CE + adversarial retention loss, LR=1e-5. Total training time: 414s.

**Clean accuracy results:**
EN 95.4%, KO 94.6% (+11pp from pretrained baseline 83.6%), ES 95.4%, FR 95.8%, JA 95.6%.
The per-language fine-tuning dramatically improved zero-shot accuracy, especially for KO.

**Adversarial retention:**

| ε | KO | ES | FR | JA |
|---|---|---|---|---|
| 0.5–8 | 0% | 0% | 0% | 0% |

All retention is 0% across all budgets and languages. The attack on the EN ViT fools all ViTs.

**Assessment:** The architecture is correct in principle. Five independent encoders
should diverge if trained long enough. But 5 epochs of fine-tuning is insufficient to
overcome the deeply ingrained symmetry from LAION-5B pretraining — all five copies
started from the same weights and learned nearly identical representations in just 5
epochs. Estimate 50–100 epochs needed for meaningful divergence.

---

#### Experiment F — Text orthogonalisation in isolation (Section 18, today)

**What:** The cleanest possible test of the text-only approach: frozen base image
encoder (`encode_image`, no adapters) + `TXT_ORTH` (the orthogonalised text embeddings
from Experiment D). The attack was computed against `TXT_ORTH["en"]` to make the
threat model realistic (attacker knows the defence is deployed).

**Results:**

| ε | EN fooled | KO | ES | FR | JA |
|---|---|---|---|---|---|
| 0.5 | 287 | 0.0% | 0.0% | 0.0% | 0.0% |
| 1   | 287 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2   | 287 | 1.7% | 0.0% | 0.0% | 0.0% |
| 4   | 287 | 2.8% | 0.0% | 0.0% | 0.0% |
| 8   | 287 | 12.5% | 0.0% | 6.3% | 4.2% |

Clean accuracy fully intact: EN 95.7%, KO 94.0%, ES 96.0%, FR 94.3%, JA 95.0%.

**Assessment:** Text orthogonalisation in isolation is insufficient. Even with
substantially more separated text embeddings, the best retention is 12.5% (KO at ε=8)
— worse than rank-64 LoRA. The image-space gradient computed against any one language's
text cluster moves the shared image embedding in a direction that is simultaneously
adversarial to all other language clusters, regardless of how spread out those clusters
are.

---

### 3. Overall Analysis

**The core result across all six experiments:** no intervention that leaves the image
encoder shared achieves over 29% retention. The rank-64 output-projection LoRA is the
best, and it operates as close to the image encoder output as possible without
touching the internal representations.

**What these results say about the architecture:**

The image encoder maps each input image to a single point in R^512. PGD computes a
gradient of the classification loss with respect to the input pixels and steps in that
gradient direction. The gradient is fundamentally a function of the image point's
position relative to all language text clusters in R^512. As long as those clusters
are approximately co-located (cosine ~0.914) or even after orthogonalisation (cosine
~0.313), the gradient step that moves the image point away from EN class-c also moves
it away from KO/ES/FR/JA class-c. The retention rate measures how often this fails —
and it almost never fails.

**Why LoRA at the output helps partially:** the per-language output projection
`Bᵢ Aᵢ x` adds a language-specific offset to the shared embedding. This means the
five languages see slightly different "views" of the shared representation. A gradient
targeting the EN view does not perfectly align with the other views. The larger the
rank (more capacity), the more the views diverge, and the more the EN gradient
misaligns — hence rank-64 > rank-8. But the fundamental point `x` (the ViT's output)
is still shared, so the divergence is limited.

**Why text orth doesn't help much:** after orthogonalisation, the text clusters are
spread further apart, but the image encoder still maps any given input to a single
shared point. PGD adapts to the new text geometry and finds a direction that crosses
all five language class boundaries simultaneously.

**Why full per-language ViT should work (given enough training):** with 5 entirely
independent ViTs, each language's gradient computation is fully decoupled. An attack
optimised against ViT_en moves ViT_en's output, but ViT_ko, ViT_es, etc. are
unaffected — they process the same perturbed pixels through entirely different weights
and (if trained long enough to diverge) produce different embeddings. 5 epochs was
insufficient because all five ViTs started from the same pretrained weights and did
not diverge enough.

---

### 4. Leaderboard (KO retention at ε = 8/255)

| Method | KO retention | Status |
|---|---|---|
| No defence (baseline) | ~86% (full transfer) | — |
| Rank-8 output-proj LoRA | 8.7% | undertrained direction |
| TXT\_ORTH isolated | 12.5% | text-only ceiling |
| Combined C+D (confounded) | ~14% | not reliable |
| Rank-64 output-proj LoRA | **29.0%** | best clean result |
| Multi-layer LoRA r16 | 0% | training diverged (LR too high) |
| Per-language ViT (5 ep) | 0% | undertrained |
| **Target** | **> 50%** | |

---

### 5. Recommended Next Steps

1. **Re-run Experiment C** with `ML_LR = 1e-4` (simple fix, ~30 min). Multi-layer
   LoRA injected at ViT blocks 6/8/10/11 could outperform output-projection LoRA if
   the training is stable.

2. **Scale Experiment E (full per-language ViT)** to 50–100 epochs on Colab A100
   (estimate 2–4 hours). This is the only approach with a sound theoretical reason to
   exceed 50% retention; it just needs sufficient training to break the pretrained
   symmetry.

3. **Combine B + E**: use rank-64 LoRA as a lightweight upper stage on top of
   diverged per-language ViTs. The LoRA already contributes ~29% on its own; combined
   with genuinely diverged image representations, it may push well past 50%.

---

## 2026-06-19 (evening) — Experiment G: Dual language-specific encoder pairs

**Notebook:** `notebooks/dual_encoder_divergence.ipynb`
**Hardware:** RTX 5070 Ti (local), 282s training time.

### Setup

Per-language LoRA adapters at every one of the 12 ViT transformer blocks (rank-32,
applied to the CLS token in NLD layout — the critical fix over Experiment C which
used the wrong LND layout) plus per-language text projection heads
(`nn.Linear(768, 512, bias=False)`) bypassing the frozen shared `model.text.proj`.
Total trainable parameters: 4.92M (2.95M image, 1.97M text). Frozen backbone: 366M.

Three-term loss:
1. Classification CE per language (clean images)
2. Image divergence: penalises cosine similarity between per-language image embeddings (λ=1.0)
3. Adversarial retention: CE for non-EN languages under EN-targeted PGD at ε=2/255 (λ=1.0)

Training: 1000 CIFAR-10 train images, BATCH=64, LR=1e-4, 15 epochs, inner PGD 7 steps.

### Results

**Clean accuracy (n=500):** EN 92.6%, KO 88.8%, ES 91.4%, FR 87.8%, JA 89.6%. No
degradation vs. the pretrained baseline.

**Adversarial retention (PGD 20 steps vs. EN pair, n=300):**

| ε | EN fooled | KO | ES | FR | JA |
|---|---|---|---|---|---|
| 0.5 | 276 | 72.8% | 68.8% | 72.8% | 73.9% |
| 1   | 279 | 75.6% | 73.5% | 69.9% | 77.8% |
| 2   | 279 | **79.6%** | **80.3%** | **74.6%** | **82.1%** |
| 4   | 279 | 69.5% | 73.1% | 69.9% | 70.6% |
| 8   | 279 | 44.8% | 35.8% | 44.8% | 39.8% |

### Analysis

**This is the first experiment to exceed the 50% retention target.** At ε ≤ 4/255,
all four non-EN languages are above 50% — peaking at ε=2 where JA reaches 82.1% and
ES reaches 80.3%. The dual-encoder design works as intended.

**Training dynamics confirm the mechanism.** The divergence loss (initially 0.927)
crossed zero and reached −0.106 by epoch 15, meaning the language-specific image
embeddings are now anti-correlated rather than merely orthogonal. The CE loss
converged cleanly from 2.44 to 0.11. This validates the theoretical prediction: when
both image and text sides are language-specific, the CE and divergence losses
reinforce each other instead of fighting.

**The ε=8 gap is explained by training budget.** The model was trained with
`DUAL_TRAIN_EPS = 2/255`. At ε=8/255, the PGD attack moves the image far enough in
pixel space to overcome the learned divergence — KO and FR drop to 44.8%, ES to
35.8%, JA to 39.8%. These are still 1.5× better than the previous best (rank-64
LoRA, 29% KO at ε=8) but below the 50% target. The fix is straightforward: retrain
with a larger training epsilon.

**Why this approach works where others failed:** previous experiments (A–F) always
left either the image or text side shared. With a shared image encoder, any EN
gradient also moves the shared representation used by all other languages. With shared
text projections, the classification and divergence losses pointed in opposite
directions, limiting how far the adapters could push embeddings apart. Experiment G
makes both sides language-specific simultaneously, so the two losses cooperate.

### Updated Leaderboard (KO retention at ε = 8/255)

| Method | KO retention | Notes |
|---|---|---|
| No defence (baseline) | ~86% (full transfer) | — |
| Rank-8 output-proj LoRA | 8.7% | too low capacity |
| TXT\_ORTH isolated | 12.5% | text-only ceiling |
| Combined C+D (confounded) | ~14% | collapsed image adapters |
| Rank-64 output-proj LoRA | 29.0% | previous best |
| Multi-layer LoRA r16 | 0% | training diverged (LR too high) |
| Per-language ViT (5 ep) | 0% | undertrained |
| **Dual enc. G (train ε=2)** | **44.8%** | **new best; >50% at ε≤4** |
| **Target** | **> 50%** | |

### Next Steps

1. **Increase training epsilon** to 4/255 or 8/255 in `dual_encoder_divergence.ipynb`
   (change `DUAL_TRAIN_EPS`) and re-run. Expect ε=8 retention to cross 50%.

2. **More epochs**: 15 epochs on 1000 samples is relatively light. 30 epochs or
   expanding to 5000 training samples may further improve all retention values.

3. **Discuss with mentor**: the 50% threshold is now crossed at moderate ε. The
   next question is whether the defence is practical — does it hold at ε=8, and what
   is the computational cost of the dual encoder at inference time?

---

## 2026-06-19 (night) — Typographic attack confusion matrices (separate per-language CLIPs)

**Notebook:** `notebooks/typographic_attack_confusion.ipynb`
**Results:** `notebooks/results/confusion_results.json`, `typographic_heatmaps.png`, `confusion_matrices.png`
**Dataset:** STL-10 test split, 200 random images (seed 0)
**Attack:** typographic — write the adversarial target class name onto the image (no
gradients). Four attack languages (EN, ZH, KO, JA) × four independent classifiers.

### Setup

Four **independently pretrained** per-language CLIP models (not a shared encoder):

| lang | model |
|---|---|
| en | OpenAI ViT-B/32 (`open_clip`) |
| zh | `OFA-Sys/chinese-clip-vit-base-patch16` |
| ko | `Bingsu/clip-vit-base-patch32-ko` |
| ja | `line-corporation/clip-japanese-base` (CLYP, trust_remote_code) |

Each image gets a random adversarial target class (≠ true class). For each attack
language, the target class name is rendered in that language at the bottom of the
image. Metrics: **accuracy** (pred == true), **ASR** (pred == written target class).

### Clean baseline (unattacked)

| model | clean accuracy |
|---|---|
| en | 98.5% |
| zh | 97.0% |
| ko | 98.5% |
| ja | **14.0%** |

EN/ZH/KO are healthy. **JA clean accuracy is broken** (likely tokenizer / transformers
5.x compatibility issues during this run). All conclusions involving `model_ja` should
be treated as provisional until JA loading is fixed and the notebook re-run.

### Accuracy matrix under attack (rows = attack language, cols = model)

|  | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| **attack_en** | 21.0% | 59.5% | 29.0% | 16.5% |
| **attack_zh** | 97.5% | 84.0% | 98.5% | 18.5% |
| **attack_ko** | 96.5% | 97.5% | 97.5% | 16.5% |
| **attack_ja** | 97.0% | 89.5% | 97.5% | 17.0% |

### ASR matrix (attack success — pred == written class)

|  | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| **attack_en** | **79.0%** | **39.5%** | **70.5%** | 5.0% |
| **attack_zh** | 0.0% | 14.5% | 0.0% | 4.5% |
| **attack_ko** | 0.5% | 0.0% | 0.0% | 5.0% |
| **attack_ja** | 0.5% | 7.5% | 0.5% | 5.0% |

Best attack per model (highest ASR in column): **English for all four models.**

Cropping sanity check (attack_en, `where="center"`): model_en center acc = 14.0% vs
bottom acc = 21.0% — attack still effective when centered, so results are not an
artefact of bottom-crop preprocessing alone.

### Conclusions

1. **English typographic attack dominates.** Writing an English target word fools the
   EN model (ASR 79%), KO model (70.5%), and ZH model (39.5%). This is the only attack
   language with high cross-model impact.

2. **Non-English script attacks do not transfer.** ZH/KO/JA attack rows keep EN/ZH/KO
   accuracy above 84% and ASR near zero on those models. A Chinese-script word on the
   image does not reliably fool the Korean or English classifier.

3. **Separate encoders disagree under English attack — unlike Q1.** Under `attack_en`,
   model_en accuracy (21%) ≠ model_zh (59.5%) ≠ model_ko (29%). With the shared
   multilingual CLIP (Q1), all languages collapsed to ~0% simultaneously and agreed
   on the wrong answer. Here, the models diverge, which is exactly the detection signal
   the consensus defence needs — at least for typographic (not PGD) attacks.

4. **No simple diagonal ASR pattern.** The strongest attack against every model is
   English, not "attack in language X fools model X." Script/language of the written
   word and model training language are not 1:1 coupled; Latin-script English text is
   visually salient across all encoders.

5. **JA column is unreliable in this run.** 14% clean accuracy means `model_ja`
   results (ASR ~5% under all attacks) reflect a broken baseline, not robustness.
   Re-run after fixing JA model loading before using JA in the ensemble.

### Connection to prior work

| prior result | typographic result (this run) |
|---|---|
| Q1 shared encoder + PGD: all languages collapse together | separate encoders + EN typographic: models disagree |
| Experiment G dual encoder + PGD: >50% retention at ε≤4 | typographic: EN word fools multiple models but not all equally |
| Consensus defence fails on shared architecture | disagreement-based **detection** may work with separate encoders under typographic threat |

### Next steps

1. Fix JA model (clean acc should be ~90%+) and re-run the full 4×4 grid.
2. Quantify **agreement rate** clean vs attacked (fraction of images where all models
   predict the same class) — the metric from `CODE_GUIDE_separate_langs_typographic.md`.
3. Test whether a **majority-vote or disagreement detector** on the four separate models
   flags typographic attacks with AUC > 0.5 (Q2 analogue for this threat model).
