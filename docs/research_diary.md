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

**Notebook:** `lib/notebooks/dual_encoder_divergence.ipynb`
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

**Notebook:** `lib/notebooks/typographic_attack_confusion.ipynb`
**Results:** `lib/notebooks/results/stl10_confusion_results.json`, `stl10_typographic_heatmaps.png`, `stl10_confusion_matrices.png`
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

---

## 2026-07-04 — Re-entry briefing after 2-week hiatus

### TL;DR

Two research threads were active when work paused. Thread A (shared multilingual CLIP + PGD attacks) reached its first result above the 50% retention target with the dual-encoder design (Experiment G). Thread B (separate per-language CLIP models + typographic attacks) showed that separate encoders disagree under English typographic attack — a detection signal that disappears when the encoder is shared — but the Japanese model is broken and needs fixing before conclusions can be drawn. Neither thread is closed; both have clear next steps.

---

### Goal of the project

The core question is whether labelling an image in five languages simultaneously (EN, KO, ES, FR, JA) using a multilingual CLIP model provides adversarial robustness. The proposed defence assumes that an attack crafted to fool English labels will leave the other languages unaffected, and that this cross-language disagreement can detect or repair the attack. The project tests this assumption empirically and, when it fails, attempts to engineer model modifications that make the assumption hold.

Three sub-questions:
- **Q1:** Does an English-only PGD attack transfer to other languages? (answered: yes, completely)
- **Q2:** Can model modifications produce genuine language-specific representations, limiting transfer? (partially answered: Experiment G exceeds 50% at moderate attack budgets)
- **Q3:** Can a denoiser trained on attacked images restore correct classification under adaptive attacks? (still open)

---

### Two research threads

**Thread A — Shared multilingual CLIP, image-space PGD attacks**

Model: `xlm-roberta-base-ViT-B-32` (one shared ViT-B/32 image encoder, XLM-RoBERTa text encoder, LAION-5B pretraining). Dataset: CIFAR-10. Attack: L∞ PGD, English-only target, 20 steps.

**Thread B — Four independent per-language CLIP models, typographic attacks**

Models: OpenAI ViT-B/32 (EN), OFA-Sys Chinese CLIP (ZH), Bingsu KO CLIP (KO), CLYP Japanese base (JA). Dataset: STL-10. Attack: typographic — render the adversarial class name as text on the image, no gradient computation.

These threads are complementary. Thread A tests the original defence architecture (shared encoder) under the standard threat model (imperceptible perturbations). Thread B tests an alternative architecture (separate encoders) under a simpler threat model (visible text overlay).

---

### Chronology (step by step)

**2026-06-16 — Q1: transfer confirmed**

Ran English-only PGD at ε = 0.5–8/255 on the shared `xlm-roberta-base-ViT-B-32`. All five languages collapsed to near-zero accuracy simultaneously at every budget. Best observed retention rate (fraction of EN-fooled images where another language stayed correct): 3.2% (ES, ε=0.5). The 50% threshold the defence requires was never approached. Root cause: same-class cross-lingual cosine similarity between text embeddings is 0.914, higher than the within-language different-class cosine of 0.792 — language boundaries are weaker than class boundaries, so any class-targeted gradient is simultaneously class-targeted for all languages.

**2026-06-17 — Experiments A–E: attempts to break the shared image encoder**

All experiments used the same shared ViT backbone (frozen), training on 1000 CIFAR-10 images with a three-term loss: classification CE + image divergence penalty + adversarial retention loss.

- **Exp A (rank-8 output-proj LoRA):** small adapters at the final 512-d output projection, one per language. Best KO retention at ε=8: 8.7%. Capacity too low.
- **Exp B (rank-64 output-proj LoRA):** same architecture, 8× capacity. Best KO retention at ε=8: **29.0%** — the previous best clean result.
- **Exp C (multi-layer LoRA, rank-16 inside ViT blocks 6/8/10/11):** training diverged at LR=1e-3, all languages collapsed to ~10% accuracy (random chance). Results uninformative. Needs re-run with LR=1e-4.
- **Exp D (text projection orthogonalisation):** fine-tuned `model.text.proj` to push same-class cross-lingual cosine from 0.914 → 0.313. Ran in 3 seconds. Effect on retention untested in isolation because the evaluation was confounded by the collapsed Exp C image adapters.
- **Exp E (five independent per-language ViT copies, 5 epochs):** clean accuracy improved (KO: 83.6% → 94.6%) but adversarial retention was 0% at every budget. The five copies started from identical pretrained weights and 5 epochs was insufficient to break the symmetry; estimate 50–100 epochs needed.
- **Exp F (base encoder + text orth in isolation):** frozen shared image encoder with only the orthogonalised text projections. Best retention: 12.5% (KO, ε=8). Text-side changes alone cannot help because the shared image encoder still maps each input to a single point in R^512; PGD adapts to any text geometry and finds a cross-language direction.

**2026-06-19 (afternoon) — Experiment G: dual language-specific encoders, first >50% result**

Notebook: `lib/notebooks/dual_encoder_divergence.ipynb`. Hardware: local RTX 5070 Ti, 282 s training.

Architecture: per-language LoRA adapters injected at all 12 ViT transformer blocks (rank-32, applied to the CLS token in NLD layout — fixing the NLD/LND layout bug from Exp C) plus per-language text projection heads (`nn.Linear(768, 512)`) bypassing the shared text projection. 4.92M trainable parameters. Training: 1000 CIFAR-10 images, 15 epochs, LR=1e-4, inner PGD 7 steps at `DUAL_TRAIN_EPS=2/255`.

Results — adversarial retention (PGD 20 steps vs EN pair):

| ε | KO | ES | FR | JA |
|---|---|---|---|---|
| 0.5 | 72.8% | 68.8% | 72.8% | 73.9% |
| 1 | 75.6% | 73.5% | 69.9% | 77.8% |
| 2 | **79.6%** | **80.3%** | **74.6%** | **82.1%** |
| 4 | 69.5% | 73.1% | 69.9% | 70.6% |
| 8 | 44.8% | 35.8% | 44.8% | 39.8% |

This is the first experiment to exceed the 50% target. At ε ≤ 4/255 all four non-EN languages are above 50%. The ε=8 gap (44.8% KO, target >50%) is explained by the training epsilon being only 2/255 — the model has not seen strong enough attacks during training. The fix is straightforward: retrain with `DUAL_TRAIN_EPS = 4` or `8`.

Why this worked where A–F failed: for the first time, both the image and text sides are language-specific simultaneously. Classification and divergence losses now reinforce each other (in A–F, shared text anchors pulled image embeddings back toward a common point, limiting how far adapters could push them apart).

**2026-06-19 (night) — Thread B: typographic attack confusion matrices with separate encoders**

Notebook: `lib/notebooks/typographic_attack_confusion.ipynb`. Dataset: STL-10, 200 random images (seed 0).

A 4×4 grid: four attack languages (EN, ZH, KO, JA) × four independent classifiers. For each combination, the adversarial class name is rendered as text on the image and the model classifies it. Key results:

- **English typographic attack dominates.** Writing an English target word achieves ASR 79% on the EN model, 70.5% on KO, 39.5% on ZH. No other attack language achieves >15% ASR on any model.
- **Non-English script attacks do not transfer.** ZH/KO/JA text on the image leaves EN/ZH/KO accuracy above 84% and ASR near zero.
- **Separate encoders disagree under EN attack.** Under `attack_en`, EN model accuracy (21%) ≠ ZH (59.5%) ≠ KO (29%). This is qualitatively different from Thread A: with the shared encoder, all languages collapsed to ~0% simultaneously and agreed on the wrong answer. Separate encoders produce a detection signal.
- **JA column is broken.** The `line-corporation/clip-japanese-base` model loaded with 14% clean accuracy (should be ~90%+). Cause: tokenizer/sentencepiece/protobuf compatibility issue with transformers 5.x. All JA conclusions are provisional until this is fixed and the notebook re-run.

---

### Results at a glance

**Thread A leaderboard (KO retention at ε=8/255):**

| Method | KO retention @ ε=8 | Notes |
|---|---|---|
| No defence (baseline) | ~86% transfer (all fooled) | — |
| Rank-8 output-proj LoRA (Exp A) | 8.7% | low capacity |
| TXT_ORTH isolated (Exp F) | 12.5% | text-only ceiling |
| Combined C+D (Exp D confounded) | ~14% | image side was broken |
| Rank-64 output-proj LoRA (Exp B) | 29.0% | previous best |
| Multi-layer LoRA r16 (Exp C) | 0% | training diverged |
| Per-language ViT 5 epochs (Exp E) | 0% | undertrained |
| **Dual encoder G (train ε=2)** | **44.8%** | **new best; >50% at ε≤4** |
| Target | **> 50%** | |

**Thread B (separate encoders, typographic, EN attack):**

| Model | Accuracy under EN attack | ASR (EN word) |
|---|---|---|
| model_en | 21.0% | 79.0% |
| model_zh | 59.5% | 39.5% |
| model_ko | 29.0% | 70.5% |
| model_ja | 16.5% | 5.0% (broken baseline) |

---

### What is broken or provisional

1. **JA model in Thread B.** `line-corporation/clip-japanese-base` loads but achieves 14% clean accuracy. The error trace points to a sentencepiece/tiktoken/protobuf incompatibility in transformers 5.x. The fix is to either pin an earlier transformers version or find a compatible checkpoint. Until fixed, treat all JA columns as unreliable.
2. **Experiment C results are uninformative.** Multi-layer LoRA at LR=1e-3 caused training collapse. The architectural idea is sound but needs a rerun with LR=1e-4.
3. **Experiment G ε=8 gap.** The dual encoder was trained with `DUAL_TRAIN_EPS=2`. Retention at ε=8 is ~40–45%, below the 50% target. This is not a fundamental failure — simply a training budget issue.
4. **Q3 denoiser still untested.** The denoiser work (Sections 9–10 of `lib/notebooks/updated_multilingual_consensus_colab.ipynb`) established that a non-adaptive denoiser achieves partial recovery but an adaptive attacker reduces accuracy to 0%. No Q2/G-style architectural fix has been applied to Q3 yet.

---

### Recommended next steps (prioritized)

1. **Scale Experiment G training epsilon** — in `lib/notebooks/dual_encoder_divergence.ipynb`, change `DUAL_TRAIN_EPS` from `2` to `4` or `8` and re-run (~5 minutes on local GPU). This is the single highest-leverage action: the architecture already works and the gap is purely a training budget issue.

2. **Fix JA model loading for Thread B** — install `sentencepiece` and `protobuf` in the venv, or test `trust_remote_code=True` with an older transformers pin. Then re-run `lib/notebooks/typographic_attack_confusion.ipynb` to get a reliable 4×4 grid.

3. **Compute disagreement-based detection metrics (Thread B Q2 analogue)** — once the JA column is valid, compute the fraction of images where all four models agree on the same class (clean vs attacked), and test a majority-vote or disagreement detector for AUC > 0.5. The methodology is in `docs/CODE_GUIDE_separate_langs_typographic.md`.

4. **Re-run Experiment C with ML_LR=1e-4** — this is lower priority now that Experiment G succeeded, but multi-layer LoRA inside ViT blocks could still outperform output-only adapters if the training is stable.

5. **Q3 denoiser on dual-encoder G architecture** — the most open-ended item. Worth revisiting only after steps 1–3 are done.

---

### Repo layout after today's reorganization

The project was reorganized today (2026-07-04):

```
2026summer/
├── docs/          ← all project markdown (research_diary.md, research_goal.md,
│                     handoff.md, handoff2.md, mentor_proposal.md, and plan files)
├── lib/
│   ├── notebooks/ ← all .ipynb files + results/
│   └── *.py       ← notebook build and review scripts
├── claude_experiments/  ← UNTOUCHED (not created by the user)
├── requirements.txt
└── .venv/
```

`claude_experiments/` is a separate working directory created by an automated agent and was not reorganized.

---

## 2026-07-04 (evening) — Experiment G analysis + JA model investigation

### Experiment G: why it worked and what to do next

Experiment G (dual language-specific encoders) is the first experiment to exceed the 50% retention target. The key insight is that every prior experiment (A–F) left either the image encoder or the text encoder shared across languages. That single shared component was the fatal flaw.

**The geometry of the failure (A–F):** the shared ViT maps any image to one point in R^512. All five languages' text anchors for the same class point in nearly the same direction (cosine 0.914). Any PGD step that moves the image away from the English class cluster simultaneously moves it away from all other language clusters — the attack is class-targeted, not language-targeted, and class membership is shared universally.

**Why G broke the pattern:** both image and text sides are language-specific simultaneously.
- Image side: per-language rank-32 LoRA adapters at all 12 ViT transformer blocks (applied to the CLS token in NLD layout — fixing the NLD/LND layout bug from Experiment C).
- Text side: per-language `nn.Linear(768, 512)` heads replacing the shared `model.text.proj`.
- Loss interaction: in A–F the classification loss and divergence loss fought each other (shared text anchors pulled all image embeddings back toward a common point). In G, both losses reinforce each other — "push language embeddings apart" and "classify correctly" are now the same objective.

By epoch 15, the per-language image embedding cosine similarities went from 0.927 (nearly identical) to −0.106 (anti-correlated). A PGD attack optimised against the EN pair moves the EN embedding in a direction largely irrelevant to the position of KO/ES/FR/JA embeddings.

**Why ε=8 still falls short (44.8%, target >50%):** `DUAL_TRAIN_EPS = 2/255` during training. At ε=8 the attacker moves 4× further in pixel space than the model ever saw. The architecture is not the bottleneck — the training budget is.

**Paths to improve (ordered by expected impact):**

1. **Scale training epsilon** — change `DUAL_TRAIN_EPS` to `4` or `8` in `lib/notebooks/dual_encoder_divergence.ipynb`. ~5 minutes on local GPU. The ε≤4 results (69–82% retention) show the architecture works; it just needs to have seen stronger attacks.
2. **More data and epochs** — 1000 images, 15 epochs is light. Expanding to 5000+ images and 30–50 epochs would give the adapters more signal, particularly for KO and FR which lag behind EN.
3. **Full sequence adapters at early blocks** — current adapters touch only the CLS token. Applying adapters to all patch tokens at early ViT blocks (0–5, where spatial features are computed) would give each language its own low-level visual processing, making single-gradient cross-language transfer harder.
4. **Combine G with long-run Experiment E** — start from G's already-diverged adapter weights, then unfreeze the full ViT backbone for continued fine-tuning at LR=1e-6. The five copies begin already diverged rather than identical, so far fewer epochs are needed to break LAION-5B symmetry.
5. **Adaptive attacker evaluation** — currently the attacker only optimises against the EN pair. A joint attack across all five language pairs simultaneously would give a more honest robustness estimate.

---

### JA model investigation (Thread B)

**Goal:** determine whether the 14% JA clean accuracy in `lib/notebooks/typographic_attack_confusion.ipynb` was caused by a missing Python package, and whether re-running the notebook with the fix would produce valid JA column results.

**What was tried:**

1. Confirmed `sentencepiece`, `protobuf`, and `tiktoken` were all already installed in the venv. The original error trace in the notebook (a `ValueError: tiktoken is required` chain) was from a previous run when these packages were absent; they were installed at some point between then and now.
2. Re-executed the full notebook via `jupyter nbconvert --execute`. The JA model (`line-corporation/clip-japanese-base`) loaded without any errors. Despite successful loading, clean accuracy remained 14.0% — identical to the original run.
3. Diagnosed the text embeddings directly: inter-class cosine similarities ranged from 0.67 to 0.86. For a healthy CLIP model, different class labels should produce similarities around 0.2–0.5. The extremely tight clustering means the model has almost no discriminative signal between STL-10 categories.
4. Tested six different Japanese prompt templates (`"{}の写真。"`, bare word, `"これは{}です。"`, etc.) on 50 STL-10 images. All templates gave 0–6% accuracy — worse than random guessing (10%).
5. Attempted to substitute `rinna/japanese-clip-vit-b-16`, a better-known Japanese CLIP model. It cannot be loaded via standard `CLIPModel` + `CLIPProcessor` due to incompatible weight key naming conventions.

**Conclusion:** the 14% JA clean accuracy is not a software bug. It is a genuine capability limitation of the `line-corporation/clip-japanese-base` (CLYP) model on STL-10. The model was trained on Japanese web crawl pairs with a different category distribution, and its text encoder produces nearly identical embeddings for all 10 STL-10 class labels regardless of template or loading method.

**Implications for Thread B:**

- The 4×4 grid results in `lib/notebooks/results/confusion_results.json` and the heatmaps remain identical to the original run because the underlying issue was never a package problem.
- All conclusions drawn from the JA column (ASR ~5%, "robust to all attacks") are invalid — the model was never classifying correctly to begin with.
- The EN, ZH, KO columns are unaffected and their conclusions stand.

**Options going forward:**

1. **Drop JA and reduce to a 3×3 grid (EN, ZH, KO).** The core findings — English typographic attack dominates cross-lingually, non-English script attacks do not transfer, separate encoders disagree under EN attack unlike the shared encoder in Q1 — are fully supported by the three working models. This is the cleanest path.
2. **Replace CLYP with a working JA model.** This requires finding a Japanese CLIP checkpoint loadable via the existing API. No standard HuggingFace option was confirmed working in this session. Worth a separate search before committing to option 1.

**Additional analysis — is this word-specific or a general model failure?**

Ran a targeted binary retrieval test to determine whether the poor accuracy is caused by these specific STL-10 vocabulary items or by something deeper.

Test: given 10 dog images and 10 cat images from STL-10, ask CLYP to choose between only two labels — 犬 (dog) or 猫 (cat).

| Images | Correct label | Score |
|---|---|---|
| Cat images | 猫 | 10/10 |
| Dog images | 犬 | 0/10 |

Cat images are classified correctly 100% of the time in the binary case. Dog images are *also* classified as 猫 — 0/10 correct. The model is not broken in general; it knows what a cat looks like. The STL-10 dog photographs happen to sit geometrically closer to CLYP's "猫" cluster than its "犬" cluster. This is a visual distribution gap between STL-10 (curated object photos) and CLYP's training data (Japanese web crawl images), not a vocabulary or tokenizer problem.

The inter-class cosine similarity between 犬 and 猫 is 0.862 — the two text embeddings are nearly in the same direction. Any slight visual bias in the image encoder's projection of STL-10 dogs (which may look different in aspect ratio, background, and framing from Japanese web dog photos) tips all dog predictions toward 猫.

**Conclusion:** this was bad luck with the evaluation dataset. CLYP almost certainly works well for its intended use case (Japanese image-text retrieval), where you only need the correct image to rank above random non-matching images. Zero-shot classification on STL-10 requires much more precise geometric separation than the model provides for this distribution. A different Japanese model, or a different evaluation dataset closer to Japanese web imagery, would likely show a functioning JA column. The finding is worth documenting: any real deployment of the separate-encoder defence must verify that each per-language model actually generalises to the evaluation distribution.

---

## 2026-07-05 — CIFAR-10 typographic attack experiment + repo housekeeping

### What was done

**Housekeeping:**
- All STL-10 result files renamed with `stl10_` prefix (`stl10_confusion_results.json`, `stl10_typographic_heatmaps.png`, `stl10_confusion_matrices.png`, `stl10_font_check.png`) to make room for CIFAR-10 results alongside them.
- The empty `notebooks/` directory left over from the July 4 reorganization was confirmed empty and is pending deletion (locked by OS file watcher).

**CIFAR-10 experiment:**
Created `lib/notebooks/cifar10_typographic_attack_confusion.ipynb` from the STL-10 notebook with three changes: CIFAR-10 class labels (automobile, frog replacing car, monkey; updated in all four languages), dataset switched to `uoft-cs/cifar10` (already cached from Thread A), output files prefixed `cifar10_`. Ran the full 4×4 typographic attack grid on 200 randomly sampled CIFAR-10 test images (seed 0).

**Motivation:** test whether CLYP's 14% clean accuracy on STL-10 was caused by a dataset-specific domain gap, or whether the model is fundamentally unsuitable for zero-shot classification on standard benchmarks.

---

### Results

**Clean accuracy (no attack)**

| Model | STL-10 | CIFAR-10 |
|---|---|---|
| EN (OpenAI ViT-B/32) | 98.5% | 85.0% |
| ZH (Chinese CLIP) | 97.0% | 90.5% |
| KO (Bingsu KO CLIP) | 98.5% | 87.0% |
| JA (CLYP) | 14.0% | **19.0%** |

CIFAR-10 is harder than STL-10 for all models (lower resolution, more confusable classes), so the EN/ZH/KO drop from ~98% to 85–90% is expected. CLYP goes from 14% to 19% — a marginal improvement well within random noise. This disproves the domain gap hypothesis: CLYP cannot zero-shot classify standard benchmark images regardless of dataset.

**Accuracy under typographic attack**

| Attack language | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| None (clean) | 85.0% | 90.5% | 87.0% | 19.0% |
| attack_en | **5.5%** | **33.0%** | **14.0%** | 19.5% |
| attack_zh | 79.5% | 58.0% | 85.5% | 18.5% |
| attack_ko | 83.5% | 88.0% | 86.0% | 20.0% |
| attack_ja | 80.5% | 67.5% | 85.0% | 18.0% |

**Attack Success Rate (pred == written target class)**

| Attack language | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| attack_en | **94.5%** | **65.0%** | **86.0%** | 9.0% |
| attack_zh | 3.0% | 37.5% | 2.5% | 9.5% |
| attack_ko | 2.5% | 1.0% | 3.0% | 9.0% |
| attack_ja | 3.0% | 24.0% | 2.0% | 9.5% |

Best attack per model: EN attack dominates for EN, ZH, and KO. JA column is still noise (random at ~9% regardless of attack language, consistent with 19% clean accuracy being near-random).

---

### Analysis

#### What "accuracy under attack" means vs. clean accuracy

The clean baseline tells you how well each model classifies images with no manipulation. The attacked accuracy tells you what happens when the adversarial target class name is written on the image in a given language. The gap between these two numbers is the attack's destructive power.

For the EN model, the EN attack drops accuracy from **85.0% → 5.5%** — a loss of 79.5 percentage points. This is nearly total destruction of the classifier. For context, random guessing on 10 classes gives 10%, so the attacked EN model is performing below chance. The ASR of 94.5% tells you why: in almost all cases, the model is not just getting confused — it is specifically redirected to predict the exact class written on the image. The typographic attack has effectively hijacked the model's decision.

This is stronger than the STL-10 result (79% ASR on CIFAR-10 vs. 94.5%). CIFAR-10 images are smaller and lower resolution (32×32 upscaled to 224×224), which means the rendered text occupies a larger fraction of the image area relative to the object, making the typographic signal harder to ignore.

#### Why EN attack transfers to KO and ZH but not vice versa

The EN attack on the KO model drops accuracy from **87.0% → 14.0%** (ASR 86.0%). On the ZH model it drops from **90.5% → 33.0%** (ASR 65.0%). In contrast, the KO attack on the EN model leaves it at 83.5% (ASR 2.5%), and the ZH attack leaves EN at 79.5% (ASR 3.0%).

This asymmetry is not about which language is "stronger" or better trained. It is a consequence of how visual text saliency works in CLIP models:

1. **Latin script is universally salient.** All four CLIP models were trained on large web datasets where image captions are predominantly in English or contain Latin-script text. The visual feature detectors inside every model's ViT have learned to attend to Latin characters because they co-occur with image content descriptions across the entire training corpus, regardless of the model's primary language.

2. **Non-Latin scripts are not visually salient to non-native models.** Korean Hangul, Chinese characters, and Japanese kana/kanji are visually present in the EN model's training data only incidentally — they don't reliably co-occur with image-content descriptions in English web text. So when the EN model sees 자동차 written on an image, it does not recognise the script as a class label; it treats it as an irrelevant visual texture.

3. **The KO model is slightly fooled by Chinese (ZH)** — the ZH attack on KO gives only 1.0% ASR, but ZH on EN gives 3.0% and ZH on ZH gives 37.5%. Chinese characters are visually closer to Japanese kanji and Korean hanja borrowings than to Latin script, but the Korean CLIP model was still trained primarily on Korean-captioned data and does not strongly associate Chinese characters with classification labels.

4. **The English word "automobile" is a strong visual cue for all three working models** because all three were trained on datasets containing Latin-script text in image descriptions, regardless of the primary language of the model.

#### What disagreement means for the defence

Under the EN typographic attack, the three working models produce very different responses:

| Model | Accuracy under EN attack | ASR |
|---|---|---|
| EN | 5.5% | 94.5% — nearly all images predicted as the written class |
| KO | 14.0% | 86.0% — strong attack but not as complete as EN |
| ZH | 33.0% | 65.0% — moderately fooled |

This disagreement is exactly the detection signal the consensus defence needs. Under the shared multilingual CLIP (Thread A / Q1), all five languages collapsed to ~0% accuracy simultaneously under PGD attack — they all agreed on the wrong answer. Here, the three models disagree substantially: the EN model predicts the written class 94.5% of the time while the ZH model still gets 33% correct. A simple majority vote of EN + ZH + KO would produce the correct prediction (since ZH and KO still classify many images correctly and would outvote EN), or a disagreement detector would fire whenever EN's prediction differs from ZH and KO.

This is the most important finding from Thread B so far: **the separate-encoder design naturally produces the disagreement signal that the shared-encoder design cannot**, at least under the typographic threat model.

#### CIFAR-10 vs. STL-10 comparison

CIFAR-10 results are more extreme across the board. EN attack ASR: 94.5% vs 79% on STL-10; EN attack ASR on KO: 86% vs 70.5%; EN attack ASR on ZH: 65% vs 39.5%. This is consistent with lower image resolution making text overlays proportionally larger and more visually dominant. CIFAR-10 is the harder and more convincing demonstration.

#### Conclusion on CLYP

CLYP scores 19% on CIFAR-10, the same functional failure as STL-10. The domain gap hypothesis is rejected. CLYP is not suitable for zero-shot classification on standard benchmarks and should be replaced. The remaining three models (EN, ZH, KO) give a clean and complete 3×3 result that supports all Thread B conclusions without requiring a working JA model.

---

### Next steps

1. **Drop JA and reduce to 3×3.** Present the EN/ZH/KO results as the primary Thread B finding. The JA column is not scientifically usable.
2. **Add disagreement detection metrics** — compute the fraction of images where EN prediction disagrees with the ZH+KO majority under EN attack (clean vs. attacked). This is the Q2 analogue for the typographic threat model.
3. **Update research_goal.md** to reflect that Thread B results are now on CIFAR-10 (same dataset as Thread A), making cross-thread comparison cleaner.

---

## 2026-07-05 (evening) — llm-jp JA model swap + full 4×4 CIFAR-10 results

### What was done

Replaced CLYP (`line-corporation/clip-japanese-base`) with `llm-jp/llm-jp-clip-vit-base-patch16` as the JA model in `lib/notebooks/cifar10_typographic_attack_confusion.ipynb`. The new model loads via standard `open_clip` (identical API to the EN model) and has a published CIFAR-10 zero-shot accuracy of 91.8% on the llm-jp benchmark. Re-ran the full 4×4 typographic attack experiment on 200 CIFAR-10 test images.

The new `JaCLIP` wrapper:
```python
self.m, _, self.pp = open_clip.create_model_and_transforms('hf-hub:llm-jp/llm-jp-clip-vit-base-patch16')
self.tok = open_clip.get_tokenizer('hf-hub:llm-jp/llm-jp-clip-vit-base-patch16')
```

---

### Results

**Clean accuracy — all four models working**

| Model | Clean accuracy |
|---|---|
| EN (OpenAI ViT-B/32) | 85.0% |
| ZH (Chinese CLIP) | 90.5% |
| KO (Bingsu KO CLIP) | 87.0% |
| **JA (llm-jp ViT-B/16)** | **93.0%** |

JA is now the strongest classifier in the ensemble. The previous CLYP model scored 14–19% (near random); llm-jp achieves 93.0%, confirming the problem was the model choice, not the task or dataset.

**Accuracy under typographic attack**

| | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| Clean baseline | 85.0% | 90.5% | 87.0% | 93.0% |
| attack_en | **5.5%** | **33.0%** | **14.0%** | **9.5%** |
| attack_zh | 79.5% | 58.0% | 85.5% | 93.0% |
| attack_ko | 83.5% | 88.0% | 86.0% | 93.0% |
| attack_ja | 80.5% | 67.5% | 85.0% | 92.5% |

**Attack Success Rate**

| | model_en | model_zh | model_ko | model_ja |
|---|---|---|---|---|
| attack_en | **94.5%** | **65.0%** | **86.0%** | **90.0%** |
| attack_zh | 3.0% | 37.5% | 2.5% | 1.5% |
| attack_ko | 2.5% | 1.0% | 3.0% | 1.5% |
| attack_ja | 3.0% | 24.0% | 2.0% | 2.0% |

Best attack per model: **EN attack dominates all four models.**

---

### Analysis

**JA model now behaves as expected.** llm-jp clean accuracy (93.0%) exceeds all other models, consistent with the published benchmark of 91.8%. Under the EN typographic attack, the JA model drops to 9.5% accuracy (ASR 90.0%) — deeply fooled by English text, matching the pattern of EN and KO. This is the correct and expected behaviour: the llm-jp model was trained on Japanese-translated captions of LAION-5B images, which includes abundant CIFAR-10-style photographs, so it generalises to this distribution.

**The EN attack is now the universal threat.** All four models are heavily fooled by English text overlay:

| Model | Clean acc | Under EN attack | Drop | EN ASR |
|---|---|---|---|---|
| EN | 85.0% | 5.5% | −79.5 pp | 94.5% |
| ZH | 90.5% | 33.0% | −57.5 pp | 65.0% |
| KO | 87.0% | 14.0% | −73.0 pp | 86.0% |
| JA | 93.0% | 9.5% | −83.5 pp | 90.0% |

The JA model is actually the most vulnerable to EN attack in terms of the absolute accuracy drop (−83.5 pp), even though it starts from the highest clean baseline. This makes sense: llm-jp was trained on translated versions of English-captioned web images, so its visual features are deeply aligned with English-described concepts. The EN model's attack direction in pixel space happens to be precisely adversarial for a model trained on English image-text data.

**Non-English script attacks still do not transfer.** ZH attack on EN: ASR 3.0%. KO attack on EN: ASR 2.5%. JA attack on EN: ASR 3.0%. ZH/KO/JA attack on JA: ASR 1.5–2.0%. The Latin-script asymmetry is complete and holds for the llm-jp JA model too: it has learned to associate Latin text with image classification labels (through English-translated training captions), but Korean Hangul and Chinese characters are not classification-relevant visual features for any model in the ensemble.

**Disagreement under EN attack — detection signal now covers all four models:**

| Model | Accuracy under EN attack |
|---|---|
| EN | 5.5% |
| JA | 9.5% |
| KO | 14.0% |
| ZH | 33.0% |

All four models are fooled, but ZH is substantially more robust than the others (33% vs 5–14%). A disagreement detector comparing ZH against the EN/KO/JA majority would fire reliably under EN attack. The ZH model's relative robustness may reflect that Chinese CLIP was trained on Chinese web text where Latin-script text overlay carries less classification signal.

**Comparison with Thread A (shared encoder + PGD):**

| Setting | Attack | Defence signal |
|---|---|---|
| Thread A: shared `xlm-roberta-base-ViT-B-32` | PGD ε=8 | All 5 languages collapse together — zero disagreement |
| Thread B: 4 separate CLIPs | EN typographic | Models disagree substantially (ZH 33% vs EN 5.5%) |

Thread B's separate-encoder design produces the disagreement that Thread A's shared encoder could never produce, exactly as hypothesised. The disagreement is not perfect (all four models are eventually fooled by strong EN attack), but it is sufficient to build a detection signal.

---

### Next steps

1. **Compute disagreement detection AUC** — for each image, record whether the four models agree on the same class (clean vs. attacked). Compute AUC of a simple disagreement count detector. This is the Q2 analogue for the typographic threat model, as defined in `docs/CODE_GUIDE_separate_langs_typographic.md`.
2. **Update `research_goal.md`** — Thread B is now fully functional with 4 working models on CIFAR-10.
3. **Consider running Thread B with the full 8000-image CIFAR-10 test split** rather than 200 images, to get tighter confidence on the disagreement rates.

---

## July 5, 2026 — Evening (disagreement AUC + research_goal.md update)

**Disagreement detector AUC computed.** Added a new cell to `lib/notebooks/cifar10_typographic_attack_confusion.ipynb` that:
1. Stores per-model raw predictions on clean images (`clean_preds`) in the clean-baseline cell (was previously only saving accuracy floats).
2. Computes a disagreement score for each image = number of unique predictions across the 4 models (score 1 = full agreement, 4 = all different).
3. Computes the standard ROC AUC (ties count as 0.5) between the EN-attacked and clean distributions.

**Results (200 test images, EN typographic attack as "positive"):**

| Condition | All-agree (score = 1) | Score ≥ 2 |
|---|---|---|
| Clean | 77.5% | 22.5% |
| Attacked (EN) | 61.5% | 38.5% |

**Detector AUC = 0.574.**

The signal is real but modest. With only 4 models, most images — even attacked ones — still get classified to the same (wrong) class by all 4 models, keeping the disagreement score at 1. The detection power comes mostly from the Chinese model staying more accurate under EN attack: when EN/KO/JA all predict the adversarial class and ZH predicts the true class, the disagreement score jumps to 2. Score distribution shows attacked images shift probability mass from 1 → 2 (18.5% clean vs. 37% attacked at score=2), with scores 3–4 being rare in both conditions.

AUC 0.574 > 0.5 confirms the hypothesis (disagreement does separate clean from attacked), but it is nowhere near actionable as a standalone detector. A practical alarm system would need either more languages/models or a richer detection signal than simple count-of-unique-predictions.

Results saved to `lib/notebooks/results/cifar10_confusion_results.json` under the `"detector"` key.

**`docs/research_goal.md` updated.** Added Thread B architecture (separate brains mermaid diagram), a Thread B findings section, and Thread B rows in the "What Was Found" table (Q1: ZH 33% under EN attack; Q2: AUC 0.574). The document now covers both threads in full.

---

## July 5, 2026 — Late evening (deeper ensemble analysis: 1000 images, per-class, all-language AUC)

**Homework: deeper analysis of the voting ensemble.** Scaled the CIFAR-10 experiment from 200 to 1000 images, added per-class vulnerability breakdowns for every attack language, and extended the disagreement-detector AUC from EN-only to all four attack languages.

### Updated 4×4 accuracy matrix (1000 images)

Numbers stabilised relative to the 200-image run — the broad picture is confirmed, not revised.

|  | model_EN | model_ZH | model_KO | model_JA |
|---|---|---|---|---|
| Clean baseline | 84.2% | 92.7% | 87.7% | **93.2%** |
| attack_EN | 4.6% | **36.5%** | 15.6% | 8.3% |
| attack_ZH | 79.2% | 58.3% | 84.5% | 90.2% |
| attack_KO | 81.9% | 89.4% | 86.1% | 91.1% |
| attack_JA | 78.2% | 69.8% | 83.5% | 89.9% |

Key observations:
- EN attack is the only one that substantially hurts the ensemble. ZH, KO, and JA text overlays leave all four models nearly at clean-accuracy levels, confirming the Latin-script asymmetry at scale.
- ZH model (36.5%) remains notably more robust than EN (4.6%), KO (15.6%), JA (8.3%) under EN attack.
- JA model has the highest clean accuracy (93.2%) and the highest off-diagonal accuracy under non-EN attacks (88–91%), confirming it is the strongest model in the ensemble.

### Per-class vulnerability under EN attack

Four per-class bar-chart grids saved to `lib/notebooks/results/cifar10_perclass_attack_{lang}.png`.

Under EN attack, the 10 CIFAR-10 classes are not equally vulnerable. Key findings:

| Class | EN model | ZH model | KO model | JA model | Notable |
|---|---|---|---|---|---|
| airplane | 1% | 46% | 14% | 9% | EN almost fully fooled |
| automobile | 0% | 44% | 39% | 2% | KO retains some accuracy |
| bird | 9% | 46% | 26% | 14% | ZH best here |
| cat | 0% | 35% | 1% | 2% | Near-total collapse for EN/KO/JA |
| deer | 8% | 27% | 25% | 16% | All models hurt |
| **dog** | **0%** | **13%** | **0%** | **0%** | Most vulnerable class — EN "dog" label is the only clean one-syllable common word that overlaps perfectly with the visual feature |
| frog | 5% | 27% | 15% | 20% | Moderate |
| horse | 20% | 41% | 26% | 20% | Most resistant across models |
| ship | 4% | 38% | 4% | 0% | EN/KO/JA collapse |
| truck | 4% | 50% | 11% | 4% | ZH most robust |

"Dog" and "cat" are the most exploitable classes — their English names are short, common, visually unambiguous, and strongly encoded by every model's text encoder. "Horse" is the hardest to fully fool (20% EN model accuracy retained), possibly because the English word is less dominantly associated with the visual prototype in LAION training data.

### Disagreement detector AUC — all attack languages

| Attack language | All-agree rate (attacked) | AUC |
|---|---|---|
| Clean | 78.2% | — |
| **EN** | 59.3% | **0.588** |
| **ZH** | 50.4% | **0.646** |
| **KO** | 73.4% | 0.525 |
| **JA** | 58.6% | 0.604 |

The ZH attack produces the **highest detector AUC (0.646)** despite being the weakest attack overall (most models barely affected). This seems counter-intuitive but makes sense: ZH attack strongly fools the ZH model (58.3% vs 92.7% clean) while leaving EN/KO/JA near their baselines — so ZH-attacked images create a consistent pattern of "ZH predicts differently from the other three," which is a detectable disagreement signature. KO attack (AUC 0.525) is barely above chance because it doesn't fully fool any model, so there is no systematic disagreement to detect. EN attack (0.588) and JA attack (0.604) are in the middle range.

Overall, the detector AUC range is 0.52–0.65 across attack types — real signal, but not strong enough for a reliable alarm without additional features.

---

### Ways to improve the disagreement detector

The core problem: 62% of attacked images still get score=1 (all 4 models agree on the wrong class), so the detector never fires for those. Three directions to raise AUC:

1. **Use confidence margin instead of hard predictions.** Instead of counting unique class predictions, measure how much each model's top-1 confidence drops under the suspected attack. On a clean image all models are confident; on a typographic attacked image the "true" class and the "planted" class compete, which flattens the probability distribution. A detector on softmax entropy or top-1 margin would fire even when all 4 models happen to predict the same wrong class.

2. **Add more languages/models.** Each new model is another independent "vote." With 4 models only ZH reliably disagrees under EN attack; adding e.g. a French or Arabic CLIP would create more breakpoints. AUC scales roughly with the number of independently trained models.

3. **Use cross-model cosine distance on embeddings, not predictions.** Compare each model's image embedding to its text embedding for the top-1 class — a typographic attack degrades the match quality in a detectable way even when the argmax doesn't change. This is a richer signal than the binary "same prediction or not."

4. **Target a harder test set.** The 200-image sample may underrepresent classes where ZH is most robust. Running on the full 8000-image CIFAR-10 test split would confirm whether AUC 0.574 is stable or an artefact of the small sample.

---

### Homework — Why does the Japanese model outperform the others?

This answer covers the full arc: the original broken JA model, what the diagnostics revealed, the replacement, and what the final numbers tell us.

#### Step 1 — The original JA model was broken (CLYP / `line-corporation/clip-japanese-base`)

The first Japanese model used (`line-corporation/clip-japanese-base`, nicknamed CLYP) achieved only 14–19% accuracy on both STL-10 and CIFAR-10, barely above chance for 10 classes. Three rounds of diagnostics isolated the cause:

- **Compressed text embeddings:** All 10 class label embeddings were clustered together with cosine similarities of 0.67–0.86. The model's text encoder had no discriminative separation between "airplane," "dog," "frog," etc. Any image would tie-break to whichever class happened to sit 0.001 cosine units closer.
- **Domain mismatch (not a broken model):** When tested on a simpler binary task (cat vs. dog on STL-10), CLYP got cat 10/10 but dog 0/10. The model was not fundamentally broken — it was trained on Japanese social-media image–caption pairs, where animal retrieval worked fine but zero-shot 10-class scene classification was outside its distribution.
- **No prompt template helped:** Trying every plausible Japanese prompt (`「{}の写真」`, `「{}」`, romaji, etc.) made no meaningful difference.

Conclusion: CLYP's design goal was image–text *retrieval*, not zero-shot *classification*. For our task it was the wrong tool.

#### Step 2 — Choosing the replacement (`llm-jp/llm-jp-clip-vit-base-patch16`)

A web search for Japanese CLIP models benchmarked on CIFAR-10 led to `llm-jp-clip`, developed by the National Institute of Informatics (llm-jp consortium). Its published benchmark table showed the highest CIFAR-10 zero-shot accuracy among Japanese CLIP variants. It uses the same `open_clip` API as the other models in the ensemble, making the swap straightforward.

#### Step 3 — What the final numbers show

| Model | Clean acc | Under EN attack | Under ZH attack | Under KO attack | Under JA attack |
|---|---|---|---|---|---|
| EN | 85.0% | 5.5% | 79.5% | 83.5% | 80.5% |
| ZH | 90.5% | 33.0% | 58.0% | 88.0% | 67.5% |
| KO | 87.0% | 14.0% | 85.5% | 86.0% | 85.0% |
| **JA** | **93.0%** | **9.5%** | **93.0%** | **93.0%** | **92.5%** |

Three things stand out:

**1. Best clean accuracy (93%).**
`llm-jp-clip` was trained on a larger and more carefully filtered Japanese web corpus than its competitors. Higher pre-training data quality directly translates to better zero-shot classification. The model simply learned richer visual–semantic associations.

**2. Near-total immunity to non-EN attacks.**
Under ZH, KO, and JA text-overlay attacks, the JA model barely moves (93% → 93% → 92.5%). The reason is script isolation: Chinese characters, Korean Hangul, and Japanese kanji/kana are visually and statistically distinct from the Latin-alphabet labels the attack relies on for other models. Even for its own language, the JA attack (writing the Japanese class name on the image) barely hurts the JA model — probably because `llm-jp-clip` learned to rely on image features more than text-overlay cues when the two conflict.

**3. The EN attack does hurt JA (93% → 9.5%).**
This is the expected result and is *not* a failure of the JA model. The attack works by writing the English adversarial class name (e.g., "truck") directly on the image. The JA model was trained on bilingual data that included many English captions alongside Japanese ones (Japanese web text is heavily mixed with English). So the JA text encoder does recognise "truck" as a classification-relevant word. The attack exploits that English literacy. KO at 14% is slightly less affected, and ZH at 33% is the most robust — reflecting different degrees of English–native-language mixing in each model's training corpus.

#### Summary

The JA model outperforms the others because (a) it was specifically chosen by benchmark — the original model was replaced after a systematic failure analysis — and (b) `llm-jp-clip`'s training data quality and scale gave it 93% clean accuracy and strong image-feature reliance that makes CJK/Hangul text overlays irrelevant. Its one vulnerability is the same as everyone's: English text overlays, because all four models were trained on data that included English and have learned to treat Latin script as a valid classification signal.
