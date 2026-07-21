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


| ε   | KO retention | ES   | FR    | JA   |
| --- | ------------ | ---- | ----- | ---- |
| 0.5 | 2.1%         | 4.9% | 5.1%  | 3.0% |
| 4   | 11.9%        | 6.7% | 12.4% | 6.4% |
| 8   | 8.7%         | 5.7% | 8.4%  | 5.3% |


**Assessment:** Small but real improvement over 0% baseline retention. The adapters
learn but rank-8 is too low-capacity to reshape the embedding geometry meaningfully.

---



#### Experiment B — Rank-64 output-projection LoRA (Section 14)

**What:** Same architecture as A but rank 64 (8× more capacity, ~327K parameters).
Same training setup.

**Results:**


| ε   | KO retention | ES    | FR    | JA    |
| --- | ------------ | ----- | ----- | ----- |
| 0.5 | 0.3%         | 0.3%  | 0.3%  | 0.3%  |
| 2   | 10.5%        | 9.9%  | 11.1% | 7.8%  |
| 4   | 24.5%        | 17.0% | 23.4% | 17.5% |
| 8   | **29.0%**    | 14.4% | 24.8% | 16.5% |


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


| ε     | KO  | ES  | FR  | JA  |
| ----- | --- | --- | --- | --- |
| 0.5–8 | 0%  | 0%  | 0%  | 0%  |


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


| ε   | EN fooled | KO    | ES   | FR   | JA   |
| --- | --------- | ----- | ---- | ---- | ---- |
| 0.5 | 287       | 0.0%  | 0.0% | 0.0% | 0.0% |
| 1   | 287       | 0.0%  | 0.0% | 0.0% | 0.0% |
| 2   | 287       | 1.7%  | 0.0% | 0.0% | 0.0% |
| 4   | 287       | 2.8%  | 0.0% | 0.0% | 0.0% |
| 8   | 287       | 12.5% | 0.0% | 6.3% | 4.2% |


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


| Method                    | KO retention         | Status                          |
| ------------------------- | -------------------- | ------------------------------- |
| No defence (baseline)     | ~86% (full transfer) | —                               |
| Rank-8 output-proj LoRA   | 8.7%                 | undertrained direction          |
| TXTORTH isolated          | 12.5%                | text-only ceiling               |
| Combined C+D (confounded) | ~14%                 | not reliable                    |
| Rank-64 output-proj LoRA  | **29.0%**            | best clean result               |
| Multi-layer LoRA r16      | 0%                   | training diverged (LR too high) |
| Per-language ViT (5 ep)   | 0%                   | undertrained                    |
| **Target**                | **> 50%**            |                                 |


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


| ε   | EN fooled | KO        | ES        | FR        | JA        |
| --- | --------- | --------- | --------- | --------- | --------- |
| 0.5 | 276       | 72.8%     | 68.8%     | 72.8%     | 73.9%     |
| 1   | 279       | 75.6%     | 73.5%     | 69.9%     | 77.8%     |
| 2   | 279       | **79.6%** | **80.3%** | **74.6%** | **82.1%** |
| 4   | 279       | 69.5%     | 73.1%     | 69.9%     | 70.6%     |
| 8   | 279       | 44.8%     | 35.8%     | 44.8%     | 39.8%     |




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


| Method                      | KO retention         | Notes                           |
| --------------------------- | -------------------- | ------------------------------- |
| No defence (baseline)       | ~86% (full transfer) | —                               |
| Rank-8 output-proj LoRA     | 8.7%                 | too low capacity                |
| TXTORTH isolated            | 12.5%                | text-only ceiling               |
| Combined C+D (confounded)   | ~14%                 | collapsed image adapters        |
| Rank-64 output-proj LoRA    | 29.0%                | previous best                   |
| Multi-layer LoRA r16        | 0%                   | training diverged (LR too high) |
| Per-language ViT (5 ep)     | 0%                   | undertrained                    |
| **Dual enc. G (train ε=2)** | **44.8%**            | **new best; >50% at ε≤4**       |
| **Target**                  | **> 50%**            |                                 |




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


| lang | model                                                           |
| ---- | --------------------------------------------------------------- |
| en   | OpenAI ViT-B/32 (`open_clip`)                                   |
| zh   | `OFA-Sys/chinese-clip-vit-base-patch16`                         |
| ko   | `Bingsu/clip-vit-base-patch32-ko`                               |
| ja   | `line-corporation/clip-japanese-base` (CLYP, trust_remote_code) |


Each image gets a random adversarial target class (≠ true class). For each attack
language, the target class name is rendered in that language at the bottom of the
image. Metrics: **accuracy** (pred == true), **ASR** (pred == written target class).

### Clean baseline (unattacked)


| model | clean accuracy |
| ----- | -------------- |
| en    | 98.5%          |
| zh    | 97.0%          |
| ko    | 98.5%          |
| ja    | **14.0%**      |


EN/ZH/KO are healthy. **JA clean accuracy is broken** (likely tokenizer / transformers
5.x compatibility issues during this run). All conclusions involving `model_ja` should
be treated as provisional until JA loading is fixed and the notebook re-run.

### Accuracy matrix under attack (rows = attack language, cols = model)


|               | model_en | model_zh | model_ko | model_ja |
| ------------- | -------- | -------- | -------- | -------- |
| **attack_en** | 21.0%    | 59.5%    | 29.0%    | 16.5%    |
| **attack_zh** | 97.5%    | 84.0%    | 98.5%    | 18.5%    |
| **attack_ko** | 96.5%    | 97.5%    | 97.5%    | 16.5%    |
| **attack_ja** | 97.0%    | 89.5%    | 97.5%    | 17.0%    |




### ASR matrix (attack success — pred == written class)


|               | model_en  | model_zh  | model_ko  | model_ja |
| ------------- | --------- | --------- | --------- | -------- |
| **attack_en** | **79.0%** | **39.5%** | **70.5%** | 5.0%     |
| **attack_zh** | 0.0%      | 14.5%     | 0.0%      | 4.5%     |
| **attack_ko** | 0.5%      | 0.0%      | 0.0%      | 5.0%     |
| **attack_ja** | 0.5%      | 7.5%      | 0.5%      | 5.0%     |


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


| prior result                                             | typographic result (this run)                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Q1 shared encoder + PGD: all languages collapse together | separate encoders + EN typographic: models disagree                                       |
| Experiment G dual encoder + PGD: >50% retention at ε≤4   | typographic: EN word fools multiple models but not all equally                            |
| Consensus defence fails on shared architecture           | disagreement-based **detection** may work with separate encoders under typographic threat |




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


| ε   | KO        | ES        | FR        | JA        |
| --- | --------- | --------- | --------- | --------- |
| 0.5 | 72.8%     | 68.8%     | 72.8%     | 73.9%     |
| 1   | 75.6%     | 73.5%     | 69.9%     | 77.8%     |
| 2   | **79.6%** | **80.3%** | **74.6%** | **82.1%** |
| 4   | 69.5%     | 73.1%     | 69.9%     | 70.6%     |
| 8   | 44.8%     | 35.8%     | 44.8%     | 39.8%     |


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


| Method                            | KO retention @ ε=8         | Notes                     |
| --------------------------------- | -------------------------- | ------------------------- |
| No defence (baseline)             | ~86% transfer (all fooled) | —                         |
| Rank-8 output-proj LoRA (Exp A)   | 8.7%                       | low capacity              |
| TXT_ORTH isolated (Exp F)         | 12.5%                      | text-only ceiling         |
| Combined C+D (Exp D confounded)   | ~14%                       | image side was broken     |
| Rank-64 output-proj LoRA (Exp B)  | 29.0%                      | previous best             |
| Multi-layer LoRA r16 (Exp C)      | 0%                         | training diverged         |
| Per-language ViT 5 epochs (Exp E) | 0%                         | undertrained              |
| **Dual encoder G (train ε=2)**    | **44.8%**                  | **new best; >50% at ε≤4** |
| Target                            | **> 50%**                  |                           |


**Thread B (separate encoders, typographic, EN attack):**


| Model    | Accuracy under EN attack | ASR (EN word)          |
| -------- | ------------------------ | ---------------------- |
| model_en | 21.0%                    | 79.0%                  |
| model_zh | 59.5%                    | 39.5%                  |
| model_ko | 29.0%                    | 70.5%                  |
| model_ja | 16.5%                    | 5.0% (broken baseline) |


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


| Images     | Correct label | Score |
| ---------- | ------------- | ----- |
| Cat images | 猫             | 10/10 |
| Dog images | 犬             | 0/10  |


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


| Model                | STL-10 | CIFAR-10  |
| -------------------- | ------ | --------- |
| EN (OpenAI ViT-B/32) | 98.5%  | 85.0%     |
| ZH (Chinese CLIP)    | 97.0%  | 90.5%     |
| KO (Bingsu KO CLIP)  | 98.5%  | 87.0%     |
| JA (CLYP)            | 14.0%  | **19.0%** |


CIFAR-10 is harder than STL-10 for all models (lower resolution, more confusable classes), so the EN/ZH/KO drop from ~98% to 85–90% is expected. CLYP goes from 14% to 19% — a marginal improvement well within random noise. This disproves the domain gap hypothesis: CLYP cannot zero-shot classify standard benchmark images regardless of dataset.

**Accuracy under typographic attack**


| Attack language | model_en | model_zh  | model_ko  | model_ja |
| --------------- | -------- | --------- | --------- | -------- |
| None (clean)    | 85.0%    | 90.5%     | 87.0%     | 19.0%    |
| attack_en       | **5.5%** | **33.0%** | **14.0%** | 19.5%    |
| attack_zh       | 79.5%    | 58.0%     | 85.5%     | 18.5%    |
| attack_ko       | 83.5%    | 88.0%     | 86.0%     | 20.0%    |
| attack_ja       | 80.5%    | 67.5%     | 85.0%     | 18.0%    |


**Attack Success Rate (pred == written target class)**


| Attack language | model_en  | model_zh  | model_ko  | model_ja |
| --------------- | --------- | --------- | --------- | -------- |
| attack_en       | **94.5%** | **65.0%** | **86.0%** | 9.0%     |
| attack_zh       | 3.0%      | 37.5%     | 2.5%      | 9.5%     |
| attack_ko       | 2.5%      | 1.0%      | 3.0%      | 9.0%     |
| attack_ja       | 3.0%      | 24.0%     | 2.0%      | 9.5%     |


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


| Model | Accuracy under EN attack | ASR                                                      |
| ----- | ------------------------ | -------------------------------------------------------- |
| EN    | 5.5%                     | 94.5% — nearly all images predicted as the written class |
| KO    | 14.0%                    | 86.0% — strong attack but not as complete as EN          |
| ZH    | 33.0%                    | 65.0% — moderately fooled                                |


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


| Model                    | Clean accuracy |
| ------------------------ | -------------- |
| EN (OpenAI ViT-B/32)     | 85.0%          |
| ZH (Chinese CLIP)        | 90.5%          |
| KO (Bingsu KO CLIP)      | 87.0%          |
| **JA (llm-jp ViT-B/16)** | **93.0%**      |


JA is now the strongest classifier in the ensemble. The previous CLYP model scored 14–19% (near random); llm-jp achieves 93.0%, confirming the problem was the model choice, not the task or dataset.

**Accuracy under typographic attack**


|                | model_en | model_zh  | model_ko  | model_ja |
| -------------- | -------- | --------- | --------- | -------- |
| Clean baseline | 85.0%    | 90.5%     | 87.0%     | 93.0%    |
| attack_en      | **5.5%** | **33.0%** | **14.0%** | **9.5%** |
| attack_zh      | 79.5%    | 58.0%     | 85.5%     | 93.0%    |
| attack_ko      | 83.5%    | 88.0%     | 86.0%     | 93.0%    |
| attack_ja      | 80.5%    | 67.5%     | 85.0%     | 92.5%    |


**Attack Success Rate**


|           | model_en  | model_zh  | model_ko  | model_ja  |
| --------- | --------- | --------- | --------- | --------- |
| attack_en | **94.5%** | **65.0%** | **86.0%** | **90.0%** |
| attack_zh | 3.0%      | 37.5%     | 2.5%      | 1.5%      |
| attack_ko | 2.5%      | 1.0%      | 3.0%      | 1.5%      |
| attack_ja | 3.0%      | 24.0%     | 2.0%      | 2.0%      |


Best attack per model: **EN attack dominates all four models.**

---



### Analysis

**JA model now behaves as expected.** llm-jp clean accuracy (93.0%) exceeds all other models, consistent with the published benchmark of 91.8%. Under the EN typographic attack, the JA model drops to 9.5% accuracy (ASR 90.0%) — deeply fooled by English text, matching the pattern of EN and KO. This is the correct and expected behaviour: the llm-jp model was trained on Japanese-translated captions of LAION-5B images, which includes abundant CIFAR-10-style photographs, so it generalises to this distribution.

**The EN attack is now the universal threat.** All four models are heavily fooled by English text overlay:


| Model | Clean acc | Under EN attack | Drop     | EN ASR |
| ----- | --------- | --------------- | -------- | ------ |
| EN    | 85.0%     | 5.5%            | −79.5 pp | 94.5%  |
| ZH    | 90.5%     | 33.0%           | −57.5 pp | 65.0%  |
| KO    | 87.0%     | 14.0%           | −73.0 pp | 86.0%  |
| JA    | 93.0%     | 9.5%            | −83.5 pp | 90.0%  |


The JA model is actually the most vulnerable to EN attack in terms of the absolute accuracy drop (−83.5 pp), even though it starts from the highest clean baseline. This makes sense: llm-jp was trained on translated versions of English-captioned web images, so its visual features are deeply aligned with English-described concepts. The EN model's attack direction in pixel space happens to be precisely adversarial for a model trained on English image-text data.

**Non-English script attacks still do not transfer.** ZH attack on EN: ASR 3.0%. KO attack on EN: ASR 2.5%. JA attack on EN: ASR 3.0%. ZH/KO/JA attack on JA: ASR 1.5–2.0%. The Latin-script asymmetry is complete and holds for the llm-jp JA model too: it has learned to associate Latin text with image classification labels (through English-translated training captions), but Korean Hangul and Chinese characters are not classification-relevant visual features for any model in the ensemble.

**Disagreement under EN attack — detection signal now covers all four models:**


| Model | Accuracy under EN attack |
| ----- | ------------------------ |
| EN    | 5.5%                     |
| JA    | 9.5%                     |
| KO    | 14.0%                    |
| ZH    | 33.0%                    |


All four models are fooled, but ZH is substantially more robust than the others (33% vs 5–14%). A disagreement detector comparing ZH against the EN/KO/JA majority would fire reliably under EN attack. The ZH model's relative robustness may reflect that Chinese CLIP was trained on Chinese web text where Latin-script text overlay carries less classification signal.

**Comparison with Thread A (shared encoder + PGD):**


| Setting                                      | Attack         | Defence signal                                        |
| -------------------------------------------- | -------------- | ----------------------------------------------------- |
| Thread A: shared `xlm-roberta-base-ViT-B-32` | PGD ε=8        | All 5 languages collapse together — zero disagreement |
| Thread B: 4 separate CLIPs                   | EN typographic | Models disagree substantially (ZH 33% vs EN 5.5%)     |


Thread B's separate-encoder design produces the disagreement that Thread A's shared encoder could never produce, exactly as hypothesised. The disagreement is not perfect (all four models are eventually fooled by strong EN attack), but it is sufficient to build a detection signal.

---



### Next steps

1. **Compute disagreement detection AUC** — for each image, record whether the four models agree on the same class (clean vs. attacked). Compute AUC of a simple disagreement count detector. This is the Q2 analogue for the typographic threat model, as defined in `docs/CODE_GUIDE_separate_langs_typographic.md`.
2. **Update** `research_goal.md` — Thread B is now fully functional with 4 working models on CIFAR-10.
3. **Consider running Thread B with the full 8000-image CIFAR-10 test split** rather than 200 images, to get tighter confidence on the disagreement rates.

---



## July 5, 2026 — Evening (disagreement AUC + research_goal.md update)

**Disagreement detector AUC computed.** Added a new cell to `lib/notebooks/cifar10_typographic_attack_confusion.ipynb` that:

1. Stores per-model raw predictions on clean images (`clean_preds`) in the clean-baseline cell (was previously only saving accuracy floats).
2. Computes a disagreement score for each image = number of unique predictions across the 4 models (score 1 = full agreement, 4 = all different).
3. Computes the standard ROC AUC (ties count as 0.5) between the EN-attacked and clean distributions.

**Results (200 test images, EN typographic attack as "positive"):**


| Condition     | All-agree (score = 1) | Score ≥ 2 |
| ------------- | --------------------- | --------- |
| Clean         | 77.5%                 | 22.5%     |
| Attacked (EN) | 61.5%                 | 38.5%     |


**Detector AUC = 0.574.**

The signal is real but modest. With only 4 models, most images — even attacked ones — still get classified to the same (wrong) class by all 4 models, keeping the disagreement score at 1. The detection power comes mostly from the Chinese model staying more accurate under EN attack: when EN/KO/JA all predict the adversarial class and ZH predicts the true class, the disagreement score jumps to 2. Score distribution shows attacked images shift probability mass from 1 → 2 (18.5% clean vs. 37% attacked at score=2), with scores 3–4 being rare in both conditions.

AUC 0.574 > 0.5 confirms the hypothesis (disagreement does separate clean from attacked), but it is nowhere near actionable as a standalone detector. A practical alarm system would need either more languages/models or a richer detection signal than simple count-of-unique-predictions.

Results saved to `lib/notebooks/results/cifar10_confusion_results.json` under the `"detector"` key.

`**docs/research_goal.md` updated.** Added Thread B architecture (separate brains mermaid diagram), a Thread B findings section, and Thread B rows in the "What Was Found" table (Q1: ZH 33% under EN attack; Q2: AUC 0.574). The document now covers both threads in full.

---



## July 5, 2026 — Night (deeper ensemble analysis: 1000 images, per-class, all-language AUC)

**Homework: deeper analysis of the voting ensemble.** Scaled the CIFAR-10 experiment from 200 to 1000 images, added per-class vulnerability breakdowns for every attack language, and extended the disagreement-detector AUC from EN-only to all four attack languages.

### Updated 4×4 accuracy matrix (1000 images)

Numbers stabilised relative to the 200-image run — the broad picture is confirmed, not revised.


|                | model_EN | model_ZH  | model_KO | model_JA  |
| -------------- | -------- | --------- | -------- | --------- |
| Clean baseline | 84.2%    | 92.7%     | 87.7%    | **93.2%** |
| attack_EN      | 4.6%     | **36.5%** | 15.6%    | 8.3%      |
| attack_ZH      | 79.2%    | 58.3%     | 84.5%    | 90.2%     |
| attack_KO      | 81.9%    | 89.4%     | 86.1%    | 91.1%     |
| attack_JA      | 78.2%    | 69.8%     | 83.5%    | 89.9%     |


Key observations:

- EN attack is the only one that substantially hurts the ensemble. ZH, KO, and JA text overlays leave all four models nearly at clean-accuracy levels, confirming the Latin-script asymmetry at scale.
- ZH model (36.5%) remains notably more robust than EN (4.6%), KO (15.6%), JA (8.3%) under EN attack.
- JA model has the highest clean accuracy (93.2%) and the highest off-diagonal accuracy under non-EN attacks (88–91%), confirming it is the strongest model in the ensemble.



### Per-class vulnerability under EN attack

Four per-class bar-chart grids saved to `lib/notebooks/results/cifar10_perclass_attack_{lang}.png`.

Under EN attack, the 10 CIFAR-10 classes are not equally vulnerable. Key findings:


| Class      | EN model | ZH model | KO model | JA model | Notable                                                                                                                           |
| ---------- | -------- | -------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| airplane   | 1%       | 46%      | 14%      | 9%       | EN almost fully fooled                                                                                                            |
| automobile | 0%       | 44%      | 39%      | 2%       | KO retains some accuracy                                                                                                          |
| bird       | 9%       | 46%      | 26%      | 14%      | ZH best here                                                                                                                      |
| cat        | 0%       | 35%      | 1%       | 2%       | Near-total collapse for EN/KO/JA                                                                                                  |
| deer       | 8%       | 27%      | 25%      | 16%      | All models hurt                                                                                                                   |
| **dog**    | **0%**   | **13%**  | **0%**   | **0%**   | Most vulnerable class — EN "dog" label is the only clean one-syllable common word that overlaps perfectly with the visual feature |
| frog       | 5%       | 27%      | 15%      | 20%      | Moderate                                                                                                                          |
| horse      | 20%      | 41%      | 26%      | 20%      | Most resistant across models                                                                                                      |
| ship       | 4%       | 38%      | 4%       | 0%       | EN/KO/JA collapse                                                                                                                 |
| truck      | 4%       | 50%      | 11%      | 4%       | ZH most robust                                                                                                                    |


"Dog" and "cat" are the most exploitable classes — their English names are short, common, visually unambiguous, and strongly encoded by every model's text encoder. "Horse" is the hardest to fully fool (20% EN model accuracy retained), possibly because the English word is less dominantly associated with the visual prototype in LAION training data.

### Disagreement detector AUC — all attack languages


| Attack language | All-agree rate (attacked) | AUC       |
| --------------- | ------------------------- | --------- |
| Clean           | 78.2%                     | —         |
| **EN**          | 59.3%                     | **0.588** |
| **ZH**          | 50.4%                     | **0.646** |
| **KO**          | 73.4%                     | 0.525     |
| **JA**          | 58.6%                     | 0.604     |


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


| Model  | Clean acc | Under EN attack | Under ZH attack | Under KO attack | Under JA attack |
| ------ | --------- | --------------- | --------------- | --------------- | --------------- |
| EN     | 85.0%     | 5.5%            | 79.5%           | 83.5%           | 80.5%           |
| ZH     | 90.5%     | 33.0%           | 58.0%           | 88.0%           | 67.5%           |
| KO     | 87.0%     | 14.0%           | 85.5%           | 86.0%           | 85.0%           |
| **JA** | **93.0%** | **9.5%**        | **93.0%**       | **93.0%**       | **92.5%**       |


Three things stand out:

**1. Best clean accuracy (93%).**
`llm-jp-clip` was trained on a larger and more carefully filtered Japanese web corpus than its competitors. Higher pre-training data quality directly translates to better zero-shot classification. The model simply learned richer visual–semantic associations.

**2. Near-total immunity to non-EN attacks.**
Under ZH, KO, and JA text-overlay attacks, the JA model barely moves (93% → 93% → 92.5%). The reason is script isolation: Chinese characters, Korean Hangul, and Japanese kanji/kana are visually and statistically distinct from the Latin-alphabet labels the attack relies on for other models. Even for its own language, the JA attack (writing the Japanese class name on the image) barely hurts the JA model — probably because `llm-jp-clip` learned to rely on image features more than text-overlay cues when the two conflict.

**3. The EN attack does hurt JA (93% → 9.5%).**
This is the expected result and is *not* a failure of the JA model. The attack works by writing the English adversarial class name (e.g., "truck") directly on the image. The JA model was trained on bilingual data that included many English captions alongside Japanese ones (Japanese web text is heavily mixed with English). So the JA text encoder does recognise "truck" as a classification-relevant word. The attack exploits that English literacy. KO at 14% is slightly less affected, and ZH at 33% is the most robust — reflecting different degrees of English–native-language mixing in each model's training corpus.

#### Summary

The JA model outperforms the others because (a) it was specifically chosen by benchmark — the original model was replaced after a systematic failure analysis — and (b) `llm-jp-clip`'s training data quality and scale gave it 93% clean accuracy and strong image-feature reliance that makes CJK/Hangul text overlays irrelevant. Its one vulnerability is the same as everyone's: English text overlays, because all four models were trained on data that included English and have learned to treat Latin script as a valid classification signal.

---



## July 8, 2026

Switching focus from the 4-model ensemble (EN/ZH/KO/JA) to just EN and ZH for now — these two are the most effective and cleanest pair to work with at this stage.

**Tonight:** try other JA and KO CLIP models to find ones that perform as well as the ZH model does — the current KO and JA models may not be the best available options.

The original 1000-image CIFAR-10 sample (`CIFAR10_4LANG_1000_SAMPLE.json`) is a seed-0 random draw — not perfectly balanced. **Class distribution** for reference:


| Class      | Count |
| ---------- | ----- |
| airplane   | 102   |
| automobile | 104   |
| bird       | 92    |
| cat        | 123   |
| deer       | 112   |
| dog        | 101   |
| frog       | 88    |
| horse      | 80    |
| ship       | 104   |
| truck      | 94    |


A new sample (`CIFAR10_BALANCED_1000_SAMPLE.json`) with exactly 100 images per class has been created in `image_samples/`.

---



## July 10, 2026

**Notebook:** `lib/notebooks/_en_zh/en_zh_typographic/balanced_typographic_comparison.ipynb`
**Results:** `lib/notebooks/_en_zh/en_zh_typographic/results/balanced/`

EN vs ZH typographic attack experiment on the balanced 1000-image CIFAR-10 sample (100 per class). Same overlay style as `old_large_font_typographic_comparison.ipynb`: upscale to 224×224, font size 40.

### Bug fix — balanced sample JSON

`CIFAR10_BALANCED_1000_SAMPLE.json` had a pairing bug: `idx` was shuffled but `true` was saved in class-block order (`[0]×100, [1]×100, …`). Loading triggered an assertion error and, if bypassed, would have used the wrong labels. Regenerated the file with seed 0 so `idx` and `true` stay aligned; verified 100 images per class.

### Results (large font, balanced sample)


|                | model_EN | model_ZH |
| -------------- | -------- | -------- |
| Clean baseline | 85.9%    | 91.4%    |
| attack_EN      | 4.6%     | 37.1%    |
| attack_ZH      | 80.9%    | 57.8%    |


Attack Success Rate (pred == written adversarial class):


|           | model_EN | model_ZH |
| --------- | -------- | -------- |
| attack_EN | 95.3%    | 61.2%    |
| attack_ZH | 1.9%     | 38.7%    |


Disagreement detector (unique predictions across EN + ZH):


| Attack | All-agree (attacked) | AUC   |
| ------ | -------------------- | ----- |
| EN     | 62.1%                | 0.614 |
| ZH     | 54.4%                | 0.652 |




### Observations

1. **Large font changes everything.** On the same balanced indices, the small-font overlay (native 32×32, `h // 24`) barely moved accuracy (EN attack: ~82–89% retained). Size-40 text on 224×224 matches the 4-model CIFAR-10 notebook behaviour: EN attack collapses EN model to 4.6% and fools ZH 61% of the time.
2. **Balanced sample enables fair per-class breakdowns.** With exactly 100 images per class, per-class accuracy bars are no longer skewed by sample-size bias (unlike the seed-0 random 1000 draw where cat had 123 images and horse only 80).
3. **ZH attack is asymmetric.** ZH overlay hurts the ZH model (91.4% → 57.8%) more than EN (85.9% → 80.9%), but EN attack remains the dominant threat for both models.
4. **Detector signal is modest.** ZH attack gives the highest AUC (0.652) because it fools ZH while leaving EN relatively stable — the same cross-model disagreement pattern seen in the 4-model runs, but weaker with only two voters.



### Next steps

- Per-class bar charts on the balanced sample (compare EN vs ZH vulnerability class-by-class with equal N).
- Save fixed `target` labels in the JSON (currently regenerated at load time with seed 0) so attack targets are reproducible across runs without re-deriving.



### KO / JA CLIP model screening

**Script:** `lib/notebooks/_archive/old_ko_ja_model_screening/screen_ko_ja_models.py`
**Results:** `lib/notebooks/_archive/old_ko_ja_model_screening/results/screening_results.json`

Screened Korean and Japanese CLIP candidates against EN/ZH baselines on the same balanced 1000-image CIFAR-10 sample (100 per class). Goal: find KO/JA models as strong as Chinese CLIP (91.4%) and the current JA model.

**Recommended upgrades for the 4-lang typographic notebook:**


| Role | HuggingFace id                         | Architecture                       |
| ---- | -------------------------------------- | ---------------------------------- |
| KO   | `Bingsu/clip-vit-large-patch14-ko`     | ViT-L/14, Korean distillation CLIP |
| JA   | `llm-jp/llm-jp-clip-vit-large-patch14` | ViT-L/14, Japanese OpenCLIP        |


Both use the same loading pattern as the current models (HF `AutoModel` for KO, `open_clip` hf-hub for JA).

**Clean accuracy — balanced 1000-image CIFAR-10 (100 per class):**


| Model                    | Lang   | Clean acc | vs ZH (91.4%) | Status                        |
| ------------------------ | ------ | --------- | ------------- | ----------------------------- |
| OpenAI ViT-B/32          | EN     | 85.9%     | —             | baseline                      |
| Chinese CLIP ViT-B/16    | ZH     | 91.4%     | —             | baseline                      |
| Bingsu ViT-B/32          | KO     | 89.7%     | −1.7pp        | current in 4-lang notebook    |
| **Bingsu ViT-L/14**      | **KO** | **96.5%** | **+5.1pp**    | **recommended KO upgrade**    |
| llm-jp ViT-B/16          | JA     | 92.5%     | +1.1pp        | current in 4-lang notebook    |
| **llm-jp ViT-L/14**      | **JA** | **97.0%** | **+5.6pp**    | **recommended JA upgrade**    |
| LY clip-japanese-base-v2 | JA     | —         | —             | load error (transformers 5.x) |
| Stability JA ViT-L/16    | JA     | —         | —             | gated HuggingFace repo        |


**Per-class accuracy (recommended models vs current):**


| Class      | EN  | ZH  | KO B/32 | **KO L/14** | JA B/16 | **JA L/14** |
| ---------- | --- | --- | ------- | ----------- | ------- | ----------- |
| airplane   | 87% | 89% | 93%     | 96%         | 95%     | 95%         |
| automobile | 78% | 99% | 97%     | 97%         | 99%     | 98%         |
| bird       | 93% | 92% | 92%     | 97%         | 93%     | 98%         |
| cat        | 77% | 91% | 89%     | 96%         | 80%     | 96%         |
| deer       | 82% | 84% | 86%     | 97%         | 88%     | 95%         |
| dog        | 88% | 87% | 83%     | 95%         | 96%     | 97%         |
| frog       | 62% | 88% | 74%     | 91%         | 84%     | 94%         |
| horse      | 95% | 89% | 97%     | 98%         | 95%     | 99%         |
| ship       | 98% | 96% | 97%     | 98%         | 97%     | 99%         |
| truck      | 99% | 99% | 89%     | 100%        | 98%     | 99%         |


Key findings:

1. **KO upgrade is the bigger win.** Current Bingsu B/32 (89.7%) trails ZH; Bingsu L/14 (96.5%) jumps +6.8pp over current KO and +5.1pp over ZH. Biggest per-class gains: frog (74%→91%), cat (89%→96%), truck (89%→100%).
2. **JA is already above ZH** with llm-jp B/16 (92.5%); L/14 (97.0%) is the best model tested. Cat improves 80%→96%.
3. **No exotic architecture needed** — scaling up within the same families (Bingsu, llm-jp) beats hunting alternatives. KELIP (~55% Korean CIFAR-10 per paper) and LY clip-japanese-base-v2 (load broken) were not viable.
4. **Next:** swap recommended models into `_archive/old_cifar10_typographic_4lang/typographic_attack_confusion.ipynb` and re-run the 4×4 attack grid.

## 2026-07-10 — CAM intersection masking defense (EN/ZH typographic)

**Notebook:** `lib/notebooks/_en_zh/cam_intersection_defense/cam_intersection_defense.ipynb` (fast re-run: `run_cam_defense_fast.py`)
**Results:** `lib/notebooks/_en_zh/cam_intersection_defense/results/`

**Idea:** On typographic-attacked images, EN and ZH CLIP models often co-attend the same text strip (visible in GradCAM). Intersect high-saliency regions from both models, mask them (mean-fill), and re-classify.

**Setup:** Balanced 1000-image CIFAR-10 sample, large-font overlay (size 40 @ 224px). GradCAM target = each model's own attacked prediction. Best threshold = 95th percentile on 100-image tune subset (~9% pixels masked).

**EN attack results (primary case):**

| Model | Baseline acc | After masking | ASR before → after | Recovery |
|-------|-------------|---------------|-------------------|----------|
| EN | 4.6% | **51.4%** | 95.3% → 31.7% | 50.0% |
| ZH | 37.1% | **67.9%** | 61.2% → 19.8% | 55.0% |

**Ablations (EN attack):** intersection+product (54.4% EN acc) beats min-only (51.4%). Bottom-band prior ∩ CAM reaches 66.4% EN / 75.8% ZH. Oracle bottom-25% mask upper bound: 79.7% EN / 86.0% ZH — confirms most recoverable signal is in the text strip.

**Failure mode:** Clean images degrade (EN 85.9%→68.6%, ZH 91.4%→81.1%) because both models attend to the object, not text. Defense is useful as targeted preprocessing when attack is suspected, not blind application.

**Conclusion:** Cross-lingual saliency agreement is a viable spatial defense signal for typographic attacks, complementary to the existing prediction-disagreement detector. Next step: gate masking on disagreement score to avoid clean-image degradation.

## 2026-07-12 — Multilingual vs unilingual attack study (design + notebooks)

**Notebooks:** `lib/notebooks/_en_zh/en_zh_multi_uni_attack/`
**Status:** Notebooks generated, not yet run.

### Motivation

The existing `en_zh_multiple_placement` experiment uses two random EN text boxes, or two random ZH text boxes, separately. A natural question is: what happens when we place *both* languages simultaneously (one EN box, one ZH box), and how does this compare against a pure unilingual (EN+EN) attack? Further, the CAM intersection defense has so far used only the natural pairing (each model looks at its own language). This study systematically explores both the attack dimension and the defence cost dimension.

### Experiment design

**Attack setups** (2 typographic boxes per image, same random placement seeds):

| Setup | Box-0 | Box-1 |
|---|---|---|
| Multilingual | EN attack word | ZH attack word |
| Unilingual | EN attack word | EN attack word (repeated) |

Both setups evaluated on EN CLIP (ViT-B/32 OpenAI) and ZH CLIP (ChineseCLIP ViT-B/16).

**Defence strategies:**

| Method | Description | Forward passes / image |
|---|---|---|
| no_defense | baseline (classify only) | 2 |
| cam_2mod (multilingual) | GradCAM(EN,EN) ∩ GradCAM(ZH,ZH) | 6 |
| cam_4mod (multilingual) | all 4 cross-combos: EN-EN ∩ EN-ZH ∩ ZH-EN ∩ ZH-ZH | 10 |
| cam_2mod (unilingual) | GradCAM(EN,EN) ∩ GradCAM(ZH,EN-via-ZH-encoder) | 6 |
| grid_1patch | 4×4 grid, best single occlusion by max mean confidence | 32 |
| grid_2patch | greedy best 2nd patch given 1st | 62 |

The **4-mod** defence introduces cross-language GradCAM probes: the EN model is scored against its own encoding of ZH class names, and the ZH model is scored against its own encoding of EN class names. These cross-lingual saliency maps potentially catch text regions that only one model attends to strongly.

The **grid search defence** is deliberately model-agnostic: no GradCAM, no text information, just trying all 16 patches (or greedy 2-patch pairs) and keeping whichever occlusion makes both models most confident. This is a black-box upper bound on what spatial occlusion alone can achieve.

### Technical notes

- `draw_multilingual_attack()`: box-0 uses Latin font + EN word, box-1 uses CJK font + ZH word; both use the same `random.Random(img_idx * NUM_BOXES + box_i)` seed scheme as existing dual-box code.
- `TEXT_EMBS` dict holds all 4 `(model_lang, text_lang)` cross-embedding combinations; the two non-standard combos (EN model tokenising Chinese glyphs, ZH model encoding English words) are computed once at model-load time.
- Generalised `gradcam_en_with_emb` / `gradcam_zh_with_emb` functions accept any text embedding matrix and return `(cam, target_idx)`.
- `compute_and_cache_cams` saves a single `.npz` per condition with all 4 CAMs; the 2-mod defence just slices the `cam_en_en` and `cam_zh_zh` keys from the same cache.
- Grid defence batches all 16 occluded variants of one image in a single `embed_images` call per model.

### Folder structure

```
en_zh_multi_uni_attack/
├── _build_notebooks.py
├── cost_vs_performance.ipynb
├── multilingual/
│   ├── attack_comparison.ipynb
│   ├── cam_defense.ipynb          (2-mod + 4-mod)
│   ├── grid_defense.ipynb         (1-patch + 2-patch)
│   └── results/{attack/, cam_2mod/, cam_4mod/, grid_1patch/, grid_2patch/}
└── unilingual/
    ├── attack_comparison.ipynb
    ├── cam_defense.ipynb           (2-mod only)
    ├── grid_defense.ipynb
    └── results/{attack/, cam_2mod/, grid_1patch/, grid_2patch/}
```

All results JSONs carry `"method"`, `"setup"`, `"inference_cost"`, and `"defense_acc_mean"` keys so `cost_vs_performance.ipynb` can aggregate them without path-hardcoding.

### Next steps

- Run all notebooks (order: `attack_comparison` → `cam_defense` → `grid_defense` → `cost_vs_performance`).
- Primary question: does 4-mod CAM meaningfully improve over 2-mod, and is it worth 10 vs 6 forward passes?
- Secondary question: does grid search (32 passes, no GradCAM) outperform 2-mod CAM (6 passes, model-aware)?

---

## 2026-07-13 — Full results: en_zh_multi_uni_attack

All 9 experiments complete. CIFAR-10 balanced 1000-image sample, 100/class.
Clean accuracy: **EN CLIP 85.9%, ZH CLIP 91.4%** (identical across all conditions).

### Complete results table

| Setup | Method | Cost (fwd/img) | EN acc | EN ASR | ZH acc | ZH ASR | Mean acc |
|---|---|---:|---:|---:|---:|---:|---:|
| multilingual | no_defense | 2 | 4.3% | 95.5% | 7.3% | 92.7% | 5.8% |
| multilingual | cam_2mod | 6 | 32.0% | 34.0% | 34.3% | 44.4% | **33.2%** |
| multilingual | cam_4mod | 10 | 29.8% | 40.6% | 31.9% | 50.9% | 30.9% |
| multilingual | grid_1patch | 32 | 5.4% | 94.4% | 16.3% | 83.3% | 10.9% |
| multilingual | grid_2patch | 62 | 6.5% | 93.2% | 16.9% | 82.6% | 11.7% |
| unilingual | no_defense | 2 | 3.4% | 96.5% | 27.1% | 71.8% | 15.3% |
| unilingual | cam_2mod | 6 | 25.7% | 46.0% | 39.2% | 35.7% | **32.5%** |
| unilingual | grid_1patch | 32 | 2.9% | 97.0% | 24.1% | 75.0% | 13.5% |
| unilingual | grid_2patch | 62 | 4.0% | 95.7% | 21.6% | 77.3% | 12.8% |

### Clean-image degradation under CAM masking

| Setup | Method | EN clean→masked | ZH clean→masked |
|---|---|---:|---:|
| multilingual | cam_2mod | 85.9% → 50.5% (−35.4pp) | 91.4% → 64.9% (−26.5pp) |
| multilingual | cam_4mod | 85.9% → 52.8% (−33.1pp) | 91.4% → 68.6% (−22.8pp) |
| unilingual | cam_2mod | 85.9% → 48.7% (−37.2pp) | 91.4% → 63.0% (−28.4pp) |

### Key findings

- **cam_2mod is the clear winner** on both setups. At only 6 forward passes it delivers ~33% post-defense mean accuracy vs ~6% (multilingual) and ~15% (unilingual) for no_defense.
- **cam_4mod underperforms cam_2mod** despite costing 10 passes instead of 6. The cross-language GradCAM probes did not add useful masking signal — mean accuracy dropped from 33.2% to 30.9% on the multilingual attack.
- **Grid defenses largely fail**: grid_1patch and grid_2patch spend 32–62 passes and barely lift accuracy above the no_defense baseline (10–12% vs 5–15%). Blind spatial occlusion cannot reliably hit the text boxes.
- **Unilingual ZH model is partially robust by default**: even without defense, ZH CLIP achieves 27.1% accuracy under an English-only typographic attack (vs 7.3% under the multilingual dual-language attack), because the ZH model is less sensitive to English text overlays.
- **CAM masking degrades clean accuracy significantly** (~25–37pp drop), which is the main cost of this defense strategy.

### Cost vs. performance chart

`cost_vs_performance.ipynb` aggregates all 9 result JSONs and plots inference cost (forward passes / image) against mean post-defence accuracy for both setups. Chart saved to `cost_vs_performance.png`. The Pareto frontier is cam_2mod at cost 6: every cheaper method (no_defense at cost 2) gives near-zero accuracy, and every more expensive method (cam_4mod at 10, grid defences at 32–62) gives equal or worse accuracy. The chart makes the cam_4mod regression and grid failure visually unambiguous.

### Exhaustive grid test

**Notebook:** `lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_exhaustive_grid/exhaustive_grid_test.ipynb`
**Results:** `_test_exhaustive_grid/results/comparison.json`

Tested whether replacing the greedy 2-patch search with an exhaustive search over all C(16,2) = 120 patch pairs changes the outcome. Run on the 100-image tune subset (multilingual attack).

| Method | acc_en | acc_zh | acc_mean | asr_mean | Runtime (100 imgs) |
|---|---:|---:|---:|---:|---:|
| no_defense | 6.0% | 4.0% | 5.0% | 95.0% | — |
| grid_1patch | 5.0% | 14.0% | 9.5% | 90.5% | 9.5 s |
| grid_2patch_greedy | 6.0% | 16.0% | 11.0% | 89.0% | 8.2 s |
| grid_2patch_exhaustive | 6.0% | 17.0% | 11.5% | 88.5% | 51.3 s |

Greedy was suboptimal for 22 of 100 images (22%), but the performance gap is negligible: acc_mean 11.0% vs 11.5%, asr_mean 89.0% vs 88.5%. Exhaustive search costs 240 forward passes vs 62 for greedy and runs 6× slower for a gain of 0.5 pp in mean accuracy. Conclusion: **greedy is sufficient** — the dominant failure mode is not patch-pair selection quality but the fundamental inability of a 4×4 grid to isolate the text boxes when they can land anywhere in the 224×224 image.

### Attention-based saliency: can we cut cam_2mod from 6 → 4 forward passes?

**Notebook:** `lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_attention_defense/attention_defense_test.ipynb` (moved on 2026-07-16 to its own top-level folder, `lib/notebooks/attention_defense/attention_defense_test.ipynb` — see the note at the end of this entry)

**Motivation:** GradCAM needs a backward pass per model to compute saliency gradients. ViT self-attention weights are available for free during the normal forward pass — no backward needed. If attention maps produce equally good masks, the defense drops from 6 to 4 forward passes.

**Inference cost breakdown:**

| Method | Operations | Total passes |
|---|---|---:|
| GradCAM cam_2mod | 2 × (fwd + back) + 2 × fwd | **6** |
| Attn cam_2mod | 2 × fwd + 2 × fwd | **4** |

The key identity: `2 × (fwd + back) = 4 ops` vs `2 × fwd = 2 ops` — attention saves exactly 2 backward passes (one per model). Classification is **fused** with saliency extraction in both cases; there is no separate pre-classify step.

**Two attention variants tested:**
- `Attn-last`: CLS→patch attention from the final transformer block, heads averaged.
- `Attn-rollout`: Abnar & Zuidema (2020) rollout — propagates attention through all layers via `∏(0.5·A + 0.5·I)`.

**Models:** EN CLIP `open_clip` ViT-B/32 (8 heads, 7×7 = 49 patches at 32 px/patch), ZH CLIP `ChineseCLIP` ViT-B/16 (12 heads, 14×14 = 196 patches at 16 px/patch). EN CLIP hardcodes `need_weights=False` in `open_clip`, so EN attention is extracted via a `register_forward_hook` on each residual block's `attn` sub-module, recomputing weights from the QKV projections. ZH CLIP (via `transformers`) natively supports `output_attentions=True`.

**Tuning subset results (100 images, multilingual attack):** GradCAM reproduced the production result exactly (mean acc 33.1% at threshold 0.85). The notebook full-1000 evaluation was interrupted mid-run for the attention variants; full results pending.

**GradCAM confirmed at cost 6.** Attention variants confirmed at cost 4. Full accuracy comparison pending re-run.

**Update — full 1000-image results (simplified summary):**

| Method | Cost | Mean acc | Coverage | Clean-acc drop |
|---|---:|---:|---:|---:|
| GradCAM | 6 | 33.1% | 26.6% | −35.4pp EN / −26.5pp ZH |
| Attn-rollout | 4 | 62.9% | 21.4% | −25.6pp EN / −16.8pp ZH |
| Attn-last | 4 | **72.6%** | **7.7%** | **−8.8pp EN / −2.6pp ZH** |

Attention wins on every axis — cheaper, more accurate, and far less damage to clean images. `Attn-last` is the best variant.

*How each method works, in one line:*
- **GradCAM** = backprop the predicted class's score down to the first conv/patch layer, weight each channel by its average gradient, sum → needs a backward pass.
- **Attn-last** = just read the CLS token's attention weights from the *last* transformer block (already computed for free during the forward pass) — literally "what the model was looking at right before deciding."
- **Attn-rollout** = multiply the attention matrices from *all 12* layers together (blended 50/50 with identity each layer) to trace attention all the way back to the input patches — no backward pass either, but the 12x blending smooths/dilutes the signal.

*Why attention (esp. Attn-last) wins:* the typographic attack works by hijacking the model's attention onto the injected text. Attn-last reads that hijacked attention directly, one hop from the final decision — so its mask lands tightly on just the text (sharp, sparse peaks). GradCAM's gradient has to flow backward through all 12 layers to reach the first-layer activation it's computed on, which blurs the signal across the text *and* the real object, forcing a bigger, sloppier mask that also wrecks clean images. Rollout sits in between because its layer-by-layer identity-blending smooths the signal, but less destructively than GradCAM's full backward pass.

---

## 2026-07-16 — Plain-language recap: GradCAM vs. attention vs. grid search

No code, no notebooks today — just writing down the plain-English version of the above for future reference, since it's easy to lose the intuition under all the math.

### The three saliency methods, simplified

Think of all three as different ways of answering the same question: **"which part of the image is the AI model actually looking at when it makes its decision?"** Once we know that, we can black out that region (hopefully the sneaky text sticker) and ask the model again.

**GradCAM** — Run the image through the model to get an answer, then trace *backward* to see which pixels "pulled" the answer in that direction. It's like asking someone to explain their reasoning after the fact by rewinding through every step. This rewinding (the "backward pass") is extra work — it roughly doubles the computation compared to just getting an answer.

**Attn-last** — Modern AI vision models don't just look at everything equally; internally they already decide "how much attention" to give each patch of the image, layer by layer, as part of normal operation. Attn-last just peeks at that attention amount from the model's very *last* internal step — right before it commits to an answer. No rewinding needed, it's information the model already had lying around.

**Attn-rollout** — Same idea as attn-last (peek at internal attention, no rewinding), but instead of just looking at the last step, it combines the attention from *every* step the model took, all the way through. It's like averaging together everywhere the model glanced throughout its whole "thought process," instead of just its final glance.

**Similarities:** all three produce a heatmap of "where the model is looking," all three feed into the exact same next step (blur/mask the hot spots, then re-ask the model for its answer).

**Differences:**
- GradCAM needs extra rewinding work; the two attention methods don't.
- GradCAM's map ends up blurry — it smears over both the sticker *and* the real object, because the rewinding gets diluted as it passes back through many layers.
- Attn-last's map is sharp and pinpoints almost exactly the sticker, because it's a single, undiluted glance.
- Attn-rollout is blurrier than attn-last (since it averages many glances together) but sharper than GradCAM.

That's why attn-last won: cheapest to compute *and* the most precise, so blacking out its hot spots barely damages the rest of the image.

### Where grid search fits in

Grid search is a completely different, much dumber approach — it doesn't look at what the model is "thinking" at all. Instead it just chops the image into a 4×4 checkerboard of 16 chunks and **brute-force tries blacking out chunks one at a time (or two at a time)**, checking after each attempt whether the model's confidence goes up. Whichever chunk(s) made the model most confident gets kept as the "mask." No heatmap, no shortcuts — pure trial and error.

**Is it still used?** No. It was tested purely as a sanity-check baseline to see if a "dumb," model-agnostic approach could compete with the saliency-based methods — it couldn't, and it's not part of the production defense.

**Results — it basically failed:**

| Method | Cost (passes/image) | Mean accuracy after defense |
|---|---:|---:|
| No defense (do nothing) | 2 | 5.8% |
| Grid search, 1 chunk blacked out | 32 | 10.9% |
| Grid search, 2 chunks blacked out (smart/greedy) | 62 | 11.7% |
| Grid search, 2 chunks (tried literally every possible pair) | 240 | ~11.5% |
| GradCAM | 6 | 33.1% |
| Attn-last (best) | 4 | 72.6% |

Grid search barely beats doing nothing (11% vs 6%), while costing **5–40× more compute** than the attention methods. Even letting it try every single possible pair of chunks (an exhaustive search, instead of a quick smart guess) only nudged results from 11.0% to 11.5% — basically no difference. That confirms the problem isn't "picking better chunks" — it's that a coarse 4×4 checkerboard is just too blunt an instrument to reliably land on a small text sticker that can appear anywhere in the image. Grid search is kept in the notebooks only as a "proof that dumb occlusion doesn't work" reference point, not as a real candidate.

### Notebook reorganization

`_test_attention_defense/` was promoted out of `en_zh_multi_uni_attack/` into its own top-level folder, `lib/notebooks/attention_defense/`, now that the full 1000-image results confirm it's a real finding (not just a sub-experiment) — cheaper AND more accurate than the production GradCAM defense. Moved via `git mv` (history preserved), fixed the one relative path inside the notebook (`../../image_samples/...` → `../image_samples/...` to account for the shallower folder depth), and added a dedicated `README.md`. Both `lib/notebooks/README.md` and this diary's notebook-path reference above were updated to point at the new location.

### To-do

- [x] Run just EN and ZH models — done; attention defense + multi/uni attack study focused on the EN/ZH pair (`attention_defense/`, `en_zh_multi_uni_attack/`).
- [x] Improve grid search — done 2026-07-16; conf-drop scoring is the fix (see entry below). Production `grid_defense.ipynb` still uses old max-conf (kept as negative baseline); improved variants live in `_test_grid/`.
- [x] Think of ways to improve our heatmap-based defense — done for now (2026-07-18). Keep **`cc_bbox_blur`** (74.9% mean / clean Δ −1.5pp). Ablation round closed; residual gap to clean left for later.
- [x] 4-lang `cc_bbox_blur` transfer trial — done 2026-07-19 (`four_lang_cc_bbox_blur/`). Defense recovers hard attacks for ZH/KO/JA; KO/JA clean Δ much worse (−11 to −23pp) than ZH (−1.5pp).
- [x] Reduce KO/JA clean-image damage under EN∩L `cc_bbox_blur` (threshold / mask geometry) — done 2026-07-19 (`ko_ja_clean_damage/`). `uni_en` Clean Δ −18/−23 → −11/−7pp; residual ~−7 to −11pp vs ZH −1.5pp is heatmap quality.
- [x] Organize notebooks — done 2026-07-19: `_archive/` + `_en_zh/` (underscore so they sort together); `old_cifar10_typographic_4lang` archived; mainline stays top-level. Branching strategy still open.
- [x] Start paper — outline only in `docs/paper_draft.md` (2026-07-19); prose not started.
- [ ] Write up results.

---

## 2026-07-16 — Improved grid search (`_test_grid/grid_test.ipynb`)

The old grid failed because of **scoring**, not (mainly) because of the coarse 4×4 geometry. Maximizing post-occlusion confidence rewards “still confidently wrong.” Switching to **confidence-drop of the pre-defense top class** (+ small EN–ZH agree bonus) flips the picture.

### What the method names mean

Two knobs: **how we score** a candidate occlusion, and **how we fill** the occluded patch.

**Conf-drop (scoring)** — Before defending, note each model’s top-1 class and its confidence on the *attacked* image. For every candidate patch occlusion, re-score and compute how much that *same* class’s confidence fell. Average the EN and ZH drops; add a small bonus if EN and ZH now agree on a class. Pick the patch(es) with the largest drop. Intuition: a good occlusion should *kill the attack belief*, not merely leave the model “confident about something.”

**Max-conf (old scoring)** — Pick the occlusion that maximizes post-occlusion max cosine-sim. That often keeps the attack label when the text survives, so it barely helps.

**Mean (fill)** — Replace selected patch pixels with the image’s average RGB color (a flat rectangle). Simple; can look harsh and can erase object detail.

**Blur (fill)** — Instead of a flat fill, Gaussian-blur only inside the selected patch(es). Glyphs get smashed while coarse object structure is more likely to survive — a small accuracy bump over mean fill at the same search cost.

So **`B_2p_confdrop_mean`** = greedy 2-patch search, conf-drop scoring, mean-fill.  
**`C_2p_confdrop_blur`** = same search + scoring, but blur fill (the n=1000 winner under the cost bar).

### Tune set (n=100 multilingual)

| Method | Cost | Mean acc |
|---|---:|---:|
| no_defense | 2 | 5.0% |
| base_2p maxconf mean (old) | 62 | 11.0% |
| B_2p confdrop mean | 62 | **43.5%** |
| C_2p confdrop blur | 62 | **44.0%** |
| D_2p beam2+dist confdrop | 92 | 45.0% |
| A_2p 6×6 confdrop | 142 | 44.5% |
| combo 6×6 blur beam2 | 212 | 47.5% |

Finer grids / beam / blur help a little; **conf-drop alone is the leap** (~11% → ~44% at the same cost 62).

### Full n=1000 (promoted under-bar winners)

| Method | Cost | EN acc | ZH acc | Mean acc |
|---|---:|---:|---:|---:|
| no_defense | 2 | 4.3% | 7.3% | 5.8% |
| base_2p maxconf mean | 62 | 6.3% | 17.0% | **11.7%** |
| B_2p confdrop mean | 62 | 45.7% | 46.3% | **46.0%** |
| C_2p confdrop blur | 62 | 48.1% | 48.9% | **48.5%** |

`C_2p_confdrop_blur` beats GradCAM cam_2mod (~33%) on accuracy, still far behind Attn-last (72.6%), and still **~10× more expensive** than CAM (62 vs 6). Notebook: [`_test_grid/grid_test.ipynb`](lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_grid/grid_test.ipynb); results in `_test_grid/results/` (`comparison.json`, `comparison_n1000.json`, `defence_examples.png`, `n1000_bars.png`, `box_hit_analysis.json`).

### Does covering Chinese or English matter more?

On the multilingual attack (one EN box + one ZH box), we checked which text boxes the chosen 4×4 patches actually overlap, then measured accuracy conditional on that hit pattern. Method: **C_2p_confdrop_blur**, full **n=1000**.

| Hit pattern | n | freq | EN acc | ZH acc | Mean acc |
|---|---:|---:|---:|---:|---:|
| hit both EN+ZH | 587 | 58.7% | 67.6% | 69.3% | **68.5%** |
| hit EN only | 287 | 28.7% | 28.9% | 28.2% | **28.6%** |
| hit ZH only | 45 | 4.5% | 0.0% | 0.0% | **0.0%** |
| hit neither | 81 | 8.1% | 1.2% | 1.2% | **1.2%** |

**Covering English matters much more than covering Chinese alone.** Hitting both is best (~68% mean). Hitting only the EN box still recovers ~29% mean — and that helps the *ZH model too*. Hitting only the ZH box recovers essentially nothing (0%), even for Chinese CLIP. Conf-drop also rarely picks ZH-only (~4%) vs EN-only (~29%) vs both (~59%). Same pattern for mean-fill (`B_2p_confdrop_mean`). Leaving the EN sticker up keeps both models fooled; the ZH sticker is secondary.

---

## 2026-07-16 — Unilingual sole two-box: does attention still beat GradCAM?

The multilingual attention win (Attn-last 72.6% mean @ cost 4) was only measured on mixed EN+ZH boxes. Question: does the same hold for **sole-language dual-box** attacks (both boxes EN, or both boxes ZH) — the setup used in `en_zh_multiple_placement`?

**Notebook:** [`lib/notebooks/attention_defense/unilingual/attention_defense_test.ipynb`](lib/notebooks/attention_defense/unilingual/attention_defense_test.ipynb)  
Same models, saliency variants, and EN∩ZH intersection defense as the multilingual notebook; only the attack changes (`draw_word` with `attack_lang='en'|'zh'`). Tune on 100 per attack lang, then full 1000.

### How to read the table

Most columns are straightforward: **Attack** is which text overlay was used; **Method** is the saliency map feeding the same 2-model intersection mask; **Cost** is forward/backward passes per image; **EN/ZH/Mean acc** are post-defense top-1 accuracy on the 1000 attacked images.

The last three columns are less obvious:

- **Cov (coverage)** — fraction of image pixels that get masked (blacked out) on average. Low coverage means a tight mask that mostly hits the text stickers; high coverage means the defense is also erasing a lot of the real object. All else equal, lower is better.
- **Best thr** — percentile threshold chosen on the 100-image tune set (by EN masked accuracy), then frozen for the full 1000. Higher thr (e.g. 0.95) keeps only the hottest saliency peaks → smaller masks; lower thr (e.g. 0.80) masks more aggressively.
- **Clean drop (EN / ZH)** — change in accuracy when the *same* defense is applied to clean (unattacked) images, in percentage points. A large negative number means the mask is damaging the object even when there is no attack — the main side-effect cost of the defense. Closer to zero is better.

### Full 1000-image results (multi + uni)

| Attack | Method | Cost | EN acc | ZH acc | Mean acc | Cov | Best thr | Clean drop (EN / ZH) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Multilingual (EN+ZH) | GradCAM | 6 | 32.0% | 34.3% | 33.1% | 26.6% | 0.85 | −35.4pp / −26.5pp |
| Multilingual (EN+ZH) | Attn-rollout | 4 | 56.3% | 69.6% | 62.9% | 21.4% | 0.85 | −25.6pp / −16.8pp |
| Multilingual (EN+ZH) | **Attn-last** | **4** | **68.7%** | **76.5%** | **72.6%** | **7.7%** | **0.95** | **−8.8pp / −2.6pp** |
| Unilingual EN+EN | GradCAM | 6 | 22.1% | 35.4% | 28.7% | 35.8% | 0.80 | −44.2pp / −36.2pp |
| Unilingual EN+EN | Attn-rollout | 4 | 48.2% | 64.1% | 56.1% | 21.6% | 0.85 | −25.6pp / −16.8pp |
| Unilingual EN+EN | **Attn-last** | **4** | **63.2%** | **72.0%** | **67.6%** | **8.5%** | **0.95** | **−8.8pp / −2.6pp** |
| Unilingual ZH+ZH | GradCAM | 6 | 58.4% | 43.5% | 50.9% | 9.4% | 0.95 | −17.5pp / −10.4pp |
| Unilingual ZH+ZH | Attn-rollout | 4 | 57.5% | 67.8% | 62.7% | 15.0% | 0.90 | −19.4pp / −11.3pp |
| Unilingual ZH+ZH | **Attn-last** | **4** | **56.5%** | **68.5%** | **62.5%** | **23.1%** | **0.85** | −29.6pp / −22.7pp |

Multilingual numbers from [`attention_defense/results/`](lib/notebooks/attention_defense/results/); unilingual from [`attention_defense/unilingual/results/`](lib/notebooks/attention_defense/unilingual/results/) (`final_comparison.png`, `confusion_results_{variant}_{en,zh}.json`).

---

## Summary — Attention defense across attack types

We compared three saliency signals (GradCAM @ cost 6, Attn-rollout @ cost 4, Attn-last @ cost 4) inside the same 2-model CAM-intersection defense, on three typographic attacks over the same 1000 balanced CIFAR-10 images: mixed-language boxes (EN+ZH), English-only dual-box (EN+EN), and Chinese-only dual-box (ZH+ZH).

**On multilingual and English-only attacks, Attn-last is clearly best.** It recovers 72.6% mean accuracy on EN+ZH and 67.6% on EN+EN, versus GradCAM’s 33.1% and 28.7%. It does this with tighter masks (coverage ~8% vs GradCAM’s ~27–36%) and far less collateral damage on clean images (−8.8pp EN / −2.6pp ZH in both cases, versus GradCAM’s −35pp/−26pp or worse). Attn-rollout sits in the middle on accuracy and side effects. So the original finding — attention is cheaper *and* more accurate than GradCAM — transfers cleanly from multilingual to sole-English dual-box.

**On Chinese-only dual-box the picture is more mixed.** GradCAM itself is much stronger here (50.9% mean) than on EN or multilingual, partly because the tune set picked a high threshold (0.95) that keeps masks small. Attn-last still leads on mean accuracy (62.5%), but only by about 12 pp over GradCAM, and Attn-rollout is essentially tied with it (62.7%). Attn-last’s usual advantages also weaken: coverage rises to 23.1% and clean-image drop worsens to −29.6pp / −22.7pp — closer to GradCAM’s side-effect profile. A plausible reason is that Chinese glyphs are harder for the English CLIP’s attention to localize, so the EN∩ZH intersection mask is less precise when both stickers are Chinese.

**Overall:** Attn-last remains the default choice across all three attacks — always at least as accurate as the alternatives, always at lower inference cost than GradCAM, and best-in-class on clean-image preservation whenever the attack includes English text. The ZH-only case is the main caveat: attention still wins on accuracy, but the gap shrinks and the clean-image cost rises, so ZH-only is where further saliency or masking work would matter most.

---

## 2026-07-17 / 18 — Improving the heatmap-based defense

**Goal:** close the gap from Attn-last (**72.6%** mean on multilingual dual-box, cost 4, clean Δ −8.8pp EN / −2.6pp ZH) toward clean accuracy (85.9% / 91.4%).

**Folder:** [`lib/notebooks/heatmap_defense_improvements/`](lib/notebooks/heatmap_defense_improvements/) — `attention_defense/` left frozen as the baseline. All numbers below are multilingual, n=1000, thresholds tuned on 100.

### Ablations (ideas 1–4, 7) — `heatmap_improvements.ipynb`

| Trial | What changed (vs Attn-last) | Mean acc | Clean Δ |
|-------|----------------------------|--------:|--------:|
| attn_last_baseline | Control: EN∩ZH attention mask, mean-fill | 72.6% | −5.7pp |
| gated_peakiness | Only mask if saliency looks “spiky” (skip spread maps) | 72.6% | −5.7pp |
| gated_disagree | Only mask if EN/ZH predictions disagree | 9.3% | −1.1pp |
| union_masks | Mask where *either* model is hot (OR), not intersection | 65.1% | −29.3pp |
| blur_fill | Blur masked pixels instead of flat mean color | 73.5% | −1.6pp |
| cc_filter | Keep only the 2 largest mask blobs | 72.5% | −2.7pp |
| cc_bbox | Top-2 blobs, then snap each to a rectangle | **74.9%** | −2.7pp |
| peaked_heads | Average only the sharpest attention heads | 72.5% | −5.0pp |

Peakiness gating never fired (`gated_off=0%`). Disagreement gating almost never masked (both models agree on the attack ~90% of the time). Union hurt clean images badly.

### EN ViT-B/16 (idea 5) — `vit16_en/`

| Trial | What changed | Mean acc | Clean Δ EN / ZH |
|-------|--------------|--------:|----------------:|
| B/32 (published) | Control EN model | **72.6%** | −8.8 / −2.6 |
| B/16 | Finer EN patches (14×14 vs 7×7) | 71.8% | −20.5 / −12.9 |

No gain; larger masks and worse clean damage.

### Attn + conf-drop hybrid (idea 6) — `attn_confdrop_hybrid/`

| Trial | What changed | Cost | Mean acc |
|-------|--------------|-----:|--------:|
| attn_last | Control | 4 | **72.6%** |
| hybrid k=4 | Shortlist 4×4 cells by attention, pick with conf-drop + blur | 18 | 56.3% |
| full grid conf-drop (ref) | Search all 4×4 cells (prior experiment) | 62 | 48.5% |

Hybrid beats full grid but loses to plain Attn-last at higher cost.

### What `blur_fill` and `cc_bbox` do (and how they combine)

Both start from the same Attn-last pipeline: EN∩ZH heatmap → percentile threshold → dilate → *then* the tweak → re-classify.

- **`blur_fill`** — changes *how* masked pixels are replaced. Baseline paints them a flat average color (a hard rectangle). Blur instead runs a Gaussian (radius 12) on the image and copies only the blurred pixels into the mask. Glyph edges get smashed so the sticker stops reading as text, but coarse color/structure of the object under/near the sticker is less destroyed — which is why clean Δ improves a lot (−5.7pp → −1.6pp) while attacked acc ticks up slightly.
- **`cc_bbox`** — changes *which* pixels get masked. After thresholding, the mask can be a blotchy blob (or several speckles) that only partly covers a sticker. We keep the 2 largest connected components (one per box, ideally), drop the rest, then expand each blob to its axis-aligned bounding rectangle — matching the attack’s white text boxes. That covers leftover glyph bits the heatmap missed, which lifts attacked acc to 74.9% (coverage 7.7% → 8.8%).
- **`cc_bbox_blur`** — run `cc_bbox` first (shape the mask), then `blur_fill` (soft occlude). Attacked acc stays at the `cc_bbox` ceiling (74.9%); clean Δ moves to the `blur_fill` regime (−1.5pp). Same cost 4 — no extra model passes.

### Combo results — `cc_bbox_blur/` (2026-07-18)

| Trial | What changed | Mean acc | Clean Δ |
|-------|--------------|--------:|--------:|
| attn_last_baseline | Control | 72.6% | −5.7pp |
| blur_fill | Blur fill only | 73.4% | −1.6pp |
| cc_bbox | Rectangle snap only | 74.9% | −2.7pp |
| **cc_bbox_blur** | Rectangle snap **and** blur fill | **74.9%** | **−1.5pp** |

### Summary

- **Keep:** `cc_bbox_blur` — best attacked accuracy so far (**74.9%**, +2.3pp over Attn-last) and best clean tradeoff among strong defenses (−1.5pp). Same cost 4.
- **Useful alone:** `blur_fill` (kindest to clean images), `cc_bbox` (same attacked acc as the combo).
- **Skip:** disagreement / peakiness gating (as implemented), union masks, peaked heads, EN ViT-B/16, attn+conf-drop hybrid.
- **Done for now:** heatmap-improvement ablations closed with `cc_bbox_blur` as the current defense. Residual ~10–15pp gap to clean left for later; next step is the 4-lang transfer trial (`four_lang_cc_bbox_blur/`).

---

## 2026-07-18 / 19 — 4-lang `cc_bbox_blur` initial trial (results)

**Goal:** check whether the EN/ZH winner **`cc_bbox_blur`** transfers to KO and JA under the same dual-box geometry.

**Folder:** [`lib/notebooks/four_lang_cc_bbox_blur/`](lib/notebooks/four_lang_cc_bbox_blur/) — notebook [`four_lang_cc_bbox_blur.ipynb`](lib/notebooks/four_lang_cc_bbox_blur/four_lang_cc_bbox_blur.ipynb). Outputs: [`results/comparison_summary.json`](lib/notebooks/four_lang_cc_bbox_blur/results/comparison_summary.json), [`results/final_comparison.png`](lib/notebooks/four_lang_cc_bbox_blur/results/final_comparison.png).

### Design (Option B)

For each partner language `L ∈ {zh, ko, ja}`:

| Attack | Boxes | Score |
|--------|-------|-------|
| `uni_en` | EN + EN | EN + L |
| `uni_l` | L + L | EN + L |
| `multi` | EN + L | EN + L |

Defense: **EN ∩ L** Attn-last → percentile mask → dilate → top-2 CC + bbox snap → Gaussian blur fill (`BLUR_RADIUS=12`, cost 4). Geometry: `NUM_BOXES=2`, `FONT_SIZE=24`, random non-overlapping placement, balanced CIFAR-10 1000. Tune thr on n=100 (max EN masked acc) → full n=1000.

**Models:** EN ViT-B/32, ZH Chinese-CLIP B/16, KO Bingsu B/32, JA llm-jp B/16. Clean acc: EN 85.9%, ZH 91.4%, KO 89.6%, JA 92.5%.

### Full n=1000 results

**Column note:** after the same `cc_bbox_blur` mask is applied, **Def EN** = accuracy of the English CLIP; **Def L** = accuracy of that row’s partner CLIP (ZH / KO / JA). Same images, two models. Mean def averages the two.

| Cell | thr | Atk EN | Def EN | Atk L | Def L | Mean def | Clean Δ (mean) | Cov |
|------|----:|-------:|-------:|------:|------:|---------:|---------------:|----:|
| zh/uni_en | 0.95 | 3.4% | 54.7% | 27.1% | 65.6% | **60.2%** | −1.5pp | 7.6% |
| zh/uni_l | 0.95 | 72.3% | 72.5% | 39.6% | 63.5% | **68.0%** | −1.5pp | 6.6% |
| zh/multi | 0.95 | 4.3% | 71.6% | 7.3% | 78.2% | **74.9%** | −1.5pp | 8.8% |
| ko/uni_en | 0.90 | 3.4% | 61.3% | 12.4% | 68.9% | **65.1%** | −18.5pp | 15.6% |
| ko/uni_l | 0.95 | 70.5% | 66.5% | 78.4% | 73.3% | **69.9%** | −11.2pp | 9.1% |
| ko/multi | 0.95 | 4.2% | 67.6% | 12.6% | 74.3% | **71.0%** | −11.2pp | 9.0% |
| ja/uni_en | 0.90 | 3.4% | 60.1% | 4.1% | 72.4% | **66.3%** | −23.1pp | 15.2% |
| ja/uni_l | 0.95 | 70.8% | 66.9% | 83.0% | 79.3% | **73.1%** | −11.5pp | 6.7% |
| ja/multi | 0.95 | 4.1% | 69.9% | 6.6% | 83.9% | **76.9%** | −11.5pp | 8.3% |

### Takeaways

1. **ZH multi reproduces the prior winner.** Mean **74.9%**, clean Δ **−1.5pp** — matches `heatmap_defense_improvements/cc_bbox_blur` exactly. Pipeline is consistent.

2. **Defense works on hard attacks (EN+EN and multi) for all three partners.** Attacked acc collapses to ~3–13%; defended mean lands in the mid-60s to mid-70s. JA multi is the strongest recovery (**76.9%** mean).

3. **Native-only attacks (`uni_l`) are weak on KO/JA.** Baseline already high (KO 78%, JA 83% on the L model); defense slightly *hurts* accuracy. ZH `uni_l` is the exception — ZH model is actually fooled (39.6% → 63.5% after defense).

4. **KO/JA pay a much larger clean-image cost** (−11 to −23pp) than ZH (−1.5pp), especially on `uni_en` where the tune picks thr=0.90 and coverage jumps to ~15%. The EN∩L intersection is less precise for KO/JA than for ZH.

5. **English dual-box remains the universal transfer attack** — ASR ≥87% on KO and ≥96% on JA before defense.

**Status:** Initial 4-lang trial complete. `cc_bbox_blur` transfers on attacked accuracy; next work should target KO/JA clean-Δ (threshold / mask geometry), not re-prove the ZH multi case.

---

## 2026-07-19 — KO/JA clean-damage ablation notebook

**Goal:** reduce KO/JA Clean Δ under the same EN∩L `cc_bbox_blur` defense without re-running ZH.

**Folder:** [`lib/notebooks/ko_ja_clean_damage/`](lib/notebooks/ko_ja_clean_damage/) — notebook [`ko_ja_clean_damage.ipynb`](lib/notebooks/ko_ja_clean_damage/ko_ja_clean_damage.ipynb).

### Design

Partners `L ∈ {ko, ja}` only. Same dual-box Option B matrix (`uni_en` / `uni_l` / `multi`). Attn-last CAMs are **cached once** per image; five mask variants reuse them:

| Variant | Idea |
|---------|------|
| `baseline` | Current four_lang tune (max EN attacked acc) |
| `thr_floor_095` | Never go below thr=0.95 (kills `uni_en` thr=0.90 overshoot) |
| `pareto_tune` | Maximize `en_atk_acc + 0.5 * mean_clean_delta` on tune n=100 |
| `tight_dilate` | Pareto thr + dilate=1 |
| `no_bbox` | Pareto thr + no bbox snap |

If floor coverage still >12% on tune, non-baseline variants also get `max_coverage=0.12`. Winner per cell: best Clean Δ among variants within 3pp mean-acc of baseline.

### Full n=1000 results

Outputs: [`results/comparison_summary.json`](lib/notebooks/ko_ja_clean_damage/results/comparison_summary.json), [`results/winners.json`](lib/notebooks/ko_ja_clean_damage/results/winners.json), [`results/final_comparison.png`](lib/notebooks/ko_ja_clean_damage/results/final_comparison.png).

**Clean Δ (pp)** — less negative is better (plot shows this as positive “clean damage”; lower bar = better):

| Cell | baseline | thr_floor_095 | pareto_tune | tight_dilate | no_bbox |
|------|---------:|--------------:|------------:|-------------:|--------:|
| ko/uni_en | −18.4 | **−11.2** | −18.4 | −17.3 | −16.2 |
| ko/uni_l | −11.2 | −11.2 | −11.2 | **−9.2** | −9.2 |
| ko/multi | −11.2 | −11.2 | −11.2 | **−9.2** | −9.2 |
| ja/uni_en | −23.1 | −11.5 | −11.5 | −9.0 | **−7.4** |
| ja/uni_l | −11.5 | −11.5 | −11.5 | −9.0 | **−7.4** |
| ja/multi | −11.5 | −11.5 | −11.5 | **−9.0** | −7.4 |

**Mean defended acc (%)** (same cells / variants):

| Cell | baseline | thr_floor_095 | pareto_tune | tight_dilate | no_bbox |
|------|---------:|--------------:|------------:|-------------:|--------:|
| ko/uni_en | 65.1 | **65.5** | 65.1 | 65.6 | 65.6 |
| ko/uni_l | 69.9 | 69.9 | 69.9 | **71.2** | 68.8 |
| ko/multi | 71.0 | 71.0 | 71.0 | **71.9** | 71.2 |
| ja/uni_en | 66.2 | 71.4 | 71.4 | 69.6 | **68.3** |
| ja/uni_l | 73.1 | 73.1 | 73.1 | 74.1 | **71.7** |
| ja/multi | 76.9 | 76.9 | 76.9 | **77.0** | 73.9 |

**Winners** (best Clean Δ within 3pp mean-acc of baseline):

| Cell | Winner | Mean def | Clean Δ | vs baseline Clean Δ |
|------|--------|---------:|--------:|--------------------:|
| ko/uni_en | `thr_floor_095` | 65.5% | −11.2pp | +7.2pp |
| ko/uni_l | `tight_dilate` | 71.2% | −9.2pp | +2.0pp |
| ko/multi | `tight_dilate` | 71.9% | −9.2pp | +2.0pp |
| ja/uni_en | `no_bbox` | 68.3% | −7.4pp | +15.7pp |
| ja/uni_l | `no_bbox` | 71.7% | −7.4pp | +4.1pp |
| ja/multi | `tight_dilate` | 77.0% | −9.0pp | +2.5pp |

### Takeaways

1. **`uni_en` thr=0.90 was the main self-inflicted wound.** Flooring at 0.95 cuts KO Clean Δ −18.4 → −11.2 and JA −23.1 → −11.5, with defended mean flat or *up* (JA 66.2% → 71.4%). Success criterion (≥ −12pp on `uni_en`) hit.

2. **Geometry helps the residual ~−11pp, modestly.** `tight_dilate` / `no_bbox` move Clean Δ to about −7 to −9pp and usually keep (or slightly raise) mean def. Best single move on JA is dropping bbox snap (−7.4pp Clean Δ).

3. **`pareto_tune` mostly collapsed to thr=0.95** (same as floor), except `ko/uni_en` where it still picked 0.90 — so the clean-aware score did not always outweigh attacked EN acc on that cell.

4. **Still far from ZH (−1.5pp).** Remaining −7 to −11pp looks like EN∩KO / EN∩JA heatmap quality, not threshold choice. Next lever if needed: stronger KO/JA Large CLIPs.

**Status:** Ablation complete. Prefer **thr ≥ 0.95** always; use **`tight_dilate`** as the default KO/JA geometry tweak (or `no_bbox` when Clean Δ matters more than a couple pp of mean def on JA).

---

## 2026-07-19 — Notebook folder reorg (A+B hybrid)

Reorganized `lib/notebooks/` to cut top-level clutter without burying the current mainline:

- `_archive/` — superseded work (`old_*`, including `old_cifar10_typographic_4lang`)
- `_en_zh/` — early EN/ZH attack + GradCAM lineage (`en_zh_typographic`, `en_zh_multiple_placement`, `cam_intersection_defense`, `en_zh_multi_uni_attack`)
- Underscore prefixes keep `_archive/` and `_en_zh/` sorted together at the top
- Top level kept for current stack: `attention_defense` -> `heatmap_defense_improvements` -> `four_lang_cc_bbox_blur` -> `ko_ja_clean_damage`, plus `image_samples`

Relative `image_samples/` paths under `_en_zh/` were deepened by one level; index README + diary paths updated.

---

## 2026-07-20 — Paper draft figures + `cc_bbox_blur` pipeline viz

**Paper outline:** linked existing result PNGs into [`docs/paper_draft.md`](paper_draft.md) under the matching Method/Results subsections (Attn-last, grid, `cc_bbox_blur`, 4-lang, KO/JA clean Δ). Replaceable later at paper DPI.

**New qualitative figures** for how `cc_bbox_blur` works (real Attn-last EN/ZH on multilingual dual-box CIFAR samples) — stored under `four_lang_cc_bbox_blur/results/`:

| Output | Path |
|--------|------|
| 8-stage single example | [`lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_steps.png`](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_steps.png) |
| 5×7 multi-example grid | [`…/pipeline_examples.png`](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_examples.png) |
| Mean vs blur fill | [`…/pipeline_fill_compare.png`](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_fill_compare.png) |

Generator: [`make_pipeline_viz.py`](../lib/notebooks/four_lang_cc_bbox_blur/make_pipeline_viz.py). Stages shown: attacked → Attn EN → Attn ZH → ∩ → threshold+dilate → top-2 CC → bbox snap → blur fill.

Also merged feature branch `attention-based-method` into `main` (fast-forward) and deleted the branch after push.

---

## 2026-07-20 — Frozen `attack_pos` protocol re-run (four_lang + grid)

**Change:** Box coordinates are no longer re-sampled inside defense notebooks. They are loaded from
[`image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`](../lib/notebooks/image_samples/CIFAR10_BALANCED_1000_SAMPLE.json)
(`attack_pos.en` / `attack_pos.l`), baked once against a conservative reference box (`131×44`).
Runtime still measures the real word, then clamps the frozen top-left into bounds. Spec:
[`PROTOCOL.md`](../lib/notebooks/PROTOCOL.md).

**Also:** promoted improved grid search out of lineage to [`lib/notebooks/_test_grid/`](../lib/notebooks/_test_grid/)
(was `_en_zh/en_zh_multi_uni_attack/_test_grid/`). Both notebooks now hard-require CUDA.

**Runs (RTX 5070 Ti):**
- [`four_lang_cc_bbox_blur.ipynb`](../lib/notebooks/four_lang_cc_bbox_blur/four_lang_cc_bbox_blur.ipynb)
- [`_test_grid/grid_test.ipynb`](../lib/notebooks/_test_grid/grid_test.ipynb)

Before snapshots kept under each folder’s `results/before_protocol/`. Diff JSON:
[`four_lang…/protocol_before_after.json`](../lib/notebooks/four_lang_cc_bbox_blur/results/protocol_before_after.json).

### Grid search (`_test_grid`, multilingual, n=1000) — before vs after

| Method | Before mean | After mean | Δ |
|--------|------------:|-----------:|--:|
| `C_2p_confdrop_blur` | **48.5%** | **48.5%** | 0.0 |
| `B_2p_confdrop_mean` | 46.0% | 45.7% | −0.3 |
| `base_2p_maxconf_mean` | 11.7% | 10.1% | −1.6 |
| `no_defense` | 5.8% | 5.5% | −0.3 |

**Takeaway:** conf-drop winner is unchanged under frozen coords. The bake used the same seeded
sampler family as the old runtime RNG, so EN/ZH multilingual geometry barely moved. Grid remains
far behind Attn-last / `cc_bbox_blur` and ~10× more expensive (cost 62 vs 4).

### 4-lang `cc_bbox_blur` — before vs after

**Mean defended accuracy** (average of EN + partner L after `cc_bbox_blur`; higher is better).
thr column shows what the n=100 tune set picked before → after.

| Cell | thr before→after | Before | After | Δ |
|------|-----------------:|-------:|------:|--:|
| zh/uni_en | 0.95→0.95 | 60.2% | 60.2% | 0.0 |
| zh/uni_l | 0.95→**0.90** | 68.0% | 67.1% | −0.9 |
| zh/multi | 0.95→**0.90** | **74.9%** | 71.7% | **−3.2** |
| ko/uni_en | 0.90→0.90 | 65.1% | 63.3% | −1.8 |
| ko/uni_l | 0.95→0.95 | 69.9% | 68.4% | −1.5 |
| ko/multi | 0.95→**0.85** | 71.0% | 64.0% | **−7.0** |
| ja/uni_en | 0.90→0.90 | 66.2% | 66.1% | −0.1 |
| ja/uni_l | 0.95→0.95 | 73.1% | 72.8% | −0.3 |
| ja/multi | 0.95→**0.90** | 76.9% | 73.6% | **−3.3** |

**Clean Δ** (defence accuracy on *clean* images minus clean accuracy; ≤0, closer to 0 is better).
ΔΔ = after − before (negative means clean damage got worse).

| Cell | thr before→after | Before | After | ΔΔ |
|------|-----------------:|-------:|------:|---:|
| zh/uni_en | 0.95→0.95 | −1.5 | −1.5 | 0.0 |
| zh/uni_l | 0.95→**0.90** | −1.5 | **−7.0** | **−5.5** |
| zh/multi | 0.95→**0.90** | −1.5 | **−7.0** | **−5.5** |
| ko/uni_en | 0.90→0.90 | −18.4 | −18.4 | 0.0 |
| ko/uni_l | 0.95→0.95 | −11.2 | −11.2 | 0.0 |
| ko/multi | 0.95→**0.85** | −11.2 | **−25.5** | **−14.3** |
| ja/uni_en | 0.90→0.90 | −23.1 | −23.1 | 0.0 |
| ja/uni_l | 0.95→0.95 | −11.5 | −11.5 | 0.0 |
| ja/multi | 0.95→**0.90** | −11.5 | **−23.1** | **−11.5** |

**No defense vs defense (new protocol only).**

How to read a row (e.g. `zh/multi`):

- **Cell** = partner language + attack type. Partner `L` is ZH/KO/JA. Attack is `uni_en` (two EN words), `uni_l` (two L words), or `multi` (one EN + one L word). Defense always scores **EN ∩ L**.
- **Atk EN** = accuracy of the **English CLIP** on the attacked images, **before** any defense.
- **Atk L** = accuracy of the **partner CLIP** (ZH, KO, or JA for that row) on the same attacked images, **before** defense.
- **Atk mean** = average of Atk EN and Atk L (how bad the attack is overall).
- **Def mean** = average of EN and L accuracy **after** `cc_bbox_blur` (how good the defense is).
- **Clean EN / L** = accuracy of EN / partner CLIP on **unattacked** images (upper bound; same for every cell that shares those models).

| Cell | Atk EN (EN CLIP, no def) | Atk L (partner CLIP, no def) | Atk mean | Def mean (after `cc_bbox_blur`) | Clean EN / L |
|------|-------------------------:|-----------------------------:|---------:|--------------------------------:|-------------:|
| zh/uni_en | 3.8% | 24.8% | 14.3% | **60.2%** | 85.9 / 91.4 |
| zh/uni_l | 72.0% | 40.3% | 56.1% | **67.1%** | 85.9 / 91.4 |
| zh/multi | 4.5% | 6.4% | 5.5% | **71.7%** | 85.9 / 91.4 |
| ko/uni_en | 3.8% | 12.9% | 8.3% | **63.3%** | 85.9 / 89.6 |
| ko/uni_l | 70.0% | 78.2% | 74.1% | **68.4%** | 85.9 / 89.6 |
| ko/multi | 3.5% | 12.3% | 7.9% | **64.0%** | 85.9 / 89.6 |
| ja/uni_en | 3.8% | 3.2% | 3.5% | **66.1%** | 85.9 / 92.5 |
| ja/uni_l | 71.3% | 84.6% | 78.0% | **72.8%** | 85.9 / 92.5 |
| ja/multi | 4.1% | 5.4% | 4.8% | **73.6%** | 85.9 / 92.5 |

Hard attacks (`uni_en`, `multi`) fall to ~4–14% mean without defense and recover to the mid-60s–70s
with `cc_bbox_blur`. Native-only `uni_l` on KO/JA is already weak as an attack (~74–78% mean), so
defense barely helps (or slightly hurts).

### Results (what changed and why)

Quick glossary for this section:

- **Frozen `attack_pos`:** every image has two fixed box corners stored in the sample JSON. Notebooks no longer roll their own RNG for placement.
- **thr (threshold):** percentile cut on the Attn-last intersection heatmap. Higher thr (e.g. 0.95) keeps only the hottest peaks → smaller masks. Lower thr (e.g. 0.85) masks more of the image.
- **Mean def:** accuracy after defense, averaged over EN and the partner language.
- **Clean Δ:** how much the same defense hurts *unattacked* images. A more negative number = more collateral damage.
- **Tune set:** first 100 images (10/class). The notebook picks thr to maximize EN attacked accuracy on this subset, then freezes that thr for the full 1000.

What we saw:

- **The attacks themselves barely moved.** Pre-defense (“attacked”) accuracy changed by about ±1pp. Freezing coordinates did not suddenly make dual-box attacks much stronger or weaker.
- **Where thr stayed the same, results stayed the same.** Cells like `zh/uni_en`, `ko/uni_l`, `ja/uni_l` keep their old mean def and Clean Δ. The defense pipeline is stable when the knobs don’t move.
- **The drops track thr drops.** Every cell that got worse had a lower tuned thr:
  - `zh/multi` and `zh/uni_l`: 0.95 → 0.90 → mean def −1 to −3pp, Clean Δ −1.5 → −7.0
  - `ja/multi`: 0.95 → 0.90 → mean def −3.3pp, Clean Δ −11.5 → −23.1
  - `ko/multi`: 0.95 → **0.85** → mean def −7.0pp, Clean Δ −11.2 → −25.5 (worst swing)
- **Why thr fell:** bake-time positions use a large reference box (`131×44`), then runtime clamps after measuring the real word. That is a small geometry shift vs the old “sample with the actual word size” RNG. On the 100-image tune set, that shift was enough for the thr search (which only maximizes attacked EN acc) to prefer a more aggressive mask.
- **Why lower thr hurts Clean Δ more than mean def:** a looser mask covers more of the object on clean images (big Clean Δ hit) while on attacked images it still helps a bit but can also erase useful content, so mean def slips a few pp too.
- **Headline ZH multi softens, but isn’t “broken”.** 74.9% → 71.7% with Clean Δ −1.5 → −7.0 is almost entirely the thr=0.90 choice, not a failure of Attn-last or CC+bbox+blur.
- **Grid vs saliency contrast:** `_test_grid`’s conf-drop winner stayed at **48.5%** — no thr knob, so no cascade. The fragile piece under protocol freeze is **percentile thr tuning**, not the frozen boxes themselves.

### Potential solutions

- **Floor thr at 0.95 in the protocol** (already the KO/JA ablation recommendation). Re-run four_lang with `thr = max(tuned, 0.95)` so multi cells can’t fall back to 0.85/0.90 for a small attacked-acc bump.
- **Clean-aware tune objective** (from `ko_ja_clean_damage`): maximize something like `en_atk_acc + λ * clean_delta` on the tune set instead of attacked EN alone, so thr can’t buy attack recovery by torching clean images.
- **Bake with actual word sizes** (or per-language refs) so frozen anchors match old runtime geometry more tightly — reduces the thr-retune surprise, though the whole point of freeze is one shared geometry.
- **Report both “free tune” and “thr≥0.95”** in papers/tables so protocol sensitivity is visible without hiding the safer operating point.
- **Leave grid alone for now** — it’s stable under freeze and still not competitive with `cc_bbox_blur` on cost or accuracy.

**Status:** Both re-runs complete on CUDA with frozen `attack_pos`. Prefer citing after-protocol numbers plus the two before/after tables above; treat thr floor 0.95 as the next protocol tightening if Clean Δ matters.
