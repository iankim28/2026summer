# Agent Handoff Document

**Project:** `d:\ian\2026summer`  
**Notebook under review:** `lib/notebooks/updated_multilingual_consensus_colab.ipynb`  
**Handoff from:** previous chat session  
**Next task:** Improve or extend the notebook based on the experimental findings below

---

## 1. What this project is about

The notebook implements and critically tests a proposed adversarial defence for
vision-language AI models (specifically multilingual CLIP). The core idea being tested:
if you label images using five languages simultaneously, an attack crafted to fool
English labels should leave other languages unaffected — causing "disagreement" that
exposes the attack.

See `research_goal.md` for a plain-English summary aimed at a general audience.

---

## 2. Repository structure

```
2026summer/
├── docs/
│   ├── research_goal.md         ← plain-English project summary
│   ├── research_diary.md        ← running lab log
│   ├── handoff.md               ← this file
│   └── ...
├── lib/
│   ├── notebooks/
│   │   ├── updated_multilingual_consensus_colab.ipynb   ← MAIN notebook (fully run)
│   │   ├── multilingual_adversarial_defence_tutorial.ipynb  ← longer tutorial version
│   │   └── intro_to_CLIP.ipynb                          ← background reading
│   └── *.py                     ← notebook build/review scripts
├── claude_experiments/          ← UNTOUCHED (not created by the user)
├── requirements.txt
└── .venv/               ← Python venv (Python 3.14, CUDA 13, RTX 5070 Ti)
```

---

## 3. Environment

- **GPU:** NVIDIA GeForce RTX 5070 Ti
- **Python:** 3.14 (venv at `d:\ian\2026summer\.venv\Scripts\python.exe`)
- **Key packages:** `open_clip_torch`, `transformers 5.12.1`, `datasets`, `torch`
- **Dataset:** CIFAR-10 loaded via HuggingFace (`uoft-cs/cifar10`) — already cached,
  loads in ~1 second. Do NOT change to `torchvision.datasets.CIFAR10`; the Toronto
  mirror is ~40 KB/s (~70 min download).
- **Model:** `xlm-roberta-base-ViT-B-32` pretrained on `laion5b_s13b_b90k` — already
  cached in `~/.cache/huggingface/hub/`

### Critical Windows/Jupyter fix already applied

`DataLoader` on Windows with `num_workers > 0` causes pickle errors in Jupyter. The
notebook already has `NUM_WORKERS = 0 if sys.platform == "win32" else 2` and the
dataset wrapper `HFCifar10` defined as a **top-level class** (not nested inside a
function). Do not revert these.

Also: `import transformers` must come **before** `import open_clip` in the imports
cell, or `open_clip` will cache `transformers = None` at import time and crash when
loading the XLM-R model.

---

## 4. Experimental results (full analysis)

### 4a. Clean zero-shot accuracy (n=500, no attacks)

| Language | Accuracy |
|---|---|
| English (en) | 94.0% |
| Korean (ko) | 83.6% |
| Spanish (es) | 94.6% |
| French (fr) | 94.8% |
| Japanese (ja) | 93.8% |
| All-5-agree | **85.6%** |

Korean is weakest — LAION-5B training data skews toward Western/English content, so
Korean text embeddings are less well calibrated. The 85.6% baseline agreement is
important: any disagreement-based detector must beat this noise floor.

### 4b. H1 — Transfer experiment (English-only PGD attack, n=300)

```
eps (/255)   en     ko     es     fr     ja   ensemble   all-5-agree
clean        ~94%   ~84%   ~95%   ~95%   ~94%   ~93%       85.6%
0.5           3.0    4.7    5.7    4.3    3.7    4.0%       89.0%
1             0.3    0.7    0.7    0.7    0.3    0.3%       92.0%
2             0.0    0.0    0.0    0.0    0.0    0.0%       93.3%
4             0.0    0.0    0.0    0.0    0.0    0.0%       94.7%
8             0.0    0.0    0.0    0.0    0.0    0.0%       94.7%
```

**Transfer fractions** (how much of the English drop carries to other languages):
- Korean: 0.85–0.87 (most resistant, still very high)
- Spanish: 0.98–1.01
- French: 1.00–1.01
- Japanese: 1.00–1.01

**Key finding:** At ε=2/255 (a routine attack strength), all five languages simultaneously
collapse to 0%. The "agreement" metric *rises* under attack (85.6% → 94.7%) — the
opposite of what the defence needs.

### 4c. Mechanism (Section 6)

```
Mean same-class cross-lingual cosine similarity  = 0.914
Mean different-class within-English cosine       = 0.792
```

The text embeddings for "cat" in EN, KO, ES, FR, JA are more similar to each other
(0.914) than "cat" vs "dog" within a single language (0.792). So the PGD gradient that
pushes the image away from the English "cat" cluster also pushes it away from every
other language's "cat" cluster. Language-specific attack is geometrically impossible
with a shared encoder.

### 4d. Disagreement detector (Section 7)

| ε (/255) | all-agree % | AUC(n_unique) | AUC(vote_entropy) | AUC(mean_JS) |
|---|---|---|---|---|
| 0.5 | 89.0% | 0.480 | 0.481 | **0.425** |
| 1 | 92.0% | 0.465 | 0.465 | **0.319** |
| 2 | 93.3% | 0.458 | 0.458 | **0.252** |
| 4 | 94.7% | 0.452 | 0.451 | **0.205** |
| 8 | 94.7% | 0.452 | 0.451 | **0.199** |

All AUCs below 0.5 — **worse than a coin flip** — and declining as the attack gets
stronger. Attacked images are *more* consensual than clean ones, so the detector fires
in the wrong direction.

### 4e. H2 — Attacker cost (Section 8)

| ε (/255) | attack English only | attack all 5 languages |
|---|---|---|
| 0.5 | 4.7% | 5.3% |
| 1 | 0.3% | 0.0% |
| 2 | 0.0% | 0.0% |

Attacking one language is essentially as effective as attacking all five. No attacker
cost growth with language count. H2 refuted.

### 4f. Consensus-purification denoiser (Section 9)

**Training:** DnCNN-style residual CNN, 2 epochs, self-supervised, ~46s total  
**Architecture:** 8-layer conv net with BatchNorm, residual connection, output clamped to [0,1]

| Condition | Accuracy |
|---|---|
| Clean, no denoiser | 92.0% |
| Clean, with denoiser | **63.0%** ← hurts clean images |

| ε (/255) | no defence | non-adaptive denoiser | adaptive denoiser |
|---|---|---|---|
| 2 | 0.0% | 56.5% | **0.0%** |
| 8 | 0.0% | 48.5% | **0.0%** |

**Three problems with the denoiser:**
1. It **hurts clean accuracy by 29 points** (92% → 63%). Root cause: CIFAR-10 images
   are 32×32 upscaled to 224×224 — the denoiser learns to smooth out upscaling
   artifacts, damaging genuine detail. This is a dataset/resolution artefact, not a
   fundamental flaw.
2. The non-adaptive "recovery" (0% → 56.5%) is misleading. The ceiling is 63%, not 92%.
   It's just returning adversarial accuracy to the denoiser's own degraded baseline.
3. The adaptive attack (which backpropagates through the denoiser end-to-end) reduces
   accuracy to **exactly 0%** — identical to having no defence at all.

---

## 5. Config (current notebook settings)

```python
DATASET      = "cifar10"   # fast; "stl10" slower but better for denoiser
N_CLEAN      = 500
N_TRANSFER   = 300
EPS_LIST     = [0.5, 1, 2, 4, 8]  # L-inf /255
PGD_STEPS    = 20
BATCH        = 100
DEN_TRAIN_N  = 1000
DEN_EPOCHS   = 2
DEN_TEST_N   = 200
DEN_TRAIN_STEPS = 7
DEN_EVAL_STEPS  = 15
DEN_EVAL_EPS    = [2, 8]
DEN_LAM_FID     = 5.0      # fidelity weight in denoiser loss
DEN_BATCH       = 24
RUN_DENOISER    = True
RUN_ABLATION    = False    # en-only vs all-5 denoiser comparison
```

---

## 6. What to work on next

The main findings are solid and reproducible, but there are several concrete directions
for improvement:

### Priority 1 — Fix the denoiser's clean-accuracy problem
The 29-point drop on clean images (92% → 63%) is the most actionable issue. Options:
- **Switch to STL-10** (`DATASET = "stl10"`): native 96×96 images upscale to 224×224
  much more cleanly; the notebook notes non-adaptive recovery reaches ~82% on STL-10
  vs 56.5% on CIFAR. This is the simplest change.
- **Tune `DEN_LAM_FID`**: the fidelity weight (currently 5.0) penalises the denoiser
  for straying too far from the input. Raising it should preserve clean accuracy better.
- **Add a direct clean accuracy term to the training loss**: train the denoiser to also
  be a near-identity on clean images, not just to restore consensus on adversarial ones.

### Priority 2 — Run the optional ablation (`RUN_ABLATION = True`)
The ablation trains an English-only denoiser and compares it to the all-5 version. If
they match, multilingual consensus contributes nothing (the notebook predicts they
match). This is ~4 minutes of extra runtime and would close the argument completely.

### Priority 3 — Stronger adaptive attack for the denoiser
The current adaptive attack uses PGD with `DEN_EVAL_STEPS = 15` steps, no random
restarts. For a publishable evaluation, add:
- Multiple random restarts (e.g. 5×)
- More steps (e.g. 40–50)
- AutoAttack or similar for a standardised benchmark

### Priority 4 — Improve plots
The current plots are functional but plain. Suggested improvements:
- Add clean accuracy as a horizontal reference line on the robust accuracy plot
- Label the denoiser ceiling (63%) on the denoiser plot so the misleading appearance
  of non-adaptive "recovery" is immediately obvious
- Add a plot specifically for the mechanism section showing the cosine similarity matrix
  as a heatmap

---

## 7. What the two notebooks are (for context)

| Notebook | Purpose |
|---|---|
| `updated_multilingual_consensus_colab.ipynb` | Critical replication — runs experiments, shows the defence fails, includes the mechanism proof. **This is the main one.** |
| `multilingual_adversarial_defence_tutorial.ipynb` | Longer pedagogical tutorial — walks through the same pipeline step by step, assumes the defence works, designed for learning. Use `intro_to_CLIP.ipynb` as a prerequisite. |
