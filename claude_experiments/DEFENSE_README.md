# Typographic-Attack Defense: occlusion baseline + attention-overlap / cascade method

Defends a CLIP zero-shot classifier against **typographic attacks** (a misleading word
written onto the image) by **locating the attack region and re-classifying with it
removed**. This folder contains a strong **sweeping baseline** (occlusion) and a cheap
**attention-overlap method** plus a **cascade** that trades compute for accuracy.

All numbers are on **STL-10** under an **English** typographic attack (a random wrong-class
word written at the bottom of the image). The classifier is **OpenAI CLIP ViT-B/32**; a
second model (**Chinese-CLIP ViT-B/16**) is used only to provide a second attention map.

---

## 1. The shared idea: locate → mask → re-classify

A typographic attack works because the image encoder *reads* the written word and reports
it. So the defense is:

1. **Predict** on the (possibly attacked) image.
2. **Localize** the region driving that prediction.
3. **Mask** that region (blank it) and **re-classify** — with the text gone, the model
   sees the real object.

The methods differ only in **how they localize** (step 2), which sets the compute cost.

---

## 2. Sweeping baseline — OCCLUSION  (`cost_accuracy.py`, `attn_defense.py`)

**How it works.** Split the image into a `G×G` grid. For **each** cell, blank it (set the
pixels to gray/zero) and re-run the model; measure how much the predicted class's score
**drops**. The cell whose removal drops the score most is the region the model relied on —
the text. Threshold the drop-map (cells `≥ 0.5·max`), mask those cells, re-classify.

```
for each of G*G cells:
    occluded = image with that cell blanked
    drop[cell] = score(image, pred_class) - score(occluded, pred_class)   # high = important
region = cells where drop >= 0.5 * max(drop)
answer = classify(image with `region` masked)
```

- **Cost** = `G² + 2` image-encoder forward passes per image (the `G²` occlusions are run
  as **one batched** forward; +1 for the initial prediction, +1 for the re-classify).
- `cost_accuracy.py` **sweeps** `G ∈ {2,4,7,14}` and plots cost vs robust accuracy.
- `attn_defense.py` runs the full occlusion defense at `G=7` **plus**: an **oracle** (mask
  the known text box = the accuracy ceiling), the **clean-image cost** of blindly masking,
  a **2-model occlusion overlap**, and **localization quality** (in-box focus ratio, IoU).

**Result:** coarse grids are the sweet spot — `4×4` recovers **27% → 84%** at ~18 forwards;
finer grids waste compute and even hurt (`14×14` → 64%). Oracle ceiling ≈ **96%**.

## 3. Our method — ATTENTION-OVERLAP  (`attn_cheap_defense.py`)

**How it works.** Attention is produced *for free* during a normal forward pass. Get the
attention-**rollout** saliency from **both** models (EN and ZH), take their **overlap**
(element-wise min), threshold, mask, re-classify.

```
sal_EN = rollout(EN.attentions)          # 1 forward
sal_ZH = rollout(ZH.attentions)          # 1 forward
overlap = min(sal_EN, sal_ZH)            # both models agree -> the text region
answer  = classify(image with high-overlap region masked)   # 1 forward
```

- **Cost** = **~3 forward passes** per image.
- **Why the overlap (not a single model):** one model's attention sits on the *object*, so
  masking it *destroys* accuracy (single-model attention defense = **8%**, worse than no
  defense). Requiring **both** models to agree avoids masking the object → recovers
  **27% → 57%** at 3 forwards.

> Note: raw attention localizes the text only weakly (in-box focus ~1.2 vs occlusion's 4.7),
> which is why the cheap method trails occlusion in accuracy. `attn_feasibility.py` and
> `attn_gradcam.py` show the raw-attention and input-gradient maps that motivated this.

## 4. Our method — CASCADE (cheap → expensive)  (`cascade.py`)

Run the cheap 3-forward attention-overlap on **every** image, then **escalate** only the
least-confident fraction (lowest softmax margin) to the occlusion defense.

```
for each image:
    cheap_pred, margin = attention_overlap_defense(image)   # 3 forwards
escalate the (100-conf)% lowest-margin images -> occlusion 4x4 (+18 forwards each)
```

- **Avg cost** = `3 + 18·f` where `f` = escalation fraction (each image is *either* 3 *or*
  21 forwards; the graph shows the dataset average).
- **Result:** a tunable Pareto curve from **57% @ 3** to **~85% @ ~17** forwards. Because the
  escalated images are exactly the ones the cheap pass failed, the curve bulges **above**
  linear and at ~80% escalation slightly **beats** pure occlusion-4×4 (84.7% vs 84%).

---

## 5. Results at a glance  (`results/`)

| defense | forward passes / img | robust acc (English attack) |
|---|---|---|
| no defense | 1 | 27% |
| attn single-model | 2 | 8% (masks the object) |
| **attn 2-model overlap (ours)** | **3** | **57%** |
| occlusion 2×2 (baseline) | 6 | 76% |
| occlusion 4×4 (baseline) | 18 | 84% |
| **cascade (ours), 50% escalate** | **12** | **80%** |
| **cascade (ours), 80% escalate** | **17.4** | **84.7%** |
| oracle (mask known box) | 2 | 96% (ceiling) |

Figures: `results/cost_accuracy.png` (occlusion sweep), `results/cost_accuracy_v2.png`
(cheap attention vs occlusion), `results/cascade.png` (the cascade Pareto curve).

⚠️ **All results are NON-ADAPTIVE** — they measure operating cost/accuracy, not robustness
against an attacker who knows the defense. An adaptive evaluation is required before any
robustness claim.

---

## 6. Setup & how to run

```bash
# Python 3.10+; a GPU is recommended (CPU works, slower).
pip install torch torchvision transformers open_clip_torch ftfy sentencepiece \
            matplotlib numpy pillow
# Fonts for rendering the attack text (Linux):
#   DejaVuSans (Latin) + Noto Sans CJK (Korean/Japanese/Chinese)
#   e.g. apt-get install fonts-dejavu fonts-noto-cjk
```

STL-10 downloads automatically to `./data` on first run. Model weights download from
Hugging Face (OpenAI CLIP, Chinese-CLIP) on first use.

```bash
# 1. Sweeping baseline: occlusion grid sweep -> results/cost_accuracy.png + .json
python cost_accuracy.py

# 2. Full occlusion defense + oracle ceiling + clean-image cost + localization quality
python attn_defense.py --n 100

# 3. Our cheap method: attention-overlap (2-3 forwards) -> results/cost_accuracy_v2.png
python attn_cheap_defense.py      # reads results/cost_accuracy.json to overlay occlusion points

# 4. Our cascade: cheap -> escalate-to-occlusion Pareto curve -> results/cascade.png
python cascade.py

# (optional) inspect why raw attention / input-gradient fail to localize:
python attn_feasibility.py        # attention-rollout heatmaps -> results/attn_feasibility.png
python attn_gradcam.py            # input-gradient heatmaps    -> results/attn_gradcam.png
```

Each script prints its numbers and writes a figure to `results/`. Sample sizes (`--n`, or
the `N=` constant near the top of each script) are small (100–150) for speed; increase for
tighter estimates.

---

## 7. File index (defense)

| file | role |
|---|---|
| `cost_accuracy.py` | **baseline**: occlusion grid sweep (cost vs accuracy) |
| `attn_defense.py` | occlusion defense + oracle ceiling + clean-cost + localization quality |
| `attn_cheap_defense.py` | **our method**: attention-overlap defense (~3 forwards) |
| `cascade.py` | **our method**: cheap→occlusion cascade (tunable Pareto curve) |
| `attn_feasibility.py` | attention-rollout maps (shows raw attention doesn't localize) |
| `attn_gradcam.py` | input-gradient maps (also poor) — motivates occlusion |
| `results/*.png` | the three cost-vs-accuracy figures |

The rest of the folder (`typographic_attack.py`, `mechanism_experiment.py`,
`perlang_models.py`, `shared_vs_separate.py`, the `*.md` writeups, the Colab notebook) is
the surrounding attack/mechanism study; see `README.md` and the `*_PROJECT.md` docs.
