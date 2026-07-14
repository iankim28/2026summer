# Attention-based defense against typographic attacks on multilingual CLIP

A **cheap (~3 forward-pass) defense**: use the CLIP image encoders' own **attention** to
localize the adversarial text, mask it, and re-classify — instead of the expensive
occlusion grid search. This folder contains the method, its evaluation, and the honest
comparison to occlusion.

Models: **English** = OpenAI CLIP ViT-B/32, **Chinese** = `OFA-Sys/chinese-clip-vit-base-patch16`
(both via `transformers`). Dataset: **STL-10**. Attack: typographic (a wrong-class word
written on the image), 1 box or 2 boxes, random placement.

---

## 1. The method: attention-overlap localization

Attention comes **for free** during a normal forward pass. The defense:

1. Get the **attention-rollout** saliency from **both** models (EN + ZH) — 1 forward each.
2. **Combine** the two maps and threshold to get a text region.
3. **Mask** that region and **re-classify** — 1 forward.

Total ≈ **3 forward passes** per image.

**The combination rule is critical:**
| combine | what it does | result |
|---|---|---|
| **min (overlap)** | mask only where *both* models agree | **works** (avoids masking the object) |
| sum (union) | mask where *either* is high | **hurts** (masks the object) |

Single-model attention alone is *harmful* — one model's attention sits on the object, so
masking it destroys accuracy. **Requiring two models to agree (min-overlap) is what makes
the method work.**

---

## 2. Results (robust accuracy = correct-class rate under attack)

**Single-box English typographic attack** (EN classifier):
| defense | fwd passes | robust acc |
|---|---|---|
| no defense | 1 | 27% |
| attention single-model | 2 | 8% (harmful) |
| **attention 2-model overlap** | **3** | **57%** |
| occlusion 4×4 (for reference) | 18 | 84% |

**2-box English typographic attack** (EN classifier, best = min-overlap):
| placement | no defense | **attention overlap** | occlusion |
|---|---|---|---|
| fixed | 26% | **44%** | 38% |
| random | 27% | **37%** | 21% |

- Best 2-box attention result ≈ **44% (fixed) / 37% (random)** at ~3 forwards.
- On the *random* 2-box attack it even **beats occlusion** (37% vs 21%).
- With the **sum** combination it collapses to 11–18% (below no-defense) — use **min**.

---

## 3. The cascade: cheap attention → expensive occlusion

`cascade.py` runs the 3-forward attention-overlap on every image, then **escalates only the
least-confident fraction** to occlusion. Sweeping the escalation fraction traces a tunable
Pareto curve from **57% @ 3 forwards** to **~85% @ ~17 forwards** (single-box); it bulges
above linear because the escalated images are exactly the ones attention got wrong.

---

## 4. Honest assessment

- **Attention is cheap but weak.** It clears no-defense but stays well below occlusion on
  the harder cases. Raw attention localizes the text only weakly (in-box focus ~1.2 vs
  occlusion's ~4.7); `attn_feasibility.py` / `attn_gradcam.py` show why (attention sits on
  the object; input-gradient is noisy).
- **Its value is the ~3-forward cost** and, on the random 2-box attack, actually beating
  occlusion.
- **All results are NON-ADAPTIVE** — operating cost/accuracy, not a robustness guarantee
  against an attacker who knows the defense.

---

## 5. Files

| file | what it does |
|---|---|
| `attn_cheap_defense.py` | the core attention 2-model overlap defense (single-box), vs occlusion |
| `cascade.py` | attention → occlusion cascade (tunable Pareto curve) |
| `multi_typographic.py` | single vs double (2-box) attack; attention-overlap result |
| `multi_lingual_defense.py` | multilingual vs unilingual 2-box; attention 2-map / 4-map (sum-combine) |
| `occ_complexity.py` | attention vs occlusion grid-search patch selection (1-patch / 2-patch) |
| `attn_feasibility.py` | attention-rollout heatmaps (shows raw attention doesn't localize) |
| `attn_gradcam.py` | input-gradient heatmaps (also poor) |
| `attn_defense.py` | occlusion + oracle reference (context for the comparison) |
| `results/*.png` | all figures |

## 6. Run

```bash
pip install torch torchvision transformers open_clip_torch ftfy sentencepiece matplotlib numpy pillow
# fonts (Linux): apt-get install fonts-dejavu fonts-noto-cjk
# STL-10 auto-downloads to ./data on first run; model weights from Hugging Face.

python attn_cheap_defense.py     # attention overlap vs occlusion (single box) -> results/cost_accuracy_v2.png
python cascade.py                # attention->occlusion cascade -> results/cascade.png
python multi_typographic.py --n 100   # single vs double attack (attention-overlap col) -> results/multi_typo_vis.png
python occ_complexity.py --n 80  # attention vs grid-search patch selection -> results/occ_complexity.png
```

Each script prints its numbers and writes a figure to `results/`. Note: `attn_cheap_defense.py`
reads `results/cost_accuracy.json` (run `cost_accuracy.py` first, included here).
