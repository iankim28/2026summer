# Experiment Protocol — Shared Notebook Stack

> **Purpose:** Reproducible shared conventions for every dual-box typographic-attack
> notebook under `lib/notebooks/`. Geometry, sample indices, targets, and defense
> defaults live here so new experiments do not silently diverge.

Folder index and “last used” dates: [`README.md`](README.md). Narrative: `docs/research_diary.md`.

---

## 1. What is current (2026-07-20)

| Stage | Folder | Role |
|---|---|---|
| Baseline saliency | [`attention_defense/`](attention_defense/) | Attn-last beats GradCAM (72.6% mean, cost 4) |
| Defense winner | [`heatmap_defense_improvements/cc_bbox_blur/`](heatmap_defense_improvements/cc_bbox_blur/) | Attn-last → CC+bbox+blur (**74.9%** mean, clean Δ −1.5pp) |
| 4-lang transfer | [`four_lang_cc_bbox_blur/`](four_lang_cc_bbox_blur/) | EN∩L `cc_bbox_blur` for L∈{zh,ko,ja}; Option B attack matrix |
| KO/JA clean Δ | [`ko_ja_clean_damage/`](ko_ja_clean_damage/) | Threshold / dilate / bbox ablations on KO/JA only |
| Shared sample | [`image_samples/`](image_samples/) | Fixed CIFAR-10 indices + **frozen attack coordinates** |

Early EN/ZH GradCAM + grid work lives under [`_en_zh/`](_en_zh/) (archived lineage, not the active defense).

---

## 2. Environment

```
open_clip_torch, transformers, datasets, matplotlib, Pillow, torch, numpy
```

```bash
pip install -q open_clip_torch transformers datasets matplotlib Pillow
```

- `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`
- All models `.eval()`

---

## 3. Models

| Lang | Library | ID / arch |
|---|---|---|
| EN | `open_clip` | `ViT-B-32` / `openai` |
| ZH | `transformers` | `OFA-Sys/chinese-clip-vit-base-patch16` |
| KO | `transformers` | `Bingsu/clip-vit-base-patch32-ko` |
| JA | `transformers` | `llm-jp/llm-jp-clip-vit-base-patch16` |

**Prompts:** EN `"a photo of a {class}."`; ZH `"一张{class}的照片。"`; KO/JA follow each notebook’s native template.

**Classification:** L2-normalised image/text cosine similarity; `argmax` over 10 class texts; batch size 128.

| Index | EN | ZH | KO | JA |
|---:|---|---|---|---|
| 0 | airplane | 飞机 | 비행기 | 飛行機 |
| 1 | automobile | 汽车 | 자동차 | 自動車 |
| 2 | bird | 鸟 | 새 | 鳥 |
| 3 | cat | 猫 | 고양이 | 猫 |
| 4 | deer | 鹿 | 사슴 | 鹿 |
| 5 | dog | 狗 | 개 | 犬 |
| 6 | frog | 青蛙 | 개구리 | カエル |
| 7 | horse | 马 | 말 | 馬 |
| 8 | ship | 船 | 배 | 船 |
| 9 | truck | 卡车 | 트럭 | トラック |

---

## 4. Dataset

| Field | Value |
|---|---|
| Dataset | CIFAR-10 test (`uoft-cs/cifar10`) |
| Sample file | [`image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`](image_samples/CIFAR10_BALANCED_1000_SAMPLE.json) |
| Sample | 1000 images, 100 per class |
| Contents | `idx`, `true`, `attack_pos`, plus metadata (`n_images`, `seed`, …) |
| Resize | All images → **224×224** bicubic before attack / classify |
| Targets | `random.Random(0)`; target ≠ true; one target per image for all experiments |
| Tune subset | 100 images = first 10 indices of each class |

```python
rng = random.Random(0)
target = np.array([
    rng.choice([c for c in range(10) if c != int(true[k])])
    for k in range(len(idx))
])
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
```

---

## 5. Frozen attack coordinates (`attack_pos`)

Box positions are **not** re-sampled inside defense notebooks. They are baked once into
the sample JSON and loaded at runtime.

### 5.1 Where and how to regenerate

| Item | Path / command |
|---|---|
| Storage | `CIFAR10_BALANCED_1000_SAMPLE.json` → key `attack_pos` |
| Helper | [`image_samples/attack_placement.py`](image_samples/attack_placement.py) |
| Bake | `python lib/notebooks/image_samples/attack_placement.py` |

Re-bake **only** if the sample set itself changes. Do not re-sample per defense.

### 5.2 Baked metadata (current file)

| Field | Value |
|---|---|
| `display_size` | `224` |
| `font_size` | `24` |
| `pad` | `8` |
| `bh_extra` | `12` |
| `num_boxes` | `2` |
| `ref_bw` × `ref_bh` | **131 × 44** (max box over EN/ZH/KO/JA class words) |
| `en` | 1000 top-left `[x, y]` for slot 0 |
| `l` | 1000 top-left `[x, y]` for slot 1 |
| Fonts (Windows bake) | lat `arial.ttf`, cjk `msyh.ttc`, ko `malgun.ttf` |

First three anchors (sanity check):

| `img_idx` | `en` (slot 0) | `l` (slot 1) |
|---:|---|---|
| 0 | `[49, 107]` | `[26, 24]` |
| 1 | `[7, 23]` | `[30, 151]` |
| 2 | `[30, 77]` | `[45, 176]` |

### 5.3 Historical bake (one-time)

Anchors were generated with the old seeded non-overlapping sampler and a **fixed reference
box size** (`ref_bw` × `ref_bh`), then written into the JSON:

```python
rng = random.Random(int(img_idx) * NUM_BOXES + box_i)  # bake only
```

Runtime code must **not** call this RNG for placement.

### 5.4 Runtime drawing

```python
attack_pos = _saved['attack_pos']
xy0 = attack_pos['en'][img_idx]   # slot 0
xy1 = attack_pos['l'][img_idx]    # slot 1
# measure bw, bh from the rendered word + PAD / bh_extra
# clamp xy into [0, 224-bw] × [0, 224-bh], then draw white box + black text
```

Shared helper (optional): `attack_placement.draw_dual_box_at(...)`.

Active notebooks that already load `attack_pos`:
`attention_defense/`, `heatmap_defense_improvements/`, `four_lang_cc_bbox_blur/`,
`ko_ja_clean_damage/`, and `_en_zh/en_zh_multi_uni_attack/`.

---

## 6. Attack design

### 6.1 Shared box parameters

| Parameter | Value |
|---|---|
| `DISPLAY_SIZE` | `224` |
| `NUM_BOXES` | `2` |
| `FONT_SIZE` | `24` |
| `PAD` | `8` |
| Box height extra | `+12` px |
| Text / box fill | black on white |

**Fonts:** EN → Latin (`arial` / DejaVu); ZH/JA → CJK (`msyh` / Noto CJK); KO → Malgun (fallback CJK).

### 6.2 Option B matrix (current 4-lang / KO-JA work)

For partner language `L ∈ {zh, ko, ja}`:

| Attack | Slot 0 (`en`) | Slot 1 (`l`) | Score models |
|---|---|---|---|
| `uni_en` | EN word | EN word | EN + L |
| `uni_l` | L word | L word | EN + L |
| `multi` | EN word | L word | EN + L |

Words are translations of the **same target class** for that image.

EN/ZH-only studies historically used the same dual-box geometry with `L=zh`
(multilingual = `multi`, unilingual = `uni_en`).

---

## 7. Defence designs (current)

### 7.1 Attn-last saliency

- CLS→patch attention from the **last** ViT block; heads averaged
- Resize map to 224×224
- Cost: **4** forward passes / image for two-model intersection (vs GradCAM **6**)

### 7.2 `cc_bbox_blur` (production defense)

Per image, for scoring pair EN ∩ L:

1. Attn-last map per model → intersection (`elementwise min`)
2. Percentile threshold (tuned on n=100; prefer **thr ≥ 0.95**)
3. Dilate (default 3×3, 3 iterations; KO/JA ablations also try 1)
4. Keep top-2 connected components; snap each to axis-aligned bbox
5. Fill masked region with Gaussian blur (`BLUR_RADIUS=12`)
6. Re-classify

**EN/ZH multilingual reference:** mean defended acc **74.9%**, clean Δ **−1.5pp**, cost 4.

### 7.3 KO/JA clean-damage defaults (from ablation)

Prefer **thr ≥ 0.95** always; use **`tight_dilate`** as the default geometry tweak
(or `no_bbox` when Clean Δ matters more than a couple pp of mean def on JA).

### 7.4 Historical (lineage only)

Under `_en_zh/en_zh_multi_uni_attack/`: GradCAM intersection (`cam_2mod` / `cam_4mod`)
and 4×4 grid occlusion (`grid_1patch` / `grid_2patch`). Superseded by Attn-last +
`cc_bbox_blur` for new work; still useful for cost baselines.

---

## 8. Evaluation metrics

| Metric | Definition |
|---|---|
| Clean / attacked / defence accuracy | `mean(pred == true)` |
| ASR | `mean(pred == target)` on attacked or defended images |
| Recovery rate | fraction of attacked-wrong images corrected by defence |
| Coverage | mean fraction of pixels masked |
| Clean degradation (Clean Δ) | `defence_acc_on_clean - clean_acc` (≤ 0) |
| Mean acc | average of the two scoring models |

Reported as fractions in JSON (×100 for %).

---

## 9. Notebook map

```
image_samples/attack_placement.py     ← bake / load attack_pos
attention_defense/                    ← Attn-last vs GradCAM
heatmap_defense_improvements/
  cc_bbox_blur/                       ← current EN/ZH winner
four_lang_cc_bbox_blur/               ← ZH/KO/JA transfer + pipeline figs
ko_ja_clean_damage/                   ← KO/JA Clean Δ ablations
_test_grid/                           ← improved conf-drop grid (frozen attack_pos)
_en_zh/en_zh_multi_uni_attack/        ← early multi/uni + CAM cost study
```

Typical 4-lang cell builder path: `_cells/06_data.py` loads the sample, asserts
`attack_pos`, and draws with clamped frozen anchors.

---

## 10. Key design decisions

### Why freeze coordinates in the sample JSON?
Fair comparison across defenses requires identical typographic geometry per image.
Storing `attack_pos.en` / `attack_pos.l` once prevents silent drift from font metrics
or copy-pasted RNG code.

### Why reference box size at bake time?
Positions were sampled against a conservative max box (`131×44`) so both slots stay
in-bounds for every language word; runtime clamps again after measuring the real word.

### Why Attn-last over GradCAM?
Cheaper (4 vs 6), higher defended accuracy, much lower clean-image damage.

### Why `cc_bbox_blur`?
Top-2 CC + bbox focuses the mask on text-like blobs; blur fill is kinder to clean
images than mean fill while matching best attacked accuracy.

### Why Option B (always score EN+L)?
Tests whether the EN∩L defense transfers when the partner language changes, under
uni-EN, uni-L, and multi attacks with the same frozen boxes.
