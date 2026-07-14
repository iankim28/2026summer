# Experiment Protocol: EN/ZH Multi- vs. Unilingual Typographic Attack Study

> **Purpose**: Fully reproducible specification for the `en_zh_multi_uni_attack` experiment suite.
> Every parameter, formula, and design decision is recorded here so the experiments can be
> reconstructed independently.

---

## 1. Overview

The study asks: *does writing the adversarial word in both English and Chinese on the same image
attack both language-specific CLIP models simultaneously, and can GradCAM or grid occlusion detect
and remove it?*

Two attack setups are crossed with three defence strategies and evaluated on both an English CLIP
and a Chinese CLIP model.

---

## 2. Environment

### 2.1 Python packages

```
open_clip_torch    (latest)
transformers       (latest)
datasets           (latest)
matplotlib         (latest)
Pillow             (latest)
torch              (latest, GPU or CPU)
numpy              (latest)
```

Install:
```bash
pip install -q open_clip_torch transformers datasets matplotlib Pillow
```

### 2.2 Hardware

- GPU (CUDA) if available, otherwise CPU.
- `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`
- All models are set to `.eval()` mode.

---

## 3. Models

### 3.1 English CLIP (`EnCLIP`)

| Field | Value |
|---|---|
| Library | `open_clip` |
| Architecture | `ViT-B-32` |
| Pretrained weights | `openai` |
| Tokenizer | `open_clip.get_tokenizer('ViT-B-32')` |
| Image preprocessing | Model's own `create_model_and_transforms` pipeline |
| Text prompt template | `"a photo of a {class_name}."` |
| Image embedding | `model.encode_image(x)`, L2-normalised |
| Text embedding | `model.encode_text(tokens)`, L2-normalised |
| GradCAM hook layer | `model.visual.conv1` (first conv of the ViT patch embedding) |

### 3.2 Chinese CLIP (`ZhCLIP`)

| Field | Value |
|---|---|
| Library | `transformers` |
| HuggingFace model ID | `OFA-Sys/chinese-clip-vit-base-patch16` |
| Processor | `ChineseCLIPProcessor.from_pretrained(...)` |
| Text prompt template | `"一张{class_name}的照片。"` |
| Image embedding | `model.get_image_features(pixel_values=...)`, L2-normalised |
| Text embedding | `model.get_text_features(input_ids=..., attention_mask=..., token_type_ids=...)`, L2-normalised |
| Feature extraction | If output has `pooler_output` attribute, use that; otherwise use raw tensor |
| GradCAM hook layer | `model.vision_model.embeddings.patch_embedding` |

### 3.3 Classification

Cosine similarity is computed between L2-normalised image and text embeddings.
The predicted class is the `argmax` over all 10 class text embeddings.
Batch size for inference: `128` images.

**Class names:**

| Index | English | Chinese |
|---|---|---|
| 0 | airplane | 飞机 |
| 1 | automobile | 汽车 |
| 2 | bird | 鸟 |
| 3 | cat | 猫 |
| 4 | deer | 鹿 |
| 5 | dog | 狗 |
| 6 | frog | 青蛙 |
| 7 | horse | 马 |
| 8 | ship | 船 |
| 9 | truck | 卡车 |

---

## 4. Dataset

| Field | Value |
|---|---|
| Dataset | CIFAR-10 test split (`uoft-cs/cifar10` via HuggingFace Datasets) |
| HF column: images | `img` (fallback: `image`) |
| HF column: labels | `label` (fallback: `labels`) |
| Sample | Fixed balanced 1000-image sample — 100 images per class |
| Sample file | `../../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json` |
| Sample content | `{"idx": [...], "true": [...]}` |
| Verification | `assert len(idx) == 1000`, `assert all((true == c).sum() == 100 for c in range(10))` |

### 4.1 Target assignment

Each image receives a randomly chosen **target class** (the wrong class the attack tries to induce):

```python
rng    = random.Random(0)         # fixed seed
target = np.array([
    rng.choice([c for c in range(10) if c != int(true[k])])
    for k in range(len(idx))
])
```

- Seed: `0`
- Constraint: target ≠ true label
- One target per image, consistent across all experiments

### 4.2 Image preprocessing

All images are resized to **224 × 224 px** using bicubic interpolation before any attack or
classification:

```python
clean_224 = [im.resize((224, 224), Image.BICUBIC) for im in clean]
```

The English CLIP model applies its own additional normalisation internally via `open_clip`'s
`create_model_and_transforms` preprocessing pipeline at inference time.

### 4.3 Tune subset

A 100-image subset (10 per class, first index of each class) is used for hyperparameter tuning
(threshold sweep):

```python
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
```

---

## 5. Attack Design

### 5.1 Shared text box parameters

| Parameter | Value |
|---|---|
| `DISPLAY_SIZE` | `224` px (image canvas size) |
| `NUM_BOXES` | `2` (two text boxes per image) |
| `FONT_SIZE` | `24` pt |
| `PAD` | `8` px (padding inside each text box) |
| Box height extra | `+12` px added to text bounding height |
| Text colour | Black (`fill='black'`) |
| Box background | White (`fill='white'`) |

### 5.2 Font selection

| Platform | Latin font | CJK font |
|---|---|---|
| Windows | `%WINDIR%\Fonts\arial.ttf` | `%WINDIR%\Fonts\msyh.ttc` (Microsoft YaHei) |
| Linux/Mac | `NotoSans-Regular.ttf` | `NotoSansCJK-Regular.ttc` |
| Fallback | `ImageFont.load_default()` | same |

A word containing any character with `ord(c) > 127` uses the CJK font; all-ASCII words use the
Latin font.

### 5.3 Box placement algorithm

Box positions are **deterministic, per-image, per-box-index** using a seeded RNG:

```python
rng = random.Random(int(img_idx) * NUM_BOXES + box_i)
```

- `img_idx`: integer index of the image in the 1000-sample
- `box_i`: 0 for the first box, 1 for the second box

Placement procedure:
1. Compute box size: `bw = text_width + 2*PAD`, `bh = text_height + PAD + 12`
2. Define valid x range: `[0, 224 - bw]`; y range: `[0, 224 - bh]`
3. Sample `(rect_x, rect_y)` from the RNG up to **64 attempts**
4. Accept if the proposed rectangle does not overlap any already-placed box
5. If all 64 attempts fail, use the last sampled position regardless

Non-overlap check: two rectangles overlap unless one is fully left/right/above/below the other
(standard AABB collision).

### 5.4 Multilingual attack

**Box-0**: English attack word (e.g. `"dog"`) using Latin font  
**Box-1**: Chinese attack word (e.g. `"狗"`) using CJK font  
Both boxes contain translations of the **same target class** for that image.

```python
en_word = CLASSES['en'][target[img_idx]]
zh_word = CLASSES['zh'][target[img_idx]]
```

The attack targets both CLIP models simultaneously: EN CLIP reads Box-0, ZH CLIP reads Box-1.

### 5.5 Unilingual attack

**Box-0**: English attack word using Latin font  
**Box-1**: Same English attack word again, using Latin font  
No Chinese text is present.

```python
word = CLASSES['en'][target[img_idx]]
```

The attack is designed to fool the EN CLIP model; ZH CLIP is exposed to English text (which it
was not trained to attend to).

---

## 6. Defence Designs

### 6.1 GradCAM helpers

#### EN CLIP GradCAM
- Hook registered on `model.visual.conv1`
- Forward pass through `model.visual(x)`
- Score: cosine similarity of image feature with target text embedding
- Gradient w.r.t. the conv1 activation tensor
- CAM: `relu((grad.mean(dim=[2,3], keepdim=True) * activation).sum(dim=1))`, min-max normalised

#### ZH CLIP GradCAM
- Hook registered on `model.vision_model.embeddings.patch_embedding`
- Forward pass through `model.get_image_features(pixel_values=...)`
- Score: cosine similarity of image feature with target text embedding
- Same CAM formula as above

#### CAM resolution
EN ViT-B/32 produces a ~7×7 spatial CAM; ZH ViT-B/16 produces a ~14×14 spatial CAM.
Both are resized to **224×224** via bilinear interpolation before any masking operation.

### 6.2 Cross-language text embeddings (for cam_2mod unilingual and cam_4mod multilingual)

Four `(model_lang, text_lang)` combinations are computed once after model load:

| Key | Description |
|---|---|
| `('en', 'en')` | EN model tokenises EN class names — standard |
| `('zh', 'zh')` | ZH model tokenises ZH class names — standard |
| `('en', 'zh')` | EN model tokenises ZH class names (CJK glyphs become UNK/subwords in the EN tokenizer) |
| `('zh', 'en')` | ZH model tokenises EN class names |

All four are L2-normalised.

### 6.3 cam_2mod defence

**Combos used:**

| Setup | Combos |
|---|---|
| Multilingual | `[('en','en'), ('zh','zh')]` |
| Unilingual | `[('en','en'), ('zh','en')]` |

**Procedure per image:**
1. Compute GradCAM for each combo → resize to 224×224
2. Intersect: elementwise `min` of all CAMs → saliency map
3. Threshold saliency at the **p-th percentile** (where p = best_threshold × 100) to get a binary mask
4. Dilate mask with 3 iterations of 3×3 dilation
5. Fill masked pixels with the **mean colour of the non-masked pixels** in that image
6. Classify the masked image

**Threshold tuning** (on the 100-image tune subset):
- Candidates: `[0.75, 0.80, 0.85, 0.90, 0.95]`
- Selection criterion: maximise masked accuracy for the EN model on the tune subset
- Chosen threshold for all experiments: **0.85**

**Inference cost:** `2 (classify) + 2×2 (2 GradCAM backprops + 2 re-classify) = 6` forward passes

### 6.4 cam_4mod defence (multilingual only)

**Combos:** `[('en','en'), ('en','zh'), ('zh','en'), ('zh','zh')]`

**Procedure:** identical to cam_2mod but intersecting 4 CAMs instead of 2.

**Threshold tuning:** same sweep; chosen threshold: **0.90**

**Inference cost:** `2 + 2×4 = 10` forward passes

### 6.5 CAM cache

To avoid recomputing GradCAMs multiple times, each (model_lang × text_lang × condition ×
image-set) combination is saved to a `.npz` file under `results/*/cache/`.  
Cache filename: `cams_{combo_label}_{condition}_n{n}.npz`  
Keys inside: `cam_en_en`, `cam_zh_zh`, `cam_en_zh`, `cam_zh_en` (as applicable), `indices`

### 6.6 grid_1patch defence

A **4×4 non-overlapping grid** divides the 224×224 image into **16 patches** of **56×56 px** each.

```
Patch layout (row-major):
 (0,0,56,56)    (56,0,112,56)   (112,0,168,56)  (168,0,224,56)
 (0,56,56,112)  ...             ...             ...
 ...            ...             ...             ...
 (0,168,56,224) ...             ...             (168,168,224,224)
```

**Scoring:** for each of the 16 candidate occluded images, compute:

```
score = mean over {en, zh} of max_cosine_similarity(image_features, all_10_class_text_features)
```

A higher score = both models are more confident in some class = adversarial text more likely
occluded.

**Fill colour:** mean RGB of all pixels in the original image, applied as a flat rectangle.

**Selection:** `argmax(scores)` → that patch is occluded and the image is returned.

**Inference cost:** `16 patches × 2 models = 32` forward passes

### 6.7 grid_2patch defence (greedy)

1. Run grid_1patch → select best patch p1
2. Apply p1 occlusion to get image_p1
3. Try all remaining 15 patches on image_p1, score each
4. Select best patch p2 → occlude both p1 and p2

**Inference cost:** `16 (step 1) + 15 (step 3) = 31 evaluations × 2 models = 62` forward passes

---

## 7. Evaluation Metrics

| Metric | Formula |
|---|---|
| Clean accuracy | `mean(argmax(cosine_sim) == true_label)` on unmodified images |
| Attacked accuracy | `mean(argmax(cosine_sim) == true_label)` on attacked images |
| Attack Success Rate (ASR) | `mean(argmax(cosine_sim) == target_label)` on attacked images |
| Defence accuracy | `mean(argmax(cosine_sim) == true_label)` on defended images |
| Defence ASR | `mean(argmax(cosine_sim) == target_label)` on defended images |
| Recovery rate | `(wrong_before_defence AND correct_after_defence).sum() / wrong_before_defence.sum()` |
| Mean acc | Average of EN and ZH model accuracy for a given condition |
| Coverage | Mean fraction of pixels masked per image |
| Clean degradation | `defence_acc_on_clean - clean_acc` (should be ≤ 0) |

All reported as fractions in JSON (multiply by 100 for %).

---

## 8. Notebook Execution Order

```
attack_comparison  →  cam_defense  →  grid_defense  →  cost_vs_performance
```

Each subdirectory (`multilingual/`, `unilingual/`) is independent. Run both in parallel or
sequentially. `cost_vs_performance.ipynb` must run last as it reads all output JSONs.

### 8.1 Output files per experiment

```
results/
  attack/
    confusion_results.json          ← no_defense baseline
    per_class_accuracy.png
    gradcam_heatmaps.png
    sample_viz.png
  cam_2mod/
    confusion_results_cam_defense.json
    threshold_sweep.png
    accuracy_delta_matrix.png
    mask_examples.png
    cache/                           ← .npz CAM arrays
  cam_4mod/                          ← multilingual only
    confusion_results_cam_defense.json
    accuracy_delta_matrix.png
    cache/
  grid_1patch/
    confusion_results_grid.json
    grid_defence_examples.png
  grid_2patch/
    confusion_results_grid.json
```

### 8.2 JSON schema (all result files)

All JSON files share these fields:

| Field | Type | Description |
|---|---|---|
| `setup` | str | `"multilingual"` or `"unilingual"` |
| `method` | str | `"no_defense"`, `"cam_2mod"`, `"cam_4mod"`, `"grid_1patch"`, `"grid_2patch"` |
| `attack` | str | Same as `setup` |
| `n_images` | int | `1000` |
| `inference_cost` | int | Forward passes per image |
| `clean_acc` | dict | `{"en": float, "zh": float}` |
| `baseline_acc` | dict | Attacked acc before defence |
| `baseline_asr` | dict | Attacked ASR before defence |
| `defense` | dict | Per-model `{acc, asr, recovery_rate, baseline_acc, baseline_asr}` |
| `defense_acc_mean` | float | Mean of EN + ZH defence acc |
| `defense_asr_mean` | float | Mean of EN + ZH defence ASR |

Additional fields in CAM JSONs:
- `best_threshold`: chosen percentile threshold
- `combos`: string of `(model_lang, text_lang)` pairs used
- `coverage_mean`: fraction of image masked on average
- `clean_degradation`: per-model `{baseline_acc, masked_acc, delta_acc}` on clean images

---

## 9. Regenerating Notebooks

All six notebooks and `cost_vs_performance.ipynb` are generated from a single builder script:

```bash
cd lib/notebooks/en_zh_multi_uni_attack
python _build_notebooks.py
```

This recreates all `.ipynb` files from scratch. Existing result JSONs and images are not
affected — only the notebook source code is overwritten.

---

## 10. Inference Cost Table

| Method | Description | Forward passes / image |
|---|---|---:|
| `no_defense` | classify only (EN + ZH) | 2 |
| `cam_2mod` | 2 GradCAM backprops + 2 re-classify | 6 |
| `cam_4mod` | 4 GradCAM backprops + 2 re-classify | 10 |
| `grid_1patch` | 16 patch candidates × 2 models | 32 |
| `grid_2patch` | 31 patch evaluations × 2 models | 62 |

---

## 11. Key Design Decisions and Rationale

### Why 2 boxes per image?
Two boxes were chosen to increase attack surface (more text pixels) while keeping the image
visually recognisable. Single-box attacks are a special case; dual-box with the same seed scheme
ensures reproducibility.

### Why seeded random placement?
Fixed seeds (`random.Random(img_idx * NUM_BOXES + box_i)`) make box positions deterministic and
reproducible without storing coordinates. Any re-run of the code on the same image produces
identical box positions.

### Why mean-colour fill for CAM masking?
Replacing masked pixels with the mean of the non-masked region avoids introducing a systematic
colour bias (e.g. all-black or all-white) that might itself shift CLIP's predictions.

### Why percentile threshold rather than absolute?
Saliency map magnitudes vary across images; a fixed absolute threshold would mask very little on
some images and almost everything on others. A percentile threshold gives consistent coverage.

### Why threshold tuned on EN model accuracy?
EN CLIP was expected to be the primary target of the attack. ZH model accuracy under CAM masking
was treated as a secondary metric.

### Why mean of model confidences for grid scoring?
The grid defence is model-agnostic. Averaging the max-cosine-sim of both models equally weights
both languages. This avoids privileging one model's "confidence" signal over the other.

### Why greedy 2-patch rather than exhaustive?
Exhaustive 2-patch search would require evaluating `C(16,2) = 120` pairs × 2 models = 240 forward
passes. Greedy (fix 1st, search 2nd) costs only 62 and nearly always reaches the same result when
the dominant patch is the text box.
