# Plan: Typographic Attack Confusion Matrices (Per-Language)

## Research question

The `updated_multilingual_consensus_colab.ipynb` experiment showed that a single
shared encoder means all languages collapse under attack together. The new question:
with **separate per-language CLIP models**, does a typographic attack written in
language X only fool the model trained on language X — or does it transfer?

This notebook answers that by running attacks in **all 4 languages** (en, zh, ko, ja)
and visualizing results as confusion matrices.

---

## Final outputs

### Output 1 — 4×4 Accuracy heatmap

```
              model_en  model_zh  model_ko  model_ja
attack_en     ??%       ??%       ??%       ??%
attack_zh     ??%       ??%       ??%       ??%
attack_ko     ??%       ??%       ??%       ??%
attack_ja     ??%       ??%       ??%       ??%
```

Rows = language of the word written on the image.
Cols = which per-language model is classifying.
Cell value = accuracy on attacked images.

**What to look for:** diagonal dominance would mean "a Chinese-script attack only
fools the Chinese model." Off-diagonal values reveal cross-lingual transfer.

### Output 2 — 4×4 ASR heatmap

Same grid, but cell = Attack Success Rate (`pred == written target class`).
High ASR = the attack successfully redirected the model to the written class.

### Output 3 — Per-model 10×10 sklearn confusion matrices

One 10×10 matrix per model (4 subplots), shown for that model's **most damaging
attack language** (highest ASR cell in its column). Axes = STL-10 class names in
the model's native language. Shows which classes get confused, not just aggregate
accuracy.

### Output 4 — `results/confusion_results.json`

Serialized `acc_matrix`, `asr_matrix`, and `clean_acc` for later reference.

---

## Target file

`notebooks/typographic_attack_confusion.ipynb` (new notebook)

---

## Notebook cell outline

| # | Cell | Contents |
|---|------|----------|
| 1 | Install & imports | pip install, open_clip reload guard, torch/numpy/matplotlib |
| 2 | Per-language model wrappers | `EnCLIP`, `ZhCLIP`, `KoCLIP`, `JaCLIP`, `CLASSES`, `TMPL`, `classify()` |
| 3 | `draw_word` + font map | OS-aware font paths (Colab/Linux vs Windows), font sanity-check strip |
| 4 | Load STL-10 | 200-sample index, `true` labels, `target` (adversarial target) labels, `clean` PIL list |
| 5 | Attack loop | For each `attack_lang` × `model_lang`: render word, classify, store predictions |
| 6 | Compute metrics | `acc_matrix[4×4]`, `asr_matrix[4×4]` from prediction dicts |
| 7 | 4×4 heatmaps | `seaborn.heatmap` for accuracy and ASR side by side |
| 8 | 10×10 confusion matrices | `sklearn.metrics.confusion_matrix`, 4-subplot figure |
| 9 | Save results | `results/confusion_results.json` |

---

## Key implementation notes

- **Models**: load all 4 once before the attack loop — do not reload per attack language
- **Font auto-detection**: `platform.system()` check so notebook runs on both Colab and Windows
  - Linux/Colab: `NotoSansCJK-Regular.ttc`, `DejaVuSans.ttf`
  - Windows: `msyh.ttc` (Microsoft YaHei, covers CJK), `arial.ttf`
- **Japanese tokenizer quirk**: call with `padding=True` only, no `return_tensors="pt"`
  (double-converts and throws `TypeError` — documented in CODE_GUIDE section 3.4)
- **Cropping guard**: also run `where="center"` to confirm models aren't trivially
  cropping out the attack text (CODE_GUIDE section 4 warning)
- **STL-10**: use `split="test"`, 200 random samples seeded at 0

---

## Models used

| lang | Hugging Face id | Library |
|------|----------------|---------|
| en | `ViT-B-32` (openai) | `open_clip` |
| zh | `OFA-Sys/chinese-clip-vit-base-patch16` | `transformers.ChineseCLIPModel` |
| ko | `Bingsu/clip-vit-base-patch32-ko` | `transformers.AutoModel` (VisionTextDualEncoder) |
| ja | `line-corporation/clip-japanese-base` | `transformers` + `trust_remote_code=True` |

---

## Connection to existing research

| Prior experiment | Finding | This experiment adds |
|---|---|---|
| `updated_multilingual_consensus_colab.ipynb` | Shared encoder → all languages collapse under attack simultaneously | — |
| `CODE_GUIDE_separate_langs_typographic.md` | Separate encoders disagree under EN attack | Now tests all 4 attack languages, adds confusion-matrix breakdown |
| `dual_encoder_divergence.ipynb` | LoRA-based per-language adapters | Different approach; this notebook uses zero-shot separate models |
