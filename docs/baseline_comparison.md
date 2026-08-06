# Baseline comparison (living doc)

**Protocol:** [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md)  
**Sample:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE.json`), 224×224  
**Production (ours):** gated **`cc_bbox_black`** — EN MIXED2000 **79.35%** (atk **72.9%** / clean **85.8%**); partner bilingual black ZH/KO/JA **81.65% / 78.35% / 82.53%**  
**Design-history ref:** always-on `cc_bbox_blur` (EN∩ZH multi) — mean **74.9%**, Clean Δ **−1.5pp**, cost **4**  
**Scope:** EN OpenAI ViT-B/32; ZH/KO/JA for spatial methods + Defense-Prefix (per-lang CIFAR-trained tokens). Dyslexify / SamplingTAR: EN + ZH/KO/JA hybrid n=1000 finals (ZH/JA GRID=14 HF/open_clip; KO HF).  
**Last updated:** 2026-08-05 (n=1000 finals for DP KO/JA, hybrids ZH/KO/JA, grid 4-lang)

Code: [`lib/notebooks/paper_baselines/`](../lib/notebooks/paper_baselines/)

---

## Leaderboard

| Method | Status | Acc (mean or EN) | Clean Δ | Cost | Notes |
|--------|--------|------------------|---------|------|-------|
| **Gated `cc_bbox_black` (ours)** | final (n=1000) | **72.9%** EN atk / **79.35%** MIXED | **−0.1pp** | 4 | Production; bilingual ZH black MIXED **81.65%** |
| Always-on `cc_bbox_blur` (ref) | final (n=1000) | **74.9%** mean | **−1.5pp** | 4 | Design history / localization ablation |
| OCR + blur | final (n=1000) | **78.8%** 4-lang mean | **−0.6pp** EN | 3 | Closest spatial peer; sticker hit 90.3%; KO/JA filled |
| SamplingTAR hybrid | final (n=1000) | **61.7%** 4-lang / **67.3%** EN | **−8.3pp** EN | 3 | Heads + attn-guided blur; ZH/KO/JA **68.9/59.0/51.5**; EN MIXED **72.45%** |
| Dyslexify hybrid | final (n=1000) | **60.3%** 4-lang / **66.9%** EN | **−8.1pp** EN | 3 | Heads + attn-guided blur; ZH/KO/JA **68.2/62.9/43.2**; EN MIXED **72.35%** |
| Defense-Prefix | final (n=1000) | **77.3%** 4-lang (EN/ZH/KO/JA 73.8/81.4/69.1/84.8) | **+0.5pp** EN | 2 | CIFAR-trained tokens; mean MIXED **84.1%** |
| 4×4 grid occlusion | final (n=1000) | **51.2%** 4-lang mean | **−3.2pp** EN | 62 | EN/ZH search; KO/JA score-only; mean MIXED **67.2%** |
| Dyslexify (heads) | final (n=1000) | **20.0%** EN | **0.0pp** | 2 | Head-only negative; dual-box weak |
| SamplingTAR (heads) | final (n=1000) | **11.6%** EN | **+0.2pp** | 2 | Circuit-only negative; weakest peer |

Statuses: `pending` → `smoke (n=100)` → `final (n=1000)` (or `failed` / `skipped` with reason).

---

## Reference: `cc_bbox_blur`

| Metric | Value |
|--------|------:|
| Defense mean acc (EN∩ZH multi) | 74.9% |
| EN defended / ZH defended | 71.6% / 78.2% |
| Clean Δ mean | −1.5pp |
| Inference cost | 4 |
| Source | `lib/notebooks/heatmap_defense_improvements/cc_bbox_blur/results/comparison_summary.json` |

---

## Defense-Prefix (Azuma & Matsui 2023)

**What:** Learned text-prefix token prepended to class prompts. CLIP weights frozen; one DP vector per language/model.  
**Setup:**
- **EN:** Published ImageNet `dp_vit-b32.pt` failed Gate A. Retrained 10 ep on CIFAR-10 **train** (n=20k, EN dual-box typos) → `dp_cifar10_vit-b32.pt` (OpenAI ViT-B/32). Prompt `a photo of a * {class}.`
- **ZH (retuned 2026-08-03):** Same recipe on ChineseCLIP ViT-B/16 with **EN+ZH multi** train stickers (matched to eval) — `train_cifar_dp_zh.py` / `zh_dp_encode.py`; 20 ep, lr=0.003 → `dp_cifar10_zh_vit-b16.pt`. Eval sample unused in both trains.
**Status:** final (n=1000) EN + ZH + KO + JA

| Split | EN / ZH / KO / JA atk | Cost | Notes |
|-------|------------------------|------|-------|
| smoke n=100 KO/JA | — / — / 66.0% / 88.0% | 2 | provisional ladder |
| **final n=1000** | **73.8 / 81.4 / 69.1 / 84.8** | 2 | MIXED **81.7 / 86.2 / 79.4 / 89.2**; 4-lang mean atk **77.3%**, mean MIXED **84.1%** |

**vs gated black:** EN DP still beats our EN MIXED (**81.65% > 79.35%**). ZH DP reaches **81.4%** atk / **86.2%** MIXED. 4-lang mean atk **77.3%** vs our EN∩ZH mean atk **74.7%**. Cost 2 vs 4.

---

## OCR + blur

**What:** External OCR boxes → Gaussian blur inside detections → reclassify EN (+ ZH).  
**Setup:** EasyOCR (`en`+`ch_sim`); blur radius 12 (match protocol). Log detect rate vs 2 stickers.  
**Status:** final (n=1000)

| Split | Mean acc | EN / ZH | Clean Δ | Cost | Sticker hit | Notes |
|-------|----------|---------|---------|------|-------------|-------|
| sanity n=16 | 65.6% | 50.0 / 81.2% | 0.0pp | 3 | 87.5% | detect_img 100% |
| smoke n=100 | 68.5% | 66.0 / 71.0% | 0.0pp | 3 | 88.0% | vs atk 7/3% |
| **final n=1000** | **73.8%** | 72.8 / 74.7% | **−0.7pp** | 3 | 90.3% | detect_img 100% |

**vs `cc_bbox_blur`:** Mean **73.8% < 74.9%** (−1.1pp). Clean Δ better (−0.7 vs −1.5pp). Misses ~9.7% of stickers; ZH defended 74.7% vs our 78.2%. Cost 3 vs 4. Our Attn-last localization beats OCR miss rate without an external detector.

---

## Dyslexify (Hufe et al. 2026)

**What:** Training-free ablation of typographic attention heads in the vision tower.  
**Setup:** open_clip ViT-B/32 openai (paper uses LAION ViT-B/16); mine heads by CLS→sticker-patch attn fraction; greedy + ranked-prefix select under Clean Δ ≤5pp; CLS←spatial redirect (`alpha=1`). EN + ZH/KO/JA hybrid finals. Fixed MHA tuple-return bug in hook.  
**Hybrid (2026-07-25):** same selected heads → aggregate CLS→patch attn → blur patches with score ≥ `0.35·max` (cap `top_k=4`, r=12) → classify with ablation still on. Code: `--mode hybrid` + [`_common/hybrid_spatial.py`](../lib/notebooks/paper_baselines/_common/hybrid_spatial.py).  
**Status:** final (n=1000) heads + hybrid (all langs)

| Split | Mode | EN / ZH / KO / JA atk | Clean Δ EN | Cost | Notes |
|-------|------|------------------------|------------|------|-------|
| **final n=1000** | heads | **20.0%** EN | **0.0pp** | 2 | EN-only negative |
| **final n=1000** | **hybrid** | **66.9 / 68.2 / 62.9 / 43.2** | **−8.1pp** | 3 | 4-lang mean atk **60.3%**, mean MIXED **71.5%** |

**vs `cc_bbox_blur`:** Heads-only far below (EN **20% vs 71.6%**). Hybrid clears the ~50% EN / MIXED2000 bar (EN **66.9%**, MIXED **72.35%**) but still trails OCR / ours; Clean Δ worse (−8.1pp vs −1.5pp). JA hybrid remains weak (43.2%).

---

## SamplingTAR (Liu et al., ECCV 2026)

**What:** Training-free circuit intervention — mine text-reading heads, redirect CLS attention at inference.  
**Setup:** EN ViT-B/32; head mining = CLS attn mass on sticker patches (no SAE; direct attribution proxy); z-threshold select; `fix_attn` alpha=1. EN + ZH/KO/JA hybrid finals.  
**Hybrid (2026-07-25):** same recipe as Dyslexify hybrid (attn-guided blur of selected-head patches + ablation). `--mode hybrid`. ZH port: `run_eval_zh.py` (ChineseCLIP HF, GRID=14).  
**Status:** final (n=1000) heads + hybrid (all langs)

| Split | Mode | EN / ZH / KO / JA atk | Clean Δ EN | Cost | Notes |
|-------|------|------------------------|------------|------|-------|
| **final n=1000** | heads | **11.6%** EN | **+0.2pp** | 2 | EN-only negative |
| **final n=1000** | **hybrid** | **67.3 / 68.9 / 59.0 / 51.5** | **−8.3pp** | 3 | 4-lang mean atk **61.7%**, mean MIXED **71.7%** |

**vs `cc_bbox_blur`:** Heads-only weakest peer (EN **11.6%**). Hybrid matches Dyslexify hybrid band (EN **67.3%**, MIXED **72.45%**) — confirms spatial blur on typo-attended patches, not head ablation alone, drives recovery.
