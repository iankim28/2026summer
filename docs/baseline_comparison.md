# Baseline comparison (living doc)

**Protocol:** [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md)  
**Sample:** frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE.json`), 224×224  
**Comparison target:** `cc_bbox_blur` (EN∩ZH multi) — mean **74.9%**, Clean Δ **−1.5pp**, cost **4**  
**Scope:** EN OpenAI ViT-B/32; ZH scored for spatial methods (mean). Attention / prompt methods report EN (+ note if ZH untreated).  
**Last updated:** 2026-07-23 (all four baselines final n=1000)

Code: [`lib/notebooks/paper_baselines/`](../lib/notebooks/paper_baselines/)

---

## Leaderboard

| Method | Status | Acc (mean or EN) | Clean Δ | Cost | Notes |
|--------|--------|------------------|---------|------|-------|
| **cc_bbox_blur** (ours) | final (n=1000) | **74.9%** mean | **−1.5pp** | 4 | EN∩ZH Attn-last → CC+bbox+blur |
| OCR + blur | final (n=1000) | **73.8%** mean | **−0.7pp** | 3 | Closest spatial peer; sticker hit 90.3% |
| Defense-Prefix | final (n=1000) | **73.8%** EN | **+0.5pp** | 2 | CIFAR-trained DP; EN-only; ASR 16.4% |
| Dyslexify | final (n=1000) | **20.0%** EN | **0.0pp** | 2 | Head ablation; weak on dual-box CIFAR |
| SamplingTAR | final (n=1000) | **11.6%** EN | **+0.2pp** | 2 | Circuit ablation; weakest peer |

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

**What:** Learned text-prefix token prepended to class prompts (`a photo of a [DP] {class}.`). CLIP weights frozen.  
**Setup:** Published ImageNet `dp_vit-b32.pt` failed Gate A on dual-box CIFAR (0/16 preds changed, ASR still 100%). Retrained 10 epochs on CIFAR-10 **train** (n=20k, synthetic dual-box typos; eval sample unused) → `defense_prefix/results/dp_cifar10_vit-b32.pt`. EN-only.  
**Status:** final (n=1000)

| Split | EN acc | ASR | Clean Δ | Cost | Notes |
|-------|--------|-----|---------|------|-------|
| sanity n=16 | 56.2% | 18.8% | −6.2pp | 2 | 13/16 preds changed |
| smoke n=100 | 68.0% | 16.0% | 0.0pp | 2 | vs vanilla atk 6.0% |
| **final n=1000** | **73.8%** | 16.4% | **+0.5pp** | 2 | vs vanilla atk 5.5% / ASR 94.4% |

**vs `cc_bbox_blur`:** EN defended **73.8% > 71.6%**, Clean Δ better (**+0.5pp** vs −2.2pp EN). Residual ASR **16.4%** ≫ our ~2.6%. No ZH treatment → not a drop-in for EN∩ZH mean **74.9%**. Cost 2 vs 4.

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
**Setup:** open_clip ViT-B/32 openai (paper uses LAION ViT-B/16); mine heads by CLS→sticker-patch attn fraction; greedy + ranked-prefix select under Clean Δ ≤5pp; CLS←spatial redirect (`alpha=1`). EN-only. Fixed MHA tuple-return bug in hook.  
**Status:** final (n=1000)

| Split | EN acc | Clean Δ | Cost | #heads | Notes |
|-------|--------|---------|------|--------|-------|
| sanity n=16 | 0.0% | 0.0pp | 2 | 12 | hook changes preds |
| smoke n=100 | 13.0% | −4.0pp | 2 | 4 | vs vanilla atk 7% |
| **final n=1000** | **20.0%** | **0.0pp** | 2 | 12 | ASR 78.3% (vanilla 95.3%) |

**vs `cc_bbox_blur`:** Far below (EN **20% vs 71.6%**). Modest lift over no-defense (4.5%→20%) with near-zero Clean Δ; not competitive with spatial defenses on this protocol.

---

## SamplingTAR (Liu et al., ECCV 2026)

**What:** Training-free circuit intervention — mine text-reading heads, redirect CLS attention at inference.  
**Setup:** EN ViT-B/32; head mining = CLS attn mass on sticker patches (no SAE; direct attribution proxy); z-threshold select; `fix_attn` alpha=1. EN-only.  
**Status:** final (n=1000)

| Split | EN acc | Clean Δ | Cost | #heads | Notes |
|-------|--------|---------|------|--------|-------|
| sanity n=16 | hook OK | — | 2 | — | Gate A: intervention fires |
| smoke n=100 | 12.0% | −4.0pp | 2 | 8 | z=2.0; vs atk 7% |
| **final n=1000** | **11.6%** | **+0.2pp** | 2 | 7 | z=2.0; ASR 87.5% |

**vs `cc_bbox_blur`:** Weakest peer (EN **11.6% vs 71.6%**). Same mechanistic family as Dyslexify; confirms head-only interventions are insufficient vs dual-box stickers here.
