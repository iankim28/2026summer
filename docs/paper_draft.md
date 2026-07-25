# Paper Draft — Outline Only

**Working title (placeholder):** Cross-Lingual Attention Intersection as a Spatial Defense Against Typographic Attacks on Multilingual CLIP  
**Status:** Outline / idea map — not prose. Expand each bullet into paragraphs later.  
**Scope of this draft:** Thread B defense line (separate per-language CLIPs + typographic attacks + saliency masking). Thread A (shared multilingual CLIP + PGD) can be mentioned briefly as motivation / contrast, not as the main contribution.  
**Last synced:** 2026-07-23 — includes frozen `attack_pos` protocol, thr ≥ 0.95 floor, attack-detector gating, and four published baselines (`docs/baseline_comparison.md`).

---

## 1. Introduction

### 1.1 Motivation — why this problem matters
- Vision–language models (CLIP-style) do zero-shot image classification by matching images to text class names.
- **Typographic attacks:** overlaying adversarial class-name text on the image can flip the prediction even though the object is unchanged (cite typographic-attack / CLIP vulnerability literature).
- Unlike invisible pixel-level adversarial noise, these attacks are human-visible but still fool models that “read” text in the image.
- Practical risk: any deployed CLIP-like classifier that sees photos containing text (signs, stickers, UI overlays).

### 1.2 Multilingual angle — four languages from the start
- Prior idea: use **multiple languages** so an attack tuned to one language fails on others (shared-encoder multilingual CLIP defenses vs separate encoders).
- Short contrast (1–2 sentences): on a **shared** image encoder, language disagreement often fails as a detector under gradient attacks (Thread A finding — cite as background, not main result).
- This paper uses **four separate per-language CLIP models — English (EN), Chinese (ZH), Korean (KO), Japanese (JA)** — under **typographic** (text-overlay) attacks.
- Defense is always **EN ∩ L**: English CLIP attention intersected with a partner-language CLIP `L ∈ {ZH, KO, JA}`. All three partners are first-class throughout Methods and Results (not an EN/ZH study with late transfer).

#### 1.2.1 Why English anchors every pairing (and most attacks)
- Early 4×4 typographic landscape (attack language × model): **English overlays are the universal / highest-ASR threat** across all four models (EN ASR ~94.5%, KO ~86%, JA ~90%, ZH ~65% under EN attack on n=200).
- Native-script attacks (ZH/KO/JA stickers) transfer poorly to foreign models; Pure L on KO/JA is often already weak.
- Therefore the paper **always pairs EN with partner L** for the spatial defense, and centers evaluation on **Pure E** and **E + L** (the hard cases). Pure L is kept for completeness and to expose EN-attention limits on non-Latin glyphs — not because native-only stickers are the main threat.
- One-sentence framing for prose: *English text is the dominant cross-model typographic threat, so every defense pairing includes the English CLIP and every hard attack includes at least one English sticker.*

### 1.3 Gap — detection alone is not enough
- Prediction disagreement across models can flag some attacks (modest AUC), but does not restore the correct label.
- Need a **spatial defense**: find and neutralize the text sticker(s), then reclassify.
- Existing saliency tools (GradCAM) are a natural candidate but may be costly / imprecise for small text boxes.
- Always-on spatial repair still risks **clean-image damage** (especially EN∩KO / EN∩JA) — motivates a learned **attack detector** that applies blur only when stickers are present.

### 1.4 Core idea (contribution preview)
- Cross-lingual **attention agreement**: mask where English CLIP and partner CLIP L both attend (intersection), for each `L ∈ {ZH, KO, JA}`.
- Prefer **last-layer attention** over GradCAM (cheaper + more accurate in our setting).
- Post-process the mask into sticker-shaped regions and soft-occlude with blur (`cc_bbox_blur`).
- Optionally **gate** `cc_bbox_blur` with an Attn-last heatmap-shape detector (blur only when attacked is predicted) → Clean Δ ≈ 0 while keeping attacked accuracy.
- Evaluate under three dual-box attack types — **Pure E**, **E + L**, **Pure L** — on all three partner languages.
- Compare against published peers: OCR+blur, Defense-Prefix, Dyslexify-style / SamplingTAR-style head ablations, plus grid occlusion.

![cc_bbox_blur pipeline overview](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_steps.png)

*Figure idea (intro / method teaser): attacked image → Attn-last EN ∩ L → intersection → CC+bbox → blur fill (shown for partner languages ZH / KO / JA).*

### 1.5 Contributions (bullet list for the paper’s claim set)
- Show that **EN ∩ L attention intersection** (`L ∈ {ZH, KO, JA}`) recovers accuracy under dual-box typographic attacks on a **frozen, class-balanced CIFAR-10 n=1000** evaluation set.
- Show **Attn-last beats GradCAM** on accuracy, compute cost, and clean-image side effects (Pure L / ZH-only as the main caveat).
- Introduce / validate **`cc_bbox_blur`** (connected-component bbox snap + blur fill) as a cheap refinement over raw attention masks.
- Show the recipe works for **all three partners (ZH, KO, JA)** under a shared protocol (`attack_pos` frozen; thr ≥ 0.95).
- Show always-on defense causes only a **minor Clean Δ** on EN∩ZH (≈ −1.5 pp); KO/JA remain larger until gated.
- Introduce an **Attn-last attack detector** that gates `cc_bbox_blur`: Clean Δ collapses to ~0 for ZH/KO/JA with ≤0.45 pp attacked-acc drop (`multi`).
- Beat or match published baselines on the same protocol: **OCR+blur** (closest spatial peer), **Defense-Prefix** (strong EN-only after CIFAR retrain), and show **Dyslexify / SamplingTAR**-style head ablations are weak negative baselines.
- Provide an additional **negative baseline**: coarse grid occlusion — even with **confidence-drop scoring** or **exhaustive** 2-patch search — remains weaker and much more expensive than attention.

### 1.6 Paper roadmap (one sentence)
- Section 2 methods (dataset + frozen geometry, 4 models, 3 attack types, `cc_bbox_blur`, detector gate, published + grid baselines) → Section 3 results (clean Δ / gated clean Δ; 4-lang transfer; EN∩ZH ablations; baseline leaderboard) → Section 4 conclusion / limitations / next steps.

---

## 2. Method / Materials

### 2.1 Dataset and evaluation protocol
- **Source dataset:** CIFAR-10 (10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), loaded via HuggingFace `uoft-cs/cifar10` (standard test split).
- **Evaluation set (current, frozen):**
  - From the CIFAR-10 test split, build a **balanced 1000-image sample**: exactly **100 images per class**, drawn randomly within each class with seed 0, then shuffled.
  - Saved as [`lib/notebooks/image_samples/CIFAR10_BALANCED_1000_SAMPLE.json`](../lib/notebooks/image_samples/CIFAR10_BALANCED_1000_SAMPLE.json) (indices + true labels).
  - Replaces an earlier seed-0 random 1000 draw that was class-imbalanced (e.g. cat over-represented, horse under-represented), and supersedes early Thread B STL-10 / n=200 exploratory runs for all main claims.
  - **Attack geometry frozen in the same JSON** under `attack_pos`: per-image top-left anchors for the two dual-box slots (`en` = slot 0, `l` = slot 1), baked once with a fixed reference box size (`131×44`) so every defense / baseline notebook shares identical sticker placement (see [`PROTOCOL.md`](../lib/notebooks/PROTOCOL.md)).
- **Protocol knobs (production setting, 2026-07-21+):**
  - Upscale images to 224×224 for overlays and CLIP inference.
  - Tune percentile thresholds on a **100-image** subset (10 per class), then set `threshold = max(threshold_free, 0.95)` for the full **n=1000** run; log both values.
  - Report Clean Δ on the same 1000 images **without** attack stickers, run through the identical mask pipeline (or gated pipeline).
  - Cite thr-floor four_lang numbers for transfer; EN∩ZH ablation winner from `heatmap_defense_improvements/cc_bbox_blur` remains **74.9% / −1.5 pp** as the detailed case-study quote (same recipe; pre-freeze RNG geometry).
- **Metrics:**
  - Top-1 accuracy after attack / after defense (per model and mean of the EN+L pair).
  - Attack success rate (ASR) where useful.
  - **Clean Δ:** accuracy change when the same defense is applied to unattacked images (side-effect cost).
  - **Coverage:** fraction of pixels masked (tight vs over-masking).
  - **Cost:** forward/backward / OCR / intervention passes per image.

### 2.2 Models (separate per-language CLIPs)
- **EN:** OpenAI ViT-B/32 via `open_clip`.
- **ZH:** Chinese-CLIP ViT-B/16 (`OFA-Sys/chinese-clip-vit-base-patch16`).
- **KO:** `Bingsu/clip-vit-base-patch32-ko`.
- **JA:** `llm-jp/llm-jp-clip-vit-base-patch16` (note: earlier CLYP Japanese model was broken / near-chance — replaced).
- Clean accuracy ballpark on the balanced n=1000: EN ~85.9%, ZH ~91.4%, KO ~89.6%, JA ~92.5%.
- Defense always pairs **EN with one partner L** (never a ZH-only pipeline).

### 2.3 Attack construction (typographic)
- Method: render adversarial class name as text on the image (`draw_word`); no gradients.
- **Dual-box geometry** (main threat model):
  - Two non-overlapping white text boxes at frozen `attack_pos` coordinates.
  - Font size fixed (24 @ 224×224 in defense experiments).
- **Three attack types** (poster names; notebook codes in parentheses):

| Poster name | Notebook code | Box 0 | Box 1 |
|---|---|---|---|
| **Pure E** | `uni_en` | English | English |
| **E + L** | `multi` | English | Partner language L |
| **Pure L** | `uni_l` | Partner L | Partner L |

- For each partner `L ∈ {ZH, KO, JA}`, evaluate all three attack types under the same EN ∩ L defense.
- Earlier single-box 4×4 confusion matrices (attack language × model) provide attack-landscape background.

![dual-box attack types Pure E / E+L / Pure L](../lib/notebooks/four_lang_cc_bbox_blur/results/attack_types_strip.png)

*Figure: dual-box typographic attack geometry — Pure E / E+L / Pure L (partner L = ZH here; same layout for KO/JA).*

#### 2.3.1 English-centric attack design (pointer)
- Rationale already stated in **§1.2.1**; Methods only operationalizes it: slot 0 is always the EN box position; Pure E / E+L / Pure L matrix uses the same frozen geometry for every partner L.
- Brief reminder if needed: English = highest-ASR attack language; Pure L retained as the weak / completeness cell.

### 2.4 Defense pipeline (main method)
Describe as a fixed sequence for any partner `L ∈ {ZH, KO, JA}`:

1. **Saliency maps** from EN model and partner model L (on the attacked image).
2. **Intersection** EN ∩ L (agreement = spatial cue that both models are “looking at” the same suspicious region).
3. **Percentile threshold** on the intersection heatmap (tuned on n=100; **enforce thr ≥ 0.95**).
4. Optional **dilation** (default 3×3×3; KO/JA may use tighter dilate).
5. **Mask shaping (`cc_bbox`):** keep top-2 connected components → snap each to axis-aligned bounding box (match sticker rectangles).
6. **Fill (`blur`):** Gaussian blur inside the mask (vs mean-color fill) — smash glyphs, preserve more object structure.
7. **Reclassify** both models on the defended image.

Name the full stack: **`cc_bbox_blur`** on top of Attn-last intersection.

![cc_bbox_blur full stage grid](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_examples.png)

*Figure: qualitative stages on E+L dual-box examples for partners ZH, KO, and JA.*

![mean fill vs blur fill](../lib/notebooks/four_lang_cc_bbox_blur/results/pipeline_fill_compare.png)

*Figure: after CC+bbox shaping, blur fill vs hard mean fill.*

### 2.5 Optional gate — Attn-last attack detector
- Motivation: always-on `cc_bbox_blur` still hurts clean KO/JA (~−11 pp); want blur **only when stickers are present**.
- **Features (26 scalars)** from Attn-last EN / L / intersection maps: entropy, top-k mass, Gini, kurtosis, cov95, CC count, EN–L corr/IoU, etc.
- **Classifier:** logistic + calibrated linear SVM; image-level 70/15/15 split; choose val threshold for **attack-recall ≥ 0.99** (keep gated atk drop <1 pp).
- **Policies:** `never_defend` / `always_defend` / `gated` (apply `cc_bbox_blur` iff detector says attack).
- Success bar: Clean Δ improves ≥1 pp vs always; attacked mean acc drops ≤1 pp.
- Code: [`lib/notebooks/attack_detector/`](../lib/notebooks/attack_detector/); evaluated on `multi` for ZH/KO/JA.

### 2.6 Saliency variants compared
- **GradCAM** (cost ~6) — prior / production-style baseline.
- **Attn-rollout** (cost ~4).
- **Attn-last** (cost ~4) — primary signal.
- Same EN ∩ L intersection + masking wrapper for fair comparison (detailed ablation case study: EN∩ZH).

![GradCAM vs attention heatmaps](../lib/notebooks/attention_defense/results/heatmap_comparison.png)

*Figure: GradCAM vs Attn-last vs Attn-rollout on the same attacked images.*

### 2.7 Published baselines (same frozen dual-box protocol)
Living numbers: [`docs/baseline_comparison.md`](baseline_comparison.md). Code: [`lib/notebooks/paper_baselines/`](../lib/notebooks/paper_baselines/). Scope for spatial mean: EN∩ZH `multi` unless noted EN-only.

| Method | What it does | Scope | Cost |
|---|---|---|---:|
| **OCR + blur** | EasyOCR (`en`+`ch_sim`) boxes → Gaussian blur r=12 → reclassify EN+ZH | EN+ZH | 3 |
| **Defense-Prefix** | Learned prompt prefix token; CLIP frozen. Published ImageNet DP failed Gate A → **retrain 10 ep on CIFAR-10 train** (synthetic dual-box typos; eval sample held out) | EN-only | 2 |
| **Dyslexify-style** | Mine typographic attention heads; CLS←spatial redirect at inference | EN-only | 2 |
| **SamplingTAR-style** | Circuit / head ablation via CLS-attn mass + `fix_attn` | EN-only | 2 |

Smoke ladder per method: n=16 sanity → n=100 smoke → **n=1000 final**.

### 2.8 Ablations and negative controls (Methods, not full results)
- **No defense** (attacked accuracy floor).
- **Grid occlusion search** (4×4 patches) — two axes to report:
  1. **Scoring:** old **max-confidence** vs **confidence-drop** of the pre-defense top class.
  2. **Search:** **greedy** 2-patch vs **exhaustive** search over all C(16,2)=120 patch pairs.
  - High pass count (~62 greedy; ~240 exhaustive).
- Heatmap ablations that failed or helped little (to justify design choices):
  - Union masks (EN ∪ L) — hurts clean images.
  - Disagreement / peakiness gating — rarely fires or too aggressive (superseded by learned detector).
  - Peaked-heads only; EN ViT-B/16 instead of B/32 — no gain / worse clean Δ.
  - Attention + conf-drop hybrid — better than full grid, worse than plain Attn-last at higher cost.
- KO/JA clean-damage variants: thr floor 0.95, tighter dilate, no bbox snap, coverage cap.
- Free-tune vs thr-floor sensitivity under frozen `attack_pos` (protocol note / appendix).

![grid defence examples](../lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_grid/results/defence_examples.png)

*Figure idea (optional): qualitative grid-occlusion baseline.*

### 2.9 Implementation notes (short)
- Shared protocol: [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md).
- Notebooks: `attention_defense`, `heatmap_defense_improvements`, `four_lang_cc_bbox_blur`, `ko_ja_clean_damage`, `attack_detector`, `paper_baselines/*`, grid: `_test_grid` (+ lineage under `_en_zh/`).
- Thresholds chosen on tune set with thr ≥ 0.95 floor; report frozen thr / coverage alongside accuracy.
- Pipeline figure regenerator: `four_lang_cc_bbox_blur/make_pipeline_viz.py` (EN ∩ ZH / KO / JA examples).

---

## 3. Results

*Outline of result blocks / figures / tables — fill with exact numbers when writing. Quote sets: (i) EN∩ZH case-study winner `cc_bbox_blur` **74.9% / −1.5 pp** from `heatmap_defense_improvements`; (ii) four_lang transfer under frozen `attack_pos` + thr≥0.95; (iii) gated detector; (iv) published baseline leaderboard.*

### 3.1 Clean-image side effects (no attack) — lead claim
- Apply the same `cc_bbox_blur` defense to **unattacked** images; measure Clean Δ.
- **EN ∩ ZH (always-on, thr≥0.95):** Clean Δ ≈ **−1.5 pp** — only a minor drop; defense does not meaningfully deteriorate clean performance.
- Contrast: GradCAM intersection Clean Δ ≈ **−25 to −35 pp** on the same style of protocol.
- **EN ∩ KO / EN ∩ JA (always-on, thr≥0.95):** Clean Δ ≈ **−11.2 / −11.5 pp** (restored from free-tune collapses to −25 / −23 when thr fell to 0.85/0.90). Still larger than ZH.
- **Gated `cc_bbox_blur` (`multi`):** Clean Δ → **0.0 / −0.2 / 0.0 pp** for ZH / KO / JA — the Clean-Δ story for KO/JA.
- Poster takeaway: ZH is nearly clean-safe always-on; KO/JA need the detector gate (or accept ~−11 pp).

### 3.2 Attack landscape (brief)
- English text overlays are a **universal** threat across EN/ZH/KO/JA models; native-script Pure L attacks are often weaker on KO/JA.
- Point to 4×4 accuracy / ASR matrix from CIFAR-10 typographic study if needed for context.
- Disagreement detector: works modestly (above chance) but is not the main contribution of this paper — spatial repair (+ optional heatmap gate) is.

### 3.3 Four-language transfer (L ∈ {ZH, KO, JA}) — main defense result
- Design matrix: for each L, evaluate **Pure E / Pure L / E + L** with EN ∩ L `cc_bbox_blur` under **frozen `attack_pos` + thr ≥ 0.95**.
- Headline mean defended acc (thr-floor, n=1000):

| Cell | Atk mean | Def mean | Clean Δ |
|---|---:|---:|---:|
| zh / Pure E | 14.3% | **60.2%** | −1.5 |
| zh / Pure L | 56.1% | **67.2%** | −1.5 |
| zh / E+L | 5.5% | **74.0%** | −1.5 |
| ko / Pure E | 8.3% | **63.7%** | −11.2 |
| ko / Pure L | 74.1% | **68.4%** | −11.2 |
| ko / E+L | 7.9% | **69.9%** | −11.2 |
| ja / Pure E | 3.5% | **72.3%** | −11.5 |
| ja / Pure L | 78.0% | **72.8%** | −11.5 |
| ja / E+L | 4.8% | **75.9%** | −11.5 |

- Claims:
  - ZH **E + L** nearly reproduces the EN/ZH case-study winner (74.0% vs 74.9%; −0.9 pp under frozen geometry).
  - Hard attacks (Pure E, E + L) recover to mid-60s–mid-70s mean for KO/JA as well.
  - Pure L often already weak on KO/JA (defense can slightly hurt); ZH Pure L is the exception where defense helps more.
  - Always-on KO/JA Clean Δ is the language-dependent failure mode — fixed by gating (§3.8).
- Protocol note (appendix / one paragraph): free thr-tune under freeze had dropped multi thr to 0.85–0.90 and inflated Clean Δ; flooring at 0.95 restored accuracy and clean cost. Report both if reviewing sensitivity.
- Figure: language × attack panel of attacked vs defended mean acc + clean Δ.

![4-lang cc_bbox_blur transfer](../lib/notebooks/four_lang_cc_bbox_blur/results/final_comparison.png)

### 3.4 Attn-last vs GradCAM vs Attn-rollout (EN∩ZH case study)
- Detailed saliency comparison under the same intersection wrapper (supports the Attn-last choice used for all partners).
- Key claims:
  - **E + L (EN+ZH):** Attn-last ~**72.6%** mean vs GradCAM ~**33%**; lower coverage; better clean Δ.
  - **Pure E (EN+EN):** same ranking (Attn-last ~**67.6%**).
  - **Pure L (ZH+ZH):** Attn-last still best or tied on accuracy (~**62.5%**), but clean damage / coverage worsen — limitation of EN attention on Chinese glyphs.
- Figure idea: bar chart of mean acc by method × attack; optional coverage / clean-Δ panel.

![Attn-last vs GradCAM final comparison](../lib/notebooks/attention_defense/results/final_comparison.png)

![unilingual attention defense](../lib/notebooks/attention_defense/unilingual/results/final_comparison.png)

### 3.5 Published baselines vs `cc_bbox_blur` (EN∩ZH `multi`, n=1000)
- Leaderboard (from [`baseline_comparison.md`](baseline_comparison.md)):

| Method | Acc | Clean Δ | Cost | Notes |
|---|---:|---:|---:|---|
| **`cc_bbox_blur` (ours)** | **74.9%** mean | **−1.5pp** | 4 | EN∩ZH Attn-last → CC+bbox+blur |
| OCR + blur | 73.8% mean | −0.7pp | 3 | Closest spatial peer; sticker hit 90.3% |
| Defense-Prefix (CIFAR-trained) | 73.8% EN | +0.5pp | 2 | EN-only; residual ASR 16.4% |
| Dyslexify-style | 20.0% EN | 0.0pp | 2 | Head ablation; weak on dual-box |
| SamplingTAR-style | 11.6% EN | +0.2pp | 2 | Circuit ablation; weakest peer |

- Talking points:
  - **Spatial methods win** on this protocol; ours edges OCR by **+1.1 pp** mean (better ZH recovery 78.2% vs 74.7%; no ~10% sticker misses; no external OCR).
  - **Defense-Prefix** beats our EN number (73.8% > 71.6%) with better Clean Δ, but leaves ASR **16.4%** (ours ~2.6%) and does not treat ZH — complementary / EN-only story, not a drop-in for EN∩L mean 74.9%.
  - **Head / circuit ablations** recover only ~12–20% EN — useful negatives (“attention-head surgery ≠ sticker defense”).
- Vanilla multi attack floor (same sample): EN ~4.5%, ZH ~6.4%.

### 3.6 Grid-search baseline — confidence-drop **and** exhaustive search
- **Scoring axis (n=1000, greedy 2-patch, E+L EN+ZH; frozen coords unchanged winner):**
  - Old **max-confidence** scoring: ~**10–12%** mean — barely above no defense.
  - **Confidence-drop** scoring: ~**48.5%** mean @ cost 62 — proves scoring mattered; still far behind Attn-last / `cc_bbox_blur` (~73–75%) and ~10× cost.
- **Search axis (tune n=100, old max-conf scoring):**
  - Greedy 2-patch: ~**11.0%** mean.
  - **Exhaustive** C(16,2)=120 pairs: ~**11.5%** mean (~240 passes, ~6× slower).
  - Conclusion: search completeness is **not** the failure mode — a coarse 4×4 grid cannot reliably isolate stickers.
- Hit-pattern analysis (E+L): covering the **English** box matters much more than covering the partner-language box alone; hit-both best (~68% conditional).
- Role in paper: model-agnostic sanity check / upper bound on naive search, not a competing production method.

![grid n=1000 bars](../lib/notebooks/_en_zh/en_zh_multi_uni_attack/_test_grid/results/n1000_bars.png)

*Figure: conf-drop vs max-conf grid variants (n=1000). Pair with exhaustive-grid table from `_test_exhaustive_grid/results/comparison.json`.*

### 3.7 Heatmap refinements — path to `cc_bbox_blur`
- Ablation table (EN∩ZH E+L): baseline Attn-last → blur_fill → cc_bbox → **cc_bbox_blur**.
- Headline numbers: **`cc_bbox_blur` ~74.9% mean, clean Δ ~−1.5 pp, cost 4** (vs Attn-last 72.6% / worse clean Δ).
- Negative ablations in one compact table or appendix: gating-by-peakiness, union, ViT-B/16, hybrid — “we tried X; it failed because Y.”
- Residual gap: still ~10–15 pp below clean accuracy — leave as open problem (gating removes clean cost but does not close the residual attack gap).

![cc_bbox_blur ablation bars](../lib/notebooks/heatmap_defense_improvements/cc_bbox_blur/results/final_comparison.png)

![heatmap improvements overview](../lib/notebooks/heatmap_defense_improvements/results/final_comparison.png)

### 3.8 Gated defense — attack detector results (`multi`)
- Attn-last heatmap features separate clean vs attacked almost perfectly (test AUC ≈ 1.0 for ZH/JA; 0.999 for KO).
- Phase C roll-up (attack-recall target **0.99**):

| L | Always atk | Gated atk | Δ atk | Always Clean Δ | Gated Clean Δ |
|---|---:|---:|---:|---:|---:|
| zh | 74.0% | **73.9%** | −0.10 | −1.45 | **0.00** |
| ko | 69.9% | **69.45%** | −0.45 | −11.25 | **−0.20** |
| ja | 75.9% | **75.7%** | −0.20 | −11.50 | **0.00** |

- Defend coverage (full n=1000): attacked 99.4–99.8%; clean false-blur 0.3–2.5% (KO highest).
- Biggest win: KO/JA Clean Δ **≈ −11 pp → ~0** with <0.5 pp attacked-acc sacrifice.
- Caveat for paper: evaluated on `multi` + frozen sticker geometry; Pure E / Pure L and adaptive placement still open.

![feature PCA / t-SNE](../lib/notebooks/attack_detector/results/zh/multi/pca_features.png)

*Figure idea: Phase A separation + gated vs always Clean Δ bars across ZH/KO/JA.*

### 3.9 KO/JA clean-damage mitigation (always-on ablations)
- Show baseline vs thr_floor_095 / tight_dilate / no_bbox.
- Main story: thr=0.90 (or lower) on Pure E / multi was self-inflicted overshoot; flooring at **0.95** recovers large Clean Δ without losing defended acc (sometimes gains).
- Geometry tweaks shave residual always-on clean damage but do not reach ZH’s −1.5 pp — remaining gap is **heatmap quality**; detector gate is the practical fix (§3.8).
- Interpretation: always-on residual gap is EN∩KO / EN∩JA heatmap quality, not just threshold choice.

![KO/JA clean-damage ablation](../lib/notebooks/ko_ja_clean_damage/results/final_comparison.png)

### 3.10 Summary table for the reader (results closer)
- One “leaderboard” table spanning: always-on `cc_bbox_blur` (per L × attack), gated variant (`multi`), OCR+blur, Defense-Prefix, Dyslexify, SamplingTAR, conf-drop grid.
- Highlight recommended config: Attn-last + `cc_bbox_blur`, thr ≥ 0.95; **gate with Attn-last detector when Clean Δ matters** (especially KO/JA); cite OCR/DP as closest published peers.

---

## 4. Conclusion

### 4.1 What we showed
- Cross-lingual attention intersection (EN ∩ L for ZH/KO/JA) is a practical **spatial** defense for typographic attacks on separate CLIPs — not only a disagreement alarm.
- Last-layer attention outperforms GradCAM in this dual-box setting on accuracy and cost.
- Simple mask post-processing (bbox snap + blur) improves the accuracy / clean-image tradeoff without extra model passes; Clean Δ ≈ −1.5 pp for always-on EN∩ZH.
- The recipe transfers to Chinese, Korean, and Japanese partners on Pure E / E+L recovery under a **shared frozen evaluation set**.
- A cheap **heatmap-shape detector** gates the blur so Clean Δ ≈ 0 for all three partners on `multi`, solving the KO/JA always-on clean-damage problem.
- Against published peers on the same protocol: `cc_bbox_blur` leads mean accuracy; OCR+blur is close; CIFAR-trained Defense-Prefix is strong EN-only with high residual ASR; head-ablation baselines fail; grid search fails even with conf-drop / exhaustive search.

### 4.2 Limitations (must-include honesty)
- Evaluated on CIFAR-10 dual-box stickers with **frozen** placement — not ImageNet-scale scenes, not adaptive attackers that place text to evade attention or the detector.
- Detector results reported for **E + L (`multi`)**; Pure E / Pure L gating still open.
- Pure L (especially ZH-only) stickers remain harder for the EN half of the intersection.
- Always-on KO/JA Clean Δ still worse than ZH without the gate.
- Residual gap to clean accuracy (~10–15 pp on EN/ZH attacked) unsolved — gating removes clean cost, not the residual attack gap.
- Defense-Prefix comparison is EN-only (no ZH prefix); Dyslexify / SamplingTAR are style ports (open_clip ViT-B/32 openai), not identical paper checkpoints.
- Grid / hybrid search not competitive enough to recommend despite scoring and exhaustive-search fixes.

### 4.3 Broader implications
- Separate encoders + spatial agreement may complement prediction-disagreement detectors.
- Soft occlusion (blur) is preferable to hard mean-fill when preserving object evidence matters; **selective** blur (detect → repair) is preferable to always-on when partner heatmaps are noisy.
- English text remains the dominant transfer threat across models — defenses should prioritize localizing Latin-script stickers.
- Spatial localization beats pure mechanistic head ablation on dual-box typographic attacks in this setting.

### 4.4 Next steps / future work
- Write full paper prose + paper-ready figures from existing notebooks.
- Close residual gap to clean under attack (better saliency, adaptive thresholds, or stronger gates).
- Extend detector evaluation to Pure E / Pure L and adaptive sticker placement.
- Stronger KO/JA backbones or better EN∩L heatmap fusion for always-on use.
- Test on higher-res datasets / more realistic text placements.
- Optional: ZH/multilingual Defense-Prefix for a fairer prompt-baseline comparison.
- Optional contrast paragraph with Thread A (shared encoder) if the venue wants multilingual defense narrative.

### 4.5 Closing sentence (idea)
- Multilingual CLIP defenses need not stop at “do the languages agree?” — asking **where** they agree to look, blurring that region when heatmap shape says “attack,” recovers most accuracy under typographic attack at low compute cost with near-zero clean-image damage.

---

## Appendix ideas (optional, not required for first draft)

- Full ablation tables and failed ideas (peakiness gate, free-tune vs thr-floor under freeze).
- Model cards / Hugging Face IDs and prompt templates.
- Example defended images (attacked → heatmap → mask → blur → prediction) — see `four_lang_cc_bbox_blur/results/pipeline_*.png` (ZH/KO/JA).
- Conf-drop grid hit-pattern contingency table; exhaustive-grid comparison JSON.
- Baseline smoke-ladder tables (n=16 / n=100 / n=1000) from `paper_baselines/*/results/`.
- Detector Phase A PCA/t-SNE + feature-importance plots per partner.
- Notebook index for reproducibility (`PROTOCOL.md` + folder README).

![older GradCAM mask examples (lineage)](../lib/notebooks/_en_zh/en_zh_multi_uni_attack/multilingual/results/cam_2mod/mask_examples.png)

---

## Writing checklist (when expanding outline → prose)

- [ ] Lock final title and abstract (150–200 words from §1.4–1.5 + headline numbers + baseline win).
- [ ] Related Work subsection (CLIP typographic attacks; GradCAM / attention rollout; multilingual / ensemble defenses; OCR defenses; Defense-Prefix; Dyslexify / SamplingTAR; occlusion-based defenses).
- [ ] Insert exact tables from `docs/research_diary.md` (2026-07-16 → 2026-07-23), `docs/baseline_comparison.md`, and notebook `results/*.json`.
- [ ] Decide whether Thread A appears only in Related Work / Discussion or is omitted.
- [x] Figures linked in outline (method diagram / pipeline; main bar charts; 4-lang transfer; qualitative examples) — replace/replot at paper DPI later if needed.
- [x] Four languages (EN/ZH/KO/JA) front-loaded in Intro + Methods.
- [x] Attack types named Pure E / E + L / Pure L; **why English** justified early in **§1.2.1** (Methods keeps a short pointer).
- [x] Dataset source + balanced-sample construction + frozen `attack_pos` + thr ≥ 0.95 specified.
- [x] Grid baseline covers conf-drop **and** exhaustive search.
- [x] Published baselines (OCR, Defense-Prefix, Dyslexify, SamplingTAR) on leaderboard.
- [x] Attack-detector gated Clean Δ results for ZH/KO/JA `multi`.
- [x] Clean Δ elevated as a lead Results claim (always-on + gated).
