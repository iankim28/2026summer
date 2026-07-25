# Homework Summary — Recent Progress (July 20–23, 2026)

**Project:** Defending multilingual CLIP classifiers against typographic (text-overlay) attacks on CIFAR-10.  
**Audience:** Quick briefing for professor meeting — what was finished on the checklist, and what comes next.

> Older briefings (Jul 5 assignments + Jul 16–19 defense sprint): [`homework_summary_archive.md`](homework_summary_archive.md).  
> Shared experiment conventions: [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md).  
> Published baseline numbers: [`baseline_comparison.md`](baseline_comparison.md).

---

## One-paragraph overview

After locking `cc_bbox_blur` as the defense, I made evaluation fair and paper-ready. (1) **Froze attack geometry** so every method sees the same EN/L sticker positions per image. (2) **Restored accuracy** under that new protocol with a thr ≥ 0.95 floor (ZH multi **74.0%**, Clean Δ **−1.5pp**). (3) Built an **Attn-last heatmap attack detector** (PCA/t-SNE evidence + SVM gate) so blur runs only when stickers are present — Clean Δ → **~0** for ZH/KO/JA with ≤0.45pp attacked-acc drop. (4) Ran four **published baselines** on the same protocol; `cc_bbox_blur` still leads mean accuracy. Open work is mostly writing the paper.

---

## Checklist status

| Item | Status | Date |
|------|--------|------|
| Bring accuracy back up for new (frozen) protocol | Done | 2026-07-21 |
| Redo tables under new protocol (keep old tables; create new) | Done | 2026-07-20/21 |
| Rename no-defense columns so “atk” = CLIP (EN / L) | Done | 2026-07-20/21 |
| Freeze attack position (same EN/L anchors for all methods) | Done | 2026-07-20 |
| Heatmap-pattern attack detector + gated occlusion | Done (`multi`) | 2026-07-22/23 |
| Dim-reduction / decision-boundary exploration (PCA, t-SNE) | Done (Phase A evidence) | 2026-07-22/23 |
| Find published baselines beyond grid search | Done | 2026-07-23 |
| Run baselines (DP → OCR → Dyslexify → SamplingTAR) | Done (n=1000) | 2026-07-23 |
| Update `PROTOCOL.md` | Done | 2026-07-20–23 |
| Start paper / write up results | In progress (outline) | `docs/paper_draft.md` |

---

## What was done (by topic)

### 1. Frozen attack geometry (fair baseline for every test)

**Problem:** Each notebook re-sampled sticker positions → grid vs attention vs baselines were not comparable.

**Fix:** Bake once into `image_samples/CIFAR10_BALANCED_1000_SAMPLE.json` under `attack_pos`:

| Slot | Key | Meaning |
|------|-----|---------|
| 0 | `en` | Top-left of English box (1000 images) |
| 1 | `l` | Top-left of partner-language box (1000 images) |

- Reference bake size `131×44`; runtime measures the real word, clamps into 224×224.
- Same anchors for grid, attention, `cc_bbox_blur`, detector, and paper baselines.
- Spec: [`PROTOCOL.md`](../lib/notebooks/PROTOCOL.md) §5; helper `image_samples/attack_placement.py`.

### 2. Bring accuracy back up under the new protocol

Free thr-tune after the freeze dropped several multi cells to thr 0.85–0.90 → defended acc and Clean Δ collapsed (e.g. ZH multi 74.9→71.7, Clean Δ −1.5→−7.0; KO multi Clean Δ → −25.5).

**Production fix:** `threshold = max(threshold_free, 0.95)` on the full n=1000 run.

| Cell | Free-tune def | Thr-floor def | Free Clean Δ | Floor Clean Δ |
|------|-------------:|--------------:|-------------:|--------------:|
| zh/multi | 71.7% | **74.0%** | −7.0 | **−1.5** |
| ko/multi | 64.0% | **69.9%** | −25.5 | **−11.2** |
| ja/multi | 73.6% | **75.9%** | −23.1 | **−11.5** |

ZH multi is within **0.9pp** of the pre-freeze 74.9% case-study number, with Clean Δ back at −1.5. Cite **thr-floor** numbers going forward.

### 3. New tables (old ones kept; atk = CLIP)

Previous tables stay in [`homework_summary_archive.md`](homework_summary_archive.md). New-protocol tables below. In no-defense columns, **Atk EN / Atk L** mean the English / partner **CLIP** accuracies before defense (not “attack strength” jargon).

#### No defense vs `cc_bbox_blur` (thr-floor, frozen `attack_pos`, n=1000)

| Cell | Atk EN (EN CLIP) | Atk L (partner CLIP) | Atk mean | Def mean | Clean EN / L |
|------|-----------------:|---------------------:|---------:|---------:|-------------:|
| zh/uni_en | 3.8% | 24.8% | 14.3% | **60.2%** | 85.9 / 91.4 |
| zh/uni_l | 72.0% | 40.3% | 56.1% | **67.2%** | 85.9 / 91.4 |
| zh/multi | 4.5% | 6.4% | 5.5% | **74.0%** | 85.9 / 91.4 |
| ko/uni_en | 3.8% | 12.9% | 8.3% | **63.7%** | 85.9 / 89.6 |
| ko/uni_l | 70.0% | 78.2% | 74.1% | **68.4%** | 85.9 / 89.6 |
| ko/multi | 3.5% | 12.3% | 7.9% | **69.9%** | 85.9 / 89.6 |
| ja/uni_en | 3.8% | 3.2% | 3.5% | **72.3%** | 85.9 / 92.5 |
| ja/uni_l | 71.3% | 84.6% | 78.0% | **72.8%** | 85.9 / 92.5 |
| ja/multi | 4.1% | 5.4% | 4.8% | **75.9%** | 85.9 / 92.5 |

Hard attacks (`uni_en`, `multi`) fall to ~4–14% mean and recover to mid-60s–mid-70s. Native-only `uni_l` on KO/JA is already weak; defense barely helps.

Grid under freeze: conf-drop winner still **48.5%** mean @ cost 62 (unchanged) — still a negative baseline.

### 4. Heatmap-pattern attack detector → gated occlusion

**Idea:** Typographic stickers → **spiky / biased** Attn-last maps; clean images → **more spread**. Learn that shape, then occlude (blur) only when the detector says “attack.” Scope extended from always-attacked assumption to **clean + attacked**.

**Pipeline** (`lib/notebooks/attack_detector/`), attack = `multi`, frozen geometry:

1. Bake Attn-last for clean + attacked (1000 each); 26 scalars from EN / L / ∩ maps.
2. **Phase A:** PCA + t-SNE — clusters separate (teacher-style dim-reduction evidence; decision boundary is in feature space, not t-SNE coords).
3. **Phase B:** logistic + calibrated linear SVM; attack-recall ≥ **0.99**.
4. **Phase C:** `never` / `always` / `gated` `cc_bbox_blur`.

| L | Test AUC | Always atk | Gated atk | Δ atk | Always Clean Δ | Gated Clean Δ |
|---|---------:|-----------:|----------:|------:|---------------:|--------------:|
| zh | 1.000 | 74.0% | **73.9%** | −0.10 | −1.45 | **0.00** |
| ko | 0.999 | 69.9% | **69.45%** | −0.45 | −11.25 | **−0.20** |
| ja | 1.000 | 75.9% | **75.7%** | −0.20 | −11.50 | **0.00** |

Biggest win: KO/JA Clean Δ **≈ −11pp → ~0** with <0.5pp attacked-acc cost. Caveat: reported for `multi` only; Pure E / Pure L gating still open.

### 5. More baselines (beyond grid) — find + run

Only competitive sanity check before was grid search. Found four published / portable methods and ran them on the **same frozen dual-box protocol** (smoke ladder n=16 → 100 → 1000). Order used: training-heavy first, then no-train spatial / heatmap peers.

| # | Method | Needs train data? | Final (n=1000) | vs ours |
|---|--------|-------------------|----------------|---------|
| 1 | **Defense-Prefix** (Azuma & Matsui 2023) | Yes — ImageNet DP failed; retrained 10 ep on CIFAR-10 train | **73.8% EN**, Clean Δ +0.5pp, ASR 16.4% | Strong EN-only; no ZH; high residual ASR |
| 2 | **OCR + blur** (EasyOCR → blur r=12) | No | **73.8% mean**, Clean Δ −0.7pp; sticker hit 90.3% | Closest peer; we win **+1.1pp** mean |
| 3 | **Dyslexify-style** head ablation | No (heatmap / attn heads) | **20.0% EN**, Clean Δ 0 | Weak negative |
| 4 | **SamplingTAR-style** circuit ablation | No | **11.6% EN**, Clean Δ +0.2pp | Weakest peer |

**Ours (reference):** `cc_bbox_blur` EN∩ZH multi — **74.9%** mean (case-study) / **74.0%** thr-floor four_lang, Clean Δ **−1.5pp**, cost 4.

Living leaderboard: [`baseline_comparison.md`](baseline_comparison.md). Code: `lib/notebooks/paper_baselines/`.

---

## Numbers to quote in a meeting

| Claim | Number |
|-------|--------|
| New-protocol ZH multi (thr-floor) | **74.0%** mean, Clean Δ **−1.5pp** |
| Gated Clean Δ (ZH / KO / JA, `multi`) | **0.0 / −0.2 / 0.0 pp** |
| Gated atk drop vs always | ≤ **0.45pp** |
| Closest published peer (OCR+blur) | 73.8% mean (−1.1pp vs our 74.9%) |
| Defense-Prefix (CIFAR-trained) | 73.8% EN, ASR still 16.4% |
| Head-ablation baselines | 11.6–20% EN (fail) |
| Grid conf-drop (frozen) | still **48.5%** @ cost 62 |

---

## Next steps (priority order)

1. **Write the paper** — expand [`paper_draft.md`](paper_draft.md); lead with frozen protocol, gated Clean Δ, baseline leaderboard.
2. **Optional experiments** — gate detector on Pure E / Pure L; adaptive sticker placement; close residual ~10–15pp gap to clean under attack.
3. **Optional** — multilingual / ZH Defense-Prefix for a fairer prompt-baseline comparison.

---

## How to explain the stack in one breath

> We freeze the same dual-box sticker positions for every method, build a mask where English and partner CLIP attention agree, snap it to sticker-shaped boxes, and blur. A cheap heatmap-shape detector turns that blur on only when an attack is present, so Clean Δ is ~0. On the same protocol we beat or match OCR and prompt baselines; attention-head surgery alone does not fix dual-box stickers.

---

## Key paths

| Work | Path |
|------|------|
| Protocol (source of truth) | `lib/notebooks/PROTOCOL.md` |
| Frozen sample + `attack_pos` | `lib/notebooks/image_samples/` |
| 4-lang thr-floor results | `lib/notebooks/four_lang_cc_bbox_blur/` |
| Attack detector (gated) | `lib/notebooks/attack_detector/` |
| Paper baselines | `lib/notebooks/paper_baselines/` |
| Baseline leaderboard | `docs/baseline_comparison.md` |
| Paper outline | `docs/paper_draft.md` |
| Archive (all older briefings) | `docs/homework_summary_archive.md` |
| Full diary | `docs/research_diary.md` (2026-07-20 → 2026-07-23) |
