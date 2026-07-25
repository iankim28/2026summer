# Handoff — Report always vs gated on ALL 2000 images (clean + attacked)

**Repo:** `d:\ian\2026summer`  
**Task:** Produce a single accuracy number per policy that pools **1000 clean + 1000 attacked** (= 2000), for partners `zh`, `ko`, `ja`, attack=`multi`.  
**Why:** Current tables split metrics (attacked-only acc vs Clean Δ). User wants the joint score where gated’s clean advantage is visible.

---

## 0. Read this first (do not retrain / re-bake unless necessary)

Phase C **already ran** both policies on both halves. Existing JSONs contain:

| Field | Meaning |
|-------|---------|
| `policies.*.attacked_acc` | Acc on 1000 **attacked** images after that policy |
| `policies.*.clean_acc_masked` | Acc on 1000 **clean** images after that policy’s masking |
| `policies.*.clean_acc` | Vanilla clean (no defense) |
| `policies.*.clean_delta` | `clean_acc_masked - clean_acc` |

**Preferred path (no GPU):** run the post-process script below on those JSONs.  
**Only re-run** the full detector pipeline if JSONs are missing/corrupt or user explicitly wants a fresh CUDA recompute.

---

## 1. Environment / rules

- Workspace path: `d:\ian\2026summer`
- **CUDA required** for any ML re-run (RTX 5070 Ti). Check first:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- Abort if CUDA unavailable — do not fall back to CPU for long runs.
- Protocol: [`lib/notebooks/PROTOCOL.md`](../lib/notebooks/PROTOCOL.md)
- Detector code: [`lib/notebooks/attack_detector/`](../lib/notebooks/attack_detector/)
- Sample: `lib/notebooks/image_samples/CIFAR10_BALANCED_1000_SAMPLE.json` (frozen `attack_pos`)
- Do **not** change detector thresholds / retrain unless asked. This task is reporting.

---

## 2. Metric definition (exact)

For each partner `L ∈ {zh, ko, ja}` and each policy ∈ `{never_defend, always_defend, gated}`:

```
atk_mean   = mean(EN_acc, L_acc) on 1000 multi-attacked images under policy
clean_mean = mean(EN_acc, L_acc) on 1000 clean images under policy
             (use clean_acc_masked for always/gated; clean_acc for never)

mixed_2000 = 0.5 * atk_mean + 0.5 * clean_mean
```

Equal weight on the two halves (same 1000 CIFAR indices, once clean + once attacked).

Also report separately (already in JSON):

- `atk_mean`, `clean_mean`, `clean_delta_mean`
- `defend_frac_attacked`, `defend_frac_clean`

**Expected qualitative result:** on `mixed_2000`, **gated > always**, especially for KO/JA (always pays ~−11pp on the clean half).

---

## 3. Fast path — compute from existing results (do this)

Script already in repo:

```bash
cd d:\ian\2026summer\lib\notebooks\attack_detector
python compute_mixed_2000.py
```

Writes: `results/mixed_2000_summary.json`  
Prints a table for zh/ko/ja.

### Inputs

```
results/zh/multi/gated_comparison.json
results/ko/multi/gated_comparison.json
results/ja/multi/gated_comparison.json
```

### Sanity targets (from current JSONs; regenerate if you change code)

Approximate `MIXED2000` mean acc:

| L | never | always | gated | gated − always |
|---|------:|-------:|------:|---------------:|
| zh | 47.05% | 80.60% | 81.28% | **+0.68 pp** |
| ko | 47.82% | 73.20% | 78.50% | **+5.30 pp** |
| ja | 46.98% | 76.80% | 82.45% | **+5.65 pp** |

(Exact printed values from the script win over this table.)

---

## 4. Slow path — only if user demands a full re-run

Working dir: `lib/notebooks/attack_detector/`

1. Confirm CUDA.
2. In `_cells/02_imports.py`, set `SKIP_EXISTING = False` if you must overwrite existing `gated_comparison.json` (default is `True` and will skip finished partners).
3. Rebuild/run:

```bash
python _build_notebook.py
python run_all.py
```

4. Re-run `python compute_mixed_2000.py`.
5. Restore `SKIP_EXISTING = True` when done if you changed it.

Notes:

- Full bake + detect + gate is GPU-heavy (Attn-last for clean+attack × 3 partners).
- Attack = `multi` only; thr floor `0.95`; recall target `0.99`.
- Do not alter frozen `attack_pos`.

---

## 5. Optional: add mixed metric into the pipeline permanently

If asked to persist in Phase C (not required for the fast path):

- Edit `_cells/16_viz.py` where `summary` is built (~lines 260–288).
- Add:

```python
always_clean_pol = _mean_acc(policies['always_defend']['clean_acc_masked'])
gated_clean_pol  = _mean_acc(policies['gated']['clean_acc_masked'])
never_clean_pol  = _mean_acc(clean_acc)
summary['always_mixed_2000_acc'] = 0.5 * always_mean_acc + 0.5 * always_clean_pol
summary['gated_mixed_2000_acc']  = 0.5 * gated_mean_acc  + 0.5 * gated_clean_pol
summary['never_mixed_2000_acc']  = 0.5 * _mean_acc(base_acc) + 0.5 * never_clean_pol
```

- Mirror fields into `_cells/18_detector.py` roll-up / `comparison_summary.json`.
- Re-run with `SKIP_EXISTING=False` or just keep using `compute_mixed_2000.py`.

---

## 6. Deliverables for the user

1. Printed table: per L × policy → `atk_mean`, `clean_policy_mean`, `mixed_2000`, Clean Δ, defend fracs.  
2. File: `lib/notebooks/attack_detector/results/mixed_2000_summary.json`  
3. One-sentence takeaway: on the pooled 2000, gated beats always (KO/JA by several pp) because always’s clean half is damaged.

---

## 7. What NOT to do

- Do not average only attacked acc and call it “2000.”
- Do not use the detector’s train/val/test **feature** split as the accuracy denominator — eval is the full 1000+1000 image pool under each defense policy.
- Do not compare against paper baselines here; this is detector Phase C only.
- Do not commit unless the user asks.

---

## 8. Context links

- Briefing: [`docs/homework_summary.md`](homework_summary.md) §4 (gated detector)  
- Diary: `docs/research_diary.md` — entry `2026-07-22 / 23 — Attack detector`  
- Existing roll-up: `lib/notebooks/attack_detector/results/comparison_summary.json`
