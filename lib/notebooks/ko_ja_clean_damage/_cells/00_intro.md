# KO/JA clean-damage ablation under `cc_bbox_blur`

**Goal:** cut Clean Δ for KO and JA without giving up much defended accuracy.

The 4-lang trial showed ZH Clean Δ ≈ −1.5pp while KO/JA sit at −11 to −23pp.
ZH is left alone. This notebook only runs `L ∈ {ko, ja}`.

| Attack | Boxes | Score |
|--------|-------|-------|
| `uni_en` | EN + EN | EN + L |
| `uni_l` | L + L | EN + L |
| `multi` | EN + L | EN + L |

**Variants per cell** (Attn-last CAMs cached once, then cheap mask sweeps):

1. `baseline` — thr maximizes tune EN attacked acc; dilate=3, bbox_snap=True
2. `thr_floor_095` — thr fixed at 0.95
3. `pareto_tune` — maximize `en_atk_acc + 0.5 * mean_clean_delta` on tune n=100
4. `tight_dilate` — pareto thr, dilate=1, bbox_snap=True
5. `no_bbox` — pareto thr, dilate=3, bbox_snap=False

Tune n=100 → full n=1000. Geometry: `FONT_SIZE=24`, `NUM_BOXES=2`.
