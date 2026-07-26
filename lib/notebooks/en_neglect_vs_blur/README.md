# EN neglect vs blur

Chase **EN MIXED2000 > 81.65%** (Defense-Prefix bar) by changing how sticker regions are occluded.

## Production fill (frozen): black

Solid **black** inside the `cc_bbox` mask (Attn-last EN∩L → thr≥0.95 → dilate → top-2 CC → bbox snap), applied under the Phase-C detector gate. Same fill for ZH/KO/JA — see [`../PROTOCOL.md`](../PROTOCOL.md) §7.2 and [`../partner_fill_ablation/`](../partner_fill_ablation/).

## Ablations

- **blur** — `GaussianBlur(radius=12)` (design history / soft occlude)
- **mean** — mean-color fill of unmasked RGB
- **neglect** — zero ViT-B/32 patch tokens after pos-embed (≥50% mask coverage; CLS kept)

Gated EN ranking: **black > mean > blur > neglect** (MIXED2000 **79.35%**).

## Run (CUDA)

```bash
cd lib/notebooks/en_neglect_vs_blur
python run_eval.py --n 16 --stage oracle --status sanity
python run_eval.py --n 1000 --stage all --status final
```

Results under `results/`. Winner on full pool: **DP + patch-zero** (spatial zero of neglected patches, then CIFAR-retrained Defense-Prefix text) → EN MIXED2000 **86.65%**.
