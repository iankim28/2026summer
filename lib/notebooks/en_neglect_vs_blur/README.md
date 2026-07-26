# EN neglect vs blur

Chase **EN MIXED2000 > 81.65%** (Defense-Prefix bar) by changing how sticker regions are occluded.

## Blur (current production)

`PIL.ImageFilter.GaussianBlur(radius=12)` inside the `cc_bbox` mask (Attn-last EN∩ZH → thr≥0.95 → dilate → top-2 CC → bbox snap). Soft occlude: glyphs smash, model still sees smeared RGB.

## Neglect (this experiment)

Zero ViT-B/32 **patch tokens** after pos-embed for patches with ≥50% mask coverage (CLS kept). Controls: `mean` fill, solid `black`.

## Run (CUDA)

```bash
cd lib/notebooks/en_neglect_vs_blur
python run_eval.py --n 16 --stage oracle --status sanity
python run_eval.py --n 1000 --stage all --status final
```

Results under `results/`. Winner on full pool: **DP + patch-zero** (spatial zero of neglected patches, then CIFAR-retrained Defense-Prefix text) → EN MIXED2000 **86.65%**.
