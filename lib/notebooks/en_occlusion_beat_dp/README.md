# EN occlusion-only chase of DP 81.65%

**Constraint:** no Defense-Prefix / learned prompts. Gated occlusion only.

```bash
python run_eval.py --n 16 --status sanity
python run_eval.py --n 1000 --status final
```

Best on full pool: **gated OCR∪cc_bbox + black** → EN MIXED2000 **79.70%** (still below DP **81.65%**).
