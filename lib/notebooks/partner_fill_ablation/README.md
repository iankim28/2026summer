# Partner fill ablation (ZH / KO / JA)

Follow EN’s gated fill ranking from [`../en_neglect_vs_blur/`](../en_neglect_vs_blur/): after the same Attn-last EN∩L `cc_bbox` masks + Phase-C detector gate, compare **neglect / blur / mean / black**.

- **neglect** (“ignore”) = zero ViT patch tokens with ≥50% mask coverage (CLS kept); grid is backbone-specific (B/32 → 7×7, B/16 → 14×14).
- **blur / mean / black** = image-space fills inside the mask (blur radius 12).

Reports per partner: L atk / L clean / L MIXED2000 and bilingual EN+L MIXED2000.

## Run (CUDA)

```bash
cd lib/notebooks/partner_fill_ablation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python run_eval.py --n 16 --partner zh --status sanity
python run_eval.py --n 1000 --partner all --status final
```

Results: `results/{zh,ko,ja}/gated_n1000.json`, `results/leaderboard.json`.

Uses Phase-C caches/gates under [`../attack_detector/results/{L}/multi/`](../attack_detector/results/) — no detector retrain.
