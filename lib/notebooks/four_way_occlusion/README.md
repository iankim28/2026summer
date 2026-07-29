# Four-way occlusion (EN ∩ ZH ∩ KO ∩ JA)

True 4-language Attn-last intersection → `cc_bbox_black`.

**Protocol:** frozen dual-box CIFAR-10 n=1000, attack = EN+ZH `multi`, thr=0.95, dilate=3, top_k=2, fill=black, CUDA.

**Mask:** `n_cam_intersection(cam_en, cam_zh, cam_ko, cam_ja)` then cc_bbox shaping.

**Gate:** Phase-C feature recipe on EN Attn-last vs mean(ZH, KO, JA) maps (documented in results JSON). Masks remain true 4-way.

```bash
python run_eval.py --n 16     # smoke
python run_eval.py --n 1000   # final
```

Results: [`results/four_way_n1000.json`](results/four_way_n1000.json).
