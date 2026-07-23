# Paper baselines

Four published baselines vs `cc_bbox_blur` (PROTOCOL dual-box CIFAR).

| Folder | Method | Entry |
|--------|--------|-------|
| `defense_prefix/` | Defense-Prefix | `run_eval.py` (+ `train_cifar_dp.py` if pretrained fails) |
| `ocr_blur/` | OCR + blur | `run_eval.py` |
| `dyslexify/` | Dyslexify-style head ablation | `run_eval.py` |
| `sampling_tar/` | SamplingTAR-style circuit ablation | `run_eval.py` |

Living numbers: [`docs/baseline_comparison.md`](../../../docs/baseline_comparison.md)

Vendor checkouts (read-only reference): `_vendor/`.

## Smoke ladder (each method)

```bash
python run_eval.py --n 16 --status sanity
python run_eval.py --n 100 --status smoke
python run_eval.py --n 1000 --status final
```

CUDA required.
