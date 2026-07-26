# Paper baselines

Four published baselines vs `cc_bbox_blur` (PROTOCOL dual-box CIFAR).

| Folder | Method | Entry |
|--------|--------|-------|
| `defense_prefix/` | Defense-Prefix | EN: `run_eval.py` + `train_cifar_dp.py`; ZH: `run_eval_zh.py` + `train_cifar_dp_zh.py` |
| `ocr_blur/` | OCR + blur | `run_eval.py` |
| `dyslexify/` | Dyslexify-style head ablation (+ hybrid attn-blur) | `run_eval.py --mode heads|hybrid` |
| `sampling_tar/` | SamplingTAR-style circuit ablation (+ hybrid attn-blur) | `run_eval.py --mode heads|hybrid` |

Living numbers: [`docs/baseline_comparison.md`](../../../docs/baseline_comparison.md)

Vendor checkouts (read-only reference): `_vendor/`.

## Smoke ladder (each method)

```bash
python run_eval.py --n 16 --status sanity --mode hybrid
python run_eval.py --n 100 --status smoke --mode hybrid
python run_eval.py --n 1000 --status final --mode hybrid
# heads-only negative baselines:
python run_eval.py --n 1000 --status final --mode heads
```

ZH Defense-Prefix (ChineseCLIP token):

```bash
cd defense_prefix
python train_cifar_dp_zh.py --epochs 10 --max_n 20000
python run_eval_zh.py --n 16 --status sanity
python run_eval_zh.py --n 100 --status smoke
python run_eval_zh.py --n 1000 --status final
```

CUDA required.
