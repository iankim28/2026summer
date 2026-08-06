# Paper baselines

Four published baselines vs `cc_bbox_blur` (PROTOCOL dual-box CIFAR).

| Folder | Method | Entry |
|--------|--------|-------|
| `defense_prefix/` | Defense-Prefix | EN / ZH / KO / JA: `run_eval*.py` + `train_cifar_dp*.py` |
| `ocr_blur/` | OCR + blur | `run_eval.py` / `run_eval_4lang.py` |
| `grid_occlusion/` | 4×4 grid `C_2p_confdrop_blur` | `run_eval.py` (EN+ZH search; score KO/JA; clean arm) |
| `dyslexify/` | Dyslexify-style head ablation (+ hybrid attn-blur) | `run_eval.py --lang en|ja`; KO: `run_eval_ko.py` |
| `sampling_tar/` | SamplingTAR-style circuit ablation (+ hybrid attn-blur) | `run_eval.py --lang en|ja`; KO: `run_eval_ko.py` |

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

ZH / KO / JA Defense-Prefix:

```bash
cd defense_prefix
python train_cifar_dp_zh.py --epochs 10 --max_n 20000
python train_cifar_dp_ko.py --epochs 10 --max_n 20000
python train_cifar_dp_ja.py --epochs 10 --max_n 20000 --lr 0.003   # EN+ZH stickers + star pool
python run_eval_zh.py --n 100 --status smoke
python run_eval_ko.py --n 100 --status smoke
python run_eval_ja.py --n 100 --status smoke
```

Grid + partner hybrids:

```bash
python grid_occlusion/run_eval.py --n 100 --status smoke
python sampling_tar/run_eval.py --n 100 --status smoke --mode hybrid --lang ja
python sampling_tar/run_eval_ko.py --n 100 --status smoke
python dyslexify/run_eval.py --n 100 --status smoke --mode hybrid --lang ja
python dyslexify/run_eval_ko.py --n 100 --status smoke
```

CUDA required.
