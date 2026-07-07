# lib/notebooks

Organised into topic folders; each contains its own `results/` directory.
Shared cross-experiment artifacts live in `shared/`.

| Folder | Research question | Main outputs |
|--------|-------------------|--------------|
| [`en_zh_typographic/`](en_zh_typographic/) | How does a typographic attack on EN text affect a Chinese CLIP model, and vice-versa? [`typographic_comparison.ipynb`](en_zh_typographic/typographic_comparison.ipynb) | `results/accuracy_matrix.png`, `results/confusion_results.json`, `results/gradcam_heatmaps.png`, `results/sample_viz.png` |
| [`cifar10_typographic_4lang/`](cifar10_typographic_4lang/) | 4-language (EN/ZH/KO/JA) typographic attack on CIFAR-10 with OpenAI CLIP. [`typographic_attack_confusion.ipynb`](cifar10_typographic_4lang/typographic_attack_confusion.ipynb) | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png`, `results/perclass_attack_{en,zh,ko,ja}.png` |
| [`stl10_typographic_4lang/`](stl10_typographic_4lang/) | Same 4-language attack experiment replicated on STL-10. [`typographic_attack_confusion.ipynb`](stl10_typographic_4lang/typographic_attack_confusion.ipynb) | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png` |
| [`shared/`](shared/) | Cross-experiment artifacts (fixed random subsets). | `cifar10_1000_indices.json` — seed-0 1000-image CIFAR-10 test subset used by EN/ZH and CIFAR-10 4-lang experiments |
| [`tutorials/`](tutorials/) | Introductory notebooks for CLIP and multilingual adversarial defence. | — |
| [`dual_encoder/`](dual_encoder/) | Divergence analysis between dual-encoder architectures. | — |
| [`multilingual_consensus/`](multilingual_consensus/) | Multilingual consensus colab experiments. | — |
