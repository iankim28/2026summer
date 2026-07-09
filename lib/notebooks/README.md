# lib/notebooks

Organised into topic folders; each contains its own `results/` directory.
Shared cross-experiment artifacts live in `image_samples/`.

| Folder | Research question | Main outputs |
|--------|-------------------|--------------|
| [`en_zh_typographic/`](en_zh_typographic/) | How does a typographic attack on EN text affect a Chinese CLIP model, and vice-versa? [`typographic_comparison.ipynb`](en_zh_typographic/typographic_comparison.ipynb) | `results/accuracy_matrix.png`, `results/confusion_results.json`, `results/gradcam_heatmaps.png`, `results/sample_viz.png` |
| [`cifar10_typographic_4lang/`](cifar10_typographic_4lang/) | 4-language (EN/ZH/KO/JA) typographic attack on CIFAR-10 with OpenAI CLIP. [`typographic_attack_confusion.ipynb`](cifar10_typographic_4lang/typographic_attack_confusion.ipynb) | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png`, `results/perclass_attack_{en,zh,ko,ja}.png` |
| [`old_stl10_typographic_4lang/`](old_stl10_typographic_4lang/) | Same 4-language attack experiment replicated on STL-10. Archived. | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png` |
| [`image_samples/`](image_samples/) | Fixed image subsets shared across experiments. | `CIFAR10_4LANG_1000_SAMPLE.json` — seed-0 random 1000-image CIFAR-10 subset; `CIFAR10_BALANCED_1000_SAMPLE.json` — 100 images per class, seed-0 |
| [`old_tutorials/`](old_tutorials/) | Introductory notebooks for CLIP and multilingual adversarial defence. Archived. | — |
| [`pgd_dual_encoder/`](pgd_dual_encoder/) | Per-language LoRA adapters + text heads trained with CE + divergence + adversarial retention loss. First experiment to exceed 50% non-EN retention under PGD attack (ε≤4). Paused — next step is retraining at higher ε. | `divergence.ipynb` |
| [`multilingual_consensus/`](multilingual_consensus/) | Multilingual consensus colab experiments. | — |
