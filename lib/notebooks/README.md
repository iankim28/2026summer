# lib/notebooks

Organised into topic folders; each contains its own `results/` directory.
Shared cross-experiment artifacts live in `image_samples/`.

| Folder | Research question | Main outputs |
|--------|-------------------|--------------|
| [`en_zh_typographic/`](en_zh_typographic/) | How does a typographic attack on EN text affect a Chinese CLIP model, and vice-versa? [`typographic_comparison.ipynb`](en_zh_typographic/typographic_comparison.ipynb), [`balanced_typographic_comparison.ipynb`](en_zh_typographic/balanced_typographic_comparison.ipynb) (100/class balanced sample, large font) | `results/accuracy_matrix.png`, `results/confusion_results.json`, `results/gradcam_heatmaps.png`, `results/sample_viz.png`; balanced run in `results/balanced/` |
| [`cam_intersection_defense/`](cam_intersection_defense/) | Can masking EN+ZH GradCAM intersection regions recover accuracy under typographic attack? [`cam_intersection_defense.ipynb`](cam_intersection_defense/cam_intersection_defense.ipynb) | `results/confusion_results_cam_defense.json`, `results/threshold_sweep.png`, `results/accuracy_delta_matrix.png`, `results/mask_examples.png` |
| [`cifar10_typographic_4lang/`](cifar10_typographic_4lang/) | 4-language (EN/ZH/KO/JA) typographic attack on CIFAR-10. [`typographic_attack_confusion.ipynb`](cifar10_typographic_4lang/typographic_attack_confusion.ipynb). Current KO/JA: `Bingsu/clip-vit-base-patch32-ko`, `llm-jp/llm-jp-clip-vit-base-patch16` — see [`ko_ja_model_screening/`](ko_ja_model_screening/) for upgrade candidates. | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png`, `results/perclass_attack_{en,zh,ko,ja}.png` |
| [`ko_ja_model_screening/`](ko_ja_model_screening/) | Screen KO/JA CLIP models for clean accuracy on balanced CIFAR-10 vs EN/ZH baselines. **Recommended upgrades:** `Bingsu/clip-vit-large-patch14-ko` (96.5%), `llm-jp/llm-jp-clip-vit-large-patch14` (97.0%). | `results/screening_results.json` |
| [`old_stl10_typographic_4lang/`](old_stl10_typographic_4lang/) | Same 4-language attack experiment replicated on STL-10. Archived. | `results/confusion_results.json`, `results/confusion_matrices.png`, `results/typographic_heatmaps.png` |
| [`image_samples/`](image_samples/) | Fixed image subsets shared across experiments. | `CIFAR10_4LANG_1000_SAMPLE.json` — seed-0 random 1000-image CIFAR-10 subset; `CIFAR10_BALANCED_1000_SAMPLE.json` — 100 images per class, seed-0 |
| [`old_tutorials/`](old_tutorials/) | Introductory notebooks for CLIP and multilingual adversarial defence. Archived. | — |
| [`pgd_dual_encoder/`](pgd_dual_encoder/) | Per-language LoRA adapters + text heads trained with CE + divergence + adversarial retention loss. First experiment to exceed 50% non-EN retention under PGD attack (ε≤4). Paused — next step is retraining at higher ε. | `divergence.ipynb` |
| [`multilingual_consensus/`](multilingual_consensus/) | Multilingual consensus colab experiments. | — |
