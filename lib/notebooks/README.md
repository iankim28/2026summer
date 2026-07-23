# lib/notebooks

Topic folders, each with its own `results/`. Shared samples live in
[`image_samples/`](image_samples/). Underscore-prefixed buckets sort first:
[`_archive/`](_archive/) (superseded work) and [`_en_zh/`](_en_zh/) (early
EN/ZH attack + GradCAM lineage). Current defense mainline stays at the top
level without a prefix.

"Last used" is the most recent file-modification date inside the folder
(not just the last git commit).

## Layout

```
lib/notebooks/
  _archive/                      # superseded / paused experiments
  _en_zh/                        # early EN/ZH attack + GradCAM lineage
  _test_grid/                    # improved conf-drop grid search (promoted from _en_zh)
  attention_defense/             # current mainline (frozen Attn-last baseline)
  heatmap_defense_improvements/  # cc_bbox_blur winner
  four_lang_cc_bbox_blur/        # EN∩L transfer to ZH/KO/JA
  ko_ja_clean_damage/            # KO/JA clean-Δ ablations
  attack_detector/               # gate cc_bbox_blur via heatmap pattern detector
  image_samples/                 # shared CIFAR-10 index JSONs + frozen attack_pos
```

## Current mainline

| Folder | Last used | Research question | Main outputs |
|--------|-----------|-------------------|--------------|
| [`attention_defense/`](attention_defense/) | 2026-07-16 | Can ViT self-attention weights replace GradCAM as the saliency signal for the CAM-intersection defense, at lower inference cost? Spun out of [`_en_zh/en_zh_multi_uni_attack/`](_en_zh/en_zh_multi_uni_attack/). **Attn-last wins on every axis**: 72.6% mean acc vs GradCAM's 33.1%, at 2/3 the cost and with far less clean-image degradation. Follow-up: [`unilingual/`](attention_defense/unilingual/). See [`README.md`](attention_defense/README.md). | `results/final_comparison.png`, `results/heatmap_comparison.png`, `results/threshold_sweep_comparison.png`, `results/confusion_results_{gradcam,attn_last,attn_rollout}.json` |
| [`heatmap_defense_improvements/`](heatmap_defense_improvements/) | 2026-07-20 | **Done for now.** Ablations to close the Attn-last gap; keep [`cc_bbox_blur/`](heatmap_defense_improvements/cc_bbox_blur/) (**74.9%** mean, clean Δ −1.5pp). See [`README.md`](heatmap_defense_improvements/README.md). | per-subfolder `results/` |
| [`four_lang_cc_bbox_blur/`](four_lang_cc_bbox_blur/) | 2026-07-20 | Initial trial: does EN/ZH winner **`cc_bbox_blur`** transfer to KO/JA? For each partner L∈{zh,ko,ja}: uni-EN / uni-L / multi EN+L dual-box attacks; defend with EN∩L Attn-last → CC+bbox+blur. Also holds qualitative pipeline figs (`results/pipeline_*.png`). See [`README.md`](four_lang_cc_bbox_blur/README.md). | `results/{L}/{attack}/`, `comparison_summary.json`, `pipeline_*.png` |
| [`ko_ja_clean_damage/`](ko_ja_clean_damage/) | 2026-07-19 | Ablation to cut KO/JA Clean Δ under EN∩L `cc_bbox_blur` (ZH skipped). Variants: baseline thr tune, thr floor 0.95, pareto clean-aware tune, tight dilate, no bbox. See [`README.md`](ko_ja_clean_damage/README.md). | `results/{L}/{attack}/{variant}.json`, `comparison_summary.json`, `winners.json` |
| [`attack_detector/`](attack_detector/) | 2026-07-23 | Learn Attn-last heatmap features to detect typographic attack vs clean; gate `cc_bbox_blur`. Step 1 EN∩ZH + Step 2 EN∩KO/JA (`multi`). See [`README.md`](attack_detector/README.md). | `results/{L}/multi/`, `comparison_summary.json` |
| [`_test_grid/`](_test_grid/) | 2026-07-20 | Improved conf-drop grid occlusion (promoted from `_en_zh/…/_test_grid/`). Frozen `attack_pos` protocol. Winner still `C_2p_confdrop_blur` (~48.5% mean @ cost 62). | `results/comparison_n1000.json`, `results/protocol_before_after.json` |
| [`image_samples/`](image_samples/) | 2026-07-20 | Fixed image subsets + frozen dual-box `attack_pos` shared across experiments. | `CIFAR10_BALANCED_1000_SAMPLE.json` (`attack_pos`), `attack_placement.py`, `CIFAR10_4LANG_1000_SAMPLE.json` |

## Early EN/ZH lineage — [`_en_zh/`](_en_zh/)

| Folder | Last used | Research question | Main outputs |
|--------|-----------|-------------------|--------------|
| [`_en_zh/en_zh_multi_uni_attack/`](_en_zh/en_zh_multi_uni_attack/) | 2026-07-16 | Does attacking with **both** EN+ZH text (multilingual) beat a repeated EN-only attack (unilingual), and which defense is worth its inference cost? Shared geometry/defense conventions: [`PROTOCOL.md`](PROTOCOL.md). Attention-vs-GradCAM saliency test promoted to [`attention_defense/`](attention_defense/). Improved grid search promoted to [`_test_grid/`](_test_grid/). | `cost_vs_performance.png`; per-setup `results/…` |
| [`_en_zh/en_zh_typographic/`](_en_zh/en_zh_typographic/) | 2026-07-13 | How does a typographic attack on EN text affect a Chinese CLIP model, and vice-versa? | `results/…`, `results/balanced/` |
| [`_en_zh/en_zh_multiple_placement/`](_en_zh/en_zh_multiple_placement/) | 2026-07-13 | EN/ZH typographic attacks at **random positions** (dual box) on balanced CIFAR-10 + CAM defense. | `results/…`, `results/cam_defense/` |
| [`_en_zh/cam_intersection_defense/`](_en_zh/cam_intersection_defense/) | 2026-07-10 | Original EN∩ZH GradCAM intersection masking defense (single-box); later reused in multiple-placement and multi/uni. | `results/confusion_results_cam_defense.json`, `results/threshold_sweep.png` |

## Archive — [`_archive/`](_archive/)

| Folder | Last used | Notes |
|--------|-----------|-------|
| [`_archive/old_cifar10_typographic_4lang/`](_archive/old_cifar10_typographic_4lang/) | 2026-07-09 | 4-lang typographic attack on CIFAR-10. KO/JA: `Bingsu/clip-vit-base-patch32-ko`, `llm-jp/llm-jp-clip-vit-base-patch16` — upgrade candidates in [`old_ko_ja_model_screening/`](_archive/old_ko_ja_model_screening/). |
| [`_archive/old_ko_ja_model_screening/`](_archive/old_ko_ja_model_screening/) | 2026-07-10 | KO/JA CLIP clean-acc screen. Recommended upgrades: `Bingsu/clip-vit-large-patch14-ko`, `llm-jp/llm-jp-clip-vit-large-patch14`. |
| [`_archive/old_stl10_typographic_4lang/`](_archive/old_stl10_typographic_4lang/) | 2026-07-06 | STL-10 4-lang attack; superseded by `old_cifar10_typographic_4lang/`. |
| [`_archive/old_pgd_dual_encoder/`](_archive/old_pgd_dual_encoder/) | 2026-06-19 | Per-language LoRA + divergence under PGD; paused. |
| [`_archive/old_multilingual_consensus/`](_archive/old_multilingual_consensus/) | 2026-06-19 | Original consensus/PGD colab (Experiments A–G). |
| [`_archive/old_tutorials/`](_archive/old_tutorials/) | 2026-06-19 | Intro CLIP + multilingual adversarial defence tutorials. |

## Related docs

- [`_archive/old_ko_ja_model_screening/README.md`](_archive/old_ko_ja_model_screening/README.md) — KO/JA model candidates and screening results.
- `docs/research_diary.md` — chronological narrative of every experiment above.
- [`PROTOCOL.md`](PROTOCOL.md) — shared reproducible spec: frozen `attack_pos`, Option B attacks, Attn-last / `cc_bbox_blur`, 4-lang + KO/JA stack.
