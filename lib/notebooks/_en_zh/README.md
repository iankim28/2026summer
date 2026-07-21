# _en_zh/

Early EN/ZH typographic-attack and GradCAM-defense lineage. These studies
established the dual-box attack protocol and the CAM-intersection defense
that later work (`attention_defense/`, `heatmap_defense_improvements/`)
replaced with Attn-last + `cc_bbox_blur`.

Underscore prefix keeps this bucket next to [`_archive/`](../_archive/) at
the top of `lib/notebooks/`.

| Folder | Role |
|--------|------|
| [`en_zh_typographic/`](en_zh_typographic/) | EN↔ZH single-box typographic transfer |
| [`en_zh_multiple_placement/`](en_zh_multiple_placement/) | Dual random-box attack + CAM defense |
| [`cam_intersection_defense/`](cam_intersection_defense/) | Original single-box GradCAM ∩ mask defense |
| [`en_zh_multi_uni_attack/`](en_zh_multi_uni_attack/) | Multi (EN+ZH) vs uni (EN+EN) attack + cost study; parent of attention defense |

See the parent [`../README.md`](../README.md) for the full index.
