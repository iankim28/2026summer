# attn_confdrop_hybrid (idea 6)

**Question:** Can Attn-last shortlist a few 4×4 cells, then conf-drop scoring pick
which to blur — recovering accuracy on hard cases without full grid cost (62 passes)?

Notebook: [`attn_confdrop_hybrid.ipynb`](attn_confdrop_hybrid.ipynb)

Reuses:
- Attn-last saliency from the attention-defense protocol
- Conf-drop scoring from [`../../_en_zh/en_zh_multi_uni_attack/_test_grid/`](../../_en_zh/en_zh_multi_uni_attack/_test_grid/)
