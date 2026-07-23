# Attack Detector

**Research question:** Can Attn-last heatmap *shape* (spiky vs spread) separate typographic-attacked images from clean ones — and does gating `cc_bbox_blur` on that detector cut Clean Δ without giving up attacked accuracy?

**Motivation:** Today `four_lang_cc_bbox_blur` **always** blurs. Earlier `gated_peakiness` never fired. This notebook learns a richer detector.

**Scope:** EN ∩ L for `L ∈ {zh, ko, ja}`, `multi` dual-box attack, frozen `attack_pos` (PROTOCOL).

| Step | Partners | Output |
|------|----------|--------|
| 1 | EN ∩ ZH | `results/zh/multi/` |
| 2 | EN ∩ KO, EN ∩ JA | `results/ko/multi/`, `results/ja/multi/` |

**Phases (per partner):** A Look (PCA/t-SNE) → B Learn (logistic/SVM) → C Use (gated defense).

Protocol: [`../PROTOCOL.md`](../PROTOCOL.md).
