# 4-lang `cc_bbox_blur` initial trial

Does the EN/ZH winner **`cc_bbox_blur`** transfer to KO and JA?

For each partner `L ∈ {zh, ko, ja}`:

| Attack | Boxes | Score |
|--------|-------|-------|
| `uni_en` | EN + EN | EN + L |
| `uni_l` | L + L | EN + L |
| `multi` | EN + L | EN + L |

Defense: **EN ∩ L** Attn-last → CC top-2 + bbox snap → Gaussian blur fill.
Tune threshold on n=100 → always full n=1000. Geometry: `FONT_SIZE=24`, `NUM_BOXES=2`.
