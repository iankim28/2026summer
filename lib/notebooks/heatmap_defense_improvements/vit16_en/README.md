# vit16_en (idea 5)

**Question:** Does swapping English CLIP from ViT-B/32 (7×7 patches) to ViT-B/16
(14×14 patches) sharpen Attn-last localization enough to improve the 2-model
intersection defense?

Notebook: [`attention_vit16_en.ipynb`](attention_vit16_en.ipynb)

Compare against published Attn-last B/32 numbers in
[`../../attention_defense/results/confusion_results_attn_last.json`](../../attention_defense/results/confusion_results_attn_last.json).
