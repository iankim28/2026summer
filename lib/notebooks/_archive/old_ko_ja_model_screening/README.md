# KO / JA CLIP model screening

Benchmarks Korean and Japanese CLIP candidates against EN/ZH baselines on the **balanced 1000-image CIFAR-10 sample** (`CIFAR10_BALANCED_1000_SAMPLE.json`).

## Run

```bash
python screen_ko_ja_models.py
```

Results: `results/screening_results.json`

## Candidates

| Lang | Model | HF id |
|------|-------|-------|
| EN | OpenAI ViT-B/32 | `openai` |
| ZH | Chinese CLIP | `OFA-Sys/chinese-clip-vit-base-patch16` |
| KO | Bingsu ViT-B/32 | `Bingsu/clip-vit-base-patch32-ko` *(current)* |
| KO | Bingsu ViT-L/14 | `Bingsu/clip-vit-large-patch14-ko` |
| JA | llm-jp ViT-B/16 | `llm-jp/llm-jp-clip-vit-base-patch16` *(current)* |
| JA | llm-jp ViT-L/14 | `llm-jp/llm-jp-clip-vit-large-patch14` |
| JA | LY clip-japanese-base-v2 | `line-corporation/clip-japanese-base-v2` |
| JA | Stability JA ViT-L/16 | `stabilityai/japanese-stable-clip-vit-l-16` |

## Results (2026-07-10, balanced 1000-image sample)

| Model | Lang | Clean acc | vs ZH | Status |
|-------|------|-----------|-------|--------|
| OpenAI ViT-B/32 | EN | 85.9% | — | baseline |
| Chinese CLIP ViT-B/16 | ZH | 91.4% | — | baseline |
| Bingsu ViT-B/32 | KO | 89.7% | −1.7pp | **current** in 4-lang notebook |
| **Bingsu ViT-L/14** | KO | **96.5%** | **+5.1pp** | **recommended KO upgrade** |
| llm-jp ViT-B/16 | JA | 92.5% | +1.1pp | **current** in 4-lang notebook |
| **llm-jp ViT-L/14** | JA | **97.0%** | **+5.6pp** | **recommended JA upgrade** |
| LY clip-japanese-base-v2 | JA | — | — | load error (transformers 5.x + custom code) |
| Stability JA ViT-L/16 | JA | — | — | gated HuggingFace repo |

**Takeaway:** Both current KO/JA models are usable, but the **large** variants from the same families (Bingsu L/14, llm-jp L/14) clearly match or beat ZH/EN. No need to hunt exotic architectures — scale up within the same training recipe.
