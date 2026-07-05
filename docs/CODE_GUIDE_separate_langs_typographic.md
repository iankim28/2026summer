# Code guide: separate per-language CLIPs + typographic attacks

How the code is organized for running **typographic attacks** against **independent
per-language CLIP models** (and comparing them to one shared multilingual CLIP). Every
snippet below matches the working, verified code in this folder.

---

## 1. What this does

1. **Wrap 4 independently-trained per-language CLIP models** behind one uniform interface
   (each has its *own* image encoder + text encoder).
2. **Render a misleading word onto an image** (a typographic attack — no gradients).
3. **Classify** the attacked image with each model and compare predictions.

The point: a typographic attack written in English fools some encoders and not others, so
**independent encoders disagree** (unlike a shared multilingual CLIP, where they agree on
the wrong answer).

---

## 2. File map

| file | role |
|---|---|
| `perlang_models.py` | the 4 per-language model wrappers + class names/templates + `classify()` |
| `typographic_attack.py` | render text on an image (`draw_word`) + the cross-lingual attack matrix |
| `shared_vs_separate.py` | driver: shared multilingual CLIP vs the 4 separate CLIPs under the attack |
| `results/*.json`, `results/*_strip.png` | saved numbers + visual font checks |

Environment: a venv with `torch`, `open_clip_torch`, `transformers` (4.44), `sentencepiece`,
`ftfy`, plus system fonts (`NotoSansCJK`, `DejaVuSans`). Everything is gradient-free and
runs in minutes on one GPU (or free Colab).

---

## 3. The per-language models (`perlang_models.py`)

### 3.1 The models

| lang | Hugging Face id | library / class | input size |
|---|---|---|---|
| en | `ViT-B-32` (pretrained `openai`) | `open_clip` | 224 |
| zh | `OFA-Sys/chinese-clip-vit-base-patch16` | `transformers.ChineseCLIPModel` | 224 |
| ko | `Bingsu/clip-vit-base-patch32-ko` | `transformers.AutoModel` (VisionTextDualEncoder) | 224 |
| ja | `line-corporation/clip-japanese-base` | `transformers` + `trust_remote_code=True` (custom `CLYPModel`) | 224 |

### 3.2 Uniform interface

Every wrapper exposes the same two methods, returning **L2-normalized** features so a plain
dot product gives cosine similarity:

```python
class SomeCLIP:
    lang = "xx"
    def embed_images(self, pil_list) -> Tensor[N, D]   # normalized
    def embed_texts(self, word_list) -> Tensor[C, D]   # normalized, uses TMPL[lang]
```

Classification is one shared helper:

```python
def classify(model, imgs, words):           # words = class names in model.lang
    imf = model.embed_images(imgs)
    tf  = model.embed_texts(words)
    return (imf @ tf.t()).argmax(-1).cpu().numpy()   # predicted class indices
```

### 3.3 Class names + prompt templates (STL-10)

```python
CLASSES = {
 "en": ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"],
 "zh": ["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"],
 "ko": ["비행기","새","자동차","고양이","사슴","개","말","원숭이","배","트럭"],
 "ja": ["飛行機","鳥","車","猫","鹿","犬","馬","猿","船","トラック"],
}
TMPL = {"en":"a photo of a {}.","zh":"一张{}的照片。","ko":"{}의 사진.","ja":"{}の写真。"}
```

### 3.4 The four wrappers (the parts that differ are the model APIs)

```python
import torch, torch.nn.functional as F, open_clip
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          ChineseCLIPModel, ChineseCLIPProcessor)
DEVICE = "cuda"

class EnCLIP:                              # OpenAI CLIP via open_clip
    lang = "en"
    def __init__(self):
        self.m,_,self.pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.m = self.m.to(DEVICE).eval(); self.tok = open_clip.get_tokenizer("ViT-B-32")
    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL["en"].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)

class ZhCLIP:                             # Chinese-CLIP: processor + get_*_features
    lang = "zh"
    def __init__(self):
        self.m = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(DEVICE).eval()
        self.p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(self.m.get_image_features(pixel_values=pv), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL["zh"].format(w) for w in words], padding=True, return_tensors="pt").to(DEVICE)
        return F.normalize(self.m.get_text_features(**t), dim=-1)

class KoCLIP:                            # Bingsu VisionTextDualEncoder
    lang = "ko"
    def __init__(self):
        self.m = AutoModel.from_pretrained("Bingsu/clip-vit-base-patch32-ko").to(DEVICE).eval()
        self.p = AutoProcessor.from_pretrained("Bingsu/clip-vit-base-patch32-ko")
    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(self.m.get_image_features(pixel_values=pv), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL["ko"].format(w) for w in words], padding=True, return_tensors="pt").to(DEVICE)
        return F.normalize(self.m.get_text_features(input_ids=t["input_ids"],
                                                    attention_mask=t["attention_mask"]), dim=-1)

class JaCLIP:                            # line-corp custom CLYP (trust_remote_code)
    lang = "ja"
    def __init__(self):
        self.m  = AutoModel.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True).to(DEVICE).eval()
        self.tok = AutoTokenizer.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True)
        self.ip  = AutoImageProcessor.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True)
    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.ip(images=imgs, return_tensors="pt").to(DEVICE)
        return F.normalize(self.m.get_image_features(**pv), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL["ja"].format(w) for w in words], padding=True)   # NOTE: no return_tensors!
        t = {k: v.to(DEVICE) for k, v in t.items()}
        return F.normalize(self.m.get_text_features(**t), dim=-1)
```

> **⚠️ Per-model gotchas (these cost real debugging time):**
> - **Japanese tokenizer**: the custom CLYP tokenizer tensorizes internally. Passing
>   `return_tensors="pt"` double-converts and throws
>   `TypeError: can only concatenate list (not "Tensor") to list`. Call it with just
>   `padding=True` and move the returned tensors to the device yourself.
> - **Korean / Chinese**: use `get_image_features(pixel_values=...)` and
>   `get_text_features(input_ids=..., attention_mask=...)` — *not* `encode_image/encode_text`.
> - **`trust_remote_code=True`** is required for the Japanese model (downloads custom
>   `*.py`); fine for research, but know you're running repo code.
> - Each model has its **own preprocessing** — always pass **raw PIL images** to the
>   wrappers and let each model resize/normalize. Don't pre-normalize.

---

## 4. The typographic attack (`typographic_attack.py`)

Rendering a word onto an image — the whole "attack." Use a **CJK-capable** font for
Korean/Japanese/Chinese and a Latin/Cyrillic font for the rest (verify it's not rendering
empty boxes):

```python
from PIL import Image, ImageDraw, ImageFont
CJK   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # Latin + CJK
LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"          # Latin + Cyrillic

def draw_word(pil_img, word, font_path=CJK, size=40, where="bottom"):
    img = pil_img.convert("RGB").resize((224, 224), Image.BICUBIC)
    d = ImageDraw.Draw(img); font = ImageFont.truetype(font_path, size)
    bb = d.textbbox((0, 0), word, font=font); w, h = bb[2]-bb[0], bb[3]-bb[1]
    x = (224 - w) // 2
    y = (224 - h - 16) if where == "bottom" else (224 - h) // 2
    d.rectangle([x-8, y-8, x+w+8, y+h+12], fill=(255, 255, 255))   # white box
    d.text((x - bb[0], y - bb[1]), word, fill=(0, 0, 0), font=font)  # black text
    return img
```

**Always sanity-check the fonts** by saving a strip of rendered words and *looking* at it
(missing-glyph "tofu" boxes silently invalidate results):

```python
strip = Image.new("RGB", (224*len(words), 224), (200,200,200))
for i, w in enumerate(words): strip.paste(draw_word(base_img, w, font_for(w)), (i*224, 0))
strip.save("results/font_check.png")   # then open and eyeball it
```

> **⚠️ Cropping check.** Some image processors center-crop. If you place text at the
> bottom, a crop can remove it and fake "robustness." Re-run with `where="center"` to
> confirm a robust model is genuinely ignoring the text, not just cropping it out.

---

## 5. Putting it together (`shared_vs_separate.py`)

```python
import torchvision, numpy as np, random
from perlang_models import EnCLIP, ZhCLIP, KoCLIP, JaCLIP, classify, CLASSES
from typographic_attack import draw_word          # or the draw_en helper

LANGS = ["en", "zh", "ko", "ja"]
ds = torchvision.datasets.STL10("data", split="test", download=False)
rng = random.Random(0)
idx = rng.sample(range(len(ds)), 200)
true   = np.array([ds[i][1] for i in idx])
target = np.array([rng.choice([c for c in range(10) if c != true[k]]) for k in range(len(idx))])

clean = [ds[i][0].convert("RGB") for i in idx]
# attack: write the ENGLISH target word (the strong typographic attacker)
attacked = [draw_word(ds[idx[k]][0], CLASSES["en"][target[k]]) for k in range(len(idx))]

models = {l: cls() for l, cls in zip(LANGS, [EnCLIP, ZhCLIP, KoCLIP, JaCLIP])}
def preds(imgs): return {l: classify(models[l], imgs, CLASSES[l]) for l in LANGS}

pc, pa = preds(clean), preds(attacked)
for l in LANGS:
    asr = (pa[l] == target).mean()
    print(f"{l}: clean {100*(pc[l]==true).mean():.1f}%  attacked {100*(pa[l]==true).mean():.1f}%  ASR {100*asr:.1f}%")

# disagreement = the defense signal
import numpy as np
P = lambda d: np.stack([d[l] for l in LANGS])
agree = lambda d: (P(d) == P(d)[0:1]).all(0).mean()
print("all-agree  clean -> attacked:", agree(pc), "->", agree(pa))
```

Run it:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python shared_vs_separate.py
```

---

## 6. Key metrics (what to compute)

| metric | meaning |
|---|---|
| **accuracy** | `pred == true_class` |
| **ASR** (attack success) | `pred == written/target class` |
| **agreement** | fraction of images where all models predict the same class (clean vs attacked) |
| **majority-vote ensemble** | most common prediction across models, then accuracy |
| **detector AUC** | can `#unique predictions` separate clean (neg) from attacked (pos)? `>0.5` ⇒ disagreement detects the attack |

---

## 7. Extending it (add a new language)

1. Find a per-language CLIP on HF and add a wrapper class with `embed_images` /
   `embed_texts` returning **normalized** features (copy the closest existing wrapper).
2. Add the class-name translations to `CLASSES[lang]` and a prompt to `TMPL[lang]`.
3. Pick a font that covers its script; add it to the font map and **eyeball the strip**.
4. Add the language to the `LANGS` list in the driver.

That's it — the uniform `embed_images`/`embed_texts`/`classify` interface keeps the attack
and metric code unchanged.
```
