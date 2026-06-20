"""WHY is the non-English typographic attack weak? Mechanism experiment.

Claim: there is ONE shared image encoder; the query language only changes the TEXT labels.
The attack works only if that single encoder can READ the written text and map it to a
concept. We isolate this with an "OCR probe": render ONLY a word on a blank background
(no object) and classify it. If the encoder reads it, the blank-image embedding lands on
that word's concept.

We test:
  (1) reading ability per WRITING language (Latin-European, CJK, Cyrillic, Devanagari)
  (2) ROMANIZATION: same words written in Latin letters -> separates SCRIPT vs VOCABULARY
  (3) that reading ability (per writing language) predicts the typographic attack success.
"""
import json, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from mclip_lib import load_model, encode_image, logits_for, get_logit_scale

CJK   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"      # Latin + Cyrillic
DEVA  = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"

# STL-10 classes order: airplane,bird,car,cat,deer,dog,horse,monkey,ship,truck
WORDS = {
 "en":        (["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"], LATIN),
 "es":        (["avión","pájaro","coche","gato","ciervo","perro","caballo","mono","barco","camión"], LATIN),
 "fr":        (["avion","oiseau","voiture","chat","cerf","chien","cheval","singe","bateau","camion"], LATIN),
 "de":        (["Flugzeug","Vogel","Auto","Katze","Hirsch","Hund","Pferd","Affe","Schiff","Lastwagen"], LATIN),
 "ru":        (["самолёт","птица","машина","кошка","олень","собака","лошадь","обезьяна","корабль","грузовик"], LATIN),
 "ko":        (["비행기","새","자동차","고양이","사슴","개","말","원숭이","배","트럭"], CJK),
 "ja":        (["飛行機","鳥","車","猫","鹿","犬","馬","猿","船","トラック"], CJK),
 "zh":        (["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"], CJK),
 "hi":        (["हवाई जहाज़","पक्षी","कार","बिल्ली","हिरण","कुत्ता","घोड़ा","बंदर","जहाज़","ट्रक"], DEVA),
 # romanizations (Latin script, non-English vocabulary) -> SCRIPT vs VOCABULARY test
 "ko-roman":  (["bihaenggi","sae","jadongcha","goyangi","saseum","gae","mal","wonsungi","bae","teureok"], LATIN),
 "ja-roman":  (["hikouki","tori","kuruma","neko","shika","inu","uma","saru","fune","torakku"], LATIN),
 "zh-roman":  (["feiji","niao","qiche","mao","lu","gou","ma","houzi","chuan","kache"], LATIN),
}
SCRIPT = {"en":"Latin","es":"Latin","fr":"Latin","de":"Latin","ru":"Cyrillic","ko":"CJK",
          "ja":"CJK","zh":"CJK","hi":"Devanagari","ko-roman":"Latin(roman)","ja-roman":"Latin(roman)","zh-roman":"Latin(roman)"}

device = "cuda"
model, tokenizer, _, mean, std = load_model(device)
ls = get_logit_scale(model)

# English reference label embeddings (concept space): "a photo of a {}"
EN_WORDS = WORDS["en"][0]
@torch.no_grad()
def embed_labels(words, tmpl="a photo of a {}."):
    f = model.encode_text(tokenizer([tmpl.format(w) for w in words]).to(device))
    return F.normalize(f, dim=-1)
EN_LABELS = embed_labels(EN_WORDS)

def render_textonly(word, font_path, size, bg=128):
    img = Image.new("RGB", (224, 224), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    bb = d.textbbox((0, 0), word, font=font)
    w, h = bb[2]-bb[0], bb[3]-bb[1]
    if w > 210:  # shrink to fit width
        font = ImageFont.truetype(font_path, max(12, int(size*200/w)))
        bb = d.textbbox((0, 0), word, font=font); w, h = bb[2]-bb[0], bb[3]-bb[1]
    d.text(((224-w)//2 - bb[0], (224-h)//2 - bb[1]), word, fill=(0, 0, 0), font=font)
    return img

to_tensor = __import__("torchvision").transforms.ToTensor()

@torch.no_grad()
def classify_imgs(imgs, labels):
    xs = torch.stack([to_tensor(im) for im in imgs]).to(device)
    feats = encode_image(model, xs, mean, std)
    return feats, logits_for(feats, labels, ls)

SIZES = [30, 40, 52]  # a few sizes per word for a smoother estimate

# control: blank gray, no text
blank = [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(10)]
_, lg = classify_imgs(blank, EN_LABELS)
print(f"control (blank, no text) accuracy vs English labels: {100*(lg.argmax(-1)==torch.arange(10,device=device)).float().mean():.1f}%  (chance=10%)\n")

print("OCR PROBE: render ONLY the word (no object), classify the blank-text image.")
print(f"{'writing lang':>12} | {'script':>13} | {'reads->English concept':>22} | {'reads->own-lang label':>21} | mean cos")
results = {}
true_idx = torch.arange(10, device=device).repeat_interleave(len(SIZES))
for lang, (words, font) in WORDS.items():
    imgs = [render_textonly(w, font, s) for w in words for s in SIZES]
    feats, lg_en = classify_imgs(imgs, EN_LABELS)
    acc_en = (lg_en.argmax(-1) == true_idx).float().mean().item()
    # cosine of each text-image to its correct English concept label
    cos = (F.normalize(feats, dim=-1) * EN_LABELS[true_idx]).sum(-1).mean().item()
    # own-language labels (skip romanizations -> use English)
    base = lang.split("-")[0]
    own_labels = embed_labels(WORDS[base][0]) if base in WORDS else EN_LABELS
    if "roman" in lang:
        acc_own = float("nan")
    else:
        _, lg_own = classify_imgs(imgs, own_labels)
        acc_own = (lg_own.argmax(-1) == true_idx).float().mean().item()
    results[lang] = {"script": SCRIPT[lang], "acc_english_concept": acc_en,
                     "acc_own_label": (None if np.isnan(acc_own) else acc_own), "mean_cos": cos}
    own_str = "   (n/a)" if np.isnan(acc_own) else f"{100*acc_own:20.1f}%"
    print(f"{lang:>12} | {SCRIPT[lang]:>13} | {100*acc_en:21.1f}% | {own_str} | {cos:7.3f}")

# save a verification strip so we can confirm fonts render (not tofu)
strip = Image.new("RGB", (224*len(WORDS), 224), (200,200,200))
for i,(lang,(words,font)) in enumerate(WORDS.items()):
    strip.paste(render_textonly(words[5], font, 44), (i*224,0))  # word[5] = "dog" concept
strip.save("results/ocr_probe_strip.png")

json.dump(results, open("results/mechanism.json","w"), indent=2, ensure_ascii=False)
print("\nsaved results/mechanism.json and results/ocr_probe_strip.png")
