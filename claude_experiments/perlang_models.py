"""Uniform wrappers for 4 INDEPENDENT per-language CLIP models (separate encoders),
plus the shared multilingual model. Each exposes embed_images(PIL list) / embed_texts(str list)
returning L2-normalized features, and classify(imgs, labels)->pred indices."""
import torch, torch.nn.functional as F
import open_clip
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          ChineseCLIPModel, ChineseCLIPProcessor)

DEVICE = "cuda"

# STL-10 class order
CLASSES = {
 "en": ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"],
 "zh": ["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"],
 "ko": ["비행기","새","자동차","고양이","사슴","개","말","원숭이","배","트럭"],
 "ja": ["飛行機","鳥","車","猫","鹿","犬","馬","猿","船","トラック"],
}
TMPL = {"en":"a photo of a {}.","zh":"一张{}的照片。","ko":"{}의 사진.","ja":"{}の写真。"}


class EnCLIP:  # OpenAI CLIP ViT-B/32 via open_clip
    lang = "en"
    def __init__(self):
        self.m,_,self.pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.m=self.m.to(DEVICE).eval(); self.tok=open_clip.get_tokenizer("ViT-B-32")
    @torch.no_grad()
    def embed_images(self, imgs):
        x=torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x),dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t=self.tok([TMPL["en"].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t),dim=-1)


class ZhCLIP:  # OFA-Sys Chinese-CLIP via transformers
    lang = "zh"
    def __init__(self):
        self.m=ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(DEVICE).eval()
        self.p=ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    @torch.no_grad()
    def embed_images(self, imgs):
        pv=self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(self.m.get_image_features(pixel_values=pv),dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t=self.p(text=[TMPL["zh"].format(w) for w in words], padding=True, return_tensors="pt").to(DEVICE)
        return F.normalize(self.m.get_text_features(**t),dim=-1)


class KoCLIP:  # Bingsu Korean CLIP (VisionTextDualEncoder) via transformers
    lang = "ko"
    def __init__(self):
        self.m=AutoModel.from_pretrained("Bingsu/clip-vit-base-patch32-ko").to(DEVICE).eval()
        self.p=AutoProcessor.from_pretrained("Bingsu/clip-vit-base-patch32-ko")
    @torch.no_grad()
    def embed_images(self, imgs):
        pv=self.p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        return F.normalize(self.m.get_image_features(pixel_values=pv),dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t=self.p(text=[TMPL["ko"].format(w) for w in words], padding=True, return_tensors="pt").to(DEVICE)
        return F.normalize(self.m.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"]),dim=-1)


class JaCLIP:  # line-corporation clip-japanese-base (custom CLYP) via transformers
    lang = "ja"
    def __init__(self):
        self.m=AutoModel.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True).to(DEVICE).eval()
        self.tok=AutoTokenizer.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True)
        self.ip=AutoImageProcessor.from_pretrained("line-corporation/clip-japanese-base", trust_remote_code=True)
    @torch.no_grad()
    def embed_images(self, imgs):
        pv=self.ip(images=imgs, return_tensors="pt").to(DEVICE)
        f=self.m.get_image_features(**pv)
        return F.normalize(f,dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        # custom CLYP tokenizer tensorizes internally; do NOT pass return_tensors
        t=self.tok([TMPL["ja"].format(w) for w in words], padding=True)
        t={k:v.to(DEVICE) for k,v in t.items()}
        f=self.m.get_text_features(**t)
        return F.normalize(f,dim=-1)


def classify(model, imgs, words):
    imf = model.embed_images(imgs); tf = model.embed_texts(words)
    return (imf @ tf.t()).argmax(-1).cpu().numpy()
