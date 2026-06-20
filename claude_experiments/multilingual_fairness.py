"""Feasibility check: does multilingual CLIP work much worse in some languages?
(High-school 'fairness' angle: accuracy across diverse languages/scripts + which is the
adversarial 'weakest link'.) Clean accuracy + accuracy under a small shared FGSM attack."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from mclip_lib import load_model, encode_image, logits_for, get_logit_scale, STL10_CLASSES

# STL-10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
TR = {
 "en (English)": ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"],
 "es (Spanish)": ["avión","pájaro","coche","gato","ciervo","perro","caballo","mono","barco","camión"],
 "fr (French)":  ["avion","oiseau","voiture","chat","cerf","chien","cheval","singe","bateau","camion"],
 "de (German)":  ["Flugzeug","Vogel","Auto","Katze","Hirsch","Hund","Pferd","Affe","Schiff","Lastwagen"],
 "ko (Korean)":  ["비행기","새","자동차","고양이","사슴","개","말","원숭이","배","트럭"],
 "ja (Japanese)":["飛行機","鳥","車","猫","鹿","犬","馬","猿","船","トラック"],
 "zh (Chinese)": ["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"],
 "ru (Russian)": ["самолёт","птица","машина","кошка","олень","собака","лошадь","обезьяна","корабль","грузовик"],
 "vi (Vietnamese)":["máy bay","chim","xe hơi","mèo","hươu","chó","ngựa","khỉ","tàu thủy","xe tải"],
 "tr (Turkish)": ["uçak","kuş","araba","kedi","geyik","köpek","at","maymun","gemi","kamyon"],
 "ar (Arabic)":  ["طائرة","طائر","سيارة","قطة","غزال","كلب","حصان","قرد","سفينة","شاحنة"],
 "hi (Hindi)":   ["हवाई जहाज़","पक्षी","कार","बिल्ली","हिरण","कुत्ता","घोड़ा","बंदर","जहाज़","ट्रक"],
}
TMPL = {  # simple "a photo of a {}" per language
 "en (English)":"a photo of a {}.","es (Spanish)":"una foto de un {}.","fr (French)":"une photo d'un {}.",
 "de (German)":"ein Foto von einem {}.","ko (Korean)":"{}의 사진.","ja (Japanese)":"{}の写真。",
 "zh (Chinese)":"{}的照片。","ru (Russian)":"фотография {}.","vi (Vietnamese)":"một bức ảnh của một {}.",
 "tr (Turkish)":"bir {} fotoğrafı.","ar (Arabic)":"صورة {}.","hi (Hindi)":"{} की तस्वीर।",
}

device="cuda"
model, tokenizer, _, mean, std = load_model(device)
ls = get_logit_scale(model)

@torch.no_grad()
def embed(lang):
    prompts=[TMPL[lang].format(c) for c in TR[lang]]
    f=model.encode_text(tokenizer(prompts).to(device))
    return F.normalize(f,dim=-1)
TXT={l:embed(l) for l in TR}

tf=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor()])
ds=torchvision.datasets.STL10("data",split="test",download=False,transform=tf)
idx=np.random.default_rng(0).permutation(len(ds))[:500]
loader=DataLoader(Subset(ds,idx.tolist()),batch_size=100,num_workers=4)

def fgsm_en(x,y,eps):
    x=x.clone().detach().requires_grad_(True)
    loss=F.cross_entropy(logits_for(encode_image(model,x,mean,std),TXT["en (English)"],ls),y)
    g=torch.autograd.grad(loss,x)[0]
    return torch.clamp(x+eps*g.sign(),0,1).detach()

correct={l:0 for l in TR}; adv_correct={l:0 for l in TR}; total=0
for x,y in loader:
    x,y=x.to(device),y.to(device); total+=y.numel()
    with torch.no_grad():
        f=encode_image(model,x,mean,std)
        for l in TR: correct[l]+=(logits_for(f,TXT[l],ls).argmax(-1)==y).sum().item()
    xa=fgsm_en(x,y,4/255.)   # small attack designed on ENGLISH only
    with torch.no_grad():
        fa=encode_image(model,xa,mean,std)
        for l in TR: adv_correct[l]+=(logits_for(fa,TXT[l],ls).argmax(-1)==y).sum().item()

print(f"{'language':>16} | clean acc | acc under English-FGSM(eps=4)")
res={}
for l in sorted(TR,key=lambda k:-correct[k]/total):
    ca=100*correct[l]/total; aa=100*adv_correct[l]/total
    res[l]={"clean":ca,"adv":aa}
    print(f"{l:>16} | {ca:8.1f}% | {aa:8.1f}%")
json.dump(res,open("results/fairness.json","w"),indent=2,ensure_ascii=False)
print("\nsaved results/fairness.json")
