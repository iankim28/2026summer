"""Tie reading-ability to attack-strength: how big must written text be to OVERRIDE the
real object, per writing language? On REAL images, sweep font size; measure attack success
(prediction flips to the written word) with an English query. English should win at small
sizes; European needs big text; CJK barely ever wins -- mirroring the OCR-probe reading scores."""
import json, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
from mclip_lib import load_model, build_text_embeddings, encode_image, logits_for, get_logit_scale, STL10_CLASSES

CJK="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; LATIN="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WORDS={
 "en":(["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"],LATIN),
 "es":(["avión","pájaro","coche","gato","ciervo","perro","caballo","mono","barco","camión"],LATIN),
 "de":(["Flugzeug","Vogel","Auto","Katze","Hirsch","Hund","Pferd","Affe","Schiff","Lastwagen"],LATIN),
 "ko":(["비행기","새","자동차","고양이","사슴","개","말","원숭이","배","트럭"],CJK),
 "ja":(["飛行機","鳥","車","猫","鹿","犬","馬","猿","船","トラック"],CJK),
}
device="cuda"; rng=random.Random(0)
model,tok,_,mean,std=load_model(device); ls=get_logit_scale(model)
EN_LABELS=build_text_embeddings(model,tok,STL10_CLASSES,device)["en"]
ds=torchvision.datasets.STL10("data",split="test",download=False)
idx=list(range(len(ds))); rng.shuffle(idx); idx=idx[:300]
true=np.array([ds[i][1] for i in idx])
target=np.array([rng.choice([c for c in range(10) if c!=true[k]]) for k in range(len(idx))])
tt=torchvision.transforms.ToTensor()

def draw(pil,word,font_path,size):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    f=ImageFont.truetype(font_path,size); bb=d.textbbox((0,0),word,font=f)
    w,h=bb[2]-bb[0],bb[3]-bb[1]; x=(224-w)//2; y=224-h-16
    d.rectangle([x-8,y-8,x+w+8,y+h+12],fill=(255,255,255)); d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f)
    return img

@torch.no_grad()
def asr(imgs):
    xs=torch.stack([tt(im) for im in imgs]).to(device)
    pred=logits_for(encode_image(model,xs,mean,std),EN_LABELS,ls).argmax(-1).cpu().numpy()
    return (pred==target).mean()

SIZES=[16,24,32,40,52]
print("Attack success rate (pred -> WRITTEN word), English query, on real images:")
print(f"{'size':>5} |" + "".join(f"{l:>8}" for l in WORDS))
res={l:[] for l in WORDS}
for s in SIZES:
    row=[]
    for l,(words,font) in WORDS.items():
        imgs=[draw(ds[idx[k]][0],words[target[k]],font,s) for k in range(len(idx))]
        a=asr(imgs); res[l].append(float(a)); row.append(a)
    print(f"{s:>5} |"+"".join(f"{100*v:7.1f}%" for v in row))
json.dump({"sizes":SIZES,"asr":res},open("results/override.json","w"),indent=2)
print("\nsaved results/override.json")
