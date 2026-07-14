"""Inference-cost vs robust-accuracy curve for the occlusion-localization defense.
Cost = image-encoder forward passes per image (FLOPs proxy). Sweep grid resolution."""
import json, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device="cuda"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; N=150
EN_W=["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W=["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]

def draw(pil,word,size=40):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    f=ImageFont.truetype(FONT,size); bb=d.textbbox((0,0),word,font=f)
    w,h=bb[2]-bb[0],bb[3]-bb[1]; x=(224-w)//2; y=224-h-16; box=(x-8,y-8,x+w+8,y+h+12)
    d.rectangle(box,fill=(255,255,255)); d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f)
    return img,box
en_m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
en_p=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
zh_m=ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
zh_p=ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
@torch.no_grad()
def tfeat(m,p,words,zh):
    t=p(text=[("一张%s的照片" if zh else "a photo of a %s.")%w for w in words],padding=True,return_tensors="pt").to(device)
    return F.normalize(m.get_text_features(**t),dim=-1)
EN_T=tfeat(en_m,en_p,EN_W,False); ZH_T=tfeat(zh_m,zh_p,ZH_W,True)
@torch.no_grad()
def feats(m,pv): return F.normalize(m.get_image_features(pixel_values=pv),dim=-1)
@torch.no_grad()
def pred(m,pv,T): return (feats(m,pv)@T.t()).argmax(-1).item()
def pvof(p,pil): return p(images=[pil],return_tensors="pt").pixel_values.to(device)
@torch.no_grad()
def occ_map(m,pv0,T,G):
    cell=224//G; c=pred(m,pv0,T); base=(feats(m,pv0)@T[c:c+1].t()).item()
    occ=pv0.repeat(G*G,1,1,1)
    for k in range(G*G):
        r,cc=divmod(k,G); occ[k,:,r*cell:(r+1)*cell,cc*cell:(cc+1)*cell]=0
    drop=(base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).reshape(G,G).clamp(min=0)
    return F.interpolate(drop[None,None],size=(224,224),mode="nearest")[0,0].cpu().numpy()
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
def boxmask(box):
    m=np.zeros((224,224),bool); x0,y0,x1,y1=box; m[y0:y1,x0:x1]=True; return m

ds=torchvision.datasets.STL10("data",split="test",download=False)
idx=list(np.random.default_rng(0).permutation(len(ds))[:N]); rng=np.random.default_rng(1)
grids=[2,4,7,14]
hit={f"occ{G}":0 for G in grids}; nodef=0; oracle=0; overlap=0
for i in idx:
    true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true])
    att,box=draw(ds[i][0],EN_W[tgt]); en_pv=pvof(en_p,att); zh_pv=pvof(zh_p,att); bm=boxmask(box)
    nodef+= (pred(en_m,en_pv,EN_T)==true)
    oracle+= (pred(en_m,maskpv(en_pv,bm),EN_T)==true)
    for G in grids:
        s=occ_map(en_m,en_pv,EN_T,G); reg=s>=0.5*(s.max()+1e-9)
        hit[f"occ{G}"]+= (pred(en_m,maskpv(en_pv,reg),EN_T)==true)
    # two-model overlap at G=7
    se=occ_map(en_m,en_pv,EN_T,7); sz=occ_map(zh_m,zh_pv,ZH_T,7)
    ov=np.minimum(se/(se.max()+1e-9), sz/(sz.max()+1e-9)); ovreg=ov>=0.5*(ov.max()+1e-9)
    overlap+= (pred(en_m,maskpv(en_pv,ovreg),EN_T)==true)

pts=[("no defense",1,100*nodef/N),
     *[(f"occlusion {G}x{G}",G*G+2,100*hit[f'occ{G}']/N) for G in grids],
     ("2-model overlap 7x7",2*(49)+2,100*overlap/N),
     ("oracle (known box)",2,100*oracle/N)]
for name,cost,acc in pts: print(f"{name:>22}: cost={cost:>4} fwd/img  robust_acc={acc:.1f}%")
json.dump({"points":[{"name":n,"cost":c,"acc":a} for n,c,a in pts],"n":N},open("results/cost_accuracy.json","w"),indent=2)

# plot
plt.figure(figsize=(8,5.5))
clean=95.0
for name,cost,acc in pts:
    if name.startswith("oracle"):
        plt.scatter(cost,acc,s=90,c="green",marker="*",zorder=5); plt.annotate(name,(cost,acc),textcoords="offset points",xytext=(6,-12),fontsize=8,color="green")
    elif name.startswith("no defense"):
        plt.scatter(cost,acc,s=70,c="red",zorder=5); plt.annotate(name,(cost,acc),textcoords="offset points",xytext=(6,4),fontsize=8,color="red")
    elif name.startswith("2-model"):
        plt.scatter(cost,acc,s=70,c="gray",zorder=5); plt.annotate(name,(cost,acc),textcoords="offset points",xytext=(-10,-14),fontsize=8,color="gray")
    else:
        plt.scatter(cost,acc,s=70,c="tab:blue",zorder=5); plt.annotate(name.replace("occlusion ",""),(cost,acc),textcoords="offset points",xytext=(6,4),fontsize=8,color="tab:blue")
occ=[(G*G+2,100*hit[f'occ{G}']/N) for G in grids]
plt.plot([c for c,_ in occ],[a for _,a in occ],"--",c="tab:blue",alpha=0.6,label="occlusion defense (grid sweep)")
plt.axhline(clean,color="k",ls=":",alpha=0.5,label="clean accuracy (95%)")
plt.xscale("log"); plt.xlabel("inference cost  (image-encoder forward passes / image, log scale)")
plt.ylabel("robust accuracy under English typographic attack (%)")
plt.title(f"Inference cost vs robust accuracy — occlusion-localization defense (n={N})")
plt.ylim(20,100); plt.legend(loc="lower right",fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/cost_accuracy.png",dpi=130); print("saved results/cost_accuracy.png")
