"""Cheap->expensive CASCADE: run the 3-forward attention-overlap defense on every image,
then ESCALATE only the least-confident fraction to occlusion(4x4). Sweeping that fraction
traces a Pareto curve between the cheap (57%@3fwd) and occlusion (84%@18fwd) points."""
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
    w,h=bb[2]-bb[0],bb[3]-bb[1]; x=(224-w)//2; y=224-h-16
    d.rectangle((x-8,y-8,x+w+8,y+h+12),fill=(255,255,255)); d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f)
    return img
en_m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
en_p=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
zh_m=ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
zh_p=ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
@torch.no_grad()
def tf(m,p,w,zh):
    t=p(text=[("一张%s的照片" if zh else "a photo of a %s.")%x for x in w],padding=True,return_tensors="pt").to(device)
    return F.normalize(m.get_text_features(**t),dim=-1)
EN_T=tf(en_m,en_p,EN_W,False); ZH_T=tf(zh_m,zh_p,ZH_W,True)
def pvof(p,pil): return p(images=[pil],return_tensors="pt").pixel_values.to(device)
@torch.no_grad()
def feats(m,pv): return F.normalize(m.get_image_features(pixel_values=pv),dim=-1)
@torch.no_grad()
def probs(m,pv,T): return F.softmax(100*(feats(m,pv)@T.t()),-1)[0]
def rollout(att):
    S=att[0].shape[-1]; R=torch.eye(S,device=att[0].device)[None]
    for A in att:
        A=A.mean(1)+torch.eye(S,device=A.device); A=A/A.sum(-1,keepdim=True); R=A@R
    cls=R[:,0,1:]; g=int(round(cls.shape[1]**0.5))
    s=F.interpolate(cls.reshape(1,1,g,g),size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/s.max()).cpu().numpy()
@torch.no_grad()
def attn(m,pv): return rollout(m.vision_model(pixel_values=pv,output_attentions=True).attentions)
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
@torch.no_grad()
def occ4(m,pv0,T):   # occlusion 4x4 defense -> predicted class
    G=4;cell=56;c=probs(m,pv0,T).argmax().item();base=(feats(m,pv0)@T[c:c+1].t()).item()
    occ=pv0.repeat(16,1,1,1)
    for k in range(16):
        r,cc=divmod(k,G);occ[k,:,r*cell:(r+1)*cell,cc*cell:(cc+1)*cell]=0
    drop=(base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).reshape(G,G).clamp(min=0)
    s=F.interpolate(drop[None,None],size=(224,224),mode="nearest")[0,0].cpu().numpy()
    reg=s>=0.5*(s.max()+1e-9)
    return probs(m,maskpv(pv0,reg),T).argmax().item()

ds=torchvision.datasets.STL10("data",split="test",download=False)
idx=list(np.random.default_rng(0).permutation(len(ds))[:N]); rng=np.random.default_rng(1)
true=[]; cheap_ok=[]; occ_ok=[]; uncert=[]
for i in idx:
    t=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=t]); true.append(t)
    att_img=draw(ds[i][0],EN_W[tgt]); en_pv=pvof(en_p,att_img); zh_pv=pvof(zh_p,att_img)
    es=attn(en_m,en_pv); zs=attn(zh_m,zh_pv)
    o=np.minimum(es,zs); reg=o>=0.5*(o.max()+1e-9)
    pr=probs(en_m,maskpv(en_pv,reg),EN_T)             # cheap-defended prediction
    cheap_ok.append(int(pr.argmax().item()==t))
    top2=torch.topk(pr,2).values; uncert.append((1-(top2[0]-top2[1])).item())   # low margin -> uncertain
    occ_ok.append(int(occ4(en_m,en_pv,EN_T)==t))
true=np.array(true); cheap_ok=np.array(cheap_ok); occ_ok=np.array(occ_ok); uncert=np.array(uncert)
order=np.argsort(-uncert)   # most-uncertain first

fracs=np.linspace(0,1,11); casc=[]
for f in fracs:
    k=int(round(f*N)); esc=set(order[:k].tolist())
    acc=np.mean([occ_ok[j] if j in esc else cheap_ok[j] for j in range(N)])
    casc.append((3+f*18, 100*acc))
print("CASCADE (attention-overlap -> escalate uncertain to occlusion 4x4):")
for (c,a),f in zip(casc,fracs): print(f"  escalate {f*100:3.0f}%: cost={c:5.1f} fwd/img  acc={a:.1f}%")

occ=json.load(open("results/cost_accuracy.json"))["points"]
plt.figure(figsize=(8.5,5.5)); clean=95.0
plt.plot([c for c,_ in casc],[a for _,a in casc],"-o",c="tab:orange",ms=5,label="CASCADE (attn→occlusion, sweep escalation %)")
for p in occ:
    if p["name"].startswith("occlusion"):
        plt.scatter(p["cost"],p["acc"],s=55,c="tab:blue",zorder=5)
        plt.annotate(p["name"].replace("occlusion ","occ "),(p["cost"],p["acc"]),textcoords="offset points",xytext=(5,4),fontsize=7,color="tab:blue")
    elif p["name"].startswith("no defense"):
        plt.scatter(p["cost"],p["acc"],s=70,c="red",zorder=5); plt.annotate("no defense",(p["cost"],p["acc"]),textcoords="offset points",xytext=(5,4),fontsize=7.5,color="red")
    elif p["name"].startswith("oracle"):
        plt.scatter(p["cost"],p["acc"],s=110,c="green",marker="*",zorder=5); plt.annotate("oracle",(p["cost"],p["acc"]),textcoords="offset points",xytext=(5,-12),fontsize=7.5,color="green")
plt.scatter(3,casc[0][1],s=90,c="tab:orange",marker="s",zorder=6); plt.annotate("attn overlap (0% esc)",(3,casc[0][1]),textcoords="offset points",xytext=(6,-12),fontsize=7.5,color="tab:orange")
plt.axhline(clean,color="k",ls=":",alpha=.5,label="clean acc (95%)")
plt.xscale("log"); plt.xlabel("inference cost  (forward passes / image, log scale)")
plt.ylabel("robust accuracy under English typographic attack (%)")
plt.title(f"Cheap→expensive CASCADE traces the Pareto curve (n={N})")
plt.ylim(20,100); plt.grid(alpha=.3); plt.legend(loc="lower right",fontsize=8)
plt.tight_layout(); plt.savefig("results/cascade.png",dpi=130); print("saved results/cascade.png")
