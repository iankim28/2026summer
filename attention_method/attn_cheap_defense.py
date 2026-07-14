"""Fair forward-call comparison: CHEAP attention-based localization (attention is free
during the forward pass) vs expensive occlusion. Adds the ~2-3 forward attention points."""
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
def tfeat(m,p,w,zh):
    t=p(text=[("一张%s的照片" if zh else "a photo of a %s.")%x for x in w],padding=True,return_tensors="pt").to(device)
    return F.normalize(m.get_text_features(**t),dim=-1)
EN_T=tfeat(en_m,en_p,EN_W,False); ZH_T=tfeat(zh_m,zh_p,ZH_W,True)
def pvof(p,pil): return p(images=[pil],return_tensors="pt").pixel_values.to(device)
@torch.no_grad()
def feats(m,pv): return F.normalize(m.get_image_features(pixel_values=pv),dim=-1)
@torch.no_grad()
def predict(m,pv,T): return (feats(m,pv)@T.t()).argmax(-1).item()
def rollout(att):
    S=att[0].shape[-1]; R=torch.eye(S,device=att[0].device)[None]
    for A in att:
        A=A.mean(1)+torch.eye(S,device=A.device); A=A/A.sum(-1,keepdim=True); R=A@R
    cls=R[:,0,1:]; g=int(round(cls.shape[1]**0.5))
    s=F.interpolate(cls.reshape(1,1,g,g),size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/s.max()).cpu().numpy()
@torch.no_grad()
def attn_sal(m,pv):  # attention rollout, comes FREE with the forward pass
    return rollout(m.vision_model(pixel_values=pv,output_attentions=True).attentions)
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
def boxmask(box):
    m=np.zeros((224,224),bool); x0,y0,x1,y1=box; m[y0:y1,x0:x1]=True; return m
def inbox(s,box): m=boxmask(box); return (s[m].sum()/(s.sum()+1e-9))/(m.mean()+1e-9)

ds=torchvision.datasets.STL10("data",split="test",download=False)
idx=list(np.random.default_rng(0).permutation(len(ds))[:N]); rng=np.random.default_rng(1)
en1=0; ov=0; ib_en=[]; ib_ov=[]
for i in idx:
    true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true])
    att,box=draw(ds[i][0],EN_W[tgt]); en_pv=pvof(en_p,att); zh_pv=pvof(zh_p,att)
    es=attn_sal(en_m,en_pv); zs=attn_sal(zh_m,zh_pv)
    ib_en.append(inbox(es,box))
    # single-model attention defense (cost 2: 1 fwd for attn+pred, 1 for second pass)
    reg=es>=0.5*es.max(); en1+= (predict(en_m,maskpv(en_pv,reg),EN_T)==true)
    # 2-model attention OVERLAP defense (cost 3)
    o=np.minimum(es,zs); ib_ov.append(inbox(o,box)); oreg=o>=0.5*(o.max()+1e-9)
    ov+= (predict(en_m,maskpv(en_pv,oreg),EN_T)==true)
attn_pts=[("attn single (EN)",2,100*en1/N),("attn 2-model overlap",3,100*ov/N)]
print(f"CHEAP attention-based localization defense (n={N}):")
for nm,c,a in attn_pts: print(f"  {nm:>22}: cost={c} fwd/img  robust_acc={a:.1f}%")
print(f"  attention in-box focus ratio: EN single={np.mean(ib_en):.2f}  overlap={np.mean(ib_ov):.2f}  (occlusion was 4.7)")

# combine with occlusion points and re-plot
occ=json.load(open("results/cost_accuracy.json"))["points"]
allpts=[(p["name"],p["cost"],p["acc"]) for p in occ]+attn_pts
plt.figure(figsize=(8.5,5.5)); clean=95.0
occ_series=sorted([(c,a) for n,c,a in allpts if n.startswith("occlusion")])
plt.plot([c for c,_ in occ_series],[a for _,a in occ_series],"--",c="tab:blue",alpha=.6,label="occlusion sweep")
for nm,c,a in allpts:
    col = "red" if nm.startswith("no defense") else "green" if nm.startswith("oracle") else \
          "tab:orange" if nm.startswith("attn") else "gray" if "overlap 7x7" in nm else "tab:blue"
    mk = "*" if nm.startswith("oracle") else "s" if nm.startswith("attn") else "o"
    plt.scatter(c,a,s=110 if mk=="*" else 70,c=col,marker=mk,zorder=5)
    plt.annotate(nm.replace("occlusion ","occ "),(c,a),textcoords="offset points",xytext=(6,4),fontsize=7.5,color=col)
plt.axhline(clean,color="k",ls=":",alpha=.5,label="clean acc (95%)")
plt.xscale("log"); plt.xlabel("inference cost  (forward passes / image, log scale)")
plt.ylabel("robust accuracy under English typographic attack (%)")
plt.title(f"Forward-call cost vs robust accuracy: cheap ATTENTION vs OCCLUSION (n={N})")
plt.ylim(20,100); plt.grid(alpha=.3); plt.legend(loc="lower right",fontsize=8)
plt.tight_layout(); plt.savefig("results/cost_accuracy_v2.png",dpi=130); print("saved results/cost_accuracy_v2.png")
