"""Multilingual vs unilingual typographic attack (2 random boxes), defended by attention-map
modulation (2-map / 4-map) and occlusion (top-1 / top-2 cells, 4x4). Classify with BOTH the
English and Chinese CLIP. Produce an inference-cost vs robust-accuracy graph.

Interpretation:
  Unilingual attack  = box1=EN(target), box2=EN(target)         (same 2 random locations)
  Multilingual attack= box1=EN(target), box2=ZH(target)
  attn 2-map = combine EN-rollout + ZH-rollout                  (cost 3 fwd)
  attn 4-map = + EN-gradcam + ZH-gradcam (2 models x 2 methods) (cost 5 fwd)
  occ 1-grid = 4x4 occlusion, mask top-1 cell                   (cost 18 fwd)
  occ 2-grid = 4x4 occlusion, mask top-2 cells                  (cost 18 fwd)
Cost = image-encoder forward passes to produce the final defended prediction.
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device="cuda"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; CJK="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
EN_W=["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W=["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]
PADX,PADY=8,10
def _bbox(word,size,cjk):
    f=ImageFont.truetype(CJK if cjk else FONT,size); im=Image.new("RGB",(224,224)); d=ImageDraw.Draw(im)
    bb=d.textbbox((0,0),word,font=f); return bb[2]-bb[0],bb[3]-bb[1],f
def put(d,word,size,cx,cy,cjk):
    w,h,f=_bbox(word,size,cjk); x=int(cx-w/2); y=int(cy-h/2); box=(x-PADX,y-PADY,x+w+PADX,y+h+PADY)
    d.rectangle(box,fill=(255,255,255)); bb=d.textbbox((0,0),word,font=f)
    d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f); return box
def rand_center(bw,bh,rng,m=4):
    return int(rng.integers(bw//2+m,max(bw//2+m+1,224-bw//2-m))), int(rng.integers(bh//2+m,max(bh//2+m+1,224-bh//2-m)))
def overlap(a,b,p=4): return not (a[2]+p<=b[0] or b[2]+p<=a[0] or a[3]+p<=b[1] or b[3]+p<=a[1])
def two_locs(w1,h1,w2,h2,rng):
    bw1,bh1,bw2,bh2=w1+2*PADX,h1+2*PADY,w2+2*PADX,h2+2*PADY
    c1=rand_center(bw1,bh1,rng); b1=(c1[0]-bw1//2,c1[1]-bh1//2,c1[0]+bw1//2,c1[1]+bh1//2)
    for _ in range(300):
        c2=rand_center(bw2,bh2,rng); b2=(c2[0]-bw2//2,c2[1]-bh2//2,c2[0]+bw2//2,c2[1]+bh2//2)
        if not overlap(b1,b2): break
    return c1,c2
def make_attack(pil,tgt,rng,multilingual,size=40):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    w1,h1,_=_bbox(EN_W[tgt],size,False)
    w2,h2,_=_bbox(ZH_W[tgt] if multilingual else EN_W[tgt],size,multilingual)
    c1,c2=two_locs(w1,h1,w2,h2,rng)
    b1=put(d,EN_W[tgt],size,c1[0],c1[1],False)
    b2=put(d,(ZH_W[tgt] if multilingual else EN_W[tgt]),size,c2[0],c2[1],multilingual)
    return img,[b1,b2]

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
def predict(m,pv,T): return (feats(m,pv)@T.t()).argmax(-1).item()

def rollout(att):
    S=att[0].shape[-1]; R=torch.eye(S,device=att[0].device)[None]
    for A in att:
        A=A.mean(1)+torch.eye(S,device=A.device); A=A/A.sum(-1,keepdim=True); R=A@R
    cls=R[:,0,1:]; g=int(round(cls.shape[1]**0.5))
    s=F.interpolate(cls.reshape(1,1,g,g),size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/(s.max()+1e-9)).detach().cpu().numpy()
@torch.no_grad()
def rollout_map(m,pv): return rollout(m.vision_model(pixel_values=pv,output_attentions=True).attentions)
def gradcam_map(m,pv,T):
    acts={}
    layer=m.vision_model.encoder.layers[-1]
    h=layer.register_forward_hook(lambda mod,i,o:acts.__setitem__('a',o[0] if isinstance(o,tuple) else o))
    f=F.normalize(m.get_image_features(pixel_values=pv),dim=-1)
    c=(f@T.t()).argmax(-1).item(); S=(f@T[c:c+1].t()).sum()
    a=acts['a']; a.retain_grad(); m.zero_grad(); S.backward()
    h.remove()
    cam=F.relu((a.grad*a).sum(-1))[0,1:]           # per-patch
    g=int(round(cam.shape[0]**0.5)); cam=cam.reshape(1,1,g,g)
    s=F.interpolate(cam,size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/(s.max()+1e-9)).detach().cpu().numpy()

def combine(maps):
    s=np.sum([mp/(mp.max()+1e-9) for mp in maps],0); return s/(s.max()+1e-9)
def region_from(mapc,thr=0.5): return mapc>=thr*mapc.max()
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
@torch.no_grad()
def occ_drop(m,pv0,T,G=4):                       # compute occlusion drops ONCE
    cell=224//G; c=predict(m,pv0,T); base=(feats(m,pv0)@T[c:c+1].t()).item()
    occ=pv0.repeat(G*G,1,1,1)
    for j in range(G*G):
        r,cc=divmod(j,G); occ[j,:,r*cell:(r+1)*cell,cc*cell:(cc+1)*cell]=0
    return (base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).cpu().numpy(), cell
def topk_region(drop,cell,k,G=4):
    top=np.argsort(-drop)[:k]; reg=np.zeros((224,224),bool)
    for j in top:
        r,cc=divmod(int(j),G); reg[r*cell:(r+1)*cell,cc*cell:(cc+1)*cell]=True
    return reg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=60); args=ap.parse_args()
    ds=torchvision.datasets.STL10("data",split="test",download=False)
    idx=list(np.random.default_rng(0).permutation(len(ds))[:args.n]); trng=np.random.default_rng(1)
    setups=["unilingual","multilingual"]; clfs={"EN":(en_m,EN_T,en_p),"ZH":(zh_m,ZH_T,zh_p)}
    defs=["no_def","attn2","attn4","occ1","occ2"]; cost={"no_def":1,"attn2":3,"attn4":5,"occ1":18,"occ2":18}
    acc={s:{cl:{d:0 for d in defs} for cl in clfs} for s in setups}
    for i in idx:
        true=ds[i][1]; tgt=trng.choice([c for c in range(10) if c!=true])
        for s in setups:
            prng=np.random.default_rng(5000+i)                  # SAME locations across setups
            img,boxes=make_attack(ds[i][0],tgt,prng,multilingual=(s=="multilingual"))
            en_pv=pvof(en_p,img); zh_pv=pvof(zh_p,img)
            # attention maps (shared localization)
            r_en=rollout_map(en_m,en_pv); r_zh=rollout_map(zh_m,zh_pv)
            reg2=region_from(combine([r_en,r_zh]))
            g_en=gradcam_map(en_m,en_pv,EN_T); g_zh=gradcam_map(zh_m,zh_pv,ZH_T)
            reg4=region_from(combine([r_en,r_zh,g_en,g_zh]))
            for cl,(m,T,p) in clfs.items():
                pv=en_pv if cl=="EN" else zh_pv
                acc[s][cl]["no_def"]+= (predict(m,pv,T)==true)
                acc[s][cl]["attn2"]+= (predict(m,maskpv(pv,reg2),T)==true)
                acc[s][cl]["attn4"]+= (predict(m,maskpv(pv,reg4),T)==true)
                drop,cell=occ_drop(m,pv,T)
                acc[s][cl]["occ1"]+= (predict(m,maskpv(pv,topk_region(drop,cell,1)),T)==true)
                acc[s][cl]["occ2"]+= (predict(m,maskpv(pv,topk_region(drop,cell,2)),T)==true)
    n=args.n
    print(f"=== Multilingual vs unilingual (2 random typographic attacks), n={n} ===")
    print("robust accuracy (%) = correct-class rate under attack, per defense\n")
    for s in setups:
        print(f"[{s} attack]"); print(f"{'clf':>4} | " + " ".join(f"{d}({cost[d]})".rjust(10) for d in defs))
        for cl in clfs:
            print(f"{cl:>4} | " + " ".join(f"{100*acc[s][cl][d]/n:9.0f}%" for d in defs))
        print()
    # graph
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    for ax,s in zip(axes,setups):
        for cl,col in [("EN","tab:blue"),("ZH","tab:red")]:
            xs=[cost[d] for d in defs]; ys=[100*acc[s][cl][d]/n for d in defs]
            ax.scatter(xs,ys,c=col);
            for d in defs: ax.annotate(d,(cost[d],100*acc[s][cl][d]/n),fontsize=6,color=col,textcoords="offset points",xytext=(3,3))
            # connect attn line and occ line
            for grp in [["no_def","attn2","attn4"],["occ1","occ2"]]:
                ax.plot([cost[d] for d in grp],[100*acc[s][cl][d]/n for d in grp],"-",c=col,alpha=.5,label=cl if grp[0]=="no_def" else None)
        ax.set_title(f"{s} attack"); ax.set_xscale("log"); ax.set_xlabel("inference cost (fwd passes, log)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("robust accuracy (%)"); axes[0].legend(title="classifier"); axes[0].set_ylim(0,100)
    plt.suptitle("Inference cost vs robust accuracy: attention-modulation vs occlusion defense")
    plt.tight_layout(); plt.savefig("results/multilingual_defense.png",dpi=125); print("saved results/multilingual_defense.png")

if __name__=="__main__": main()
