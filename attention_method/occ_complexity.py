"""2 random typographic attacks: OURS (attention) vs occlusion GRID SEARCH patch-selection.
  grid-1patch  : mask the single best cell         -> n = G^2 = 16 sweeps  (linear)
  grid-2patch  : mask the best PAIR of cells        -> C(n,2) = 120 sweeps  (QUADRATIC)
  greedy-2patch: mask best cell, re-occlude, repeat -> 2n = 32 sweeps       (linear, OURS)
  attn-2map    : combine EN+ZH attention -> mask     -> ~3 forwards          (OURS, cheap)
Selection criterion is unsupervised: mask the cell(s) whose removal most DROPS the model's
current (attacked) predicted-class score. Report robust accuracy under the 2-box attack.
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device="cuda"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; CJK="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
EN_W=["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W=["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]
PADX,PADY=8,10; G=4; CELL=224//G
def _bbox(w,s,cjk):
    f=ImageFont.truetype(CJK if cjk else FONT,s); d=ImageDraw.Draw(Image.new("RGB",(224,224)))
    bb=d.textbbox((0,0),w,font=f); return bb[2]-bb[0],bb[3]-bb[1],f
def put(d,w,s,cx,cy,cjk):
    tw,th,f=_bbox(w,s,cjk); x=int(cx-tw/2); y=int(cy-th/2); box=(x-PADX,y-PADY,x+tw+PADX,y+th+PADY)
    d.rectangle(box,fill=(255,255,255)); bb=d.textbbox((0,0),w,font=f); d.text((x-bb[0],y-bb[1]),w,fill=(0,0,0),font=f); return box
def rc(bw,bh,rng,m=4): return int(rng.integers(bw//2+m,max(bw//2+m+1,224-bw//2-m))),int(rng.integers(bh//2+m,max(bh//2+m+1,224-bh//2-m)))
def ov(a,b,p=4): return not(a[2]+p<=b[0] or b[2]+p<=a[0] or a[3]+p<=b[1] or b[3]+p<=a[1])
def make_attack(pil,tgt,rng,multi,s=40):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    w1,h1,_=_bbox(EN_W[tgt],s,False); w2,h2,_=_bbox(ZH_W[tgt] if multi else EN_W[tgt],s,multi)
    bw1,bh1,bw2,bh2=w1+2*PADX,h1+2*PADY,w2+2*PADX,h2+2*PADY
    c1=rc(bw1,bh1,rng); b1=(c1[0]-bw1//2,c1[1]-bh1//2,c1[0]+bw1//2,c1[1]+bh1//2)
    for _ in range(300):
        c2=rc(bw2,bh2,rng); b2=(c2[0]-bw2//2,c2[1]-bh2//2,c2[0]+bw2//2,c2[1]+bh2//2)
        if not ov(b1,b2): break
    put(d,EN_W[tgt],s,c1[0],c1[1],False); put(d,(ZH_W[tgt] if multi else EN_W[tgt]),s,c2[0],c2[1],multi)
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
def predict(m,pv,T): return (feats(m,pv)@T.t()).argmax(-1).item()
def cellreg(j): r,c=divmod(int(j),G); m=np.zeros((224,224),bool); m[r*CELL:(r+1)*CELL,c*CELL:(c+1)*CELL]=True; return m
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
@torch.no_grad()
def single_drops(m,pv,T):                      # 16 fwd (batched)
    c=predict(m,pv,T); base=(feats(m,pv)@T[c:c+1].t()).item(); occ=pv.repeat(G*G,1,1,1)
    for j in range(G*G):
        r,cc=divmod(j,G); occ[j,:,r*CELL:(r+1)*CELL,cc*CELL:(cc+1)*CELL]=0
    return (base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).cpu().numpy()
@torch.no_grad()
def pair_drops(m,pv,T):                         # C(16,2)=120 fwd (batched) — QUADRATIC
    c=predict(m,pv,T); base=(feats(m,pv)@T[c:c+1].t()).item(); pairs=list(combinations(range(G*G),2))
    occ=pv.repeat(len(pairs),1,1,1)
    for i,(a,b) in enumerate(pairs):
        for j in (a,b):
            r,cc=divmod(j,G); occ[i,:,r*CELL:(r+1)*CELL,cc*CELL:(cc+1)*CELL]=0
    d=(base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).cpu().numpy(); return pairs,d
def rollout(att):
    S=att[0].shape[-1]; R=torch.eye(S,device=att[0].device)[None]
    for A in att:
        A=A.mean(1)+torch.eye(S,device=A.device); A=A/A.sum(-1,keepdim=True); R=A@R
    cls=R[:,0,1:]; g=int(round(cls.shape[1]**0.5)); s=F.interpolate(cls.reshape(1,1,g,g),size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/(s.max()+1e-9)).detach().cpu().numpy()
@torch.no_grad()
def rollout_map(m,pv): return rollout(m.vision_model(pixel_values=pv,output_attentions=True).attentions)

def defenses(m,pv,T,r_en,r_zh):
    # ours attention 2-map
    comb=r_en/(r_en.max()+1e-9)+r_zh/(r_zh.max()+1e-9); attn_reg=comb>=0.5*comb.max()
    # grid 1-patch
    d1=single_drops(m,pv,T); reg1=cellreg(d1.argmax())
    # greedy 2-patch (linear): mask best, re-occlude, mask best again
    pv2=maskpv(pv,reg1); d2=single_drops(m,pv2,T); regg=reg1|cellreg(d2.argmax())
    # exhaustive 2-patch (quadratic)
    pairs,dp=pair_drops(m,pv,T); a,b=pairs[int(dp.argmax())]; rege=cellreg(a)|cellreg(b)
    return {"attn2":attn_reg,"grid1":reg1,"greedy2":regg,"grid2ex":rege}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=80); args=ap.parse_args()
    ds=torchvision.datasets.STL10("data",split="test",download=False)
    idx=list(np.random.default_rng(0).permutation(len(ds))[:args.n]); trng=np.random.default_rng(1)
    setups=["unilingual","multilingual"]; clfs={"EN":(en_m,EN_T),"ZH":(zh_m,ZH_T)}
    order=["no_def","attn2","grid1","greedy2","grid2ex"]; cost={"no_def":1,"attn2":3,"grid1":17,"greedy2":33,"grid2ex":121}
    acc={s:{cl:{d:0 for d in order} for cl in clfs} for s in setups}
    for i in idx:
        true=ds[i][1]; tgt=trng.choice([c for c in range(10) if c!=true])
        for s in setups:
            prng=np.random.default_rng(5000+i)
            img=make_attack(ds[i][0],tgt,prng,multi=(s=="multilingual"))
            en_pv=pvof(en_p,img); zh_pv=pvof(zh_p,img)
            r_en=rollout_map(en_m,en_pv); r_zh=rollout_map(zh_m,zh_pv)
            for cl,(m,T) in clfs.items():
                pv=en_pv if cl=="EN" else zh_pv
                acc[s][cl]["no_def"]+=(predict(m,pv,T)==true)
                regs=defenses(m,pv,T,r_en,r_zh)
                for d,reg in regs.items():
                    acc[s][cl][d]+=(predict(m,maskpv(pv,reg),T)==true)
    n=args.n
    print(f"=== 2 random typographic attacks: OURS vs occlusion grid-search patch selection (n={n}) ===\n")
    for s in setups:
        print(f"[{s} attack]  robust accuracy %  (cost = fwd passes)")
        print(f"{'clf':>4} | "+" ".join(f"{d}({cost[d]})".rjust(13) for d in order))
        for cl in clfs: print(f"{cl:>4} | "+" ".join(f"{100*acc[s][cl][d]/n:12.0f}%" for d in order))
        print()
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    mk={"no_def":"o","attn2":"s","grid1":"^","greedy2":"D","grid2ex":"*"}
    for ax,s in zip(axes,setups):
        for cl,col in [("EN","tab:blue"),("ZH","tab:red")]:
            for d in order:
                ax.scatter(cost[d],100*acc[s][cl][d]/n,marker=mk[d],c=col,s=90 if d=="grid2ex" else 55,zorder=5)
            ax.plot([cost[d] for d in order],[100*acc[s][cl][d]/n for d in order],"-",c=col,alpha=.35,label=cl)
        for d in order: ax.annotate(d,(cost[d],2),rotation=90,fontsize=6,ha="center",va="bottom",color="gray")
        ax.set_title(f"{s} attack"); ax.set_xscale("log"); ax.set_xlabel("inference cost (fwd passes, log)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("robust accuracy (%)"); axes[0].legend(title="classifier"); axes[0].set_ylim(0,100)
    plt.suptitle("2-box attack: ours (attention, greedy) vs occlusion grid search (1-patch=16, 2-patch=120 QUADRATIC)")
    plt.tight_layout(); plt.savefig("results/occ_complexity.png",dpi=125); print("saved results/occ_complexity.png")

if __name__=="__main__": main()
