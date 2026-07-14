"""SINGLE vs DOUBLE typographic attack at MATCHED total text area.
Double = two smaller boxes of the same target word (top + bottom), each ~half the area of
the single box, so total covered area is ~equal. Tests whether spreading the attack evades
the localize-and-mask defense (occlusion / attention-overlap).
Records performance + saves visualizations."""
import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device="cuda"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
EN_W=["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W=["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]
PADX,PADY=8,10
def _bbox(word,size):
    f=ImageFont.truetype(FONT,size); im=Image.new("RGB",(224,224)); d=ImageDraw.Draw(im)
    bb=d.textbbox((0,0),word,font=f); return bb[2]-bb[0], bb[3]-bb[1], f
def box_area(word,size):
    w,h,_=_bbox(word,size); return (w+2*PADX)*(h+2*PADY)
def find_size(word,target_area,hi=44):
    best=(1e18,10)
    for s in range(10,hi+1):
        a=box_area(word,s); best=min(best,(abs(a-target_area),s))
    return best[1]
def put(d,word,size,cx,cy):
    w,h,f=_bbox(word,size); x=int(cx-w/2); y=int(cy-h/2)
    box=(x-PADX,y-PADY,x+w+PADX,y+h+PADY)
    d.rectangle(box,fill=(255,255,255)); bb=d.textbbox((0,0),word,font=f)
    d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f); return box
def _rand_center(bw,bh,rng,m=4):
    cx=int(rng.integers(bw//2+m, max(bw//2+m+1, 224-bw//2-m)))
    cy=int(rng.integers(bh//2+m, max(bh//2+m+1, 224-bh//2-m)))
    return cx,cy
def _overlap(a,b,pad=4):
    return not (a[2]+pad<=b[0] or b[2]+pad<=a[0] or a[3]+pad<=b[1] or b[3]+pad<=a[1])
def draw_single(pil,word,size,rng):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    w,h,_=_bbox(word,size); bw,bh=w+2*PADX,h+2*PADY
    cx,cy=_rand_center(bw,bh,rng); box=put(d,word,size,cx,cy); return img,[box]
def draw_double(pil,word,size,rng):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    w,h,_=_bbox(word,size); bw,bh=w+2*PADX,h+2*PADY
    cx1,cy1=_rand_center(bw,bh,rng); b1=put(d,word,size,cx1,cy1)
    for _ in range(200):                              # non-overlapping 2nd box
        cx2,cy2=_rand_center(bw,bh,rng); cand=(int(cx2-w/2)-PADX,int(cy2-h/2)-PADY,int(cx2+w/2)+PADX,int(cy2+h/2)+PADY)
        if not _overlap(b1,cand): break
    b2=put(d,word,size,cx2,cy2); return img,[b1,b2]

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
@torch.no_grad()
def occ_sal(m,pv0,T,G=4):
    cell=224//G; c=predict(m,pv0,T); base=(feats(m,pv0)@T[c:c+1].t()).item()
    occ=pv0.repeat(G*G,1,1,1)
    for k in range(G*G):
        r,cc=divmod(k,G); occ[k,:,r*cell:(r+1)*cell,cc*cell:(cc+1)*cell]=0
    drop=(base-(feats(m,occ)@T[c:c+1].t()).squeeze(-1)).reshape(G,G).clamp(min=0)
    return F.interpolate(drop[None,None],size=(224,224),mode="nearest")[0,0].cpu().numpy()
def rollout(att):
    S=att[0].shape[-1]; R=torch.eye(S,device=att[0].device)[None]
    for A in att:
        A=A.mean(1)+torch.eye(S,device=A.device); A=A/A.sum(-1,keepdim=True); R=A@R
    cls=R[:,0,1:]; g=int(round(cls.shape[1]**0.5))
    s=F.interpolate(cls.reshape(1,1,g,g),size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return (s/s.max()).cpu().numpy()
@torch.no_grad()
def attn_ov(en_pv,zh_pv):
    es=rollout(en_m.vision_model(pixel_values=en_pv,output_attentions=True).attentions)
    zs=rollout(zh_m.vision_model(pixel_values=zh_pv,output_attentions=True).attentions)
    return np.minimum(es,zs)
def maskpv(pv,reg):
    pv=pv.clone(); pv[:,:,torch.tensor(reg,device=pv.device)]=0; return pv
def boxes_mask(boxes):
    m=np.zeros((224,224),bool)
    for (x0,y0,x1,y1) in boxes: m[max(0,y0):min(224,y1),max(0,x0):min(224,x1)]=True
    return m
def box_cov(reg,box):   # fraction of a single box covered by masked region
    m=boxes_mask([box]); return (reg&m).sum()/(m.sum()+1e-9)

def evaluate(n):
    ds=torchvision.datasets.STL10("data",split="test",download=False)
    idx=list(np.random.default_rng(0).permutation(len(ds))[:n]); rng=np.random.default_rng(1)
    R={c:{"asr":0,"acc":0,"occ":0,"attn":0,"area":0,"cov_both":0,"nbox":0} for c in ["single","double"]}
    for i in idx:
        true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true])
        a1=box_area(EN_W[tgt],44); s2=find_size(EN_W[tgt],a1/2); prng=np.random.default_rng(1000+i)
        for cond,(img,boxes) in [("single",draw_single(ds[i][0],EN_W[tgt],44,prng)),
                                 ("double",draw_double(ds[i][0],EN_W[tgt],s2,prng))]:
            en_pv=pvof(en_p,img); zh_pv=pvof(zh_p,img)
            R[cond]["area"]+=boxes_mask(boxes).sum(); R[cond]["nbox"]+=len(boxes)
            p0=predict(en_m,en_pv,EN_T); R[cond]["acc"]+=(p0==true); R[cond]["asr"]+=(p0==tgt)
            s=occ_sal(en_m,en_pv,EN_T,4); reg=s>=0.5*(s.max()+1e-9)
            R[cond]["occ"]+=(predict(en_m,maskpv(en_pv,reg),EN_T)==true)
            R[cond]["cov_both"]+= int(all(box_cov(reg,b)>0.3 for b in boxes))  # both boxes masked?
            o=attn_ov(en_pv,zh_pv); oreg=o>=0.5*(o.max()+1e-9)
            R[cond]["attn"]+=(predict(en_m,maskpv(en_pv,oreg),EN_T)==true)
    print(f"=== Single vs Double typographic attack (matched area), n={n} ===")
    print(f"{'cond':>7} | {'#box':>4} {'area px2':>8} | {'ASR':>5} {'no-def acc':>10} | {'occ4x4 acc':>10} {'boxes-both-masked':>17} | {'attn-ov acc':>11}")
    for c in ["single","double"]:
        d=R[c]; pc=lambda k:100*d[k]/n
        print(f"{c:>7} | {d['nbox']/n:>4.1f} {d['area']/n:>8.0f} | {pc('asr'):>4.0f}% {pc('acc'):>9.0f}% | {pc('occ'):>9.0f}% {pc('cov_both'):>16.0f}% | {pc('attn'):>10.0f}%")

def visualize(k=4):
    ds=torchvision.datasets.STL10("data",split="test",download=False)
    idx=list(np.random.default_rng(7).permutation(len(ds))[:k]); rng=np.random.default_rng(3)
    fig,axes=plt.subplots(k,4,figsize=(12,3*k))
    col=["single: attacked","single: occ-mask","double: attacked","double: occ-mask"]
    for r,i in enumerate(idx):
        true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true])
        a1=box_area(EN_W[tgt],44); s2=find_size(EN_W[tgt],a1/2); prng=np.random.default_rng(2000+i)
        for base_c,(img,boxes) in [(0,draw_single(ds[i][0],EN_W[tgt],44,prng)),(2,draw_double(ds[i][0],EN_W[tgt],s2,prng))]:
            en_pv=pvof(en_p,img); s=occ_sal(en_m,en_pv,EN_T,4); reg=s>=0.5*(s.max()+1e-9)
            p0=EN_W[predict(en_m,en_pv,EN_T)]; pd=EN_W[predict(en_m,maskpv(en_pv,reg),EN_T)]
            base=np.array(img).astype(float)/255
            ax=axes[r,base_c]; ax.imshow(base)
            for (x0,y0,x1,y1) in boxes: ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="lime",lw=1.5))
            ax.set_title(f"{col[base_c]}\npred={p0}",fontsize=8); ax.axis("off")
            ax2=axes[r,base_c+1]; ax2.imshow(base); ax2.imshow(s,cmap="jet",alpha=0.45)
            # show masked region outline
            ys,xs=np.where(reg)
            if len(xs): ax2.add_patch(plt.Rectangle((xs.min(),ys.min()),xs.max()-xs.min(),ys.max()-ys.min(),fill=False,edgecolor="white",lw=1.5,ls="--"))
            ax2.set_title(f"{col[base_c+1]}\nafter mask -> {pd}",fontsize=8); ax2.axis("off")
    plt.suptitle("Single vs Double (matched-area) typographic attack + occlusion defense",fontsize=11)
    plt.tight_layout(); plt.savefig("results/multi_typo_vis.png",dpi=120); print("saved results/multi_typo_vis.png")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100); args=ap.parse_args()
    evaluate(args.n); visualize(4)
