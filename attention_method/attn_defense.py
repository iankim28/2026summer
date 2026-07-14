"""Decisive test of the 'locate typographic attack -> neglect region -> second pass' idea.

Localizer = OCCLUSION/counterfactual (mask a cell, measure drop in the predicted-class
score) -- reliable, and it IS the defense mechanism. Two models: EN (OpenAI CLIP) + ZH
(Chinese-CLIP); test single-model and EN&ZH-OVERLAP localization.

Reports: no-defense acc/ASR, ORACLE-mask (known box) acc [ceiling], occlusion-localized-mask
acc (EN, and EN&ZH overlap), and localization quality (in-box focus ratio, IoU with true box).
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device="cuda"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
EN_W=["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W=["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]
G=7; CELL=32  # 7x7 grid of 32px cells

def draw(pil, word, size=40):
    img=pil.convert("RGB").resize((224,224),Image.BICUBIC); d=ImageDraw.Draw(img)
    f=ImageFont.truetype(FONT,size); bb=d.textbbox((0,0),word,font=f)
    w,h=bb[2]-bb[0],bb[3]-bb[1]; x=(224-w)//2; y=224-h-16; box=(x-8,y-8,x+w+8,y+h+12)
    d.rectangle(box,fill=(255,255,255)); d.text((x-bb[0],y-bb[1]),word,fill=(0,0,0),font=f)
    return img, box

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
def pv_of(proc,pil): return proc(images=[pil],return_tensors="pt").pixel_values.to(device)
@torch.no_grad()
def feats(m,pv): return F.normalize(m.get_image_features(pixel_values=pv),dim=-1)
@torch.no_grad()
def predict(m,pv,TXT): return (feats(m,pv)@TXT.t()).argmax(-1).item()

@torch.no_grad()
def occ_saliency(m,pv0,TXT):
    """drop in predicted-class cosine when each 32px cell is masked -> [G,G], + pred + 224 map."""
    c=predict(m,pv0,TXT); base=(feats(m,pv0)@TXT[c:c+1].t()).item()
    occ=pv0.repeat(G*G,1,1,1)
    for k in range(G*G):
        r,cc=divmod(k,G); occ[k,:,r*CELL:(r+1)*CELL, cc*CELL:(cc+1)*CELL]=0
    sc=(feats(m,occ)@TXT[c:c+1].t()).squeeze(-1)          # [49]
    drop=(base-sc).reshape(G,G).clamp(min=0)
    m224=F.interpolate(drop[None,None],size=(224,224),mode="nearest")[0,0]
    return drop.cpu().numpy(), c, m224.cpu().numpy()

def mask_pv(pv,region224):   # region224: bool [224,224] to zero out
    pv=pv.clone(); mask=torch.tensor(region224,device=pv.device)
    pv[:,:,mask]=0; return pv

def box_mask(box):
    m=np.zeros((224,224),bool); x0,y0,x1,y1=box; m[y0:y1,x0:x1]=True; return m
def inbox_ratio(s,box):
    m=box_mask(box); return (s[m].sum()/(s.sum()+1e-9))/(m.mean()+1e-9)
def thr_region(s224,frac=0.5):   # cells with drop >= frac*max
    return s224 >= frac*(s224.max()+1e-9)
def iou(a,b): return (a&b).sum()/((a|b).sum()+1e-9)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100); args=ap.parse_args()
    ds=torchvision.datasets.STL10("data",split="test",download=False)
    idx=list(np.random.default_rng(0).permutation(len(ds))[:args.n]); rng=np.random.default_rng(1)
    S={k:0 for k in ["en_att_true","en_att_asr","en_oracle","en_occ","en_overlap",
                     "zh_att_true","zh_occ","enzh_agree_att","enzh_agree_def_overlap",
                     "en_clean","en_clean_def","en_clean_changed"]}
    inbox_en=[]; inbox_zh=[]; iou_en=[]; iou_overlap=[]; n=0
    for i in idx:
        true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true]); n+=1
        att,box=draw(ds[i][0],EN_W[tgt]); bm=box_mask(box)
        en_pv=pv_of(en_p,att); zh_pv=pv_of(zh_p,att)
        # CLEAN-image cost of blindly applying localize+mask (no attack present)
        clean_pil=ds[i][0].convert("RGB").resize((224,224),Image.BICUBIC); cpv=pv_of(en_p,clean_pil)
        cpred=predict(en_m,cpv,EN_T); S["en_clean"]+=(cpred==true)
        _,_,cs224=occ_saliency(en_m,cpv,EN_T); cdef=predict(en_m,mask_pv(cpv,thr_region(cs224)),EN_T)
        S["en_clean_def"]+=(cdef==true); S["en_clean_changed"]+=(cdef!=cpred)
        # no-defense
        en_pred=predict(en_m,en_pv,EN_T); zh_pred=predict(zh_m,zh_pv,ZH_T)
        S["en_att_true"]+= (en_pred==true); S["en_att_asr"]+= (en_pred==tgt)
        S["zh_att_true"]+= (zh_pred==true)
        S["enzh_agree_att"]+= (en_pred==zh_pred)
        # oracle mask (known box)
        S["en_oracle"]+= (predict(en_m,mask_pv(en_pv,bm),EN_T)==true)
        # occlusion localize (each model)
        en_s,_,en_s224=occ_saliency(en_m,en_pv,EN_T); zh_s,_,zh_s224=occ_saliency(zh_m,zh_pv,ZH_T)
        inbox_en.append(inbox_ratio(en_s224,box)); inbox_zh.append(inbox_ratio(zh_s224,box))
        en_reg=thr_region(en_s224); iou_en.append(iou(en_reg,bm))
        S["en_occ"]+= (predict(en_m,mask_pv(en_pv,en_reg),EN_T)==true)
        S["zh_occ"]+= (predict(zh_m,mask_pv(zh_pv,thr_region(zh_s224)),ZH_T)==true)
        # EN&ZH overlap localization (normalize each, take min, threshold)
        en_n=en_s224/(en_s224.max()+1e-9); zh_n=zh_s224/(zh_s224.max()+1e-9)
        over=np.minimum(en_n,zh_n); over_reg=over>=0.5*(over.max()+1e-9)
        iou_overlap.append(iou(over_reg,bm))
        S["en_overlap"]+= (predict(en_m,mask_pv(en_pv,over_reg),EN_T)==true)
        en_def=predict(en_m,mask_pv(en_pv,over_reg),EN_T); zh_def=predict(zh_m,mask_pv(zh_pv,over_reg),ZH_T)
        S["enzh_agree_def_overlap"]+= (en_def==zh_def)
    p=lambda k:100*S[k]/n
    print(f"=== Attention/occlusion localization defense (n={n}, English typographic attack) ===\n")
    print("NO DEFENSE:")
    print(f"  EN acc={p('en_att_true'):.1f}%  EN ASR(->written)={p('en_att_asr'):.1f}%   ZH acc={p('zh_att_true'):.1f}%")
    print(f"  EN==ZH agreement (attacked): {p('enzh_agree_att'):.1f}%")
    print("\nDEFENSE (mask region -> re-classify):")
    print(f"  EN, ORACLE mask (known box)         : {p('en_oracle'):.1f}%   <- ceiling")
    print(f"  EN, occlusion-localized mask        : {p('en_occ'):.1f}%")
    print(f"  EN, EN&ZH-OVERLAP-localized mask    : {p('en_overlap'):.1f}%")
    print(f"  ZH, occlusion-localized mask        : {p('zh_occ'):.1f}%")
    print(f"  EN==ZH agreement after overlap-def  : {p('enzh_agree_def_overlap'):.1f}%")
    print("\nCLEAN-IMAGE COST (blindly masking the top region when NO attack is present):")
    print(f"  EN clean acc (no defense)={p('en_clean'):.1f}%  ->  after localize+mask={p('en_clean_def'):.1f}%"
          f"   (prediction changed on {p('en_clean_changed'):.1f}% of clean imgs)")
    print("\nLOCALIZATION QUALITY:")
    print(f"  occlusion in-box focus ratio  EN={np.mean(inbox_en):.2f}  ZH={np.mean(inbox_zh):.2f}  (>1 concentrated on text)")
    print(f"  IoU(localized region, true box)  EN={np.mean(iou_en):.2f}  OVERLAP={np.mean(iou_overlap):.2f}")

if __name__=="__main__": main()
