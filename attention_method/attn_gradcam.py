"""Gradient-based localization of the typographic attack (the 'right' version of the
attention idea): saliency = |d cos(image, PREDICTED-class text) / d pixels|, smoothed.
Test whether EN + ZH gradient saliency concentrates on the text box, and whether their
OVERLAP localizes it robustly."""
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device = "cuda"; FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
EN_W = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
ZH_W = ["飞机","鸟","汽车","猫","鹿","狗","马","猴子","船","卡车"]

def draw(pil, word, size=40):
    img = pil.convert("RGB").resize((224,224), Image.BICUBIC); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size); bb = d.textbbox((0,0), word, font=f)
    w,h = bb[2]-bb[0], bb[3]-bb[1]; x=(224-w)//2; y=224-h-16; box=(x-8,y-8,x+w+8,y+h+12)
    d.rectangle(box, fill=(255,255,255)); d.text((x-bb[0],y-bb[1]), word, fill=(0,0,0), font=f)
    return img, box

en_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
en_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
zh_m = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
zh_p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

@torch.no_grad()
def text_feats(model, proc, words, is_zh):
    t = proc(text=[("一张%s的照片" if is_zh else "a photo of a %s.") % w for w in words],
             padding=True, return_tensors="pt").to(device)
    return F.normalize(model.get_text_features(**t), dim=-1)
EN_T = text_feats(en_m, en_p, EN_W, False); ZH_T = text_feats(zh_m, zh_p, ZH_W, True)

def img_feats(model, pv):
    return F.normalize(model.get_image_features(pixel_values=pv), dim=-1)

def smoothgrad_saliency(model, proc, pil, TXT, nsamp=12, sigma=0.08):
    pv0 = proc(images=[pil], return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        pred = (img_feats(model, pv0) @ TXT.t()).argmax(-1).item()
    sal = torch.zeros(pv0.shape[-2:], device=device)
    for _ in range(nsamp):
        pv = (pv0 + sigma*torch.randn_like(pv0)).detach().requires_grad_(True)
        s = (img_feats(model, pv) @ TXT[pred:pred+1].t()).sum()
        g = torch.autograd.grad(s, pv)[0][0]           # [3,H,W]
        sal += g.abs().sum(0)
    sal = sal / sal.max()
    # upsample to 224 if model input != 224 (both are 224 here) then to numpy
    return F.interpolate(sal[None,None], size=(224,224), mode="bilinear", align_corners=False)[0,0].cpu().numpy(), pred

def inbox_ratio(sal, box):
    x0,y0,x1,y1 = box; m=np.zeros((224,224),bool); m[y0:y1,x0:x1]=True
    return (sal[m].sum()/(sal.sum()+1e-9))/(m.mean()+1e-9)

ds = torchvision.datasets.STL10("data", split="test", download=False)
idx = list(np.random.default_rng(0).permutation(len(ds))[:4]); rng=np.random.default_rng(1)
fig, axes = plt.subplots(4,3, figsize=(10,13))
for r,i in enumerate(idx):
    true=ds[i][1]; tgt=rng.choice([c for c in range(10) if c!=true])
    att_img, box = draw(ds[i][0], EN_W[tgt])
    en_sal, en_pred = smoothgrad_saliency(en_m, en_p, att_img, EN_T)
    zh_sal, zh_pred = smoothgrad_saliency(zh_m, zh_p, att_img, ZH_T)
    over = np.minimum(en_sal, zh_sal)
    er,zr,orr = inbox_ratio(en_sal,box), inbox_ratio(zh_sal,box), inbox_ratio(over,box)
    print(f"img{r}: wrote '{EN_W[tgt]}' | EN pred={EN_W[en_pred]} ZH pred={ZH_W[zh_pred]} | in-box ratio EN={er:.2f} ZH={zr:.2f} OVERLAP={orr:.2f}")
    base=np.array(att_img).astype(float)/255
    for c,(sal,nm) in enumerate([(en_sal,f"EN r={er:.1f}"),(zh_sal,f"ZH r={zr:.1f}"),(over,f"overlap r={orr:.1f}")]):
        ax=axes[r,c]; ax.imshow(base); ax.imshow(sal,cmap="jet",alpha=0.5)
        ax.add_patch(plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor="lime",lw=2))
        ax.set_title(nm,fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("results/attn_gradcam.png", dpi=110); print("saved results/attn_gradcam.png")
