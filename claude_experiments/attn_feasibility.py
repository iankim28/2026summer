"""Feasibility: do EN (OpenAI CLIP) and ZH (Chinese-CLIP) visual-attention maps concentrate
on the typographic-text region? Extract attention rollout, overlay, and report in-box mass."""
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

device = "cuda"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
EN_W = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

def draw(pil, word, size=40):
    img = pil.convert("RGB").resize((224,224), Image.BICUBIC); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size); bb = d.textbbox((0,0), word, font=f)
    w,h = bb[2]-bb[0], bb[3]-bb[1]; x=(224-w)//2; y=224-h-16
    box = (x-8, y-8, x+w+8, y+h+12)
    d.rectangle(box, fill=(255,255,255)); d.text((x-bb[0], y-bb[1]), word, fill=(0,0,0), font=f)
    return img, box   # box in 224-space (x0,y0,x1,y1)

def rollout(attentions):
    # attentions: tuple of [B,heads,S,S]
    S = attentions[0].shape[-1]; B = attentions[0].shape[0]
    R = torch.eye(S, device=attentions[0].device).unsqueeze(0).repeat(B,1,1)
    for A in attentions:
        A = A.mean(1) + torch.eye(S, device=A.device)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    cls = R[:, 0, 1:]                      # [B, num_patches]
    g = int(round(cls.shape[1] ** 0.5))
    sal = cls.reshape(B, 1, g, g)
    sal = F.interpolate(sal, size=(224,224), mode="bilinear", align_corners=False)[:, 0]
    sal = sal / sal.amax(dim=(1,2), keepdim=True)
    return sal.cpu().numpy()               # [B,224,224] in [0,1]

# models (transformers -> easy output_attentions)
en_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
en_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
zh_m = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
zh_p = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

@torch.no_grad()
def saliency(model, proc, imgs):
    pv = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
    att = model.vision_model(pixel_values=pv, output_attentions=True).attentions
    return rollout(att), pv.shape[-1]

def inbox_ratio(sal, box):
    x0,y0,x1,y1 = box; m = np.zeros((224,224), bool); m[y0:y1, x0:x1] = True
    area_frac = m.mean()
    return (sal[m].sum() / (sal.sum()+1e-9)) / (area_frac+1e-9)   # >1 => over-attended

ds = torchvision.datasets.STL10("data", split="test", download=False)
idx = list(np.random.default_rng(0).permutation(len(ds))[:4])
rng = np.random.default_rng(1)
fig, axes = plt.subplots(4, 3, figsize=(10, 13))
for r, i in enumerate(idx):
    true = ds[i][1]; tgt = rng.choice([c for c in range(10) if c != true])
    att_img, box = draw(ds[i][0], EN_W[tgt])
    en_sal,_ = saliency(en_m, en_p, [att_img]); zh_sal,_ = saliency(zh_m, zh_p, [att_img])
    en_sal, zh_sal = en_sal[0], zh_sal[0]
    en_r = inbox_ratio(en_sal, box); zh_r = inbox_ratio(zh_sal, box)
    over = np.minimum(en_sal, zh_sal)   # attention OVERLAP (min = both attend)
    ov_r = inbox_ratio(over, box)
    print(f"img{r}: wrote '{EN_W[tgt]}'  in-box focus ratio  EN={en_r:.2f}  ZH={zh_r:.2f}  OVERLAP={ov_r:.2f}  (>1 = concentrated on text)")
    base = np.array(att_img).astype(float)/255
    for c,(sal,name) in enumerate([(en_sal,f"EN r={en_r:.1f}"),(zh_sal,f"ZH r={zh_r:.1f}"),(over,f"overlap r={ov_r:.1f}")]):
        ax = axes[r,c]; ax.imshow(base); ax.imshow(sal, cmap="jet", alpha=0.5)
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="lime", lw=2))
        ax.set_title(name, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("results/attn_feasibility.png", dpi=110)
print("saved results/attn_feasibility.png")
