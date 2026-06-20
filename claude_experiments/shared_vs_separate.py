"""SHARED vs SEPARATE encoders under the same (English) typographic attack.

Tests the original proposal's premise directly: does cross-lingual DISAGREEMENT appear
when the image encoders are NOT shared?

  SHARED   : one multilingual CLIP (xlm-roberta-base-ViT-B-32); 4 language label sets
             share ONE image embedding.
  SEPARATE : 4 independently-trained per-language CLIPs (en=OpenAI, zh=Chinese-CLIP,
             ko=Bingsu, ja=line-corp); each has its OWN image encoder.

Attack: write the English target word on the image (the strong typographic attacker).
Metrics (clean vs attacked, both conditions): per-language accuracy/ASR, all-4 agreement,
majority-vote ensemble accuracy, and disagreement-detector ROC-AUC.
"""
import json, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
from mclip_lib import load_model, encode_image, logits_for, get_logit_scale
from perlang_models import EnCLIP, ZhCLIP, KoCLIP, JaCLIP, classify, CLASSES, TMPL

LANGS4 = ["en", "zh", "ko", "ja"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
device = "cuda"
rng = random.Random(0)

def draw_en(pil, word, size=40):
    img = pil.convert("RGB").resize((224, 224), Image.BICUBIC); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size); bb = d.textbbox((0, 0), word, font=f)
    w, h = bb[2]-bb[0], bb[3]-bb[1]; x = (224-w)//2; y = 224-h-16
    d.rectangle([x-8, y-8, x+w+8, y+h+12], fill=(255, 255, 255))
    d.text((x-bb[0], y-bb[1]), word, fill=(0, 0, 0), font=f)
    return img

# ---- data ----
ds = torchvision.datasets.STL10("data", split="test", download=False)
idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[:200]
true = np.array([ds[i][1] for i in idx])
target = np.array([rng.choice([c for c in range(10) if c != true[k]]) for k in range(len(idx))])
clean_imgs = [ds[i][0].convert("RGB") for i in idx]
att_imgs = [draw_en(ds[idx[k]][0], CLASSES["en"][target[k]]) for k in range(len(idx))]

# ---- SHARED multilingual model ----
sm, stok, _, mean, std = load_model(device); ls = get_logit_scale(sm)
pixel_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor()])
@torch.no_grad()
def shared_txt():
    out = {}
    for l in LANGS4:
        t = stok([TMPL[l].format(w) for w in CLASSES[l]]).to(device)
        out[l] = F.normalize(sm.encode_text(t), dim=-1)
    return out
STXT = shared_txt()
@torch.no_grad()
def shared_preds(imgs):
    xs = torch.stack([pixel_tf(im) for im in imgs]).to(device)
    feats = encode_image(sm, xs, mean, std)
    return {l: logits_for(feats, STXT[l], ls).argmax(-1).cpu().numpy() for l in LANGS4}

# ---- SEPARATE per-language models ----
SEP = {"en": EnCLIP(), "zh": ZhCLIP(), "ko": KoCLIP(), "ja": JaCLIP()}
def separate_preds(imgs):
    return {l: classify(SEP[l], imgs, CLASSES[l]) for l in LANGS4}

# ---- metrics ----
def agreement(preds):
    P = np.stack([preds[l] for l in LANGS4]); return (P == P[0:1]).all(0).mean()
def majority(preds):
    P = np.stack([preds[l] for l in LANGS4])  # [4,N]
    out = np.zeros(P.shape[1], int)
    for i in range(P.shape[1]):
        v, c = np.unique(P[:, i], return_counts=True); out[i] = v[c.argmax()]
    return out
def n_unique(preds):
    P = np.stack([preds[l] for l in LANGS4]); return np.array([len(np.unique(P[:, i])) for i in range(P.shape[1])], float)
def auc(neg, pos):
    s = np.concatenate([neg, pos]); y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt)-(cnt-1)/2.0)[inv]
    return (r[y == 1].sum()-len(pos)*(len(pos)+1)/2.0)/(len(pos)*len(neg))

def report(name, clean, att):
    print(f"\n===== {name} =====")
    print(f"{'lang':>5} | {'clean acc':>9} | {'attacked acc':>12} | {'ASR(->written)':>14}")
    res = {"per_lang": {}}
    for l in LANGS4:
        ca = (clean[l] == true).mean(); aa = (att[l] == true).mean(); asr = (att[l] == target).mean()
        res["per_lang"][l] = {"clean": float(ca), "attacked_acc": float(aa), "asr": float(asr)}
        print(f"{l:>5} | {100*ca:8.1f}% | {100*aa:11.1f}% | {100*asr:13.1f}%")
    ag_c, ag_a = agreement(clean), agreement(att)
    ens_c = (majority(clean) == true).mean(); ens_a = (majority(att) == true).mean()
    det_auc = auc(n_unique(clean), n_unique(att))
    res.update({"agree_clean": float(ag_c), "agree_attacked": float(ag_a),
                "ensemble_clean": float(ens_c), "ensemble_attacked": float(ens_a),
                "detector_auc": float(det_auc)})
    print(f"all-4-agree:   clean {100*ag_c:5.1f}%  ->  attacked {100*ag_a:5.1f}%")
    print(f"ensemble(vote):clean {100*ens_c:5.1f}%  ->  attacked {100*ens_a:5.1f}%   <- defense under attack")
    print(f"disagreement-detector AUC (clean vs attacked): {det_auc:.3f}   [>0.5 = detects attack]")
    return res

out = {"n": len(idx)}
out["shared"] = report("SHARED encoder (one multilingual CLIP, 4 label sets)", shared_preds(clean_imgs), shared_preds(att_imgs))
out["separate"] = report("SEPARATE encoders (4 independent per-language CLIPs)", separate_preds(clean_imgs), separate_preds(att_imgs))
json.dump(out, open("results/shared_vs_separate.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved results/shared_vs_separate.json")
