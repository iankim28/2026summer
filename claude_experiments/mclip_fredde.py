"""Confirmation on the ACTUAL FreddeFrallan M-CLIP named in the proposal.

M-CLIP = frozen OpenAI CLIP ViT-B/32 image encoder + an XLM-R text encoder distilled
to match OpenAI CLIP's English text space. Because every language is distilled to the
same English teacher, cross-lingual alignment is even stronger -> transfer should be
at least as complete as the jointly-trained open_clip model.

image features: open_clip ViT-B-32 (openai)
text features : M-CLIP/XLM-Roberta-Large-Vit-B-32
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers
from data_utils import get_loader
from mclip_lib import TRANSLATIONS, TEMPLATES, LANGS

MCLIP_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="stl10")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eps_grid", default="2,4,8")
    args = ap.parse_args()
    device = "cuda"
    eps_grid = [float(e) for e in args.eps_grid.split(",")]

    # image encoder = OpenAI CLIP ViT-B/32
    img_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    img_model = img_model.to(device).eval()
    for p in img_model.parameters():
        p.requires_grad_(False)
    ls = img_model.logit_scale.exp().detach()
    mean = CLIP_MEAN.to(device); std = CLIP_STD.to(device)

    # text encoder = FreddeFrallan M-CLIP (kept on CPU: only 10 prompts/lang, no grad;
    # M-CLIP.forward tokenizes on CPU internally so running the model on CPU avoids a
    # device mismatch). Embeddings are moved to GPU after.
    txt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(MCLIP_NAME).eval()
    tok = transformers.AutoTokenizer.from_pretrained(MCLIP_NAME)

    loader, classes = get_loader(args.dataset, n=args.n, batch_size=args.bs)

    @torch.no_grad()
    def text_embeds():
        out = {}
        for l in LANGS:
            prompts = [TEMPLATES[l].format(TRANSLATIONS[c][l]) for c in classes]
            emb = txt_model.forward(prompts, tok)
            out[l] = F.normalize(emb, dim=-1).to(device)
        return out
    txt = text_embeds()

    def enc_img(x):
        feats = img_model.encode_image((x - mean) / std)
        return F.normalize(feats, dim=-1)

    def pgd_en(x, y, eps):
        x0 = x.clone().detach(); alpha = 2.5 * eps / args.steps
        xa = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0, 1).detach()
        for _ in range(args.steps):
            xa.requires_grad_(True)
            loss = F.cross_entropy(ls * enc_img(xa) @ txt["en"].t(), y)
            g = torch.autograd.grad(loss, xa)[0]
            with torch.no_grad():
                xa = torch.min(torch.max(xa + alpha * g.sign(), x0 - eps), x0 + eps)
                xa = torch.clamp(xa, 0, 1)
            xa = xa.detach()
        return xa

    # clean accuracy
    corr = {l: 0 for l in LANGS}; tot = 0
    advcorr = {e: {l: 0 for l in LANGS} for e in eps_grid}
    agree_clean = 0; agree_adv = {e: 0 for e in eps_grid}
    for x, y in loader:
        x = x.to(device); y = y.to(device); tot += y.numel()
        with torch.no_grad():
            f = enc_img(x)
            preds = {}
            for l in LANGS:
                p = (ls * f @ txt[l].t()).argmax(-1); preds[l] = p
                corr[l] += (p == y).sum().item()
            st = torch.stack([preds[l] for l in LANGS]); agree_clean += (st == st[0:1]).all(0).sum().item()
        for e in eps_grid:
            xa = pgd_en(x, y, e / 255.0)
            with torch.no_grad():
                fa = enc_img(xa); preds = {}
                for l in LANGS:
                    p = (ls * fa @ txt[l].t()).argmax(-1); preds[l] = p
                    advcorr[e][l] += (p == y).sum().item()
                st = torch.stack([preds[l] for l in LANGS]); agree_adv[e] += (st == st[0:1]).all(0).sum().item()

    print(f"\n=== FreddeFrallan M-CLIP ({MCLIP_NAME}) on {args.dataset} (n={tot}) ===")
    print("CLEAN  " + "  ".join(f"{l}:{corr[l]/tot*100:5.1f}" for l in LANGS) +
          f"   agree:{agree_clean/tot*100:.1f}")
    print(f"{'eps':>4}  " + "  ".join(f"{l:>8}" for l in LANGS) + "    agree")
    for e in eps_grid:
        print(f"{e:>4}  " + "  ".join(f"{advcorr[e][l]/tot*100:8.1f}" for l in LANGS) +
              f"   {agree_adv[e]/tot*100:6.1f}")


if __name__ == "__main__":
    main()
