"""L-inf FGSM / PGD attacks in pixel space against multilingual CLIP.

An attack maximizes the summed cross-entropy loss over a chosen set of
*attacked languages*. Single-language attack -> attacked_langs=["en"].
The image encoder is shared, so the perturbation is crafted once and then
scored against every language's labels to measure transfer.
"""
import torch
import torch.nn.functional as F
from mclip_lib import encode_image, logits_for


def _loss_over_langs(model, x_pixel, mean, std, txt, ls, y, attacked_langs):
    """Summed CE over attacked languages (untargeted: maximize this)."""
    feats = encode_image(model, x_pixel, mean, std)
    loss = 0.0
    for l in attacked_langs:
        lg = logits_for(feats, txt[l], ls)
        loss = loss + F.cross_entropy(lg, y)
    return loss


def fgsm(model, x, y, mean, std, txt, ls, eps, attacked_langs=("en",)):
    x = x.clone().detach()
    x.requires_grad_(True)
    loss = _loss_over_langs(model, x, mean, std, txt, ls, y, attacked_langs)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv


def pgd(model, x, y, mean, std, txt, ls, eps, steps=20, alpha=None,
        attacked_langs=("en",), random_start=True):
    x_orig = x.clone().detach()
    if alpha is None:
        alpha = 2.5 * eps / steps
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_orig + delta, 0, 1).detach()
    else:
        x_adv = x_orig.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = _loss_over_langs(model, x_adv, mean, std, txt, ls, y, attacked_langs)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv
