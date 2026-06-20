"""Summary figure: cross-lingual transfer, agreement-under-attack, detector AUC."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mclip_lib import LANGS

def load(p):
    with open(p) as f: return json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel A: robust accuracy per language vs eps (STL-10, single-language EN attack)
ax = axes[0, 0]
t = load("results/transfer_stl10_pgd.json")
eps = t["eps_grid"]
for l in LANGS:
    ys = [t["clean"]["acc"][l]*100] + [t["results"][str(e)]["acc"][l]*100 for e in eps]
    ax.plot([0]+eps, ys, marker="o", label=l)
ax.plot([0]+eps, [t["clean"]["ens"]*100]+[t["results"][str(e)]["ens"]*100 for e in eps],
        "k--", marker="s", label="ensemble")
ax.set_title("A. STL-10: English-only PGD attack collapses ALL languages")
ax.set_xlabel("eps (/255)"); ax.set_ylabel("accuracy (%)"); ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.3)

# Panel B: transfer fraction vs eps (CIFAR low-eps) -- flat, near 1.0
ax = axes[0, 1]
tl = load("results/transfer_cifar10_pgd_loweps.json")
eps_l = tl["eps_grid"]
for l in [x for x in LANGS if x != "en"]:
    ys = [tl["results"][str(e)]["transfer_fraction"][l] for e in eps_l]
    ax.plot(eps_l, ys, marker="o", label=l)
ax.axhline(1.0, color="gray", ls=":")
ax.set_title("B. CIFAR-10: transfer fraction is ~1.0 at ALL eps\n(H1 predicted partial transfer rising with eps -- REFUTED)")
ax.set_xlabel("eps (/255)"); ax.set_ylabel("transfer fraction (non-EN drop / EN drop)")
ax.set_ylim(0, 1.2); ax.legend(fontsize=8); ax.grid(alpha=.3)

# Panel C: agreement clean vs under attack
ax = axes[1, 0]
for name, ds, color in [("STL-10","stl10","tab:blue"), ("CIFAR-10","cifar10","tab:orange")]:
    tt = load(f"results/transfer_{ds}_pgd.json")
    e = tt["eps_grid"]
    ys = [tt["clean"]["agreement"]*100] + [tt["results"][str(x)]["agreement"]*100 for x in e]
    ax.plot([0]+e, ys, marker="o", color=color, label=name)
    ax.axhline(tt["clean"]["agreement"]*100, color=color, ls=":", alpha=.5)
ax.set_title("C. All-language AGREEMENT rises under attack\n(disagreement defense needs the opposite)")
ax.set_xlabel("eps (/255)  [0 = clean]"); ax.set_ylabel("all-5-agree (%)"); ax.legend(); ax.grid(alpha=.3)

# Panel D: detector AUC vs eps
ax = axes[1, 1]
for name, ds, color in [("STL-10","stl10","tab:blue"), ("CIFAR-10","cifar10","tab:orange")]:
    dd = load(f"results/detector_{ds}_pgd.json")
    es = sorted(dd["by_eps"].keys(), key=lambda x: float(x))
    ax.plot([float(x) for x in es], [dd["by_eps"][x]["auc_mean_js"] for x in es],
            marker="o", color=color, label=f"{name} (mean-JS)")
    ax.plot([float(x) for x in es], [dd["by_eps"][x]["auc_n_unique"] for x in es],
            marker="^", ls="--", color=color, label=f"{name} (vote)")
ax.axhline(0.5, color="red", ls="-", label="random (0.5)")
ax.set_title("D. Disagreement-detector ROC-AUC < 0.5\n(worse than random: adv images agree MORE)")
ax.set_xlabel("eps (/255)"); ax.set_ylabel("detector AUC"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig("results/summary.png", dpi=130)
print("saved results/summary.png")
