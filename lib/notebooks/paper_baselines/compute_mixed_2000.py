"""Compute pooled 2000-image (1000 clean + 1000 attacked) accuracy for paper baselines.

No GPU / re-run — recombines fields already in comparison_summary_final_n1000.json.
Same frozen sample as attack_detector / cc_bbox_blur: CIFAR10_BALANCED_1000_SAMPLE.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent
OUT_DIR = HERE / "results"

BASELINES = [
    ("ocr_blur", "OCR + blur", HERE / "ocr_blur" / "results" / "comparison_summary_final_n1000.json"),
    # Prefer EN+ZH merged summary when ZH CIFAR DP token exists; else EN-only.
    (
        "defense_prefix",
        "Defense-Prefix",
        HERE / "defense_prefix" / "results" / "comparison_summary_final_n1000_en_zh.json"
        if (HERE / "defense_prefix" / "results" / "comparison_summary_final_n1000_en_zh.json").is_file()
        else HERE / "defense_prefix" / "results" / "comparison_summary_final_n1000.json",
    ),
    ("dyslexify", "Dyslexify (heads)", HERE / "dyslexify" / "results" / "comparison_summary_final_n1000.json"),
    (
        "dyslexify_hybrid",
        "Dyslexify hybrid",
        HERE / "dyslexify" / "results" / "comparison_summary_final_n1000_hybrid.json",
    ),
    ("sampling_tar", "SamplingTAR (heads)", HERE / "sampling_tar" / "results" / "comparison_summary_final_n1000.json"),
    (
        "sampling_tar_hybrid",
        "SamplingTAR hybrid",
        HERE / "sampling_tar" / "results" / "comparison_summary_final_n1000_hybrid.json",
    ),
]

# Ours refs: Phase C gated mixed-2000 for partner ZH (same table as diary §2026-07-24).
# Do NOT use older heatmap confusion_results (atk 74.9%); Phase C always is atk 74.0%.
GATED_MIXED = NOTEBOOKS / "attack_detector" / "results" / "mixed_2000_summary.json"

SAMPLE_ID = "CIFAR10_BALANCED_1000_SAMPLE"


def mean_lang(d: dict, key: str) -> float:
    return sum(v[key] for v in d.values()) / len(d)


def policies_from_summary(g: dict) -> dict:
    """never / always from a baseline (or ours) comparison_summary-style JSON."""
    defense = g["defense"]
    clean = g["clean_degradation"]
    never_atk = mean_lang(defense, "baseline_acc")
    never_cln = mean_lang(clean, "baseline_acc")
    always_atk = mean_lang(defense, "acc")
    always_cln = mean_lang(clean, "masked_acc")
    clean_delta = mean_lang(clean, "delta_acc")
    return {
        "never": {
            "attacked_mean_acc": never_atk,
            "clean_policy_mean_acc": never_cln,
            "mixed_2000_mean_acc": 0.5 * never_atk + 0.5 * never_cln,
            "clean_delta_mean": 0.0,
        },
        "always": {
            "attacked_mean_acc": always_atk,
            "clean_policy_mean_acc": always_cln,
            "mixed_2000_mean_acc": 0.5 * always_atk + 0.5 * always_cln,
            "clean_delta_mean": clean_delta,
        },
    }


def scope_label(g: dict) -> str:
    scope = g.get("scope", "")
    if "en_zh" in scope or len(g.get("defense", {})) > 1:
        return "EN+ZH"
    if scope in ("zh_only", "zh"):
        return "ZH"
    return "EN"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def fmt_pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def print_method(name: str, scope: str, policies: dict, extra: str = "") -> None:
    print(f"=== {name} ({scope}){extra} ===")
    for pol in policies:
        p = policies[pol]
        print(
            f"  {pol:8s}  atk={fmt_pct(p['attacked_mean_acc']):>7s}  "
            f"clean_pol={fmt_pct(p['clean_policy_mean_acc']):>7s}  "
            f"MIXED2000={fmt_pct(p['mixed_2000_mean_acc']):>7s}  "
            f"Clean_d={fmt_pp(p['clean_delta_mean'])}"
        )
    if "never" in policies and "always" in policies:
        d = (
            policies["always"]["mixed_2000_mean_acc"]
            - policies["never"]["mixed_2000_mean_acc"]
        )
        print(f"  --> always - never (mixed2000) = {100 * d:+.2f} pp")


def main() -> None:
    methods = []
    print("Sample:", SAMPLE_ID, "(same 1000 idx + frozen attack_pos as our model)\n")

    for method_id, display, path in BASELINES:
        if not path.is_file():
            print(f"SKIP {display}: missing {path}")
            continue
        g = json.loads(path.read_text(encoding="utf-8"))
        scope = scope_label(g)
        policies = policies_from_summary(g)
        print_method(display, scope, policies)
        methods.append(
            {
                "method": method_id,
                "display": display,
                "scope": scope,
                "source": str(path.relative_to(NOTEBOOKS)),
                "policies": policies,
            }
        )

    # Reference: our cc_bbox_blur from attack_detector mixed_2000 (partner ZH)
    gated_doc = json.loads(GATED_MIXED.read_text(encoding="utf-8"))
    zh = next(p for p in gated_doc["partners"] if p["L"] == "zh")
    pol_map = {
        "never_defend": "never",
        "always_defend": "always",
        "gated": "gated",
    }
    ours_policies = {}
    for src, dst in pol_map.items():
        p = zh["policies"][src]
        ours_policies[dst] = {
            "attacked_mean_acc": p["attacked_mean_acc"],
            "clean_policy_mean_acc": p["clean_policy_mean_acc"],
            "mixed_2000_mean_acc": p["mixed_2000_mean_acc"],
            "clean_delta_mean": p["clean_delta_mean"],
            "defend_frac_attacked": p.get("defend_frac_attacked"),
            "defend_frac_clean": p.get("defend_frac_clean"),
        }
    print_method("cc_bbox_blur ZH (ref)", "EN+ZH", ours_policies, extra=" - Phase C")
    methods.append(
        {
            "method": "cc_bbox_blur_zh",
            "display": "cc_bbox_blur ZH (ref)",
            "scope": "EN+ZH",
            "source": str(GATED_MIXED.relative_to(NOTEBOOKS)),
            "partner": "zh",
            "policies": ours_policies,
            "reference": True,
        }
    )

    # Markdown tables for diary / chat
    print("\n### MIXED2000 by policy\n")
    print("| Method | scope | never | always | always - never |")
    print("|--------|-------|------:|-------:|---------------:|")
    for m in methods:
        pols = m["policies"]
        if "never" in pols and "always" in pols:
            n, a = pols["never"]["mixed_2000_mean_acc"], pols["always"]["mixed_2000_mean_acc"]
            print(
                f"| {m['display']} | {m['scope']} | {fmt_pct(n)} | {fmt_pct(a)} | "
                f"{100 * (a - n):+.2f} pp |"
            )
        if "gated" in pols:
            g = pols["gated"]["mixed_2000_mean_acc"]
            print(
                f"| {m['display']} gated | {m['scope']} | - | {fmt_pct(g)} | - |"
            )

    print("\n### Per-policy breakdown\n")
    print("| Method | scope | policy | atk | clean_pol | MIXED2000 | Clean d |")
    print("|--------|-------|--------|----:|----------:|----------:|--------:|")
    for m in methods:
        for pol, p in m["policies"].items():
            print(
                f"| {m['display']} | {m['scope']} | {pol} | "
                f"{fmt_pct(p['attacked_mean_acc'])} | "
                f"{fmt_pct(p['clean_policy_mean_acc'])} | "
                f"{fmt_pct(p['mixed_2000_mean_acc'])} | "
                f"{fmt_pp(p['clean_delta_mean'])} |"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mixed_2000_summary.json"
    payload = {
        "definition": (
            "mixed_2000_mean_acc = 0.5 * attacked_mean_acc + 0.5 * clean_policy_mean_acc "
            "on the same 1000 CIFAR indices (1000 clean + 1000 multi-attacked = 2000 "
            "equal-weight eval). Lang mean is over keys present in each JSON "
            "(EN+ZH for ocr_blur / our ZH ref; EN-only otherwise). "
            "Our ref uses attack_detector mixed_2000 ZH (Phase C always atk=74.0%), "
            "not older heatmap confusion_results (atk=74.9%)."
        ),
        "sample": SAMPLE_ID,
        "sample_note": (
            "Identical pool to attack_detector / cc_bbox_blur: frozen idx + attack_pos "
            "from lib/notebooks/image_samples/CIFAR10_BALANCED_1000_SAMPLE.json. "
            "No re-sampling; post-process of final_n1000 JSONs only."
        ),
        "methods": methods,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
