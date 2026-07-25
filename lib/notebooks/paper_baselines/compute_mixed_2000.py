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
    ("defense_prefix", "Defense-Prefix", HERE / "defense_prefix" / "results" / "comparison_summary_final_n1000.json"),
    ("dyslexify", "Dyslexify", HERE / "dyslexify" / "results" / "comparison_summary_final_n1000.json"),
    ("sampling_tar", "SamplingTAR", HERE / "sampling_tar" / "results" / "comparison_summary_final_n1000.json"),
]

OURS_ALWAYS = (
    NOTEBOOKS
    / "heatmap_defense_improvements"
    / "cc_bbox_blur"
    / "results"
    / "confusion_results_cc_bbox_blur.json"
)
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

    # Reference: always-on cc_bbox_blur (EN∩ZH)
    ours = json.loads(OURS_ALWAYS.read_text(encoding="utf-8"))
    ours_policies = policies_from_summary(ours)
    print_method("cc_bbox_blur (ref)", "EN+ZH", ours_policies, extra=" - always-on")
    methods.append(
        {
            "method": "cc_bbox_blur",
            "display": "cc_bbox_blur (ref)",
            "scope": "EN+ZH",
            "source": str(OURS_ALWAYS.relative_to(NOTEBOOKS)),
            "policies": ours_policies,
            "reference": True,
        }
    )

    # Reference: gated ZH from attack_detector mixed_2000
    gated_doc = json.loads(GATED_MIXED.read_text(encoding="utf-8"))
    zh = next(p for p in gated_doc["partners"] if p["L"] == "zh")
    gpol = zh["policies"]["gated"]
    gated_policies = {
        "gated": {
            "attacked_mean_acc": gpol["attacked_mean_acc"],
            "clean_policy_mean_acc": gpol["clean_policy_mean_acc"],
            "mixed_2000_mean_acc": gpol["mixed_2000_mean_acc"],
            "clean_delta_mean": gpol["clean_delta_mean"],
            "defend_frac_attacked": gpol.get("defend_frac_attacked"),
            "defend_frac_clean": gpol.get("defend_frac_clean"),
        }
    }
    print_method("gated ZH (ref)", "EN+ZH", gated_policies)
    methods.append(
        {
            "method": "gated_zh",
            "display": "gated ZH (ref)",
            "scope": "EN+ZH",
            "source": str(GATED_MIXED.relative_to(NOTEBOOKS)),
            "policies": gated_policies,
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
        elif "gated" in pols:
            g = pols["gated"]["mixed_2000_mean_acc"]
            print(f"| {m['display']} | {m['scope']} | - | {fmt_pct(g)} (gated) | - |")

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
            "(EN+ZH for ocr_blur / cc_bbox_blur; EN-only otherwise)."
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
