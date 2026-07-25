"""Compute pooled 2000-image (1000 clean + 1000 attacked) accuracy from existing Phase C JSONs.

No GPU / re-run needed — uses attacked_acc + clean_acc_masked already in gated_comparison.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
PARTNERS = ["zh", "ko", "ja"]


def mean_dict(d: dict) -> float:
    return sum(d.values()) / len(d)


def policy_clean_acc(p: dict) -> dict:
    # Under a defense policy, accuracy on clean images after the policy's masking.
    if "clean_acc_masked" in p:
        return p["clean_acc_masked"]
    return p["clean_acc"]


def main() -> None:
    rows = []
    for L in PARTNERS:
        path = ROOT / L / "multi" / "gated_comparison.json"
        g = json.loads(path.read_text(encoding="utf-8"))
        out = {"L": L, "policies": {}}
        print(f"=== {L} ===")
        for pol in ["never_defend", "always_defend", "gated"]:
            p = g["policies"][pol]
            atk_mean = mean_dict(p["attacked_acc"])
            cln_mean = mean_dict(policy_clean_acc(p))
            mixed = 0.5 * atk_mean + 0.5 * cln_mean  # 1000 + 1000 equal weight
            rec = {
                "attacked_mean_acc": atk_mean,
                "clean_policy_mean_acc": cln_mean,
                "mixed_2000_mean_acc": mixed,
                "defend_frac_attacked": p.get("defend_frac_attacked", 0.0),
                "defend_frac_clean": p.get("defend_frac_clean", 0.0),
                "clean_delta_mean": mean_dict(p.get("clean_delta", {L: 0.0})),
            }
            out["policies"][pol] = rec
            print(
                f"  {pol:14s}  atk={100*atk_mean:5.2f}%  "
                f"clean_pol={100*cln_mean:5.2f}%  "
                f"MIXED2000={100*mixed:5.2f}%  "
                f"frac_atk/cln={rec['defend_frac_attacked']:.3f}/"
                f"{rec['defend_frac_clean']:.3f}"
            )
        a = out["policies"]["always_defend"]["mixed_2000_mean_acc"]
        gate = out["policies"]["gated"]["mixed_2000_mean_acc"]
        print(f"  --> gated - always (mixed2000) = {100*(gate-a):+.2f} pp")
        rows.append(out)

    out_path = ROOT / "mixed_2000_summary.json"
    payload = {
        "definition": (
            "mixed_2000_mean_acc = 0.5 * mean(EN,L attacked_acc) + "
            "0.5 * mean(EN,L clean_policy_acc) on the same 1000 indices "
            "(1000 clean + 1000 multi-attacked = 2000 equal-weight eval)."
        ),
        "partners": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
