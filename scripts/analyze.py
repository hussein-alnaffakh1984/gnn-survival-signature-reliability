"""
analyze.py  —  command-line decision-support tool
--------------------------------------------------
Reads a network specification (JSON), runs the full analysis, prints a
human-readable report, and (optionally) saves a JSON report and plots.

USAGE
-----
    python analyze.py network.json
    python analyze.py network.json --t 0.5 --out report.json --plots
    python analyze.py --example          # run a built-in 6-component example

INPUT JSON FORMAT
-----------------
    {
      "edges":  [["S", 1], ["S", 2], [1, 3], [3, "T"], ...],
      "types":  {"1": 1, "2": 1, "3": 2, ...},
      "lambda1": 0.8,
      "lambda2": 1.5,
      "sub_networks": [[1], [6], [2, 6]]      # optional, components to remove
    }
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app_api import analyze_network, format_report


EXAMPLE_SPEC = {
    "edges": [["S", 1], ["S", 2], [1, 3], [1, 4], [2, 3], [2, 4],
              [3, 5], [3, 6], [4, 5], [4, 6], [5, "T"], [6, "T"]],
    "types": {"1": 1, "2": 1, "3": 2, "4": 2, "5": 1, "6": 2},
    "lambda1": 0.8,
    "lambda2": 1.5,
    "sub_networks": [[1], [6], [2, 6]]
}


def save_plots(report, prefix):
    """Save reliability curve and importance bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reliability curve
    ts = [p[0] for p in report["reliability_curve"]]
    rs = [p[1] for p in report["reliability_curve"]]
    plt.figure(figsize=(6, 4))
    plt.plot(ts, rs, color="#1f4e79", lw=2.2)
    plt.xlabel("Time t"); plt.ylabel("Network reliability R(t)")
    plt.title("Reliability curve"); plt.grid(True, alpha=0.3, ls="--")
    plt.tight_layout(); plt.savefig(f"{prefix}_reliability.png", dpi=130)
    plt.close()

    # importance ranking (Birnbaum)
    ranking = report["component_ranking_by_birnbaum"]
    im = report["importance_measures"]
    bis = [im[c]["BI"] for c in ranking]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(ranking)), bis, color="#5b8a5a", alpha=0.85)
    plt.xticks(range(len(ranking)), ranking)
    plt.xlabel("Component (ranked)"); plt.ylabel("Birnbaum importance")
    plt.title("Component importance ranking")
    plt.grid(True, alpha=0.3, axis="y", ls="--")
    plt.tight_layout(); plt.savefig(f"{prefix}_importance.png", dpi=130)
    plt.close()
    print(f"Plots saved: {prefix}_reliability.png, {prefix}_importance.png")


def main():
    ap = argparse.ArgumentParser(description="GNN network-reliability decision-support tool")
    ap.add_argument("network", nargs="?", help="path to network JSON spec")
    ap.add_argument("--example", action="store_true", help="run the built-in example")
    ap.add_argument("--t", type=float, default=0.5, help="time for importance measures")
    ap.add_argument("--out", default=None, help="path to save JSON report")
    ap.add_argument("--plots", action="store_true", help="save reliability + importance plots")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = ap.parse_args()

    if args.example:
        spec = EXAMPLE_SPEC
    elif args.network:
        with open(args.network) as f:
            spec = json.load(f)
    else:
        ap.error("provide a network JSON file or use --example")

    sub = spec.get("sub_networks")
    report = analyze_network(spec, t_eval=args.t, sub_networks=sub,
                             verbose=not args.quiet)

    print("\n" + format_report(report))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to {args.out}")

    if args.plots:
        prefix = os.path.splitext(args.out)[0] if args.out else "report"
        save_plots(report, prefix)


if __name__ == "__main__":
    main()
