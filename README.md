# A Unified GNN Framework for Network Reliability and Dynamic Component Importance Measures via the Survival Signature

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

This repository contains the code, benchmark networks, and reproduction
scripts for the paper:

> *A Unified Graph Neural Network Framework for Network Reliability and
> Dynamic Component Importance Measures via the Survival Signature.*

> **Note.** This repository deliberately contains **only the source code, data,
> and figures** needed to reproduce the results. The manuscript text (PDF/DOCX)
> is *not* included, to avoid self-similarity during the journal's plagiarism
> screening. The DOI and full citation will be added here upon acceptance.

The framework estimates, from a **single trained graph neural network**:

1. the **survival signature** of a two-terminal network,
2. the time-dependent **network reliability** R(t),
3. three classical **component importance measures** — Birnbaum (BI),
   Criticality (CI), and Fussell–Vesely (FVI) — via the survival signature
   (Proposition 1 in the paper), and
4. the reliability and importance of **sub-networks without retraining**
   (within a bounded, honestly reported scope).

## Using the framework as a decision-support tool

Beyond reproducing the paper, the repository ships a ready-to-use
**decision-support application** that takes a user-defined network and returns
a full reliability report.

**Input** (a JSON file): network edges, component types, failure rates, and
optionally a list of sub-networks to analyze.
**Output**: the survival-signature-based reliability R(t), the three importance
measures (BI/CI/FVI) for every component, a ranking of the most critical
components, and retraining-free sub-network reliability — plus optional plots.

Command line:

```bash
cd scripts
python analyze.py --example                        # built-in 6-component demo
python analyze.py ../examples/benchmark6.json      # analyze your own network
python analyze.py ../examples/ieee14.json --plots --out report.json
```

Example input (`examples/benchmark6.json`):

```json
{
  "edges": [["S",1],["S",2],[1,3],[3,"T"], "..."],
  "types": {"1":1,"2":1,"3":2},
  "lambda1": 0.8, "lambda2": 1.5,
  "sub_networks": [[1],[6],[2,6]]
}
```

Python API:

```python
from app_api import analyze_network, format_report
report = analyze_network(spec, t_eval=0.5, sub_networks=[[1],[6]])
print(format_report(report))             # human-readable report
report["component_ranking_by_birnbaum"]  # e.g. ['3','4','5',...]
```

The application is a thin wrapper around the research modules in `src/`, so its
outputs are identical to those used in the paper.

## Repository structure

```
.
├── src/
│   ├── networks.py     # network builders, structure function,
│   │                   # exact + Monte-Carlo survival signatures, reliability
│   ├── models.py       # FastReliabilityGNN (proposed) + ANN/GCNN/GraphSAGE/GAT
│   ├── framework.py    # training, adaptive scheme (Eq. 10),
│   │                   # importance measures (Prop. 1), sub-network analysis
│   └── app_api.py      # decision-support API: analyze_network(...) -> report
├── scripts/
│   ├── analyze.py               # command-line decision-support tool
│   ├── run_all_experiments.py   # reproduces the principal results
│   ├── statistical_comparison.py# 10-seed test, runtime std, generalization
│   ├── backbone_swap.py         # architecture-agnostic test (GraphSAGE backbone)
│   ├── make_figures.py          # regenerates Figs. 2–10
│   └── make_framework_diagram.py# regenerates Fig. 1
├── examples/
│   ├── benchmark6.json # 6-component example network
│   └── ieee14.json     # IEEE 14-bus example network
├── figures/            # generated figures (Figs. 1–10)
├── data/
│   └── results.json    # recorded numerical results (single source of truth)
└── docs/
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+; a CUDA-capable GPU is recommended for the larger
networks but not required (the code falls back to CPU).

## Reproducing the results

```bash
cd scripts
python run_all_experiments.py          # full run
python run_all_experiments.py --quick  # faster smoke test
```

This prints the benchmark importance measures (Table 1), sub-network
reliability (Table 2), the IEEE 14-bus results (Table 8), the ablation
study (Table 6), and the scalability/runtime figures (Section 5.4).

To regenerate the figures:

```bash
cd scripts
python make_framework_diagram.py
python make_figures.py
```

## Networks included

| Network                    | Components | Reference        | Section |
|----------------------------|-----------:|------------------|---------|
| Benchmark (6)              | 6          | exact enumeration| 5.1–5.2 |
| Complex irregular (13)     | 13         | exact enumeration| 5.5–5.6 |
| Scalable (30, 50)          | 30 / 50    | Monte Carlo      | 5.4, 5.7|
| IEEE 14-bus (bus failures) | 12         | exact enumeration| 5.9     |

## Notes on reproducibility

- Component failure rates: λ₁ = 0.8 (type 1), λ₂ = 1.5 (type 2).
- Optimizer: Adam, lr = 0.01, weight decay 1e-4, hidden dim = 128.
- Monte-Carlo reference: up to 2000 samples per survival-signature cell
  (cells smaller than this are enumerated exactly).
- Runtimes reported in the paper are averaged over five independent runs;
  absolute values depend on hardware (paper used an NVIDIA T4-class GPU).
- Results on very large networks (30–50) and on extensive sub-network
  removals are reported honestly, including their accuracy limits
  (see Sections 5.4 and 5.7 of the paper).

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
