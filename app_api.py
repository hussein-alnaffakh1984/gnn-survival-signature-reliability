"""
app_api.py
----------
Decision-support API for the unified GNN survival-signature framework.

Single entry point:  analyze_network(...)
Given a user-defined network (nodes, edges, component types), it returns a
complete reliability report:

    INPUT                         ->     OUTPUT
    -----                                ------
    topology G(N, L)                     survival signature  Phi
    component types (1/2)                reliability curve   R(t)
    failure rates (lambda1, lambda2)     importance measures BI / CI / FVI
    (optional) sub-networks              component ranking (most critical first)
                                         sub-network reliability (no retraining)

This wraps the research modules in src/ behind a clean, user-facing function.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networks as N
import framework as Fw


# ----------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------
def build_network_from_spec(spec):
    """
    Build a network context from a plain dict specification.

    spec = {
        "edges": [["S", 1], [1, 2], [2, "T"], ...],   # required
        "types": {"1": 1, "2": 2, ...},               # component -> type (1/2)
        "lambda1": 0.8,                               # optional
        "lambda2": 1.5                                # optional
    }

    Components are every node that is not 'S' or 'T' and is not listed as an
    always-on intermediate bus. By default, every non-S/T node that appears in
    `types` is a failable component.
    """
    import networkx as nx
    G = nx.Graph()
    for u, v in spec["edges"]:
        G.add_edge(u, v)

    types = {}
    for k, val in spec.get("types", {}).items():
        # keys may be strings from JSON; keep original node id type if possible
        try:
            key = int(k)
        except (ValueError, TypeError):
            key = k
        types[key] = int(val)

    comps = sorted([c for c in types.keys()], key=lambda x: str(x))
    if not comps:
        # default: all non-S/T nodes alternate types
        nodes = [n for n in G.nodes() if n not in ('S', 'T')]
        nodes = sorted(nodes, key=lambda x: str(x))
        comps = nodes
        types = {c: (1 if i % 2 == 0 else 2) for i, c in enumerate(nodes)}

    net = N.make_network_context(G, types, comps)
    net = Fw.setup_net_tensors(net, hidden_dim=128)
    return net


# ----------------------------------------------------------------------
# Main analysis entry point
# ----------------------------------------------------------------------
def analyze_network(spec, t_eval=0.5, time_grid=None, sub_networks=None,
                    use_exact_if_small=True, max_iter=6, seed=42, verbose=True):
    """
    Run the full decision-support analysis on a network.

    Parameters
    ----------
    spec : dict (see build_network_from_spec) OR an already-built net context
    t_eval : time at which importance measures are reported
    time_grid : array of times for the reliability curve (default 0..3)
    sub_networks : list of lists, each a set of components to remove (no retrain)
    use_exact_if_small : if state space <= 2^16, use exact reference for training
    max_iter : adaptive-training iterations
    seed : random seed
    verbose : print progress

    Returns
    -------
    report : dict with all outputs (also JSON-serializable)
    """
    lam1 = spec.get("lambda1", N.LAMBDA_TYPE1) if isinstance(spec, dict) else N.LAMBDA_TYPE1
    lam2 = spec.get("lambda2", N.LAMBDA_TYPE2) if isinstance(spec, dict) else N.LAMBDA_TYPE2

    net = spec if (isinstance(spec, dict) and "nn" in spec) else build_network_from_spec(spec)

    m = len(net['comps'])
    if time_grid is None:
        time_grid = list(np.round(np.linspace(0, 3, 31), 3))

    if verbose:
        print(f"[1/5] Network: {m} components "
              f"(type-1: {net['m1']}, type-2: {net['m2']}), "
              f"state space 2^{m}.")

    # --- reference survival signature (for training target / validation) ---
    small = m <= 16
    if small and use_exact_if_small:
        phi_ref = N.exact_survival_signature(net)
        ref_kind = "exact enumeration"
    else:
        phi_ref, _ = N.mcs_survival_signature(net, n_mcs=2000)
        ref_kind = "Monte Carlo (2000/cell)"
    if verbose:
        print(f"[2/5] Reference survival signature computed ({ref_kind}).")

    # --- train the GNN (adaptive) ---
    if verbose:
        print(f"[3/5] Training GNN (adaptive)...")
    model, phi_gnn, train_time, history, train_states, _ = Fw.adaptive_train(
        net, phi_ref, seed=seed, max_iter=max_iter,
        sigma0=0.015, verbose=verbose)

    # --- reliability curve ---
    R_curve = [(float(t), float(N.network_reliability(t, phi_gnn, net['m1'],
               net['m2'], lam1, lam2))) for t in time_grid]
    R_at_eval = float(N.network_reliability(t_eval, phi_gnn, net['m1'],
                      net['m2'], lam1, lam2))
    if verbose:
        print(f"[4/5] Reliability curve computed. R({t_eval}) = {R_at_eval:.4f}")

    # --- importance measures + ranking ---
    importance = {}
    for c in net['comps']:
        im = Fw.importance_measures(net, c, t=t_eval)
        importance[c] = {k: float(v) for k, v in im.items()}
    # ranking by Birnbaum (most critical first)
    ranking = sorted(net['comps'], key=lambda c: -importance[c]['BI'])
    if verbose:
        top = ranking[0]
        print(f"[5/5] Importance computed. Most critical component: "
              f"{top} (BI={importance[top]['BI']:.4f})")

    # --- accuracy vs reference (validation) ---
    ss_err = max(abs(phi_ref[k] - phi_gnn.get(k, 0)) for k in phi_ref)
    R_ref = N.network_reliability(t_eval, phi_ref, net['m1'], net['m2'], lam1, lam2)
    rel_err = abs(R_ref - R_at_eval) / R_ref if R_ref else 0.0

    # --- optional sub-network analysis (no retraining) ---
    sub_results = []
    if sub_networks:
        for removed in sub_networks:
            removed = set(removed)
            phi_sub, st1, st2 = Fw.subnetwork_signature(net, model, removed)
            R_sub = float(N.network_reliability(t_eval, phi_sub, len(st1),
                          len(st2), lam1, lam2))
            sub_results.append({"removed": sorted(str(x) for x in removed),
                                "reliability": R_sub})
            if verbose:
                print(f"      sub-network (remove {sorted(removed)}): "
                      f"R({t_eval}) = {R_sub:.4f}")

    report = {
        "network": {
            "components": m, "type1": net['m1'], "type2": net['m2'],
            "state_space": f"2^{m}", "lambda1": lam1, "lambda2": lam2,
        },
        "reference_kind": ref_kind,
        "training": {
            "states_used": len(train_states),
            "fraction_of_state_space": round(len(train_states) / (2 ** m), 4),
            "iterations": len(history),
            "train_time_seconds": round(train_time, 2),
        },
        "validation": {
            "max_survival_signature_error": round(ss_err, 4),
            "reliability_relative_error": round(rel_err, 4),
        },
        "reliability_at_t_eval": {"t": t_eval, "R": round(R_at_eval, 4)},
        "reliability_curve": R_curve,
        "importance_measures": {str(c): importance[c] for c in net['comps']},
        "component_ranking_by_birnbaum": [str(c) for c in ranking],
        "sub_networks": sub_results,
    }
    return report


def format_report(report):
    """Return a human-readable text version of the report."""
    L = []
    n = report["network"]
    L.append("=" * 60)
    L.append("NETWORK RELIABILITY & IMPORTANCE REPORT")
    L.append("=" * 60)
    L.append(f"Components: {n['components']}  (type-1: {n['type1']}, "
             f"type-2: {n['type2']})   state space: {n['state_space']}")
    L.append(f"Failure rates: lambda1={n['lambda1']}, lambda2={n['lambda2']}")
    L.append(f"Reference: {report['reference_kind']}")
    tr = report["training"]
    L.append(f"Training: {tr['states_used']} states "
             f"({tr['fraction_of_state_space']*100:.1f}% of space), "
             f"{tr['iterations']} adaptive iterations, "
             f"{tr['train_time_seconds']}s")
    v = report["validation"]
    L.append(f"Validation: max SS error = {v['max_survival_signature_error']}, "
             f"reliability rel. error = {v['reliability_relative_error']*100:.2f}%")
    rl = report["reliability_at_t_eval"]
    L.append("")
    L.append(f"RELIABILITY at t={rl['t']}:  R = {rl['R']}")
    L.append("")
    L.append("COMPONENT IMPORTANCE MEASURES (at t_eval):")
    L.append(f"  {'rank':<5}{'comp':<8}{'BI':<10}{'CI':<10}{'FVI':<10}")
    im = report["importance_measures"]
    for i, c in enumerate(report["component_ranking_by_birnbaum"], 1):
        d = im[c]
        L.append(f"  {i:<5}{c:<8}{d['BI']:<10.4f}{d['CI']:<10.4f}{d['FVI']:<10.4f}")
    L.append("")
    L.append(f"MOST CRITICAL COMPONENT: "
             f"{report['component_ranking_by_birnbaum'][0]} "
             f"(maintain/reinforce first)")
    if report["sub_networks"]:
        L.append("")
        L.append("SUB-NETWORK ANALYSIS (no retraining):")
        for s in report["sub_networks"]:
            L.append(f"  remove {s['removed']}: R = {s['reliability']:.4f}")
    L.append("=" * 60)
    return "\n".join(L)
