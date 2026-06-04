"""
run_all_experiments.py
----------------------
Reproduces the principal results of the paper:
  - Table 1 : importance measures on the 6-component benchmark
  - Table 2 : sub-network analysis without retraining
  - Section 5.4 : scalability + runtime (30, 50 components)
  - Table 6 : ablation study (complex 13)
  - Section 5.7 : sub-network removal scope (30 components)
  - Table 8 : IEEE 14-bus

Usage:
    python run_all_experiments.py            # run everything
    python run_all_experiments.py --quick    # smaller budgets for a fast check
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import random
import torch
from scipy.stats import spearmanr, ttest_rel

import networks as N
import framework as Fw
from framework import device

def banner(t): print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def exp_benchmark6():
    banner("Benchmark 6-component: importance measures (Table 1)")
    net = Fw.setup_net_tensors(N.build_benchmark6())
    phi_ref = N.exact_survival_signature(net)
    print(f"  exact R(0.5) = {N.network_reliability(0.5, phi_ref, net['m1'], net['m2']):.4f}")
    print(f"  {'comp':<6}{'type':<6}{'BI':<10}{'CI':<10}{'FVI':<10}")
    for c in net['comps']:
        im = Fw.importance_measures(net, c, t=0.5)
        print(f"  {c:<6}{net['ctype'][c]:<6}{im['BI']:<10.4f}{im['CI']:<10.4f}{im['FVI']:<10.4f}")


def exp_subnetwork6():
    banner("Benchmark 6-component: sub-networks without retraining (Table 2)")
    net = Fw.setup_net_tensors(N.build_benchmark6())
    phi_ref = N.exact_survival_signature(net)
    model, _, _, _, _, _ = Fw.adaptive_train(net, phi_ref, seed=42, max_iter=6)
    for removed in [{1}, {6}, {2, 6}]:
        phi_sub, st1, st2 = Fw.subnetwork_signature(net, model, removed)
        R = N.network_reliability(0.5, phi_sub, len(st1), len(st2))
        print(f"  remove {sorted(removed)}: GNN R(0.5) = {R:.4f}")


def exp_scalability(quick=False):
    banner("Scalability + runtime (Section 5.4, Table 4)")
    for n_comp in ([30] if quick else [30, 50]):
        net = Fw.setup_net_tensors(N.build_scalable_network(n_comp,
                                   seed=11 if n_comp == 30 else 23))
        phi_ref, t_mcs = N.mcs_survival_signature(net, n_mcs=1500)
        model, phi_gnn, t_train, hist, _, _ = Fw.adaptive_train(
            net, phi_ref, seed=42, max_iter=5, sigma0=0.02, verbose=False)
        _, t_inf = Fw.survival_signature(net, model)
        R_ref = N.network_reliability(0.5, phi_ref, net['m1'], net['m2'])
        R_gnn = N.network_reliability(0.5, phi_gnn, net['m1'], net['m2'])
        print(f"  {n_comp}-comp: MCS={t_mcs:.1f}s  GNN(tr+inf)={t_train + t_inf:.1f}s  "
              f"R_rel_err={abs(R_ref - R_gnn) / R_ref:.4f}")


def exp_ablation():
    banner("Ablation study (Table 6) on complex 13")
    from models import FastReliabilityGNN
    net = Fw.setup_net_tensors(N.build_complex13())
    phi_ref = N.exact_survival_signature(net)
    states, labels = Fw.stratified_training_states(net, 1000, seed=42)

    def ss_err(model):
        phi, _ = Fw.survival_signature(net, model)
        e = max(abs(phi_ref[k] - phi.get(k, 0)) for k in phi_ref)
        R_ref = N.network_reliability(0.5, phi_ref, net['m1'], net['m2'])
        R_g = N.network_reliability(0.5, phi, net['m1'], net['m2'])
        return e, abs(R_ref - R_g) / R_ref

    m, _ = Fw.train_with_states(net, states, labels, ModelClass=FastReliabilityGNN)
    e, r = ss_err(m)
    print(f"  full arch (no adaptive): SS_err={e:.4f} R_err={r:.4f}")
    m2, phi2, _, _, st2, _ = Fw.adaptive_train(net, phi_ref, seed=42, max_iter=5, verbose=False)
    e2 = max(abs(phi_ref[k] - phi2.get(k, 0)) for k in phi_ref)
    R_ref = N.network_reliability(0.5, phi_ref, net['m1'], net['m2'])
    R_g = N.network_reliability(0.5, phi2, net['m1'], net['m2'])
    print(f"  full framework (adaptive): SS_err={e2:.4f} R_err={abs(R_ref-R_g)/R_ref:.4f} "
          f"(train={len(st2)})")


def exp_subnet_scope():
    banner("Sub-network removal scope (Section 5.7) on 30-comp")
    net = Fw.setup_net_tensors(N.build_scalable_network(30, seed=11))
    phi_ref, _ = N.mcs_survival_signature(net, n_mcs=1500)
    model, _, _, _, _, _ = Fw.adaptive_train(net, phi_ref, seed=42, max_iter=5,
                                             sigma0=0.02, verbose=False)
    n = len(net['comps'])
    for pct in [0.20, 0.30, 0.40]:
        removed = random.Random(int(pct * 100)).sample(net['comps'], int(pct * n))
        # reference for sub-network
        def sub_sf(working):
            failed = (set(net['comps']) - set(working)) | set(removed)
            H = net['G'].copy(); H.remove_nodes_from(failed)
            return 1 if ('S' in H and 'T' in H and __import__('networkx').has_path(H, 'S', 'T')) else 0
        phi_gnn, st1, st2 = Fw.subnetwork_signature(net, model, set(removed))
        R_gnn = N.network_reliability(0.5, phi_gnn, len(st1), len(st2))
        print(f"  remove {int(pct*100)}% ({len(removed)}): GNN R(0.5)={R_gnn:.4f}")


def exp_ieee14():
    banner("IEEE 14-bus power network (Table 8)")
    net = Fw.setup_net_tensors(N.build_ieee14_buses())
    phi_ref = N.exact_survival_signature(net)
    print(f"  buses={14} failable={len(net['comps'])} state-space=2^{len(net['comps'])}")
    model, phi_gnn, t_train, hist, st, _ = Fw.adaptive_train(
        net, phi_ref, seed=42, max_iter=6, sigma0=0.015)
    ss = max(abs(phi_ref[k] - phi_gnn.get(k, 0)) for k in phi_ref)
    R_ref = N.network_reliability(0.5, phi_ref, net['m1'], net['m2'])
    R_gnn = N.network_reliability(0.5, phi_gnn, net['m1'], net['m2'])
    print(f"  train states={len(st)} ({100*len(st)/2**len(net['comps']):.1f}%)")
    print(f"  max SS error={ss:.4f}  R_ref={R_ref:.4f} R_gnn={R_gnn:.4f} "
          f"rel_err={abs(R_ref-R_gnn)/R_ref:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='smaller budgets')
    args = ap.parse_args()
    print(f"Device: {device}")
    exp_benchmark6()
    exp_subnetwork6()
    exp_ieee14()
    exp_ablation()
    if not args.quick:
        exp_scalability()
        exp_subnet_scope()
    print("\nAll experiments completed.")
