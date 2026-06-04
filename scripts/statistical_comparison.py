"""
statistical_comparison.py
-------------------------
Reproduces the statistical analyses reported in Sections 5.4, 5.5, and 5.10:

  - 10-seed comparison of the proposed model vs GCNN, with paired t-test,
    Cohen's d, and 95% confidence intervals (Section 5.5)
  - runtime mean +/- std over five runs on the 30-component network (5.4)
  - exploratory cross-topology generalization test (Section 5.10)

Usage:
    python statistical_comparison.py
    python statistical_comparison.py --seeds 10 --quick
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel

import networks as N
import framework as Fw
from framework import device, build_indicator_matrix
from models import StandardGCNN, FastReliabilityGNN


def _normalize_adj(A):
    A = A + np.eye(A.shape[0])
    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    dinv[nz] = np.power(deg[nz], -0.5)
    D = np.diag(dinv)
    return D @ A @ D


def _train(net, ModelClass, states, labels, adj, seed, epochs=250):
    torch.manual_seed(seed)
    m = ModelClass(net['nn'], net['hidden']).to(device)
    X = build_indicator_matrix(net, states)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(); idx = np.arange(len(states))
    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 256):
            bb = idx[b:b + 256]; opt.zero_grad()
            loss = crit(m(X[bb], adj, net['node_features']), y[bb])
            loss.backward(); opt.step()
    return m


def _full_accuracy(net, model, adj, X_all, y_all):
    model.eval()
    with torch.no_grad():
        pr = []
        for b in range(0, len(X_all), 2048):
            pr.append(model(X_all[b:b + 2048], adj, net['node_features']).argmax(dim=1))
        return (torch.cat(pr) == y_all).float().mean().item()


def ten_seed_comparison(n_seeds=10):
    print("=" * 56)
    print(f"{n_seeds}-SEED COMPARISON (complex 13, 10% data)")
    print("=" * 56)
    net = Fw.setup_net_tensors(N.build_complex13())
    net['adj_norm'] = torch.tensor(_normalize_adj(net['A']), dtype=torch.float32).to(device)
    seeds = [42, 7, 123, 2024, 99, 1, 256, 512, 77, 2025][:n_seeds]
    n_tr = int(0.10 * 2 ** 13)
    allst = [set(c for i, c in enumerate(net['comps']) if (mk >> i) & 1)
             for mk in range(2 ** 13)]
    X_all = build_indicator_matrix(net, allst)
    y_all = torch.tensor([N.structure_fn(net, s) for s in allst],
                         dtype=torch.long).to(device)
    res = {'GCNN': [], 'Proposed': []}
    for seed in seeds:
        states, labels = Fw.stratified_training_states(net, n_tr, seed=seed)
        mg = _train(net, StandardGCNN, states, labels, net['adj_norm'], seed)
        res['GCNN'].append(_full_accuracy(net, mg, net['adj_norm'], X_all, y_all))
        mp = _train(net, FastReliabilityGNN, states, labels, net['adj_tensor'], seed)
        res['Proposed'].append(_full_accuracy(net, mp, net['adj_tensor'], X_all, y_all))
    for name in ['GCNN', 'Proposed']:
        a = np.array(res[name]); ci = 1.96 * a.std() / np.sqrt(len(a))
        print(f"  {name:<10}: {a.mean():.4f} +/- {a.std():.4f}  "
              f"(95% CI [{a.mean()-ci:.4f}, {a.mean()+ci:.4f}], n={len(a)})")
    t, p = ttest_rel(res['Proposed'], res['GCNN'])
    d = t / np.sqrt(len(seeds))
    print(f"\n  paired t = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}")
    print(f"  conclusion: {'no significant difference' if p > 0.05 else 'significant'}")


def runtime_std(n_runs=5):
    print("\n" + "=" * 56)
    print(f"RUNTIME mean +/- std (30-comp, {n_runs} runs)")
    print("=" * 56)
    net = Fw.setup_net_tensors(N.build_scalable_network(30, seed=11))
    mcs, gnn = [], []
    for run in range(n_runs):
        _, tm = N.mcs_survival_signature(net, n_mcs=1500, seed=run)
        mcs.append(tm)
        states, labels = Fw.stratified_training_states(net, 1000, seed=run)
        t0 = time.time()
        m = _train(net, FastReliabilityGNN, states, labels,
                   net['adj_tensor'], run, epochs=300)
        _, ti = Fw.survival_signature(net, m)
        gnn.append(time.time() - t0)
    mcs, gnn = np.array(mcs), np.array(gnn)
    print(f"  MCS: {mcs.mean():.1f} +/- {mcs.std():.1f} s")
    print(f"  GNN: {gnn.mean():.1f} +/- {gnn.std():.1f} s")
    print(f"  speedup: {mcs.mean()/gnn.mean():.1f}x")


def cross_topology():
    print("\n" + "=" * 56)
    print("CROSS-TOPOLOGY GENERALIZATION (exploratory, Section 5.10)")
    print("=" * 56)
    netA = Fw.setup_net_tensors(N.build_complex13())
    netB = Fw.setup_net_tensors(N.build_complex13_variant())
    sA, lA = Fw.stratified_training_states(netA, 1000, seed=42)
    modelA = _train(netA, FastReliabilityGNN, sA, lA, netA['adj_tensor'], 42, epochs=300)
    allB = [set(c for i, c in enumerate(netB['comps']) if (mk >> i) & 1)
            for mk in range(2 ** 13)]
    XB = build_indicator_matrix(netB, allB)
    yB = torch.tensor([N.structure_fn(netB, s) for s in allB],
                      dtype=torch.long).to(device)
    cross = _full_accuracy(netB, modelA, netB['adj_tensor'], XB, yB)
    sB, lB = Fw.stratified_training_states(netB, 1000, seed=42)
    modelB = _train(netB, FastReliabilityGNN, sB, lB, netB['adj_tensor'], 42, epochs=300)
    same = _full_accuracy(netB, modelB, netB['adj_tensor'], XB, yB)
    print(f"  train A, test B : {cross:.4f}")
    print(f"  train B, test B : {same:.4f}")
    print(f"  transfer gap    : {same - cross:.4f}")
    print("  note: partial transfer; framework is per-network by design.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    print(f"Device: {device}")
    ten_seed_comparison(n_seeds=5 if args.quick else args.seeds)
    if not args.quick:
        runtime_std()
    cross_topology()
    print("\nDone.")
