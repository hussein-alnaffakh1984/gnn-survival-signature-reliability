"""
framework.py
------------
Training utilities, the adaptive learning scheme with the survival-signature
convergence criterion, the importance-measure bridge (Proposition 1), and
retraining-free sub-network analysis.
"""
import numpy as np
import torch
import torch.nn as nn
import random
import time
from itertools import combinations
from math import comb

from networks import (structure_fn, network_reliability,
                      component_reliability, LAMBDA_TYPE1, LAMBDA_TYPE2)
from models import FastReliabilityGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------
# Tensor setup and indicator construction
# ----------------------------------------------------------------------
def setup_net_tensors(net, hidden_dim=128):
    net['node_features'] = torch.eye(net['nn'], dtype=torch.float32).to(device)
    net['adj_tensor'] = torch.tensor(net['A'], dtype=torch.float32).to(device)
    net['hidden'] = hidden_dim
    return net


def build_indicator_matrix(net, states):
    """[B, nn] indicator; S, T and intermediate buses are always on."""
    B = len(states)
    M = np.zeros((B, net['nn']), dtype=np.float32)
    comp_set = set(net['comps'])
    for n, idx in net['nidx'].items():
        if n not in comp_set:          # S, T, buses
            M[:, idx] = 1.0
    for b, s in enumerate(states):
        for c in s:
            M[b, net['nidx'][c]] = 1.0
    return torch.from_numpy(M).to(device)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def stratified_training_states(net, n_samples, seed=0):
    """Sample states covering every (l1,l2) stratum, then fill randomly."""
    rng = random.Random(seed)
    t1, t2, m1, m2 = net['t1'], net['t2'], net['m1'], net['m2']
    states, seen = [], set()
    per_cell = max(1, n_samples // ((m1 + 1) * (m2 + 1)))
    for l1 in range(m1 + 1):
        for l2 in range(m2 + 1):
            for _ in range(per_cell):
                w1 = rng.sample(t1, l1) if l1 <= len(t1) else t1
                w2 = rng.sample(t2, l2) if l2 <= len(t2) else t2
                key = tuple(sorted(w1 + w2))
                if key not in seen:
                    seen.add(key); states.append(set(w1 + w2))
    while len(states) < n_samples:
        k = rng.randint(0, len(net['comps']))
        w = tuple(sorted(rng.sample(net['comps'], k)))
        if w not in seen:
            seen.add(w); states.append(set(w))
    labels = [structure_fn(net, s) for s in states]
    return states, labels


# ----------------------------------------------------------------------
# Training and survival-signature assembly
# ----------------------------------------------------------------------
def train_with_states(net, states, labels, epochs=300, lr=0.01, seed=42,
                      ModelClass=FastReliabilityGNN):
    torch.manual_seed(seed)
    model = ModelClass(net['nn'], net['hidden']).to(device)
    X = build_indicator_matrix(net, states)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    idx = np.arange(len(states)); bs = 256
    t0 = time.time()
    for _ in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), bs):
            batch = idx[b:b + bs]
            opt.zero_grad()
            loss = crit(model(X[batch], net['adj_tensor'], net['node_features']),
                        y[batch])
            loss.backward(); opt.step()
    return model, time.time() - t0


def survival_signature(net, model, n_mcs=2000, seed=1):
    """Assemble survival signature from a trained model; returns (phi, time)."""
    rng = random.Random(seed)
    t1, t2, m1, m2 = net['t1'], net['t2'], net['m1'], net['m2']
    phi = {}
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for l1 in range(m1 + 1):
            for l2 in range(m2 + 1):
                total = comb(m1, l1) * comb(m2, l2)
                states = []
                if total <= n_mcs:
                    for w1 in combinations(t1, l1):
                        for w2 in combinations(t2, l2):
                            states.append(set(list(w1) + list(w2)))
                else:
                    for _ in range(n_mcs):
                        states.append(set(rng.sample(t1, l1) + rng.sample(t2, l2)))
                inds = build_indicator_matrix(net, states)
                preds = []
                for b in range(0, len(inds), 4096):
                    preds.append(model(inds[b:b + 4096], net['adj_tensor'],
                                       net['node_features']).argmax(dim=1))
                phi[(l1, l2)] = torch.cat(preds).float().mean().item()
    return phi, time.time() - t0


# ----------------------------------------------------------------------
# Adaptive framework (smooth survival-signature convergence criterion, Eq.10)
# ----------------------------------------------------------------------
def adaptive_train(net, phi_ref, n_init=None, n_set=4, sigma0=0.015,
                   max_iter=6, seed=42, verbose=True):
    rng = random.Random(seed)
    m1, m2 = net['m1'], net['m2']
    if n_init is None:
        n_init = (m1 + 1) * (m2 + 1) * 3
    states, labels = stratified_training_states(net, n_init, seed=seed)
    total_time = 0.0
    history = []
    main, phi_main = None, {}
    for it in range(max_iter):
        main, tt = train_with_states(net, states, labels, epochs=300, seed=100)
        total_time += tt
        phi_main, _ = survival_signature(net, main)
        ens = []
        for s in range(n_set):
            sub = rng.sample(range(len(states)), max(20, int(0.8 * len(states))))
            em, et = train_with_states(net, [states[i] for i in sub],
                                       [labels[i] for i in sub],
                                       epochs=200, seed=200 + s)
            total_time += et
            ep, _ = survival_signature(net, em)
            ens.append(ep)
        sigma = max(np.std([phi_main[k]] + [e[k] for e in ens]) for k in phi_main)
        ss_err = max(abs(phi_ref[k] - phi_main.get(k, 0)) for k in phi_ref)
        R_ref = network_reliability(0.5, phi_ref, m1, m2)
        R_gnn = network_reliability(0.5, phi_main, m1, m2)
        rel = abs(R_ref - R_gnn) / R_ref if R_ref else 0
        history.append(dict(iter=it, train=len(states), sigma=sigma,
                            ss_err=ss_err, rel_err=rel))
        if verbose:
            print(f"  iter {it}: train={len(states)} sigma={sigma:.4f} "
                  f"SS_err={ss_err:.4f} R_err={rel:.4f}")
        if sigma <= sigma0:
            if verbose:
                print(f"  converged (sigma={sigma:.4f})")
            break
        # add most-informative cells (highest ensemble disagreement)
        disagree = {k: np.std([e[k] for e in ens]) for k in phi_main}
        worst = sorted(disagree, key=lambda k: -disagree[k])[:8]
        seen = set(tuple(sorted(s)) for s in states)
        for (l1, l2) in worst:
            for _ in range(10):
                w1 = rng.sample(net['t1'], l1) if l1 <= len(net['t1']) else net['t1']
                w2 = rng.sample(net['t2'], l2) if l2 <= len(net['t2']) else net['t2']
                key = tuple(sorted(w1 + w2))
                if key not in seen:
                    seen.add(key); states.append(set(w1 + w2))
                    labels.append(structure_fn(net, set(w1 + w2)))
    return main, phi_main, total_time, history, states, labels


# ----------------------------------------------------------------------
# Importance measures (Proposition 1)
# ----------------------------------------------------------------------
def conditional_signature(net, comp, force):
    """
    Conditional survival signature with `comp` forced 'up' or 'down'
    (Eq. 11), computed exactly over the remaining components of its type.
    Used for the exact reference importance values.
    """
    tp = net['ctype'][comp]
    t1 = [c for c in net['t1'] if c != comp]
    t2 = [c for c in net['t2'] if c != comp]
    phi = {}
    for l1 in range(len(t1) + 1):
        for l2 in range(len(t2) + 1):
            tot = comb(len(t1), l1) * comb(len(t2), l2)
            cnt = 0
            for w1 in combinations(t1, l1):
                for w2 in combinations(t2, l2):
                    base = list(w1) + list(w2)
                    work = base + [comp] if force == 'up' else base
                    cnt += structure_fn(net, work)
            phi[(l1, l2)] = cnt / tot if tot > 0 else 0.0
    return phi, len(t1), len(t2)


def importance_measures(net, comp, t=0.5):
    """Birnbaum, Criticality, Fussell-Vesely for `comp` at time t (Eqs. 4-6)."""
    pw, n1, n2 = conditional_signature(net, comp, 'up')
    pf, _, _ = conditional_signature(net, comp, 'down')
    Rup = network_reliability(t, pw, n1, n2)
    Rdn = network_reliability(t, pf, n1, n2)
    BI = Rup - Rdn
    lam = LAMBDA_TYPE1 if net['ctype'][comp] == 1 else LAMBDA_TYPE2
    p_i = component_reliability(t, lam)
    # system reliability from the full conditional decomposition
    Rs = p_i * Rup + (1 - p_i) * Rdn
    CI = BI * p_i / Rs if Rs > 0 else 0.0
    Fs = 1 - Rs
    FVI = (Fs - (1 - Rup)) / Fs if Fs > 0 else 0.0
    return dict(BI=BI, CI=CI, FVI=FVI)


# ----------------------------------------------------------------------
# Sub-network analysis without retraining
# ----------------------------------------------------------------------
def subnetwork_signature(net, model, removed, n_mcs=2000, seed=1):
    """
    Survival signature of a sub-network (components in `removed` deleted)
    using the SAME trained model, no retraining. Removed components are
    simply held in the failed state in the indicator.
    """
    rng = random.Random(seed)
    sub_t1 = [c for c in net['t1'] if c not in removed]
    sub_t2 = [c for c in net['t2'] if c not in removed]
    phi = {}
    model.eval()
    with torch.no_grad():
        for l1 in range(len(sub_t1) + 1):
            for l2 in range(len(sub_t2) + 1):
                total = comb(len(sub_t1), l1) * comb(len(sub_t2), l2)
                states = []
                if total <= n_mcs:
                    for w1 in combinations(sub_t1, l1):
                        for w2 in combinations(sub_t2, l2):
                            states.append(set(list(w1) + list(w2)))
                else:
                    for _ in range(n_mcs):
                        states.append(set(rng.sample(sub_t1, l1) + rng.sample(sub_t2, l2)))
                inds = build_indicator_matrix(net, states)
                preds = []
                for b in range(0, len(inds), 4096):
                    preds.append(model(inds[b:b + 4096], net['adj_tensor'],
                                       net['node_features']).argmax(dim=1))
                phi[(l1, l2)] = torch.cat(preds).float().mean().item()
    return phi, sub_t1, sub_t2
