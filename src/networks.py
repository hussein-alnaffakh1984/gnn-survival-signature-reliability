"""
networks.py
-----------
Network construction and exact/Monte-Carlo survival-signature computation
for the unified GNN survival-signature framework.

All networks are two-terminal (source S, terminal T). Failable components are
nodes of two types; S, T, and any intermediate buses are perfectly reliable.
"""
import numpy as np
import networkx as nx
import random
import time
from itertools import combinations
from math import comb
from scipy.stats import binom

# Default component failure rates (exponential) for the two types
LAMBDA_TYPE1 = 0.8
LAMBDA_TYPE2 = 1.5


# ----------------------------------------------------------------------
# Network context
# ----------------------------------------------------------------------
def make_network_context(G, ctype, comps):
    """
    Build a context dict for a network.

    Parameters
    ----------
    G : networkx.Graph with nodes 'S', 'T', component ids, and optional buses
    ctype : dict mapping component id -> type (1 or 2)
    comps : sorted list of failable component ids

    Returns
    -------
    dict with adjacency, node index, type partitions, etc.
    Auto-detects intermediate (non-component) nodes such as IEEE buses.
    """
    t1 = sorted([c for c in comps if ctype[c] == 1])
    t2 = sorted([c for c in comps if ctype[c] == 2])
    other = sorted([n for n in G.nodes() if n not in ('S', 'T') and n not in comps],
                   key=lambda x: str(x))
    nodes = ['S'] + list(comps) + other + ['T']
    nidx = {n: i for i, n in enumerate(nodes)}
    nn = len(nodes)
    A = np.zeros((nn, nn))
    for u, v in G.edges():
        A[nidx[u], nidx[v]] = 1
        A[nidx[v], nidx[u]] = 1
    return dict(G=G, ctype=ctype, comps=comps, t1=t1, t2=t2,
                nodes=nodes, nidx=nidx, nn=nn, A=A,
                m1=len(t1), m2=len(t2))


def structure_fn(net, working):
    """Structure function: 1 if S-T connected with `working` components up."""
    failed = set(net['comps']) - set(working)
    H = net['G'].copy()
    H.remove_nodes_from(failed)
    if 'S' not in H or 'T' not in H:
        return 0
    return 1 if nx.has_path(H, 'S', 'T') else 0


# ----------------------------------------------------------------------
# Reference survival signatures
# ----------------------------------------------------------------------
def exact_survival_signature(net):
    """Exact survival signature by complete enumeration (small networks)."""
    t1, t2, m1, m2 = net['t1'], net['t2'], net['m1'], net['m2']
    phi = {}
    for l1 in range(m1 + 1):
        for l2 in range(m2 + 1):
            tot = comb(m1, l1) * comb(m2, l2)
            cnt = 0
            for w1 in combinations(t1, l1):
                for w2 in combinations(t2, l2):
                    cnt += structure_fn(net, list(w1) + list(w2))
            phi[(l1, l2)] = cnt / tot if tot > 0 else 0.0
    return phi


def mcs_survival_signature(net, n_mcs=2000, seed=1):
    """
    Monte-Carlo survival signature (large networks where enumeration is
    infeasible). Cells with <= n_mcs combinations are enumerated exactly.
    Returns (phi, elapsed_seconds).
    """
    rng = random.Random(seed)
    t1, t2, m1, m2 = net['t1'], net['t2'], net['m1'], net['m2']
    phi = {}
    t0 = time.time()
    for l1 in range(m1 + 1):
        for l2 in range(m2 + 1):
            total = comb(m1, l1) * comb(m2, l2)
            if total <= n_mcs:
                cnt = 0
                for w1 in combinations(t1, l1):
                    for w2 in combinations(t2, l2):
                        cnt += structure_fn(net, list(w1) + list(w2))
                phi[(l1, l2)] = cnt / total if total > 0 else 0.0
            else:
                cnt = 0
                for _ in range(n_mcs):
                    w1 = rng.sample(t1, l1)
                    w2 = rng.sample(t2, l2)
                    cnt += structure_fn(net, w1 + w2)
                phi[(l1, l2)] = cnt / n_mcs
    return phi, time.time() - t0


# ----------------------------------------------------------------------
# Reliability from survival signature
# ----------------------------------------------------------------------
def component_reliability(t, lam):
    return np.exp(-lam * t)


def network_reliability(t, phi, m1, m2,
                        lam1=LAMBDA_TYPE1, lam2=LAMBDA_TYPE2):
    """R(t) = sum over cells of Phi * binomial weights (Eq. 3)."""
    p1 = component_reliability(t, lam1)
    p2 = component_reliability(t, lam2)
    R = 0.0
    for l1 in range(m1 + 1):
        for l2 in range(m2 + 1):
            R += phi.get((l1, l2), 0.0) * binom.pmf(l1, m1, p1) * binom.pmf(l2, m2, p2)
    return R


# ----------------------------------------------------------------------
# Benchmark network builders
# ----------------------------------------------------------------------
def build_benchmark6():
    """6-component two-terminal benchmark (Section 5.1)."""
    G = nx.Graph()
    G.add_edges_from([('S', 1), ('S', 2), (1, 3), (1, 4), (2, 3), (2, 4),
                      (3, 5), (3, 6), (4, 5), (4, 6), (5, 'T'), (6, 'T')])
    ctype = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 2}
    return make_network_context(G, ctype, [1, 2, 3, 4, 5, 6])


def build_complex13():
    """13-component complex irregular network (Sections 5.5, 5.6)."""
    G = nx.Graph()
    G.add_edges_from([('S', 1), ('S', 2), ('S', 3), (1, 4), (2, 4), (2, 5),
                      (3, 5), (3, 6), (4, 7), (5, 7), (5, 8), (6, 8), (6, 9),
                      (7, 10), (8, 10), (8, 11), (9, 11), (10, 12), (11, 12),
                      (11, 13), (12, 'T'), (13, 'T'),
                      (1, 5), (4, 8), (7, 11), (9, 13)])
    ctype = {1: 1, 2: 2, 3: 1, 4: 1, 5: 2, 6: 1, 7: 2, 8: 1, 9: 2,
             10: 2, 11: 1, 12: 1, 13: 2}
    return make_network_context(G, ctype, list(range(1, 14)))


def build_scalable_network(n_comp, seed=7):
    """Layered scalable network with n_comp components (Section 5.4)."""
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    comps = list(range(1, n_comp + 1))
    n_layers = max(3, int(np.sqrt(n_comp)))
    layers = np.array_split(comps, n_layers)
    for c in layers[0]:
        G.add_edge('S', c)
    for c in layers[-1]:
        G.add_edge(c, 'T')
    for li in range(len(layers) - 1):
        cur, nxt = layers[li], layers[li + 1]
        for c in cur:
            k = rng.randint(2, min(4, len(nxt) + 1))
            for t in rng.choice(nxt, size=min(k, len(nxt)), replace=False):
                G.add_edge(int(c), int(t))
        for t in nxt:
            if not any(G.has_edge(c, t) for c in cur):
                G.add_edge(int(rng.choice(cur)), int(t))
    for _ in range(n_comp // 5):
        a, b = rng.choice(comps, 2, replace=False)
        if abs(a - b) > 2:
            G.add_edge(int(a), int(b))
    ctype = {c: (1 if (c + (c // 3)) % 2 == 0 else 2) for c in comps}
    return make_network_context(G, ctype, comps)


def build_ieee14_buses():
    """
    IEEE 14-bus power network, bus-failure representation (Section 5.9).
    Buses 2..13 are failable components; lines are edges.
    """
    branches = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5),
                (4, 7), (4, 9), (5, 6), (6, 11), (6, 12), (6, 13), (7, 8),
                (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14)]
    G = nx.Graph()
    G.add_edges_from(branches)
    G = nx.relabel_nodes(G, {1: 'S', 14: 'T'})
    comps = sorted([n for n in G.nodes() if n not in ('S', 'T')])
    hv = {2, 4, 5, 6, 7, 9}
    ctype = {c: (1 if c in hv else 2) for c in comps}
    return make_network_context(G, ctype, comps)


def build_complex13_variant(seed=3):
    """
    A structurally different 13-node network (same size as build_complex13)
    used for the exploratory cross-topology generalization test (Section 5.10).
    """
    G = nx.Graph()
    edges = [('S', 1), ('S', 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5),
             (4, 6), (5, 7), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10),
             (9, 11), (10, 11), (10, 12), (11, 13), (12, 13), (12, 'T'),
             (13, 'T'), (1, 4), (3, 6), (5, 8), (9, 12)]
    G.add_edges_from(edges)
    ctype = {i: (1 if i % 2 == 1 else 2) for i in range(1, 14)}
    return make_network_context(G, ctype, list(range(1, 14)))


def build_ieee30_buses():
    """
    IEEE 30-bus test system, bus-failure representation (Section 5.10).
    Buses 2..29 are failable components; lines are edges. Source = bus 1,
    terminal = bus 30. Requires a high-fidelity Monte Carlo reference
    (2^28 states; enumeration infeasible).
    """
    branches = [(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (2, 6), (4, 6),
                (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (9, 11), (9, 10),
                (4, 12), (12, 13), (12, 14), (12, 15), (12, 16), (14, 15),
                (16, 17), (15, 18), (18, 19), (19, 20), (10, 20), (10, 17),
                (10, 21), (10, 22), (21, 22), (15, 23), (22, 24), (23, 24),
                (24, 25), (25, 26), (25, 27), (28, 27), (27, 29), (27, 30),
                (29, 30), (8, 28), (6, 28)]
    G = nx.Graph()
    G.add_edges_from(branches)
    G = nx.relabel_nodes(G, {1: 'S', 30: 'T'})
    comps = sorted([n for n in G.nodes() if n not in ('S', 'T')])
    hv_buses = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    ctype = {c: (1 if c in hv_buses else 2) for c in comps}
    return make_network_context(G, ctype, comps)
