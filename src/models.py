"""
models.py
---------
Graph neural network architectures for the unified framework.

- FastReliabilityGNN : the proposed flow-aware + HGNN backbone (vectorized)
- PlainANN           : topology-agnostic baseline
- StandardGCNN       : graph convolutional baseline
- GraphSAGE, GAT     : modern graph baselines

All models map a batch of component-state indicators to a 2-class logit
(network working / failed), i.e. they learn the structure function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Proposed backbone: flow-aware aggregation + higher-order (HGNN) layer
# ----------------------------------------------------------------------
class FastReliabilityGNN(nn.Module):
    """
    Vectorized implementation that processes a batch of states at once.

    Forward inputs:
        indicators : [B, nn]  (1 for working node / always-on bus, else 0)
        adj        : [nn, nn] adjacency
        node_feat  : [nn, nn] identity node features
    Output:
        logits     : [B, 2]
    """
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.lin_agg1 = nn.Linear(num_nodes, hidden_dim)   # Eq. 7 transform
        self.W_self = nn.Linear(hidden_dim, hidden_dim)    # Eq. 8 self
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim)   # Eq. 8 neighbor
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, indicators, adj, node_feat):
        feat = self.lin_agg1(node_feat)                       # [nn, hid]
        gated = feat.unsqueeze(0) * indicators.unsqueeze(2)   # flow gating
        x = torch.einsum('ij,bjh->bih', adj, gated)           # Eq. 7
        x = F.relu(x)
        self_part = F.relu(self.W_self(x))                    # Eq. 8
        neigh = F.relu(self.W_neigh(x))
        x = self_part + torch.einsum('ij,bjh->bih', adj, neigh)
        x = F.relu(self.lin1(x))                              # Eq. 9
        x, _ = torch.max(x, dim=1)                            # max-pool
        return self.lin2(x)


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
class PlainANN(nn.Module):
    """Topology-agnostic MLP over the node indicator vector."""
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2))

    def forward(self, indicators, adj, node_feat):
        if indicators.dim() == 1:
            indicators = indicators.unsqueeze(0)
        return self.net(indicators)


class StandardGCNN(nn.Module):
    """Standard graph convolutional network (symmetric-normalized adj)."""
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.lin0 = nn.Linear(num_nodes, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, indicators, adj_norm, node_feat):
        x = self.lin0(node_feat).unsqueeze(0) * indicators.unsqueeze(2)
        x = F.relu(torch.einsum('ij,bjh->bih', adj_norm, self.W1(x)))
        x = F.relu(torch.einsum('ij,bjh->bih', adj_norm, self.W2(x)))
        x, _ = torch.max(x, dim=1)
        return self.lin2(x)


class GraphSAGE(nn.Module):
    """Simplified GraphSAGE with mean aggregation."""
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.lin0 = nn.Linear(num_nodes, hidden_dim)
        self.Ws1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wn1 = nn.Linear(hidden_dim, hidden_dim)
        self.Ws2 = nn.Linear(hidden_dim, hidden_dim)
        self.Wn2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, indicators, adj, node_feat):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_mean = adj / deg
        x = self.lin0(node_feat).unsqueeze(0) * indicators.unsqueeze(2)
        neigh = torch.einsum('ij,bjh->bih', adj_mean, x)
        x = F.relu(self.Ws1(x) + self.Wn1(neigh))
        neigh = torch.einsum('ij,bjh->bih', adj_mean, x)
        x = F.relu(self.Ws2(x) + self.Wn2(neigh))
        x, _ = torch.max(x, dim=1)
        return self.lin2(x)


class GAT(nn.Module):
    """Simplified single-head graph attention network."""
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.lin0 = nn.Linear(num_nodes, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.att1 = nn.Linear(2 * hidden_dim, 1)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.att2 = nn.Linear(2 * hidden_dim, 1)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def _attn(self, x, adj, W, att):
        B, N, H = x.shape
        h = W(x)
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        e = F.leaky_relu(att(torch.cat([hi, hj], dim=-1)).squeeze(-1))
        e = e.masked_fill(~(adj.unsqueeze(0) > 0), float('-inf'))
        alpha = torch.nan_to_num(torch.softmax(e, dim=2))
        return F.relu(torch.einsum('bij,bjh->bih', alpha, h))

    def forward(self, indicators, adj, node_feat):
        x = self.lin0(node_feat).unsqueeze(0) * indicators.unsqueeze(2)
        x = self._attn(x, adj, self.W1, self.att1)
        x = self._attn(x, adj, self.W2, self.att2)
        x, _ = torch.max(x, dim=1)
        return self.lin2(x)
