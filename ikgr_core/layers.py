import torch
import torch.nn as nn
import torch.nn.functional as F

class KGCNLayer(nn.Module):
    """
    Implements Eq.(kgcn): v_out = σ(W * [v + softmax(R[S_v] E[S_v]^T) E[S_v]] + b)
    Here we assume neighbor aggregation is precomputed as adjacency lists with typed edges.
    For simplicity, use an adjacency index list `neighbors[v] -> List[(nbr_id, rel_id)]`.
    R and E are embedding matrices; we'll index them during message passing.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, E: torch.Tensor, R: torch.Tensor, neighbors):
        """
        E: [N, d]  entity embeddings
        R: [T, d]  relation embeddings
        neighbors: List[List[Tuple[int, int]]] for each node v: [(nbr_id, rel_id), ...]
        """
        N, d = E.shape
        out = torch.zeros_like(E)
        for v in range(N):
            if len(neighbors[v]) == 0:
                agg = torch.zeros(d, device=E.device)
            else:
                nbr_e = torch.stack([E[n] for n, _ in neighbors[v]], dim=0)           # [k, d]
                nbr_r = torch.stack([R[r] for _, r in neighbors[v]], dim=0)           # [k, d]
                # attention via softmax(R E^T) E  (cosine-like by learned scale)
                scores = (nbr_r * nbr_e).sum(dim=1)                                   # [k]
                alpha = torch.softmax(scores, dim=0)
                agg = (alpha.unsqueeze(1) * nbr_e).sum(dim=0)                         # [d]
            fused = E[v] + agg
            out[v] = torch.tanh(self.proj(fused) + self.bias)
        return out

class TransHScore(nn.Module):
    """
    f(e_h, e_t, r) = || e_h^⊥ + r - e_t^⊥ ||
    with e_h^⊥ = e_h - (w_r^T e_h) w_r  ;  e_t^⊥ similarly.
    """
    def __init__(self, embed_dim: int, num_rel: int):
        super().__init__()
        self.r = nn.Embedding(num_rel, embed_dim)
        self.w = nn.Embedding(num_rel, embed_dim)
        nn.init.xavier_uniform_(self.r.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def project(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # project e to hyperplane orthogonal to w (w is unit-normalized implicitly by training)
        return e - (e * w).sum(dim=-1, keepdim=True) * w

    def triplet_energy(self, e_h, e_t, rel_ids):
        r = self.r(rel_ids)
        w = F.normalize(self.w(rel_ids), dim=-1)
        e_h_p = self.project(e_h, w)
        e_t_p = self.project(e_t, w)
        return torch.norm(e_h_p + r - e_t_p, p=2, dim=-1)  # lower is better

def intent_relation_vector(z_u: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
    """
    r^{u,i} = softmax(P^{u,i})^T D^{u,i}
    where P contains pairwise cosine similarities; D contains (z_i[q] - z_u[p]).
    Inputs:
      z_u: [m, d] intents of user
      z_i: [n, d] intents of item
    Returns:
      r_ui: [d]
    """
    if z_u.numel() == 0 or z_i.numel() == 0:
        # fallback to zero vector to avoid instability
        return torch.zeros(z_u.shape[-1] if z_u.numel() else z_i.shape[-1], device=z_u.device if z_u.numel() else z_i.device)

    z_u_n = F.normalize(z_u, dim=-1)  # [m, d]
    z_i_n = F.normalize(z_i, dim=-1)  # [n, d]
    P = torch.einsum("md,nd->mn", z_u_n, z_i_n).reshape(-1)  # [m*n]
    # build D of diffs for all pairs
    Du = z_u.unsqueeze(1).expand(-1, z_i.shape[0], -1)       # [m, n, d]
    Di = z_i.unsqueeze(0).expand(z_u.shape[0], -1, -1)       # [m, n, d]
    D = (Di - Du).reshape(-1, z_u.shape[-1])                 # [m*n, d]
    alpha = torch.softmax(P, dim=0).unsqueeze(0)             # [1, m*n]
    r_ui = alpha @ D                                         # [1, d]
    return r_ui.squeeze(0)

def intent_aware_score(e_u: torch.Tensor, e_i: torch.Tensor, Z_u: torch.Tensor, Z_i: torch.Tensor, lam: float = 0.1):
    """
    z_{u,i} = cos(e_u, e_i)
    y_{u,i} = max_{nu in Ωu, ni in Ωi} cos(e_nu, e_ni) * (0.5 if no overlap else 1.0)
    score = y_{u,i} + λ z_{u,i}
    """
    e_u_n, e_i_n = F.normalize(e_u, dim=-1), F.normalize(e_i, dim=-1)
    z_ui = (e_u_n * e_i_n).sum(dim=-1)

    if Z_u.numel() == 0 or Z_i.numel() == 0:
        y_ui = torch.zeros_like(z_ui)
    else:
        Zu_n = F.normalize(Z_u, dim=-1)       # [m, d]
        Zi_n = F.normalize(Z_i, dim=-1)       # [n, d]
        sims = Zu_n @ Zi_n.T                   # [m, n]
        y_ui = sims.max().unsqueeze(0) if sims.ndim == 2 else torch.tensor([0.0], device=e_u.device)
        # overlap check by exact intent-id equality should be done outside;
        # here we approximate: if max > 0.9 we treat as overlapping synonym
        penalty = 1.0 if float(y_ui.item()) > 0.9 else 0.5
        y_ui = y_ui * penalty
    return y_ui + lam * z_ui
