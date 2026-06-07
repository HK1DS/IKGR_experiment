import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Robust imports for different RecBole versions
try:
    from recbole.model.general_recommender import GeneralRecommender
except Exception:
    try:
        from recbole.model.abstract_recommender import GeneralRecommender  # very old fallback
    except Exception as e:
        raise ImportError(
            "Cannot import GeneralRecommender from RecBole. "
            "Please upgrade RecBole (pip install 'recbole>=1.2.0')."
        ) from e

try:
    from recbole.utils import InputType
except Exception:
    # Fallback enum stub if very old RecBole; PAIRWISE is common in BPR-like training
    class InputType:
        PAIRWISE = type("X", (), {"value": "pairwise"})
        POINTWISE = type("X", (), {"value": "pointwise"})

# --- Layers
class KGCNLayer(nn.Module):
    """
    Implements Eq.(kgcn):
    v_out = tanh( W * [v + softmax(R[S_v]·E[S_v]) E[S_v] ] + b )
    neighbors[v] -> List[(nbr_id, rel_id)].
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, E: torch.Tensor, R: torch.Tensor, neighbors):
        N, d = E.shape
        device = E.device
        out = torch.zeros_like(E)
        for v in range(N):
            if len(neighbors[v]) == 0:
                agg = torch.zeros(d, device=device)
            else:
                nbr_idx = [n for n, _ in neighbors[v]]
                rel_idx = [r for _, r in neighbors[v]]
                nbr_e = E[nbr_idx]                     # [k, d]
                nbr_r = R[rel_idx]                     # [k, d]
                scores = (nbr_r * nbr_e).sum(dim=1)    # [k]
                alpha = torch.softmax(scores, dim=0)   # [k]
                agg = (alpha.unsqueeze(1) * nbr_e).sum(dim=0)  # [d]
            fused = E[v] + agg
            out[v] = torch.tanh(self.proj(fused) + self.bias)
        return out


class TransHScore(nn.Module):
    """
    f(e_h, e_t, r) = || e_h^⊥ + r - e_t^⊥ ||_2
    e^⊥ = e - (e·w) w, with w normalized.
    """
    def __init__(self, embed_dim: int, num_rel: int):
        super().__init__()
        self.r = nn.Embedding(num_rel, embed_dim)
        self.w = nn.Embedding(num_rel, embed_dim)
        nn.init.xavier_uniform_(self.r.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def project(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return e - (e * w).sum(dim=-1, keepdim=True) * w

    def triplet_energy(self, e_h, e_t, rel_ids):
        r = self.r(rel_ids)
        w = F.normalize(self.w(rel_ids), dim=-1)
        e_h_p = self.project(e_h, w)
        e_t_p = self.project(e_t, w)
        return torch.norm(e_h_p + r - e_t_p, p=2, dim=-1)


def intent_aware_score(e_u: torch.Tensor,
                       e_i: torch.Tensor,
                       Z_u: torch.Tensor,
                       Z_i: torch.Tensor,
                       lam: float = 0.1) -> torch.Tensor:
    """
    z_{u,i} = cos(e_u, e_i)
    y_{u,i} = max_{nu in Ωu, ni in Ωi} cos(e_nu, e_ni) * penalty
    score = y_{u,i} + λ z_{u,i}
    """
    e_u_n, e_i_n = F.normalize(e_u, dim=-1), F.normalize(e_i, dim=-1)
    z_ui = (e_u_n * e_i_n).sum(dim=-1)

    if Z_u.numel() == 0 or Z_i.numel() == 0:
        y_ui = torch.zeros_like(z_ui)
    else:
        Zu_n = F.normalize(Z_u, dim=-1)
        Zi_n = F.normalize(Z_i, dim=-1)
        sims = Zu_n @ Zi_n.T
        # take global max similarity as proxy
        y_val = sims.max()
        # simple overlap proxy: if max>0.9, treat as overlapping, else apply 0.5 penalty
        penalty = 1.0 if float(y_val.item()) > 0.9 else 0.5
        y_ui = y_val * penalty

    return y_ui + lam * z_ui


def _cfg(config, key: str, default):
    """Safe access for RecBole Config: supports `key in config` + `config[key]`."""
    return config[key] if key in config else default


class IKGR(GeneralRecommender):
    """
    IKGR (intent knowledge-graph recommender), redesigned to be LEARNABLE and to
    actually use the intent KG:

      - Base learnable user/item embeddings (like MF/BPR).
      - Shared learnable INTENT NODES initialized from the LLM/mpnet intent
        embeddings (run/kg_pack.pt) via a learnable projection.
      - One vectorized KGCN-style propagation layer: each user/item embedding is
        enriched by the (row-normalized) mean of its connected intent-node
        embeddings -> u' = e_u + a * (A_u_hat @ E_intent), i' = e_i + a * ...
      - Score = <u', i'> (inner product), trained with BPR. No frozen heuristic.

    Set `use_kg=False` to disable propagation (-> plain MF/BPR), which gives the
    KG-on vs KG-off ablation with everything else identical.

    Why the rewrite: the previous version scored y_ui (frozen max-cosine over
    intent pairs, range ~0.5-1.0) + 0.1*cos(emb). The frozen term dominated and
    saturated (a 0.9 penalty cliff tied hundreds of items at the top per user),
    so the learnable signal was suppressed and it ranked below Pop.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embed_dim = int(_cfg(config, "embedding_size", 64))
        self.dropout = float(_cfg(config, "dropout_prob", 0.1))
        self.reg_weight = float(_cfg(config, "reg_weight", 1e-6))
        self.use_kg = bool(_cfg(config, "use_kg", True))
        self.kg_pack_path = _cfg(config, "kg_pack_path", "run/kg_pack.pt")

        n_users, n_items = self.n_users, self.n_items
        self.user_embedding = nn.Embedding(n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(n_items, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.dropout_layer = nn.Dropout(self.dropout)

        self._kg_ready = False
        if self.use_kg:
            self._build_kg(dataset)

    # ---- KG construction (intent nodes + sparse user/item -> intent adjacency)
    def _build_kg(self, dataset):
        pack = torch.load(self.kg_pack_path, map_location="cpu")
        intent_emb = pack["intent_emb"].float()      # [n_intents, in_dim]
        self.n_intents = int(pack["n_intents"])
        in_dim = intent_emb.shape[1]

        # Frozen semantic init + learnable projection to embed_dim.
        self.register_buffer("intent_emb_raw", intent_emb)
        self.intent_proj = nn.Linear(in_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.intent_proj.weight)
        # Learnable scalar gate on the intent contribution.
        self.intent_alpha = nn.Parameter(torch.tensor(1.0))

        u_tok2id = dataset.field2token_id[self.USER_ID]
        i_tok2id = dataset.field2token_id[self.ITEM_ID]

        def build_adj(token_intents, tok2id, n_rows):
            rows, cols, vals = [], [], []
            for tok, ids in token_intents.items():
                rid = tok2id.get(str(tok))
                if rid is None:
                    continue
                ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
                k = len(ids)
                if k == 0:
                    continue
                w = 1.0 / k  # row-normalized -> mean of connected intent nodes
                for j in ids:
                    rows.append(rid); cols.append(j); vals.append(w)
            idx = torch.tensor([rows, cols], dtype=torch.long)
            val = torch.tensor(vals, dtype=torch.float32)
            return idx, val

        au_idx, au_val = build_adj(pack["user_intents"], u_tok2id, self.n_users)
        ai_idx, ai_val = build_adj(pack["item_intents"], i_tok2id, self.n_items)
        # Store COO pieces as buffers so .to(device) moves them with the model.
        self.register_buffer("A_u_idx", au_idx)
        self.register_buffer("A_u_val", au_val)
        self.register_buffer("A_i_idx", ai_idx)
        self.register_buffer("A_i_val", ai_val)
        self._kg_ready = True

    def _propagate(self):
        """Return (u_all, i_all): [n_users, d], [n_items, d] enriched embeddings."""
        if not (self.use_kg and self._kg_ready):
            return self.user_embedding.weight, self.item_embedding.weight
        E_int = self.intent_proj(self.intent_emb_raw)  # [n_intents, d]
        A_u = torch.sparse_coo_tensor(self.A_u_idx, self.A_u_val,
                                      (self.n_users, self.n_intents))
        A_i = torch.sparse_coo_tensor(self.A_i_idx, self.A_i_val,
                                      (self.n_items, self.n_intents))
        u_agg = torch.sparse.mm(A_u, E_int)            # [n_users, d]
        i_agg = torch.sparse.mm(A_i, E_int)            # [n_items, d]
        u_all = self.user_embedding.weight + self.intent_alpha * u_agg
        i_all = self.item_embedding.weight + self.intent_alpha * i_agg
        return u_all, i_all

    def forward(self, user, item):
        u_all, i_all = self._propagate()
        u = self.dropout_layer(u_all[user])
        i = self.dropout_layer(i_all[item])
        return (u * i).sum(dim=-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos = interaction[self.ITEM_ID]
        u_all, i_all = self._propagate()
        u = u_all[user]
        pos_e = i_all[pos]
        pos_score = (u * pos_e).sum(dim=-1)

        has_neg = hasattr(self, "NEG_ITEM_ID") and (self.NEG_ITEM_ID in interaction)
        if has_neg:
            neg = interaction[self.NEG_ITEM_ID]
            neg_e = i_all[neg]
            neg_score = (u * neg_e).sum(dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2))
        else:
            if hasattr(self, "LABEL"):
                label_field = self.LABEL
            elif "LABEL_FIELD" in self.config:
                label_field = self.config["LABEL_FIELD"]
            else:
                raise KeyError("Pointwise training requires LABEL_FIELD in config.")
            label = interaction[label_field].float()
            loss = F.binary_cross_entropy_with_logits(pos_score, label)
            reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2))

        return loss + self.reg_weight * reg / user.shape[0]

    def predict(self, interaction):
        u_all, i_all = self._propagate()
        u = u_all[interaction[self.USER_ID]]
        i = i_all[interaction[self.ITEM_ID]]
        return (u * i).sum(dim=-1)

    def full_sort_predict(self, interaction):
        u_all, i_all = self._propagate()
        u = u_all[interaction[self.USER_ID]]      # [B, d]
        return torch.matmul(u, i_all.t())          # [B, n_items]

