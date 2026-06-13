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
        self.kg_layers = int(_cfg(config, "kg_layers", 1))
        self.intent_learnable = bool(_cfg(config, "intent_learnable", True))
        self.kg_cap = int(_cfg(config, "kg_cap", 64))
        # Heterogeneous metadata KG (brand/category/attribute), LLM-free.
        self.use_meta_kg = bool(_cfg(config, "use_meta_kg", False))
        self.meta_kg_path = _cfg(config, "meta_kg_path", "run/meta_kg_pack.pt")
        self._meta_ready = False

        n_users, n_items = self.n_users, self.n_items
        self.user_embedding = nn.Embedding(n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(n_items, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.dropout_layer = nn.Dropout(self.dropout)

        self._kg_ready = False
        if self.use_kg:
            self._build_kg(dataset)
        if self.use_meta_kg:
            self._build_meta_kg(dataset)

    # ---- KG construction (learnable intent nodes + normalized user/item<->intent adjacency)
    def _build_kg(self, dataset):
        pack = torch.load(self.kg_pack_path, map_location="cpu")
        raw = pack["intent_emb"].float()             # [n_intents, in_dim] (mpnet)
        self.n_intents = int(pack["n_intents"])
        in_dim = raw.shape[1]
        d = self.embed_dim

        # Intent NODE embeddings, initialized from a fixed random projection of
        # the LLM/mpnet intent vectors (preserves semantic geometry, JL lemma),
        # then learned jointly (genuine shared KG nodes).
        if in_dim == d:
            init = raw.clone()
        else:
            g = torch.Generator().manual_seed(12345)
            P = torch.randn(in_dim, d, generator=g) / (d ** 0.5)
            init = raw @ P                           # [n_intents, d]
        self.intent_embedding = nn.Embedding(self.n_intents, d)
        self.intent_embedding.weight.data.copy_(init)
        if not self.intent_learnable:
            self.intent_embedding.weight.requires_grad_(False)
        self.intent_alpha = nn.Parameter(torch.tensor(1.0))

        u_tok2id = dataset.field2token_id[self.USER_ID]
        i_tok2id = dataset.field2token_id[self.ITEM_ID]

        def edges(token_intents, tok2id):
            rs, cs = [], []
            for tok, ids in token_intents.items():
                rid = tok2id.get(str(tok))
                if rid is None:
                    continue
                ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
                for j in ids:
                    rs.append(rid); cs.append(int(j))
            return rs, cs

        ur, uc = edges(pack["user_intents"], u_tok2id)   # user<->intent edges
        ir, ic = edges(pack["item_intents"], i_tok2id)   # item<->intent edges

        def norm_coo(rows, cols, n_rows):
            """Row-normalized COO: value = 1/deg(row)."""
            rt = torch.tensor(rows, dtype=torch.long)
            ct = torch.tensor(cols, dtype=torch.long)
            deg = torch.zeros(n_rows)
            deg.index_add_(0, rt, torch.ones(len(rows)))
            val = 1.0 / deg[rt].clamp(min=1.0)
            return torch.stack([rt, ct]), val

        # Mu : user-normalized  user->intent  (user gathers mean of its intents)
        # MuT: intent-normalized intent->user (intent gathers mean of its users)
        mats = {
            "Mu":  norm_coo(ur, uc, self.n_users),
            "Mi":  norm_coo(ir, ic, self.n_items),
            "MuT": norm_coo(uc, ur, self.n_intents),
            "MiT": norm_coo(ic, ir, self.n_intents),
        }
        for name, (idx, val) in mats.items():
            self.register_buffer(f"{name}_idx", idx)
            self.register_buffer(f"{name}_val", val)

        # Padded intent-id tensors for FAST mini-batch L1 aggregation (gather the
        # batch's intent-node embeddings directly, avoiding a full-graph sparse
        # mm every step). Equivalent to Mu/Mi row-mean but only for batch rows.
        def padded(rows, cols, n_rows, cap):
            from collections import defaultdict
            d = defaultdict(list)
            for r, c in zip(rows, cols):
                if len(d[r]) < cap:
                    d[r].append(c)
            ids = torch.zeros(n_rows, cap, dtype=torch.long)
            mask = torch.zeros(n_rows, cap, dtype=torch.bool)
            for r, lst in d.items():
                k = len(lst)
                ids[r, :k] = torch.tensor(lst, dtype=torch.long)
                mask[r, :k] = True
            return ids, mask
        ui, um = padded(ur, uc, self.n_users, self.kg_cap)
        ii, im = padded(ir, ic, self.n_items, self.kg_cap)
        self.register_buffer("user_intent_ids", ui)
        self.register_buffer("user_intent_mask", um)
        self.register_buffer("item_intent_ids", ii)
        self.register_buffer("item_intent_mask", im)
        self._kg_ready = True

    def _sp(self, name, shape):
        return torch.sparse_coo_tensor(getattr(self, f"{name}_idx"),
                                       getattr(self, f"{name}_val"), shape)

    def _agg(self, emb, ids_buf, mask_buf, rows):
        ids = ids_buf[rows]                                  # [B, cap]
        m = mask_buf[rows].unsqueeze(-1).float()             # [B, cap, 1]
        e = emb(ids) * m                                     # [B, cap, d]
        return e.sum(1) / m.sum(1).clamp(min=1.0)            # [B, d]

    def _build_meta_kg(self, dataset):
        """Heterogeneous item-side KG from book metadata (brand/category/attr).
        Learnable id-node embeddings + padded item->node gather (LLM-free)."""
        pack = torch.load(self.meta_kg_path, map_location="cpu")
        i_tok2id = dataset.field2token_id[self.ITEM_ID]
        d, cap = self.embed_dim, self.kg_cap

        def build(rel_key, n_key, emb_name, ids_name, mask_name, alpha_name):
            n = max(int(pack[n_key]), 1)
            emb = nn.Embedding(n, d); nn.init.xavier_uniform_(emb.weight)
            setattr(self, emb_name, emb)
            setattr(self, alpha_name, nn.Parameter(torch.tensor(1.0)))
            ids = torch.zeros(self.n_items, cap, dtype=torch.long)
            mask = torch.zeros(self.n_items, cap, dtype=torch.bool)
            present = False
            for tok, node_ids in pack[rel_key].items():
                rid = i_tok2id.get(str(tok))
                if rid is None:
                    continue
                lst = (node_ids.tolist() if torch.is_tensor(node_ids) else list(node_ids))[:cap]
                if not lst:
                    continue
                present = True
                ids[rid, :len(lst)] = torch.tensor(lst, dtype=torch.long)
                mask[rid, :len(lst)] = True
            self.register_buffer(ids_name, ids)
            self.register_buffer(mask_name, mask)
            return present

        self._has_author = build("item_authors", "n_authors", "author_embedding",
                                  "item_author_ids", "item_author_mask", "author_alpha")
        self._has_pub = build("item_publishers", "n_publishers", "publisher_embedding",
                              "item_pub_ids", "item_pub_mask", "pub_alpha")
        self._has_shelf = build("item_shelves", "n_shelves", "shelf_embedding",
                                "item_shelf_ids", "item_shelf_mask", "shelf_alpha")
        self._meta_ready = True

    def _emb_users(self, uids):
        e = self.user_embedding(uids)
        if self.use_kg and self._kg_ready:
            if self.kg_layers == 1:
                e = e + self.intent_alpha * self._agg(self.intent_embedding,
                                                      self.user_intent_ids, self.user_intent_mask, uids)
            else:
                e = self._propagate()[0][uids]
        return e

    def _emb_items(self, iids):
        e = self.item_embedding(iids)
        if self.use_kg and self._kg_ready and self.kg_layers >= 2:
            return self._propagate()[1][iids]   # intent-only L2 path
        if self.use_kg and self._kg_ready:
            e = e + self.intent_alpha * self._agg(self.intent_embedding,
                                                  self.item_intent_ids, self.item_intent_mask, iids)
        if self.use_meta_kg and self._meta_ready:
            if self._has_author:
                e = e + self.author_alpha * self._agg(self.author_embedding,
                                                      self.item_author_ids, self.item_author_mask, iids)
            if self._has_pub:
                e = e + self.pub_alpha * self._agg(self.publisher_embedding,
                                                   self.item_pub_ids, self.item_pub_mask, iids)
            if self._has_shelf:
                e = e + self.shelf_alpha * self._agg(self.shelf_embedding,
                                                     self.item_shelf_ids, self.item_shelf_mask, iids)
        return e

    def _propagate(self):
        """Return (u_all, i_all): KG-enriched user/item embeddings.

        L=1: users/items gather a mean of their connected intent nodes.
        L=2: intents additionally gather from their users+items, then propagate
             back -> user->intent->item collaborative signal flows across hops.
        """
        E_u = self.user_embedding.weight
        E_i = self.item_embedding.weight
        if not (self.use_kg and self._kg_ready):
            return E_u, E_i
        ni, nu, nit = self.n_intents, self.n_users, self.n_items
        Mu = self._sp("Mu", (nu, ni)); Mi = self._sp("Mi", (nit, ni))
        E_int = self.intent_embedding.weight

        u_acc = torch.sparse.mm(Mu, E_int)           # [n_users, d]
        i_acc = torch.sparse.mm(Mi, E_int)           # [n_items, d]
        if self.kg_layers >= 2:
            MuT = self._sp("MuT", (ni, nu)); MiT = self._sp("MiT", (ni, nit))
            int1 = torch.sparse.mm(MuT, E_u) + torch.sparse.mm(MiT, E_i)  # intents <- users+items
            u_acc = u_acc + torch.sparse.mm(Mu, int1)
            i_acc = i_acc + torch.sparse.mm(Mi, int1)

        u_all = E_u + self.intent_alpha * u_acc
        i_all = E_i + self.intent_alpha * i_acc
        return u_all, i_all

    def forward(self, user, item):
        u = self.dropout_layer(self._emb_users(user))
        i = self.dropout_layer(self._emb_items(item))
        return (u * i).sum(dim=-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos = interaction[self.ITEM_ID]
        u = self._emb_users(user)
        pos_e = self._emb_items(pos)
        pos_score = (u * pos_e).sum(dim=-1)

        has_neg = hasattr(self, "NEG_ITEM_ID") and (self.NEG_ITEM_ID in interaction)
        if has_neg:
            neg = interaction[self.NEG_ITEM_ID]
            neg_e = self._emb_items(neg)
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
        u = self._emb_users(interaction[self.USER_ID])
        i = self._emb_items(interaction[self.ITEM_ID])
        return (u * i).sum(dim=-1)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        if self.use_kg and self._kg_ready and self.kg_layers >= 2:
            u_all, i_all = self._propagate()
            return torch.matmul(u_all[users], i_all.t())
        u = self._emb_users(users)                                   # [B, d]
        all_items = torch.arange(self.n_items, device=users.device)
        i_all = self._emb_items(all_items)                           # [n_items, d]
        return torch.matmul(u, i_all.t())

