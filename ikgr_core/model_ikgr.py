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
    Minimal IKGR model:
      - user/item base embeddings
      - one KGCN layer touching global entity table (neighbors should be loaded in production)
      - TransH params present (you can add KG loss with real triples)
      - intent-aware hybrid scoring at inference
    """
    # choose pairwise by default (BPR-like); switch to POINTWISE if needed
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embed_dim = int(_cfg(config, "embedding_size", 64))
        self.lambda_mix = float(_cfg(config, "lambda_mix", 0.1))
        self.dropout = float(_cfg(config, "dropout_prob", 0.1))
        # Memory-safety knobs (important on small-VRAM GPUs):
        #  - max_intents_per_node: cap intents kept per user/item (padding width).
        #    Mean ~25, p99 ~50, max ~111; cap=64 covers >99% with far less VRAM.
        #  - intent_score_chunk: rows processed per chunk in forward() to bound the
        #    [chunk, max_ku, max_ki] bmm during full-sort evaluation.
        self.max_intents = int(_cfg(config, "max_intents_per_node", 64))
        self.intent_chunk = int(_cfg(config, "intent_score_chunk", 2048))

        num_users, num_items = self.n_users, self.n_items

        # Base embeddings
        self.user_embedding = nn.Embedding(num_users, self.embed_dim)
        self.item_embedding = nn.Embedding(num_items, self.embed_dim)

        # Global entity/relation tables (toy; can be merged with user/item tables)
        self.entity_embeddings = nn.Embedding(num_users + num_items, self.embed_dim)
        self.relation_embeddings = nn.Embedding(3, self.embed_dim)  # user_has_intent, item_has_intent, user_consumes_item

        self.kgcn = KGCNLayer(self.embed_dim)
        self.transh = TransHScore(self.embed_dim, num_rel=3)

        self.dropout_layer = nn.Dropout(self.dropout)
        self._reset_parameters()

        # Placeholder neighbor list for KGCN; in production, fill from KG adjacency
        self.neighbors = [[] for _ in range(num_users + num_items)]

        # Intent banks: map internal user/item ids to intent embedding tensors [k, d]
        self.user_intent_bank = {}
        self.item_intent_bank = {}

    def _reset_parameters(self):
        for m in [self.user_embedding, self.item_embedding, self.entity_embeddings]:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    # Optional: hook to preload intent banks
    def load_intent_banks(self, user_bank: dict, item_bank: dict):
        """
        user_bank: Dict[int, torch.Tensor(k,d)]
        item_bank: Dict[int, torch.Tensor(k,d)]
        """
        self.user_intent_bank = user_bank
        self.item_intent_bank = item_bank

        # Pre-build 3D padded tensors to avoid slow python loops during training/eval
        device = self.user_embedding.weight.device

        cap = self.max_intents

        def _cap(t):
            # Keep at most `cap` intents per node to bound padding width / VRAM.
            return t[:cap] if (t is not None and t.shape[0] > cap) else t

        # 1) User Padded Tensor
        user_lists = [_cap(user_bank.get(u, torch.zeros(1, 768, device=device))) for u in range(self.n_users)]
        intent_dim = user_lists[0].shape[1] if user_lists else 768
        max_ku = max(u.shape[0] for u in user_lists) if user_lists else 1
        self.user_intents_padded = torch.zeros(self.n_users, max_ku, intent_dim, device=device)
        self.user_intents_mask = torch.zeros(self.n_users, max_ku, device=device, dtype=torch.bool)
        for u in range(self.n_users):
            k = user_lists[u].shape[0]
            if k > 0 and user_bank.get(u) is not None:
                self.user_intents_padded[u, :k, :] = user_lists[u].to(device)
                self.user_intents_mask[u, :k] = True

        # 2) Item Padded Tensor
        item_lists = [_cap(item_bank.get(i, torch.zeros(1, intent_dim, device=device))) for i in range(self.n_items)]
        max_ki = max(i.shape[0] for i in item_lists) if item_lists else 1
        self.item_intents_padded = torch.zeros(self.n_items, max_ki, intent_dim, device=device)
        self.item_intents_mask = torch.zeros(self.n_items, max_ki, device=device, dtype=torch.bool)
        for i in range(self.n_items):
            k = item_lists[i].shape[0]
            if k > 0 and item_bank.get(i) is not None:
                self.item_intents_padded[i, :k, :] = item_lists[i].to(device)
                self.item_intents_mask[i, :k] = True

    def forward(self, user, item):
        """
        user: [B] internal user ids
        item: [B] internal item ids
        """
        # Chunk evaluation input if batch size is too large to prevent VRAM OOM / paging
        batch_size = user.shape[0]
        max_chunk = self.intent_chunk
        if batch_size > max_chunk:
            scores_list = []
            for start in range(0, batch_size, max_chunk):
                end = min(start + max_chunk, batch_size)
                u_chunk = user[start:end]
                i_chunk = item[start:end]
                scores_list.append(self.forward(u_chunk, i_chunk))
            return torch.cat(scores_list, dim=0)

        u = self.user_embedding(user)  # [B, d]
        i = self.item_embedding(item)  # [B, d]
        u = self.dropout_layer(u)
        i = self.dropout_layer(i)

        # Touch KGCN parameters to let them train
        E = self.entity_embeddings.weight
        R = self.relation_embeddings.weight
        _ = self.kgcn(E, R, self.neighbors)

        # Base intent-free similarity score
        u_n, i_n = F.normalize(u, dim=-1), F.normalize(i, dim=-1)
        z_ui = (u_n * i_n).sum(dim=-1)  # [B]

        # Vectorized Intent-aware scoring
        Zu = self.user_intents_padded[user]      # [B, max_ku, d]
        Zi = self.item_intents_padded[item]      # [B, max_ki, d]

        Zu_mask = self.user_intents_mask[user]    # [B, max_ku]
        Zi_mask = self.item_intents_mask[item]    # [B, max_ki]

        # Normalize intent tensors
        Zu_n = F.normalize(Zu, dim=-1)
        Zi_n = F.normalize(Zi, dim=-1)

        # Batched matrix multiplication: [B, max_ku, d] x [B, d, max_ki] -> [B, max_ku, max_ki]
        sims = torch.bmm(Zu_n, Zi_n.transpose(1, 2))

        # Mask out padded intent slots
        mask_2d = Zu_mask.unsqueeze(2) & Zi_mask.unsqueeze(1)
        sims = torch.where(mask_2d, sims, torch.tensor(-1e9, device=sims.device))

        # Max similarity over all intent pairs
        y_val = sims.view(user.shape[0], -1).max(dim=-1)[0]  # [B]

        # Zero out invalid values
        y_val = torch.where(y_val < -1.0, torch.zeros_like(y_val), y_val)

        # Overlap penalty
        penalty = torch.where(y_val > 0.9, 1.0, 0.5)
        y_ui = y_val * penalty

        return y_ui + self.lambda_mix * z_ui

    def calculate_loss(self, interaction):
        """
        Prefer pairwise BPR if NEG_ITEM_ID exists in interaction.
        Otherwise fall back to pointwise BCE using LABEL_FIELD from config.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        pos_score = self.forward(user, item)

        # ---- Pairwise branch (BPR) if negative items are provided
        has_neg = hasattr(self, "NEG_ITEM_ID") and (self.NEG_ITEM_ID in interaction)
        if has_neg:
            neg_item = interaction[self.NEG_ITEM_ID]
            neg_score = self.forward(user, neg_item)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-12).mean()
            return loss

        # ---- Pointwise branch (BCE) otherwise
        # Robustly resolve label field name
        if hasattr(self, "LABEL"):
            label_field = self.LABEL  # some RecBole versions expose this alias
        else:
            # fall back to config
            if "LABEL_FIELD" in self.config:
                label_field = self.config["LABEL_FIELD"]
            else:
                raise KeyError(
                    "Pointwise training requires LABEL_FIELD in config, "
                    "but neither self.LABEL nor config['LABEL_FIELD'] is available."
                )

        if label_field not in interaction:
            raise KeyError(
                f"Interaction does not contain label field '{label_field}'. "
                f"Available keys: {list(interaction.keys())}"
            )

        label = interaction[label_field].float()
        return F.binary_cross_entropy_with_logits(pos_score, label)



    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        all_items = torch.arange(self.n_items, device=user.device)  # [I]
        user_expand = user.view(-1, 1).repeat(1, self.n_items).view(-1)  # [B*I]
        items_expand = all_items.unsqueeze(0).repeat(user.shape[0], 1).view(-1)  # [B*I]
        scores = self.forward(user_expand, items_expand).view(-1, self.n_items)
        return scores

