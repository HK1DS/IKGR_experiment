import pandas as pd
from typing import List, Tuple
from .utils import write_csv

def intents_to_kg_triples(step2_df: pd.DataFrame, kg_out_path: str) -> None:
    """
    Build KG triples with three relation types:
    - user_has_intent
    - item_has_intent
    - user_consumes_item (from interactions if provided)
    The triples file for Recbole is space-separated: head  relation  tail
    """
    rows: List[Tuple[str, str, str]] = []

    def _safe_eval(s):
        if isinstance(s, str) and s.strip():
            try: return list(eval(s))
            except: return []
        return []

    for _, r in step2_df.iterrows():
        uid, iid = str(r["user_id"]), str(r["item_id"])
        u_ints = _safe_eval(r.get("user_intents_related", "[]"))
        i_ints = _safe_eval(r.get("item_intents_related", "[]"))
        for it in u_ints:
            rows.append((f"u_{uid}", "user_has_intent", f"intent::{it}"))
        for it in i_ints:
            rows.append((f"i_{iid}", "item_has_intent", f"intent::{it}"))

    with open(kg_out_path, "w", encoding="utf-8") as f:
        for h, r, t in rows:
            f.write(f"{h}\t{r}\t{t}\n")
