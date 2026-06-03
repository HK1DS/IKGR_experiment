'''
Build KG + train/eval IKGR in Recbole. This step is optional if you want to run intent graph KG with other infra.
'''

import os, yaml, torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from ikgr_core.model_ikgr import IKGR as IKGRModel

import torch

def _load_bank_pt(path: str):
    """Load a saved .pt: returns dict(raw_token -> tensor[k,d]), and dim."""
    pkg = torch.load(path, map_location="cpu")
    return pkg.get("bank", {}), int(pkg.get("dim", 0))

def _dataset_token2id_map(dataset, field: str):
    """
    Build token->internal_id mapping from RecBole dataset.
    Compatible with multiple versions by trying common attributes.
    """
    # Preferred: dataset.field2token_id[field] (dict token->id)
    try:
        token2id = dataset.field2token_id[field]
        if isinstance(token2id, dict):
            return token2id
    except Exception:
        pass

    # Fallback: invert id2token list/array
    try:
        id2token = dataset.field2id_token[field]  # array/list by internal id
        token2id = {str(tok): i for i, tok in enumerate(id2token)}
        return token2id
    except Exception:
        pass

    # Last resort: use dataset.token2id if exists
    if hasattr(dataset, "token2id") and isinstance(dataset.token2id, dict) and field in dataset.token2id:
        return dataset.token2id[field]

    raise RuntimeError(f"Cannot build token->id map for field '{field}'. Check RecBole version and dataset.")

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    paths, rb = cfg["paths"], cfg["recbole"]

    config_dict = {
        "epochs": rb["epochs"],
        "metrics": rb["metrics"],
        "topk": rb["topk"],
        "embedding_size": rb["embedding_size"],
        "learning_rate": 1e-3,
        "reg_weight": 1e-6,
        "dropout_prob": rb.get("dropout", 0.1),
        "lambda_mix": rb["lambda_mix"],

        "data_path": os.path.dirname(paths["inter_file"]),  # directory contains {dataset}.inter/.kg
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LABEL_FIELD": "rating",
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": True,
        "checkpoint_dir": paths["recbole_dump"],
    }

    config = Config(model=IKGRModel, dataset=rb["dataset"], config_dict=config_dict)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    user_bank_path = cfg["paths"].get("user_bank_pt", "run/user_bank.pt")
    item_bank_path = cfg["paths"].get("item_bank_pt", "run/item_bank.pt")

    user_bank_raw, dim_u = _load_bank_pt(user_bank_path) if os.path.exists(user_bank_path) else ({}, 0)
    item_bank_raw, dim_i = _load_bank_pt(item_bank_path) if os.path.exists(item_bank_path) else ({}, 0)

    # Build token->id maps using RecBole dataset
    u_field = config["USER_ID_FIELD"] if "USER_ID_FIELD" in config else "user_id"
    i_field = config["ITEM_ID_FIELD"] if "ITEM_ID_FIELD" in config else "item_id"
    u_token2id = _dataset_token2id_map(dataset, u_field)
    i_token2id = _dataset_token2id_map(dataset, i_field)

    # Convert raw-token keyed banks into internal-id keyed banks
    user_bank_by_id = {}
    for tok, tensor in user_bank_raw.items():
        tok_s = str(tok)
        if tok_s in u_token2id:
            user_bank_by_id[u_token2id[tok_s]] = tensor

    item_bank_by_id = {}
    for tok, tensor in item_bank_raw.items():
        tok_s = str(tok)
        if tok_s in i_token2id:
            item_bank_by_id[i_token2id[tok_s]] = tensor

    print(f"[banks] users={len(user_bank_by_id)} items={len(item_bank_by_id)} (mapped to internal ids)")

    # ===== construct model and inject banks =====
    model = IKGRModel(config, train_data.dataset).to(config["device"])
    # move tensors to device
    dev = config["device"]
    user_bank_by_id = {k: v.to(dev) for k, v in user_bank_by_id.items()}
    item_bank_by_id = {k: v.to(dev) for k, v in item_bank_by_id.items()}
    model.load_intent_banks(user_bank_by_id, item_bank_by_id)


    
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    print("[valid]", best_valid_result)
    print("[test ]", test_result)

if __name__ == "__main__":
    main()
