from typing import List, Sequence
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np
from .utils import save_json, load_json

class IntentEncoderIndex:
    def __init__(self, encoder_name: str, dim: int = 768):
        self.encoder = SentenceTransformer(encoder_name)
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def build_ann(emb: np.ndarray, dim: int, trees: int, out_path: str):
        ann = AnnoyIndex(dim, "angular")
        for i, v in enumerate(emb):
            ann.add_item(i, v)
        ann.build(trees)
        ann.save(out_path)

    @staticmethod
    def load_ann(dim: int, path: str) -> AnnoyIndex:
        ann = AnnoyIndex(dim, "angular")
        ann.load(path)
        return ann

def knn_strings(ann: AnnoyIndex, emb_query: np.ndarray, vocab: List[str], k: int) -> List[str]:
    idxs = ann.get_nns_by_vector(emb_query, k)
    return [vocab[i] for i in idxs]
