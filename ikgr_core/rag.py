from typing import List, Sequence
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from .utils import save_json, load_json

class NearestNeighborsIndex:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.items = []
        self.nn = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute")
        self.emb = None

    def add_item(self, i: int, v: List[float]):
        self.items.append(v)

    def build(self, trees: int):
        self.emb = np.array(self.items, dtype=np.float32)
        self.nn.fit(self.emb)

    def save(self, out_path: str):
        with open(out_path, "wb") as f:
            np.save(f, self.emb)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.emb = np.load(f)
        self.nn.fit(self.emb)

    def get_nns_by_vector(self, query: np.ndarray, k: int) -> List[int]:
        query = query.reshape(1, -1)
        _, indices = self.nn.kneighbors(query, n_neighbors=k)
        return indices[0].tolist()

class IntentEncoderIndex:
    def __init__(self, encoder_name: str, dim: int = 768):
        self.encoder = SentenceTransformer(encoder_name)
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def build_ann(emb: np.ndarray, dim: int, trees: int, out_path: str):
        ann = NearestNeighborsIndex(dim)
        for i, v in enumerate(emb):
            ann.add_item(i, v.tolist())
        ann.build(trees)
        ann.save(out_path)

    @staticmethod
    def load_ann(dim: int, path: str) -> NearestNeighborsIndex:
        ann = NearestNeighborsIndex(dim)
        ann.load(path)
        return ann

def knn_strings(ann: NearestNeighborsIndex, emb_query: np.ndarray, vocab: List[str], k: int) -> List[str]:
    idxs = ann.get_nns_by_vector(emb_query, k)
    return [vocab[i] for i in idxs]
