"""
State encoder: converts (query_text, cache_state) → observation vector.
Uses all-MiniLM-L6-v2 for query embeddings (384-dim, CPU-only).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List
from pathlib import Path


class StateEncoder:
    """
    Encodes query text into a fixed-size embedding using sentence-transformers.
    Falls back to zero vectors if the model can't be loaded (for testing).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embed_dim: int = 384):
        self.embed_dim = embed_dim
        self.model = None
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device="cpu")
            print(f"[StateEncoder] Loaded {model_name} (dim={embed_dim})")
        except Exception as e:
            print(f"[StateEncoder] Could not load {model_name}: {e}")
            print(f"[StateEncoder] Using zero embeddings as fallback")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single query text → (embed_dim,) float32 array."""
        if self.model is not None:
            emb = self.model.encode(text, show_progress_bar=False)
            return emb.astype(np.float32)
        return np.zeros(self.embed_dim, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts → (n, embed_dim) float32 array."""
        if self.model is not None:
            embs = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            return embs.astype(np.float32)
        return np.zeros((len(texts), self.embed_dim), dtype=np.float32)

    def encode_and_save(self, texts: List[str], save_path: str | Path) -> np.ndarray:
        """Encode texts and save to a .npy file for reuse."""
        embs = self.encode_batch(texts)
        np.save(str(save_path), embs)
        print(f"[StateEncoder] Saved {embs.shape} embeddings → {save_path}")
        return embs

    @staticmethod
    def load_embeddings(path: str | Path) -> np.ndarray:
        """Load pre-computed embeddings from a .npy file."""
        embs = np.load(str(path))
        print(f"[StateEncoder] Loaded embeddings: {embs.shape}")
        return embs
