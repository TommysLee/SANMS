"""
Retrieval engine using cosine similarity.

Supports both brute-force and Faiss-based nearest-neighbor search.
"""

import numpy as np
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class Retriever:
    """Image retrieval using cosine similarity.

    For L2-normalized embeddings, cosine similarity = inner product.
    Uses Faiss for efficient search when available.
    """

    def __init__(self, embedding_dim: int = 512, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.gallery_embeddings = None
        self.gallery_labels = None
        self.index = None

    def build_index(self, gallery_embeddings: np.ndarray, gallery_labels: np.ndarray):
        """Build the gallery index.

        Args:
            gallery_embeddings: (G, D) L2-normalized.
            gallery_labels: (G,) class labels.
        """
        self.gallery_embeddings = gallery_embeddings.astype(np.float32)
        self.gallery_labels = np.asarray(gallery_labels)

        if self.use_faiss:
            # Inner product index (works for L2-normalized = cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.gallery_embeddings)
        # else: brute-force (numpy)

    def search(self, query_embeddings: np.ndarray, k: int = 100) -> tuple:
        """Search for top-K matches.

        Args:
            query_embeddings: (Q, D) L2-normalized.
            k: number of nearest neighbors.

        Returns:
            indices: (Q, K) gallery indices.
            similarities: (Q, K) cosine similarities.
        """
        query_embeddings = query_embeddings.astype(np.float32)

        if self.use_faiss:
            similarities, indices = self.index.search(query_embeddings, k)
        else:
            # Brute-force
            sim = query_embeddings @ self.gallery_embeddings.T
            if k >= sim.shape[1]:
                indices = np.argsort(-sim, axis=1)
                similarities = -np.sort(-sim, axis=1)
            else:
                # argpartition for top-K
                indices = np.argpartition(-sim, k, axis=1)[:, :k]
                # Sort within top-K
                for i in range(len(query_embeddings)):
                    order = np.argsort(-sim[i, indices[i]])
                    indices[i] = indices[i][order]
                    similarities = sim[np.arange(len(query_embeddings))[:, None], indices]

        return indices, similarities

    def recall_at_k(
        self,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        k_values: list = [1, 10, 100],
    ) -> dict:
        """Compute Recall@K using the gallery index.

        Args:
            query_embeddings: (Q, D) L2-normalized.
            query_labels: (Q,)
            k_values: list of K values.

        Returns:
            dict 'R@K' -> recall %.
        """
        max_k = max(k_values)
        max_k = min(max_k, len(self.gallery_embeddings))
        indices, _ = self.search(query_embeddings, max_k)

        results = {}
        for k in k_values:
            k = min(k, indices.shape[1])
            top_k_labels = self.gallery_labels[indices[:, :k]]
            correct = (top_k_labels == query_labels[:, None]).any(axis=1)
            results[f"R@{k}"] = correct.mean() * 100

        return results

    def top1_accuracy(
        self, query_embeddings: np.ndarray, query_labels: np.ndarray
    ) -> float:
        """Top-1 accuracy."""
        indices, _ = self.search(query_embeddings, 1)
        top1_labels = self.gallery_labels[indices[:, 0]]
        return (top1_labels == query_labels).mean() * 100
