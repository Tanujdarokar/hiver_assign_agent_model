"""
Phase 8 — Historical Response Retrieval Module
==============================================
Retrieves top-k historical AppleSupport customer-agent conversation pairs
using vector embeddings and FAISS / NearestNeighbors.
Guarantees zero data leakage by excluding Golden Set conversations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "apple_support_processed.csv"
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
INDEX_PATH = ROOT / "data" / "processed" / "retriever_index.pkl"


class HistoricalRetriever:
    """Vector Retrieval System for historical AppleSupport resolution grounding."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = None
        self.nn_model = None
        self.corpus_df = None
        self.is_built = False

    def build_index(self, max_samples: int = 15000):
        """
        Builds vector index on historical dataset EXCLUDING Golden Set items.
        """
        print("Building Historical Response Retrieval Index ...")
        proc_df = pd.read_csv(PROCESSED_CSV)
        golden_df = pd.read_csv(GOLDEN_CSV)

        # 1. Leakage Prevention: Exclude Golden Set conversations
        golden_ids = set(golden_df["conversation_id"].astype(str))
        corpus_df = proc_df[~proc_df["conversation_id"].astype(str).isin(golden_ids)].copy()

        # Limit index size for speed & efficiency
        if len(corpus_df) > max_samples:
            corpus_df = corpus_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.corpus_df = corpus_df.reset_index(drop=True)
        texts = self.corpus_df["customer_message"].tolist()

        # 2. Vectorizer & Nearest Neighbors
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
        X_vecs = self.vectorizer.fit_transform(texts)

        self.nn_model = NearestNeighbors(n_neighbors=self.top_k, metric="cosine")
        self.nn_model.fit(X_vecs)
        self.is_built = True

        # Save index
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "nn_model": self.nn_model,
                "corpus_df": self.corpus_df
            }, f)
        print(f"Retrieval index built on {len(self.corpus_df)} conversations -> Saved to {INDEX_PATH}")

    def load(self):
        """Loads index from disk."""
        if INDEX_PATH.exists():
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.nn_model = data["nn_model"]
                self.corpus_df = data["corpus_df"]
            self.is_built = True
        else:
            self.build_index()

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieves top-k historical customer-agent pairs matching the query.
        Returns:
            list of {"conversation_id", "customer_message", "support_response", "similarity_score"}
        """
        if not self.is_built or self.vectorizer is None:
            self.load()

        k = top_k if top_k is not None else self.top_k
        query_vec = self.vectorizer.transform([query])
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            sim_score = float(1.0 - dist)  # Cosine similarity = 1 - cosine distance
            row = self.corpus_df.iloc[idx]
            results.append({
                "conversation_id": str(row["conversation_id"]),
                "customer_message": str(row["customer_message"]),
                "support_response": str(row["support_response"]),
                "similarity_score": round(sim_score, 4)
            })

        return results


if __name__ == "__main__":
    retriever = HistoricalRetriever(top_k=3)
    retriever.build_index()

    test_query = "My iPhone 6 is stuck on the Apple logo while updating iOS"
    results = retriever.retrieve(test_query)
    print(f"\nTop 3 Retrieved Historical Examples for: '{test_query}'")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Score: {res['similarity_score']} (Conv ID: {res['conversation_id']})")
        print(f"    Customer: {res['customer_message']}")
        print(f"    Support : {res['support_response']}")
