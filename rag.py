"""
rag.py
------
A lightweight RAG (Retrieval-Augmented Generation) retriever.

Uses TF-IDF + cosine similarity instead of a heavy embedding model so the
project runs anywhere with zero external services or GPU/API dependency.
Swap `TfidfRetriever` for a real vector DB (Chroma, Pinecone, pgvector...)
in production without changing the rest of the agent.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.doc_names = []
        self.doc_texts = []
        self._load_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _load_documents(self):
        paths = sorted(glob.glob(os.path.join(self.knowledge_dir, "*.txt")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.doc_names.append(os.path.basename(path))
            self.doc_texts.append(text)
        if not self.doc_texts:
            raise ValueError(f"No .txt documents found in {self.knowledge_dir}")

    def retrieve(self, query: str, top_k: int = 2):
        """Return the top_k most relevant documents for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked:
            if scores[idx] > 0:
                results.append(
                    {
                        "source": self.doc_names[idx],
                        "text": self.doc_texts[idx],
                        "score": float(scores[idx]),
                    }
                )
        return results
