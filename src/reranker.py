"""
Reranker Module
Uses cross-encoder models to rerank retrieved documents
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved documents using cross-encoder models"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return (None = return all)

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc['text']] for doc in documents]

        # Get reranking scores
        logger.info(f"Reranking {len(documents)} documents...")
        scores = self.model.predict(pairs)

        # Add reranking scores to documents
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            doc_copy['original_score'] = doc.get('score', doc.get('hybrid_score', 0))
            reranked_docs.append(doc_copy)

        # Sort by reranking score
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Update ranks
        for i, doc in enumerate(reranked_docs):
            doc['rerank_position'] = i + 1

        if top_k:
            reranked_docs = reranked_docs[:top_k]

        logger.info(f"Reranking complete. Returning top {len(reranked_docs)} documents")
        return reranked_docs

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        threshold: float = 0.0,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents and filter by score threshold

        Args:
            query: Search query
            documents: List of retrieved documents
            threshold: Minimum rerank score to include
            top_k: Maximum number of documents to return

        Returns:
            Filtered and reranked documents
        """
        reranked = self.rerank(query, documents, top_k=None)

        # Filter by threshold
        filtered = [doc for doc in reranked if doc['rerank_score'] >= threshold]

        if top_k:
            filtered = filtered[:top_k]

        logger.info(
            f"Filtered {len(reranked)} documents to {len(filtered)} "
            f"using threshold {threshold}"
        )

        return filtered
