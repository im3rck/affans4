"""
Hybrid Retriever Module
Combines dense (vector/semantic) and sparse (BM25/keyword) retrieval
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining dense embeddings and sparse BM25"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
        persist_directory: str = "./data/chroma_db"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")

        # BM25 components
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents to both vector store and BM25 index"""
        if not chunks:
            logger.warning("No chunks to add")
            return

        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(texts))]

        # Add to ChromaDB (dense retrieval)
        logger.info(f"Adding {len(texts)} documents to vector store...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        # Build BM25 index (sparse retrieval)
        logger.info("Building BM25 index...")
        self.documents = texts
        self.tokenized_docs = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        logger.info(f"Successfully indexed {len(texts)} documents")

    def dense_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform dense vector search"""
        query_embedding = self.embedding_model.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        retrieved_docs = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            retrieved_docs.append({
                'text': doc,
                'metadata': metadata,
                'score': 1 - distance,  # Convert distance to similarity
                'rank': i + 1,
                'method': 'dense'
            })

        return retrieved_docs

    def sparse_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform sparse BM25 search"""
        if self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        retrieved_docs = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include documents with positive scores
                retrieved_docs.append({
                    'text': self.documents[idx],
                    'metadata': {},
                    'score': float(scores[idx]),
                    'rank': rank + 1,
                    'method': 'sparse'
                })

        return retrieved_docs

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse results

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for dense search (0-1). sparse weight = 1 - alpha

        Returns:
            List of retrieved documents with combined scores
        """
        # Get results from both methods
        dense_results = self.dense_search(query, k=k * 2)
        sparse_results = self.sparse_search(query, k=k * 2)

        # Normalize scores
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            min_dense = min(r['score'] for r in dense_results)
            if max_dense > min_dense:
                for r in dense_results:
                    r['score'] = (r['score'] - min_dense) / (max_dense - min_dense)

        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            min_sparse = min(r['score'] for r in sparse_results)
            if max_sparse > min_sparse:
                for r in sparse_results:
                    r['score'] = (r['score'] - min_sparse) / (max_sparse - min_sparse)

        # Combine results using Reciprocal Rank Fusion (RRF)
        doc_scores = {}

        # Add dense results
        for doc in dense_results:
            text = doc['text']
            if text not in doc_scores:
                doc_scores[text] = {'doc': doc, 'dense_score': 0, 'sparse_score': 0}
            doc_scores[text]['dense_score'] = doc['score']

        # Add sparse results
        for doc in sparse_results:
            text = doc['text']
            if text not in doc_scores:
                doc_scores[text] = {'doc': doc, 'dense_score': 0, 'sparse_score': 0}
            doc_scores[text]['sparse_score'] = doc['score']

        # Calculate hybrid scores
        hybrid_results = []
        for text, scores in doc_scores.items():
            combined_score = (
                alpha * scores['dense_score'] +
                (1 - alpha) * scores['sparse_score']
            )

            doc = scores['doc'].copy()
            doc['dense_score'] = scores['dense_score']
            doc['sparse_score'] = scores['sparse_score']
            doc['hybrid_score'] = combined_score
            doc['method'] = 'hybrid'

            hybrid_results.append(doc)

        # Sort by combined score
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return hybrid_results[:k]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'bm25_initialized': self.bm25 is not None
        }
