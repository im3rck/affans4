"""
RAG Chatbot Module
Main orchestration of the RAG system with Gemini
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import os
from document_processor import DocumentProcessor
from hybrid_retriever import HybridRetriever
from query_rewriter import QueryRewriter
from reranker import Reranker
from clustering_analyzer import ClusteringAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Advanced RAG Chatbot with multiple techniques:
    - Hybrid Search (Dense + Sparse)
    - Query Rewriting
    - Reranking
    - Clustering Analysis
    - Gemini for generation
    """

    def __init__(
        self,
        api_key: str = None,
        gemini_model: str = "gemini-1.5-pro-latest",
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_query_rewriting: bool = True,
        enable_reranking: bool = True,
        enable_clustering: bool = True
    ):
        """
        Initialize RAG Chatbot

        Args:
            api_key: Google API key for Gemini
            gemini_model: Gemini model name
            embedding_model: Sentence transformer model for embeddings
            enable_query_rewriting: Enable query rewriting
            enable_reranking: Enable reranking
            enable_clustering: Enable clustering analysis
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=self.api_key)
        self.gemini = genai.GenerativeModel(gemini_model)
        self.gemini_model = gemini_model

        # Initialize components
        logger.info("Initializing RAG Chatbot components...")

        self.document_processor = DocumentProcessor()
        self.retriever = HybridRetriever(embedding_model=embedding_model)

        self.enable_query_rewriting = enable_query_rewriting
        self.enable_reranking = enable_reranking
        self.enable_clustering = enable_clustering

        if enable_query_rewriting:
            self.query_rewriter = QueryRewriter(api_key=self.api_key)

        if enable_reranking:
            self.reranker = Reranker()

        if enable_clustering:
            self.clustering_analyzer = ClusteringAnalyzer(
                embedding_model=embedding_model,
                api_key=self.api_key
            )

        self.conversation_history = []
        logger.info("RAG Chatbot initialized successfully")

    def index_documents(self, source: str, source_type: str = "directory"):
        """
        Index documents from various sources

        Args:
            source: Path to directory, file, or URL
            source_type: Type of source ('directory', 'file', 'url')
        """
        logger.info(f"Indexing documents from {source_type}: {source}")

        if source_type == "directory":
            chunks = self.document_processor.process_directory(source)
        elif source_type == "file":
            chunks = self.document_processor.process_file(source)
        elif source_type == "url":
            chunks = self.document_processor.process_url(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        if chunks:
            self.retriever.add_documents(chunks)
            logger.info(f"Successfully indexed {len(chunks)} chunks")
        else:
            logger.warning("No documents were indexed")

    def retrieve_documents(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        use_hyde: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search

        Args:
            query: Search query
            k: Number of documents to retrieve
            alpha: Weight for dense vs sparse search
            use_hyde: Use hypothetical document embeddings

        Returns:
            List of retrieved documents
        """
        # Query rewriting
        if self.enable_query_rewriting:
            if use_hyde:
                # HyDE: Generate hypothetical document and use it for search
                hypothetical_doc = self.query_rewriter.generate_hypothetical_document(query)
                search_query = hypothetical_doc
                logger.info(f"Using HyDE with hypothetical document")
            else:
                # Generate query variations and search with each
                query_variations = self.query_rewriter.rewrite_query(query)
                logger.info(f"Generated {len(query_variations)} query variations")

                # Retrieve with each variation and merge results
                all_results = []
                for variant in query_variations[:2]:  # Use top 2 variants
                    results = self.retriever.hybrid_search(variant, k=k, alpha=alpha)
                    all_results.extend(results)

                # Deduplicate by text
                seen_texts = set()
                unique_results = []
                for doc in all_results:
                    if doc['text'] not in seen_texts:
                        seen_texts.add(doc['text'])
                        unique_results.append(doc)

                return unique_results[:k * 2]  # Return more for reranking
        else:
            search_query = query

        # Standard hybrid search
        results = self.retriever.hybrid_search(search_query, k=k * 2, alpha=alpha)
        return results

    def answer_query(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        use_hyde: bool = False,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG

        Args:
            query: User query
            k: Number of documents to use for context
            alpha: Hybrid search weight
            use_hyde: Use HyDE technique
            include_sources: Include source documents in response

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {query}")

        # Retrieve documents
        retrieved_docs = self.retrieve_documents(
            query,
            k=k * 2 if self.enable_reranking else k,
            alpha=alpha,
            use_hyde=use_hyde
        )

        if not retrieved_docs:
            return {
                'query': query,
                'answer': "I couldn't find any relevant documents to answer your question.",
                'sources': [],
                'num_sources': 0
            }

        # Rerank documents
        if self.enable_reranking:
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=k)
            final_docs = reranked_docs
            logger.info(f"Reranked to top {len(final_docs)} documents")
        else:
            final_docs = retrieved_docs[:k]

        # Build context from documents
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['text']}"
            for i, doc in enumerate(final_docs)
        ])

        # Generate answer with Gemini
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Provide a comprehensive and accurate answer based on the context
2. If the context doesn't contain enough information, say so
3. Cite specific documents when making claims (e.g., "According to Document 2...")
4. Be concise but thorough

Answer:"""

        try:
            response = self.gemini.generate_content(prompt)
            answer = response.text.strip()

            # Store in conversation history
            self.conversation_history.append(f"User: {query}")
            self.conversation_history.append(f"Assistant: {answer}")

            result = {
                'query': query,
                'answer': answer,
                'num_sources': len(final_docs),
                'retrieval_method': 'hybrid',
                'reranked': self.enable_reranking,
                'query_rewritten': self.enable_query_rewriting
            }

            if include_sources:
                result['sources'] = [
                    {
                        'text': doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'],
                        'score': doc.get('rerank_score', doc.get('hybrid_score', 0)),
                        'metadata': doc.get('metadata', {})
                    }
                    for doc in final_docs
                ]

            logger.info("Answer generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'query': query,
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'num_sources': 0
            }

    def get_cluster_insights(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze document collection using clustering

        Args:
            n_clusters: Number of clusters (auto-detect if None)

        Returns:
            Dictionary with cluster analyses and insights
        """
        if not self.enable_clustering:
            return {'error': 'Clustering is not enabled'}

        # Get all documents from retriever
        collection_stats = self.retriever.get_collection_stats()
        if collection_stats['total_documents'] == 0:
            return {'error': 'No documents indexed'}

        documents = self.retriever.documents

        logger.info(f"Analyzing {len(documents)} documents with clustering...")
        insights = self.clustering_analyzer.get_cluster_insights(
            documents,
            n_clusters=n_clusters
        )

        return insights

    def chat(
        self,
        query: str,
        use_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat interface with conversation history

        Args:
            query: User query
            use_context: Use conversation history for context
            **kwargs: Additional arguments for answer_query

        Returns:
            Response dictionary
        """
        # Rewrite query with conversation context if enabled
        if use_context and self.conversation_history and self.enable_query_rewriting:
            query = self.query_rewriter.rewrite_with_context(
                query,
                self.conversation_history[-6:]
            )

        return self.answer_query(query, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            'model': self.gemini_model,
            'conversation_turns': len(self.conversation_history) // 2,
            'features': {
                'query_rewriting': self.enable_query_rewriting,
                'reranking': self.enable_reranking,
                'clustering': self.enable_clustering
            },
            'retriever_stats': self.retriever.get_collection_stats()
        }
