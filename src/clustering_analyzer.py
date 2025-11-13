"""
Clustering Analyzer Module
Combines traditional data analysis (clustering) with Gemini's generative capabilities
to find insights and patterns in document collections
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """
    Analyze document collections using clustering and LLM interpretation
    Combines traditional ML (clustering) with generative AI (Gemini)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: str = None,
        gemini_model: str = "gemini-1.5-flash-latest"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")

        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(gemini_model)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 10
    ) -> int:
        """Find optimal number of clusters using silhouette score"""
        if len(embeddings) < min_clusters:
            return min(len(embeddings), 2)

        best_score = -1
        best_k = min_clusters

        for k in range(min_clusters, min(max_clusters + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        logger.info(f"Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k

    def cluster_documents(
        self,
        documents: List[str],
        n_clusters: Optional[int] = None,
        method: str = "kmeans"
    ) -> Dict[str, Any]:
        """
        Cluster documents and return cluster assignments

        Args:
            documents: List of document texts
            n_clusters: Number of clusters (auto-detect if None)
            method: Clustering method ('kmeans' or 'dbscan')

        Returns:
            Dictionary with cluster info
        """
        embeddings = self.create_embeddings(documents)

        if method == "kmeans":
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(embeddings)

            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)

        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Organize documents by cluster
        clusters = {}
        for doc, label in zip(documents, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)

        logger.info(f"Clustered {len(documents)} documents into {len(clusters)} clusters")

        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'embeddings': embeddings,
            'n_clusters': n_clusters
        }

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization"""
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "umap":
            reducer = UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        reduced = reducer.fit_transform(embeddings)
        logger.info(f"Reduced dimensions to {n_components}D using {method}")
        return reduced

    def analyze_cluster_with_llm(
        self,
        cluster_documents: List[str],
        cluster_id: int
    ) -> Dict[str, str]:
        """
        Use Gemini to analyze and summarize a cluster

        Args:
            cluster_documents: Documents in the cluster
            cluster_id: Cluster identifier

        Returns:
            Dictionary with cluster analysis
        """
        # Sample documents if too many
        sample_docs = cluster_documents[:10] if len(cluster_documents) > 10 else cluster_documents

        docs_text = "\n\n---\n\n".join([f"Document {i+1}:\n{doc[:500]}"
                                        for i, doc in enumerate(sample_docs)])

        prompt = f"""Analyze the following cluster of documents and provide:
1. A descriptive theme/topic (2-3 words)
2. A brief summary of the main concepts (1-2 sentences)
3. Key insights or patterns (2-3 bullet points)

Cluster ID: {cluster_id}
Number of documents: {len(cluster_documents)}

Sample documents from this cluster:
{docs_text}

Provide your analysis in this format:
Theme: [theme]
Summary: [summary]
Insights:
- [insight 1]
- [insight 2]
- [insight 3]
"""

        try:
            response = self.gemini.generate_content(prompt)
            analysis_text = response.text.strip()

            # Parse the response
            lines = analysis_text.split('\n')
            theme = ""
            summary = ""
            insights = []

            for line in lines:
                if line.startswith("Theme:"):
                    theme = line.replace("Theme:", "").strip()
                elif line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.strip().startswith("-"):
                    insights.append(line.strip())

            return {
                'cluster_id': cluster_id,
                'theme': theme,
                'summary': summary,
                'insights': insights,
                'document_count': len(cluster_documents),
                'full_analysis': analysis_text
            }

        except Exception as e:
            logger.error(f"Error analyzing cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'theme': f"Cluster {cluster_id}",
                'summary': f"Contains {len(cluster_documents)} documents",
                'insights': [],
                'document_count': len(cluster_documents),
                'full_analysis': ""
            }

    def analyze_all_clusters(
        self,
        cluster_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Analyze all clusters using Gemini"""
        clusters = cluster_result['clusters']
        analyses = []

        for cluster_id, documents in clusters.items():
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue

            logger.info(f"Analyzing cluster {cluster_id} ({len(documents)} documents)...")
            analysis = self.analyze_cluster_with_llm(documents, cluster_id)
            analyses.append(analysis)

        return analyses

    def get_cluster_insights(
        self,
        documents: List[str],
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: cluster documents and generate insights

        Args:
            documents: List of document texts
            n_clusters: Number of clusters (auto-detect if None)

        Returns:
            Dictionary with clustering results and LLM-generated insights
        """
        # Cluster documents
        cluster_result = self.cluster_documents(documents, n_clusters=n_clusters)

        # Analyze clusters with LLM
        analyses = self.analyze_all_clusters(cluster_result)

        # Reduce dimensions for visualization
        reduced_2d = self.reduce_dimensions(
            cluster_result['embeddings'],
            method="umap",
            n_components=2
        )

        return {
            'cluster_result': cluster_result,
            'analyses': analyses,
            'reduced_embeddings': reduced_2d,
            'n_clusters': cluster_result['n_clusters']
        }
