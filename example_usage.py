"""
Example Usage of Advanced RAG Chatbot
This script demonstrates various features and use cases
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_chatbot import RAGChatbot

# Load environment variables
load_dotenv()


def example_basic_qa():
    """Example 1: Basic Question-Answering"""
    print("\n" + "="*60)
    print("Example 1: Basic Question-Answering")
    print("="*60)

    # Initialize chatbot
    chatbot = RAGChatbot(
        api_key=os.getenv("GOOGLE_API_KEY"),
        enable_query_rewriting=True,
        enable_reranking=True
    )

    # Index sample documents
    print("\n[1] Indexing documents...")
    # Create a sample document directory if it doesn't exist
    docs_dir = Path("./documents")
    docs_dir.mkdir(exist_ok=True)

    # Create a sample document if none exist
    sample_doc = docs_dir / "sample.txt"
    if not sample_doc.exists():
        sample_doc.write_text("""
        Machine Learning Basics

        Machine learning is a subset of artificial intelligence that enables systems to learn
        and improve from experience without being explicitly programmed. It focuses on the
        development of computer programs that can access data and use it to learn for themselves.

        There are three main types of machine learning:
        1. Supervised Learning: The algorithm learns from labeled training data
        2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
        3. Reinforcement Learning: The algorithm learns through trial and error

        Deep learning is a subset of machine learning that uses neural networks with multiple
        layers. These deep neural networks can learn complex patterns and representations from
        large amounts of data.
        """)

    chatbot.index_documents(str(docs_dir), source_type="directory")

    # Ask a question
    print("\n[2] Asking question...")
    response = chatbot.answer_query(
        "What are the types of machine learning?",
        k=5,
        alpha=0.5
    )

    print(f"\nQuestion: What are the types of machine learning?")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nMetadata:")
    print(f"  - Sources used: {response['num_sources']}")
    print(f"  - Query rewritten: {response['query_rewritten']}")
    print(f"  - Reranked: {response['reranked']}")


def example_hybrid_search():
    """Example 2: Hybrid Search with Different Alpha Values"""
    print("\n" + "="*60)
    print("Example 2: Hybrid Search Comparison")
    print("="*60)

    chatbot = RAGChatbot()
    chatbot.index_documents("./documents", source_type="directory")

    query = "neural networks"

    # Test different alpha values
    for alpha in [0.0, 0.5, 1.0]:
        print(f"\n[Alpha = {alpha}] ", end="")
        if alpha == 0.0:
            print("(Pure Sparse/BM25)")
        elif alpha == 1.0:
            print("(Pure Dense/Semantic)")
        else:
            print("(Balanced Hybrid)")

        response = chatbot.answer_query(
            query,
            k=3,
            alpha=alpha,
            include_sources=False
        )

        print(f"Answer: {response['answer'][:200]}...")


def example_query_rewriting():
    """Example 3: Query Rewriting and HyDE"""
    print("\n" + "="*60)
    print("Example 3: Query Rewriting Techniques")
    print("="*60)

    chatbot = RAGChatbot(enable_query_rewriting=True)
    chatbot.index_documents("./documents", source_type="directory")

    # Standard query
    print("\n[1] Standard Query")
    response1 = chatbot.answer_query(
        "ML types",
        k=3,
        use_hyde=False
    )
    print(f"Answer: {response1['answer'][:200]}...")

    # HyDE query
    print("\n[2] With HyDE (Hypothetical Document Embeddings)")
    response2 = chatbot.answer_query(
        "ML types",
        k=3,
        use_hyde=True
    )
    print(f"Answer: {response2['answer'][:200]}...")


def example_clustering():
    """Example 4: Document Clustering and Analysis"""
    print("\n" + "="*60)
    print("Example 4: Document Clustering Analysis")
    print("="*60)

    chatbot = RAGChatbot(enable_clustering=True)
    chatbot.index_documents("./documents", source_type="directory")

    print("\n[1] Performing cluster analysis...")
    insights = chatbot.get_cluster_insights(n_clusters=3)

    if 'error' not in insights:
        print(f"\nFound {insights['n_clusters']} clusters:\n")

        for i, analysis in enumerate(insights['analyses'], 1):
            print(f"Cluster {i}: {analysis['theme']}")
            print(f"  Documents: {analysis['document_count']}")
            print(f"  Summary: {analysis['summary']}")
            print(f"  Insights:")
            for insight in analysis['insights']:
                print(f"    {insight}")
            print()


def example_conversation():
    """Example 5: Multi-turn Conversation"""
    print("\n" + "="*60)
    print("Example 5: Multi-turn Conversation")
    print("="*60)

    chatbot = RAGChatbot()
    chatbot.index_documents("./documents", source_type="directory")

    # Conversation turns
    queries = [
        "What is machine learning?",
        "What are its main types?",
        "Can you explain supervised learning in more detail?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Turn {i}] User: {query}")
        response = chatbot.chat(query, use_context=True, k=3)
        print(f"Assistant: {response['answer'][:300]}...")


def example_url_indexing():
    """Example 6: Index Content from URL"""
    print("\n" + "="*60)
    print("Example 6: Indexing from URL")
    print("="*60)

    chatbot = RAGChatbot()

    # Index from a URL (example - Wikipedia article)
    url = "https://en.wikipedia.org/wiki/Machine_learning"
    print(f"\n[1] Indexing content from: {url}")

    try:
        chatbot.index_documents(url, source_type="url")

        print("\n[2] Asking question about the URL content...")
        response = chatbot.answer_query(
            "What is machine learning according to this source?",
            k=5
        )

        print(f"\nAnswer: {response['answer']}")

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if there are network issues)")


def example_statistics():
    """Example 7: Get Chatbot Statistics"""
    print("\n" + "="*60)
    print("Example 7: Chatbot Statistics")
    print("="*60)

    chatbot = RAGChatbot()
    chatbot.index_documents("./documents", source_type="directory")

    # Make a few queries
    chatbot.answer_query("What is machine learning?")
    chatbot.answer_query("Explain deep learning")

    # Get statistics
    stats = chatbot.get_stats()

    print("\n[Statistics]")
    print(f"Model: {stats['model']}")
    print(f"Conversation turns: {stats['conversation_turns']}")
    print(f"Total documents: {stats['retriever_stats']['total_documents']}")
    print(f"\nEnabled features:")
    for feature, enabled in stats['features'].items():
        print(f"  - {feature}: {'✓' if enabled else '✗'}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Advanced RAG Chatbot - Example Usage")
    print("="*60)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nError: GOOGLE_API_KEY not found in environment variables")
        print("Please set it in .env file or environment")
        return

    # Run examples
    try:
        example_basic_qa()
        # example_hybrid_search()
        # example_query_rewriting()
        # example_clustering()
        # example_conversation()
        # example_url_indexing()
        # example_statistics()

        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
