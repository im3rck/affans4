"""
Streamlit Web Interface for RAG Chatbot
"""

import streamlit as st
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_chatbot import RAGChatbot

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .stat-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot (cached)"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables!")
        st.stop()

    return RAGChatbot(
        api_key=api_key,
        gemini_model="gemini-1.5-pro",
        enable_query_rewriting=True,
        enable_reranking=True,
        enable_clustering=True
    )


def main():
    # Header
    st.markdown('<div class="main-header">🤖 Advanced RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by Gemini with Hybrid Search, Query Rewriting, Reranking & Clustering</div>',
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = initialize_chatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'documents_indexed' not in st.session_state:
        st.session_state.documents_indexed = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Document indexing section
        st.subheader("📚 Document Management")

        index_option = st.selectbox(
            "Index documents from:",
            ["Directory", "Single File", "URL"]
        )

        if index_option == "Directory":
            directory_path = st.text_input(
                "Directory path:",
                value="./documents",
                help="Path to directory containing documents (PDF, TXT, DOCX)"
            )
            if st.button("Index Directory"):
                if os.path.exists(directory_path):
                    with st.spinner(f"Indexing documents from {directory_path}..."):
                        st.session_state.chatbot.index_documents(
                            directory_path,
                            source_type="directory"
                        )
                        st.session_state.documents_indexed = True
                        st.success(f"Documents indexed successfully!")
                else:
                    st.error(f"Directory not found: {directory_path}")

        elif index_option == "Single File":
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=['pdf', 'txt', 'docx']
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.button("Index File"):
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        st.session_state.chatbot.index_documents(
                            temp_path,
                            source_type="file"
                        )
                        st.session_state.documents_indexed = True
                        st.success("File indexed successfully!")
                    os.remove(temp_path)

        elif index_option == "URL":
            url = st.text_input("Enter URL:")
            if st.button("Index URL"):
                if url:
                    with st.spinner(f"Indexing content from {url}..."):
                        st.session_state.chatbot.index_documents(
                            url,
                            source_type="url"
                        )
                        st.session_state.documents_indexed = True
                        st.success("URL content indexed successfully!")
                else:
                    st.warning("Please enter a URL")

        st.divider()

        # Query settings
        st.subheader("🔍 Query Settings")

        num_results = st.slider(
            "Number of results to retrieve:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of documents to use for answering"
        )

        alpha = st.slider(
            "Dense vs Sparse weight (alpha):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = only sparse (BM25), 1 = only dense (vector), 0.5 = balanced"
        )

        use_hyde = st.checkbox(
            "Use HyDE (Hypothetical Document Embeddings)",
            value=False,
            help="Generate hypothetical documents for better retrieval"
        )

        st.divider()

        # Statistics
        if st.session_state.documents_indexed:
            st.subheader("📊 Statistics")
            stats = st.session_state.chatbot.get_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Docs", stats['retriever_stats']['total_documents'])
            with col2:
                st.metric("Conversations", stats['conversation_turns'])

            with st.expander("Advanced Features"):
                features = stats['features']
                st.write("✓ Query Rewriting" if features['query_rewriting'] else "✗ Query Rewriting")
                st.write("✓ Reranking" if features['reranking'] else "✗ Reranking")
                st.write("✓ Clustering" if features['clustering'] else "✗ Clustering")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Cluster Analysis", "ℹ️ About"])

    with tab1:
        # Chat interface
        if not st.session_state.documents_indexed:
            st.info("👈 Please index some documents first using the sidebar to start chatting!")
        else:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    # Show sources if available
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander(f"📚 View {len(message['sources'])} sources"):
                            for i, source in enumerate(message['sources']):
                                st.markdown(f"**Source {i+1}** (Score: {source['score']:.3f})")
                                st.markdown(f'<div class="source-box">{source["text"]}</div>',
                                          unsafe_allow_html=True)
                                if source.get('metadata'):
                                    st.caption(f"Metadata: {source['metadata']}")

            # Chat input
            if query := st.chat_input("Ask a question about your documents..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})

                with st.chat_message("user"):
                    st.markdown(query)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.answer_query(
                            query,
                            k=num_results,
                            alpha=alpha,
                            use_hyde=use_hyde,
                            include_sources=True
                        )

                        st.markdown(response['answer'])

                        # Show metadata (use .get() to handle errors gracefully)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"📄 {response.get('num_sources', 0)} sources")
                        with col2:
                            st.caption(f"🔄 {'Reranked' if response.get('reranked', False) else 'Not reranked'}")
                        with col3:
                            st.caption(f"✍️ {'Query rewritten' if response.get('query_rewritten', False) else 'Original query'}")

                        # Save assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": response.get('sources', [])
                        })

            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.chatbot.conversation_history = []
                st.rerun()

    with tab2:
        # Clustering analysis
        st.subheader("🔍 Document Cluster Analysis")
        st.write("Analyze your document collection using clustering and AI-powered insights")

        if not st.session_state.documents_indexed:
            st.info("Please index documents first to use cluster analysis")
        else:
            col1, col2 = st.columns([3, 1])

            with col2:
                n_clusters = st.number_input(
                    "Number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=None,
                    help="Leave empty for auto-detection"
                )

            with col1:
                if st.button("🔬 Analyze Clusters", type="primary"):
                    with st.spinner("Clustering documents and generating insights..."):
                        insights = st.session_state.chatbot.get_cluster_insights(
                            n_clusters=n_clusters
                        )

                        if 'error' in insights:
                            st.error(insights['error'])
                        else:
                            st.success(f"Found {insights['n_clusters']} clusters!")

                            # Visualization
                            st.subheader("📊 Cluster Visualization")

                            reduced_emb = insights['reduced_embeddings']
                            labels = insights['cluster_result']['labels']

                            df_viz = pd.DataFrame({
                                'x': reduced_emb[:, 0],
                                'y': reduced_emb[:, 1],
                                'cluster': [f"Cluster {l}" for l in labels]
                            })

                            fig = px.scatter(
                                df_viz,
                                x='x',
                                y='y',
                                color='cluster',
                                title='Document Clusters (UMAP Projection)',
                                width=800,
                                height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Cluster insights
                            st.subheader("💡 Cluster Insights")

                            for analysis in insights['analyses']:
                                with st.expander(
                                    f"**{analysis['theme']}** - "
                                    f"{analysis['document_count']} documents",
                                    expanded=True
                                ):
                                    st.write(f"**Summary:** {analysis['summary']}")

                                    if analysis['insights']:
                                        st.write("**Key Insights:**")
                                        for insight in analysis['insights']:
                                            st.write(insight)

    with tab3:
        # About section
        st.subheader("About This Chatbot")

        st.markdown("""
        ### 🎯 Features

        This is an advanced RAG (Retrieval-Augmented Generation) chatbot built with multiple cutting-edge techniques:

        #### 1. **Hybrid Search** 🔍
        - Combines dense (semantic/vector) and sparse (BM25/keyword) retrieval
        - Adjustable weight between methods for optimal results
        - Reciprocal Rank Fusion for result merging

        #### 2. **Query Rewriting** ✍️
        - Uses Gemini to reformulate queries for better retrieval
        - Generates multiple query variations
        - Supports HyDE (Hypothetical Document Embeddings)
        - Context-aware rewriting for follow-up questions

        #### 3. **Reranking** 🎯
        - Cross-encoder model to rerank retrieved documents
        - Improves relevance of final results
        - ms-marco-MiniLM model for fast reranking

        #### 4. **Clustering Analysis** 🧬
        - KMeans/DBSCAN clustering of document collections
        - UMAP dimensionality reduction for visualization
        - AI-powered cluster interpretation using Gemini
        - Automatic optimal cluster detection

        #### 5. **Document Processing** 📄
        - Supports PDF, TXT, DOCX files
        - Web scraping from URLs
        - Smart chunking with overlap
        - Metadata preservation

        ### 🚀 Technology Stack

        - **LLM:** Google Gemini (1.5 Pro)
        - **Vector Store:** ChromaDB
        - **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
        - **Sparse Retrieval:** BM25 (rank-bm25)
        - **Reranking:** Cross-Encoder (ms-marco-MiniLM-L-6-v2)
        - **Clustering:** scikit-learn, UMAP
        - **Web Interface:** Streamlit

        ### 📖 How to Use

        1. **Index Documents:** Use the sidebar to index documents from a directory, file, or URL
        2. **Configure Settings:** Adjust retrieval parameters in the sidebar
        3. **Ask Questions:** Chat with your documents in the Chat tab
        4. **Analyze Clusters:** Discover patterns in the Cluster Analysis tab

        ### 🔧 Advanced Settings

        - **Alpha (0-1):** Controls dense vs sparse search weight
          - 0 = Only keyword search (BM25)
          - 1 = Only semantic search (vectors)
          - 0.5 = Balanced hybrid
        - **HyDE:** Generates hypothetical documents that would answer the query
        - **Number of Results:** How many documents to use for context

        ---
        Built with ❤️ using Google Gemini and modern RAG techniques
        """)


if __name__ == "__main__":
    main()
