"""
Optimized Streamlit Web Interface for RAG Chatbot
Includes API quota management and usage tracking
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_chatbot import RAGChatbot
from config import (
    ACTIVE_CONFIG,
    ACTIVE_PROFILE,
    APIUsageTracker,
    estimate_cost_per_query,
    print_active_config
)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Chatbot (Quota Optimized)",
    page_icon="🤖",
    layout="wide",
)

# Initialize usage tracker
if 'usage_tracker' not in st.session_state:
    st.session_state.usage_tracker = APIUsageTracker(max_rpm=15)

@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot with optimized config"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found!")
        st.stop()

    return RAGChatbot(
        api_key=api_key,
        gemini_model=ACTIVE_CONFIG.get("gemini_model", "gemini-pro"),
        enable_query_rewriting=ACTIVE_CONFIG.get("enable_query_rewriting", False),
        enable_reranking=ACTIVE_CONFIG.get("enable_reranking", True),
        enable_clustering=ACTIVE_CONFIG.get("enable_clustering", False),
    )

def main():
    st.title("🤖 RAG Chatbot (Quota Optimized)")
    st.caption(f"Profile: {ACTIVE_PROFILE} | API calls/query: ~{estimate_cost_per_query(ACTIVE_CONFIG)}")

    # Initialize
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing..."):
            st.session_state.chatbot = initialize_chatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'api_calls_made' not in st.session_state:
        st.session_state.api_calls_made = 0

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Usage Display
        st.subheader("📊 API Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Calls Made", st.session_state.api_calls_made)
        with col2:
            calls_in_last_min = len(st.session_state.usage_tracker.calls)
            st.metric("Last Min", calls_in_last_min, delta=f"{15-calls_in_last_min} left")

        if st.session_state.api_calls_made > 0:
            st.progress(min(calls_in_last_min / 15, 1.0))

        st.caption(f"Free tier limit: 15 calls/min")

        st.divider()

        # Document indexing
        st.subheader("📚 Documents")
        directory_path = st.text_input("Directory:", value="./documents")

        if st.button("Index Documents"):
            if os.path.exists(directory_path):
                with st.spinner("Indexing..."):
                    st.session_state.chatbot.index_documents(
                        directory_path,
                        source_type="directory"
                    )
                    st.success("✅ Indexed!")
            else:
                st.error("Directory not found")

        st.divider()

        # Query settings
        st.subheader("🔍 Query Settings")
        num_results = st.slider("Results:", 1, 10, 5)
        alpha = st.slider("Dense/Sparse:", 0.0, 1.0, 0.5, 0.1)

        # Show if expensive features are enabled
        if ACTIVE_CONFIG.get("enable_query_rewriting"):
            st.warning("⚠️ Query rewriting enabled (+1 API call)")

        use_hyde = st.checkbox(
            "HyDE (+1 API call)",
            value=False,
            disabled=not ACTIVE_CONFIG.get("enable_query_rewriting")
        )

        st.divider()

        # Profile info
        with st.expander("ℹ️ Active Profile"):
            st.code(f"""
Profile: {ACTIVE_PROFILE}
Model: {ACTIVE_CONFIG.get('gemini_model')}
Query Rewriting: {ACTIVE_CONFIG.get('enable_query_rewriting')}
Reranking: {ACTIVE_CONFIG.get('enable_reranking')}
Clustering: {ACTIVE_CONFIG.get('enable_clustering')}
            """)

    # Main chat area
    st.subheader("💬 Chat")

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Check rate limit
        if not st.session_state.usage_tracker.can_make_call():
            st.error("⏳ Rate limit reached! Please wait a minute.")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Track API usage
                estimated_calls = estimate_cost_per_query({
                    **ACTIVE_CONFIG,
                    "use_hyde": use_hyde
                })

                # Make query
                start_time = time.time()

                response = st.session_state.chatbot.answer_query(
                    query,
                    k=num_results,
                    alpha=alpha,
                    use_hyde=use_hyde,
                    include_sources=True
                )

                elapsed = time.time() - start_time

                # Record API calls
                for _ in range(estimated_calls):
                    st.session_state.usage_tracker.record_call()
                st.session_state.api_calls_made += estimated_calls

                # Display answer
                st.markdown(response['answer'])

                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {elapsed:.2f}s")
                with col2:
                    st.caption(f"📄 {response['num_sources']} sources")
                with col3:
                    st.caption(f"🔥 {estimated_calls} API calls")

                # Save message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer']
                })

    # Clear button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
