# Advanced RAG Chatbot with Gemini

A production-ready Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini, featuring multiple advanced techniques for superior document retrieval and answer generation.

## Features

### Core RAG Techniques

1. **Hybrid Search**
   - Combines dense (semantic/vector) and sparse (BM25/keyword) retrieval
   - Adjustable weighting between dense and sparse methods
   - Reciprocal Rank Fusion for optimal result merging
   - Significantly improves retrieval accuracy over single-method approaches

2. **Query Rewriting**
   - Automatic query reformulation using Gemini
   - Generates multiple query variations for comprehensive retrieval
   - HyDE (Hypothetical Document Embeddings) support
   - Context-aware rewriting for follow-up questions

3. **Reranking**
   - Cross-encoder model for precise document reranking
   - Filters and reorders initial retrieval results
   - Improves relevance of documents used for generation
   - Uses ms-marco-MiniLM-L-6-v2 model

4. **Clustering Analysis**
   - Unsupervised document clustering (KMeans/DBSCAN)
   - UMAP dimensionality reduction for visualization
   - AI-powered cluster interpretation using Gemini
   - Combines traditional ML with generative AI for insights

### Additional Features

- Multi-format document support (PDF, TXT, DOCX)
- Web scraping from URLs
- Smart text chunking with overlap
- Interactive Streamlit web interface
- Conversation history and context awareness
- Real-time statistics and monitoring
- Source citation and transparency

## Technology Stack

- **LLM**: Google Gemini (1.5 Pro)
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Sparse Retrieval**: BM25 (rank-bm25)
- **Reranking**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **Clustering**: scikit-learn, UMAP
- **Web Framework**: Streamlit
- **Document Processing**: PyPDF, python-docx, BeautifulSoup4

## Installation

### Prerequisites

- Python 3.8+
- Google API Key for Gemini

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd affans4
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Usage

### Web Interface (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Index documents using the sidebar (directory, file, or URL)
3. Configure search parameters
4. Start chatting with your documents
5. Explore cluster analysis for document insights

### Programmatic Usage

```python
from src.rag_chatbot import RAGChatbot
import os

# Initialize chatbot
chatbot = RAGChatbot(
    api_key=os.getenv("GOOGLE_API_KEY"),
    enable_query_rewriting=True,
    enable_reranking=True,
    enable_clustering=True
)

# Index documents
chatbot.index_documents("./documents", source_type="directory")

# Ask a question
response = chatbot.answer_query(
    "What is the main topic discussed in the documents?",
    k=5,
    alpha=0.5,
    use_hyde=False
)

print(response['answer'])
print(f"Used {response['num_sources']} sources")

# Get cluster insights
insights = chatbot.get_cluster_insights(n_clusters=5)
for analysis in insights['analyses']:
    print(f"Cluster: {analysis['theme']}")
    print(f"Summary: {analysis['summary']}")
```

## Advanced Configuration

### Hybrid Search Parameters

- **k**: Number of documents to retrieve (default: 5)
- **alpha**: Weight for dense vs sparse search (0-1)
  - 0 = Only sparse (BM25/keyword)
  - 1 = Only dense (semantic/vector)
  - 0.5 = Balanced hybrid
- **use_hyde**: Enable HyDE technique (default: False)

### Component Customization

```python
chatbot = RAGChatbot(
    gemini_model="gemini-1.5-pro-latest",  # Gemini model version
    embedding_model="all-MiniLM-L6-v2",     # Embedding model
    enable_query_rewriting=True,            # Query rewriting
    enable_reranking=True,                  # Reranking
    enable_clustering=True                  # Clustering analysis
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   Query Rewriter      │
           │   (Gemini)            │
           └───────────┬───────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │    Hybrid Retriever         │
         ├─────────────────────────────┤
         │  Dense      │    Sparse     │
         │  (Vector)   │    (BM25)     │
         └──────┬──────┴────────┬──────┘
                │               │
                └───────┬───────┘
                        │
                        ▼
              ┌─────────────────┐
              │   Reranker      │
              │  (Cross-Encoder)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Answer Gen     │
              │  (Gemini)       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Final Answer  │
              └─────────────────┘
```

## Project Structure

```
affans4/
├── src/
│   ├── document_processor.py      # Document loading and chunking
│   ├── hybrid_retriever.py        # Hybrid search implementation
│   ├── query_rewriter.py          # Query rewriting with Gemini
│   ├── reranker.py                # Document reranking
│   ├── clustering_analyzer.py     # Clustering and analysis
│   └── rag_chatbot.py             # Main RAG orchestration
├── data/                          # Vector database storage
├── documents/                     # Sample documents
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## API Reference

### RAGChatbot

Main class for RAG chatbot functionality.

#### Methods

- `index_documents(source, source_type)`: Index documents from various sources
- `answer_query(query, k, alpha, use_hyde)`: Answer a query using RAG
- `chat(query, use_context)`: Chat with conversation history
- `get_cluster_insights(n_clusters)`: Analyze document collection
- `get_stats()`: Get chatbot statistics

### DocumentProcessor

Handle document loading and processing.

#### Methods

- `process_file(file_path)`: Process a single file
- `process_directory(directory)`: Process all files in directory
- `process_url(url)`: Process content from URL
- `chunk_text(text, metadata)`: Split text into chunks

### HybridRetriever

Perform hybrid search combining dense and sparse retrieval.

#### Methods

- `add_documents(chunks)`: Add documents to index
- `hybrid_search(query, k, alpha)`: Perform hybrid search
- `dense_search(query, k)`: Dense vector search only
- `sparse_search(query, k)`: Sparse BM25 search only

## Performance Tips

1. **Document Chunking**: Adjust chunk size based on document type
   - Technical docs: 500-800 tokens
   - Narrative text: 800-1200 tokens

2. **Hybrid Search Alpha**: Tune based on query type
   - Factual queries: alpha=0.3 (more keyword)
   - Conceptual queries: alpha=0.7 (more semantic)

3. **Reranking**: Essential for accuracy but adds latency
   - Enable for production
   - Disable for rapid prototyping

4. **Query Rewriting**: Improves recall but increases API calls
   - Enable for complex queries
   - Consider caching for repeated queries

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in embedding generation
   - Process documents in smaller batches

2. **Slow Retrieval**
   - Reduce number of retrieved documents (k)
   - Disable reranking for faster responses

3. **Poor Retrieval Quality**
   - Adjust alpha parameter
   - Enable query rewriting
   - Use smaller chunk sizes

4. **API Rate Limits**
   - Implement exponential backoff
   - Cache embeddings and responses
   - Use batch processing

## Examples

### Example 1: Basic Q&A

```python
chatbot = RAGChatbot()
chatbot.index_documents("./documents", source_type="directory")

response = chatbot.answer_query("What is machine learning?")
print(response['answer'])
```

### Example 2: Advanced Retrieval

```python
response = chatbot.answer_query(
    query="Explain neural networks",
    k=10,              # Retrieve 10 documents
    alpha=0.7,         # Favor semantic search
    use_hyde=True      # Use hypothetical documents
)
```

### Example 3: Cluster Analysis

```python
insights = chatbot.get_cluster_insights(n_clusters=5)

for analysis in insights['analyses']:
    print(f"\nCluster: {analysis['theme']}")
    print(f"Documents: {analysis['document_count']}")
    print(f"Summary: {analysis['summary']}")
    print("Insights:")
    for insight in analysis['insights']:
        print(f"  {insight}")
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Google Gemini for LLM capabilities
- HuggingFace for transformer models
- ChromaDB for vector storage
- Streamlit for web interface

## Citation

If you use this project in your research, please cite:

```bibtex
@software{advanced_rag_chatbot,
  title={Advanced RAG Chatbot with Gemini},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/affans4}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review examples and troubleshooting guide

## Roadmap

- [ ] Support for more document formats (Markdown, HTML, etc.)
- [ ] Multi-modal support (images, tables)
- [ ] Advanced caching mechanisms
- [ ] Distributed vector storage
- [ ] Fine-tuning support for embeddings
- [ ] Multi-language support
- [ ] Custom reranking models
- [ ] API endpoint deployment
- [ ] Docker containerization

---

Built with advanced RAG techniques and powered by Google Gemini
