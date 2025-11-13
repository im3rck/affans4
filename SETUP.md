# Setup Guide

## Quick Start

### 1. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Create a new API key or use an existing one
4. Copy the API key

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# Replace 'your_google_api_key_here' with your actual key
nano .env  # or use any text editor
```

Your `.env` file should look like:
```
GOOGLE_API_KEY=AIzaSy...your_actual_key_here
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Prepare Documents

Put your documents in the `documents/` folder:
- Supported formats: PDF, TXT, DOCX
- Or use URLs to fetch content from the web

Example:
```bash
cp /path/to/your/documents/*.pdf documents/
```

### 5. Run the Application

#### Option A: Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

#### Option B: Python Script

```bash
python example_usage.py
```

#### Option C: Programmatic Usage

```python
from src.rag_chatbot import RAGChatbot
import os

chatbot = RAGChatbot(api_key=os.getenv("GOOGLE_API_KEY"))
chatbot.index_documents("./documents", source_type="directory")
response = chatbot.answer_query("Your question here")
print(response['answer'])
```

## Detailed Configuration

### Model Selection

You can customize which models to use:

```python
chatbot = RAGChatbot(
    gemini_model="gemini-1.5-pro-latest",      # or "gemini-pro"
    embedding_model="all-MiniLM-L6-v2",        # or other sentence-transformers models
)
```

### Feature Toggles

Enable/disable features:

```python
chatbot = RAGChatbot(
    enable_query_rewriting=True,    # Use Gemini to rewrite queries
    enable_reranking=True,          # Use cross-encoder for reranking
    enable_clustering=True          # Enable clustering analysis
)
```

### Retrieval Parameters

Fine-tune retrieval:

```python
response = chatbot.answer_query(
    query="Your question",
    k=5,              # Number of documents to retrieve
    alpha=0.5,        # 0=sparse only, 1=dense only, 0.5=balanced
    use_hyde=False    # Use hypothetical document embeddings
)
```

## Troubleshooting

### Issue: API Key Error

```
Error: GOOGLE_API_KEY not found in environment variables
```

**Solution**:
- Make sure you created the `.env` file
- Check that the API key is correctly set
- Verify the API key is valid at Google AI Studio

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: ChromaDB Error

```
sqlite3.OperationalError: unable to open database file
```

**Solution**:
```bash
# Create data directory
mkdir -p data

# Set permissions
chmod 755 data
```

### Issue: Out of Memory

**Solution**:
- Reduce batch size in document processing
- Process documents in smaller chunks
- Reduce the number of documents indexed at once

### Issue: Slow Performance

**Solution**:
- Disable reranking for faster responses (less accurate)
- Reduce `k` parameter (fewer documents retrieved)
- Use a smaller embedding model
- Enable caching

## Testing

### Basic Test

```bash
python -c "from src.rag_chatbot import RAGChatbot; print('Import successful!')"
```

### Full Test

```bash
python example_usage.py
```

### Web Interface Test

```bash
streamlit run app.py
```

Then:
1. Index some documents
2. Ask a test question
3. Check the response

## Advanced Setup

### Using Custom Embedding Models

```python
from src.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
)
```

### Using Custom Reranking Models

```python
from src.reranker import Reranker

reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"  # Larger, more accurate
)
```

### Persistent Storage

By default, ChromaDB stores data in `./data/chroma_db`. To change:

```python
retriever = HybridRetriever(
    persist_directory="/path/to/your/db"
)
```

## Production Deployment

### Docker (Future)

```bash
# Build
docker build -t rag-chatbot .

# Run
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key rag-chatbot
```

### Cloud Deployment

For cloud deployment (AWS, GCP, Azure):
1. Set up environment variables
2. Configure persistent storage
3. Set up load balancing
4. Enable caching
5. Monitor API usage

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review [example_usage.py](example_usage.py) for code examples
- Open an issue on GitHub for bugs
- Check Google AI Studio documentation for API issues

## Next Steps

1. Index your documents
2. Try different alpha values for retrieval
3. Experiment with query rewriting and HyDE
4. Explore cluster analysis
5. Fine-tune parameters for your use case

Happy chatting!
