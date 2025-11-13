# Documents Folder

This folder is where you should place your documents for the RAG chatbot to index and search.

## Supported Formats

- **PDF** (.pdf)
- **Text** (.txt)
- **Word Documents** (.docx)

## Usage

1. Place your documents in this folder
2. Use the Streamlit interface or Python API to index them:
   ```python
   chatbot.index_documents("./documents", source_type="directory")
   ```

## Sample Documents

Sample documents have been created but are gitignored by default:
- `sample_ml_basics.txt` - Machine learning fundamentals
- `sample_data_science.txt` - Data science overview

To create your own sample documents, run the application and it will generate them automatically,
or add your own PDF, TXT, or DOCX files here.

## Notes

- Documents in this folder are gitignored by default to protect your data
- The folder structure is preserved in git with a .gitkeep file
- You can modify .gitignore if you want to commit specific documents
