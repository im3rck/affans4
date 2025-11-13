"""
Document Processing Module
Handles loading and processing of various document types (PDF, TXT, DOCX, web pages)
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents from various sources"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, file_path: str) -> str:
        """Load content from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

    def load_txt(self, file_path: str) -> str:
        """Load content from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return ""

    def load_docx(self, file_path: str) -> str:
        """Load content from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return ""

    def load_url(self, url: str) -> str:
        """Load content from web page"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return ""

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings
                for punct in ['. ', '.\n', '! ', '?\n', '? ']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'metadata': metadata or {},
                    'start_idx': start,
                    'end_idx': end
                }
                chunks.append(chunk_data)

            start = end - self.chunk_overlap

        return chunks

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file and return chunks"""
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        # Determine file type and load content
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            content = self.load_pdf(str(file_path))
        elif suffix == '.txt':
            content = self.load_txt(str(file_path))
        elif suffix == '.docx':
            content = self.load_docx(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []

        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': suffix
        }

        return self.chunk_text(content, metadata)

    def process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process a URL and return chunks"""
        content = self.load_url(url)
        metadata = {
            'source': url,
            'source_type': 'url'
        }
        return self.chunk_text(content, metadata)

    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Process all supported files in a directory"""
        all_chunks = []
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        supported_extensions = ['.pdf', '.txt', '.docx']

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing: {file_path}")
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)

        logger.info(f"Processed {len(all_chunks)} chunks from {directory}")
        return all_chunks
