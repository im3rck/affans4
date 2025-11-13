"""
Query Rewriter Module
Uses Gemini to rewrite and expand queries for better retrieval
"""

import google.generativeai as genai
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrite queries using Gemini for improved retrieval"""

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash-latest"):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite a query into multiple variations for better retrieval

        Returns:
            List of query variations including the original
        """
        prompt = f"""Given the following user query, generate 3 alternative phrasings that would help retrieve relevant documents.
The variations should:
1. Expand abbreviations and add context
2. Rephrase using synonyms
3. Break down complex queries into simpler components

Original Query: {query}

Generate ONLY the 3 alternative queries, one per line, without numbering or explanations."""

        try:
            response = self.model.generate_content(prompt)
            variations = response.text.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]

            # Add original query
            all_queries = [query] + variations[:3]

            logger.info(f"Generated {len(all_queries)} query variations")
            return all_queries

        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return [query]

    def rewrite_with_context(self, query: str, conversation_history: List[str]) -> str:
        """
        Rewrite query with conversation context for follow-up questions

        Args:
            query: Current user query
            conversation_history: List of previous queries/responses

        Returns:
            Standalone rewritten query
        """
        if not conversation_history:
            return query

        history_text = "\n".join(conversation_history[-6:])  # Last 3 turns

        prompt = f"""Given the conversation history and the current query, rewrite the query to be standalone and fully self-contained.

Conversation History:
{history_text}

Current Query: {query}

Rewritten Query (standalone, incorporating necessary context):"""

        try:
            response = self.model.generate_content(prompt)
            rewritten = response.text.strip()
            logger.info(f"Rewrote query with context: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Error rewriting query with context: {e}")
            return query

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical ideal document that would answer the query (HyDE technique)

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        prompt = f"""Given this question, write a short hypothetical passage (2-3 sentences) that would perfectly answer it.
Do not write "Here is" or similar - just write the passage directly.

Question: {query}

Hypothetical Answer:"""

        try:
            response = self.model.generate_content(prompt)
            hypothetical_doc = response.text.strip()
            logger.info(f"Generated hypothetical document for query: {query}")
            return hypothetical_doc

        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return query
