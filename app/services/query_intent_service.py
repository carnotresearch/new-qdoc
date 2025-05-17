"""
Query intent classification service.

This module provides functionality to classify user queries into different types:
- General chat queries (no document context needed)
- Summary queries (request for document summaries)
- Document-specific queries (need document context)
- SQL/data queries (need database context)
"""

import logging
from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional, List

from langchain_openai import ChatOpenAI
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    GENERAL_CHAT = auto()  # General conversation, no document context needed
    SUMMARY = auto()       # Request for document summary
    DOCUMENT = auto()      # Document-specific query needing vector search
    DATA_QUERY = auto()    # Query about structured data (SQL tables)
    HYBRID = auto()        # Needs both document and data context

class QueryIntentService:
    """Service for classifying the intent of user queries."""
    
    def __init__(self):
        """Initialize the query intent service."""
        logger.info("Initializing query intent service")
        self.openai_api_key = Config.OPENAI_API_KEY
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
        
    def classify_intent(self, query: str, has_documents: bool = True, 
                      has_data_tables: bool = False) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a user query.
        
        Args:
            query: The user's query text
            has_documents: Whether the user has uploaded document files
            has_data_tables: Whether the user has uploaded CSV/Excel files
            
        Returns:
            Tuple containing the query intent and confidence score
        """
        # Short circuit for empty queries
        if not query or not query.strip():
            return QueryIntent.GENERAL_CHAT, 1.0
            
        # First check if it's a general chat query
        if self._is_general_chat(query):
            return QueryIntent.GENERAL_CHAT, 0.9
            
        # Then check if it's a summary query
        if self._is_summary_query(query):
            return QueryIntent.SUMMARY, 0.9
            
        # If they have data tables, check if it's a data query
        if has_data_tables and self._is_data_query(query):
            # If they also have documents, check if it needs both contexts
            if has_documents and self._needs_document_context(query):
                return QueryIntent.HYBRID, 0.8
            return QueryIntent.DATA_QUERY, 0.9
            
        # Default to document query if they have documents
        if has_documents:
            return QueryIntent.DOCUMENT, 0.8
            
        # Fall back to general chat if no documents or tables
        return QueryIntent.GENERAL_CHAT, 0.7
    
    def _is_general_chat(self, query: str) -> bool:
        """
        Determine if a query is general conversation rather than document-specific.
        
        Args:
            query: The user's query text
            
        Returns:
            True if query is general chat, False otherwise
        """
        # Simple pattern matching for common conversational queries
        common_greetings = [
            "hello", "hi", "hey", "hi there", "hello there", "greetings",
            "how are you", "how's it going", "what's up", "good morning",
            "good afternoon", "good evening", "how do you work", 
            "what can you do", "who are you", "tell me about yourself"
        ]
        
        normalized_query = query.lower().strip().rstrip('?')
        
        for greeting in common_greetings:
            if normalized_query == greeting or normalized_query.startswith(greeting + " "):
                return True
                
        # Check for other conversational patterns
        return False
    
    def _is_summary_query(self, query: str) -> bool:
        """
        Determine if a query is requesting a document summary.
        
        Args:
            query: The user's query text
            
        Returns:
            True if query is a summary request, False otherwise
        """
        prompt = f"""User is asking questions regarding a document, which can be in any format. If the user question is requesting a general summary of the entire document, respond with the single integer 1. If the question asks to summarize a specific section, extract information, or answer specific queries about the document, respond with 0. For general conversation or if any confusion, respond with 0. Only respond with the single integer 0 or 1 as the answer.

        Examples:

        Question: Who is Bill Gates?
        Expected Response: 0

        Question: Summarize this PDF in 5 sentences.
        Expected Response: 1

        Question: Summarize the conclusion of this document.
        Expected Response: 0

        Question: Summarize this.
        Expected Response: 1

        Question: Can you extract the key points from page 10?
        Expected Response: 0

        Question: Hi, how are you?
        Expected Response: 0

        User question:
        {query}"""
        
        try:
            response = self.llm.invoke(prompt)
            logger.info(f'Summary classification response: {response.content}')
            
            num = int(response.content.strip())
            return bool(num)
        except Exception as e:
            logger.error(f'Error classifying summary query intent: {e}')
            return False
    
    def _is_data_query(self, query: str) -> bool:
        """
        Determine if a query is related to structured data (SQL tables).
        
        Args:
            query: The user's query text
            
        Returns:
            True if query is for structured data, False otherwise
        """
        prompt = f"""The user has uploaded documents (like PDF, DOCX, TXT) and also spreadsheets/CSV files. Determine if the user's query is asking specifically about data that would be stored in tables, spreadsheets, or requires numerical analysis.

        Respond with:
        1 - If the query is primarily about tabular data, statistics, calculations, or analysis of numerical information
        0 - If the query is primarily about textual information, conceptual understanding, or general content of documents

        Examples:
        "What's the average sales in Q3?" → 1
        "Plot a graph of the revenue trend" → 1  
        "How many employees have over 5 years experience?" → 1
        "What does the document say about climate change?" → 0
        "Summarize the introduction" → 0
        "Find mentions of AI technology" → 0

        User query: {query}

        Only respond with a single digit: 0 or 1."""
        
        try:
            response = self.llm.invoke(prompt)
            logger.info(f'Data query classification response: {response.content}')
            
            result = int(response.content.strip())
            return bool(result)
        except Exception as e:
            logger.error(f'Error classifying data query intent: {e}')
            return False
            
    def _needs_document_context(self, query: str) -> bool:
        """
        For data queries, determine if document context is also needed.
        
        Args:
            query: The user's query text
            
        Returns:
            True if query also needs document context, False otherwise
        """
        prompt = f"""The user has uploaded both documents (PDF, DOCX, TXT) and tabular data (CSV, Excel). 
        The query appears to be primarily related to the tabular data, but determine if understanding 
        the documents would also be important for answering this query comprehensively.

        Respond with:
        1 - If the query might benefit from both tabular data AND document context
        0 - If the query can be fully answered with just the tabular data alone

        Examples:
        "What's the revenue compared to the projections mentioned in the report?" → 1
        "Find discrepancies between the financial data and what's written in the document" → 1
        "Calculate the average sales for each region" → 0
        "What's the total expense by department?" → 0

        User query: {query}

        Only respond with a single digit: 0 or 1."""
        
        try:
            response = self.llm.invoke(prompt)
            logger.info(f'Hybrid context classification response: {response.content}')
            
            result = int(response.content.strip())
            return bool(result)
        except Exception as e:
            logger.error(f'Error classifying hybrid context needs: {e}')
            return False

# Create a singleton instance
_query_intent_service = None

def get_query_intent_service() -> QueryIntentService:
    """
    Get the query intent service singleton instance.
    
    Returns:
        QueryIntentService: The query intent service instance
    """
    global _query_intent_service
    if _query_intent_service is None:
        _query_intent_service = QueryIntentService()
    return _query_intent_service