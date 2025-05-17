"""
Context provider service.

This module provides context for user queries from different sources:
- Document-based context from Elasticsearch
- Summary-based context from abstractive summaries
- SQL-based context from database tables
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple

from elastic.retriever import ElasticRetriever
from controllers.sql_db import query_database
from controllers.doc_summary import get_summary_service
from utils.extractText import clean_filename

# Configure logging
logger = logging.getLogger(__name__)

class ContextProviderService:
    """Service for providing context from different sources."""
    
    def __init__(self):
        """Initialize the context provider service."""
        logger.info("Initializing context provider service")
        self.summary_service = get_summary_service()
        
    def get_document_context(self, user_session: str, user_query: str) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Get document context from Elasticsearch.
        
        Args:
            user_session: User's session identifier
            user_query: User's query
            
        Returns:
            Tuple containing formatted context, file name, and page number
        """
        try:
            # Retrieve documents using Elastic search
            retriever = ElasticRetriever(user_session)
            docs = retriever.search(user_query)
            logger.info(f"Retrieved {len(docs)} documents for query: {user_query}")

            if not docs:
                return "", None, None
            
            # Extract page number and file name of the first document
            page_no = 0
            file_name = ''
            try:
                actual_metadata = docs[0].metadata.get('_source', {}).get('metadata', {})
                # Get filename directly from metadata or from source path
                file_name = actual_metadata.get('filename', '')
                if not file_name:
                    source = actual_metadata.get('source', '')
                    file_name = clean_filename(source)
                    
                # Get page number, ensuring it's an integer
                page_no = actual_metadata.get('page', 0)
                if isinstance(page_no, str):
                    try:
                        page_no = int(page_no)
                    except ValueError:
                        page_no = 0
                page_no = page_no + 1
                
                logger.info(f'File name: {file_name}')
                logger.info(f'Page number: {page_no}')
            except Exception as e: 
                logger.error(f'Error extracting metadata: {e}')
                # Don't return immediately on metadata extraction error
            
            # Process document information
            try:
                docs = self._extract_doc_info(docs)
                logger.info(f'Extracted {len(docs)} documents')
                if not docs:  # If extraction failed completely
                    return "", file_name, page_no
            except Exception as e:
                logger.error(f'Error restructuring documents: {e}')
                return "", file_name, page_no

            # Format context for LLM
            formatted_context = ""
            for i, chunk in enumerate(docs, start=1):
                try:
                    formatted_context += f"[{i}] \"{chunk['text']}\"  \n(Source: {chunk['source']}, Page {chunk['page']})\n\n"
                except Exception as e:
                    logger.error(f'Error formatting context: {e}')
                    continue
            
            return formatted_context, file_name, page_no
        except Exception as e:
            logger.error(f'Error retrieving document context: {e}')
            return "", None, None
            
    def get_data_context(self, user_session: str, user_query: str) -> str:
        """
        Get data context from SQL database.
        
        Args:
            user_session: User's session identifier
            user_query: User's query
            
        Returns:
            SQL query results formatted as context
        """
        try:
            sql_doc, error = query_database(user_session, user_query)
            if error:
                logger.error(f"SQL query error: {error}")
                return ""
                
            if not sql_doc:
                logger.info(f"No results found for SQL query")
                return ""
                
            logger.info(f'SQL query results added to context')
            return sql_doc
        except Exception as e:
            logger.error(f'Error getting data context: {e}')
            return ""
            
    def get_summary_context(self, user_session: str, user_query: str, 
                         language: Optional[str] = None, 
                         folder_names: Optional[List[str]] = None) -> str:
        """
        Get summary context from abstractive summaries.
        
        Args:
            user_session: User's session identifier
            user_query: User's query
            language: Language for the summary
            folder_names: List of folder names to include in the summary
            
        Returns:
            Abstractive summary
        """
        try:
            summary = self.summary_service.summarize_document(
                user_query, user_session, language, folder_names
            )
            return summary
        except Exception as e:
            logger.error(f'Error getting summary context: {e}')
            return f"I'm sorry, but I couldn't generate a summary at this time. Error: {str(e)}"
            
    def get_previous_question(self, user_session: str) -> str:
        """
        Get the previous question asked by the user.
        
        Args:
            user_session: User's session identifier
            
        Returns:
            Previous question or empty string
        """
        prev_question_filename = f'users/{user_session}/prev_question.txt'
        
        if os.path.exists(prev_question_filename):
            with open(prev_question_filename, 'r', encoding='utf-8') as file:
                prev_qn = str(file.read())
        else:
            prev_qn = ""
        
        return prev_qn
        
    def check_resources_exist(self, user_session: str) -> Dict[str, bool]:
        """
        Check which resources exist for the given user session.
        
        Args:
            user_session: User's session identifier
            
        Returns:
            Dictionary with resource availability flags
        """
        # Check for document files in the files directory
        files_dir = os.path.join('users', user_session, 'files')
        legacy_content = os.path.join('users', user_session, 'content.txt')
        has_documents = os.path.exists(files_dir) or os.path.exists(legacy_content)
        
        # Check for summaries
        summary_exists = False
        if os.path.exists(files_dir):
            import glob
            for file_dir in glob.glob(os.path.join(files_dir, '*')):
                if os.path.exists(os.path.join(file_dir, 'imp_sents.txt')):
                    summary_exists = True
                    break
        elif os.path.exists(os.path.join('users', user_session, 'imp_sents.txt')):
            summary_exists = True
            
        # Check for database tables by looking for sheet_metadata.json
        sheet_metadata_path = os.path.join('users', user_session, "files", "sheet_metadata.json")
        has_data_tables = os.path.exists(sheet_metadata_path)
        
        return {
            "has_documents": has_documents,
            "has_data_tables": has_data_tables,
            "has_summaries": summary_exists
        }
    
    def _extract_doc_info(self, docs):
        """
        Extract and process information from search results.
        
        Args:
            docs: List of document objects from search results
            
        Returns:
            Processed list of document information
        """
        extracted = []
        
        try:
            # Set threshold score as median of first few documents
            threshold_score = (
                docs[1].metadata.get("_score") or 
                docs[1].metadata.get("_source", {}).get("_score") or 
                3
            )
        except Exception as e:  
            logger.warning(f'Error determining threshold score: {e}')
            threshold_score = 3
        
        for doc in docs:
            try:
                # Extract metadata
                meta = (
                    doc.metadata.get('_source', {}).get('metadata', {}) 
                    if '_source' in doc.metadata else doc.metadata
                )
                source = meta.get('source', '')
                filename = meta.get('filename', '')  # Get filename directly from metadata
                page = meta.get('page', 0)

                if isinstance(page, str):
                    try:
                        page = int(page)
                    except Exception as e:
                        page = 0  # Default to 0 if conversion fails
                
                # Add 1 to get 1-based page number
                page = page + 1
                
                # If filename is missing but source exists, extract clean filename from source
                if not filename and source:
                    filename = clean_filename(source)
                
                logger.debug(f'File name: {filename}, page number: {page}')
                
                # Add document info to extracted list
                extracted.append({
                    "text": doc.page_content,
                    "source": filename,
                    "page": page
                })
            except Exception as e:
                logger.warning(f"Failed to extract from doc: {e}")
        
        return extracted

# Create a singleton instance
_context_provider_service = None

def get_context_provider_service() -> ContextProviderService:
    """
    Get the context provider service singleton instance.
    
    Returns:
        ContextProviderService: The context provider service instance
    """
    global _context_provider_service
    if _context_provider_service is None:
        _context_provider_service = ContextProviderService()
    return _context_provider_service