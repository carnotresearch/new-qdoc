"""
Document summarization module.

This module provides functions for:
- Extracting important sentences from documents
- Creating abstractive summaries using LLMs
- Managing summaries for individual files
"""

# Standard library imports
import json
import logging
import os
import glob
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
from langchain_openai import ChatOpenAI
from summarizer import Summarizer
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Load config keys
openai_api_key = Config.OPENAI_API_KEY

class DocumentSummaryService:
    """Service for document summarization and management."""
    
    def __init__(self):
        """Initialize the document summary service."""
        logger.info("Initializing document summary service")
        self.bert_model = Summarizer()
    
    def create_abstractive_summary(self, user_session: str) -> None:
        """
        Create abstractive summaries for all files in a session.
        
        Args:
            user_session (str): User's session identifier
        """
        logger.info(f'Creating abstractive summaries for user session: {user_session}')
        
        # Define the files directory path
        files_dir = os.path.join('users', user_session, 'files')
        
        # Check if files directory exists
        if not os.path.exists(files_dir):
            # Legacy mode - check for a single content.txt file
            content_path = os.path.join('users', user_session, 'content.txt')
            if os.path.exists(content_path):
                self._create_legacy_summary(user_session)
            logger.info(f"No files directory found for session {user_session}")
            return
        
        # Get all file directories
        file_dirs = glob.glob(os.path.join(files_dir, '*'))
        logger.info(f"Found {len(file_dirs)} file directories")
        
        # Process each file directory
        for file_dir in file_dirs:
            # Get folder_name from directory name
            folder_name = os.path.basename(file_dir)
            
            # Try to create summary for this file
            try:
                self._create_file_summary(user_session, folder_name)
                logger.info(f'Created summary for file in folder {folder_name}')
            except Exception as e:
                logger.error(f'Error creating summary for file in folder {folder_name}: {e}')
    
    def _create_legacy_summary(self, user_session: str) -> None:
        """
        Create summary for a session using the legacy single-file approach.
        
        Args:
            user_session (str): User's session identifier
        """
        filename = os.path.join('users', user_session, 'content.txt')
        
        try:
            # Read the text from file
            with open(filename, "r", encoding='utf8') as file:
                full_text = file.read()
            
            # Use BERT extractive summarizer model
            most_important_sents = self.bert_model(full_text, num_sentences=60)  # Extract most important sentences
            
            # Save the most important sentences to a file
            output_path = os.path.join('users', user_session, 'imp_sents.txt')
            with open(output_path, 'w', encoding='utf8') as file:
                file.write(''.join(most_important_sents))
            
            logger.info(f'Created legacy summary for user session: {user_session}')
        except Exception as e:
            logger.error(f'Error creating legacy summary: {e}')
    
    def _create_file_summary(self, user_session: str, folder_name: str) -> None:
        """
        Create summary for a specific file.
        
        Args:
            user_session (str): User's session identifier
            folder_name (str): Name of the folder containing the file
        """
        # Define file paths
        file_dir = os.path.join('users', user_session, 'files', folder_name)
        content_path = os.path.join(file_dir, 'content.txt')
        summary_path = os.path.join(file_dir, 'imp_sents.txt')
        
        # Check if content file exists
        if not os.path.exists(content_path):
            logger.warning(f'Content file not found for file {folder_name}')
            raise FileNotFoundError(f'Content file not found for file {folder_name}')
        
        # Read the text from file
        with open(content_path, "r", encoding='utf8') as file:
            full_text = file.read()
        
        # Use BERT extractive summarizer model
        most_important_sents = self.bert_model(full_text, num_sentences=60)  # Extract most important sentences
        
        # Save the most important sentences to a file
        with open(summary_path, 'w', encoding='utf8') as file:
            file.write(''.join(most_important_sents))
        
        # Update metadata to include summary status
        self._update_file_metadata(file_dir, summary_path)
    
    def _update_file_metadata(self, file_dir: str, summary_path: str) -> None:
        """
        Update file metadata with summary information.
        
        Args:
            file_dir (str): Directory containing the file
            summary_path (str): Path to the summary file
        """
        metadata_path = os.path.join(file_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf8') as file:
                metadata = json.load(file)
            
            metadata['has_summary'] = True
            metadata['summary_created_at'] = os.path.getmtime(summary_path)
            
            with open(metadata_path, 'w', encoding='utf8') as file:
                json.dump(metadata, file)
                
            logger.debug(f"Updated metadata with summary information")
    
    def get_file_summaries(self, user_session: str) -> List[Dict[str, Any]]:
        """
        Get a list of files with summaries in a session.
        
        Args:
            user_session (str): User's session identifier
            
        Returns:
            list: List of file information with summary status
        """
        # Define the files directory path
        files_dir = os.path.join('users', user_session, 'files')
        
        # Check if files directory exists
        if not os.path.exists(files_dir):
            # Legacy mode - check for a single content.txt file
            if os.path.exists(os.path.join('users', user_session, 'imp_sents.txt')):
                return [{'legacy': True, 'folder_name': 'legacy', 'filename': 'All Files'}]
            return []
        
        # Get all file directories
        file_dirs = glob.glob(os.path.join(files_dir, '*'))
        
        # Collect information about each file
        file_summaries = []
        
        for file_dir in file_dirs:
            folder_name = os.path.basename(file_dir)
            metadata_path = os.path.join(file_dir, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf8') as file:
                    metadata = json.load(file)
                
                summary_info = {
                    'folder_name': folder_name,
                    'filename': metadata.get('filename', folder_name),
                    'has_summary': os.path.exists(os.path.join(file_dir, 'imp_sents.txt'))
                }
                
                file_summaries.append(summary_info)
        
        return file_summaries
    
    def summarize_document(self, query: str, user_session: str, language: Optional[str] = None, 
                           folder_names: Optional[List[str]] = None) -> str:
        """
        Create an abstractive summary from important sentences based on a query.
        
        Args:
            query (str): User's query for the summary
            user_session (str): User's session identifier
            language (str, optional): Language for the summary
            folder_names (list, optional): List of folder names to include in the summary
            
        Returns:
            str: Abstractive summary of the document
            
        Raises:
            Exception: If no important sentences are found or summary creation fails
        """
        start_time = time.time()
        logger.info(f"Generating document summary for query: {query}")
        
        # Get important sentences from specified files or all files
        combined_important_sentences = self._get_combined_important_sentences(user_session, folder_names)
        
        if not combined_important_sentences:
            logger.warning("No important sentences found")
            raise Exception("No important sentences found. Please run the summarization first.")
        
        # Create abstractive summary using LLM
        try:
            prompt = self._create_summary_prompt(query, combined_important_sentences, language)
            logger.info(f"Approx token count for prompt: {len(prompt.split()) * 1.33}")
            
            # Create summary with LLM
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
            summary = llm.invoke(prompt)
            logger.info(f'Generated summary in {time.time() - start_time:.2f} seconds')
            
            return summary.content
        except Exception as e:
            logger.error(f'Error creating abstractive summary with LLM: {e}')
            raise Exception("Cannot create abstractive summary")
    
    def _get_combined_important_sentences(self, user_session: str, 
                                         folder_names: Optional[List[str]] = None) -> str:
        """
        Get combined important sentences from specified files or all files.
        
        Args:
            user_session (str): User's session identifier
            folder_names (list, optional): List of folder names to include
            
        Returns:
            str: Combined important sentences
        """
        combined_important_sentences = ""
        
        # If specific folder names are provided, use only those files' important sentences
        if folder_names and isinstance(folder_names, list) and len(folder_names) > 0:
            combined_important_sentences = self._get_sentences_from_specified_folders(user_session, folder_names)
        
        # If no specific folder name is provided or fallback is triggered, use all available files
        if not combined_important_sentences:
            combined_important_sentences = self._get_sentences_from_all_files(user_session)
        
        return combined_important_sentences
    
    def _get_sentences_from_specified_folders(self, user_session: str, folder_names: List[str]) -> str:
        """
        Get important sentences from specified folders.
        
        Args:
            user_session (str): User's session identifier
            folder_names (list): List of folder names to include
            
        Returns:
            str: Combined important sentences
        """
        combined_text = ""
        files_dir = os.path.join('users', user_session, 'files')
        found_any = False
        
        if os.path.exists(files_dir):
            # Process each provided folder name
            for folder_name in folder_names:
                imp_sents_path = os.path.join('users', user_session, 'files', folder_name, 'imp_sents.txt')
                metadata_path = os.path.join('users', user_session, 'files', folder_name, 'metadata.json')
                
                if os.path.exists(imp_sents_path):
                    with open(imp_sents_path, "r", encoding='utf8') as file:
                        file_sentences = file.read()
                        if not file_sentences:
                            logger.warning(f"No important sentences found in {folder_name}")
                            continue
                        
                        # Filename from metadata or default to folder name
                        filename = folder_name
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf8') as meta_file:
                                    metadata = json.load(meta_file)
                                    filename = metadata.get('filename', folder_name)
                            except Exception as e:
                                logger.error(f"Error reading metadata for {folder_name}: {e}")
                        
                        # Add file identifier and its content
                        combined_text += f"\n--- {filename} ---\n{file_sentences}\n"
                        logger.info(f"Added important sentences for {filename}")
                        found_any = True
        
        # If no files were found with the specified folder names, return empty string
        if not found_any:
            logger.warning(f"No important sentences found for specified folder names: {folder_names}")
            return ""
            
        return combined_text
    
    def _get_sentences_from_all_files(self, user_session: str) -> str:
        """
        Get important sentences from all files.
        
        Args:
            user_session (str): User's session identifier
            
        Returns:
            str: Combined important sentences
        """
        # Check for legacy mode - session-level summary
        legacy_path = os.path.join('users', user_session, 'imp_sents.txt')
        if os.path.exists(legacy_path):
            with open(legacy_path, "r", encoding='utf8') as file:
                return file.read()
        
        # New approach: Combine important sentences from all files
        combined_text = ""
        files_dir = os.path.join('users', user_session, 'files')
        if os.path.exists(files_dir):
            file_dirs = glob.glob(os.path.join(files_dir, '*'))
            
            # If no files found, return empty string
            if not file_dirs:
                logger.warning("No files found with summaries.")
                return ""
            
            # Process each file directory
            for file_dir in file_dirs:
                curr_folder_name = os.path.basename(file_dir)
                imp_sents_path = os.path.join(file_dir, 'imp_sents.txt')
                metadata_path = os.path.join(file_dir, 'metadata.json')
                
                if os.path.exists(imp_sents_path):
                    with open(imp_sents_path, "r", encoding='utf8') as file:
                        file_sentences = file.read()
                        if not file_sentences:
                            logger.warning(f"No important sentences found in {curr_folder_name}")
                            continue
                        
                        # Filename from metadata or default to folder name
                        filename = curr_folder_name
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf8') as meta_file:
                                    metadata = json.load(meta_file)
                                    filename = metadata.get('filename', curr_folder_name)
                            except Exception as e:
                                logger.error(f"Error reading metadata for {curr_folder_name}: {e}")
                        
                        # Add file identifier and its content
                        combined_text += f"\n--- {filename} ---\n{file_sentences}\n"
                        logger.info(f"Added important sentences for {filename}")
        
        return combined_text
    
    def _create_summary_prompt(self, query: str, sentences: str, language: Optional[str] = None) -> str:
        """
        Create a prompt for the LLM to generate a summary.
        
        Args:
            query (str): User's query for the summary
            sentences (str): Important sentences to summarize
            language (str, optional): Language for the summary
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = f'''You are given most important sentences extracted from one or more documents/files/papers.
            Answer the question based on the given sentences. Creatively infer and summarize the information in the sentences to provide a comprehensive answer.
            Do not make up information that is not in the sentences. Answer the question in a concise and clear manner.
            The sentences may come from multiple files, include relevant information from each source. Do not mention about extracted sentences, only include the file name in the answer.
            
            ```Sentences```
            {sentences}
            
            ```Query```
            {query}
            '''

        # Add language in prompt if language is not English
        if language and language != 'English':
            prompt = prompt + f"\nAnswer in user's preferred language - {language}."
            
        return prompt

# Create a singleton instance
_summary_service = None

def get_summary_service() -> DocumentSummaryService:
    """
    Get the document summary service singleton instance.
    
    Returns:
        DocumentSummaryService: The document summary service instance
    """
    global _summary_service
    if _summary_service is None:
        _summary_service = DocumentSummaryService()
    return _summary_service

# Keep these functions for backward compatibility
def create_abstractive_summary(user_session: str) -> None:
    """Legacy function to maintain backward compatibility."""
    return get_summary_service().create_abstractive_summary(user_session)

def summarize_document(query: str, user_session: str, language: Optional[str] = None, 
                      folder_names: Optional[List[str]] = None) -> str:
    """Legacy function to maintain backward compatibility."""
    return get_summary_service().summarize_document(query, user_session, language, folder_names)