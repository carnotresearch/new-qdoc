"""
Document upload and processing service module.

This module handles file upload, vector storage, processing, and management.
"""

import logging
import os
import threading
import time
from typing import Dict, Any, List, Tuple, Optional

from werkzeug.utils import secure_filename

from controllers.sql_db import create_database_with_tables, store_table_info, add_tables_to_existing_db
from controllers.doc_summary import create_abstractive_summary
from controllers.upload import store_vector
from controllers.delete_session import delete_session
from utils.extractText import get_text_from_files

# Configure logging
logger = logging.getLogger(__name__)

class DocumentService:
    """Service for handling document upload and processing."""
    
    def __init__(self):
        """Initialize the document service."""
        logger.info("Initializing document service")
        self._valid_extensions = ['.pdf', '.docx', '.txt', '.xlsx', '.csv']
    
    def validate_file_type(self, filename: str) -> bool:
        """
        Check if a file has a valid extension.
        
        Args:
            filename (str): Name of the file to check
            
        Returns:
            bool: True if valid, False otherwise
        """
        return any(filename.lower().endswith(ext) for ext in self._valid_extensions)
    
    def process_files(self, files: Any, user_session: str, is_new_container: bool = True, 
                     is_trial: bool = False) -> Dict[str, Any]:
        """
        Process uploaded files (documents and/or CSV/Excel).
        
        Args:
            files: The files from the request
            user_session: User session identifier
            is_new_container: Whether to create a new container or add to existing
            is_trial: Whether this is for a trial user
            
        Returns:
            Dict containing processing results and status
        """
        start_time = time.time()
        logger.info(f"Processing files for session {user_session}, new container: {is_new_container}, trial: {is_trial}")
        
        file_list = files.getlist("files")
        
        # Check if any files were uploaded
        if not file_list or file_list[0].filename == '':
            message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
            logger.warning(f"No files uploaded for session {user_session}")
            return {"status": "error", "message": message}
        
        # Separate files into valid document types and CSV files
        valid_files = [f for f in file_list if f.filename.endswith(('.pdf', '.docx', '.txt'))]
        csv_files = [f for f in file_list if f.filename.endswith(('.xlsx', '.csv'))]
        
        # Process CSV files if any
        if csv_files and is_trial:
            logger.warning("CSV files are not supported in free trial mode")
            return {"status": "error", "message": "CSV files are not supported in free trial mode."}
        
        # Process CSV/Excel files for authenticated users
        if csv_files and not is_trial:
            logger.info(f"Processing {len(csv_files)} CSV/Excel files")
            csv_result = self._process_data_files(user_session, csv_files, is_new_container)
            if not csv_result["success"]:
                return {"status": "error", "message": csv_result["message"]}
        
        # Delete previous session if needed for trial users
        if is_trial and is_new_container:
            try:
                logger.info(f"Deleting previous trial session for {user_session}")
                delete_session(user_session)
            except Exception as e:
                logger.warning(f"Error deleting previous session: {e}")
        
        # Process valid document files
        file_results = self._process_document_files(valid_files, user_session, is_new_container, start_time)
        
        # Prepare response message
        if not valid_files and not csv_files:
            message = "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
            return {"status": "error", "message": message}
        elif file_results:
            successful_files = sum(1 for r in file_results if r.get('success', False))
            response_data = {
                "status": "ok",
                "message": "Files successfully uploaded.",
                "files_processed": len(file_results),
                "files_successful": successful_files,
                "file_details": file_results
            }
            return response_data
        else:
            return {
                "status": "ok",
                "message": "Files successfully uploaded."
            }
    
    def _process_data_files(self, user_session: str, data_files: List, is_new_container: bool) -> Dict[str, Any]:
        """
        Process CSV and Excel files.
        
        Args:
            user_session: User session identifier
            data_files: List of CSV/Excel files
            is_new_container: Whether to create a new container
            
        Returns:
            Dict with success status and message
        """
        try:
            # Add CSV files to the database
            if is_new_container:
                success, message = create_database_with_tables(user_session, data_files)
            else:
                success, message = add_tables_to_existing_db(user_session, data_files)
                
            if not success:
                logger.error(f"Error processing data files: {message}")
                return {"success": False, "message": message}
                
            logger.info('CSV/Excel files successfully uploaded!')
            
            # Create summaries asynchronously
            for file in data_files:
                threading.Thread(target=store_table_info, args=(user_session, file.filename,)).start()
            
            return {"success": True, "message": "Data files processed successfully"}
        except Exception as e:
            logger.error(f"Error processing data files: {e}")
            return {"success": False, "message": f"Error processing data files: {str(e)}"}
    
    def _process_document_files(self, doc_files: List, user_session: str, is_new_container: bool, start_time: float) -> List[Dict[str, Any]]:
        """
        Process document files (PDF, DOCX, TXT).
        
        Args:
            doc_files: List of document files
            user_session: User session identifier
            is_new_container: Whether to create a new container
            start_time: Start time for performance logging
            
        Returns:
            List of processing results for each file
        """
        if not doc_files:
            logger.info("No document files to process")
            return []
        
        try:
            # Process each file individually
            all_docs = []
            file_infos = []
            
            for file in doc_files:
                # Extract text from the file
                docs, file_info = get_text_from_files([file])
                if docs:
                    all_docs.append(docs)
                    file_infos.append(file_info[0])
            
            # Log the time taken to extract data from files
            logger.info('--- %s seconds to extract data from files ---' % (time.time() - start_time))
            
            file_results = []
            if all_docs:
                # Store the extracted text as vectors
                file_results = store_vector(all_docs, user_session, is_new_container, file_infos)
                logger.info('--- %s seconds to create vector indexes ---' % (time.time() - start_time))
            
                # Create document summaries asynchronously
                threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
            
            return file_results
        except Exception as e:
            logger.error(f'Error processing document files: {e}')
            return []
        
    def delete_container(self, user_session: str) -> bool:
        """
        Delete a user's container and associated data.
        
        Args:
            user_session: User session identifier
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            logger.info(f"Deleting container for session {user_session}")
            delete_session(user_session)
            return True
        except Exception as e:
            logger.error(f'Error deleting container: {e}')
            return False

# Create a singleton instance
_document_service = None

def get_document_service() -> DocumentService:
    """
    Get the document service singleton instance.
    
    Returns:
        DocumentService: The document service instance
    """
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service