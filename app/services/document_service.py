"""
Document upload and processing service.

This module handles document upload, vector storage, and file processing.
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

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for handling document upload and processing."""
    
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
        file_list = files.getlist("files")
        
        # Check if any files were uploaded
        if not file_list or file_list[0].filename == '':
            message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
            return {"status": "error", "message": message}
        
        # Separate files into valid document types and CSV files
        valid_files = [f for f in file_list if f.filename.endswith(('.pdf', '.docx', '.txt'))]
        csv_files = [f for f in file_list if f.filename.endswith(('.xlsx', '.csv'))]
        
        # Process CSV files if any
        if csv_files and is_trial:
            return {"status": "error", "message": "CSV files are not supported in free trial mode."}
        
        if csv_files and not is_trial:
            # Add CSV files to the database
            if is_new_container:
                success, message = create_database_with_tables(user_session, csv_files)
            else:
                success, message = add_tables_to_existing_db(user_session, csv_files)
                
            if not success:
                return {"status": "error", "message": message}
                
            logger.info('CSV files successfully uploaded!')
            
            # Create summaries asynchronously
            for file in csv_files:
                threading.Thread(target=store_table_info, args=(user_session, file.filename,)).start()
        
        # Delete previous session if needed for trial users
        if is_trial and is_new_container:
            try:
                delete_session(user_session)
            except Exception as e:
                logger.info(f'Error deleting previous session: {e}')
        
        # Process valid document files
        file_results = []
        if valid_files:
            try:
                # Process each file individually
                all_docs = []
                file_infos = []
                
                for file in valid_files:
                    # Extract text from the file
                    docs, file_info = get_text_from_files([file])
                    if docs:
                        all_docs.append(docs)
                        file_infos.append(file_info[0])
                
                # Log the time taken to extract data from files
                logger.info('--- %s seconds to extract data from files ---' % (time.time() - start_time))
                
                if all_docs:
                    # Store the extracted text as vectors
                    file_results = store_vector(all_docs, user_session, is_new_container, file_infos)
                    logger.info('--- %s seconds to create vector indexes ---' % (time.time() - start_time))
                
                # Create document summaries asynchronously
                threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
            except Exception as e:
                logger.error(f'Error processing document files: {e}')
                return {"status": "error", "message": f'Error processing files: {str(e)}'}
        
        # Prepare response message
        message = "Files successfully uploaded."
        if not valid_files and not csv_files:
            message = "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
        
        # Return success response
        response_data = {
            "status": "ok", 
            "message": message
        }
        
        # Include file details if available
        if file_results:
            successful_files = sum(1 for r in file_results if r.get('success', False))
            response_data["files_processed"] = len(file_results)
            response_data["files_successful"] = successful_files
            response_data["file_details"] = file_results
        
        return response_data
        
    def delete_container(self, user_session: str) -> bool:
        """
        Delete a user's container and associated data.
        
        Args:
            user_session: User session identifier
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            delete_session(user_session)
            return True
        except Exception as e:
            logger.exception(f'Error deleting container: {e}')
            return False

# Create a singleton instance
document_service = DocumentService()

def get_document_service() -> DocumentService:
    """
    Get the document service singleton instance.
    
    Returns:
        DocumentService: The document service instance
    """
    return document_service