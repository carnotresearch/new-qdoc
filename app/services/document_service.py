"""
Simplified document upload and processing service.

This module cleanly separates document processing (PDF, DOCX, TXT) from 
data processing (CSV, XLSX) while eliminating redundancy with extractText.py.
"""

import logging
import os
import threading
import time
from typing import Dict, Any, List, Tuple

from controllers.sql_db import create_database_with_tables, store_table_info, add_tables_to_existing_db
from controllers.doc_summary import create_abstractive_summary
from controllers.upload import store_vector
from controllers.delete_session import delete_session
from utils.extractText import get_text_from_files

# Configure logging
logger = logging.getLogger(__name__)

def classify_files(file_list) -> Tuple[List, List, List]:
    """
    Separate files into document files and data files
    
    Returns:
        Tuple of (document_files, data_files, unsupported_files)
    """
    document_extensions = {'.pdf', '.docx', '.txt', '.pptx', '.doc'}
    data_extensions = {'.csv', '.xlsx', '.xls'}
    
    document_files = []
    data_files = []
    unsupported_files = []
    
    for file in file_list:
        if not file.filename:
            unsupported_files.append(file)
            continue
            
        ext = os.path.splitext(file.filename)[1].lower()
        
        if ext in document_extensions:
            document_files.append(file)
        elif ext in data_extensions:
            data_files.append(file)
        else:
            unsupported_files.append(file)
    
    return document_files, data_files, unsupported_files

def process_document_files(doc_files: List, user_session: str, is_new_container: bool) -> Dict[str, Any]:
    """
    Process document files using the enhanced extractText.py system
    
    Returns:
        Dict with processing results
    """
    if not doc_files:
        return {"success": True, "files_processed": 0, "file_details": []}
    
    try:
        logger.info(f"Processing {len(doc_files)} document files")
        
        # Use the enhanced extractText system for batch processing
        all_docs, file_infos = get_text_from_files(doc_files)
        
        if not all_docs:
            return {
                "success": False,
                "message": "No text content could be extracted from document files",
                "files_processed": len(doc_files),
                "file_details": file_infos
            }
        
        # Create session directory if needed
        user_session_dir = os.path.join('users', user_session)
        os.makedirs(user_session_dir, exist_ok=True)
        
        # Store documents as vectors
        store_vector([all_docs], user_session, is_new_container, file_infos)
        
        # Create summaries asynchronously
        threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
        
        successful_files = sum(1 for info in file_infos if info.get('success', False))
        
        return {
            "success": True,
            "files_processed": len(doc_files),
            "files_successful": successful_files,
            "file_details": file_infos
        }
        
    except Exception as e:
        logger.error(f"Error processing document files: {e}")
        return {
            "success": False,
            "message": f"Error processing document files: {str(e)}",
            "files_processed": len(doc_files),
            "file_details": []
        }

def process_data_files(data_files: List, user_session: str, is_new_container: bool) -> Dict[str, Any]:
    """
    Process CSV/Excel files using existing SQL table creation
    
    Returns:
        Dict with processing results
    """
    if not data_files:
        return {"success": True, "files_processed": 0, "file_details": []}
    
    try:
        logger.info(f"Processing {len(data_files)} data files")
        
        # Use existing SQL table creation logic
        if is_new_container:
            success, message = create_database_with_tables(user_session, data_files)
        else:
            success, message = add_tables_to_existing_db(user_session, data_files)
        
        if not success:
            return {
                "success": False,
                "message": f"Failed to create database tables: {message}",
                "files_processed": len(data_files),
                "file_details": []
            }
        
        # Create summaries asynchronously
        for file in data_files:
            threading.Thread(target=store_table_info, args=(user_session, file.filename)).start()
        
        file_details = [{"filename": f.filename, "success": True} for f in data_files]
        
        return {
            "success": True,
            "files_processed": len(data_files),
            "files_successful": len(data_files),
            "file_details": file_details
        }
        
    except Exception as e:
        logger.error(f"Error processing data files: {e}")
        return {
            "success": False,
            "message": f"Error processing data files: {str(e)}",
            "files_processed": len(data_files),
            "file_details": []
        }

class DocumentService:
    """Enhanced document service with clean separation of concerns"""
    
    def __init__(self):
        logger.info("Initializing DocumentService")
    
    def process_files(self, files: Any, user_session: str, is_new_container: bool = True, 
                     is_trial: bool = False) -> Dict[str, Any]:
        """
        Process uploaded files with automatic type detection and routing
        
        Args:
            files: The files from the request
            user_session: User session identifier
            is_new_container: Whether to create a new container or add to existing
            is_trial: Whether this is for a trial user
            
        Returns:
            Dict containing processing results and status
        """
        start_time = time.time()
        logger.info(f"Processing files for session {user_session}")
        
        # Extract file list
        file_list = files.getlist("files")
        
        # Check if any files were uploaded
        if not file_list or file_list[0].filename == '':
            return {
                "status": "error",
                "message": "Please upload files in PDF, DOC, DOCX, PPTX, TXT, CSV, or XLSX format."
            }
        
        # Classify files by type
        document_files, data_files, unsupported_files = classify_files(file_list)
        
        # Check for unsupported files
        if unsupported_files:
            unsupported_names = [f.filename for f in unsupported_files]
            logger.warning(f"Unsupported files: {unsupported_names}")
        
        # Validate trial restrictions
        if is_trial and data_files:
            return {
                "status": "error",
                "message": "CSV files are not supported in free trial mode."
            }
        
        # Check if we have any processable files
        if not document_files and not data_files:
            return {
                "status": "error",
                "message": "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
            }
        
        # Delete previous session if needed for trial users
        if is_trial and is_new_container:
            try:
                logger.info(f"Deleting previous trial session for {user_session}")
                delete_session(user_session)
            except Exception as e:
                logger.warning(f"Error deleting previous session: {e}")
        
        # Process data files first (CSV/Excel)
        data_result = process_data_files(data_files, user_session, is_new_container)
        
        # Process document files (PDF, DOCX, TXT)
        doc_result = process_document_files(document_files, user_session, is_new_container)
        
        # Combine results
        total_processed = data_result["files_processed"] + doc_result["files_processed"]
        total_successful = data_result.get("files_successful", 0) + doc_result.get("files_successful", 0)
        
        all_file_details = data_result["file_details"] + doc_result["file_details"]
        
        # Add unsupported files to details
        for unsupported_file in unsupported_files:
            all_file_details.append({
                "filename": unsupported_file.filename,
                "success": False,
                "error": "Unsupported file type"
            })
        
        # Determine overall success
        overall_success = data_result["success"] and doc_result["success"]
        
        if overall_success and total_successful > 0:
            return {
                "status": "ok",
                "message": "Files successfully uploaded.",
                "files_processed": total_processed,
                "files_successful": total_successful,
                "file_details": all_file_details
            }
        else:
            error_messages = []
            if not data_result["success"]:
                error_messages.append(data_result.get("message", "Data file processing failed"))
            if not doc_result["success"]:
                error_messages.append(doc_result.get("message", "Document file processing failed"))
            
            return {
                "status": "error",
                "message": "; ".join(error_messages) if error_messages else "Processing failed",
                "files_processed": total_processed,
                "files_successful": total_successful,
                "file_details": all_file_details
            }
    
    def delete_container(self, user_session: str) -> bool:
        """Delete a user's container and associated data"""
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
    """Get the document service singleton instance"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service