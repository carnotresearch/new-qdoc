"""
Document upload and vector storage module.

This module provides functions for:
- Converting documents to text chunks
- Creating vector stores from text chunks
- Storing documents in Elasticsearch
"""

# Standard library imports
import logging
import os
import json

# Third-party imports
from elastic.document_manager import ElasticDocumentManager
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename

# Local imports
from utils.extractText import clean_filename
from config import Config

# Load environment variables
openai_api_key = Config.OPENAI_API_KEY


def process_file_content(pages, user_session, folder_name, filename):
    """
    Process file content to create chunks and store the full text.
    
    Args:
        pages (list): List of Document objects representing pages
        user_session (str): User's session identifier
        folder_name (str): Name to use for the folder
        filename (str): Original name of the file
        
    Returns:
        list: List of Document objects representing text chunks
    """
    # Clean the filename
    clean_file_name = clean_filename(filename)
    
    # Create file directory if it doesn't exist
    file_dir = os.path.join('users', user_session, 'files', folder_name)
    os.makedirs(file_dir, exist_ok=True)
    
    # Combine all text from the file
    full_text = ""
    total_pages = len(pages)
    
    for page in pages:
        full_text += page.page_content
        full_text = full_text.replace("\n", " ")
    
    # Save the full text to file
    content_path = os.path.join(file_dir, 'content.txt')
    with open(content_path, "w", encoding='utf-8') as file:
        file.write(full_text)
    
    # Save file metadata
    metadata = {
        'filename': clean_file_name,
        'page_count': total_pages,
        'extracted_at': os.path.getmtime(content_path),
        'file_size': os.path.getsize(content_path)
    }
    
    metadata_path = os.path.join(file_dir, 'metadata.json')
    with open(metadata_path, "w", encoding='utf-8') as file:
        json.dump(metadata, file)
    
    # Create chunks from the pages
    all_chunks = get_hierarchical_chunks(pages, clean_file_name)
    
    return all_chunks


def get_hierarchical_chunks(pages, filename=None):
    """
    Extract hierarchical chunks from document pages.
    
    Args:
        pages (list): List of Document objects representing pages
        filename (str, optional): Name of the file
        
    Returns:
        list: List of Document objects representing hierarchical chunks
    """
    # Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Initialize list for final chunks
    final_chunks = []
    
    # Process each page
    for page in pages:
        page_number = page.metadata.get('page', 0)
        source = page.metadata.get('source', filename or '')
        
        # Split by markdown headers
        md_header_splits = markdown_splitter.split_text(page.page_content)
        
        # Configure text splitter for further chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Process each header section
        for doc in md_header_splits:
            # Split into smaller chunks
            smaller_chunks = text_splitter.split_text(doc.page_content)
            
            # Create Document objects for each chunk
            for chunk in smaller_chunks:
                # Prepare metadata
                metadata = {
                    "source": source,
                    "page": page_number,
                    "header": " > ".join([
                        doc.metadata.get(f"Header {i}", "") 
                        for i in range(1, 4) 
                        if f"Header {i}" in doc.metadata
                    ])
                }
                
                # Add filename if provided
                if filename:
                    metadata["filename"] = filename
                
                # Create document with cleaned content and metadata
                final_chunks.append(Document(
                    page_content=chunk.replace("\n", " "),
                    metadata=metadata
                ))

    # Log the number of chunks created
    logging.info(f"Created {len(final_chunks)} chunks from {len(pages)} pages in file: {filename}")
    return final_chunks


def process_single_file(raw_text, user_session, filename):
    """
    Process a single file and add it to the vector store.
    
    Args:
        raw_text (list): List of Document objects for the file
        user_session (str): User's session identifier
        filename (str): Name of the file
        
    Returns:
        dict: File processing information
    """
    # Clean the filename
    clean_file_name = clean_filename(filename)
    
    # Use filename (without extension) as the folder name
    file_name = os.path.splitext(clean_file_name)[0]
    # Clean filename to make it safe for directory use
    safe_folder_name = secure_filename(file_name)
    
    # Process file content and create chunks
    text_chunks = process_file_content(raw_text, user_session, safe_folder_name, clean_file_name)
    
    # Create the main session directory if it doesn't exist
    user_session_dir = os.path.join('users', user_session)
    if not os.path.exists(user_session_dir):
        os.makedirs(user_session_dir)
    
    # Store documents
    doc_manager = ElasticDocumentManager(user_session)
    doc_manager.store_documents(text_chunks)
    
    # Return file processing information
    return {
        'filename': clean_file_name,
        'chunk_count': len(text_chunks),
        'page_count': len(raw_text),
        'success': True
    }


def store_vector(raw_text_list, user_session, new_container=True, file_info_list=None):
    """
    Process and store multiple files as vectors.
    
    Args:
        raw_text_list (list): List of lists of Document objects, one list per file
        user_session (str): User's session identifier
        new_container (bool): Whether to create a new container or update existing
        file_info_list (list, optional): List of file information dictionaries
        
    Returns:
        list: List of file processing results
    """
    # Process each file separately
    file_results = []
    
    for i, raw_text in enumerate(raw_text_list):
        try:
            # Skip empty files
            if not raw_text:
                continue
                
            # Get filename from file_info if available
            filename = file_info_list[i]['filename'] if file_info_list else f"file_{i}.txt"
            
            # Process the file
            file_result = process_single_file(raw_text, user_session, filename)
            file_results.append(file_result)
            
            # Log page tracking information
            logging.info(f"File {filename}: {file_result['page_count']} pages, "
                        f"{file_result['chunk_count']} chunks, "
                        f"max page number: {file_result['max_page_number']}")
            
        except Exception as e:
            logging.error(f"Error processing file {i}: {e}")
            file_results.append({
                'filename': filename if 'filename' in locals() else f"file_{i}.txt",
                'success': False,
                'error': str(e)
            })
    
    total_successful = sum(1 for r in file_results if r.get('success', False))
    total_pages = sum(r.get('page_count', 0) for r in file_results if r.get('success', False))
    total_chunks = sum(r.get('chunk_count', 0) for r in file_results if r.get('success', False))
    
    logging.info(f"Processed {len(file_results)} files with {total_successful} successes, "
                f"{total_pages} total pages, {total_chunks} total chunks")
    
    return file_results