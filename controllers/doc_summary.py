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

# Third-party imports
from langchain_openai import ChatOpenAI
from summarizer import Summarizer
from config import Config

# Load config keys
openai_api_key = Config.OPENAI_API_KEY


def create_abstractive_summary(usersession):
    """
    Create abstractive summaries for all files in a session.
    
    Args:
        usersession (str): User's session identifier
    """
    logging.info(f'Creating abstractive summaries for user session: {usersession}')
    
    # Define the files directory path
    files_dir = os.path.join('users', usersession, 'files')
    
    # Check if files directory exists
    if not os.path.exists(files_dir):
        # Legacy mode - check for a single content.txt file
        content_path = os.path.join('users', usersession, 'content.txt')
        if os.path.exists(content_path):
            create_legacy_summary(usersession)
        return
    
    # Get all file directories
    file_dirs = glob.glob(os.path.join(files_dir, '*'))
    
    # Process each file directory
    for file_dir in file_dirs:
        # Get folder_name from directory name
        folder_name = os.path.basename(file_dir)
        
        # Try to create summary for this file
        try:
            create_file_summary(usersession, folder_name)
            logging.info(f'Created summary for file in folder {folder_name}')
        except Exception as e:
            logging.error(f'Error creating summary for file in folder {folder_name}: {e}')


def create_legacy_summary(usersession):
    """
    Create summary for a session using the legacy single-file approach.
    
    Args:
        usersession (str): User's session identifier
    """
    filename = os.path.join('users', usersession, 'content.txt')
    
    try:
        # Read the text from file
        with open(filename, "r", encoding='utf8') as file:
            full_text = file.read()
        
        # Use BERT extractive summarizer model
        model = Summarizer()
        most_important_sents = model(full_text, num_sentences=60)  # Extract most important sentences
        
        # Save the most important sentences to a file
        with open(os.path.join(usersession, 'imp_sents.txt'), 'w', encoding='utf8') as file:
            file.write(''.join(most_important_sents))
        
        logging.info(f'Created legacy summary for user session: {usersession}')
    except Exception as e:
        logging.error(f'Error creating legacy summary: {e}')


def create_file_summary(usersession, folder_name):
    """
    Create summary for a specific file.
    
    Args:
        usersession (str): User's session identifier
        folder_name (str): Name of the folder containing the file
    """
    # Define file paths
    file_dir = os.path.join('users', usersession, 'files', folder_name)
    content_path = os.path.join(file_dir, 'content.txt')
    summary_path = os.path.join(file_dir, 'imp_sents.txt')
    
    # Check if content file exists
    if not os.path.exists(content_path):
        raise FileNotFoundError(f'Content file not found for file {folder_name}')
    
    # Read the text from file
    with open(content_path, "r", encoding='utf8') as file:
        full_text = file.read()
    
    # Use BERT extractive summarizer model
    model = Summarizer()
    most_important_sents = model(full_text, num_sentences=60)  # Extract most important sentences
    
    # Save the most important sentences to a file
    with open(summary_path, 'w', encoding='utf8') as file:
        file.write(''.join(most_important_sents))
    
    # Update metadata to include summary status
    metadata_path = os.path.join(file_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf8') as file:
            metadata = json.load(file)
        
        metadata['has_summary'] = True
        metadata['summary_created_at'] = os.path.getmtime(summary_path)
        
        with open(metadata_path, 'w', encoding='utf8') as file:
            json.dump(metadata, file)


def get_file_summaries(usersession):
    """
    Get a list of files with summaries in a session.
    
    Args:
        usersession (str): User's session identifier
        
    Returns:
        list: List of file information with summary status
    """
    # Define the files directory path
    files_dir = os.path.join('users', usersession, 'files')
    
    # Check if files directory exists
    if not os.path.exists(files_dir):
        # Legacy mode - check for a single content.txt file
        if os.path.exists(os.path.join('users', usersession, 'imp_sents.txt')):
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


def summarize_document(query, usersession, language, folder_names=None):
    """
    Create an abstractive summary from important sentences based on a query.
    
    Args:
        query (str): User's query for the summary
        usersession (str): User's session identifier
        folder_names (list, optional): List of folder names to include in the summary
        
    Returns:
        str: Abstractive summary of the document
        
    Raises:
        Exception: If no important sentences are found or summary creation fails
    """
    combined_important_sentences = ""
    
    # If specific folder names are provided, use only those files' important sentences
    if folder_names and isinstance(folder_names, list) and len(folder_names) > 0:
        files_dir = os.path.join('users', usersession, 'files')
        found_any = False
        
        if os.path.exists(files_dir):
            # Process each provided folder name
            for folder_name in folder_names:
                imp_sents_path = os.path.join('users', usersession, 'files', folder_name, 'imp_sents.txt')
                metadata_path = os.path.join('users', usersession, 'files', folder_name, 'metadata.json')
                
                if os.path.exists(imp_sents_path):
                    with open(imp_sents_path, "r", encoding='utf8') as file:
                        file_sentences = file.read()
                        filename = folder_name  # Default to folder_name if metadata not available
                        
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf8') as meta_file:
                                    metadata = json.load(meta_file)
                                    filename = metadata.get('filename', folder_name)
                            except Exception as e:
                                logging.error(f"Error reading metadata for {folder_name}: {e}")
                        
                        # Add file identifier and its content
                        combined_important_sentences += f"\n--- {filename} ---\n{file_sentences}\n"
                        found_any = True
        
        # If no files were found with the specified folder names, fall back to all files
        if not found_any:
            logging.info(f"No important sentences found for specified folder names: {folder_names}. Falling back to all files.")
            folder_names = None  # Trigger fallback to all files
    
    # If no specific folder name is provided or fallback is triggered, use all available files
    if not folder_names or not isinstance(folder_names, list) or len(folder_names) == 0:
        # Check for legacy mode - session-level summary
        legacy_path = os.path.join('users', usersession, 'imp_sents.txt')
        if os.path.exists(legacy_path):
            with open(legacy_path, "r", encoding='utf8') as file:
                combined_important_sentences = file.read()
        else:
            # New approach: Combine important sentences from all files
            files_dir = os.path.join('users', usersession, 'files')
            if os.path.exists(files_dir):
                file_dirs = glob.glob(os.path.join(files_dir, '*'))
                
                # If no files found, raise exception
                if not file_dirs:
                    raise Exception("No files found with summaries.")
                
                # Process each file directory
                for file_dir in file_dirs:
                    curr_folder_name = os.path.basename(file_dir)
                    imp_sents_path = os.path.join(file_dir, 'imp_sents.txt')
                    metadata_path = os.path.join(file_dir, 'metadata.json')
                    
                    if os.path.exists(imp_sents_path):
                        with open(imp_sents_path, "r", encoding='utf8') as file:
                            file_sentences = file.read()
                            filename = curr_folder_name  # Default to folder_name if metadata not available
                            
                            if os.path.exists(metadata_path):
                                try:
                                    with open(metadata_path, 'r', encoding='utf8') as meta_file:
                                        metadata = json.load(meta_file)
                                        filename = metadata.get('filename', curr_folder_name)
                                except Exception as e:
                                    logging.error(f"Error reading metadata for {curr_folder_name}: {e}")
                            
                            # Add file identifier and its content
                            combined_important_sentences += f"\n--- {filename} ---\n{file_sentences}\n"
    
    if not combined_important_sentences:
        raise Exception("No important sentences found. Please run the summarization first.")
    
    # Create abstractive summary using LLM
    try:
        prompt = f'''You are given most important sentences extracted from one or more documents/files/papers.
            Answer the question based on the given sentences. Creatively infer and summarize the information in the sentences to provide a comprehensive answer.
            Do not make up information that is not in the sentences. Answer the question in a concise and clear manner.
            The sentences may come from multiple files, include relevant information from each source. Do not mention about extracted sentences, only include the file name in the answer.
            
            ```Sentences```
            {combined_important_sentences}
            
            ```Query```
            {query}
            '''

        # Add language in prompt if language is not English
        if language and language != 'English':
            prompt = prompt +  f"\nAnswer in user's preferred language - {language}."
        # Guess no of tokens in the prompt
        num_tokens = len(prompt.split())
        logging.info(f'Number of tokens in the prompt: {num_tokens}')
        # if num_tokens > 4096:
        #     raise Exception("Prompt is too long. Please shorten your query or the sentences.")

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        summary = llm.invoke(prompt)
        logging.info(f'Summary result: {summary}')
        
        return summary.content
    except Exception as e:
        logging.error(f'Error creating abstractive summary with LLM: {e}')
        raise Exception("Cannot create abstractive summary")
