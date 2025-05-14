import os
import shutil
import logging
from elastic.client import ElasticClient

def delete_directory(directory):
    """
    Delete a directory and its contents.
    
    Args:
        directory (str): Path to the directory to be deleted
    """
    # Check if the directory exists
    if os.path.exists(directory):
        # Remove the directory and its contents
        shutil.rmtree(directory)
        logging.info(f"Deleted directory: {directory}")
    else:
        logging.warning(f"Directory not found: {directory}")

def delete_elastic_index(index_name):
    """
    Delete an Elasticsearch index.
    
    Args:
        index_name (str): Name of the Elasticsearch index to be deleted
    """
    # Check if the index exists
    if ElasticClient().client.indices.exists(index=index_name):
        # Delete the index
        ElasticClient().client.indices.delete(index=index_name)
        logging.info(f"Deleted index: {index_name}")
    else:
        logging.warning(f"Index not found: {index_name}")

def delete_session(user_session):
    """
    Delete a specific session (directory) for a user.
    
    Args:
        user_session (str): User's session identifier
    """
    # Construct the path to the directory
    session_path = os.path.join('users', user_session)
    
    # Delete the directory
    delete_directory(session_path)

    # Delete the Elasticsearch index
    delete_elastic_index(user_session)