"""
Document management API routes.

This module defines routes for document upload, processing, and container management.
"""

import logging
from flask import Blueprint, request, jsonify
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from app.services.auth_service import get_auth_service
from app.services.document_service import get_document_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
documents_bp = Blueprint('documents', __name__)

# Get service instances
auth_service = get_auth_service()
document_service = get_document_service()

# ----------------------------------------
# Free Trial Routes
# ----------------------------------------

@documents_bp.route('/freeTrial', methods=['POST', 'OPTIONS'])
def free_trial(): 
    """
    Handle file uploads for users in free trial mode.
    
    This endpoint processes files for users without authentication:
    1. Validates the user's fingerprint
    2. Processes uploaded files
    3. Stores the extracted text as vector embeddings
    4. Creates summaries asynchronously
    
    Returns:
        JSON response with status, message, and file details
    """
    try:
        # Extract and validate fingerprint
        fingerprint = request.form.get('fingerprint')
        if not fingerprint:
            return jsonify({'message': 'Fingerprint is missing'}), 400
            
        # Process files
        result = document_service.process_files(
            request.files, fingerprint, is_new_container=True, is_trial=True
        )
        
        if result.get("status") == "error":
            return jsonify({'message': result.get("message")}), 400
            
        return jsonify(result), 200
    except Exception as e:
        logger.exception(f'Error in free trial: {e}')
        return jsonify({'message': f'Error processing files: {str(e)}'}), 500

# ----------------------------------------
# Authenticated User Routes
# ----------------------------------------

@documents_bp.route('/upload', methods=['POST', 'OPTIONS'])
def upload(): 
    """
    Handle file uploads for authenticated users to create a new container.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Processes uploaded files
    3. Creates a new document container
    
    Returns:
        JSON response with status, message, and file details
    """
    # Authenticate user
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Process files
    result = document_service.process_files(
        request.files, user_session, is_new_container=True, is_trial=False
    )
    
    if result.get("status") == "error":
        return jsonify({'message': result.get("message")}), 400
        
    return jsonify(result), 200

@documents_bp.route('/add-upload', methods=['POST', 'OPTIONS'])
def add_upload():
    """
    Handle additional file uploads for authenticated users.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Processes uploaded files
    3. Adds to an existing document container
    
    Returns:
        JSON response with status, message, and file details
    """
    # Authenticate user
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Process files
    result = document_service.process_files(
        request.files, user_session, is_new_container=False, is_trial=False
    )
    
    if result.get("status") == "error":
        return jsonify({'message': result.get("message")}), 400
        
    return jsonify(result), 200

@documents_bp.route('/delete-container', methods=['POST'])
def delete_container_route():
    """
    Delete a user's container and associated data.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Deletes the user's container from the database
    
    Returns:
        JSON response with status message
    """
    # Extract request parameters
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Delete the user's container from the database
    success = document_service.delete_container(user_session)
    if success:
        return jsonify({'message': 'Container deleted successfully'}), 200
    else:
        return jsonify({'message': 'Error deleting container'}), 500