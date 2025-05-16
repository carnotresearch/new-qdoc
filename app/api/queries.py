"""
Document querying API routes.

This module defines routes for document querying and LLM interactions.
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from app.services.auth_service import get_auth_service
from app.services.query_service import get_query_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
queries_bp = Blueprint('queries', __name__)

# Get service instances
auth_service = get_auth_service()
query_service = get_query_service()

# Get limiter extension
def get_limiter():
    return current_app.extensions.get("limiter")

# ----------------------------------------
# Trial User Query Routes
# ----------------------------------------

@queries_bp.route('/trialAsk', methods=['POST'])
def trial_ask():
    """
    Handle document queries for users in free trial mode.
    
    This endpoint:
    1. Validates the user's fingerprint and trial limits
    2. Processes the user's question
    3. Generates a response using vector search and LLMs
    
    Returns:
        JSON response with answer
    """
    # Apply rate limiting
    limiter = get_limiter()
    if limiter:
        limiter.limit("20 per minute")(trial_ask)
    
    # Process query
    data = request.get_json()
    response, status_code = query_service.process_trial_query(data)
    return jsonify(response), status_code

# ----------------------------------------
# Authenticated User Query Routes
# ----------------------------------------

@queries_bp.route('/ask', methods=['POST'])
def ask():
    """
    Handle document queries for authenticated users.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Checks if the user has exceeded query limits
    3. Processes the user's question
    4. Generates a response using vector search and LLMs
    
    Returns:
        JSON response with answer
    """
    data = request.get_json()
    
    # Authenticate user
    try:
        token = data.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_name = user_email
        context = data.get('context', False)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Authentication failed!'}), 401
    
    # We only need session_id when context is True
    if context:
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({'message': 'Session ID is required for context queries!'}), 400
            
        session_name = user_email + str(session_id.lower())

    # Process query
    response, status_code = query_service.process_authenticated_query(data, user_email, session_name)
    return jsonify(response), status_code

# ----------------------------------------
# Demo Query Routes
# ----------------------------------------

@queries_bp.route('/demo', methods=['POST'])
def demo():
    """
    Handle demo queries for public transport information.
    
    This endpoint:
    1. Processes the user's question about public transport
    2. Generates a response using a pre-defined database
    
    Returns:
        JSON response with answer
    """
    # Apply rate limiting
    limiter = get_limiter()
    if limiter:
        limiter.limit("10 per minute")(demo)
    
    # Process query
    data = request.get_json()
    response, status_code = query_service.process_demo_query(data)
    return jsonify(response), status_code