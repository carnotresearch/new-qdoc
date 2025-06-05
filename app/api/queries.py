"""
Document querying API routes.

This module defines routes for document querying and LLM interactions.
"""

import logging
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import json
import time

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

@queries_bp.route('/ask-stream', methods=['GET'])
def ask_stream():
    """
    Handle document queries with streaming for creative mode.
    
    This endpoint streams progress updates and the final answer for better UX.
    """
    data = request.args
    
    # Authenticate user
    try:
        token = data.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = data.get('sessionId')
        
        if not session_id:
            return jsonify({'message': 'Session ID is required!'}), 400
            
        session_name = user_email + str(session_id.lower())
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Authentication failed!'}), 401
    
    # Check if it's creative mode
    mode = data.get('mode', 'default')
    if mode != 'creative':
        # For non-creative mode, redirect to regular endpoint
        return query_service.process_authenticated_query(data, user_email, session_name)
    
    # Generate streaming response
    def generate():
        try:
            # Extract query parameters
            user_query = query_service._extract_query_parameters(data)
            
            # Apply guardrails
            guardrail_response = query_service._guardrail.process_input(user_query["message"])
            if guardrail_response.get("status") == "blocked":
                yield f"data: {json.dumps({'type': 'error', 'content': guardrail_response})}\n\n"
                return
                
            user_query["message"] = guardrail_response.get("sanitized_input", user_query["message"])
            
            # Get streaming response from creative service
            from app.services.creative_reasoning_service import get_creative_reasoning_service
            creative_service = get_creative_reasoning_service()
            
            for event in creative_service.process_creative_query_stream(
                user_query["message"],
                session_name,
                user_query["input_language"],
                user_query["output_language"],
                user_query["filenames"],
                user_query["hascsvxl"]
            ):
                yield f"data: {json.dumps(event)}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable Nginx buffering
            'Connection': 'keep-alive'
        }
    )

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