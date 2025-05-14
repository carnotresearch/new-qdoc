"""
Authentication API routes.

This module defines routes for user authentication, token validation, and session management.
"""

import logging
from flask import Blueprint, request, jsonify

from app.services.auth_service import get_auth_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Get service instance
auth_service = get_auth_service()

@auth_bp.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Simple health check endpoint to verify the application is running.
    
    Returns:
        JSON response with status message
    """
    return jsonify({"message": "app is up and running"})

# Note: Other authentication endpoints would go here.
# For now, we include only the healthcheck endpoint since the current codebase
# doesn't appear to have explicit authentication endpoints (just token validation).

# TODO: Add proper user login, registration, and token refresh endpoints when needed.