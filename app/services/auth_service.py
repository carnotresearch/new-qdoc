"""
Authentication service module.

This module provides functions for handling user authentication and token management.
"""

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication and token management."""
    
    def __init__(self, jwt_secret_key: str, jwt_algorithm: str = "HS256"):
        """
        Initialize the authentication service.
        
        Args:
            jwt_secret_key (str): Secret key for JWT token encoding/decoding
            jwt_algorithm (str): Algorithm used for JWT token encoding/decoding
        """
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
    
    def authenticate(self, token: str) -> str:
        """
        Authenticate a user with the provided JWT token.
        
        Args:
            token (str): JWT token to authenticate
            
        Returns:
            str: User email extracted from the token
            
        Raises:
            ExpiredSignatureError: If the token has expired
            InvalidTokenError: If the token is invalid
        """
        data = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
        current_user = data['email']
        logger.info(f'User authenticated: {current_user}')
        return current_user
    
    def get_full_session_name(self, token: str, session_id: str) -> str:
        """
        Get the full session name by combining user email and session ID.
        
        Args:
            token (str): JWT token containing user email
            session_id (str): Session identifier
            
        Returns:
            str: Combined session name (email + session_id)
            
        Raises:
            ExpiredSignatureError: If the token has expired
            InvalidTokenError: If the token is invalid
        """
        user_email = self.authenticate(token)
        return user_email + str(session_id.lower())
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new JWT token for a user.
        
        Args:
            user_data (Dict[str, Any]): User data to encode in the token
            
        Returns:
            str: Generated JWT token
        """
        return jwt.encode(user_data, self.jwt_secret_key, algorithm=self.jwt_algorithm)

# Create a singleton instance
auth_service = None

def get_auth_service(app=None):
    """
    Get the authentication service singleton instance.
    
    Args:
        app (Flask, optional): Flask application for configuration
        
    Returns:
        AuthService: The authentication service instance
    """
    global auth_service
    
    if auth_service is None:
        if app is None:
            # Use default values if app is not provided
            auth_service = AuthService("secret")
        else:
            auth_service = AuthService(
                app.config.get("JWT_SECRET_KEY", "secret"),
                app.config.get("JWT_ALGORITHM", "HS256")
            )
    
    return auth_service