"""
Application factory module for the Flask backend.

This module creates and configures the Flask application using the factory pattern,
which allows for better testing and the creation of multiple app instances.
"""

import logging
from flask import Flask
from flask_cors import CORS

from app.extensions import register_extensions
from app.api import register_blueprints
from app.config import ConfigFactory

def create_app(config_name="default"):
    """
    Create and configure the Flask application.
    
    Args:
        config_name (str): The configuration to use (default, development, testing, production)
        
    Returns:
        Flask: The configured Flask application
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Creating app with {config_name} configuration")
    
    # Create and configure the app
    app = Flask(__name__)
    
    # Load configuration
    config = ConfigFactory.get_config(config_name)
    app.config.from_object(config)
    
    # Configure CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Initialize extensions
    register_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register shell context processors (optional, for Flask shell)
    @app.shell_context_processor
    def make_shell_context():
        return {'app': app}
    
    return app

def register_error_handlers(app):
    """
    Register error handlers for the application.
    
    Args:
        app (Flask): The Flask application
    """
    from flask import jsonify
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400
    
    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({"error": "Unauthorized", "message": str(e)}), 401
    
    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({"error": "Forbidden", "message": str(e)}), 403
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404
    
    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({"error": "Rate limit exceeded", "message": str(e)}), 429
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({"error": "Internal server error", "message": str(e)}), 500