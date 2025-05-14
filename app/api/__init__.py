"""
API blueprints registration module.

This module registers all API blueprints with the Flask application.
"""

def register_blueprints(app):
    """
    Register all blueprints with the application.
    
    Args:
        app (Flask): The Flask application
    """
    # Import all blueprints
    from app.api.auth import auth_bp
    from app.api.documents import documents_bp
    from app.api.queries import queries_bp
    from app.api.accounts import accounts_bp
    from webhook import webhook_blueprint  # Import existing webhook blueprint
    
    # Register blueprints with the app
    app.register_blueprint(auth_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(queries_bp)
    app.register_blueprint(accounts_bp)
    
    # Register the existing webhook blueprint
    app.register_blueprint(webhook_blueprint, url_prefix="/api")
    
    return app