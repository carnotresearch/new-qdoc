"""
Extensions initialization module.
This module configures Flask extensions used throughout the application.
"""

from flask import request
from flask_limiter import Limiter
import asyncio

def get_real_ip():
    """Get the real IP address from proxy headers."""
    # Skip rate limiting for OPTIONS requests
    if request.method == 'OPTIONS':
        return None  # This tells Flask-Limiter to exempt this request

    # Check for X-Forwarded-For header
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs - use the first one (client IP)
        return forwarded_for.split(',')[0].strip()
    
    # If no X-Forwarded-For, try X-Real-IP
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fall back to remote address as last resort
    return request.remote_addr

def register_extensions(app):
    """
    Initialize and register Flask extensions with the application.
    
    Args:
        app (Flask): The Flask application
    """
    # Initialize rate limiter with custom key function
    limiter = Limiter(
        get_real_ip,  # Use custom function for getting the real IP
        app=app,
        storage_uri=app.config.get("RATELIMIT_STORAGE_URL"),
        default_limits=[app.config.get("RATELIMIT_DEFAULT")],
        strategy=app.config.get("RATELIMIT_STRATEGY")
    )
    app.extensions["limiter"] = limiter
    
    # Initialize asyncio event loop
    get_or_create_eventloop()
    
    return app

def get_or_create_eventloop():
    """
    Get the current asyncio event loop or create a new one if none exists.
    
    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop