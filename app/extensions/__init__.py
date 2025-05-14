"""
Extensions initialization module.

This module configures Flask extensions used throughout the application.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import asyncio

def register_extensions(app):
    """
    Initialize and register Flask extensions with the application.
    
    Args:
        app (Flask): The Flask application
    """
    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
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