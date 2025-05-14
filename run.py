"""
Main application entry point.

This module runs the Flask application using application factory pattern.
"""

from app import create_app

if __name__ == "__main__":
    app = create_app('production')
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        ssl_context=(
            '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/fullchain.pem', 
            '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/privkey.pem'
        )
    )