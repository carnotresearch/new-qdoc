from flask import Blueprint

webhook_blueprint = Blueprint("webhook", __name__)

# Register routes
from . import webhook  # Import routes here
