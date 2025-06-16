# config.py in root directory
from app.config import ConfigFactory

# Create a compatibility class
class Config:
    """Compatibility wrapper for old Config class"""
    
    # Get the actual config
    _config = ConfigFactory.get_config('default')
    
    # Define properties that redirect to the new config
    ES_CLOUD_ID = _config.ES_CLOUD_ID
    ES_API_KEY = _config.ES_API_KEY
    OPENAI_API_KEY = _config.OPENAI_API_KEY
    MONGO_URL = _config.MONGO_URL
    MYSQL_HOST = _config.MYSQL_HOST
    MYSQL_USERNAME = _config.MYSQL_USERNAME
    MYSQL_PASSWORD = _config.MYSQL_PASSWORD
    MYSQL_PORT = _config.MYSQL_PORT
    NEO4J_URI = _config.NEO4J_URI
    NEO4J_USERNAME = _config.NEO4J_USERNAME
    NEO4J_PASSWORD = _config.NEO4J_PASSWORD
    GEMINI_API_KEY = _config.GEMINI_API_KEY