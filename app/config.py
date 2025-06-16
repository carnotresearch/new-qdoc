"""
Configuration module for the Flask application.

This module provides configuration management through environment variables
and different configuration classes for different environments.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaseConfig:
    """Base configuration with common settings."""
    
    # App settings
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    DEBUG = False
    TESTING = False
    
    # Elasticsearch
    ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
    ES_API_KEY = os.getenv("ES_API_KEY")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # MongoDB
    MONGO_URL = os.getenv("MONGO_URL")
    
    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USERNAME = os.getenv("MYSQL_USERNAME")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    
    # JWT Settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret")
    JWT_ALGORITHM = "HS256"
    
    # Rate limiting
    RATELIMIT_DEFAULT = "300 per day, 30 per hour"
    RATELIMIT_STRATEGY = "fixed-window"
    RATELIMIT_STORAGE_URL = "memory://"


    # Neo4j Settings
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
    
    @staticmethod
    def validate():
        """Validate that all required settings are present."""
        required_vars = [
            "ES_CLOUD_ID", "ES_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "MONGO_URL", "MYSQL_HOST", "MYSQL_USERNAME", "MYSQL_PASSWORD","NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD",
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                f"Missing environment variables: {', '.join(missing)}"
            )

class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    DEBUG = True
    # Add development-specific settings

class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = True
    # Add testing-specific settings
    
    # Use in-memory databases for testing
    MONGO_URL = "mongodb://localhost:27017/test"
    
    # Override validation for testing
    @staticmethod
    def validate():
        """Skip validation in testing environment."""
        pass

class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    # Add production-specific settings
    
    # Override with stronger secret key for production
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required in production")
    
    # More strict rate limiting for production
    RATELIMIT_DEFAULT = "200 per day, 200 per hour"

class ConfigFactory:
    """Factory class to create configuration objects."""
    
    @staticmethod
    def get_config(config_name):
        """
        Get the appropriate configuration class based on the name.
        
        Args:
            config_name (str): Name of the configuration to use
            
        Returns:
            Config class: The configuration class to use
            
        Raises:
            ValueError: If an invalid configuration name is provided
        """
        configs = {
            "development": DevelopmentConfig,
            "testing": TestingConfig,
            "production": ProductionConfig,
            "default": DevelopmentConfig
        }
        
        if config_name not in configs:
            raise ValueError(f"Invalid configuration name: {config_name}")
        
        config_class = configs[config_name]
        
        # Validate configuration
        if config_name != "testing":
            config_class.validate()
            
        return config_class
    


    