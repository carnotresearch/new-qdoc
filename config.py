from dotenv import load_dotenv
import os

load_dotenv()  # Load once on app startup

class Config:
    # Elasticsearch
    ES_CLOUD_ID: str = os.getenv("ES_CLOUD_ID")
    ES_API_KEY: str = os.getenv("ES_API_KEY")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # MongoDB
    MONGO_URL: str = os.getenv("MONGO_URL")

    # MySQL
    MYSQL_HOST: str = os.getenv("MYSQL_HOST")
    MYSQL_USERNAME: str = os.getenv("MYSQL_USERNAME")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", 3306))
    
    # App Settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def validate(cls):
        missing = [var for var, value in vars(cls).items() 
                 if not var.startswith("__") and value is None]
        if missing:
            raise EnvironmentError(
                f"Missing environment variables: {', '.join(missing)}"
            )

# Validate on import
Config.validate()