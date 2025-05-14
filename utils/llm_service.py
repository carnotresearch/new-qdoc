"""
Centralized LLM Service for icarKno

This module provides a single source of truth for all language model instances
throughout the application, ensuring consistent configuration and efficient 
resource utilization through instance caching.
"""

import os
import logging
import base64
from functools import lru_cache
from typing import Dict, Optional, List, Any, Union
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings  
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMService:
    """
    Centralized service for LLM instances used throughout the application.
    Implements singleton pattern and caching to avoid redundant initializations.
    """
    _instance = None
    _models = {}
    _embeddings = {}
    _vector_stores = {}
    _es_clients = {}
    
    def __new__(cls):
        """Ensure single instance following the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize service with configuration from environment variables."""
        # Load or reload environment variables
        load_dotenv()
        
        # Core API keys for LLM services
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Elasticsearch configuration
        self.es_cloud_id = os.getenv("ES_CLOUD_ID")
        self.es_api_key = os.getenv("ES_API_KEY")
        
        # MySQL database configuration
        self.mysql_username = os.getenv("MYSQL_USERNAME")
        self.mysql_password = os.getenv("MYSQL_PASSWORD")
        
        # App configuration
        self.app_secret = os.getenv("APP_SECRET", "7be8c589cfe1e5d7f37a2703404bfed1")  # Default from webhook/decorators/security.py
        
        # Validate required configuration
        if not self.openai_api_key:
            logging.warning("OPENAI_API_KEY not found in environment - LLM features will not work")
        
        if not self.es_cloud_id or not self.es_api_key:
            logging.warning("Elasticsearch configuration incomplete - Vector search features will not work")
            
        if not self.mysql_username or not self.mysql_password:
            logging.warning("MySQL configuration incomplete - Database features will not work")
    
    def get_openai_chat(self, model_name: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
        """
        Get a ChatOpenAI instance with caching based on parameters.
        
        Args:
            model_name: The OpenAI model to use
            temperature: Controls randomness (0=deterministic, 1=creative)
            
        Returns:
            ChatOpenAI instance
        """
        # Create a cache key based on model parameters
        cache_key = f"openai_{model_name}_{temperature}"
        
        # Return cached instance if available
        if cache_key in self._models:
            return self._models[cache_key]
        
        # Create new instance
        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=self.openai_api_key,
                temperature=temperature
            )
            
            # Cache the instance
            self._models[cache_key] = llm
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize ChatOpenAI model {model_name}: {str(e)}")
            raise
    
    def get_embeddings(self, model: str = "text-embedding-3-large") -> OpenAIEmbeddings:
        """
        Get OpenAI embeddings instance with caching.
        
        Args:
            model: The embedding model to use
            
        Returns:
            OpenAIEmbeddings instance
        """
        if model in self._embeddings:
            return self._embeddings[model]
        
        try:
            embeddings = OpenAIEmbeddings(
                model=model,
                api_key=self.openai_api_key
            )
            
            self._embeddings[model] = embeddings
            return embeddings
        except Exception as e:
            logging.error(f"Failed to initialize embeddings model {model}: {str(e)}")
            raise
    
    def get_elasticsearch_client(self) -> Elasticsearch:
        """
        Get an Elasticsearch client instance.
        
        Returns:
            Elasticsearch client
        """
        cache_key = "default"
        
        if cache_key in self._es_clients:
            return self._es_clients[cache_key]
        
        try:
            client = Elasticsearch(
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key
            )
            
            self._es_clients[cache_key] = client
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Elasticsearch client: {str(e)}")
            raise
    
    def get_vector_store(self, index_name: str, embedding: Optional[Any] = None) -> ElasticsearchStore:
        """
        Get Elasticsearch vector store with caching.
        
        Args:
            index_name: The name of the Elasticsearch index
            embedding: Optional custom embedding model
            
        Returns:
            ElasticsearchStore instance
        """
        cache_key = f"vector_store_{index_name}"
        
        if cache_key in self._vector_stores:
            return self._vector_stores[cache_key]
        
        if embedding is None:
            embedding = self.get_embeddings()
            
        try:
            store = ElasticsearchStore(
                es_cloud_id=self.es_cloud_id,
                es_api_key=self.es_api_key,
                index_name=index_name,
                embedding=embedding
            )
            
            self._vector_stores[cache_key] = store
            return store
        except Exception as e:
            logging.error(f"Failed to initialize vector store for index {index_name}: {str(e)}")
            raise
    
    def get_openai_chat_with_fallback(self, 
                                     model_names: List[str] = ["gpt-4o", "gpt-4o-mini"], 
                                     temperature: float = 0) -> ChatOpenAI:
        """
        Try to get a model from a list of fallback options.
        Useful when primary models might hit rate limits.
        
        Args:
            model_names: List of models to try in order
            temperature: Temperature setting for the model
            
        Returns:
            ChatOpenAI instance
            
        Raises:
            Exception: If all models fail
        """
        last_error = None
        for model_name in model_names:
            try:
                return self.get_openai_chat(model_name, temperature)
            except Exception as e:
                last_error = e
                logging.warning(f"Model {model_name} failed, trying fallback. Error: {str(e)}")
                continue
        
        # If we get here, all models failed
        logging.error(f"All fallback models failed. Last error: {str(last_error)}")
        raise last_error or Exception("All fallback models failed")
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear the cache of instances.
        
        Args:
            model_type: Optional type to clear (models, embeddings, vector_stores, es_clients)
                       If None, clear all caches
        """
        if model_type == "models" or model_type is None:
            self._models = {}
        if model_type == "embeddings" or model_type is None:
            self._embeddings = {}
        if model_type == "vector_stores" or model_type is None:
            self._vector_stores = {}
        if model_type == "es_clients" or model_type is None:
            self._es_clients = {}
            
    def process_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Process an image using a vision-capable model.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            
        Returns:
            Model's response to the image
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Get a client (current OpenAI library requires using client directly for vision)
            # This is still using our centralized API key
            client = get_openai_chat().client
            
            # Use a vision-capable model
            response = client.chat.completions.create(
                model="gpt-4.5-preview",  # Vision capable model
                messages=[
                    {"role": "system", "content": "You are an AI that accurately describes images."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error processing image with VLM: {str(e)}")
            raise


# Create a singleton instance
llm_service = LLMService()

# Convenience functions for easier imports throughout the codebase
def get_openai_chat(model_name: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    """Get a ChatOpenAI instance."""
    return llm_service.get_openai_chat(model_name, temperature)

def get_embeddings(model: str = "text-embedding-3-large") -> OpenAIEmbeddings:
    """Get an OpenAI embeddings instance."""
    return llm_service.get_embeddings(model)

def get_elasticsearch_client() -> Elasticsearch:
    """Get an Elasticsearch client."""
    return llm_service.get_elasticsearch_client()

def get_vector_store(index_name: str, embedding: Optional[Any] = None) -> ElasticsearchStore:
    """Get an Elasticsearch vector store."""
    return llm_service.get_vector_store(index_name, embedding)

def get_openai_chat_with_fallback(model_names: List[str] = ["gpt-4o", "gpt-4o-mini"], 
                                temperature: float = 0) -> ChatOpenAI:
    """Get a ChatOpenAI instance with fallback options."""
    return llm_service.get_openai_chat_with_fallback(model_names, temperature)

def process_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """Process an image with a vision-capable model."""
    return llm_service.process_image(image_path, prompt)