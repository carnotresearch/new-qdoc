"""
Centralized LLM service for managing all LLM interactions across the application.

This module provides a factory pattern for creating and managing LLM instances,
making it easy to switch between different providers and configurations.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackHandler
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"

class LLMType(Enum):
    """Different LLM configurations for different use cases."""
    FAST = "fast"           # Quick responses, lower cost
    STANDARD = "standard"   # Balanced performance
    COMPREHENSIVE = "comprehensive"  # High quality, complex tasks
    CREATIVE = "creative"   # Creative and reasoning tasks
    CODE = "code"          # Code generation and analysis

@dataclass
class LLMConfig:
    """Configuration for LLM instances."""
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    streaming: bool = False
    callbacks: Optional[List[BaseCallbackHandler]] = None
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}
        if self.callbacks is None:
            self.callbacks = []

class BaseLLMFactory(ABC):
    """Abstract base class for LLM factories."""
    
    @abstractmethod
    def create_llm(self, config: LLMConfig) -> BaseLanguageModel:
        """Create an LLM instance with the given configuration."""
        pass
    
    @abstractmethod
    def get_default_configs(self) -> Dict[LLMType, LLMConfig]:
        """Get default configurations for different LLM types."""
        pass

class OpenAILLMFactory(BaseLLMFactory):
    """Factory for creating OpenAI LLM instances."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_llm(self, config: LLMConfig) -> ChatOpenAI:
        """Create a ChatOpenAI instance."""
        # Prepare parameters
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "streaming": config.streaming,
            "api_key": self.api_key,
            **config.additional_params
        }
        
        # Add callbacks if provided
        if config.callbacks:
            params["callbacks"] = config.callbacks
            
        return ChatOpenAI(**params)
    
    def get_default_configs(self) -> Dict[LLMType, LLMConfig]:
        """Get default OpenAI configurations."""
        return {
            LLMType.FAST: LLMConfig(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1024,
                timeout=30
            ),
            LLMType.STANDARD: LLMConfig(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2048,
                timeout=60
            ),
            LLMType.COMPREHENSIVE: LLMConfig(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=8000,
                timeout=120
            ),
            LLMType.CREATIVE: LLMConfig(
                model="gpt-4o",
                temperature=0.4,
                max_tokens=8000,
                timeout=120
            ),
            LLMType.CODE: LLMConfig(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=4096,
                timeout=90
            )
        }

class GoogleLLMFactory(BaseLLMFactory):
    """Factory for creating Google Gemini LLM instances."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_llm(self, config: LLMConfig) -> ChatGoogleGenerativeAI:
        """Create a ChatGoogleGenerativeAI instance."""
        # Prepare parameters
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "google_api_key": self.api_key,
            **config.additional_params
        }
        
        # Add callbacks if provided (Gemini also supports callbacks)
        if config.callbacks:
            params["callbacks"] = config.callbacks
            
        return ChatGoogleGenerativeAI(**params)
    
    def get_default_configs(self) -> Dict[LLMType, LLMConfig]:
        """Get default Google configurations."""
        return {
            LLMType.FAST: LLMConfig(
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=1024
            ),
            LLMType.STANDARD: LLMConfig(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=2048
            ),
            LLMType.COMPREHENSIVE: LLMConfig(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=8000
            ),
            LLMType.CREATIVE: LLMConfig(
                model="gemini-2.0-flash",
                temperature=0.4,
                max_tokens=8000
            ),
            LLMType.CODE: LLMConfig(
                model="gemini-2.0-flash",
                temperature=0.0,
                max_tokens=4096
            )
        }

class LLMService:
    """
    Centralized LLM service using Singleton pattern.
    
    This service manages all LLM instances across the application,
    providing a consistent interface for different LLM providers and configurations.
    """
    
    _instance: Optional['LLMService'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'LLMService':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the LLM service."""
        if not self._initialized:
            self._factories: Dict[LLMProvider, BaseLLMFactory] = {}
            self._cached_llms: Dict[str, BaseLanguageModel] = {}
            self._default_provider = LLMProvider.GOOGLE
            self._initialize_factories()
            self._initialized = True
            logger.info("LLM Service initialized")
    
    def _initialize_factories(self):
        """Initialize LLM factories for available providers."""
        try:
            # Initialize OpenAI factory
            if Config.OPENAI_API_KEY:
                self._factories[LLMProvider.OPENAI] = OpenAILLMFactory(Config.OPENAI_API_KEY)
                logger.info("OpenAI factory initialized")
            
            # Initialize Google factory
            if Config.GEMINI_API_KEY:
                self._factories[LLMProvider.GOOGLE] = GoogleLLMFactory(Config.GEMINI_API_KEY)
                logger.info("Google factory initialized")
            
            if not self._factories:
                raise ValueError("No LLM providers configured. Please check your API keys.")
                
        except Exception as e:
            logger.error(f"Error initializing LLM factories: {e}")
            raise
    
    def set_default_provider(self, provider: LLMProvider):
        """Set the default LLM provider."""
        if provider not in self._factories:
            raise ValueError(f"Provider {provider} is not available")
        self._default_provider = provider
        logger.info(f"Default LLM provider set to {provider.value}")
    
    def get_llm(
        self,
        llm_type: LLMType = LLMType.STANDARD,
        provider: Optional[LLMProvider] = None,
        custom_config: Optional[LLMConfig] = None,
        cache: bool = True
    ) -> BaseLanguageModel:
        """
        Get an LLM instance.
        
        Args:
            llm_type: Type of LLM configuration to use
            provider: LLM provider (uses default if None)
            custom_config: Custom configuration (overrides default)
            cache: Whether to cache the LLM instance
            
        Returns:
            Configured LLM instance
        """
        # Use default provider if none specified
        if provider is None:
            provider = self._default_provider
        
        # Check if provider is available
        if provider not in self._factories:
            logger.warning(f"Provider {provider} not available, using default {self._default_provider}")
            provider = self._default_provider
        
        # Create cache key (don't cache if custom config has callbacks)
        cache_key = f"{provider.value}_{llm_type.value}"
        if custom_config:
            cache_key += f"_custom_{hash(str(custom_config))}"
            # Don't cache if callbacks are present
            if custom_config.callbacks:
                cache = False
        
        # Return cached instance if available
        if cache and cache_key in self._cached_llms:
            return self._cached_llms[cache_key]
        
        # Get factory and configuration
        factory = self._factories[provider]
        
        if custom_config:
            config = custom_config
        else:
            default_configs = factory.get_default_configs()
            config = default_configs.get(llm_type)
            if config is None:
                logger.warning(f"No default config for {llm_type}, using STANDARD")
                config = default_configs[LLMType.STANDARD]
        
        # Create LLM instance
        try:
            llm = factory.create_llm(config)
            
            # Cache if enabled and no callbacks
            if cache and not config.callbacks:
                self._cached_llms[cache_key] = llm
            
            logger.debug(f"Created LLM: {provider.value} {llm_type.value} {config.model}")
            return llm
            
        except Exception as e:
            logger.error(f"Error creating LLM: {e}")
            raise
    
    def get_streaming_llm(
        self,
        llm_type: LLMType = LLMType.STANDARD,
        provider: Optional[LLMProvider] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> BaseLanguageModel:
        """
        Get a streaming-enabled LLM instance.
        
        Args:
            llm_type: Type of LLM configuration to use
            provider: LLM provider (uses default if None)
            callbacks: List of callback handlers for streaming
            
        Returns:
            Streaming-enabled LLM instance
        """
        # Get base config and enable streaming
        factory = self._factories.get(provider or self._default_provider)
        if not factory:
            raise ValueError(f"Provider {provider or self._default_provider} not available")
        
        base_config = factory.get_default_configs()[llm_type]
        streaming_config = LLMConfig(
            model=base_config.model,
            temperature=base_config.temperature,
            max_tokens=base_config.max_tokens,
            timeout=base_config.timeout,
            streaming=True,
            callbacks=callbacks or [],
            additional_params=base_config.additional_params
        )
        
        return self.get_llm(
            llm_type=llm_type,
            provider=provider,
            custom_config=streaming_config,
            cache=False  # Don't cache streaming instances with callbacks
        )
    
    def clear_cache(self):
        """Clear the LLM cache."""
        self._cached_llms.clear()
        logger.info("LLM cache cleared")
    
    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of available LLM providers."""
        return list(self._factories.keys())
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers and configurations."""
        info = {
            "default_provider": self._default_provider.value,
            "available_providers": [p.value for p in self._factories.keys()],
            "llm_types": [t.value for t in LLMType],
            "configurations": {}
        }
        
        for provider, factory in self._factories.items():
            configs = factory.get_default_configs()
            info["configurations"][provider.value] = {
                llm_type.value: {
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
                for llm_type, config in configs.items()
            }
        
        return info

# Global instance - Singleton
_llm_service_instance: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance.
    
    Returns:
        LLMService: The global LLM service instance
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance

# Convenience functions for common use cases
def get_fast_llm() -> BaseLanguageModel:
    """Get a fast LLM for quick responses."""
    return get_llm_service().get_llm(LLMType.FAST)

def get_standard_llm() -> BaseLanguageModel:
    """Get a standard LLM for general use."""
    return get_llm_service().get_llm(LLMType.STANDARD)

def get_comprehensive_llm() -> BaseLanguageModel:
    """Get a comprehensive LLM for complex tasks."""
    return get_llm_service().get_llm(LLMType.COMPREHENSIVE)

def get_creative_llm() -> BaseLanguageModel:
    """Get a creative LLM for reasoning and creative tasks."""
    return get_llm_service().get_llm(LLMType.CREATIVE)

def get_code_llm() -> BaseLanguageModel:
    """Get a code-focused LLM."""
    return get_llm_service().get_llm(LLMType.CODE)

def get_streaming_llm(llm_type: LLMType = LLMType.STANDARD, 
                     callbacks: Optional[List[BaseCallbackHandler]] = None) -> BaseLanguageModel:
    """Get a streaming LLM with callbacks."""
    return get_llm_service().get_streaming_llm(llm_type, callbacks=callbacks)

# Configuration functions
def set_default_provider(provider: Union[LLMProvider, str]):
    """Set the default LLM provider."""
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    get_llm_service().set_default_provider(provider)

def get_provider_info() -> Dict[str, Any]:
    """Get information about available providers."""
    return get_llm_service().get_provider_info()