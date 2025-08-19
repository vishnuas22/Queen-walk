"""
Configuration management for Quantum Intelligence Engine
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables first
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# Try to import pydantic settings, provide fallback
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator
        PYDANTIC_AVAILABLE = True
    except ImportError:
        PYDANTIC_AVAILABLE = False

        # Create fallback BaseSettings
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def Field(default=None, **kwargs):
            return default

        def validator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class DatabaseBackend(str, Enum):
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class QuantumEngineConfig(BaseSettings):
    """
    Comprehensive configuration for Quantum Intelligence Engine
    """

    def __init__(self, **kwargs):
        if not PYDANTIC_AVAILABLE:
            # Fallback initialization without pydantic
            # Set default values for all fields
            self.app_name = kwargs.get("app_name", "Quantum Learning Intelligence Engine")
            self.version = kwargs.get("version", "2.0.0")
            self.debug = kwargs.get("debug", False)
            self.environment = kwargs.get("environment", "development")
            
            # Load API keys from environment with fallback to kwargs
            self.groq_api_key = os.getenv("GROQ_API_KEY") or kwargs.get("groq_api_key")
            self.openai_api_key = os.getenv("OPENAI_API_KEY") or kwargs.get("openai_api_key") 
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or kwargs.get("anthropic_api_key")
            self.gemini_api_key = os.getenv("GEMINI_API_KEY") or kwargs.get("gemini_api_key")
            
            self.primary_model = kwargs.get("primary_model", "deepseek-r1-distill-llama-70b")
            self.enable_neural_networks = kwargs.get("enable_neural_networks", True)
            self.enable_caching = kwargs.get("enable_caching", True)
            self.enable_metrics = kwargs.get("enable_metrics", True)
            self.cache_backend = kwargs.get("cache_backend", "memory")
            self.max_cache_size = kwargs.get("max_cache_size", 1000)
            self.cache_ttl = kwargs.get("cache_ttl", 3600)
            
            # Database settings
            self.mongo_url = os.getenv("MONGO_URL") or kwargs.get("mongo_url")
            self.database_name = os.getenv("DB_NAME") or kwargs.get("database_name", "masterx_quantum")
        else:
            super().__init__(**kwargs)
    
    # Core Settings
    app_name: str = "Quantum Learning Intelligence Engine"
    version: str = "2.0.0"
    debug: bool = False
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # AI Provider Settings
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"), env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"), env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"), env="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"), env="GEMINI_API_KEY")
    
    # Model Configuration
    primary_model: str = "deepseek-r1-distill-llama-70b"
    fallback_models: List[str] = ["gpt-3.5-turbo", "claude-3-haiku"]
    model_timeout: int = 30
    max_retries: int = 3
    
    # Neural Network Settings
    enable_neural_networks: bool = True
    neural_network_device: str = "cpu"  # cpu, cuda, mps
    model_cache_size: int = 3
    torch_threads: int = 4
    
    # Database Configuration
    database_backend: DatabaseBackend = DatabaseBackend.MONGODB
    mongo_url: Optional[str] = Field(default=None, env="MONGO_URL")
    database_name: str = Field(default="masterx_quantum", env="DB_NAME")
    postgres_url: Optional[str] = Field(default=None, env="POSTGRES_URL")
    sqlite_path: str = "quantum_engine.db"
    
    # Caching Configuration
    cache_backend: CacheBackend = CacheBackend.MEMORY
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout: int = 60
    enable_compression: bool = True
    enable_metrics: bool = True
    
    # Security Settings
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    enable_cors: bool = True
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    secret_key: str = Field(default="quantum-secret-key", env="SECRET_KEY")
    
    # Logging Configuration
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    enable_structured_logging: bool = True
    log_file: Optional[str] = None
    
    # Feature Flags
    enable_multimodal: bool = True
    enable_emotional_ai: bool = True
    enable_collaboration: bool = True
    enable_gamification: bool = True
    enable_quantum_algorithms: bool = True
    enable_enterprise_features: bool = False
    
    # Streaming Settings
    streaming_chunk_size: int = 1024
    streaming_delay: float = 0.05
    enable_adaptive_streaming: bool = True
    
    # Monitoring Settings
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_prometheus_metrics: bool = False
    metrics_port: int = 9090
    
    @validator("groq_api_key", "openai_api_key", "anthropic_api_key")
    def validate_api_keys(cls, v):
        if v and len(v) < 10:
            raise ValueError("API key appears to be invalid")
        return v
    
    @validator("allowed_origins")
    def validate_origins(cls, v):
        if not v:
            return ["*"]  # Allow all origins if none specified
        return v
    
    @property
    def has_ai_provider(self) -> bool:
        """Check if at least one AI provider is configured"""
        return bool(self.groq_api_key or self.openai_api_key or self.anthropic_api_key or self.gemini_api_key)
    
    @property
    def database_url(self) -> Optional[str]:
        """Get the appropriate database URL based on backend"""
        if self.database_backend == DatabaseBackend.MONGODB:
            return self.mongo_url
        elif self.database_backend == DatabaseBackend.POSTGRESQL:
            return self.postgres_url
        elif self.database_backend == DatabaseBackend.SQLITE:
            return f"sqlite:///{self.sqlite_path}"
        return None
    
    class Config:
        env_file = ".env"
        env_prefix = "QUANTUM_"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment


# Global configuration instance
config = QuantumEngineConfig()


def get_config() -> QuantumEngineConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> QuantumEngineConfig:
    """Reload configuration from environment"""
    global config
    config = QuantumEngineConfig()
    return config
