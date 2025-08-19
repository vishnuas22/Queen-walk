"""
Dependency injection container for Quantum Intelligence Engine
"""

import asyncio
from typing import Dict, Any, Optional, TypeVar, Type, Callable
from functools import lru_cache, wraps
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from .settings import QuantumEngineConfig, get_config
from ..core.exceptions import ConfigurationError, ErrorCodes
from ..utils.caching import CacheService, MemoryCache, RedisCache
from ..utils.monitoring import MetricsService, HealthCheckService

T = TypeVar('T')


class DependencyContainer:
    """Dependency injection container for managing service instances"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._config = get_config()
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance"""
        self._singletons[name] = instance
        logger.info(f"Registered singleton: {name}")
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function"""
        self._factories[name] = factory
        logger.info(f"Registered factory: {name}")
    
    def get(self, name: str) -> Any:
        """Get a service instance"""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Check if we have a factory
        if name in self._factories:
            instance = self._factories[name]()
            # Cache as singleton if it's a service
            if hasattr(instance, '__class__') and 'Service' in instance.__class__.__name__:
                self._singletons[name] = instance
            return instance
        
        raise ConfigurationError(
            f"Service '{name}' not found in dependency container",
            ErrorCodes.INVALID_CONFIG,
            {"available_services": list(self._singletons.keys()) + list(self._factories.keys())}
        )
    
    def has(self, name: str) -> bool:
        """Check if a service is registered"""
        return name in self._singletons or name in self._factories
    
    def clear(self) -> None:
        """Clear all registered services"""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global container instance
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get the global dependency container"""
    return _container


@lru_cache(maxsize=1)
def get_cache_service() -> CacheService:
    """Get cache service instance"""
    config = get_config()
    
    # Handle both enum and string values for cache_backend
    cache_backend = getattr(config.cache_backend, 'value', config.cache_backend)
    
    if cache_backend == "redis" and config.redis_url:
        return RedisCache(
            redis_url=config.redis_url,
            default_ttl=config.cache_ttl,
            max_size=config.max_cache_size
        )
    else:
        # Create simple memory cache without async tasks for now
        class SimpleMemoryCache:
            def __init__(self, max_size=1000, default_ttl=3600):
                self.max_size = max_size
                self.default_ttl = default_ttl
                self._cache = {}
            
            async def get(self, key: str):
                return self._cache.get(key)
            
            async def set(self, key: str, value, ttl=None):
                self._cache[key] = value
                return True
            
            async def delete(self, key: str):
                return self._cache.pop(key, None) is not None
            
            async def clear(self):
                self._cache.clear()
                return True
            
            async def close(self):
                self._cache.clear()
        
        return SimpleMemoryCache(
            max_size=config.max_cache_size,
            default_ttl=config.cache_ttl
        )


@lru_cache(maxsize=1)
def get_metrics_service() -> MetricsService:
    """Get metrics service instance"""
    # Create simple metrics service without async complexity for initialization
    class SimpleMetricsService:
        def __init__(self, enabled=True, prometheus_enabled=False, port=8080):
            self.enabled = enabled
            self.prometheus_enabled = prometheus_enabled
            self.port = port
            self.metrics = {}
        
        async def record_metric(self, name: str, value: float, labels=None):
            self.metrics[name] = value
        
        async def get_metrics(self):
            return self.metrics
        
        async def close(self):
            pass
    
    config = get_config()
    return SimpleMetricsService(
        enabled=config.enable_metrics,
        prometheus_enabled=config.enable_prometheus_metrics,
        port=config.metrics_port
    )


@lru_cache(maxsize=1)
def get_health_service() -> HealthCheckService:
    """Get health check service instance"""
    # Create simple health service without async tasks for initialization
    class SimpleHealthService:
        def __init__(self, enabled=True, check_interval=30):
            self.enabled = enabled
            self.check_interval = check_interval
            self.status = "healthy"
        
        async def check_health(self):
            return {"status": "healthy", "checks": []}
        
        async def close(self):
            pass
    
    config = get_config()
    return SimpleHealthService(
        enabled=config.enable_health_checks,
        check_interval=config.health_check_interval
    )


def get_quantum_engine():
    """Get quantum engine instance with dependency injection"""
    from ..core.engine import QuantumLearningIntelligenceEngine
    
    container = get_container()
    
    # Check if already exists as singleton (not factory!)
    if "quantum_engine" in container._singletons:
        return container._singletons["quantum_engine"]
    
    # Create new instance with dependencies
    config = get_config()
    cache_service = get_cache_service()
    metrics_service = get_metrics_service()
    health_service = get_health_service()
    
    engine = QuantumLearningIntelligenceEngine(
        config=config,
        cache_service=cache_service,
        metrics_service=metrics_service,
        health_service=health_service
    )
    
    # Register as singleton directly
    container.register_singleton("quantum_engine", engine)
    return engine


def setup_dependencies() -> None:
    """Setup all dependencies and services"""
    logger.info("Setting up Quantum Intelligence Engine dependencies...")
    
    config = get_config()
    container = get_container()
    
    # Validate configuration
    if not config.has_ai_provider:
        raise ConfigurationError(
            "No AI provider configured. Please set at least one API key.",
            ErrorCodes.MISSING_API_KEY
        )
    
    # Register core services
    container.register_factory("cache_service", get_cache_service)
    container.register_factory("metrics_service", get_metrics_service)
    container.register_factory("health_service", get_health_service)
    
    # Quantum engine uses direct singleton pattern, no factory needed
    
    logger.info("Dependencies setup complete")


def dependency_injection(service_name: str):
    """Decorator for dependency injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            service = container.get(service_name)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


async def cleanup_dependencies() -> None:
    """Cleanup all dependencies and close connections"""
    logger.info("Cleaning up dependencies...")
    
    container = get_container()
    
    # Cleanup cache service
    if container.has("cache_service"):
        cache_service = container.get("cache_service")
        if hasattr(cache_service, 'close'):
            await cache_service.close()
    
    # Cleanup metrics service
    if container.has("metrics_service"):
        metrics_service = container.get("metrics_service")
        if hasattr(metrics_service, 'close'):
            await metrics_service.close()
    
    # Cleanup quantum engine
    if container.has("quantum_engine"):
        engine = container.get("quantum_engine")
        if hasattr(engine, 'close'):
            await engine.close()
    
    container.clear()
    logger.info("Dependencies cleanup complete")


# Context manager for dependency lifecycle
class DependencyManager:
    """Context manager for dependency lifecycle management"""
    
    async def __aenter__(self):
        setup_dependencies()
        return get_container()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await cleanup_dependencies()


def get_dependency_manager() -> DependencyManager:
    """Get dependency manager for context management"""
    return DependencyManager()
