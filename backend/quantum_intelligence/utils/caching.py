"""
Caching utilities with TTL and memory management
"""

import asyncio
import time
import pickle
import weakref
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, TypeVar, Generic
from dataclasses import dataclass
from collections import OrderedDict
import logging
from functools import wraps

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.exceptions import CacheError, ErrorCodes

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    value: Any
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_accessed: float = 0
    
    def __post_init__(self):
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the value and update statistics"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class CacheService(ABC, Generic[T]):
    """Abstract base class for cache services"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass
    
    async def get_or_compute(
        self, 
        key: str, 
        compute_func: callable, 
        ttl: Optional[int] = None
    ) -> T:
        """Get value from cache or compute and cache it"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value
        if asyncio.iscoroutinefunction(compute_func):
            computed_value = await compute_func()
        else:
            computed_value = compute_func()
        
        # Cache the computed value
        await self.set(key, computed_value, ttl)
        return computed_value


class MemoryCache(CacheService[T]):
    """In-memory cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            
            return entry.access()
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._stats["sets"] += 1
            
            # Evict if necessary
            await self._evict_if_needed()
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        async with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return False
            
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self._stats["hits"] / max(self._stats["hits"] + self._stats["misses"], 1)
            }
    
    async def _evict_if_needed(self) -> None:
        """Evict entries if cache is full"""
        while len(self._cache) > self.max_size:
            # Remove least recently used item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats["evictions"] += 1
    
    async def _cleanup_expired(self) -> None:
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                        
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def close(self) -> None:
        """Close cache and cleanup resources"""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.clear()


if REDIS_AVAILABLE:
    class RedisCache(CacheService[T]):
        """Redis-based cache implementation"""
else:
    class RedisCache(CacheService[T]):
        """Placeholder Redis cache when redis is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Redis not available. Install redis package.")

        async def get(self, key: str):
            raise ImportError("Redis not available")

        async def set(self, key: str, value, ttl: Optional[int] = None):
            raise ImportError("Redis not available")

        async def delete(self, key: str) -> bool:
            raise ImportError("Redis not available")

        async def clear(self):
            raise ImportError("Redis not available")

        async def exists(self, key: str) -> bool:
            raise ImportError("Redis not available")

        async def get_stats(self):
            raise ImportError("Redis not available")

        async def close(self):
            pass

if REDIS_AVAILABLE:
    class _RedisCache(CacheService[T]):
        """Redis-based cache implementation"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600, max_size: int = 10000):
        if not REDIS_AVAILABLE:
            raise CacheError(
                "Redis not available. Install redis package.",
                ErrorCodes.CACHE_UNAVAILABLE
            )
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._redis = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def _get_redis(self):
        """Get Redis connection"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url)
                await self._redis.ping()
            except Exception as e:
                raise CacheError(
                    f"Failed to connect to Redis: {e}",
                    ErrorCodes.CACHE_UNAVAILABLE
                )
        return self._redis
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(key)
            
            if data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        try:
            redis_client = await self._get_redis()
            
            if ttl is None:
                ttl = self.default_ttl
            
            data = pickle.dumps(value)
            await redis_client.setex(key, ttl, data)
            self._stats["sets"] += 1
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            raise CacheError(
                f"Failed to set cache value: {e}",
                ErrorCodes.CACHE_WRITE_FAILED
            )
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            
            if result > 0:
                self._stats["deletes"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        try:
            redis_client = await self._get_redis()
            await redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self._get_redis()
            return bool(await redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            redis_client = await self._get_redis()
            info = await redis_client.info("memory")
            
            return {
                **self._stats,
                "redis_memory_used": info.get("used_memory", 0),
                "redis_memory_peak": info.get("used_memory_peak", 0),
                "hit_rate": self._stats["hits"] / max(self._stats["hits"] + self._stats["misses"], 1)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self._stats
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Cache decorators
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = ":".join(filter(None, key_parts))
            
            # Try to get from cache
            from ..config.dependencies import get_cache_service
            cache = get_cache_service()
            
            return await cache.get_or_compute(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl
            )
        
        return wrapper
    return decorator
