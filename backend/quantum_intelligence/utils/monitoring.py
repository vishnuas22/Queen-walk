"""
Monitoring utilities for metrics and health checks
"""

import time
import asyncio
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from functools import wraps

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..core.exceptions import ConfigurationError


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_func: Callable
    timeout: float = 5.0
    critical: bool = True
    last_result: Optional[bool] = None
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsService:
    """Service for collecting and exposing metrics"""
    
    def __init__(self, enabled: bool = True, prometheus_enabled: bool = False, port: int = 9090):
        self.enabled = enabled
        self.prometheus_enabled = prometheus_enabled
        self.port = port
        
        # In-memory metrics storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Prometheus metrics
        self._prometheus_counters: Dict[str, Counter] = {}
        self._prometheus_gauges: Dict[str, Gauge] = {}
        self._prometheus_histograms: Dict[str, Histogram] = {}
        
        if self.enabled and self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if not self.enabled:
            return
        
        self._counters[name] += value
        
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            if name not in self._prometheus_counters:
                self._prometheus_counters[name] = Counter(
                    name.replace('.', '_'), 
                    f'Counter metric: {name}',
                    list(labels.keys()) if labels else []
                )
            
            if labels:
                self._prometheus_counters[name].labels(**labels).inc(value)
            else:
                self._prometheus_counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        if not self.enabled:
            return
        
        self._gauges[name] = value
        
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            if name not in self._prometheus_gauges:
                self._prometheus_gauges[name] = Gauge(
                    name.replace('.', '_'),
                    f'Gauge metric: {name}',
                    list(labels.keys()) if labels else []
                )
            
            if labels:
                self._prometheus_gauges[name].labels(**labels).set(value)
            else:
                self._prometheus_gauges[name].set(value)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        if not self.enabled:
            return
        
        self._histograms[name].append(MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        ))
        
        if self.prometheus_enabled and PROMETHEUS_AVAILABLE:
            if name not in self._prometheus_histograms:
                self._prometheus_histograms[name] = Histogram(
                    name.replace('.', '_'),
                    f'Histogram metric: {name}',
                    list(labels.keys()) if labels else []
                )
            
            if labels:
                self._prometheus_histograms[name].labels(**labels).observe(value)
            else:
                self._prometheus_histograms[name].observe(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        if not self.enabled:
            return {}
        
        # Calculate histogram statistics
        histogram_stats = {}
        for name, points in self._histograms.items():
            if points:
                values = [p.value for p in points]
                histogram_stats[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "recent_points": len([p for p in points if time.time() - p.timestamp < 300])  # Last 5 minutes
                }
        
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": histogram_stats,
            "timestamp": time.time()
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
    
    async def close(self):
        """Close metrics service"""
        if self.enabled:
            logger.info("Metrics service closed")


class HealthCheckService:
    """Service for health checks and system monitoring"""
    
    def __init__(self, enabled: bool = True, check_interval: int = 30):
        self.enabled = enabled
        self.check_interval = check_interval
        self._health_checks: Dict[str, HealthCheck] = {}
        self._system_stats: Dict[str, Any] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        
        if self.enabled:
            self._register_default_checks()
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def _register_default_checks(self):
        """Register default system health checks"""
        self.register_check("memory_usage", self._check_memory_usage, critical=True)
        self.register_check("cpu_usage", self._check_cpu_usage, critical=False)
        self.register_check("disk_usage", self._check_disk_usage, critical=True)
    
    def register_check(self, name: str, check_func: Callable, timeout: float = 5.0, critical: bool = True):
        """Register a health check"""
        self._health_checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            critical=critical
        )
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> bool:
        """Run a specific health check"""
        if name not in self._health_checks:
            return False
        
        check = self._health_checks[name]
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._run_check_safely(check.check_func),
                timeout=check.timeout
            )
            
            check.last_result = result
            check.last_check = datetime.utcnow()
            check.error_message = None
            
            return result
            
        except asyncio.TimeoutError:
            check.last_result = False
            check.last_check = datetime.utcnow()
            check.error_message = f"Check timed out after {check.timeout}s"
            return False
            
        except Exception as e:
            check.last_result = False
            check.last_check = datetime.utcnow()
            check.error_message = str(e)
            logger.error(f"Health check {name} failed: {e}")
            return False
    
    async def _run_check_safely(self, check_func: Callable) -> bool:
        """Run check function safely"""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        else:
            return check_func()
    
    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name in self._health_checks:
            results[name] = await self.run_check(name)
        
        return results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        if not self.enabled:
            return {"status": "disabled"}
        
        check_results = await self.run_all_checks()
        
        # Determine overall status
        critical_checks = [
            name for name, check in self._health_checks.items()
            if check.critical
        ]
        
        critical_failures = [
            name for name in critical_checks
            if not check_results.get(name, False)
        ]
        
        if critical_failures:
            overall_status = "unhealthy"
        elif any(not result for result in check_results.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "checks": {
                name: {
                    "status": "pass" if result else "fail",
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "error": check.error_message,
                    "critical": check.critical
                }
                for name, check in self._health_checks.items()
                for result in [check_results.get(name, False)]
            },
            "system_stats": self._system_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Update system stats
                self._system_stats = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                    "uptime": time.time() - psutil.boot_time()
                }
                
                # Run health checks
                await self.run_all_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    # Default health check implementations
    def _check_memory_usage(self) -> bool:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Fail if memory usage > 90%
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 95  # Fail if CPU usage > 95%
    
    def _check_disk_usage(self) -> bool:
        """Check disk usage"""
        disk = psutil.disk_usage('/')
        return disk.percent < 95  # Fail if disk usage > 95%
    
    async def close(self):
        """Close health check service"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.enabled:
            logger.info("Health check service closed")


# Decorators for automatic metrics collection
def timed(metric_name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}.duration"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                from ..config.dependencies import get_metrics_service
                metrics = get_metrics_service()
                metrics.record_histogram(name, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                from ..config.dependencies import get_metrics_service
                metrics = get_metrics_service()
                metrics.record_histogram(name, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def counted(metric_name: str = None):
    """Decorator to count function calls"""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}.calls"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            from ..config.dependencies import get_metrics_service
            metrics = get_metrics_service()
            metrics.increment_counter(name)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
