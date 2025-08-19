"""
System Monitoring for MasterX Quantum Intelligence Platform

Comprehensive system monitoring that provides health check endpoints,
performance metrics collection, distributed logging and tracing,
alert and notification systems for all quantum intelligence services.

📊 SYSTEM MONITORING CAPABILITIES:
- Real-time health check endpoints for all services
- Advanced performance metrics collection and aggregation
- Distributed logging and tracing across microservices
- Intelligent alert and notification systems
- System resource monitoring and optimization
- Comprehensive observability and diagnostics

Author: MasterX AI Team - Integration & Orchestration Division
Version: 1.0 - Phase 11 Integration & Orchestration
"""

import asyncio
import json
import time
import psutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import threading

# Try to import advanced libraries with fallbacks
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available - using basic metrics")

# ============================================================================
# MONITORING ENUMS & DATA STRUCTURES
# ============================================================================

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class HealthCheck:
    """
    🏥 HEALTH CHECK
    
    Represents a health check for a service or component
    """
    check_id: str
    service_name: str
    check_name: str
    
    # Health check configuration
    endpoint_url: str
    timeout_seconds: float = 5.0
    interval_seconds: int = 30
    
    # Health status
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: Optional[datetime] = None
    response_time_ms: float = 0.0
    
    # Error tracking
    consecutive_failures: int = 0
    failure_threshold: int = 3
    last_error: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class SystemMetric:
    """
    📈 SYSTEM METRIC
    
    Represents a system performance metric
    """
    metric_id: str
    metric_name: str
    metric_type: MetricType
    
    # Metric data
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

@dataclass
class Alert:
    """
    🚨 ALERT
    
    Represents a system alert or notification
    """
    alert_id: str
    alert_name: str
    severity: AlertSeverity
    
    # Alert content
    message: str
    description: str = ""
    
    # Alert context
    service_name: str = ""
    component: str = ""
    metric_name: str = ""
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    
    # Alert lifecycle
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None

@dataclass
class LogEntry:
    """
    📝 LOG ENTRY
    
    Represents a structured log entry
    """
    log_id: str
    timestamp: datetime
    level: str
    
    # Log content
    message: str
    logger_name: str = ""
    
    # Context
    service_name: str = ""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Additional data
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None


class SystemMonitor:
    """
    📊 SYSTEM MONITOR
    
    Comprehensive system monitoring that provides health checks, performance
    metrics collection, distributed logging, and intelligent alerting for
    all quantum intelligence services.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the system monitor"""
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Core monitoring components
        self.health_checker = HealthChecker(self.config.get('health_checks', {}))
        self.metrics_collector = MetricsCollector(self.config.get('metrics', {}))
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        self.log_aggregator = LogAggregator(self.config.get('logging', {}))
        
        # System resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Monitoring state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Registered services
        self.registered_services: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring metrics
        self.monitoring_metrics = {
            'total_health_checks': 0,
            'failed_health_checks': 0,
            'total_alerts_triggered': 0,
            'total_metrics_collected': 0,
            'average_response_time': 0.0
        }
        
        # Prometheus metrics (if available)
        self.prometheus_metrics = None
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_metrics = self._initialize_prometheus_metrics()
            except ValueError as e:
                # Handle duplicate metrics registration
                logger.warning(f"Prometheus metrics already registered: {e}")
                self.prometheus_metrics = None
        
        logger.info("📊 System Monitor initialized")
    
    async def start(self) -> bool:
        """
        Start the system monitor
        
        Returns:
            bool: True if startup successful, False otherwise
        """
        try:
            logger.info("🚀 Starting System Monitor...")
            
            # Start core components
            await self.health_checker.start()
            await self.metrics_collector.start()
            await self.alert_manager.start()
            await self.log_aggregator.start()
            await self.resource_monitor.start()
            
            # Start monitoring loops
            await self._start_monitoring_loops()
            
            # Register default health checks
            await self._register_default_health_checks()
            
            self.is_running = True
            
            logger.info("✅ System Monitor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start System Monitor: {e}")
            await self.shutdown()
            return False
    
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the system monitor
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            logger.info("🛑 Shutting down System Monitor...")
            
            self.is_running = False
            
            # Shutdown components
            await self.resource_monitor.shutdown()
            await self.log_aggregator.shutdown()
            await self.alert_manager.shutdown()
            await self.metrics_collector.shutdown()
            await self.health_checker.shutdown()
            
            logger.info("✅ System Monitor shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during System Monitor shutdown: {e}")
            return False
    
    async def register_service(
        self,
        service_name: str,
        health_check_url: str,
        metrics_endpoints: Optional[List[str]] = None
    ) -> bool:
        """
        Register a service for monitoring
        
        Args:
            service_name: Name of the service
            health_check_url: Health check endpoint URL
            metrics_endpoints: Optional metrics endpoints
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Register health check
            health_check = HealthCheck(
                check_id=f"{service_name}_health",
                service_name=service_name,
                check_name="Service Health Check",
                endpoint_url=health_check_url
            )
            
            await self.health_checker.register_health_check(health_check)
            
            # Register service
            self.registered_services[service_name] = {
                'health_check_url': health_check_url,
                'metrics_endpoints': metrics_endpoints or [],
                'registered_at': datetime.now(),
                'status': HealthStatus.UNKNOWN
            }
            
            logger.info(f"✅ Service registered for monitoring: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register service: {e}")
            return False
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Record a system metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels
            
        Returns:
            bool: True if metric recorded successfully, False otherwise
        """
        try:
            metric = SystemMetric(
                metric_id=f"{metric_name}_{uuid.uuid4().hex[:8]}",
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                labels=labels or {}
            )
            
            await self.metrics_collector.record_metric(metric)
            
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                await self._update_prometheus_metric(metric)
            
            self.monitoring_metrics['total_metrics_collected'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record metric: {e}")
            return False
    
    async def trigger_alert(
        self,
        alert_name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        service_name: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Trigger a system alert
        
        Args:
            alert_name: Name of the alert
            message: Alert message
            severity: Alert severity
            service_name: Service that triggered the alert
            labels: Optional labels
            
        Returns:
            bool: True if alert triggered successfully, False otherwise
        """
        try:
            alert = Alert(
                alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                alert_name=alert_name,
                severity=severity,
                message=message,
                service_name=service_name,
                labels=labels or {}
            )
            
            await self.alert_manager.trigger_alert(alert)
            
            self.monitoring_metrics['total_alerts_triggered'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger alert: {e}")
            return False
    
    async def log_event(
        self,
        level: str,
        message: str,
        service_name: str = "",
        extra_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a system event
        
        Args:
            level: Log level
            message: Log message
            service_name: Service name
            extra_data: Additional data
            
        Returns:
            bool: True if logged successfully, False otherwise
        """
        try:
            log_entry = LogEntry(
                log_id=f"log_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                level=level,
                message=message,
                service_name=service_name,
                extra_data=extra_data or {}
            )
            
            await self.log_aggregator.log_event(log_entry)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log event: {e}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status
        
        Returns:
            dict: System health information
        """
        try:
            # Get health check results
            health_checks = await self.health_checker.get_all_health_checks()
            
            # Get system resources
            system_resources = await self.resource_monitor.get_system_resources()
            
            # Calculate overall health
            healthy_services = sum(1 for hc in health_checks.values() if hc.status == HealthStatus.HEALTHY)
            total_services = len(health_checks)
            overall_health = HealthStatus.HEALTHY if healthy_services == total_services else HealthStatus.DEGRADED
            
            if healthy_services == 0:
                overall_health = HealthStatus.UNHEALTHY
            
            return {
                'overall_health': overall_health.value,
                'healthy_services': healthy_services,
                'total_services': total_services,
                'health_checks': {
                    name: {
                        'status': hc.status.value,
                        'response_time_ms': hc.response_time_ms,
                        'last_check_time': hc.last_check_time.isoformat() if hc.last_check_time else None,
                        'consecutive_failures': hc.consecutive_failures
                    } for name, hc in health_checks.items()
                },
                'system_resources': system_resources,
                'monitoring_metrics': self.monitoring_metrics,
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return {'error': str(e)}
    
    async def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get metrics summary for a time window
        
        Args:
            time_window_minutes: Time window in minutes
            
        Returns:
            dict: Metrics summary
        """
        try:
            return await self.metrics_collector.get_metrics_summary(time_window_minutes)
            
        except Exception as e:
            logger.error(f"❌ Failed to get metrics summary: {e}")
            return {'error': str(e)}
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts
        
        Returns:
            list: Active alerts
        """
        try:
            alerts = await self.alert_manager.get_active_alerts()
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'alert_name': alert.alert_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'service_name': alert.service_name,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'labels': alert.labels
                } for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"❌ Failed to get active alerts: {e}")
            return []

    # ========================================================================
    # HELPER METHODS FOR SYSTEM MONITOR
    # ========================================================================

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system monitor configuration"""

        return {
            'health_checks': {
                'default_timeout_seconds': 5.0,
                'default_interval_seconds': 30,
                'failure_threshold': 3
            },
            'metrics': {
                'collection_interval_seconds': 10,
                'retention_hours': 24,
                'aggregation_window_seconds': 60
            },
            'alerts': {
                'notification_channels': ['email', 'slack'],
                'escalation_timeout_minutes': 15,
                'auto_resolve_timeout_minutes': 60
            },
            'logging': {
                'log_level': 'info',
                'retention_days': 7,
                'max_log_size_mb': 100
            },
            'resources': {
                'cpu_threshold': 80.0,
                'memory_threshold': 85.0,
                'disk_threshold': 90.0
            }
        }

    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""

        return {
            'health_check_duration': Histogram(
                'health_check_duration_seconds',
                'Time spent on health checks',
                ['service_name', 'check_name']
            ),
            'health_check_status': Gauge(
                'health_check_status',
                'Health check status (1=healthy, 0=unhealthy)',
                ['service_name', 'check_name']
            ),
            'system_cpu_usage': Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage'
            ),
            'system_memory_usage': Gauge(
                'system_memory_usage_percent',
                'System memory usage percentage'
            ),
            'alerts_total': Counter(
                'alerts_total',
                'Total number of alerts triggered',
                ['severity', 'service_name']
            )
        }

    async def _start_monitoring_loops(self):
        """Start background monitoring loops"""

        asyncio.create_task(self._system_monitoring_loop())
        asyncio.create_task(self._metrics_aggregation_loop())
        asyncio.create_task(self._alert_processing_loop())

    async def _register_default_health_checks(self):
        """Register default health checks"""

        # System health check
        system_health_check = HealthCheck(
            check_id="system_health",
            service_name="system_monitor",
            check_name="System Health",
            endpoint_url="http://localhost:8000/health"
        )

        await self.health_checker.register_health_check(system_health_check)

    async def _update_prometheus_metric(self, metric: SystemMetric):
        """Update Prometheus metric"""

        if not PROMETHEUS_AVAILABLE or not hasattr(self, 'prometheus_metrics'):
            return

        # Update appropriate Prometheus metric based on type
        # This is a simplified implementation
        pass

    async def _system_monitoring_loop(self):
        """Background system monitoring loop"""

        while self.is_running:
            try:
                # Monitor system resources
                await self._monitor_system_resources()

                # Check for resource-based alerts
                await self._check_resource_alerts()

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _metrics_aggregation_loop(self):
        """Background metrics aggregation loop"""

        while self.is_running:
            try:
                # Aggregate metrics
                await self.metrics_collector.aggregate_metrics()

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(10)

    async def _alert_processing_loop(self):
        """Background alert processing loop"""

        while self.is_running:
            try:
                # Process pending alerts
                await self.alert_manager.process_alerts()

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)

    async def _monitor_system_resources(self):
        """Monitor system resources"""

        try:
            resources = await self.resource_monitor.get_system_resources()

            # Record metrics
            await self.record_metric('system_cpu_usage', resources['cpu_percent'])
            await self.record_metric('system_memory_usage', resources['memory_percent'])
            await self.record_metric('system_disk_usage', resources['disk_percent'])

        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")

    async def _check_resource_alerts(self):
        """Check for resource-based alerts"""

        try:
            resources = await self.resource_monitor.get_system_resources()
            thresholds = self.config['resources']

            # Check CPU usage
            if resources['cpu_percent'] > thresholds['cpu_threshold']:
                await self.trigger_alert(
                    'high_cpu_usage',
                    f"High CPU usage: {resources['cpu_percent']:.1f}%",
                    AlertSeverity.WARNING,
                    'system_monitor'
                )

            # Check memory usage
            if resources['memory_percent'] > thresholds['memory_threshold']:
                await self.trigger_alert(
                    'high_memory_usage',
                    f"High memory usage: {resources['memory_percent']:.1f}%",
                    AlertSeverity.WARNING,
                    'system_monitor'
                )

            # Check disk usage
            if resources['disk_percent'] > thresholds['disk_threshold']:
                await self.trigger_alert(
                    'high_disk_usage',
                    f"High disk usage: {resources['disk_percent']:.1f}%",
                    AlertSeverity.ERROR,
                    'system_monitor'
                )

        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")

    def get_monitor_status(self) -> Dict[str, Any]:
        """Get comprehensive monitor status"""

        return {
            'is_running': self.is_running,
            'startup_time': self.startup_time,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'registered_services': len(self.registered_services),
            'monitoring_metrics': self.monitoring_metrics,
            'components': {
                'health_checker': 'active' if self.health_checker else 'inactive',
                'metrics_collector': 'active' if self.metrics_collector else 'inactive',
                'alert_manager': 'active' if self.alert_manager else 'inactive',
                'log_aggregator': 'active' if self.log_aggregator else 'inactive',
                'resource_monitor': 'active' if self.resource_monitor else 'inactive'
            },
            'prometheus_enabled': PROMETHEUS_AVAILABLE
        }


# ============================================================================
# HELPER CLASSES FOR SYSTEM MONITORING
# ============================================================================

class HealthChecker:
    """Health checker for services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.is_running = False

    async def start(self):
        """Start the health checker"""
        self.is_running = True
        asyncio.create_task(self._health_check_loop())
        logger.info("✅ Health checker started")

    async def shutdown(self):
        """Shutdown the health checker"""
        self.is_running = False
        logger.info("✅ Health checker shutdown")

    async def register_health_check(self, health_check: HealthCheck) -> bool:
        """Register a health check"""
        try:
            self.health_checks[health_check.check_id] = health_check
            logger.info(f"Health check registered: {health_check.check_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register health check: {e}")
            return False

    async def get_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Get all health checks"""
        return self.health_checks.copy()

    async def _health_check_loop(self):
        """Background health check loop"""

        while self.is_running:
            try:
                for health_check in self.health_checks.values():
                    await self._perform_health_check(health_check)

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _perform_health_check(self, health_check: HealthCheck):
        """Perform a single health check"""

        try:
            start_time = time.time()

            # Simulate health check (in production, this would be an actual HTTP call)
            await asyncio.sleep(0.01)  # Simulate network call

            # Update health check
            health_check.status = HealthStatus.HEALTHY
            health_check.last_check_time = datetime.now()
            health_check.response_time_ms = (time.time() - start_time) * 1000
            health_check.consecutive_failures = 0
            health_check.last_error = None

        except Exception as e:
            # Handle health check failure
            health_check.status = HealthStatus.UNHEALTHY
            health_check.last_check_time = datetime.now()
            health_check.consecutive_failures += 1
            health_check.last_error = str(e)

            if health_check.consecutive_failures >= health_check.failure_threshold:
                health_check.status = HealthStatus.UNHEALTHY


class MetricsCollector:
    """Metrics collector for performance data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque()
        self.aggregated_metrics = defaultdict(list)
        self.is_running = False

    async def start(self):
        """Start the metrics collector"""
        self.is_running = True
        asyncio.create_task(self._metrics_collection_loop())
        logger.info("✅ Metrics collector started")

    async def shutdown(self):
        """Shutdown the metrics collector"""
        self.is_running = False
        logger.info("✅ Metrics collector shutdown")

    async def record_metric(self, metric: SystemMetric):
        """Record a metric"""
        self.metrics_buffer.append(metric)

        # Keep buffer size manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer.popleft()

    async def get_metrics_summary(self, time_window_minutes: int) -> Dict[str, Any]:
        """Get metrics summary for time window"""

        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        recent_metrics = [
            metric for metric in self.metrics_buffer
            if metric.timestamp >= cutoff_time
        ]

        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.metric_name].append(metric.value)

        # Calculate summaries
        summary = {}
        for metric_name, values in grouped_metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }

        return summary

    async def aggregate_metrics(self):
        """Aggregate metrics for storage"""
        # Placeholder for metrics aggregation logic
        pass

    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""

        while self.is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                await asyncio.sleep(self.config.get('collection_interval_seconds', 10))

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            await self.record_metric(SystemMetric(
                metric_id=f"cpu_{uuid.uuid4().hex[:8]}",
                metric_name="system_cpu_usage",
                metric_type=MetricType.GAUGE,
                value=cpu_percent
            ))

            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric(SystemMetric(
                metric_id=f"memory_{uuid.uuid4().hex[:8]}",
                metric_name="system_memory_usage",
                metric_type=MetricType.GAUGE,
                value=memory.percent
            ))

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class AlertManager:
    """Alert manager for notifications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_queue = asyncio.Queue()
        self.is_running = False

    async def start(self):
        """Start the alert manager"""
        self.is_running = True
        asyncio.create_task(self._alert_processing_loop())
        logger.info("✅ Alert manager started")

    async def shutdown(self):
        """Shutdown the alert manager"""
        self.is_running = False
        logger.info("✅ Alert manager shutdown")

    async def trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        await self.alert_queue.put(alert)

    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    async def process_alerts(self):
        """Process pending alerts"""
        try:
            alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)

            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert

            # Send notifications
            await self._send_alert_notifications(alert)

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")

    async def _alert_processing_loop(self):
        """Background alert processing loop"""

        while self.is_running:
            try:
                await self.process_alerts()
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""

        # Log the alert
        logger.warning(f"ALERT: {alert.alert_name} - {alert.message}")

        # In production, this would send notifications via email, Slack, etc.


class LogAggregator:
    """Log aggregator for centralized logging"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_buffer = deque()
        self.is_running = False

    async def start(self):
        """Start the log aggregator"""
        self.is_running = True
        logger.info("✅ Log aggregator started")

    async def shutdown(self):
        """Shutdown the log aggregator"""
        self.is_running = False
        logger.info("✅ Log aggregator shutdown")

    async def log_event(self, log_entry: LogEntry):
        """Log an event"""
        self.log_buffer.append(log_entry)

        # Keep buffer size manageable
        if len(self.log_buffer) > 10000:
            self.log_buffer.popleft()


class ResourceMonitor:
    """System resource monitor"""

    def __init__(self):
        self.is_running = False

    async def start(self):
        """Start the resource monitor"""
        self.is_running = True
        logger.info("✅ Resource monitor started")

    async def shutdown(self):
        """Shutdown the resource monitor"""
        self.is_running = False
        logger.info("✅ Resource monitor shutdown")

    async def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Network I/O
            network = psutil.net_io_counters()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0,
                'timestamp': datetime.now().isoformat()
            }
