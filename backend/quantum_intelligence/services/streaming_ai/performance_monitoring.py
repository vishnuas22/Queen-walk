"""
Real-Time Performance Monitoring

Advanced monitoring system for tracking and analyzing streaming AI performance
with sub-millisecond precision, anomaly detection, and predictive scaling.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import random

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Provide fallback functions
    class np:
        @staticmethod
        def random():
            class RandomModule:
                @staticmethod
                def uniform(low, high):
                    return random.uniform(low, high)
                @staticmethod
                def randint(low, high):
                    return random.randint(low, high)
                @staticmethod
                def choice(choices):
                    return random.choice(choices)
            return RandomModule()

        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0

        @staticmethod
        def var(values):
            if not values:
                return 0
            mean_val = sum(values) / len(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)

        @staticmethod
        def std(values):
            return (np.var(values)) ** 0.5

        @staticmethod
        def polyfit(x, y, deg):
            # Very simple linear regression for deg=1
            if deg != 1 or not x or not y:
                return [0, 0]

            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_xx = sum(x[i] * x[i] for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            return [slope, intercept]

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .data_structures import (
    StreamingMetrics, NetworkCondition, PerformanceAlert, AlertSeverity
)


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    PACKET_LOSS = "packet_loss"
    JITTER = "jitter"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_QUALITY = "network_quality"
    USER_ENGAGEMENT = "user_engagement"
    CONTENT_DELIVERY = "content_delivery"
    ADAPTIVE_ADJUSTMENTS = "adaptive_adjustments"


class AlertType(Enum):
    """Types of performance alerts"""
    HIGH_LATENCY = "high_latency"
    LOW_THROUGHPUT = "low_throughput"
    EXCESSIVE_PACKET_LOSS = "excessive_packet_loss"
    HIGH_JITTER = "high_jitter"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_DEGRADATION = "network_degradation"
    USER_DISENGAGEMENT = "user_disengagement"
    CONTENT_DELIVERY_FAILURE = "content_delivery_failure"
    SYSTEM_OVERLOAD = "system_overload"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class MetricThresholds:
    """Thresholds for performance metrics"""
    warning_thresholds: Dict[MetricType, float]
    critical_thresholds: Dict[MetricType, float]
    recovery_thresholds: Dict[MetricType, float]


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    session_id: str
    timestamp: datetime
    metrics: StreamingMetrics
    network_conditions: Dict[str, NetworkCondition]
    active_users: int
    active_features: List[str]
    system_load: Dict[str, float]
    anomaly_scores: Dict[str, float]


class RealTimePerformanceMonitor:
    """
    📊 REAL-TIME PERFORMANCE MONITOR
    
    Advanced monitoring system for tracking and analyzing streaming AI performance
    with sub-millisecond precision, anomaly detection, and predictive scaling.
    """
    
    def __init__(self, session_id: str, cache_service: Optional[CacheService] = None):
        self.session_id = session_id
        self.cache = cache_service
        
        # Performance tracking
        self.metrics_history: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=1000) for metric_type in MetricType
        }
        self.performance_snapshots: deque = deque(maxlen=100)
        self.alerts_history: deque = deque(maxlen=100)
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 1.0  # seconds
        self.last_snapshot_time = datetime.now()
        
        # Metric thresholds
        self.metric_thresholds = MetricThresholds(
            warning_thresholds={
                MetricType.LATENCY: 150.0,  # ms
                MetricType.THROUGHPUT: 200.0,  # kbps
                MetricType.PACKET_LOSS: 0.02,  # 2%
                MetricType.JITTER: 50.0,  # ms
                MetricType.CPU_USAGE: 0.7,  # 70%
                MetricType.MEMORY_USAGE: 0.7,  # 70%
                MetricType.NETWORK_QUALITY: 0.6,  # score
                MetricType.USER_ENGAGEMENT: 0.5,  # score
                MetricType.CONTENT_DELIVERY: 0.8,  # success rate
                MetricType.ADAPTIVE_ADJUSTMENTS: 10  # count
            },
            critical_thresholds={
                MetricType.LATENCY: 250.0,  # ms
                MetricType.THROUGHPUT: 100.0,  # kbps
                MetricType.PACKET_LOSS: 0.05,  # 5%
                MetricType.JITTER: 100.0,  # ms
                MetricType.CPU_USAGE: 0.9,  # 90%
                MetricType.MEMORY_USAGE: 0.9,  # 90%
                MetricType.NETWORK_QUALITY: 0.4,  # score
                MetricType.USER_ENGAGEMENT: 0.3,  # score
                MetricType.CONTENT_DELIVERY: 0.6,  # success rate
                MetricType.ADAPTIVE_ADJUSTMENTS: 20  # count
            },
            recovery_thresholds={
                MetricType.LATENCY: 120.0,  # ms
                MetricType.THROUGHPUT: 250.0,  # kbps
                MetricType.PACKET_LOSS: 0.01,  # 1%
                MetricType.JITTER: 30.0,  # ms
                MetricType.CPU_USAGE: 0.6,  # 60%
                MetricType.MEMORY_USAGE: 0.6,  # 60%
                MetricType.NETWORK_QUALITY: 0.7,  # score
                MetricType.USER_ENGAGEMENT: 0.6,  # score
                MetricType.CONTENT_DELIVERY: 0.9,  # success rate
                MetricType.ADAPTIVE_ADJUSTMENTS: 5  # count
            }
        )
        
        # Current metrics
        self.current_metrics = StreamingMetrics(
            latency_ms=0.0,
            throughput_kbps=0.0,
            packet_loss_rate=0.0,
            jitter_ms=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            network_quality_score=1.0,
            user_engagement_score=0.8,
            content_delivery_success_rate=1.0,
            adaptive_adjustments_count=0
        )
        
        # Network conditions by user
        self.network_conditions: Dict[str, NetworkCondition] = {}
        
        logger.info(f"Real-Time Performance Monitor initialized for session {session_id}")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started performance monitoring for session {self.session_id}")
    
    async def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info(f"Stopped performance monitoring for session {self.session_id}")
    
    async def update_metrics(self, metrics: StreamingMetrics):
        """Update current performance metrics"""
        self.current_metrics = metrics
        
        # Store in history
        self.metrics_history[MetricType.LATENCY].append((datetime.now(), metrics.latency_ms))
        self.metrics_history[MetricType.THROUGHPUT].append((datetime.now(), metrics.throughput_kbps))
        self.metrics_history[MetricType.PACKET_LOSS].append((datetime.now(), metrics.packet_loss_rate))
        self.metrics_history[MetricType.JITTER].append((datetime.now(), metrics.jitter_ms))
        self.metrics_history[MetricType.CPU_USAGE].append((datetime.now(), metrics.cpu_usage))
        self.metrics_history[MetricType.MEMORY_USAGE].append((datetime.now(), metrics.memory_usage))
        self.metrics_history[MetricType.NETWORK_QUALITY].append((datetime.now(), metrics.network_quality_score))
        self.metrics_history[MetricType.USER_ENGAGEMENT].append((datetime.now(), metrics.user_engagement_score))
        self.metrics_history[MetricType.CONTENT_DELIVERY].append((datetime.now(), metrics.content_delivery_success_rate))
        self.metrics_history[MetricType.ADAPTIVE_ADJUSTMENTS].append((datetime.now(), metrics.adaptive_adjustments_count))
        
        # Check for alerts
        await self._check_metric_alerts()
    
    async def update_network_conditions(self, user_id: str, conditions: NetworkCondition):
        """Update network conditions for a user"""
        self.network_conditions[user_id] = conditions
    
    def get_current_metrics(self) -> StreamingMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Calculate metric trends
            trends = await self._calculate_metric_trends()
            
            # Generate anomaly analysis
            anomalies = await self._detect_anomalies()
            
            # Generate performance predictions
            predictions = await self._predict_future_performance()
            
            # Create performance report
            report = {
                'session_id': self.session_id,
                'report_timestamp': datetime.now().isoformat(),
                'current_metrics': self.current_metrics.__dict__,
                'metric_trends': trends,
                'active_alerts': [alert.__dict__ for alert in self.active_alerts.values()],
                'anomaly_analysis': anomalies,
                'performance_predictions': predictions,
                'network_conditions_summary': await self._summarize_network_conditions(),
                'optimization_recommendations': await self._generate_optimization_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                'session_id': self.session_id,
                'error': str(e),
                'report_timestamp': datetime.now().isoformat()
            }
    
    async def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Take performance snapshot
                await self._take_performance_snapshot()
                
                # Analyze current performance
                await self._analyze_current_performance()
                
                # Check for alert recovery
                await self._check_alert_recovery()
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)  # Wait longer on error
    
    async def _take_performance_snapshot(self):
        """Take a point-in-time performance snapshot"""
        snapshot = PerformanceSnapshot(
            session_id=self.session_id,
            timestamp=datetime.now(),
            metrics=self.current_metrics,
            network_conditions=self.network_conditions.copy(),
            active_users=len(self.network_conditions),
            active_features=[],  # Would be populated in production
            system_load={
                'cpu': self.current_metrics.cpu_usage,
                'memory': self.current_metrics.memory_usage,
                'network': 1.0 - self.current_metrics.network_quality_score
            },
            anomaly_scores={}  # Would be populated in production
        )
        
        self.performance_snapshots.append(snapshot)
        self.last_snapshot_time = snapshot.timestamp
    
    async def _analyze_current_performance(self):
        """Analyze current performance state"""
        # This would contain more sophisticated analysis in production
        pass
    
    async def _check_metric_alerts(self):
        """Check metrics against thresholds for alerts"""
        # Check latency
        await self._check_metric_threshold(
            MetricType.LATENCY,
            self.current_metrics.latency_ms,
            AlertType.HIGH_LATENCY,
            "High latency detected",
            lambda val, threshold: val > threshold
        )
        
        # Check throughput
        await self._check_metric_threshold(
            MetricType.THROUGHPUT,
            self.current_metrics.throughput_kbps,
            AlertType.LOW_THROUGHPUT,
            "Low throughput detected",
            lambda val, threshold: val < threshold
        )
        
        # Check packet loss
        await self._check_metric_threshold(
            MetricType.PACKET_LOSS,
            self.current_metrics.packet_loss_rate,
            AlertType.EXCESSIVE_PACKET_LOSS,
            "Excessive packet loss detected",
            lambda val, threshold: val > threshold
        )
        
        # Check network quality
        await self._check_metric_threshold(
            MetricType.NETWORK_QUALITY,
            self.current_metrics.network_quality_score,
            AlertType.NETWORK_DEGRADATION,
            "Network quality degradation detected",
            lambda val, threshold: val < threshold
        )
        
        # Check user engagement
        await self._check_metric_threshold(
            MetricType.USER_ENGAGEMENT,
            self.current_metrics.user_engagement_score,
            AlertType.USER_DISENGAGEMENT,
            "User disengagement detected",
            lambda val, threshold: val < threshold
        )
    
    async def _check_metric_threshold(self,
                                    metric_type: MetricType,
                                    value: float,
                                    alert_type: AlertType,
                                    message: str,
                                    comparison_func):
        """Check a metric against thresholds"""
        warning_threshold = self.metric_thresholds.warning_thresholds[metric_type]
        critical_threshold = self.metric_thresholds.critical_thresholds[metric_type]
        
        alert_id = f"{alert_type.value}_{self.session_id}"
        
        # Check for critical threshold
        if comparison_func(value, critical_threshold):
            if alert_id not in self.active_alerts or self.active_alerts[alert_id].severity != AlertSeverity.CRITICAL:
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    session_id=self.session_id,
                    alert_type=alert_type.value,
                    severity=AlertSeverity.CRITICAL,
                    message=f"CRITICAL: {message}",
                    metric_type=metric_type.value,
                    current_value=value,
                    threshold_value=critical_threshold,
                    first_detected=datetime.now(),
                    last_updated=datetime.now(),
                    active=True,
                    related_metrics={}
                )
                self.active_alerts[alert_id] = alert
                self.alerts_history.append(alert)
                logger.warning(f"CRITICAL ALERT: {message} - {value} vs threshold {critical_threshold}")
            else:
                self.active_alerts[alert_id].last_updated = datetime.now()
                self.active_alerts[alert_id].current_value = value
        
        # Check for warning threshold
        elif comparison_func(value, warning_threshold):
            if alert_id not in self.active_alerts:
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    session_id=self.session_id,
                    alert_type=alert_type.value,
                    severity=AlertSeverity.WARNING,
                    message=f"WARNING: {message}",
                    metric_type=metric_type.value,
                    current_value=value,
                    threshold_value=warning_threshold,
                    first_detected=datetime.now(),
                    last_updated=datetime.now(),
                    active=True,
                    related_metrics={}
                )
                self.active_alerts[alert_id] = alert
                self.alerts_history.append(alert)
                logger.warning(f"WARNING ALERT: {message} - {value} vs threshold {warning_threshold}")
            elif self.active_alerts[alert_id].severity != AlertSeverity.CRITICAL:
                self.active_alerts[alert_id].last_updated = datetime.now()
                self.active_alerts[alert_id].current_value = value
    
    async def _check_alert_recovery(self):
        """Check if any active alerts have recovered"""
        alerts_to_recover = []
        
        for alert_id, alert in self.active_alerts.items():
            metric_type = MetricType(alert.metric_type)
            recovery_threshold = self.metric_thresholds.recovery_thresholds[metric_type]
            
            # Get current value
            current_value = getattr(self.current_metrics, self._get_metric_attribute(metric_type))
            
            # Check if recovered
            if alert.alert_type == AlertType.LOW_THROUGHPUT.value:
                recovered = current_value > recovery_threshold
            else:
                recovered = current_value < recovery_threshold if alert.alert_type in [
                    AlertType.HIGH_LATENCY.value,
                    AlertType.EXCESSIVE_PACKET_LOSS.value,
                    AlertType.HIGH_JITTER.value
                ] else current_value > recovery_threshold
            
            if recovered:
                alerts_to_recover.append(alert_id)
                logger.info(f"Alert recovered: {alert.alert_type} - {current_value} vs recovery threshold {recovery_threshold}")
        
        # Mark alerts as recovered
        for alert_id in alerts_to_recover:
            self.active_alerts[alert_id].active = False
            self.active_alerts[alert_id].last_updated = datetime.now()
            self.active_alerts[alert_id].message += " (RECOVERED)"
            del self.active_alerts[alert_id]
    
    def _get_metric_attribute(self, metric_type: MetricType) -> str:
        """Get attribute name for metric type"""
        attribute_map = {
            MetricType.LATENCY: "latency_ms",
            MetricType.THROUGHPUT: "throughput_kbps",
            MetricType.PACKET_LOSS: "packet_loss_rate",
            MetricType.JITTER: "jitter_ms",
            MetricType.CPU_USAGE: "cpu_usage",
            MetricType.MEMORY_USAGE: "memory_usage",
            MetricType.NETWORK_QUALITY: "network_quality_score",
            MetricType.USER_ENGAGEMENT: "user_engagement_score",
            MetricType.CONTENT_DELIVERY: "content_delivery_success_rate",
            MetricType.ADAPTIVE_ADJUSTMENTS: "adaptive_adjustments_count"
        }
        return attribute_map.get(metric_type, "")
    
    async def _calculate_metric_trends(self) -> Dict[str, Any]:
        """Calculate trends for all metrics"""
        trends = {}
        
        for metric_type in MetricType:
            history = self.metrics_history[metric_type]
            if len(history) < 2:
                trends[metric_type.value] = "stable"
                continue
            
            # Get recent values
            recent_values = [value for _, value in list(history)[-10:]]
            
            if len(recent_values) < 2:
                trends[metric_type.value] = "stable"
                continue
            
            # Calculate trend
            first_half = np.mean(recent_values[:len(recent_values)//2])
            second_half = np.mean(recent_values[len(recent_values)//2:])
            
            if metric_type in [MetricType.LATENCY, MetricType.PACKET_LOSS, MetricType.JITTER]:
                # For these metrics, lower is better
                if second_half < first_half * 0.9:
                    trends[metric_type.value] = "improving"
                elif second_half > first_half * 1.1:
                    trends[metric_type.value] = "degrading"
                else:
                    trends[metric_type.value] = "stable"
            else:
                # For these metrics, higher is better
                if second_half > first_half * 1.1:
                    trends[metric_type.value] = "improving"
                elif second_half < first_half * 0.9:
                    trends[metric_type.value] = "degrading"
                else:
                    trends[metric_type.value] = "stable"
        
        return trends
    
    async def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in performance metrics"""
        # Simplified anomaly detection
        anomalies = {}
        
        for metric_type in MetricType:
            history = self.metrics_history[metric_type]
            if len(history) < 10:
                continue
            
            # Get recent values
            recent_values = [value for _, value in list(history)[-20:]]
            
            # Calculate mean and standard deviation
            mean = np.mean(recent_values)
            std = np.std(recent_values)
            
            # Get current value
            current_value = getattr(self.current_metrics, self._get_metric_attribute(metric_type))
            
            # Check for anomaly (z-score > 2)
            if std > 0 and abs(current_value - mean) > 2 * std:
                anomalies[metric_type.value] = {
                    'current_value': current_value,
                    'mean': mean,
                    'std': std,
                    'z_score': (current_value - mean) / std,
                    'is_anomaly': True
                }
        
        return {
            'detected_anomalies': len(anomalies),
            'anomaly_details': anomalies
        }
    
    async def _predict_future_performance(self) -> Dict[str, Any]:
        """Predict future performance based on trends"""
        # Simplified prediction
        predictions = {}
        
        for metric_type in MetricType:
            history = self.metrics_history[metric_type]
            if len(history) < 10:
                continue
            
            # Get recent values with timestamps
            recent_history = list(history)[-20:]
            timestamps = [(ts - recent_history[0][0]).total_seconds() for ts, _ in recent_history]
            values = [value for _, value in recent_history]
            
            if len(timestamps) < 2:
                continue
            
            # Simple linear regression
            try:
                slope, intercept = np.polyfit(timestamps, values, 1)
                
                # Predict value in 5 minutes
                future_time = 300  # 5 minutes in seconds
                predicted_value = slope * (timestamps[-1] + future_time) + intercept
                
                # Ensure prediction is within reasonable bounds
                if metric_type in [MetricType.NETWORK_QUALITY, MetricType.USER_ENGAGEMENT, MetricType.CONTENT_DELIVERY]:
                    predicted_value = max(0.0, min(1.0, predicted_value))
                elif metric_type in [MetricType.CPU_USAGE, MetricType.MEMORY_USAGE]:
                    predicted_value = max(0.0, min(1.0, predicted_value))
                elif metric_type in [MetricType.LATENCY, MetricType.THROUGHPUT, MetricType.JITTER]:
                    predicted_value = max(0.0, predicted_value)
                
                predictions[metric_type.value] = {
                    'current_value': values[-1],
                    'predicted_value': predicted_value,
                    'prediction_time': '5 minutes',
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'trend_strength': abs(slope)
                }
            except Exception as e:
                logger.error(f"Error predicting {metric_type.value}: {e}")
        
        return predictions
    
    async def _summarize_network_conditions(self) -> Dict[str, Any]:
        """Summarize network conditions across users"""
        if not self.network_conditions:
            return {
                'average_bandwidth': 0,
                'average_latency': 0,
                'average_packet_loss': 0,
                'average_stability': 0,
                'user_count': 0
            }
        
        # Calculate averages
        avg_bandwidth = np.mean([nc.bandwidth_kbps for nc in self.network_conditions.values()])
        avg_latency = np.mean([nc.latency_ms for nc in self.network_conditions.values()])
        avg_packet_loss = np.mean([nc.packet_loss_rate for nc in self.network_conditions.values()])
        avg_stability = np.mean([nc.connection_stability for nc in self.network_conditions.values()])
        
        return {
            'average_bandwidth': avg_bandwidth,
            'average_latency': avg_latency,
            'average_packet_loss': avg_packet_loss,
            'average_stability': avg_stability,
            'user_count': len(self.network_conditions),
            'quality_distribution': self._calculate_quality_distribution()
        }
    
    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """Calculate distribution of stream quality across users"""
        quality_counts = defaultdict(int)
        
        for nc in self.network_conditions.values():
            quality_counts[nc.optimal_quality.value] += 1
        
        return dict(quality_counts)
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check latency
        if self.current_metrics.latency_ms > self.metric_thresholds.warning_thresholds[MetricType.LATENCY]:
            recommendations.append("Reduce content complexity to improve latency")
        
        # Check throughput
        if self.current_metrics.throughput_kbps < self.metric_thresholds.warning_thresholds[MetricType.THROUGHPUT]:
            recommendations.append("Optimize bandwidth usage with adaptive content")
        
        # Check packet loss
        if self.current_metrics.packet_loss_rate > self.metric_thresholds.warning_thresholds[MetricType.PACKET_LOSS]:
            recommendations.append("Implement packet loss recovery mechanisms")
        
        # Check user engagement
        if self.current_metrics.user_engagement_score < self.metric_thresholds.warning_thresholds[MetricType.USER_ENGAGEMENT]:
            recommendations.append("Enhance interactive elements to improve engagement")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Current performance is within optimal parameters")
        
        return recommendations
