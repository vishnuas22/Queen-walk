"""
Stream Quality Optimizer

AI-powered system for optimizing streaming quality based on network conditions,
device capabilities, and learning context requirements with adaptive quality scaling.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
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

# Try to import ML libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .data_structures import (
    StreamQuality, NetworkCondition, StreamingMetrics, ContentAdaptation
)


class ContentType(Enum):
    """Types of streaming content"""
    TEXT_ONLY = "text_only"
    BASIC_INTERACTIVE = "basic_interactive"
    RICH_MEDIA = "rich_media"
    REAL_TIME_COLLABORATION = "real_time_collaboration"
    VIDEO_CONTENT = "video_content"
    AUDIO_CONTENT = "audio_content"


class OptimizationStrategy(Enum):
    """Stream optimization strategies"""
    QUALITY_FIRST = "quality_first"
    LATENCY_FIRST = "latency_first"
    BANDWIDTH_EFFICIENT = "bandwidth_efficient"
    ADAPTIVE_BALANCED = "adaptive_balanced"


@dataclass
class QualityProfile:
    """Quality profile for user/session"""
    user_id: str
    session_id: str
    preferred_quality: StreamQuality
    quality_tolerance: float
    latency_sensitivity: float
    bandwidth_constraints: Dict[str, float]
    device_capabilities: Dict[str, Any]
    historical_performance: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class QualityAdjustment:
    """Quality adjustment recommendation"""
    adjustment_id: str
    user_id: str
    session_id: str
    current_quality: StreamQuality
    recommended_quality: StreamQuality
    adjustment_reason: str
    expected_improvement: Dict[str, float]
    implementation_priority: int
    bandwidth_impact: float
    latency_impact: float
    learning_impact_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class StreamQualityOptimizer:
    """
    📊 STREAM QUALITY OPTIMIZER
    
    AI-powered system for optimizing streaming quality based on network
    conditions, device capabilities, and learning context requirements.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Quality management
        self.quality_profiles: Dict[str, QualityProfile] = {}
        self.network_monitoring: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.quality_adjustments_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.performance_metrics: Dict[str, StreamingMetrics] = {}
        
        # ML models (if available)
        if SKLEARN_AVAILABLE:
            self.quality_predictor = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                random_state=42
            )
        else:
            self.quality_predictor = None
        
        # Quality requirements for different content types
        self.content_quality_requirements = {
            ContentType.TEXT_ONLY: {
                'min_bandwidth_kbps': 64,
                'latency_tolerance_ms': 200,
                'quality_importance': 0.3
            },
            ContentType.BASIC_INTERACTIVE: {
                'min_bandwidth_kbps': 128,
                'latency_tolerance_ms': 150,
                'quality_importance': 0.6
            },
            ContentType.RICH_MEDIA: {
                'min_bandwidth_kbps': 256,
                'latency_tolerance_ms': 100,
                'quality_importance': 0.8
            },
            ContentType.REAL_TIME_COLLABORATION: {
                'min_bandwidth_kbps': 512,
                'latency_tolerance_ms': 50,
                'quality_importance': 0.9
            },
            ContentType.VIDEO_CONTENT: {
                'min_bandwidth_kbps': 1024,
                'latency_tolerance_ms': 75,
                'quality_importance': 0.9
            },
            ContentType.AUDIO_CONTENT: {
                'min_bandwidth_kbps': 128,
                'latency_tolerance_ms': 100,
                'quality_importance': 0.7
            }
        }
        
        # Quality level specifications
        self.quality_specifications = {
            StreamQuality.ULTRA_LOW: {
                'bandwidth_kbps': 64,
                'resolution': '240p',
                'features': ['text', 'basic_audio']
            },
            StreamQuality.LOW: {
                'bandwidth_kbps': 128,
                'resolution': '360p',
                'features': ['text', 'audio', 'basic_video']
            },
            StreamQuality.MEDIUM: {
                'bandwidth_kbps': 256,
                'resolution': '480p',
                'features': ['text', 'audio', 'video', 'basic_interactive']
            },
            StreamQuality.HIGH: {
                'bandwidth_kbps': 512,
                'resolution': '720p',
                'features': ['text', 'audio', 'video', 'interactive', 'collaboration']
            },
            StreamQuality.ULTRA_HIGH: {
                'bandwidth_kbps': 1024,
                'resolution': '1080p',
                'features': ['text', 'audio', 'video', 'interactive', 'collaboration', 'advanced']
            }
        }
        
        logger.info("Stream Quality Optimizer initialized")
    
    async def optimize_stream_quality(self,
                                    user_id: str,
                                    session_id: str,
                                    network_conditions: NetworkCondition,
                                    content_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize stream quality based on current conditions
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            network_conditions: Current network conditions
            content_context: Content context and requirements
            
        Returns:
            Dict: Quality optimization result
        """
        try:
            # Get or create quality profile
            profile = await self._get_quality_profile(user_id, session_id)
            
            # Analyze network performance
            network_analysis = await self._analyze_network_performance(network_conditions)
            
            # Determine content requirements
            content_requirements = await self._determine_content_requirements(content_context)
            
            # Calculate optimal quality settings
            optimal_quality = await self._calculate_optimal_quality(
                network_analysis, content_requirements, profile
            )
            
            # Generate quality adjustment recommendations
            quality_adjustments = await self._generate_quality_adjustments(
                profile, optimal_quality, network_analysis
            )
            
            # Update monitoring data
            self.network_monitoring[user_id].append(network_conditions)
            
            # Create optimization result
            optimization_result = {
                'user_id': user_id,
                'session_id': session_id,
                'current_quality': profile.preferred_quality.value,
                'recommended_quality': optimal_quality.value,
                'network_analysis': network_analysis,
                'content_requirements': content_requirements,
                'quality_adjustments': [adj.__dict__ for adj in quality_adjustments],
                'optimization_confidence': await self._calculate_optimization_confidence(
                    network_analysis, content_requirements
                ),
                'expected_improvements': await self._predict_quality_improvements(
                    profile.preferred_quality, optimal_quality
                ),
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Update profile
            profile.preferred_quality = optimal_quality
            profile.last_updated = datetime.now()
            
            return {
                'status': 'success',
                'optimization_result': optimization_result
            }
            
        except Exception as e:
            logger.error(f"Error optimizing stream quality: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def optimize_for_conditions(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize stream quality for changing network conditions
        
        Args:
            event: Network condition change event
            
        Returns:
            Dict: Optimization result
        """
        try:
            user_id = event.get('user_id', '')
            session_id = event.get('session_id', '')
            new_conditions = NetworkCondition(**event.get('event_data', {}))
            
            # Get current content context (simplified)
            content_context = {
                'content_type': 'rich_media',
                'interaction_level': 'high',
                'learning_criticality': 0.8
            }
            
            # Optimize for new conditions
            result = await self.optimize_stream_quality(
                user_id, session_id, new_conditions, content_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing for conditions: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_quality_profile(self, user_id: str, session_id: str) -> QualityProfile:
        """Get or create quality profile for user/session"""
        profile_key = f"{user_id}_{session_id}"
        
        if profile_key not in self.quality_profiles:
            # Create default profile
            self.quality_profiles[profile_key] = QualityProfile(
                user_id=user_id,
                session_id=session_id,
                preferred_quality=StreamQuality.MEDIUM,
                quality_tolerance=0.7,
                latency_sensitivity=0.8,
                bandwidth_constraints={'max_kbps': 1000, 'min_kbps': 128},
                device_capabilities={
                    'video_support': True,
                    'audio_support': True,
                    'interactive_support': True,
                    'max_resolution': '1080p'
                }
            )
        
        return self.quality_profiles[profile_key]
    
    async def _analyze_network_performance(self, conditions: NetworkCondition) -> Dict[str, Any]:
        """Analyze current network performance"""
        # Calculate network quality score
        bandwidth_score = min(1.0, conditions.bandwidth_kbps / 1000.0)  # Normalize to 1Mbps
        latency_score = max(0.0, 1.0 - conditions.latency_ms / 200.0)   # Normalize to 200ms
        stability_score = conditions.connection_stability
        packet_loss_score = max(0.0, 1.0 - conditions.packet_loss_rate * 100)  # Normalize packet loss
        
        overall_quality = (
            bandwidth_score * 0.4 +
            latency_score * 0.3 +
            stability_score * 0.2 +
            packet_loss_score * 0.1
        )
        
        # Determine network tier
        if overall_quality >= 0.8:
            network_tier = "excellent"
        elif overall_quality >= 0.6:
            network_tier = "good"
        elif overall_quality >= 0.4:
            network_tier = "fair"
        else:
            network_tier = "poor"
        
        return {
            'overall_quality_score': overall_quality,
            'network_tier': network_tier,
            'bandwidth_score': bandwidth_score,
            'latency_score': latency_score,
            'stability_score': stability_score,
            'packet_loss_score': packet_loss_score,
            'bottleneck_factor': self._identify_bottleneck(conditions),
            'optimization_potential': 1.0 - overall_quality
        }
    
    async def _determine_content_requirements(self, content_context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine content quality requirements"""
        content_type_str = content_context.get('content_type', 'basic_interactive')
        
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.BASIC_INTERACTIVE
        
        base_requirements = self.content_quality_requirements[content_type]
        
        # Adjust requirements based on context
        interaction_level = content_context.get('interaction_level', 'medium')
        learning_criticality = content_context.get('learning_criticality', 0.7)
        
        # Scale requirements based on interaction level
        interaction_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'critical': 1.5
        }.get(interaction_level, 1.0)
        
        adjusted_requirements = {
            'min_bandwidth_kbps': base_requirements['min_bandwidth_kbps'] * interaction_multiplier,
            'latency_tolerance_ms': base_requirements['latency_tolerance_ms'] / interaction_multiplier,
            'quality_importance': min(1.0, base_requirements['quality_importance'] * learning_criticality),
            'content_type': content_type.value,
            'interaction_level': interaction_level,
            'learning_criticality': learning_criticality
        }
        
        return adjusted_requirements
    
    async def _calculate_optimal_quality(self,
                                       network_analysis: Dict[str, Any],
                                       content_requirements: Dict[str, Any],
                                       profile: QualityProfile) -> StreamQuality:
        """Calculate optimal quality level"""
        available_bandwidth = network_analysis['bandwidth_score'] * 1000  # Convert back to kbps
        required_bandwidth = content_requirements['min_bandwidth_kbps']
        quality_importance = content_requirements['quality_importance']
        
        # Find highest quality that meets requirements
        suitable_qualities = []
        
        for quality, specs in self.quality_specifications.items():
            if specs['bandwidth_kbps'] <= available_bandwidth:
                # Check if quality meets content requirements
                if specs['bandwidth_kbps'] >= required_bandwidth * 0.8:  # 80% threshold
                    suitable_qualities.append((quality, specs))
        
        if not suitable_qualities:
            # Fallback to lowest quality
            return StreamQuality.ULTRA_LOW
        
        # Select optimal quality based on importance and user preferences
        if quality_importance > 0.8:
            # High importance: choose highest available quality
            return max(suitable_qualities, key=lambda x: x[1]['bandwidth_kbps'])[0]
        elif quality_importance < 0.4:
            # Low importance: choose most efficient quality
            return min(suitable_qualities, key=lambda x: x[1]['bandwidth_kbps'])[0]
        else:
            # Medium importance: balance quality and efficiency
            mid_index = len(suitable_qualities) // 2
            return sorted(suitable_qualities, key=lambda x: x[1]['bandwidth_kbps'])[mid_index][0]
    
    async def _generate_quality_adjustments(self,
                                          profile: QualityProfile,
                                          optimal_quality: StreamQuality,
                                          network_analysis: Dict[str, Any]) -> List[QualityAdjustment]:
        """Generate quality adjustment recommendations"""
        adjustments = []
        
        if profile.preferred_quality != optimal_quality:
            adjustment = QualityAdjustment(
                adjustment_id=f"adj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                user_id=profile.user_id,
                session_id=profile.session_id,
                current_quality=profile.preferred_quality,
                recommended_quality=optimal_quality,
                adjustment_reason=await self._determine_adjustment_reason(
                    profile.preferred_quality, optimal_quality, network_analysis
                ),
                expected_improvement=await self._calculate_expected_improvement(
                    profile.preferred_quality, optimal_quality
                ),
                implementation_priority=await self._calculate_adjustment_priority(
                    profile.preferred_quality, optimal_quality, network_analysis
                ),
                bandwidth_impact=await self._calculate_bandwidth_impact(
                    profile.preferred_quality, optimal_quality
                ),
                latency_impact=await self._calculate_latency_impact(
                    profile.preferred_quality, optimal_quality
                ),
                learning_impact_score=await self._calculate_learning_impact(
                    profile.preferred_quality, optimal_quality
                )
            )
            adjustments.append(adjustment)
        
        return adjustments
    
    def _identify_bottleneck(self, conditions: NetworkCondition) -> str:
        """Identify primary network bottleneck"""
        if conditions.bandwidth_kbps < 256:
            return "bandwidth"
        elif conditions.latency_ms > 150:
            return "latency"
        elif conditions.packet_loss_rate > 0.02:
            return "packet_loss"
        elif conditions.connection_stability < 0.8:
            return "stability"
        else:
            return "none"
    
    async def _calculate_optimization_confidence(self,
                                               network_analysis: Dict[str, Any],
                                               content_requirements: Dict[str, Any]) -> float:
        """Calculate confidence in optimization recommendation"""
        # Base confidence on network quality and stability
        network_confidence = network_analysis['overall_quality_score']
        stability_confidence = network_analysis['stability_score']
        
        # Adjust based on how well network meets requirements
        bandwidth_ratio = network_analysis['bandwidth_score'] * 1000 / content_requirements['min_bandwidth_kbps']
        requirement_confidence = min(1.0, bandwidth_ratio)
        
        # Combined confidence
        overall_confidence = (
            network_confidence * 0.4 +
            stability_confidence * 0.3 +
            requirement_confidence * 0.3
        )
        
        return max(0.1, min(1.0, overall_confidence))
    
    async def _predict_quality_improvements(self,
                                          current_quality: StreamQuality,
                                          optimal_quality: StreamQuality) -> Dict[str, float]:
        """Predict improvements from quality adjustment"""
        current_specs = self.quality_specifications[current_quality]
        optimal_specs = self.quality_specifications[optimal_quality]
        
        bandwidth_change = (optimal_specs['bandwidth_kbps'] - current_specs['bandwidth_kbps']) / current_specs['bandwidth_kbps']
        
        # Predict improvements (simplified)
        improvements = {
            'user_experience': max(0.0, bandwidth_change * 0.3),
            'learning_effectiveness': max(0.0, bandwidth_change * 0.2),
            'engagement': max(0.0, bandwidth_change * 0.25),
            'content_delivery_success': max(0.0, bandwidth_change * 0.4)
        }
        
        return improvements
    
    # Helper methods for adjustment calculations
    async def _determine_adjustment_reason(self, current: StreamQuality, optimal: StreamQuality, network: Dict[str, Any]) -> str:
        """Determine reason for quality adjustment"""
        if optimal.value > current.value:
            return f"Network conditions improved - {network['network_tier']} quality detected"
        else:
            return f"Network conditions degraded - optimizing for {network['bottleneck_factor']} bottleneck"
    
    async def _calculate_expected_improvement(self, current: StreamQuality, optimal: StreamQuality) -> Dict[str, float]:
        """Calculate expected improvement from adjustment"""
        return {
            'quality_score': 0.15 if optimal != current else 0.0,
            'user_satisfaction': 0.10 if optimal != current else 0.0,
            'learning_effectiveness': 0.08 if optimal != current else 0.0
        }
    
    async def _calculate_adjustment_priority(self, current: StreamQuality, optimal: StreamQuality, network: Dict[str, Any]) -> int:
        """Calculate adjustment priority (1-10)"""
        if network['network_tier'] == 'poor':
            return 9  # High priority for poor networks
        elif network['network_tier'] == 'excellent':
            return 6  # Medium priority for excellent networks
        else:
            return 7  # Default priority
    
    async def _calculate_bandwidth_impact(self, current: StreamQuality, optimal: StreamQuality) -> float:
        """Calculate bandwidth impact of adjustment"""
        current_bw = self.quality_specifications[current]['bandwidth_kbps']
        optimal_bw = self.quality_specifications[optimal]['bandwidth_kbps']
        return (optimal_bw - current_bw) / current_bw
    
    async def _calculate_latency_impact(self, current: StreamQuality, optimal: StreamQuality) -> float:
        """Calculate latency impact of adjustment"""
        # Higher quality typically means slightly higher latency due to processing
        if optimal.value > current.value:
            return 0.05  # 5% increase
        else:
            return -0.03  # 3% decrease
    
    async def _calculate_learning_impact(self, current: StreamQuality, optimal: StreamQuality) -> float:
        """Calculate learning impact of quality adjustment"""
        # Better quality generally improves learning outcomes
        quality_diff = list(StreamQuality).index(optimal) - list(StreamQuality).index(current)
        return quality_diff * 0.1  # 10% per quality level
