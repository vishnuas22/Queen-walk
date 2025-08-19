"""
Bandwidth-Adaptive Content System

Intelligent content adaptation system that dynamically adjusts content delivery
based on available bandwidth while maintaining learning effectiveness and
providing optimal user experience across varying network conditions.
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

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .data_structures import (
    StreamQuality, NetworkCondition, ContentAdaptation
)


class BandwidthCategory(Enum):
    """Bandwidth categories for content adaptation"""
    ULTRA_LOW = "ultra_low"      # < 64 kbps
    LOW = "low"                  # 64-128 kbps
    MEDIUM = "medium"            # 128-256 kbps
    HIGH = "high"                # 256-512 kbps
    ULTRA_HIGH = "ultra_high"    # > 512 kbps


class AdaptationStrategy(Enum):
    """Content adaptation strategies"""
    TEXT_ONLY = "text_only"
    MINIMAL_GRAPHICS = "minimal_graphics"
    COMPRESSED_DATA = "compressed_data"
    OPTIMIZED_TEXT = "optimized_text"
    BASIC_GRAPHICS = "basic_graphics"
    REDUCED_MEDIA = "reduced_media"
    STANDARD_CONTENT = "standard_content"
    COMPRESSED_MEDIA = "compressed_media"
    ADAPTIVE_LOADING = "adaptive_loading"
    RICH_CONTENT = "rich_content"
    FULL_MEDIA = "full_media"
    INTERACTIVE_ELEMENTS = "interactive_elements"
    PREMIUM_CONTENT = "premium_content"
    HIGH_QUALITY_MEDIA = "high_quality_media"
    REAL_TIME_FEATURES = "real_time_features"


@dataclass
class ContentVariant:
    """Content variant for different bandwidth conditions"""
    variant_id: str
    original_content_id: str
    bandwidth_category: BandwidthCategory
    adaptation_strategies: List[AdaptationStrategy]
    estimated_size_kb: float
    estimated_bandwidth_kbps: float
    quality_score: float
    learning_effectiveness_score: float
    user_experience_score: float
    creation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationProfile:
    """User adaptation profile"""
    user_id: str
    bandwidth_history: deque = field(default_factory=lambda: deque(maxlen=50))
    adaptation_preferences: Dict[str, float] = field(default_factory=dict)
    device_capabilities: Dict[str, Any] = field(default_factory=dict)
    learning_style_preferences: Dict[str, float] = field(default_factory=dict)
    quality_tolerance: float = 0.7
    last_updated: datetime = field(default_factory=datetime.now)


class BandwidthAdaptiveContent:
    """
    🌐 BANDWIDTH-ADAPTIVE CONTENT SYSTEM
    
    Intelligent content adaptation system that dynamically adjusts content
    delivery based on available bandwidth while maintaining learning effectiveness.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Content adaptation management
        self.content_adaptation_models: Dict[str, Any] = {}
        self.bandwidth_monitoring: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.adaptation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.content_variants: Dict[str, Dict[str, ContentVariant]] = defaultdict(dict)
        self.adaptation_profiles: Dict[str, AdaptationProfile] = {}
        
        # Content adaptation strategies by bandwidth category
        self.adaptation_strategies = {
            BandwidthCategory.ULTRA_LOW: {
                'max_bandwidth_kbps': 64,
                'strategies': [
                    AdaptationStrategy.TEXT_ONLY,
                    AdaptationStrategy.MINIMAL_GRAPHICS,
                    AdaptationStrategy.COMPRESSED_DATA
                ],
                'quality_target': 0.4,
                'learning_effectiveness_target': 0.7
            },
            BandwidthCategory.LOW: {
                'max_bandwidth_kbps': 128,
                'strategies': [
                    AdaptationStrategy.OPTIMIZED_TEXT,
                    AdaptationStrategy.BASIC_GRAPHICS,
                    AdaptationStrategy.REDUCED_MEDIA
                ],
                'quality_target': 0.6,
                'learning_effectiveness_target': 0.8
            },
            BandwidthCategory.MEDIUM: {
                'max_bandwidth_kbps': 256,
                'strategies': [
                    AdaptationStrategy.STANDARD_CONTENT,
                    AdaptationStrategy.COMPRESSED_MEDIA,
                    AdaptationStrategy.ADAPTIVE_LOADING
                ],
                'quality_target': 0.8,
                'learning_effectiveness_target': 0.9
            },
            BandwidthCategory.HIGH: {
                'max_bandwidth_kbps': 512,
                'strategies': [
                    AdaptationStrategy.RICH_CONTENT,
                    AdaptationStrategy.FULL_MEDIA,
                    AdaptationStrategy.INTERACTIVE_ELEMENTS
                ],
                'quality_target': 0.9,
                'learning_effectiveness_target': 0.95
            },
            BandwidthCategory.ULTRA_HIGH: {
                'max_bandwidth_kbps': float('inf'),
                'strategies': [
                    AdaptationStrategy.PREMIUM_CONTENT,
                    AdaptationStrategy.HIGH_QUALITY_MEDIA,
                    AdaptationStrategy.REAL_TIME_FEATURES
                ],
                'quality_target': 1.0,
                'learning_effectiveness_target': 1.0
            }
        }
        
        logger.info("Bandwidth-Adaptive Content System initialized")
    
    async def adapt_content_for_bandwidth(self,
                                        user_id: str,
                                        session_id: str,
                                        content_request: Dict[str, Any],
                                        current_bandwidth: float,
                                        network_conditions: NetworkCondition) -> Dict[str, Any]:
        """
        Adapt content for current bandwidth conditions
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            content_request: Content request details
            current_bandwidth: Current available bandwidth in kbps
            network_conditions: Current network conditions
            
        Returns:
            Dict: Content adaptation result
        """
        try:
            # Get or create adaptation profile
            profile = await self._get_adaptation_profile(user_id)
            
            # Determine bandwidth category
            bandwidth_category = self._categorize_bandwidth(current_bandwidth)
            
            # Analyze adaptation requirements
            adaptation_requirements = await self._analyze_adaptation_requirements(
                content_request, bandwidth_category, network_conditions, profile
            )
            
            # Generate content adaptations
            content_adaptations = await self._generate_content_adaptations(
                content_request, adaptation_requirements, bandwidth_category
            )
            
            # Predict learning impact of adaptations
            learning_impact = await self._predict_adaptation_learning_impact(
                content_adaptations, content_request
            )
            
            # Create adaptation record
            adaptation_record = ContentAdaptation(
                adaptation_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                user_id=user_id,
                session_id=session_id,
                original_content_id=content_request.get('content_id', ''),
                adapted_content_id=content_adaptations.get('adapted_content_id', ''),
                adaptation_type=bandwidth_category.value,
                adaptation_reason=f"Bandwidth optimization for {current_bandwidth:.0f} kbps",
                quality_reduction=content_adaptations.get('quality_reduction', 0.0),
                bandwidth_saved=content_adaptations.get('bandwidth_saved', 0.0),
                learning_impact_score=learning_impact,
                user_satisfaction_prediction=content_adaptations.get('user_satisfaction_prediction', 0.8)
            )
            
            # Store adaptation
            self.adaptation_history[user_id].append(adaptation_record)
            
            # Update bandwidth monitoring
            self.bandwidth_monitoring[user_id].append(current_bandwidth)
            
            return {
                'status': 'success',
                'adaptation_result': {
                    'original_content': content_request,
                    'adapted_content': content_adaptations,
                    'bandwidth_category': bandwidth_category.value,
                    'adaptation_strategies': adaptation_requirements.get('selected_strategies', []),
                    'quality_impact': content_adaptations.get('quality_reduction', 0.0),
                    'learning_impact': learning_impact,
                    'bandwidth_savings': content_adaptations.get('bandwidth_saved', 0.0),
                    'estimated_load_time': content_adaptations.get('estimated_load_time', 0.0),
                    'user_experience_prediction': content_adaptations.get('user_satisfaction_prediction', 0.8)
                },
                'adaptation_record': adaptation_record.__dict__,
                'adaptation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adapting content for bandwidth: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def adapt_to_quality_change(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt content to quality change event
        
        Args:
            event: Quality change event data
            
        Returns:
            Dict: Adaptation result
        """
        try:
            user_id = event.get('user_id', '')
            session_id = event.get('session_id', '')
            new_quality = StreamQuality(event.get('event_data', {}).get('new_quality', 'medium'))
            
            # Estimate bandwidth from quality level
            quality_bandwidth_map = {
                StreamQuality.ULTRA_LOW: 64,
                StreamQuality.LOW: 128,
                StreamQuality.MEDIUM: 256,
                StreamQuality.HIGH: 512,
                StreamQuality.ULTRA_HIGH: 1024
            }
            
            estimated_bandwidth = quality_bandwidth_map.get(new_quality, 256)
            
            # Create mock network conditions
            network_conditions = NetworkCondition(
                bandwidth_kbps=estimated_bandwidth,
                latency_ms=100.0,
                packet_loss_rate=0.01,
                connection_stability=0.8,
                device_capabilities={'video': True, 'audio': True},
                optimal_quality=new_quality,
                adaptive_recommendations=[]
            )
            
            # Create content request (simplified)
            content_request = {
                'content_id': event.get('content_id', 'current_content'),
                'content_type': 'mixed_media',
                'learning_objectives': ['maintain_engagement', 'preserve_effectiveness']
            }
            
            # Adapt content
            result = await self.adapt_content_for_bandwidth(
                user_id, session_id, content_request, estimated_bandwidth, network_conditions
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error adapting to quality change: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_adaptation_profile(self, user_id: str) -> AdaptationProfile:
        """Get or create adaptation profile for user"""
        if user_id not in self.adaptation_profiles:
            self.adaptation_profiles[user_id] = AdaptationProfile(
                user_id=user_id,
                adaptation_preferences={
                    'quality_over_speed': 0.7,
                    'interactivity_importance': 0.8,
                    'media_preference': 0.6,
                    'text_readability': 0.9
                },
                device_capabilities={
                    'screen_size': 'medium',
                    'processing_power': 'medium',
                    'storage_capacity': 'high',
                    'network_type': 'wifi'
                },
                learning_style_preferences={
                    'visual': 0.8,
                    'auditory': 0.6,
                    'kinesthetic': 0.7,
                    'reading': 0.9
                },
                quality_tolerance=0.7
            )
        
        return self.adaptation_profiles[user_id]
    
    def _categorize_bandwidth(self, bandwidth_kbps: float) -> BandwidthCategory:
        """Categorize bandwidth into appropriate category"""
        if bandwidth_kbps < 64:
            return BandwidthCategory.ULTRA_LOW
        elif bandwidth_kbps < 128:
            return BandwidthCategory.LOW
        elif bandwidth_kbps < 256:
            return BandwidthCategory.MEDIUM
        elif bandwidth_kbps < 512:
            return BandwidthCategory.HIGH
        else:
            return BandwidthCategory.ULTRA_HIGH
    
    async def _analyze_adaptation_requirements(self,
                                             content_request: Dict[str, Any],
                                             bandwidth_category: BandwidthCategory,
                                             network_conditions: NetworkCondition,
                                             profile: AdaptationProfile) -> Dict[str, Any]:
        """Analyze content adaptation requirements"""
        # Get strategy configuration for bandwidth category
        strategy_config = self.adaptation_strategies[bandwidth_category]
        
        # Analyze content characteristics
        content_type = content_request.get('content_type', 'mixed_media')
        content_size = content_request.get('estimated_size_kb', 1000)
        interactivity_level = content_request.get('interactivity_level', 'medium')
        
        # Determine adaptation priorities
        adaptation_priorities = {
            'preserve_learning_objectives': 1.0,
            'maintain_user_engagement': 0.9,
            'optimize_load_time': 0.8,
            'preserve_interactivity': profile.adaptation_preferences.get('interactivity_importance', 0.8),
            'maintain_visual_quality': profile.adaptation_preferences.get('quality_over_speed', 0.7)
        }
        
        # Select appropriate strategies
        selected_strategies = await self._select_adaptation_strategies(
            strategy_config['strategies'], content_type, adaptation_priorities
        )
        
        return {
            'bandwidth_category': bandwidth_category.value,
            'target_bandwidth': strategy_config['max_bandwidth_kbps'],
            'quality_target': strategy_config['quality_target'],
            'learning_effectiveness_target': strategy_config['learning_effectiveness_target'],
            'selected_strategies': [s.value for s in selected_strategies],
            'adaptation_priorities': adaptation_priorities,
            'content_characteristics': {
                'type': content_type,
                'size_kb': content_size,
                'interactivity': interactivity_level
            },
            'network_constraints': {
                'bandwidth_kbps': network_conditions.bandwidth_kbps,
                'latency_ms': network_conditions.latency_ms,
                'stability': network_conditions.connection_stability
            }
        }
    
    async def _select_adaptation_strategies(self,
                                          available_strategies: List[AdaptationStrategy],
                                          content_type: str,
                                          priorities: Dict[str, float]) -> List[AdaptationStrategy]:
        """Select optimal adaptation strategies"""
        selected = []
        
        # Strategy selection based on content type and priorities
        for strategy in available_strategies:
            if strategy == AdaptationStrategy.TEXT_ONLY:
                if content_type in ['text', 'document'] or priorities['optimize_load_time'] > 0.8:
                    selected.append(strategy)
            
            elif strategy == AdaptationStrategy.COMPRESSED_MEDIA:
                if content_type in ['video', 'audio', 'mixed_media']:
                    selected.append(strategy)
            
            elif strategy == AdaptationStrategy.INTERACTIVE_ELEMENTS:
                if priorities['preserve_interactivity'] > 0.7:
                    selected.append(strategy)
            
            elif strategy == AdaptationStrategy.ADAPTIVE_LOADING:
                if priorities['optimize_load_time'] > 0.6:
                    selected.append(strategy)
            
            else:
                # Include other strategies by default
                selected.append(strategy)
        
        return selected[:3]  # Limit to top 3 strategies
    
    async def _generate_content_adaptations(self,
                                          content_request: Dict[str, Any],
                                          requirements: Dict[str, Any],
                                          bandwidth_category: BandwidthCategory) -> Dict[str, Any]:
        """Generate content adaptations based on requirements"""
        original_size = content_request.get('estimated_size_kb', 1000)
        target_bandwidth = requirements['target_bandwidth']
        selected_strategies = requirements['selected_strategies']
        
        # Calculate adaptation effects
        size_reduction = await self._calculate_size_reduction(selected_strategies, original_size)
        quality_reduction = await self._calculate_quality_reduction(selected_strategies)
        load_time_improvement = await self._calculate_load_time_improvement(size_reduction, target_bandwidth)
        
        # Generate adapted content specification
        adapted_content = {
            'adapted_content_id': f"adapted_{content_request.get('content_id', 'content')}_{bandwidth_category.value}",
            'adaptation_strategies_applied': selected_strategies,
            'original_size_kb': original_size,
            'adapted_size_kb': original_size * (1 - size_reduction),
            'size_reduction_percentage': size_reduction * 100,
            'quality_reduction': quality_reduction,
            'bandwidth_saved': original_size * size_reduction * 8 / 1024,  # Convert to kbps
            'estimated_load_time': (original_size * (1 - size_reduction)) / target_bandwidth,
            'user_satisfaction_prediction': await self._predict_user_satisfaction(
                quality_reduction, load_time_improvement
            ),
            'learning_effectiveness_preservation': requirements['learning_effectiveness_target']
        }
        
        return adapted_content
    
    async def _predict_adaptation_learning_impact(self,
                                                content_adaptations: Dict[str, Any],
                                                original_request: Dict[str, Any]) -> float:
        """Predict learning impact of content adaptations"""
        quality_reduction = content_adaptations.get('quality_reduction', 0.0)
        load_time_improvement = content_adaptations.get('estimated_load_time', 5.0)
        
        # Learning impact factors
        quality_impact = 1.0 - (quality_reduction * 0.3)  # Quality reduction affects learning
        accessibility_impact = min(1.0, 10.0 / max(load_time_improvement, 1.0))  # Faster loading improves access
        
        # Combined learning impact
        learning_impact = (quality_impact * 0.6 + accessibility_impact * 0.4)
        
        return max(0.3, min(1.0, learning_impact))
    
    async def _calculate_size_reduction(self, strategies: List[str], original_size: float) -> float:
        """Calculate size reduction from adaptation strategies"""
        reduction_factors = {
            'text_only': 0.8,
            'minimal_graphics': 0.6,
            'compressed_data': 0.4,
            'optimized_text': 0.3,
            'basic_graphics': 0.4,
            'reduced_media': 0.5,
            'compressed_media': 0.3,
            'adaptive_loading': 0.2
        }
        
        total_reduction = 0.0
        for strategy in strategies:
            total_reduction += reduction_factors.get(strategy, 0.1)
        
        return min(0.8, total_reduction)  # Cap at 80% reduction
    
    async def _calculate_quality_reduction(self, strategies: List[str]) -> float:
        """Calculate quality reduction from adaptation strategies"""
        quality_impacts = {
            'text_only': 0.6,
            'minimal_graphics': 0.4,
            'compressed_data': 0.3,
            'optimized_text': 0.1,
            'basic_graphics': 0.2,
            'reduced_media': 0.3,
            'compressed_media': 0.2,
            'adaptive_loading': 0.05
        }
        
        total_impact = 0.0
        for strategy in strategies:
            total_impact += quality_impacts.get(strategy, 0.1)
        
        return min(0.7, total_impact)  # Cap at 70% quality reduction
    
    async def _calculate_load_time_improvement(self, size_reduction: float, target_bandwidth: float) -> float:
        """Calculate load time improvement from size reduction"""
        # Simplified calculation: improvement proportional to size reduction
        return size_reduction * 0.8  # 80% of size reduction translates to load time improvement
    
    async def _predict_user_satisfaction(self, quality_reduction: float, load_time_improvement: float) -> float:
        """Predict user satisfaction with adaptations"""
        # Balance between quality loss and speed gain
        quality_satisfaction = 1.0 - quality_reduction
        speed_satisfaction = min(1.0, load_time_improvement * 2)  # Speed improvements are valued
        
        # Weighted combination
        overall_satisfaction = quality_satisfaction * 0.6 + speed_satisfaction * 0.4
        
        return max(0.3, min(1.0, overall_satisfaction))
