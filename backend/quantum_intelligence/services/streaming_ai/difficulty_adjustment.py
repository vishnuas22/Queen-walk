"""
Real-Time Difficulty Adjustment System

AI-powered system that continuously monitors learning performance and adjusts
content difficulty in real-time for optimal challenge-skill balance with
sub-200ms response times.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import time
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
    from sklearn.linear_model import LinearRegression
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
    DifficultyAdjustment, DifficultyAdjustmentReason,
    StreamingEvent, StreamingEventType
)


class PerformanceZone(Enum):
    """Performance zones for difficulty calibration"""
    FRUSTRATION = "frustration"      # < 0.4 - Too difficult
    SUPPORT = "support"              # 0.4-0.65 - Needs support
    OPTIMAL = "optimal"              # 0.65-0.85 - Sweet spot
    CHALLENGE = "challenge"          # 0.85-0.95 - Increase difficulty
    MASTERY = "mastery"              # > 0.95 - Mastered, advance


class AdjustmentUrgency(Enum):
    """Urgency levels for difficulty adjustments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    accuracy: float
    response_time: float
    engagement_level: float
    cognitive_load: float
    confidence_level: float
    help_seeking_frequency: float
    error_patterns: List[str]
    learning_velocity: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DifficultyModel:
    """User-specific difficulty model"""
    user_id: str
    current_difficulty: float
    target_performance: float
    adjustment_rate: float
    stability_preference: float
    challenge_tolerance: float
    learning_velocity: float
    performance_history: deque = field(default_factory=lambda: deque(maxlen=50))
    adjustment_history: List[DifficultyAdjustment] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class RealTimeDifficultyAdjustment:
    """
    ⚡ REAL-TIME DIFFICULTY ADJUSTMENT SYSTEM

    AI-powered system that continuously monitors learning performance
    and adjusts content difficulty in real-time for optimal challenge-skill balance.
    """

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service

        # User difficulty models
        self.difficulty_models: Dict[str, DifficultyModel] = {}
        self.performance_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.adjustment_queue: asyncio.Queue = asyncio.Queue()

        # Real-time adjustment parameters
        self.adjustment_sensitivity = 0.7  # How quickly to adjust
        self.stability_threshold = 0.1     # Minimum change to trigger adjustment
        self.max_adjustment_per_step = 0.2 # Maximum difficulty change per adjustment

        # Performance zones for difficulty calibration
        self.performance_zones = {
            PerformanceZone.FRUSTRATION: (0.0, 0.4),
            PerformanceZone.SUPPORT: (0.4, 0.65),
            PerformanceZone.OPTIMAL: (0.65, 0.85),
            PerformanceZone.CHALLENGE: (0.85, 0.95),
            PerformanceZone.MASTERY: (0.95, 1.0)
        }

        # ML models (if available)
        if SKLEARN_AVAILABLE:
            self.difficulty_predictor = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                random_state=42
            )
            self.performance_predictor = LinearRegression()
        else:
            self.difficulty_predictor = None
            self.performance_predictor = None

        # Start real-time processing (will be created when needed)
        self.processing_task = None

        logger.info("Real-Time Difficulty Adjustment System initialized")

    def _ensure_processing_task(self):
        """Ensure processing task is running"""
        if self.processing_task is None:
            try:
                self.processing_task = asyncio.create_task(self._process_adjustments())
            except RuntimeError:
                # No event loop running, task will be created when needed
                pass

    async def initialize_user_model(self,
                                  user_id: str,
                                  learning_profile: Dict[str, Any],
                                  subject_domain: str) -> DifficultyModel:
        """
        Initialize personalized difficulty model for a user

        Args:
            user_id: User identifier
            learning_profile: User's learning profile data
            subject_domain: Subject domain for difficulty modeling

        Returns:
            DifficultyModel: Initialized difficulty model
        """
        try:
            # Analyze baseline capabilities
            baseline_analysis = await self._analyze_baseline_capabilities(
                user_id, learning_profile, subject_domain
            )

            # Create difficulty model
            model = DifficultyModel(
                user_id=user_id,
                current_difficulty=baseline_analysis['starting_difficulty'],
                target_performance=0.75,  # Target 75% success rate
                adjustment_rate=baseline_analysis['adjustment_rate'],
                stability_preference=baseline_analysis['stability_preference'],
                challenge_tolerance=baseline_analysis['challenge_tolerance'],
                learning_velocity=baseline_analysis['learning_velocity']
            )

            # Store model
            self.difficulty_models[user_id] = model

            logger.info(f"Difficulty model initialized for user: {user_id}")
            return model

        except Exception as e:
            logger.error(f"Error initializing difficulty model: {e}")
            raise QuantumEngineError(f"Failed to initialize difficulty model: {e}")

    async def analyze_performance_change(self,
                                       event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance change and determine if difficulty adjustment is needed

        Args:
            event: User action event data

        Returns:
            Dict: Performance analysis and adjustment recommendation
        """
        start_time = time.time()

        # Ensure processing task is running
        self._ensure_processing_task()

        try:
            user_id = event.get('user_id', '')
            session_id = event.get('session_id', '')

            if user_id not in self.difficulty_models:
                # Initialize model if not exists
                await self.initialize_user_model(
                    user_id,
                    event.get('learning_profile', {}),
                    event.get('subject_domain', 'general')
                )

            # Extract performance metrics
            performance_metrics = self._extract_performance_metrics(event.get('event_data', {}))

            # Update performance tracking
            self.performance_tracking[user_id].append(performance_metrics)
            self.difficulty_models[user_id].performance_history.append(performance_metrics)

            # Analyze current performance zone
            current_zone = self._determine_performance_zone(performance_metrics.accuracy)

            # Calculate difficulty adjustment recommendation
            adjustment_recommendation = await self._calculate_difficulty_adjustment(
                user_id, performance_metrics, current_zone
            )

            # Generate analysis result
            analysis_result = {
                'user_id': user_id,
                'session_id': session_id,
                'current_performance': performance_metrics.__dict__,
                'performance_zone': current_zone.value,
                'adjustment_recommendation': adjustment_recommendation,
                'processing_latency_ms': (time.time() - start_time) * 1000,
                'analysis_timestamp': datetime.now().isoformat()
            }

            # Queue adjustment if needed
            if adjustment_recommendation['adjustment_needed']:
                await self.adjustment_queue.put({
                    'user_id': user_id,
                    'session_id': session_id,
                    'adjustment': adjustment_recommendation,
                    'timestamp': datetime.now()
                })

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing performance change: {e}")
            return {
                'error': str(e),
                'processing_latency_ms': (time.time() - start_time) * 1000
            }

    async def apply_difficulty_adjustment(self,
                                        user_id: str,
                                        session_id: str,
                                        adjustment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply difficulty adjustment with smooth transitions

        Args:
            user_id: User identifier
            session_id: Session identifier
            adjustment_data: Adjustment parameters

        Returns:
            Dict: Adjustment application result
        """
        try:
            if user_id not in self.difficulty_models:
                return {'error': 'User model not found'}

            model = self.difficulty_models[user_id]
            current_difficulty = model.current_difficulty
            target_difficulty = adjustment_data['target_difficulty']

            # Calculate smooth transition
            transition_step = self._calculate_transition_step(
                current_difficulty, target_difficulty, adjustment_data.get('urgency', AdjustmentUrgency.MEDIUM)
            )

            # Apply adjustment
            new_difficulty = current_difficulty + transition_step
            new_difficulty = max(0.1, min(1.0, new_difficulty))  # Clamp to valid range

            # Update model
            model.current_difficulty = new_difficulty
            model.last_updated = datetime.now()

            # Create adjustment record
            adjustment_record = DifficultyAdjustment(
                adjustment_id=f"adj_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                user_id=user_id,
                session_id=session_id,
                previous_difficulty=current_difficulty,
                new_difficulty=new_difficulty,
                adjustment_magnitude=abs(transition_step),
                reason=DifficultyAdjustmentReason(adjustment_data.get('reason', 'performance_too_low')),
                confidence_score=adjustment_data.get('confidence', 0.8),
                expected_impact=adjustment_data.get('expected_impact', {}),
                adjustment_timestamp=datetime.now(),
                performance_context=adjustment_data.get('performance_context', {}),
                learning_context=adjustment_data.get('learning_context', {})
            )

            # Store adjustment
            model.adjustment_history.append(adjustment_record)

            return {
                'status': 'success',
                'user_id': user_id,
                'session_id': session_id,
                'previous_difficulty': current_difficulty,
                'new_difficulty': new_difficulty,
                'adjustment_magnitude': abs(transition_step),
                'adjustment_reason': adjustment_data.get('reason'),
                'expected_impact': adjustment_data.get('expected_impact', {}),
                'adjustment_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error applying difficulty adjustment: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _process_adjustments(self):
        """Process difficulty adjustments from queue"""
        while True:
            try:
                # Get adjustment from queue with timeout
                try:
                    adjustment_request = await asyncio.wait_for(
                        self.adjustment_queue.get(), timeout=1.0
                    )

                    # Apply adjustment
                    result = await self.apply_difficulty_adjustment(
                        adjustment_request['user_id'],
                        adjustment_request['session_id'],
                        adjustment_request['adjustment']
                    )

                    # Log result
                    if result.get('status') == 'success':
                        logger.info(f"Applied difficulty adjustment for user {adjustment_request['user_id']}")
                    else:
                        logger.error(f"Failed to apply difficulty adjustment: {result.get('error')}")

                except asyncio.TimeoutError:
                    # No adjustments to process
                    pass

                # Brief sleep to prevent CPU spinning
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in adjustment processing: {e}")
                await asyncio.sleep(1.0)

    def _extract_performance_metrics(self, event_data: Dict[str, Any]) -> PerformanceMetrics:
        """Extract performance metrics from event data"""
        return PerformanceMetrics(
            accuracy=event_data.get('accuracy', 0.5),
            response_time=event_data.get('response_time', 5.0),
            engagement_level=event_data.get('engagement_level', 0.7),
            cognitive_load=event_data.get('cognitive_load', 0.5),
            confidence_level=event_data.get('confidence_level', 0.6),
            help_seeking_frequency=event_data.get('help_seeking_frequency', 0.2),
            error_patterns=event_data.get('error_patterns', []),
            learning_velocity=event_data.get('learning_velocity', 0.5)
        )

    def _determine_performance_zone(self, accuracy: float) -> PerformanceZone:
        """Determine performance zone based on accuracy"""
        for zone, (min_acc, max_acc) in self.performance_zones.items():
            if min_acc <= accuracy < max_acc:
                return zone
        return PerformanceZone.OPTIMAL  # Default

    async def _calculate_difficulty_adjustment(self,
                                             user_id: str,
                                             performance_metrics: PerformanceMetrics,
                                             current_zone: PerformanceZone) -> Dict[str, Any]:
        """Calculate optimal difficulty adjustment"""
        model = self.difficulty_models[user_id]
        current_difficulty = model.current_difficulty

        # Determine if adjustment is needed
        adjustment_needed = False
        target_difficulty = current_difficulty
        reason = None
        urgency = AdjustmentUrgency.LOW

        if current_zone == PerformanceZone.FRUSTRATION:
            # Performance too low, decrease difficulty
            adjustment_needed = True
            target_difficulty = max(0.1, current_difficulty - 0.15)
            reason = DifficultyAdjustmentReason.PERFORMANCE_TOO_LOW
            urgency = AdjustmentUrgency.HIGH

        elif current_zone == PerformanceZone.SUPPORT:
            # Performance low, slight decrease
            if performance_metrics.accuracy < 0.5:
                adjustment_needed = True
                target_difficulty = max(0.1, current_difficulty - 0.1)
                reason = DifficultyAdjustmentReason.PERFORMANCE_TOO_LOW
                urgency = AdjustmentUrgency.MEDIUM

        elif current_zone == PerformanceZone.CHALLENGE:
            # Performance high, increase difficulty
            adjustment_needed = True
            target_difficulty = min(1.0, current_difficulty + 0.1)
            reason = DifficultyAdjustmentReason.PERFORMANCE_TOO_HIGH
            urgency = AdjustmentUrgency.MEDIUM

        elif current_zone == PerformanceZone.MASTERY:
            # Mastery achieved, significant increase
            adjustment_needed = True
            target_difficulty = min(1.0, current_difficulty + 0.2)
            reason = DifficultyAdjustmentReason.MASTERY_ACHIEVED
            urgency = AdjustmentUrgency.HIGH

        # Check for cognitive overload
        if performance_metrics.cognitive_load > 0.8:
            adjustment_needed = True
            target_difficulty = max(0.1, current_difficulty - 0.1)
            reason = DifficultyAdjustmentReason.COGNITIVE_OVERLOAD
            urgency = AdjustmentUrgency.HIGH

        # Check for engagement issues
        if performance_metrics.engagement_level < 0.4:
            adjustment_needed = True
            # Adjust based on current performance
            if performance_metrics.accuracy > 0.8:
                target_difficulty = min(1.0, current_difficulty + 0.1)  # Increase challenge
            else:
                target_difficulty = max(0.1, current_difficulty - 0.1)  # Decrease difficulty
            reason = DifficultyAdjustmentReason.ENGAGEMENT_DROPPING
            urgency = AdjustmentUrgency.MEDIUM

        # Calculate confidence score
        confidence_score = self._calculate_adjustment_confidence(
            model, performance_metrics, target_difficulty
        )

        return {
            'adjustment_needed': adjustment_needed,
            'target_difficulty': target_difficulty,
            'current_difficulty': current_difficulty,
            'adjustment_magnitude': abs(target_difficulty - current_difficulty),
            'reason': reason.value if reason else None,
            'urgency': urgency.value,
            'confidence': confidence_score,
            'expected_impact': {
                'accuracy_change': self._predict_accuracy_change(target_difficulty - current_difficulty),
                'engagement_change': self._predict_engagement_change(target_difficulty - current_difficulty),
                'learning_velocity_change': self._predict_velocity_change(target_difficulty - current_difficulty)
            },
            'performance_context': performance_metrics.__dict__,
            'learning_context': {
                'current_zone': current_zone.value,
                'learning_velocity': model.learning_velocity,
                'stability_preference': model.stability_preference
            }
        }

    def _calculate_transition_step(self,
                                 current_difficulty: float,
                                 target_difficulty: float,
                                 urgency: AdjustmentUrgency) -> float:
        """Calculate smooth transition step"""
        difference = target_difficulty - current_difficulty

        # Adjust step size based on urgency
        urgency_multipliers = {
            AdjustmentUrgency.LOW: 0.3,
            AdjustmentUrgency.MEDIUM: 0.6,
            AdjustmentUrgency.HIGH: 0.8,
            AdjustmentUrgency.CRITICAL: 1.0
        }

        multiplier = urgency_multipliers.get(urgency, 0.6)
        step = difference * multiplier

        # Clamp to maximum adjustment per step
        return max(-self.max_adjustment_per_step, min(self.max_adjustment_per_step, step))

    def _calculate_adjustment_confidence(self,
                                       model: DifficultyModel,
                                       performance_metrics: PerformanceMetrics,
                                       target_difficulty: float) -> float:
        """Calculate confidence in adjustment recommendation"""
        # Base confidence on data availability
        data_points = len(model.performance_history)
        data_confidence = min(1.0, data_points / 10.0)  # Full confidence with 10+ data points

        # Confidence based on performance consistency
        if data_points > 1:
            recent_accuracies = [p.accuracy for p in list(model.performance_history)[-5:]]
            consistency = 1.0 - np.std(recent_accuracies) if recent_accuracies else 0.5
        else:
            consistency = 0.5

        # Confidence based on adjustment magnitude
        adjustment_magnitude = abs(target_difficulty - model.current_difficulty)
        magnitude_confidence = 1.0 - min(1.0, adjustment_magnitude / 0.5)  # Lower confidence for large adjustments

        # Combined confidence
        overall_confidence = (data_confidence * 0.4 + consistency * 0.3 + magnitude_confidence * 0.3)

        return max(0.1, min(1.0, overall_confidence))

    # Prediction methods (simplified implementations)
    def _predict_accuracy_change(self, difficulty_change: float) -> float:
        """Predict accuracy change based on difficulty adjustment"""
        # Simplified: inverse relationship between difficulty and accuracy
        return -difficulty_change * 0.3

    def _predict_engagement_change(self, difficulty_change: float) -> float:
        """Predict engagement change based on difficulty adjustment"""
        # Simplified: moderate difficulty changes can improve engagement
        if abs(difficulty_change) < 0.1:
            return abs(difficulty_change) * 0.2
        else:
            return -abs(difficulty_change) * 0.1

    def _predict_velocity_change(self, difficulty_change: float) -> float:
        """Predict learning velocity change based on difficulty adjustment"""
        # Simplified: optimal difficulty improves velocity
        return -abs(difficulty_change) * 0.1

    async def _analyze_baseline_capabilities(self,
                                           user_id: str,
                                           learning_profile: Dict[str, Any],
                                           subject_domain: str) -> Dict[str, Any]:
        """Analyze user's baseline capabilities for difficulty modeling"""
        # Simplified baseline analysis
        return {
            'starting_difficulty': learning_profile.get('estimated_ability', 0.5),
            'adjustment_rate': learning_profile.get('adaptation_speed', 0.7),
            'stability_preference': learning_profile.get('stability_preference', 0.6),
            'challenge_tolerance': learning_profile.get('challenge_tolerance', 0.7),
            'learning_velocity': learning_profile.get('learning_velocity', 0.5)
        }