"""
Behavioral Intelligence System

Advanced user behavior modeling and engagement analytics for learning platforms.
Implements Hidden Markov Models, clustering analysis, personalization insights,
behavioral anomaly detection, and motivation prediction using behavioral economics.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import random

# Try to import advanced libraries
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class BehaviorState(Enum):
    """User behavior states"""
    HIGHLY_ENGAGED = "highly_engaged"
    MODERATELY_ENGAGED = "moderately_engaged"
    PASSIVELY_ENGAGED = "passively_engaged"
    DISENGAGED = "disengaged"
    STRUGGLING = "struggling"
    EXPLORING = "exploring"
    FOCUSED_LEARNING = "focused_learning"
    PROCRASTINATING = "procrastinating"


class EngagementLevel(Enum):
    """Engagement level categories"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class LearningStyle(Enum):
    """Learning style categories"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"


@dataclass
class BehaviorPattern:
    """Individual behavior pattern"""
    pattern_id: str = ""
    pattern_type: str = ""
    frequency: float = 0.0
    duration_avg: float = 0.0
    intensity: float = 0.0
    context_factors: Dict[str, Any] = field(default_factory=dict)
    temporal_distribution: Dict[str, float] = field(default_factory=dict)
    predictive_indicators: List[str] = field(default_factory=list)


@dataclass
class UserBehaviorProfile:
    """Comprehensive user behavior profile"""
    user_id: str = ""
    behavior_state: BehaviorState = BehaviorState.MODERATELY_ENGAGED
    engagement_level: EngagementLevel = EngagementLevel.MODERATE
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    motivation_indicators: Dict[str, float] = field(default_factory=dict)
    anomaly_scores: Dict[str, float] = field(default_factory=dict)
    personalization_insights: Dict[str, Any] = field(default_factory=dict)
    behavior_trends: Dict[str, List[float]] = field(default_factory=dict)
    last_updated: str = ""


@dataclass
class EngagementAnalysis:
    """Engagement analysis result"""
    user_id: str = ""
    overall_engagement_score: float = 0.0
    engagement_components: Dict[str, float] = field(default_factory=dict)
    engagement_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    peak_engagement_periods: List[Dict[str, Any]] = field(default_factory=list)
    engagement_barriers: List[str] = field(default_factory=list)
    engagement_drivers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_timestamp: str = ""


class BehavioralIntelligenceSystem:
    """
    🧠 BEHAVIORAL INTELLIGENCE SYSTEM
    
    Advanced user behavior modeling and engagement analytics using machine learning
    and behavioral economics principles.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # System configuration
        self.config = {
            'behavior_window_hours': 24,
            'pattern_detection_threshold': 0.7,
            'anomaly_detection_threshold': 0.3,
            'engagement_update_frequency': 'hourly',
            'clustering_algorithms': ['kmeans', 'dbscan'],
            'hmm_states': 8,
            'feature_importance_threshold': 0.1
        }
        
        # Behavior tracking
        self.user_profiles = {}
        self.behavior_models = {}
        self.engagement_history = {}
        self.anomaly_detectors = {}
        
        # Initialize ML models if available
        if SKLEARN_AVAILABLE:
            self._initialize_behavior_models()
        
        logger.info("Behavioral Intelligence System initialized")
    
    def _initialize_behavior_models(self):
        """Initialize machine learning models for behavior analysis"""
        # Clustering models for user segmentation
        self.behavior_models['kmeans_clusterer'] = KMeans(n_clusters=5, random_state=42)
        self.behavior_models['dbscan_clusterer'] = DBSCAN(eps=0.5, min_samples=5)
        
        # Anomaly detection model
        self.anomaly_detectors['isolation_forest'] = IsolationForest(
            contamination=0.1, random_state=42
        )
        
        # Feature scaler
        self.behavior_models['scaler'] = StandardScaler()
    
    async def analyze_user_behavior(self,
                                  user_id: str,
                                  behavioral_data: Dict[str, Any],
                                  learning_activities: List[Dict[str, Any]],
                                  context_data: Optional[Dict[str, Any]] = None) -> UserBehaviorProfile:
        """
        Comprehensive user behavior analysis
        
        Args:
            user_id: User identifier
            behavioral_data: Raw behavioral data (clicks, time spent, etc.)
            learning_activities: Learning activity history
            context_data: Additional context information
            
        Returns:
            UserBehaviorProfile: Comprehensive behavior profile
        """
        try:
            # Extract behavior features
            behavior_features = await self._extract_behavior_features(
                behavioral_data, learning_activities, context_data
            )
            
            # Detect behavior patterns
            behavior_patterns = await self._detect_behavior_patterns(behavior_features)
            
            # Classify behavior state
            behavior_state = await self._classify_behavior_state(behavior_features)
            
            # Analyze engagement level
            engagement_level = await self._analyze_engagement_level(behavior_features)
            
            # Infer learning style
            learning_style = await self._infer_learning_style(behavioral_data, learning_activities)
            
            # Calculate engagement metrics
            engagement_metrics = await self._calculate_engagement_metrics(behavior_features)
            
            # Analyze motivation indicators
            motivation_indicators = await self._analyze_motivation_indicators(
                behavior_features, learning_activities
            )
            
            # Detect behavioral anomalies
            anomaly_scores = await self._detect_behavioral_anomalies(behavior_features)
            
            # Generate personalization insights
            personalization_insights = await self._generate_personalization_insights(
                behavior_patterns, engagement_metrics, learning_style
            )
            
            # Analyze behavior trends
            behavior_trends = await self._analyze_behavior_trends(behavioral_data)
            
            # Create behavior profile
            profile = UserBehaviorProfile(
                user_id=user_id,
                behavior_state=behavior_state,
                engagement_level=engagement_level,
                learning_style=learning_style,
                behavior_patterns=behavior_patterns,
                engagement_metrics=engagement_metrics,
                motivation_indicators=motivation_indicators,
                anomaly_scores=anomaly_scores,
                personalization_insights=personalization_insights,
                behavior_trends=behavior_trends,
                last_updated=datetime.utcnow().isoformat()
            )
            
            # Store profile
            self.user_profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            raise QuantumEngineError(f"Behavior analysis failed: {e}")
    
    async def _extract_behavior_features(self,
                                       behavioral_data: Dict[str, Any],
                                       learning_activities: List[Dict[str, Any]],
                                       context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for behavior analysis"""
        features = {}
        
        # Time-based features
        if 'session_durations' in behavioral_data:
            sessions = behavioral_data['session_durations']
            features['avg_session_duration'] = np.mean(sessions)
            features['session_duration_variance'] = np.var(sessions)
            features['total_time_spent'] = sum(sessions)
            features['session_frequency'] = len(sessions)
        
        # Interaction features
        if 'click_patterns' in behavioral_data:
            clicks = behavioral_data['click_patterns']
            features['click_rate'] = len(clicks) / max(1, features.get('total_time_spent', 1))
            features['click_diversity'] = len(set(c.get('target_type', '') for c in clicks))
        
        if 'navigation_patterns' in behavioral_data:
            nav = behavioral_data['navigation_patterns']
            features['page_views'] = nav.get('page_views', 0)
            features['unique_pages'] = nav.get('unique_pages', 0)
            features['back_button_usage'] = nav.get('back_button_clicks', 0)
        
        # Learning activity features
        if learning_activities:
            features['activities_completed'] = len([a for a in learning_activities if a.get('completed', False)])
            features['activities_attempted'] = len(learning_activities)
            features['completion_rate'] = features['activities_completed'] / max(1, features['activities_attempted'])
            
            # Performance features
            scores = [a.get('score', 0) for a in learning_activities if 'score' in a]
            if scores:
                features['avg_performance'] = np.mean(scores)
                features['performance_variance'] = np.var(scores)
                features['performance_trend'] = self._calculate_trend(scores)
        
        # Temporal features
        if 'timestamps' in behavioral_data:
            timestamps = behavioral_data['timestamps']
            if timestamps:
                # Convert to hours of day
                hours = [datetime.fromisoformat(ts).hour for ts in timestamps if isinstance(ts, str)]
                if hours:
                    features['preferred_hour'] = max(set(hours), key=hours.count)
                    features['time_diversity'] = len(set(hours))
        
        # Content interaction features
        if 'content_interactions' in behavioral_data:
            content = behavioral_data['content_interactions']
            features['content_types_accessed'] = len(set(c.get('type', '') for c in content))
            features['content_engagement_depth'] = np.mean([c.get('engagement_score', 0.5) for c in content])
        
        # Context features
        if context_data:
            features['device_type'] = context_data.get('device_type', 'unknown')
            features['location_consistency'] = context_data.get('location_consistency', 0.5)
            features['social_context'] = context_data.get('social_context', 'individual')
        
        return features
    
    async def _detect_behavior_patterns(self, features: Dict[str, Any]) -> List[BehaviorPattern]:
        """Detect recurring behavior patterns"""
        patterns = []
        
        # Session duration patterns
        if 'avg_session_duration' in features and 'session_duration_variance' in features:
            if features['session_duration_variance'] < 100:  # Consistent sessions
                patterns.append(BehaviorPattern(
                    pattern_id="consistent_sessions",
                    pattern_type="temporal",
                    frequency=0.8,
                    duration_avg=features['avg_session_duration'],
                    intensity=0.7,
                    context_factors={'consistency': 'high'},
                    predictive_indicators=['regular_schedule', 'habit_formation']
                ))
        
        # High engagement pattern
        if features.get('completion_rate', 0) > 0.8 and features.get('click_rate', 0) > 0.5:
            patterns.append(BehaviorPattern(
                pattern_id="high_engagement",
                pattern_type="engagement",
                frequency=0.9,
                duration_avg=features.get('avg_session_duration', 0),
                intensity=0.9,
                context_factors={'engagement_type': 'active'},
                predictive_indicators=['task_completion', 'active_interaction']
            ))
        
        # Exploration pattern
        if features.get('content_types_accessed', 0) > 3 and features.get('unique_pages', 0) > 10:
            patterns.append(BehaviorPattern(
                pattern_id="exploration",
                pattern_type="learning_style",
                frequency=0.6,
                duration_avg=features.get('avg_session_duration', 0),
                intensity=0.6,
                context_factors={'exploration_breadth': 'high'},
                predictive_indicators=['curiosity', 'broad_interests']
            ))
        
        # Focused learning pattern
        if (features.get('avg_session_duration', 0) > 1800 and  # 30+ minutes
            features.get('back_button_usage', 0) < 5):
            patterns.append(BehaviorPattern(
                pattern_id="focused_learning",
                pattern_type="attention",
                frequency=0.7,
                duration_avg=features.get('avg_session_duration', 0),
                intensity=0.8,
                context_factors={'focus_depth': 'high'},
                predictive_indicators=['sustained_attention', 'goal_oriented']
            ))
        
        # Struggling pattern
        if (features.get('completion_rate', 1) < 0.3 and 
            features.get('performance_variance', 0) > 0.5):
            patterns.append(BehaviorPattern(
                pattern_id="struggling",
                pattern_type="performance",
                frequency=0.5,
                duration_avg=features.get('avg_session_duration', 0),
                intensity=0.4,
                context_factors={'difficulty_level': 'high'},
                predictive_indicators=['low_completion', 'inconsistent_performance']
            ))
        
        return patterns
    
    async def _classify_behavior_state(self, features: Dict[str, Any]) -> BehaviorState:
        """Classify current behavior state"""
        # Calculate engagement indicators
        completion_rate = features.get('completion_rate', 0)
        session_duration = features.get('avg_session_duration', 0)
        click_rate = features.get('click_rate', 0)
        performance = features.get('avg_performance', 0.5)
        
        # Engagement score
        engagement_score = (completion_rate + min(session_duration / 3600, 1) + 
                          min(click_rate, 1) + performance) / 4
        
        # Classify based on engagement and performance
        if engagement_score > 0.8 and performance > 0.8:
            return BehaviorState.HIGHLY_ENGAGED
        elif engagement_score > 0.6 and performance > 0.6:
            return BehaviorState.MODERATELY_ENGAGED
        elif engagement_score > 0.4:
            return BehaviorState.PASSIVELY_ENGAGED
        elif completion_rate < 0.3 and performance < 0.4:
            return BehaviorState.STRUGGLING
        elif features.get('content_types_accessed', 0) > 3:
            return BehaviorState.EXPLORING
        elif session_duration > 1800 and click_rate < 0.3:
            return BehaviorState.FOCUSED_LEARNING
        elif session_duration < 300:  # Less than 5 minutes
            return BehaviorState.PROCRASTINATING
        else:
            return BehaviorState.DISENGAGED
    
    async def _analyze_engagement_level(self, features: Dict[str, Any]) -> EngagementLevel:
        """Analyze overall engagement level"""
        engagement_indicators = []
        
        # Time engagement
        if 'total_time_spent' in features:
            time_score = min(features['total_time_spent'] / 7200, 1.0)  # 2 hours max
            engagement_indicators.append(time_score)
        
        # Activity engagement
        if 'completion_rate' in features:
            engagement_indicators.append(features['completion_rate'])
        
        # Interaction engagement
        if 'click_rate' in features:
            interaction_score = min(features['click_rate'], 1.0)
            engagement_indicators.append(interaction_score)
        
        # Content engagement
        if 'content_engagement_depth' in features:
            engagement_indicators.append(features['content_engagement_depth'])
        
        # Calculate overall engagement
        if engagement_indicators:
            overall_engagement = np.mean(engagement_indicators)
            
            if overall_engagement > 0.8:
                return EngagementLevel.VERY_HIGH
            elif overall_engagement > 0.6:
                return EngagementLevel.HIGH
            elif overall_engagement > 0.4:
                return EngagementLevel.MODERATE
            elif overall_engagement > 0.2:
                return EngagementLevel.LOW
            else:
                return EngagementLevel.VERY_LOW
        
        return EngagementLevel.MODERATE
    
    async def _infer_learning_style(self,
                                  behavioral_data: Dict[str, Any],
                                  learning_activities: List[Dict[str, Any]]) -> LearningStyle:
        """Infer learning style from behavior patterns"""
        style_indicators = {
            LearningStyle.VISUAL: 0.0,
            LearningStyle.AUDITORY: 0.0,
            LearningStyle.KINESTHETIC: 0.0,
            LearningStyle.READING_WRITING: 0.0
        }
        
        # Analyze content preferences
        if 'content_interactions' in behavioral_data:
            content_types = [c.get('type', '') for c in behavioral_data['content_interactions']]
            
            visual_content = sum(1 for t in content_types if t in ['video', 'image', 'diagram'])
            audio_content = sum(1 for t in content_types if t in ['audio', 'podcast'])
            text_content = sum(1 for t in content_types if t in ['text', 'article', 'document'])
            interactive_content = sum(1 for t in content_types if t in ['simulation', 'game', 'exercise'])
            
            total_content = len(content_types)
            if total_content > 0:
                style_indicators[LearningStyle.VISUAL] = visual_content / total_content
                style_indicators[LearningStyle.AUDITORY] = audio_content / total_content
                style_indicators[LearningStyle.READING_WRITING] = text_content / total_content
                style_indicators[LearningStyle.KINESTHETIC] = interactive_content / total_content
        
        # Analyze activity preferences
        if learning_activities:
            activity_types = [a.get('type', '') for a in learning_activities]
            
            hands_on = sum(1 for t in activity_types if t in ['lab', 'project', 'simulation'])
            reading = sum(1 for t in activity_types if t in ['reading', 'research'])
            
            if activity_types:
                style_indicators[LearningStyle.KINESTHETIC] += hands_on / len(activity_types) * 0.5
                style_indicators[LearningStyle.READING_WRITING] += reading / len(activity_types) * 0.5
        
        # Determine dominant style
        max_score = max(style_indicators.values())
        if max_score < 0.4:  # No clear preference
            return LearningStyle.MULTIMODAL
        
        dominant_style = max(style_indicators.items(), key=lambda x: x[1])[0]
        return dominant_style
    
    async def _calculate_engagement_metrics(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed engagement metrics"""
        metrics = {}
        
        # Time-based engagement
        if 'total_time_spent' in features:
            metrics['time_engagement'] = min(features['total_time_spent'] / 3600, 2.0) / 2.0
        
        # Activity engagement
        if 'completion_rate' in features:
            metrics['activity_engagement'] = features['completion_rate']
        
        # Performance engagement
        if 'avg_performance' in features:
            metrics['performance_engagement'] = features['avg_performance']
        
        # Interaction engagement
        if 'click_rate' in features:
            metrics['interaction_engagement'] = min(features['click_rate'], 1.0)
        
        # Consistency engagement
        if 'session_duration_variance' in features and 'avg_session_duration' in features:
            consistency = 1 / (1 + features['session_duration_variance'] / max(features['avg_session_duration'], 1))
            metrics['consistency_engagement'] = consistency
        
        # Content diversity engagement
        if 'content_types_accessed' in features:
            metrics['diversity_engagement'] = min(features['content_types_accessed'] / 5, 1.0)
        
        # Overall engagement (weighted average)
        if metrics:
            weights = {
                'time_engagement': 0.2,
                'activity_engagement': 0.25,
                'performance_engagement': 0.2,
                'interaction_engagement': 0.15,
                'consistency_engagement': 0.1,
                'diversity_engagement': 0.1
            }
            
            weighted_sum = sum(metrics.get(key, 0) * weight for key, weight in weights.items())
            total_weight = sum(weight for key, weight in weights.items() if key in metrics)
            
            if total_weight > 0:
                metrics['overall_engagement'] = weighted_sum / total_weight
        
        return metrics
    
    async def _analyze_motivation_indicators(self,
                                           features: Dict[str, Any],
                                           learning_activities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze motivation indicators using behavioral economics principles"""
        indicators = {}
        
        # Intrinsic motivation indicators
        if 'content_types_accessed' in features and features['content_types_accessed'] > 3:
            indicators['curiosity'] = min(features['content_types_accessed'] / 5, 1.0)
        
        if 'avg_session_duration' in features and features['avg_session_duration'] > 1800:
            indicators['persistence'] = min(features['avg_session_duration'] / 3600, 1.0)
        
        # Extrinsic motivation indicators
        if 'completion_rate' in features:
            indicators['achievement_orientation'] = features['completion_rate']
        
        if 'performance_trend' in features and features['performance_trend'] > 0:
            indicators['progress_motivation'] = min(features['performance_trend'], 1.0)
        
        # Self-efficacy indicators
        if 'avg_performance' in features and 'performance_variance' in features:
            consistency = 1 / (1 + features['performance_variance'])
            indicators['self_efficacy'] = (features['avg_performance'] + consistency) / 2
        
        # Social motivation (if available)
        if learning_activities:
            social_activities = sum(1 for a in learning_activities if a.get('social_component', False))
            if social_activities > 0:
                indicators['social_motivation'] = social_activities / len(learning_activities)
        
        # Goal orientation
        if 'session_frequency' in features and features['session_frequency'] > 5:
            indicators['goal_commitment'] = min(features['session_frequency'] / 10, 1.0)
        
        return indicators
    
    async def _detect_behavioral_anomalies(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Detect behavioral anomalies using statistical methods"""
        anomalies = {}
        
        # Simple statistical anomaly detection
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        if numeric_features:
            values = list(numeric_features.values())
            
            # Z-score based anomaly detection
            if len(values) > 1:
                z_scores = np.abs(stats.zscore(values)) if SCIPY_AVAILABLE else [0] * len(values)
                
                for i, (feature_name, value) in enumerate(numeric_features.items()):
                    if i < len(z_scores):
                        anomaly_score = min(z_scores[i] / 3, 1.0)  # Normalize to 0-1
                        if anomaly_score > 0.5:  # Threshold for anomaly
                            anomalies[feature_name] = anomaly_score
        
        # Domain-specific anomaly detection
        
        # Sudden performance drop
        if 'performance_trend' in features and features['performance_trend'] < -0.3:
            anomalies['performance_drop'] = abs(features['performance_trend'])
        
        # Unusual session patterns
        if ('avg_session_duration' in features and 
            'session_duration_variance' in features and
            features['session_duration_variance'] > features['avg_session_duration']):
            anomalies['session_inconsistency'] = min(
                features['session_duration_variance'] / features['avg_session_duration'], 1.0
            )
        
        # Engagement anomalies
        if 'completion_rate' in features and features['completion_rate'] < 0.1:
            anomalies['low_completion'] = 1 - features['completion_rate']
        
        return anomalies
    
    async def _generate_personalization_insights(self,
                                               patterns: List[BehaviorPattern],
                                               engagement_metrics: Dict[str, float],
                                               learning_style: LearningStyle) -> Dict[str, Any]:
        """Generate personalization insights for adaptive learning"""
        insights = {
            'content_recommendations': [],
            'interaction_preferences': [],
            'timing_recommendations': [],
            'difficulty_adjustments': [],
            'motivation_strategies': []
        }
        
        # Content recommendations based on learning style
        if learning_style == LearningStyle.VISUAL:
            insights['content_recommendations'].extend([
                'Prioritize visual content (videos, diagrams, infographics)',
                'Use mind maps and visual organizers',
                'Include interactive visualizations'
            ])
        elif learning_style == LearningStyle.AUDITORY:
            insights['content_recommendations'].extend([
                'Include audio explanations and podcasts',
                'Use discussion-based learning',
                'Provide verbal instructions and feedback'
            ])
        elif learning_style == LearningStyle.KINESTHETIC:
            insights['content_recommendations'].extend([
                'Include hands-on activities and simulations',
                'Use interactive exercises and labs',
                'Provide opportunities for physical manipulation'
            ])
        elif learning_style == LearningStyle.READING_WRITING:
            insights['content_recommendations'].extend([
                'Provide text-based materials and articles',
                'Include writing assignments and note-taking',
                'Use structured reading activities'
            ])
        
        # Interaction preferences based on patterns
        for pattern in patterns:
            if pattern.pattern_id == 'high_engagement':
                insights['interaction_preferences'].append('Provide challenging, interactive content')
            elif pattern.pattern_id == 'exploration':
                insights['interaction_preferences'].append('Offer multiple learning paths and choices')
            elif pattern.pattern_id == 'focused_learning':
                insights['interaction_preferences'].append('Minimize distractions and interruptions')
        
        # Timing recommendations
        if engagement_metrics.get('consistency_engagement', 0) > 0.7:
            insights['timing_recommendations'].append('Maintain regular learning schedule')
        else:
            insights['timing_recommendations'].append('Provide flexible scheduling options')
        
        # Difficulty adjustments
        if engagement_metrics.get('performance_engagement', 0.5) > 0.8:
            insights['difficulty_adjustments'].append('Increase challenge level gradually')
        elif engagement_metrics.get('performance_engagement', 0.5) < 0.4:
            insights['difficulty_adjustments'].append('Provide additional support and scaffolding')
        
        # Motivation strategies
        if engagement_metrics.get('overall_engagement', 0.5) < 0.5:
            insights['motivation_strategies'].extend([
                'Implement gamification elements',
                'Provide frequent positive feedback',
                'Set achievable short-term goals'
            ])
        
        return insights
    
    async def _analyze_behavior_trends(self, behavioral_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Analyze trends in behavior over time"""
        trends = {}
        
        # Session duration trends
        if 'session_durations' in behavioral_data:
            sessions = behavioral_data['session_durations']
            if len(sessions) > 1:
                trends['session_duration'] = sessions
                trends['session_duration_trend'] = [self._calculate_trend(sessions[:i+1]) 
                                                   for i in range(1, len(sessions))]
        
        # Performance trends
        if 'performance_history' in behavioral_data:
            performance = behavioral_data['performance_history']
            if len(performance) > 1:
                trends['performance'] = performance
                trends['performance_trend'] = [self._calculate_trend(performance[:i+1]) 
                                             for i in range(1, len(performance))]
        
        # Engagement trends (if available)
        if 'engagement_history' in behavioral_data:
            engagement = behavioral_data['engagement_history']
            if len(engagement) > 1:
                trends['engagement'] = engagement
                trends['engagement_trend'] = [self._calculate_trend(engagement[:i+1]) 
                                            for i in range(1, len(engagement))]
        
        return trends
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in time series data"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression slope
        if SCIPY_AVAILABLE:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        else:
            # Manual calculation
            n = len(data)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope


class UserBehaviorModeler:
    """
    🎭 USER BEHAVIOR MODELER

    Advanced behavior modeling using Hidden Markov Models and recurrent neural networks.
    """

    def __init__(self):
        self.behavior_models = {}
        self.state_transitions = {}
        logger.info("User Behavior Modeler initialized")

    async def build_behavior_model(self,
                                 user_id: str,
                                 behavior_sequence: List[Dict[str, Any]],
                                 model_type: str = 'hmm') -> Dict[str, Any]:
        """
        Build behavior model for user

        Args:
            user_id: User identifier
            behavior_sequence: Sequence of behavior observations
            model_type: Type of model ('hmm', 'markov_chain')

        Returns:
            Dict: Behavior model parameters and predictions
        """
        try:
            if model_type == 'hmm':
                return await self._build_hmm_model(user_id, behavior_sequence)
            elif model_type == 'markov_chain':
                return await self._build_markov_chain_model(user_id, behavior_sequence)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            logger.error(f"Error building behavior model: {e}")
            raise QuantumEngineError(f"Behavior modeling failed: {e}")

    async def _build_hmm_model(self, user_id: str, behavior_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Hidden Markov Model for behavior prediction"""
        # Simplified HMM implementation
        # In production, you'd use libraries like hmmlearn

        # Extract behavior states
        states = [self._extract_behavior_state(obs) for obs in behavior_sequence]
        unique_states = list(set(states))
        n_states = len(unique_states)

        if n_states < 2:
            return {'error': 'Insufficient state diversity for HMM'}

        # Calculate transition probabilities
        transition_matrix = np.zeros((n_states, n_states))
        state_to_idx = {state: i for i, state in enumerate(unique_states)}

        for i in range(len(states) - 1):
            current_state = state_to_idx[states[i]]
            next_state = state_to_idx[states[i + 1]]
            transition_matrix[current_state][next_state] += 1

        # Normalize transition matrix
        for i in range(n_states):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum

        # Calculate emission probabilities (simplified)
        emission_matrix = np.random.rand(n_states, 3)  # 3 observable features
        emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

        # Initial state probabilities
        initial_probs = np.zeros(n_states)
        if states:
            initial_state = state_to_idx[states[0]]
            initial_probs[initial_state] = 1.0

        model = {
            'model_type': 'hmm',
            'states': unique_states,
            'transition_matrix': transition_matrix.tolist(),
            'emission_matrix': emission_matrix.tolist(),
            'initial_probabilities': initial_probs.tolist(),
            'model_accuracy': self._estimate_model_accuracy(states, transition_matrix, state_to_idx),
            'next_state_prediction': self._predict_next_state(states[-1], transition_matrix, unique_states, state_to_idx)
        }

        self.behavior_models[user_id] = model
        return model

    async def _build_markov_chain_model(self, user_id: str, behavior_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Markov Chain model for behavior prediction"""
        # Extract behavior states
        states = [self._extract_behavior_state(obs) for obs in behavior_sequence]
        unique_states = list(set(states))

        # Build transition matrix
        transitions = {}
        for state in unique_states:
            transitions[state] = {}
            for next_state in unique_states:
                transitions[state][next_state] = 0

        # Count transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transitions[current_state][next_state] += 1

        # Normalize to probabilities
        for state in unique_states:
            total = sum(transitions[state].values())
            if total > 0:
                for next_state in unique_states:
                    transitions[state][next_state] /= total

        model = {
            'model_type': 'markov_chain',
            'states': unique_states,
            'transition_probabilities': transitions,
            'steady_state': self._calculate_steady_state(transitions, unique_states),
            'next_state_prediction': self._predict_next_state_markov(states[-1], transitions)
        }

        self.behavior_models[user_id] = model
        return model

    def _extract_behavior_state(self, observation: Dict[str, Any]) -> str:
        """Extract behavior state from observation"""
        # Simplified state extraction
        engagement = observation.get('engagement_score', 0.5)
        performance = observation.get('performance_score', 0.5)

        if engagement > 0.7 and performance > 0.7:
            return 'high_performance'
        elif engagement > 0.5 and performance > 0.5:
            return 'moderate_performance'
        elif engagement < 0.3 or performance < 0.3:
            return 'struggling'
        else:
            return 'average'

    def _estimate_model_accuracy(self, states: List[str], transition_matrix: np.ndarray,
                                state_to_idx: Dict[str, int]) -> float:
        """Estimate model accuracy using cross-validation"""
        if len(states) < 10:
            return 0.5  # Insufficient data

        # Simple accuracy estimation
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(states) - 1):
            current_state_idx = state_to_idx[states[i]]
            actual_next_state = states[i + 1]

            # Predict next state
            predicted_state_idx = np.argmax(transition_matrix[current_state_idx])
            predicted_state = list(state_to_idx.keys())[predicted_state_idx]

            if predicted_state == actual_next_state:
                correct_predictions += 1
            total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0.5

    def _predict_next_state(self, current_state: str, transition_matrix: np.ndarray,
                          unique_states: List[str], state_to_idx: Dict[str, int]) -> Dict[str, float]:
        """Predict next state probabilities"""
        if current_state not in state_to_idx:
            return {state: 1.0/len(unique_states) for state in unique_states}

        current_idx = state_to_idx[current_state]
        next_state_probs = transition_matrix[current_idx]

        return {unique_states[i]: float(prob) for i, prob in enumerate(next_state_probs)}

    def _predict_next_state_markov(self, current_state: str, transitions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Predict next state using Markov chain"""
        if current_state in transitions:
            return transitions[current_state]
        else:
            # Uniform distribution if state not seen
            states = list(transitions.keys())
            return {state: 1.0/len(states) for state in states}

    def _calculate_steady_state(self, transitions: Dict[str, Dict[str, float]], states: List[str]) -> Dict[str, float]:
        """Calculate steady-state probabilities"""
        n = len(states)
        if n == 0:
            return {}

        # Convert to matrix
        matrix = np.zeros((n, n))
        state_to_idx = {state: i for i, state in enumerate(states)}

        for i, state in enumerate(states):
            for j, next_state in enumerate(states):
                matrix[i][j] = transitions[state].get(next_state, 0)

        # Find steady state (eigenvector with eigenvalue 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            steady_state_idx = np.argmin(np.abs(eigenvalues - 1))
            steady_state_vector = np.real(eigenvectors[:, steady_state_idx])
            steady_state_vector = steady_state_vector / np.sum(steady_state_vector)

            return {states[i]: float(prob) for i, prob in enumerate(steady_state_vector)}
        except:
            # Fallback to uniform distribution
            return {state: 1.0/n for state in states}


class EngagementAnalytics:
    """
    📊 ENGAGEMENT ANALYTICS

    Advanced engagement analytics with clustering analysis and trend detection.
    """

    def __init__(self):
        self.engagement_clusters = {}
        self.engagement_models = {}
        logger.info("Engagement Analytics initialized")

    async def analyze_engagement_patterns(self,
                                        user_data: List[Dict[str, Any]],
                                        clustering_method: str = 'kmeans') -> Dict[str, Any]:
        """
        Analyze engagement patterns across users

        Args:
            user_data: List of user engagement data
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')

        Returns:
            Dict: Engagement pattern analysis results
        """
        try:
            if not SKLEARN_AVAILABLE:
                return await self._simple_engagement_analysis(user_data)

            # Extract engagement features
            features = await self._extract_engagement_features(user_data)

            # Perform clustering
            clusters = await self._perform_engagement_clustering(features, clustering_method)

            # Analyze cluster characteristics
            cluster_analysis = await self._analyze_engagement_clusters(features, clusters, user_data)

            # Identify engagement patterns
            patterns = await self._identify_engagement_patterns(cluster_analysis)

            # Generate insights
            insights = await self._generate_engagement_insights(patterns, cluster_analysis)

            return {
                'clustering_method': clustering_method,
                'n_clusters': len(set(clusters)) if clusters else 0,
                'cluster_assignments': clusters,
                'cluster_analysis': cluster_analysis,
                'engagement_patterns': patterns,
                'insights': insights,
                'feature_importance': await self._calculate_feature_importance(features)
            }

        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {e}")
            raise QuantumEngineError(f"Engagement analysis failed: {e}")

    async def _extract_engagement_features(self, user_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for engagement clustering"""
        features = []

        for user in user_data:
            user_features = [
                user.get('total_time_spent', 0) / 3600,  # Hours
                user.get('session_frequency', 0),
                user.get('completion_rate', 0),
                user.get('avg_performance', 0),
                user.get('content_diversity', 0),
                user.get('interaction_rate', 0),
                user.get('consistency_score', 0),
                user.get('progress_rate', 0)
            ]
            features.append(user_features)

        return np.array(features)

    async def _perform_engagement_clustering(self, features: np.ndarray, method: str) -> List[int]:
        """Perform clustering on engagement features"""
        if len(features) < 2:
            return [0] * len(features)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == 'kmeans':
            # Determine optimal number of clusters
            n_clusters = min(5, len(features) // 2)
            if n_clusters < 2:
                n_clusters = 2

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            clusters = dbscan.fit_predict(features_scaled)

        else:
            # Fallback to simple clustering
            return await self._simple_clustering(features)

        return clusters.tolist()

    async def _analyze_engagement_clusters(self, features: np.ndarray, clusters: List[int],
                                         user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of engagement clusters"""
        cluster_analysis = {}
        unique_clusters = list(set(clusters))

        feature_names = [
            'time_spent', 'session_frequency', 'completion_rate', 'avg_performance',
            'content_diversity', 'interaction_rate', 'consistency_score', 'progress_rate'
        ]

        for cluster_id in unique_clusters:
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_features = features[cluster_indices]
            cluster_users = [user_data[i] for i in cluster_indices]

            # Calculate cluster statistics
            cluster_stats = {
                'size': len(cluster_indices),
                'percentage': len(cluster_indices) / len(user_data) * 100,
                'feature_means': np.mean(cluster_features, axis=0).tolist(),
                'feature_stds': np.std(cluster_features, axis=0).tolist(),
                'user_ids': [user.get('user_id', f'user_{i}') for i, user in enumerate(cluster_users)]
            }

            # Characterize cluster
            cluster_stats['characteristics'] = await self._characterize_cluster(
                cluster_stats['feature_means'], feature_names
            )

            cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats

        return cluster_analysis

    async def _characterize_cluster(self, feature_means: List[float], feature_names: List[str]) -> Dict[str, str]:
        """Characterize cluster based on feature means"""
        characteristics = {}

        # Time engagement
        if feature_means[0] > 2.0:  # > 2 hours
            characteristics['time_engagement'] = 'high'
        elif feature_means[0] > 1.0:
            characteristics['time_engagement'] = 'moderate'
        else:
            characteristics['time_engagement'] = 'low'

        # Session frequency
        if feature_means[1] > 10:
            characteristics['frequency'] = 'very_frequent'
        elif feature_means[1] > 5:
            characteristics['frequency'] = 'frequent'
        else:
            characteristics['frequency'] = 'infrequent'

        # Performance
        if feature_means[3] > 0.8:
            characteristics['performance'] = 'high'
        elif feature_means[3] > 0.6:
            characteristics['performance'] = 'moderate'
        else:
            characteristics['performance'] = 'low'

        # Overall engagement level
        overall_score = np.mean(feature_means)
        if overall_score > 0.7:
            characteristics['overall_engagement'] = 'highly_engaged'
        elif overall_score > 0.5:
            characteristics['overall_engagement'] = 'moderately_engaged'
        else:
            characteristics['overall_engagement'] = 'low_engagement'

        return characteristics

    async def _identify_engagement_patterns(self, cluster_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify common engagement patterns"""
        patterns = []

        for cluster_id, cluster_data in cluster_analysis.items():
            characteristics = cluster_data['characteristics']

            pattern = {
                'pattern_id': f"pattern_{cluster_id}",
                'cluster_id': cluster_id,
                'pattern_type': characteristics.get('overall_engagement', 'unknown'),
                'size': cluster_data['size'],
                'percentage': cluster_data['percentage'],
                'key_characteristics': characteristics,
                'typical_behaviors': await self._describe_typical_behaviors(characteristics)
            }

            patterns.append(pattern)

        return patterns

    async def _describe_typical_behaviors(self, characteristics: Dict[str, str]) -> List[str]:
        """Describe typical behaviors for engagement pattern"""
        behaviors = []

        engagement_level = characteristics.get('overall_engagement', 'unknown')

        if engagement_level == 'highly_engaged':
            behaviors.extend([
                'Spends significant time in learning sessions',
                'Maintains consistent learning schedule',
                'Achieves high completion rates',
                'Demonstrates strong performance'
            ])
        elif engagement_level == 'moderately_engaged':
            behaviors.extend([
                'Regular but moderate learning activity',
                'Good completion rates with room for improvement',
                'Consistent but not intensive engagement'
            ])
        else:
            behaviors.extend([
                'Irregular learning patterns',
                'Low completion rates',
                'May need additional motivation and support'
            ])

        # Add specific behavioral indicators
        if characteristics.get('frequency') == 'very_frequent':
            behaviors.append('Very frequent learning sessions')

        if characteristics.get('performance') == 'high':
            behaviors.append('Consistently high performance scores')

        return behaviors

    async def _generate_engagement_insights(self, patterns: List[Dict[str, Any]],
                                          cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from engagement analysis"""
        insights = []

        # Overall engagement distribution
        total_users = sum(pattern['size'] for pattern in patterns)
        highly_engaged = sum(pattern['size'] for pattern in patterns
                           if pattern['pattern_type'] == 'highly_engaged')

        if highly_engaged / total_users < 0.3:
            insights.append("Less than 30% of users are highly engaged - consider engagement interventions")

        # Identify largest engagement segment
        largest_pattern = max(patterns, key=lambda p: p['size'])
        insights.append(f"Largest engagement segment: {largest_pattern['pattern_type']} "
                       f"({largest_pattern['percentage']:.1f}% of users)")

        # Performance vs engagement insights
        for pattern in patterns:
            characteristics = pattern['key_characteristics']
            if (characteristics.get('time_engagement') == 'high' and
                characteristics.get('performance') == 'low'):
                insights.append(f"Cluster {pattern['cluster_id']}: High time investment but low performance - "
                               "may need difficulty adjustment or learning support")

        # Frequency insights
        infrequent_users = sum(pattern['size'] for pattern in patterns
                             if pattern['key_characteristics'].get('frequency') == 'infrequent')
        if infrequent_users / total_users > 0.4:
            insights.append("Over 40% of users have infrequent learning sessions - "
                           "consider habit formation interventions")

        return insights

    async def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for engagement clustering"""
        feature_names = [
            'time_spent', 'session_frequency', 'completion_rate', 'avg_performance',
            'content_diversity', 'interaction_rate', 'consistency_score', 'progress_rate'
        ]

        # Simple variance-based importance
        feature_variances = np.var(features, axis=0)
        total_variance = np.sum(feature_variances)

        if total_variance > 0:
            importance_scores = feature_variances / total_variance
        else:
            importance_scores = np.ones(len(feature_names)) / len(feature_names)

        return {name: float(score) for name, score in zip(feature_names, importance_scores)}

    async def _simple_engagement_analysis(self, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple engagement analysis when sklearn is not available"""
        if not user_data:
            return {'error': 'No user data provided'}

        # Calculate basic statistics
        engagement_scores = [user.get('overall_engagement', 0.5) for user in user_data]

        high_engagement = sum(1 for score in engagement_scores if score > 0.7)
        moderate_engagement = sum(1 for score in engagement_scores if 0.4 <= score <= 0.7)
        low_engagement = sum(1 for score in engagement_scores if score < 0.4)

        return {
            'clustering_method': 'simple_thresholding',
            'n_clusters': 3,
            'engagement_distribution': {
                'high_engagement': {'count': high_engagement, 'percentage': high_engagement / len(user_data) * 100},
                'moderate_engagement': {'count': moderate_engagement, 'percentage': moderate_engagement / len(user_data) * 100},
                'low_engagement': {'count': low_engagement, 'percentage': low_engagement / len(user_data) * 100}
            },
            'insights': [
                f"Average engagement score: {np.mean(engagement_scores):.2f}",
                f"Engagement standard deviation: {np.std(engagement_scores):.2f}",
                f"Percentage of highly engaged users: {high_engagement / len(user_data) * 100:.1f}%"
            ]
        }

    async def _simple_clustering(self, features: np.ndarray) -> List[int]:
        """Simple clustering when advanced methods are not available"""
        # Use overall engagement score for simple clustering
        overall_scores = np.mean(features, axis=1)

        clusters = []
        for score in overall_scores:
            if score > 0.7:
                clusters.append(0)  # High engagement
            elif score > 0.4:
                clusters.append(1)  # Moderate engagement
            else:
                clusters.append(2)  # Low engagement

        return clusters


class PersonalizationInsights:
    """
    🎯 PERSONALIZATION INSIGHTS

    Advanced personalization insights through collaborative filtering and matrix factorization.
    """

    def __init__(self):
        self.user_item_matrix = None
        self.personalization_models = {}
        logger.info("Personalization Insights initialized")

    async def generate_personalization_insights(self,
                                              user_profiles: List[UserBehaviorProfile],
                                              content_interactions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate personalization insights using collaborative filtering

        Args:
            user_profiles: List of user behavior profiles
            content_interactions: User-content interaction data

        Returns:
            Dict: Personalization insights and recommendations
        """
        try:
            # Build user-item interaction matrix
            interaction_matrix = await self._build_interaction_matrix(content_interactions)

            # Perform collaborative filtering
            recommendations = await self._collaborative_filtering(interaction_matrix, user_profiles)

            # Analyze user similarities
            user_similarities = await self._analyze_user_similarities(user_profiles)

            # Generate content recommendations
            content_recommendations = await self._generate_content_recommendations(
                user_profiles, interaction_matrix, user_similarities
            )

            # Identify personalization opportunities
            personalization_opportunities = await self._identify_personalization_opportunities(
                user_profiles, content_interactions
            )

            return {
                'user_similarities': user_similarities,
                'content_recommendations': content_recommendations,
                'personalization_opportunities': personalization_opportunities,
                'collaborative_filtering_results': recommendations,
                'interaction_matrix_stats': await self._analyze_interaction_matrix(interaction_matrix)
            }

        except Exception as e:
            logger.error(f"Error generating personalization insights: {e}")
            raise QuantumEngineError(f"Personalization insights generation failed: {e}")

    async def _build_interaction_matrix(self, content_interactions: Dict[str, List[Dict[str, Any]]]) -> np.ndarray:
        """Build user-item interaction matrix"""
        # Get all users and content items
        all_users = list(content_interactions.keys())
        all_content = set()

        for user_interactions in content_interactions.values():
            for interaction in user_interactions:
                all_content.add(interaction.get('content_id', ''))

        all_content = list(all_content)

        # Build matrix
        matrix = np.zeros((len(all_users), len(all_content)))

        for i, user_id in enumerate(all_users):
            user_interactions = content_interactions[user_id]
            for interaction in user_interactions:
                content_id = interaction.get('content_id', '')
                if content_id in all_content:
                    j = all_content.index(content_id)
                    # Use engagement score or rating as interaction strength
                    matrix[i, j] = interaction.get('engagement_score', 1.0)

        self.user_item_matrix = matrix
        return matrix

    async def _collaborative_filtering(self, interaction_matrix: np.ndarray,
                                     user_profiles: List[UserBehaviorProfile]) -> Dict[str, Any]:
        """Perform collaborative filtering for recommendations"""
        if interaction_matrix.size == 0:
            return {'error': 'Empty interaction matrix'}

        # Simple user-based collaborative filtering
        # Calculate user similarities using cosine similarity
        user_similarities = np.zeros((interaction_matrix.shape[0], interaction_matrix.shape[0]))

        for i in range(interaction_matrix.shape[0]):
            for j in range(interaction_matrix.shape[0]):
                if i != j:
                    user_i = interaction_matrix[i]
                    user_j = interaction_matrix[j]

                    # Cosine similarity
                    dot_product = np.dot(user_i, user_j)
                    norm_i = np.linalg.norm(user_i)
                    norm_j = np.linalg.norm(user_j)

                    if norm_i > 0 and norm_j > 0:
                        user_similarities[i, j] = dot_product / (norm_i * norm_j)

        # Generate recommendations for each user
        recommendations = {}
        for i in range(interaction_matrix.shape[0]):
            user_recommendations = await self._generate_user_recommendations(
                i, interaction_matrix, user_similarities
            )
            recommendations[f'user_{i}'] = user_recommendations

        return {
            'user_similarities': user_similarities.tolist(),
            'recommendations': recommendations,
            'matrix_sparsity': np.count_nonzero(interaction_matrix) / interaction_matrix.size
        }

    async def _generate_user_recommendations(self, user_idx: int, interaction_matrix: np.ndarray,
                                           user_similarities: np.ndarray) -> List[Dict[str, Any]]:
        """Generate recommendations for a specific user"""
        user_interactions = interaction_matrix[user_idx]
        similar_users = np.argsort(user_similarities[user_idx])[::-1][:5]  # Top 5 similar users

        recommendations = []

        for content_idx in range(interaction_matrix.shape[1]):
            if user_interactions[content_idx] == 0:  # User hasn't interacted with this content
                # Calculate predicted rating based on similar users
                weighted_sum = 0
                similarity_sum = 0

                for similar_user_idx in similar_users:
                    if interaction_matrix[similar_user_idx, content_idx] > 0:
                        similarity = user_similarities[user_idx, similar_user_idx]
                        rating = interaction_matrix[similar_user_idx, content_idx]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)

                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations.append({
                        'content_idx': content_idx,
                        'predicted_rating': predicted_rating,
                        'confidence': similarity_sum / len(similar_users)
                    })

        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:10]  # Top 10 recommendations

    async def _analyze_user_similarities(self, user_profiles: List[UserBehaviorProfile]) -> Dict[str, Any]:
        """Analyze similarities between users"""
        if len(user_profiles) < 2:
            return {'error': 'Insufficient users for similarity analysis'}

        similarities = {}

        for i, profile_i in enumerate(user_profiles):
            for j, profile_j in enumerate(user_profiles):
                if i < j:  # Avoid duplicate pairs
                    similarity = await self._calculate_profile_similarity(profile_i, profile_j)
                    similarities[f'{profile_i.user_id}_{profile_j.user_id}'] = similarity

        # Find most similar user pairs
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return {
            'user_similarities': similarities,
            'most_similar_pairs': sorted_similarities[:5],
            'average_similarity': np.mean(list(similarities.values())),
            'similarity_distribution': await self._analyze_similarity_distribution(list(similarities.values()))
        }

    async def _calculate_profile_similarity(self, profile1: UserBehaviorProfile,
                                          profile2: UserBehaviorProfile) -> float:
        """Calculate similarity between two user profiles"""
        similarity_factors = []

        # Learning style similarity
        if profile1.learning_style == profile2.learning_style:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Engagement level similarity
        engagement_levels = {
            EngagementLevel.VERY_HIGH: 5,
            EngagementLevel.HIGH: 4,
            EngagementLevel.MODERATE: 3,
            EngagementLevel.LOW: 2,
            EngagementLevel.VERY_LOW: 1
        }

        eng1 = engagement_levels.get(profile1.engagement_level, 3)
        eng2 = engagement_levels.get(profile2.engagement_level, 3)
        engagement_similarity = 1 - abs(eng1 - eng2) / 4
        similarity_factors.append(engagement_similarity)

        # Behavior pattern similarity
        patterns1 = set(p.pattern_type for p in profile1.behavior_patterns)
        patterns2 = set(p.pattern_type for p in profile2.behavior_patterns)

        if patterns1 and patterns2:
            pattern_similarity = len(patterns1.intersection(patterns2)) / len(patterns1.union(patterns2))
            similarity_factors.append(pattern_similarity)

        # Engagement metrics similarity
        metrics1 = profile1.engagement_metrics
        metrics2 = profile2.engagement_metrics

        common_metrics = set(metrics1.keys()).intersection(set(metrics2.keys()))
        if common_metrics:
            metric_similarities = []
            for metric in common_metrics:
                diff = abs(metrics1[metric] - metrics2[metric])
                metric_similarities.append(1 - diff)
            similarity_factors.append(np.mean(metric_similarities))

        return np.mean(similarity_factors) if similarity_factors else 0.0

    async def _analyze_similarity_distribution(self, similarities: List[float]) -> Dict[str, float]:
        """Analyze distribution of user similarities"""
        if not similarities:
            return {}

        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75))
        }

    async def _generate_content_recommendations(self,
                                              user_profiles: List[UserBehaviorProfile],
                                              interaction_matrix: np.ndarray,
                                              user_similarities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate content recommendations for each user"""
        recommendations = {}

        for i, profile in enumerate(user_profiles):
            user_recommendations = []

            # Learning style based recommendations
            if profile.learning_style == LearningStyle.VISUAL:
                user_recommendations.extend([
                    'video_content', 'infographics', 'interactive_visualizations'
                ])
            elif profile.learning_style == LearningStyle.AUDITORY:
                user_recommendations.extend([
                    'audio_content', 'podcasts', 'discussion_forums'
                ])
            elif profile.learning_style == LearningStyle.KINESTHETIC:
                user_recommendations.extend([
                    'hands_on_activities', 'simulations', 'interactive_exercises'
                ])
            elif profile.learning_style == LearningStyle.READING_WRITING:
                user_recommendations.extend([
                    'text_articles', 'written_exercises', 'note_taking_tools'
                ])

            # Engagement level based recommendations
            if profile.engagement_level in [EngagementLevel.LOW, EngagementLevel.VERY_LOW]:
                user_recommendations.extend([
                    'gamified_content', 'short_form_content', 'achievement_systems'
                ])
            elif profile.engagement_level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
                user_recommendations.extend([
                    'advanced_content', 'challenge_problems', 'peer_collaboration'
                ])

            # Behavior pattern based recommendations
            for pattern in profile.behavior_patterns:
                if pattern.pattern_type == 'exploration':
                    user_recommendations.append('diverse_content_library')
                elif pattern.pattern_type == 'focused_learning':
                    user_recommendations.append('deep_dive_content')
                elif pattern.pattern_type == 'struggling':
                    user_recommendations.extend(['remedial_content', 'tutoring_support'])

            recommendations[profile.user_id] = list(set(user_recommendations))  # Remove duplicates

        return recommendations

    async def _identify_personalization_opportunities(self,
                                                    user_profiles: List[UserBehaviorProfile],
                                                    content_interactions: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify opportunities for better personalization"""
        opportunities = []

        # Analyze engagement gaps
        low_engagement_users = [p for p in user_profiles
                               if p.engagement_level in [EngagementLevel.LOW, EngagementLevel.VERY_LOW]]

        if len(low_engagement_users) > len(user_profiles) * 0.3:
            opportunities.append({
                'type': 'engagement_improvement',
                'description': f'{len(low_engagement_users)} users have low engagement',
                'recommendation': 'Implement targeted engagement interventions',
                'priority': 'high',
                'affected_users': len(low_engagement_users)
            })

        # Analyze learning style distribution
        style_distribution = {}
        for profile in user_profiles:
            style = profile.learning_style.value
            style_distribution[style] = style_distribution.get(style, 0) + 1

        # Check for underserved learning styles
        total_users = len(user_profiles)
        for style, count in style_distribution.items():
            if count / total_users > 0.4:  # Over 40% have same learning style
                opportunities.append({
                    'type': 'content_diversification',
                    'description': f'Over 40% of users prefer {style} learning style',
                    'recommendation': f'Expand content variety for {style} learners',
                    'priority': 'medium',
                    'affected_users': count
                })

        # Analyze content interaction patterns
        if content_interactions:
            interaction_counts = {}
            for user_interactions in content_interactions.values():
                for interaction in user_interactions:
                    content_type = interaction.get('content_type', 'unknown')
                    interaction_counts[content_type] = interaction_counts.get(content_type, 0) + 1

            # Identify underutilized content types
            total_interactions = sum(interaction_counts.values())
            for content_type, count in interaction_counts.items():
                if count / total_interactions < 0.05:  # Less than 5% of interactions
                    opportunities.append({
                        'type': 'content_promotion',
                        'description': f'{content_type} content is underutilized',
                        'recommendation': f'Promote {content_type} content or improve its discoverability',
                        'priority': 'low',
                        'affected_users': 'all'
                    })

        return opportunities

    async def _analyze_interaction_matrix(self, interaction_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze interaction matrix statistics"""
        if interaction_matrix.size == 0:
            return {'error': 'Empty interaction matrix'}

        total_elements = interaction_matrix.size
        non_zero_elements = np.count_nonzero(interaction_matrix)
        sparsity = 1 - (non_zero_elements / total_elements)

        return {
            'matrix_shape': interaction_matrix.shape,
            'total_elements': total_elements,
            'non_zero_elements': non_zero_elements,
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'mean_interaction_strength': float(np.mean(interaction_matrix[interaction_matrix > 0])) if non_zero_elements > 0 else 0,
            'max_interaction_strength': float(np.max(interaction_matrix)),
            'users_with_interactions': int(np.sum(np.any(interaction_matrix > 0, axis=1))),
            'content_with_interactions': int(np.sum(np.any(interaction_matrix > 0, axis=0)))
        }
