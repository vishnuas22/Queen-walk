"""
User Preference Modeling and Prediction Engine

Advanced preference modeling system that learns, predicts, and adapts to user
preferences using machine learning, behavioral analysis, and quantum-enhanced
pattern recognition for revolutionary personalized experiences.

🎯 PREFERENCE ENGINE CAPABILITIES:
- Dynamic preference learning from user interactions
- Predictive preference modeling using advanced algorithms
- Context-aware preference adaptation
- Multi-dimensional preference space modeling
- Temporal preference evolution tracking
- Cross-domain preference transfer learning

Author: MasterX AI Team - Personalization Division
Version: 1.0 - Phase 9 Advanced Personalization Engine
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import random
import math

# Import personalization components
from .user_profiling import LearningDNA, LearningStyle, CognitivePattern

# Try to import advanced libraries with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def mean(array):
            return sum(array) / len(array) if array else 0
        
        @staticmethod
        def std(array):
            if not array:
                return 0
            mean_val = sum(array) / len(array)
            variance = sum((x - mean_val) ** 2 for x in array) / len(array)
            return math.sqrt(variance)
        
        @staticmethod
        def corrcoef(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            den_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            den_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            if den_x == 0 or den_y == 0:
                return 0
            return num / math.sqrt(den_x * den_y)

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ============================================================================
# PREFERENCE ENGINE ENUMS & DATA STRUCTURES
# ============================================================================

class PreferenceCategory(Enum):
    """Categories of user preferences"""
    CONTENT_TYPE = "content_type"
    INTERACTION_STYLE = "interaction_style"
    DIFFICULTY_LEVEL = "difficulty_level"
    PACING = "pacing"
    FEEDBACK_STYLE = "feedback_style"
    VISUAL_DESIGN = "visual_design"
    AUDIO_PREFERENCES = "audio_preferences"
    SOCIAL_INTERACTION = "social_interaction"
    LEARNING_ENVIRONMENT = "learning_environment"
    ASSESSMENT_TYPE = "assessment_type"

class PreferenceStrength(Enum):
    """Strength levels of preferences"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class PreferenceSource(Enum):
    """Sources of preference information"""
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    PERFORMANCE_CORRELATION = "performance_correlation"
    ENGAGEMENT_PATTERN = "engagement_pattern"
    COMPLETION_RATE = "completion_rate"
    TIME_SPENT = "time_spent"

@dataclass
class UserPreference:
    """
    🎯 USER PREFERENCE
    
    Individual user preference with detailed metadata
    """
    user_id: str
    category: PreferenceCategory
    preference_key: str
    preference_value: Any
    strength: PreferenceStrength
    confidence: float  # 0.0-1.0
    
    # Preference metadata
    source: PreferenceSource
    evidence_count: int
    last_reinforced: datetime
    first_observed: datetime
    
    # Context information
    context_tags: List[str] = field(default_factory=list)
    domain_specific: bool = False
    temporal_stability: float = 0.7  # How stable this preference is over time
    
    # Performance correlation
    performance_correlation: float = 0.0
    engagement_correlation: float = 0.0
    
    # Evolution tracking
    preference_history: List[Dict[str, Any]] = field(default_factory=list)
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"

@dataclass
class PreferenceProfile:
    """
    📊 COMPREHENSIVE PREFERENCE PROFILE
    
    Complete preference profile for a user across all categories
    """
    user_id: str
    preferences: Dict[PreferenceCategory, List[UserPreference]]
    
    # Profile metadata
    profile_completeness: float
    confidence_score: float
    last_updated: datetime
    
    # Preference patterns
    preference_clusters: List[Dict[str, Any]] = field(default_factory=list)
    cross_category_correlations: Dict[str, float] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Prediction capabilities
    predictive_accuracy: float = 0.0
    prediction_confidence: float = 0.0
    
    # Adaptation metrics
    adaptation_responsiveness: float = 0.7
    preference_volatility: float = 0.3


class PreferenceEngine:
    """
    🎯 ADVANCED PREFERENCE ENGINE
    
    Revolutionary preference modeling and prediction system that learns user
    preferences from multiple sources, predicts future preferences, and adapts
    to changing user needs using advanced machine learning and behavioral analysis.
    """
    
    def __init__(self, cache_service=None):
        """Initialize the preference engine"""
        
        # Core preference systems
        self.user_preferences = defaultdict(lambda: defaultdict(list))
        self.preference_profiles = {}
        self.preference_models = {}
        
        # Learning and prediction systems
        self.preference_learner = PreferenceLearner()
        self.preference_predictor = PreferencePredictor()
        self.preference_adapter = PreferenceAdapter()
        
        # Preference tracking
        self.interaction_history = defaultdict(deque)
        self.preference_evolution = defaultdict(list)
        self.prediction_accuracy_tracking = defaultdict(list)
        
        # Configuration
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
        self.preference_decay_rate = 0.05  # How quickly preferences fade without reinforcement
        
        # Performance metrics
        self.engine_metrics = {
            'total_preferences_learned': 0,
            'prediction_accuracy': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Cache service
        self.cache_service = cache_service
        
        logger.info("🎯 Advanced Preference Engine initialized")
    
    async def learn_preferences_from_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn user preferences from interaction data
        
        Args:
            user_id: User identifier
            interaction_data: Data about user interaction
            context: Context of the interaction
            
        Returns:
            dict: Learning results and updated preferences
        """
        try:
            # Extract preference signals from interaction
            preference_signals = await self.preference_learner.extract_preference_signals(
                interaction_data, context
            )
            
            # Update existing preferences or create new ones
            updated_preferences = []
            for signal in preference_signals:
                preference = await self._update_or_create_preference(
                    user_id, signal, interaction_data
                )
                updated_preferences.append(preference)
            
            # Update preference profile
            await self._update_preference_profile(user_id, updated_preferences)
            
            # Track learning metrics
            self.engine_metrics['total_preferences_learned'] += len(updated_preferences)
            
            return {
                'user_id': user_id,
                'preferences_updated': len(updated_preferences),
                'new_preferences': [p for p in updated_preferences if p.evidence_count == 1],
                'reinforced_preferences': [p for p in updated_preferences if p.evidence_count > 1],
                'learning_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error learning preferences for {user_id}: {e}")
            return {'error': str(e), 'preferences_updated': 0}
    
    async def predict_user_preferences(
        self,
        user_id: str,
        context: Dict[str, Any],
        prediction_categories: Optional[List[PreferenceCategory]] = None
    ) -> Dict[str, Any]:
        """
        Predict user preferences for given context
        
        Args:
            user_id: User identifier
            context: Context for prediction
            prediction_categories: Specific categories to predict (optional)
            
        Returns:
            dict: Predicted preferences with confidence scores
        """
        try:
            # Get user's preference profile
            preference_profile = await self._get_or_create_preference_profile(user_id)
            
            # Determine categories to predict
            if prediction_categories is None:
                prediction_categories = list(PreferenceCategory)
            
            # Generate predictions for each category
            predictions = {}
            for category in prediction_categories:
                category_prediction = await self.preference_predictor.predict_category_preference(
                    preference_profile, category, context
                )
                predictions[category.value] = category_prediction
            
            # Calculate overall prediction confidence
            overall_confidence = np.mean([
                pred.get('confidence', 0.5) for pred in predictions.values()
            ])
            
            # Generate contextual recommendations
            recommendations = await self._generate_preference_recommendations(
                predictions, context, preference_profile
            )
            
            return {
                'user_id': user_id,
                'predictions': predictions,
                'overall_confidence': overall_confidence,
                'recommendations': recommendations,
                'context': context,
                'prediction_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error predicting preferences for {user_id}: {e}")
            return {'error': str(e), 'predictions': {}}
    
    async def adapt_to_feedback(
        self,
        user_id: str,
        feedback: Dict[str, Any],
        applied_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt preferences based on user feedback
        
        Args:
            user_id: User identifier
            feedback: User feedback on applied preferences
            applied_preferences: Preferences that were applied
            
        Returns:
            dict: Adaptation results and updated preferences
        """
        try:
            # Analyze feedback for preference adjustments
            feedback_analysis = await self.preference_adapter.analyze_feedback(
                feedback, applied_preferences
            )
            
            # Apply preference adjustments
            adaptation_results = await self.preference_adapter.adapt_preferences(
                user_id, feedback_analysis, self.user_preferences[user_id]
            )
            
            # Update preference profile
            await self._update_preference_profile(user_id, adaptation_results.get('updated_preferences', []))
            
            # Track adaptation success
            adaptation_success = feedback_analysis.get('satisfaction_score', 0.5)
            self.prediction_accuracy_tracking[user_id].append(adaptation_success)
            
            # Update engine metrics
            recent_accuracy = np.mean(list(self.prediction_accuracy_tracking[user_id])[-10:])
            self.engine_metrics['prediction_accuracy'] = recent_accuracy
            
            return {
                'user_id': user_id,
                'adaptation_applied': True,
                'adaptation_strength': adaptation_results.get('adaptation_strength', 0.0),
                'updated_preferences': adaptation_results.get('updated_preferences', []),
                'satisfaction_improvement': feedback_analysis.get('satisfaction_score', 0.5),
                'adaptation_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error adapting preferences for {user_id}: {e}")
            return {'error': str(e), 'adaptation_applied': False}
    
    async def get_preference_insights(
        self,
        user_id: str,
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Get comprehensive insights about user preferences
        
        Args:
            user_id: User identifier
            analysis_depth: Depth of analysis ("basic", "detailed", "comprehensive")
            
        Returns:
            dict: Preference insights and analytics
        """
        try:
            # Get preference profile
            preference_profile = await self._get_or_create_preference_profile(user_id)
            
            # Analyze preference patterns
            pattern_analysis = await self._analyze_preference_patterns(preference_profile)
            
            # Analyze preference evolution
            evolution_analysis = await self._analyze_preference_evolution(user_id)
            
            # Generate preference recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                preference_profile, pattern_analysis
            )
            
            insights = {
                'user_id': user_id,
                'preference_summary': {
                    'total_preferences': sum(len(prefs) for prefs in preference_profile.preferences.values()),
                    'strong_preferences': await self._count_strong_preferences(preference_profile),
                    'profile_completeness': preference_profile.profile_completeness,
                    'confidence_score': preference_profile.confidence_score
                },
                'pattern_analysis': pattern_analysis,
                'evolution_analysis': evolution_analysis,
                'optimization_recommendations': optimization_recommendations,
                'analysis_timestamp': datetime.now()
            }
            
            if analysis_depth == "comprehensive":
                insights.update({
                    'cross_category_correlations': preference_profile.cross_category_correlations,
                    'temporal_patterns': preference_profile.temporal_patterns,
                    'predictive_accuracy': preference_profile.predictive_accuracy
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating preference insights for {user_id}: {e}")
            return {'error': str(e), 'insights_available': False}

    # ========================================================================
    # HELPER METHODS FOR PREFERENCE MANAGEMENT
    # ========================================================================

    async def _update_or_create_preference(
        self,
        user_id: str,
        preference_signal: Dict[str, Any],
        interaction_data: Dict[str, Any]
    ) -> UserPreference:
        """Update existing preference or create new one"""

        category = PreferenceCategory(preference_signal['category'])
        preference_key = preference_signal['key']
        preference_value = preference_signal['value']

        # Check if preference already exists
        existing_preferences = self.user_preferences[user_id][category]
        existing_pref = None

        for pref in existing_preferences:
            if pref.preference_key == preference_key:
                existing_pref = pref
                break

        if existing_pref:
            # Update existing preference
            existing_pref.evidence_count += 1
            existing_pref.last_reinforced = datetime.now()
            existing_pref.confidence = min(1.0, existing_pref.confidence + 0.1)

            # Update strength based on evidence
            if existing_pref.evidence_count >= 10:
                existing_pref.strength = PreferenceStrength.VERY_STRONG
            elif existing_pref.evidence_count >= 5:
                existing_pref.strength = PreferenceStrength.STRONG
            elif existing_pref.evidence_count >= 3:
                existing_pref.strength = PreferenceStrength.MODERATE

            # Track preference evolution
            existing_pref.preference_history.append({
                'timestamp': datetime.now(),
                'value': preference_value,
                'evidence_count': existing_pref.evidence_count,
                'confidence': existing_pref.confidence
            })

            return existing_pref
        else:
            # Create new preference
            new_preference = UserPreference(
                user_id=user_id,
                category=category,
                preference_key=preference_key,
                preference_value=preference_value,
                strength=PreferenceStrength.WEAK,
                confidence=0.3,
                source=PreferenceSource(preference_signal.get('source', 'implicit_behavior')),
                evidence_count=1,
                last_reinforced=datetime.now(),
                first_observed=datetime.now(),
                context_tags=preference_signal.get('context_tags', [])
            )

            self.user_preferences[user_id][category].append(new_preference)
            return new_preference

    async def _get_or_create_preference_profile(self, user_id: str) -> PreferenceProfile:
        """Get existing preference profile or create new one"""

        if user_id in self.preference_profiles:
            return self.preference_profiles[user_id]

        # Create new preference profile
        profile = PreferenceProfile(
            user_id=user_id,
            preferences=dict(self.user_preferences[user_id]),
            profile_completeness=0.1,
            confidence_score=0.3,
            last_updated=datetime.now()
        )

        self.preference_profiles[user_id] = profile
        return profile

    async def _update_preference_profile(
        self,
        user_id: str,
        updated_preferences: List[UserPreference]
    ):
        """Update preference profile with new preferences"""

        profile = await self._get_or_create_preference_profile(user_id)

        # Update preferences in profile
        for pref in updated_preferences:
            if pref.category not in profile.preferences:
                profile.preferences[pref.category] = []

            # Update or add preference
            existing_idx = None
            for i, existing_pref in enumerate(profile.preferences[pref.category]):
                if existing_pref.preference_key == pref.preference_key:
                    existing_idx = i
                    break

            if existing_idx is not None:
                profile.preferences[pref.category][existing_idx] = pref
            else:
                profile.preferences[pref.category].append(pref)

        # Recalculate profile metrics
        profile.profile_completeness = await self._calculate_profile_completeness(profile)
        profile.confidence_score = await self._calculate_profile_confidence(profile)
        profile.last_updated = datetime.now()

    async def _calculate_profile_completeness(self, profile: PreferenceProfile) -> float:
        """Calculate how complete the preference profile is"""

        total_categories = len(PreferenceCategory)
        covered_categories = len(profile.preferences)

        # Base completeness on category coverage
        category_completeness = covered_categories / total_categories

        # Adjust for preference depth within categories
        depth_scores = []
        for category, preferences in profile.preferences.items():
            if preferences:
                avg_evidence = np.mean([p.evidence_count for p in preferences])
                depth_score = min(1.0, avg_evidence / 5.0)  # Normalize to 5 evidence points
                depth_scores.append(depth_score)

        depth_completeness = np.mean(depth_scores) if depth_scores else 0.0

        return (category_completeness * 0.6 + depth_completeness * 0.4)

    async def _calculate_profile_confidence(self, profile: PreferenceProfile) -> float:
        """Calculate overall confidence in the preference profile"""

        all_confidences = []
        for preferences in profile.preferences.values():
            for pref in preferences:
                all_confidences.append(pref.confidence)

        if not all_confidences:
            return 0.0

        return np.mean(all_confidences)

    async def _analyze_preference_patterns(self, profile: PreferenceProfile) -> Dict[str, Any]:
        """Analyze patterns in user preferences"""

        patterns = {
            'dominant_categories': [],
            'preference_clusters': [],
            'consistency_score': 0.0,
            'volatility_score': 0.0
        }

        # Find dominant categories
        category_strengths = {}
        for category, preferences in profile.preferences.items():
            if preferences:
                avg_strength = np.mean([
                    {'weak': 1, 'moderate': 2, 'strong': 3, 'very_strong': 4}[p.strength.value]
                    for p in preferences
                ])
                category_strengths[category.value] = avg_strength

        # Sort by strength
        sorted_categories = sorted(category_strengths.items(), key=lambda x: x[1], reverse=True)
        patterns['dominant_categories'] = [cat for cat, strength in sorted_categories[:3]]

        # Calculate consistency score
        all_confidences = []
        for preferences in profile.preferences.values():
            for pref in preferences:
                all_confidences.append(pref.confidence)

        if all_confidences:
            patterns['consistency_score'] = 1.0 - np.std(all_confidences)

        return patterns

    async def _analyze_preference_evolution(self, user_id: str) -> Dict[str, Any]:
        """Analyze how preferences have evolved over time"""

        evolution_data = self.preference_evolution.get(user_id, [])

        if len(evolution_data) < 2:
            return {'evolution_trend': 'insufficient_data', 'stability_score': 0.5}

        # Analyze trend direction
        recent_changes = evolution_data[-5:]  # Last 5 changes
        change_directions = []

        for change in recent_changes:
            if change.get('confidence_delta', 0) > 0:
                change_directions.append(1)
            elif change.get('confidence_delta', 0) < 0:
                change_directions.append(-1)
            else:
                change_directions.append(0)

        avg_direction = np.mean(change_directions) if change_directions else 0

        if avg_direction > 0.2:
            trend = 'strengthening'
        elif avg_direction < -0.2:
            trend = 'weakening'
        else:
            trend = 'stable'

        # Calculate stability score
        confidence_changes = [abs(change.get('confidence_delta', 0)) for change in recent_changes]
        stability_score = 1.0 - np.mean(confidence_changes) if confidence_changes else 0.5

        return {
            'evolution_trend': trend,
            'stability_score': stability_score,
            'total_changes': len(evolution_data),
            'recent_activity': len(recent_changes)
        }


class PreferenceLearner:
    """
    📚 PREFERENCE LEARNER

    Specialized system for learning preferences from user interactions
    """

    async def extract_preference_signals(
        self,
        interaction_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract preference signals from interaction data"""

        signals = []

        # Extract content type preferences
        if 'content_type' in interaction_data:
            engagement = interaction_data.get('engagement_score', 0.5)
            if engagement > 0.6:  # Only learn from positive interactions
                signals.append({
                    'category': 'content_type',
                    'key': 'preferred_content_type',
                    'value': interaction_data['content_type'],
                    'source': 'engagement_pattern',
                    'strength': engagement
                })

        # Extract difficulty preferences
        if 'difficulty_level' in interaction_data:
            performance = interaction_data.get('performance_score', 0.5)
            completion = interaction_data.get('completion_rate', 0.5)

            if performance > 0.7 and completion > 0.8:
                signals.append({
                    'category': 'difficulty_level',
                    'key': 'optimal_difficulty',
                    'value': interaction_data['difficulty_level'],
                    'source': 'performance_correlation',
                    'strength': (performance + completion) / 2
                })

        # Extract pacing preferences
        if 'session_duration' in interaction_data:
            completion = interaction_data.get('completion_rate', 0.5)
            engagement = interaction_data.get('engagement_score', 0.5)

            if completion > 0.7 and engagement > 0.6:
                signals.append({
                    'category': 'pacing',
                    'key': 'preferred_session_duration',
                    'value': interaction_data['session_duration'],
                    'source': 'completion_rate',
                    'strength': (completion + engagement) / 2
                })

        # Extract feedback preferences
        if 'feedback_received' in interaction_data:
            feedback_engagement = interaction_data.get('feedback_engagement', 0.5)
            if feedback_engagement > 0.6:
                signals.append({
                    'category': 'feedback_style',
                    'key': 'feedback_frequency',
                    'value': interaction_data.get('feedback_frequency', 'medium'),
                    'source': 'engagement_pattern',
                    'strength': feedback_engagement
                })

        return signals


class PreferencePredictor:
    """
    🔮 PREFERENCE PREDICTOR

    Specialized system for predicting user preferences
    """

    async def predict_category_preference(
        self,
        profile: PreferenceProfile,
        category: PreferenceCategory,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict preference for a specific category"""

        # Get existing preferences for category
        existing_prefs = profile.preferences.get(category, [])

        if not existing_prefs:
            return await self._generate_default_prediction(category, context)

        # Find strongest preference
        strongest_pref = max(existing_prefs, key=lambda p: p.confidence *
                           {'weak': 1, 'moderate': 2, 'strong': 3, 'very_strong': 4}[p.strength.value])

        # Calculate prediction confidence
        base_confidence = strongest_pref.confidence
        evidence_boost = min(0.3, strongest_pref.evidence_count * 0.05)
        context_adjustment = await self._calculate_context_adjustment(strongest_pref, context)

        prediction_confidence = min(1.0, base_confidence + evidence_boost + context_adjustment)

        return {
            'predicted_value': strongest_pref.preference_value,
            'confidence': prediction_confidence,
            'evidence_count': strongest_pref.evidence_count,
            'last_reinforced': strongest_pref.last_reinforced.isoformat(),
            'prediction_rationale': f"Based on {strongest_pref.evidence_count} observations with {strongest_pref.strength.value} strength"
        }

    async def _generate_default_prediction(
        self,
        category: PreferenceCategory,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate default prediction when no preferences exist"""

        defaults = {
            PreferenceCategory.CONTENT_TYPE: 'mixed',
            PreferenceCategory.DIFFICULTY_LEVEL: 0.6,
            PreferenceCategory.PACING: 'moderate',
            PreferenceCategory.FEEDBACK_STYLE: 'balanced',
            PreferenceCategory.SOCIAL_INTERACTION: 0.5
        }

        return {
            'predicted_value': defaults.get(category, 'unknown'),
            'confidence': 0.3,
            'evidence_count': 0,
            'prediction_rationale': 'Default prediction - no user data available'
        }

    async def _calculate_context_adjustment(
        self,
        preference: UserPreference,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence adjustment based on context"""

        # Check if context matches preference context tags
        context_match_score = 0.0

        if preference.context_tags:
            context_keywords = context.get('keywords', [])
            matching_tags = set(preference.context_tags) & set(context_keywords)
            context_match_score = len(matching_tags) / len(preference.context_tags)

        # Temporal adjustment
        time_since_reinforcement = (datetime.now() - preference.last_reinforced).days
        temporal_adjustment = max(0, 0.1 - (time_since_reinforcement * 0.01))

        return context_match_score * 0.1 + temporal_adjustment


class PreferenceAdapter:
    """
    🔄 PREFERENCE ADAPTER

    Specialized system for adapting preferences based on feedback
    """

    async def analyze_feedback(
        self,
        feedback: Dict[str, Any],
        applied_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user feedback for preference adjustments"""

        satisfaction_score = feedback.get('satisfaction_score', 0.5)
        specific_feedback = feedback.get('specific_feedback', {})

        analysis = {
            'satisfaction_score': satisfaction_score,
            'adjustment_needed': satisfaction_score < 0.6,
            'adjustment_strength': max(0, 0.8 - satisfaction_score),
            'specific_adjustments': []
        }

        # Analyze specific feedback
        for category, feedback_value in specific_feedback.items():
            if feedback_value.get('rating', 0.5) < 0.5:
                analysis['specific_adjustments'].append({
                    'category': category,
                    'current_value': applied_preferences.get(category),
                    'suggested_adjustment': feedback_value.get('suggestion'),
                    'adjustment_strength': 0.6 - feedback_value.get('rating', 0.5)
                })

        return analysis

    async def adapt_preferences(
        self,
        user_id: str,
        feedback_analysis: Dict[str, Any],
        current_preferences: Dict[PreferenceCategory, List[UserPreference]]
    ) -> Dict[str, Any]:
        """Adapt preferences based on feedback analysis"""

        if not feedback_analysis.get('adjustment_needed', False):
            return {'adaptation_applied': False, 'updated_preferences': []}

        updated_preferences = []
        adaptation_strength = feedback_analysis.get('adjustment_strength', 0.0)

        # Apply specific adjustments
        for adjustment in feedback_analysis.get('specific_adjustments', []):
            category_name = adjustment['category']

            try:
                category = PreferenceCategory(category_name)
                category_prefs = current_preferences.get(category, [])

                # Find relevant preference to adjust
                for pref in category_prefs:
                    if pref.preference_value == adjustment['current_value']:
                        # Reduce confidence based on negative feedback
                        confidence_reduction = adjustment['adjustment_strength'] * 0.3
                        pref.confidence = max(0.1, pref.confidence - confidence_reduction)

                        # Update strength if confidence drops significantly
                        if pref.confidence < 0.4:
                            if pref.strength == PreferenceStrength.STRONG:
                                pref.strength = PreferenceStrength.MODERATE
                            elif pref.strength == PreferenceStrength.MODERATE:
                                pref.strength = PreferenceStrength.WEAK

                        updated_preferences.append(pref)
                        break

            except ValueError:
                continue  # Skip invalid category names

        return {
            'adaptation_applied': True,
            'adaptation_strength': adaptation_strength,
            'updated_preferences': updated_preferences,
            'adjustments_made': len(updated_preferences)
        }
