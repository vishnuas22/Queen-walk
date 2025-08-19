"""
Attention Optimization Engine

Advanced attention analysis and optimization system for learning platforms.
Implements attention span analysis, focus enhancement algorithms, distraction mitigation,
attention state classification, and mindfulness/flow state detection.
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.signal as signal
    from scipy.stats import zscore
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


class AttentionState(Enum):
    """Types of attention states"""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    FLOW = "flow"
    FATIGUED = "fatigued"
    HYPERVIGILANT = "hypervigilant"
    MIND_WANDERING = "mind_wandering"
    DEEP_FOCUS = "deep_focus"
    SCATTERED = "scattered"


class DistractionType(Enum):
    """Types of distractions"""
    INTERNAL = "internal"  # Mind wandering, thoughts
    EXTERNAL_VISUAL = "external_visual"  # Visual distractions
    EXTERNAL_AUDITORY = "external_auditory"  # Sound distractions
    DIGITAL = "digital"  # Notifications, devices
    ENVIRONMENTAL = "environmental"  # Temperature, lighting
    SOCIAL = "social"  # People, conversations
    PHYSIOLOGICAL = "physiological"  # Hunger, fatigue


@dataclass
class AttentionMetrics:
    """Attention measurement metrics"""
    attention_span_seconds: float = 0.0
    focus_intensity: float = 0.0  # 0-1 scale
    distraction_frequency: float = 0.0  # per minute
    attention_stability: float = 0.0  # variance measure
    flow_probability: float = 0.0  # 0-1 scale
    cognitive_load: float = 0.0  # 0-1 scale
    sustained_attention_duration: float = 0.0
    attention_switching_rate: float = 0.0


@dataclass
class AttentionAnalysis:
    """Comprehensive attention analysis result"""
    user_id: str = ""
    session_id: str = ""
    attention_metrics: AttentionMetrics = field(default_factory=AttentionMetrics)
    attention_state: AttentionState = AttentionState.FOCUSED
    attention_timeline: List[Dict[str, Any]] = field(default_factory=list)
    distraction_analysis: Dict[str, Any] = field(default_factory=dict)
    focus_patterns: Dict[str, Any] = field(default_factory=dict)
    optimization_recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_timestamp: str = ""


@dataclass
class FocusEnhancementPlan:
    """Focus enhancement intervention plan"""
    plan_id: str = ""
    user_id: str = ""
    enhancement_strategies: List[Dict[str, Any]] = field(default_factory=list)
    environmental_optimizations: List[str] = field(default_factory=list)
    behavioral_interventions: List[str] = field(default_factory=list)
    technology_recommendations: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0
    implementation_timeline: Dict[str, str] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    created_at: str = ""


class AttentionOptimizationEngine:
    """
    🎯 ATTENTION OPTIMIZATION ENGINE
    
    Advanced attention analysis and optimization system for learning platforms.
    Implements cutting-edge attention research and neurofeedback principles.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Engine configuration
        self.config = {
            'attention_window_seconds': 30,
            'distraction_threshold': 0.3,
            'flow_state_threshold': 0.8,
            'fatigue_threshold': 0.2,
            'analysis_frequency': 'real_time',
            'baseline_calibration_duration': 300,  # 5 minutes
            'attention_models': {
                'focus_classifier': None,
                'distraction_detector': None,
                'flow_predictor': None
            }
        }
        
        # Attention tracking
        self.user_attention_profiles = {}
        self.attention_sessions = {}
        self.baseline_measurements = {}
        
        # Initialize ML models if sklearn is available
        if SKLEARN_AVAILABLE:
            self._initialize_attention_models()
        
        logger.info("Attention Optimization Engine initialized")
    
    def _initialize_attention_models(self):
        """Initialize machine learning models for attention analysis"""
        # Focus state classifier
        self.config['attention_models']['focus_classifier'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        
        # Distraction detector
        self.config['attention_models']['distraction_detector'] = SVC(
            kernel='rbf', probability=True, random_state=42
        )
        
        # Flow state predictor
        self.config['attention_models']['flow_predictor'] = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
    
    async def analyze_attention_patterns(self,
                                       user_id: str,
                                       behavioral_data: Dict[str, Any],
                                       physiological_data: Optional[Dict[str, Any]] = None,
                                       environmental_data: Optional[Dict[str, Any]] = None) -> AttentionAnalysis:
        """
        Comprehensive attention pattern analysis
        
        Args:
            user_id: User identifier
            behavioral_data: Behavioral indicators (response times, click patterns, etc.)
            physiological_data: Physiological indicators (simulated eye-tracking, etc.)
            environmental_data: Environmental factors
            
        Returns:
            AttentionAnalysis: Comprehensive attention analysis
        """
        try:
            # Extract attention features
            attention_features = await self._extract_attention_features(
                behavioral_data, physiological_data, environmental_data
            )
            
            # Calculate attention metrics
            attention_metrics = await self._calculate_attention_metrics(attention_features)
            
            # Classify attention state
            attention_state = await self._classify_attention_state(attention_features)
            
            # Analyze attention timeline
            attention_timeline = await self._analyze_attention_timeline(behavioral_data)
            
            # Detect and analyze distractions
            distraction_analysis = await self._analyze_distractions(
                attention_features, environmental_data
            )
            
            # Identify focus patterns
            focus_patterns = await self._identify_focus_patterns(attention_timeline)
            
            # Generate optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                attention_metrics, distraction_analysis, focus_patterns
            )
            
            # Calculate confidence score
            confidence_score = await self._calculate_analysis_confidence(attention_features)
            
            # Create analysis result
            analysis = AttentionAnalysis(
                user_id=user_id,
                session_id=f"attention_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                attention_metrics=attention_metrics,
                attention_state=attention_state,
                attention_timeline=attention_timeline,
                distraction_analysis=distraction_analysis,
                focus_patterns=focus_patterns,
                optimization_recommendations=optimization_recommendations,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.utcnow().isoformat()
            )
            
            # Store analysis
            if user_id not in self.user_attention_profiles:
                self.user_attention_profiles[user_id] = []
            self.user_attention_profiles[user_id].append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {e}")
            raise QuantumEngineError(f"Attention analysis failed: {e}")
    
    async def _extract_attention_features(self,
                                        behavioral_data: Dict[str, Any],
                                        physiological_data: Optional[Dict[str, Any]],
                                        environmental_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for attention analysis"""
        features = {}
        
        # Behavioral features
        if 'response_times' in behavioral_data:
            response_times = behavioral_data['response_times']
            features['avg_response_time'] = np.mean(response_times)
            features['response_time_variance'] = np.var(response_times)
            features['response_time_trend'] = self._calculate_trend(response_times)
        
        if 'click_patterns' in behavioral_data:
            click_data = behavioral_data['click_patterns']
            features['click_frequency'] = len(click_data) / max(1, behavioral_data.get('session_duration', 1))
            features['click_precision'] = self._calculate_click_precision(click_data)
        
        if 'task_switching' in behavioral_data:
            features['task_switch_frequency'] = behavioral_data['task_switching'].get('frequency', 0)
            features['task_switch_duration'] = behavioral_data['task_switching'].get('avg_duration', 0)
        
        # Physiological features (simulated)
        if physiological_data:
            if 'eye_tracking' in physiological_data:
                eye_data = physiological_data['eye_tracking']
                features['fixation_duration'] = eye_data.get('avg_fixation_duration', 250)
                features['saccade_velocity'] = eye_data.get('avg_saccade_velocity', 300)
                features['blink_rate'] = eye_data.get('blink_rate', 15)
            
            if 'heart_rate_variability' in physiological_data:
                features['hrv'] = physiological_data['heart_rate_variability']
        
        # Environmental features
        if environmental_data:
            features['noise_level'] = environmental_data.get('noise_level', 0.5)
            features['lighting_quality'] = environmental_data.get('lighting_quality', 0.8)
            features['temperature_comfort'] = environmental_data.get('temperature_comfort', 0.7)
            features['digital_distractions'] = environmental_data.get('digital_distractions', 0)
        
        # Time-based features
        current_time = datetime.now()
        features['time_of_day'] = current_time.hour + current_time.minute / 60
        features['day_of_week'] = current_time.weekday()
        
        return features
    
    async def _calculate_attention_metrics(self, features: Dict[str, Any]) -> AttentionMetrics:
        """Calculate comprehensive attention metrics"""
        metrics = AttentionMetrics()
        
        # Attention span estimation
        if 'avg_response_time' in features and 'response_time_variance' in features:
            # Stable response times indicate sustained attention
            stability = 1 / (1 + features['response_time_variance'] / max(features['avg_response_time'], 1))
            metrics.attention_span_seconds = stability * 300  # Max 5 minutes
        
        # Focus intensity
        focus_indicators = []
        if 'fixation_duration' in features:
            focus_indicators.append(min(features['fixation_duration'] / 500, 1.0))
        if 'click_precision' in features:
            focus_indicators.append(features['click_precision'])
        if 'response_time_variance' in features:
            focus_indicators.append(1 - min(features['response_time_variance'] / 1000, 1.0))
        
        if focus_indicators:
            metrics.focus_intensity = np.mean(focus_indicators)
        
        # Distraction frequency
        distraction_factors = []
        if 'task_switch_frequency' in features:
            distraction_factors.append(features['task_switch_frequency'])
        if 'digital_distractions' in features:
            distraction_factors.append(features['digital_distractions'] / 10)
        
        if distraction_factors:
            metrics.distraction_frequency = np.mean(distraction_factors)
        
        # Attention stability
        if 'response_time_variance' in features:
            metrics.attention_stability = 1 / (1 + features['response_time_variance'] / 100)
        
        # Flow probability
        flow_indicators = []
        if metrics.focus_intensity > 0.7:
            flow_indicators.append(metrics.focus_intensity)
        if metrics.distraction_frequency < 0.2:
            flow_indicators.append(1 - metrics.distraction_frequency)
        if 'hrv' in features and features['hrv'] > 0.6:
            flow_indicators.append(features['hrv'])
        
        if flow_indicators:
            metrics.flow_probability = np.mean(flow_indicators)
        
        # Cognitive load estimation
        load_indicators = []
        if 'avg_response_time' in features:
            load_indicators.append(min(features['avg_response_time'] / 2000, 1.0))
        if 'blink_rate' in features:
            # Higher blink rate can indicate cognitive load
            load_indicators.append(min(features['blink_rate'] / 25, 1.0))
        
        if load_indicators:
            metrics.cognitive_load = np.mean(load_indicators)
        
        return metrics
    
    async def _classify_attention_state(self, features: Dict[str, Any]) -> AttentionState:
        """Classify current attention state using ML or rule-based approach"""
        if SKLEARN_AVAILABLE and self.config['attention_models']['focus_classifier']:
            # Use ML classifier if available and trained
            # For now, use rule-based classification
            pass
        
        # Rule-based classification
        focus_score = features.get('click_precision', 0.5)
        distraction_score = features.get('task_switch_frequency', 0)
        response_stability = 1 / (1 + features.get('response_time_variance', 100) / 100)
        
        # Flow state detection
        if focus_score > 0.8 and distraction_score < 0.1 and response_stability > 0.8:
            return AttentionState.FLOW
        
        # Deep focus
        elif focus_score > 0.7 and distraction_score < 0.2:
            return AttentionState.DEEP_FOCUS
        
        # Regular focus
        elif focus_score > 0.5 and distraction_score < 0.4:
            return AttentionState.FOCUSED
        
        # Distracted
        elif distraction_score > 0.5:
            return AttentionState.DISTRACTED
        
        # Scattered attention
        elif response_stability < 0.3:
            return AttentionState.SCATTERED
        
        # Fatigue detection
        elif features.get('avg_response_time', 0) > 1500:
            return AttentionState.FATIGUED
        
        # Default to mind wandering
        else:
            return AttentionState.MIND_WANDERING
    
    async def _analyze_attention_timeline(self, behavioral_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze attention patterns over time"""
        timeline = []
        
        if 'response_times' in behavioral_data:
            response_times = behavioral_data['response_times']
            timestamps = behavioral_data.get('timestamps', list(range(len(response_times))))
            
            # Sliding window analysis
            window_size = 10
            for i in range(0, len(response_times) - window_size + 1, window_size // 2):
                window_data = response_times[i:i + window_size]
                window_time = timestamps[i + window_size // 2] if i + window_size // 2 < len(timestamps) else timestamps[-1]
                
                # Calculate attention metrics for this window
                avg_rt = np.mean(window_data)
                rt_variance = np.var(window_data)
                attention_score = 1 / (1 + rt_variance / max(avg_rt, 1))
                
                timeline.append({
                    'timestamp': window_time,
                    'attention_score': attention_score,
                    'response_time_avg': avg_rt,
                    'response_time_variance': rt_variance,
                    'window_start': i,
                    'window_end': i + window_size
                })
        
        return timeline
    
    async def _analyze_distractions(self,
                                  features: Dict[str, Any],
                                  environmental_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distraction patterns and sources"""
        distraction_analysis = {
            'total_distraction_score': 0.0,
            'distraction_sources': {},
            'distraction_timeline': [],
            'mitigation_strategies': []
        }
        
        # Analyze distraction sources
        if environmental_data:
            if environmental_data.get('noise_level', 0) > 0.7:
                distraction_analysis['distraction_sources']['auditory'] = environmental_data['noise_level']
                distraction_analysis['mitigation_strategies'].append('noise_reduction')
            
            if environmental_data.get('digital_distractions', 0) > 5:
                distraction_analysis['distraction_sources']['digital'] = environmental_data['digital_distractions'] / 10
                distraction_analysis['mitigation_strategies'].append('digital_minimization')
            
            if environmental_data.get('lighting_quality', 1) < 0.5:
                distraction_analysis['distraction_sources']['visual'] = 1 - environmental_data['lighting_quality']
                distraction_analysis['mitigation_strategies'].append('lighting_optimization')
        
        # Internal distraction indicators
        if features.get('task_switch_frequency', 0) > 0.3:
            distraction_analysis['distraction_sources']['internal'] = features['task_switch_frequency']
            distraction_analysis['mitigation_strategies'].append('mindfulness_training')
        
        # Calculate total distraction score
        if distraction_analysis['distraction_sources']:
            distraction_analysis['total_distraction_score'] = np.mean(
                list(distraction_analysis['distraction_sources'].values())
            )
        
        return distraction_analysis
    
    async def _identify_focus_patterns(self, attention_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in focus and attention"""
        if not attention_timeline:
            return {}
        
        attention_scores = [point['attention_score'] for point in attention_timeline]
        timestamps = [point['timestamp'] for point in attention_timeline]
        
        patterns = {
            'peak_focus_periods': [],
            'attention_cycles': {},
            'focus_duration_distribution': {},
            'optimal_focus_times': []
        }
        
        # Find peak focus periods
        if SCIPY_AVAILABLE:
            peaks, properties = signal.find_peaks(attention_scores, height=0.7, distance=5)
            for peak_idx in peaks:
                if peak_idx < len(attention_timeline):
                    patterns['peak_focus_periods'].append({
                        'timestamp': timestamps[peak_idx],
                        'attention_score': attention_scores[peak_idx],
                        'duration_estimate': properties.get('widths', [1])[0] if 'widths' in properties else 1
                    })
        
        # Analyze attention cycles
        if len(attention_scores) > 10:
            # Simple cycle detection using autocorrelation
            autocorr = np.correlate(attention_scores, attention_scores, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find potential cycle length
            if len(autocorr) > 5:
                cycle_candidates = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
                if len(cycle_candidates[0]) > 0:
                    patterns['attention_cycles']['cycle_length'] = cycle_candidates[0][0] + 1
                    patterns['attention_cycles']['cycle_strength'] = autocorr[cycle_candidates[0][0] + 1]
        
        return patterns
    
    async def _generate_optimization_recommendations(self,
                                                   attention_metrics: AttentionMetrics,
                                                   distraction_analysis: Dict[str, Any],
                                                   focus_patterns: Dict[str, Any]) -> List[str]:
        """Generate personalized attention optimization recommendations"""
        recommendations = []
        
        # Focus intensity recommendations
        if attention_metrics.focus_intensity < 0.5:
            recommendations.append("Implement focused breathing exercises before learning sessions")
            recommendations.append("Use the Pomodoro Technique with 25-minute focus intervals")
            recommendations.append("Minimize multitasking and focus on single tasks")
        
        # Distraction mitigation
        if attention_metrics.distraction_frequency > 0.4:
            recommendations.append("Create a dedicated, distraction-free learning environment")
            recommendations.append("Use website blockers during study sessions")
            recommendations.append("Practice mindfulness meditation to improve attention control")
        
        # Flow state enhancement
        if attention_metrics.flow_probability < 0.3:
            recommendations.append("Adjust task difficulty to match your skill level")
            recommendations.append("Set clear, achievable goals for each learning session")
            recommendations.append("Eliminate external interruptions during deep work")
        
        # Specific distraction sources
        for source, score in distraction_analysis.get('distraction_sources', {}).items():
            if score > 0.5:
                if source == 'auditory':
                    recommendations.append("Use noise-canceling headphones or white noise")
                elif source == 'digital':
                    recommendations.append("Turn off non-essential notifications")
                elif source == 'visual':
                    recommendations.append("Optimize lighting and reduce visual clutter")
                elif source == 'internal':
                    recommendations.append("Practice attention training exercises")
        
        # Cognitive load management
        if attention_metrics.cognitive_load > 0.7:
            recommendations.append("Break complex tasks into smaller, manageable chunks")
            recommendations.append("Take regular breaks to prevent cognitive overload")
            recommendations.append("Use visual aids and diagrams to reduce mental effort")
        
        # Attention span improvement
        if attention_metrics.attention_span_seconds < 600:  # Less than 10 minutes
            recommendations.append("Gradually increase focus session duration")
            recommendations.append("Practice sustained attention exercises")
            recommendations.append("Ensure adequate sleep and nutrition for optimal attention")
        
        return recommendations
    
    async def _calculate_analysis_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the attention analysis"""
        confidence_factors = []
        
        # Data completeness
        expected_features = ['avg_response_time', 'response_time_variance', 'click_precision']
        available_features = sum(1 for feature in expected_features if feature in features)
        confidence_factors.append(available_features / len(expected_features))
        
        # Data quality indicators
        if 'response_time_variance' in features:
            # Lower variance indicates more reliable data
            variance_quality = 1 / (1 + features['response_time_variance'] / 1000)
            confidence_factors.append(variance_quality)
        
        # Sample size (if available)
        if 'sample_size' in features:
            sample_quality = min(features['sample_size'] / 100, 1.0)
            confidence_factors.append(sample_quality)
        
        # Environmental data availability
        env_features = ['noise_level', 'lighting_quality', 'temperature_comfort']
        env_available = sum(1 for feature in env_features if feature in features)
        confidence_factors.append(env_available / len(env_features))
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in time series data"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope
    
    def _calculate_click_precision(self, click_data: List[Dict[str, Any]]) -> float:
        """Calculate click precision metric"""
        if not click_data:
            return 0.5
        
        # Simulate click precision based on click patterns
        # In real implementation, this would analyze actual click coordinates
        precision_scores = []
        
        for click in click_data:
            # Simulate precision based on click timing and context
            if 'target_distance' in click:
                precision = 1 / (1 + click['target_distance'] / 100)
            else:
                precision = random.uniform(0.6, 0.9)  # Simulated precision
            precision_scores.append(precision)
        
        return np.mean(precision_scores)


class FocusEnhancementAlgorithms:
    """
    🧠 FOCUS ENHANCEMENT ALGORITHMS
    
    Advanced algorithms for enhancing focus and attention based on neurofeedback principles.
    """
    
    def __init__(self):
        self.enhancement_history = {}
        logger.info("Focus Enhancement Algorithms initialized")
    
    async def create_focus_enhancement_plan(self,
                                          user_id: str,
                                          attention_analysis: AttentionAnalysis,
                                          user_preferences: Dict[str, Any]) -> FocusEnhancementPlan:
        """
        Create personalized focus enhancement plan
        
        Args:
            user_id: User identifier
            attention_analysis: Recent attention analysis
            user_preferences: User preferences and constraints
            
        Returns:
            FocusEnhancementPlan: Comprehensive enhancement plan
        """
        try:
            plan_id = f"focus_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user_id}"
            
            # Analyze current attention state
            current_metrics = attention_analysis.attention_metrics
            
            # Generate enhancement strategies
            enhancement_strategies = await self._generate_enhancement_strategies(
                current_metrics, user_preferences
            )
            
            # Environmental optimizations
            environmental_optimizations = await self._recommend_environmental_changes(
                attention_analysis.distraction_analysis, user_preferences
            )
            
            # Behavioral interventions
            behavioral_interventions = await self._design_behavioral_interventions(
                current_metrics, attention_analysis.focus_patterns
            )
            
            # Technology recommendations
            technology_recommendations = await self._suggest_technology_aids(
                current_metrics, user_preferences
            )
            
            # Estimate expected improvement
            expected_improvement = await self._estimate_improvement_potential(
                current_metrics, enhancement_strategies
            )
            
            # Create implementation timeline
            implementation_timeline = await self._create_implementation_timeline(
                enhancement_strategies, user_preferences
            )
            
            # Define success metrics
            success_metrics = await self._define_success_metrics(current_metrics)
            
            plan = FocusEnhancementPlan(
                plan_id=plan_id,
                user_id=user_id,
                enhancement_strategies=enhancement_strategies,
                environmental_optimizations=environmental_optimizations,
                behavioral_interventions=behavioral_interventions,
                technology_recommendations=technology_recommendations,
                expected_improvement=expected_improvement,
                implementation_timeline=implementation_timeline,
                success_metrics=success_metrics,
                created_at=datetime.utcnow().isoformat()
            )
            
            # Store plan
            if user_id not in self.enhancement_history:
                self.enhancement_history[user_id] = []
            self.enhancement_history[user_id].append(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating focus enhancement plan: {e}")
            raise QuantumEngineError(f"Focus enhancement plan creation failed: {e}")
    
    async def _generate_enhancement_strategies(self,
                                             metrics: AttentionMetrics,
                                             preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized enhancement strategies"""
        strategies = []
        
        # Focus intensity strategies
        if metrics.focus_intensity < 0.6:
            strategies.append({
                'type': 'attention_training',
                'name': 'Focused Attention Meditation',
                'description': 'Daily 10-minute focused breathing exercises',
                'duration_minutes': 10,
                'frequency': 'daily',
                'difficulty': 'beginner',
                'expected_benefit': 0.2
            })
            
            strategies.append({
                'type': 'cognitive_training',
                'name': 'Attention Control Exercises',
                'description': 'Structured exercises to improve attention control',
                'duration_minutes': 15,
                'frequency': '3x_weekly',
                'difficulty': 'intermediate',
                'expected_benefit': 0.25
            })
        
        # Distraction management
        if metrics.distraction_frequency > 0.3:
            strategies.append({
                'type': 'distraction_management',
                'name': 'Progressive Distraction Resistance',
                'description': 'Gradually increase resistance to distractions',
                'duration_minutes': 20,
                'frequency': 'daily',
                'difficulty': 'intermediate',
                'expected_benefit': 0.3
            })
        
        # Flow state cultivation
        if metrics.flow_probability < 0.4:
            strategies.append({
                'type': 'flow_cultivation',
                'name': 'Flow State Training',
                'description': 'Techniques to enter and maintain flow states',
                'duration_minutes': 25,
                'frequency': '4x_weekly',
                'difficulty': 'advanced',
                'expected_benefit': 0.35
            })
        
        # Attention span building
        if metrics.attention_span_seconds < 900:  # Less than 15 minutes
            strategies.append({
                'type': 'span_building',
                'name': 'Progressive Attention Extension',
                'description': 'Gradually increase sustained attention duration',
                'duration_minutes': 30,
                'frequency': 'daily',
                'difficulty': 'beginner',
                'expected_benefit': 0.4
            })
        
        return strategies
    
    async def _recommend_environmental_changes(self,
                                             distraction_analysis: Dict[str, Any],
                                             preferences: Dict[str, Any]) -> List[str]:
        """Recommend environmental optimizations"""
        recommendations = []
        
        distraction_sources = distraction_analysis.get('distraction_sources', {})
        
        if 'auditory' in distraction_sources:
            recommendations.extend([
                "Use noise-canceling headphones during focus sessions",
                "Create a quiet study space away from high-traffic areas",
                "Use white noise or nature sounds to mask distracting sounds"
            ])
        
        if 'visual' in distraction_sources:
            recommendations.extend([
                "Optimize lighting with adjustable desk lamps",
                "Remove visual clutter from workspace",
                "Use neutral colors in study environment"
            ])
        
        if 'digital' in distraction_sources:
            recommendations.extend([
                "Use app blockers during study sessions",
                "Turn off non-essential notifications",
                "Keep phone in another room during deep work"
            ])
        
        # General environmental recommendations
        recommendations.extend([
            "Maintain comfortable temperature (68-72°F)",
            "Ensure good air circulation",
            "Use ergonomic furniture to reduce physical discomfort",
            "Add plants to improve air quality and reduce stress"
        ])
        
        return recommendations
    
    async def _design_behavioral_interventions(self,
                                             metrics: AttentionMetrics,
                                             focus_patterns: Dict[str, Any]) -> List[str]:
        """Design behavioral interventions"""
        interventions = []
        
        # Time management interventions
        interventions.extend([
            "Implement the Pomodoro Technique (25 min work, 5 min break)",
            "Schedule demanding tasks during peak attention hours",
            "Use time-blocking to allocate specific times for focused work"
        ])
        
        # Attention training interventions
        if metrics.focus_intensity < 0.5:
            interventions.extend([
                "Practice mindfulness meditation for 10 minutes daily",
                "Use attention training apps with progressive difficulty",
                "Perform single-tasking exercises to improve focus"
            ])
        
        # Energy management
        interventions.extend([
            "Take regular breaks to prevent attention fatigue",
            "Maintain consistent sleep schedule for optimal attention",
            "Use physical exercise to boost cognitive function"
        ])
        
        # Habit formation
        interventions.extend([
            "Create consistent pre-work rituals to signal focus time",
            "Use implementation intentions (if-then planning)",
            "Track attention metrics to build awareness"
        ])
        
        return interventions
    
    async def _suggest_technology_aids(self,
                                     metrics: AttentionMetrics,
                                     preferences: Dict[str, Any]) -> List[str]:
        """Suggest technology aids for attention enhancement"""
        recommendations = []
        
        # Focus apps and tools
        recommendations.extend([
            "Use focus apps like Forest or Freedom for distraction blocking",
            "Try brain training apps for attention improvement",
            "Use meditation apps like Headspace or Calm"
        ])
        
        # Productivity tools
        recommendations.extend([
            "Use task management apps with focus timers",
            "Try ambient sound apps for concentration",
            "Use website blockers during study sessions"
        ])
        
        # Monitoring tools
        if preferences.get('tech_comfort_level', 'medium') in ['medium', 'high']:
            recommendations.extend([
                "Use attention tracking software to monitor progress",
                "Try biofeedback devices for real-time attention feedback",
                "Use smart lighting that adjusts based on time of day"
            ])
        
        # Hardware recommendations
        recommendations.extend([
            "Invest in noise-canceling headphones",
            "Use blue light filtering glasses for screen work",
            "Consider a standing desk for better alertness"
        ])
        
        return recommendations
    
    async def _estimate_improvement_potential(self,
                                            current_metrics: AttentionMetrics,
                                            strategies: List[Dict[str, Any]]) -> float:
        """Estimate potential improvement from enhancement plan"""
        # Calculate baseline improvement potential
        baseline_potential = 1.0 - max(
            current_metrics.focus_intensity,
            current_metrics.attention_stability,
            1 - current_metrics.distraction_frequency
        )
        
        # Calculate strategy benefits
        strategy_benefits = [s.get('expected_benefit', 0.1) for s in strategies]
        total_strategy_benefit = min(sum(strategy_benefits), 0.8)  # Cap at 80% improvement
        
        # Combine baseline potential with strategy benefits
        estimated_improvement = baseline_potential * total_strategy_benefit
        
        return min(estimated_improvement, 0.6)  # Cap at 60% improvement
    
    async def _create_implementation_timeline(self,
                                            strategies: List[Dict[str, Any]],
                                            preferences: Dict[str, Any]) -> Dict[str, str]:
        """Create implementation timeline for enhancement plan"""
        timeline = {}
        
        # Immediate actions (Week 1)
        timeline['week_1'] = "Begin environmental optimizations and basic attention exercises"
        
        # Short-term goals (Weeks 2-4)
        timeline['weeks_2_4'] = "Establish daily attention training routine and distraction management"
        
        # Medium-term goals (Months 2-3)
        timeline['months_2_3'] = "Advanced attention techniques and flow state cultivation"
        
        # Long-term goals (Months 4-6)
        timeline['months_4_6'] = "Maintain improvements and fine-tune personalized approach"
        
        # Ongoing maintenance
        timeline['ongoing'] = "Regular assessment and plan adjustments based on progress"
        
        return timeline
    
    async def _define_success_metrics(self, current_metrics: AttentionMetrics) -> List[str]:
        """Define success metrics for the enhancement plan"""
        metrics = []
        
        # Focus intensity improvement
        target_focus = min(current_metrics.focus_intensity + 0.3, 1.0)
        metrics.append(f"Increase focus intensity to {target_focus:.2f}")
        
        # Distraction reduction
        target_distraction = max(current_metrics.distraction_frequency - 0.2, 0.0)
        metrics.append(f"Reduce distraction frequency to {target_distraction:.2f}")
        
        # Attention span extension
        target_span = current_metrics.attention_span_seconds + 300  # Add 5 minutes
        metrics.append(f"Extend attention span to {target_span:.0f} seconds")
        
        # Flow state improvement
        target_flow = min(current_metrics.flow_probability + 0.25, 1.0)
        metrics.append(f"Increase flow probability to {target_flow:.2f}")
        
        # Stability improvement
        target_stability = min(current_metrics.attention_stability + 0.2, 1.0)
        metrics.append(f"Improve attention stability to {target_stability:.2f}")
        
        return metrics
