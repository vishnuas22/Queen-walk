"""
Advanced Learning Analytics Dashboard

Comprehensive learning analytics system that provides real-time learning progress
visualization, predictive insights, performance trend analysis, and comparative
analytics with intelligent recommendations and forecasting capabilities.

📊 LEARNING ANALYTICS CAPABILITIES:
- Real-time learning progress visualization and tracking
- Predictive insights and recommendation generation
- Performance trend analysis and forecasting
- Comparative analytics and benchmarking
- Interactive dashboard with advanced visualizations
- Multi-dimensional analytics and reporting

Author: MasterX AI Team - Predictive Analytics Division
Version: 1.0 - Phase 10 Advanced Predictive Learning Analytics Engine
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
import math

# Import predictive modeling components
from .predictive_modeling import (
    PredictiveModelingEngine, PredictionRequest, PredictionResult,
    PredictionType, PredictionHorizon, RiskLevel
)

# Import personalization components
from ..personalization import (
    LearningDNA, PersonalizationSession, BehaviorEvent, BehaviorType,
    LearningStyle, CognitivePattern
)

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

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ============================================================================
# LEARNING ANALYTICS ENUMS & DATA STRUCTURES
# ============================================================================

class AnalyticsView(Enum):
    """Types of analytics views"""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    ENGAGEMENT = "engagement"
    PREDICTIONS = "predictions"
    COMPARISONS = "comparisons"
    INTERVENTIONS = "interventions"

class TimeRange(Enum):
    """Time ranges for analytics"""
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    ALL_TIME = "all_time"

class MetricType(Enum):
    """Types of learning metrics"""
    ACCURACY = "accuracy"
    COMPLETION_RATE = "completion_rate"
    ENGAGEMENT_SCORE = "engagement_score"
    LEARNING_VELOCITY = "learning_velocity"
    SKILL_MASTERY = "skill_mastery"
    TIME_SPENT = "time_spent"
    PROGRESS_RATE = "progress_rate"

@dataclass
class AnalyticsRequest:
    """
    📊 ANALYTICS REQUEST
    
    Comprehensive analytics request with parameters and context
    """
    user_id: str
    analytics_view: AnalyticsView
    time_range: TimeRange
    
    # Specific metrics to include
    metrics: List[MetricType] = field(default_factory=list)
    
    # Comparison parameters
    compare_with_peers: bool = False
    compare_with_goals: bool = True
    
    # Prediction parameters
    include_predictions: bool = True
    prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    
    # Visualization parameters
    include_visualizations: bool = True
    chart_types: List[str] = field(default_factory=lambda: ['line', 'bar', 'radar'])
    
    # Context
    request_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AnalyticsDashboard:
    """
    📈 ANALYTICS DASHBOARD
    
    Comprehensive analytics dashboard with visualizations and insights
    """
    user_id: str
    dashboard_id: str
    analytics_view: AnalyticsView
    time_range: TimeRange
    
    # Core analytics data
    performance_metrics: Dict[str, Any]
    engagement_analytics: Dict[str, Any]
    learning_progress: Dict[str, Any]
    predictive_insights: Dict[str, Any]
    
    # Visualizations
    charts: List[Dict[str, Any]]
    trend_visualizations: List[Dict[str, Any]]
    comparison_charts: List[Dict[str, Any]]
    
    # Insights and recommendations
    key_insights: List[str]
    performance_summary: Dict[str, Any]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]
    
    # Comparative analytics
    peer_comparisons: Dict[str, Any]
    goal_progress: Dict[str, Any]
    benchmarks: Dict[str, Any]
    
    # Metadata
    dashboard_timestamp: datetime = field(default_factory=datetime.now)
    data_freshness: datetime = field(default_factory=datetime.now)
    analytics_confidence: float = 0.8

@dataclass
class LearningInsight:
    """
    💡 LEARNING INSIGHT
    
    Individual learning insight with context and recommendations
    """
    insight_id: str
    insight_type: str
    title: str
    description: str
    
    # Insight data
    metric_values: Dict[str, Any]
    trend_direction: str
    significance_level: float
    
    # Recommendations
    recommended_actions: List[str]
    priority_level: str
    
    # Context
    time_period: str
    affected_areas: List[str]
    
    # Metadata
    confidence_score: float
    insight_timestamp: datetime = field(default_factory=datetime.now)


class LearningAnalyticsEngine:
    """
    📊 LEARNING ANALYTICS ENGINE
    
    Advanced learning analytics system that provides comprehensive real-time
    learning progress visualization, predictive insights, performance analysis,
    and intelligent recommendations with interactive dashboard capabilities.
    """
    
    def __init__(self, predictive_engine: Optional[PredictiveModelingEngine] = None):
        """Initialize the learning analytics engine"""
        
        # Core engines
        self.predictive_engine = predictive_engine or PredictiveModelingEngine()
        
        # Analytics data storage
        self.user_analytics = defaultdict(dict)
        self.dashboard_cache = {}
        self.insights_history = defaultdict(list)
        
        # Analytics processors
        self.performance_analyzer = PerformanceAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.comparison_analyzer = ComparisonAnalyzer()
        
        # Configuration
        self.cache_ttl_minutes = 15
        self.max_insights_per_user = 50
        self.confidence_threshold = 0.6
        
        # Performance tracking
        self.analytics_metrics = {
            'dashboards_generated': 0,
            'insights_created': 0,
            'predictions_integrated': 0,
            'average_response_time': 0.0
        }
        
        logger.info("📊 Learning Analytics Engine initialized")
    
    async def generate_analytics_dashboard(
        self,
        analytics_request: AnalyticsRequest,
        learning_dna: LearningDNA,
        recent_performance: List[Dict[str, Any]],
        behavioral_history: List[BehaviorEvent]
    ) -> AnalyticsDashboard:
        """
        Generate comprehensive analytics dashboard
        
        Args:
            analytics_request: Analytics request parameters
            learning_dna: User's learning DNA
            recent_performance: Recent performance data
            behavioral_history: Behavioral event history
            
        Returns:
            AnalyticsDashboard: Comprehensive analytics dashboard
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_dashboard_cache_key(analytics_request)
            cached_dashboard = self._get_cached_dashboard(cache_key)
            if cached_dashboard:
                return cached_dashboard
            
            # Generate performance metrics
            performance_metrics = await self.performance_analyzer.analyze_performance(
                analytics_request.user_id, recent_performance, analytics_request.time_range
            )
            
            # Generate engagement analytics
            engagement_analytics = await self.engagement_analyzer.analyze_engagement(
                analytics_request.user_id, behavioral_history, analytics_request.time_range
            )
            
            # Generate learning progress analysis
            learning_progress = await self._analyze_learning_progress(
                analytics_request, learning_dna, recent_performance
            )
            
            # Generate predictive insights
            predictive_insights = {}
            if analytics_request.include_predictions:
                predictive_insights = await self._generate_predictive_insights(
                    analytics_request, learning_dna, recent_performance, behavioral_history
                )
            
            # Generate visualizations
            charts = await self._generate_charts(
                analytics_request, performance_metrics, engagement_analytics
            )
            
            # Generate trend visualizations
            trend_visualizations = await self.trend_analyzer.generate_trend_visualizations(
                analytics_request.user_id, recent_performance, behavioral_history
            )
            
            # Generate comparison charts
            comparison_charts = []
            if analytics_request.compare_with_peers or analytics_request.compare_with_goals:
                comparison_charts = await self.comparison_analyzer.generate_comparison_charts(
                    analytics_request, performance_metrics, engagement_analytics
                )
            
            # Generate insights and recommendations
            key_insights = await self._generate_key_insights(
                performance_metrics, engagement_analytics, predictive_insights
            )
            
            performance_summary = await self._generate_performance_summary(
                performance_metrics, engagement_analytics, learning_progress
            )
            
            recommendations = await self._generate_recommendations(
                analytics_request, performance_metrics, engagement_analytics, predictive_insights
            )
            
            alerts = await self._generate_alerts(
                analytics_request, performance_metrics, predictive_insights
            )
            
            # Generate comparative analytics
            peer_comparisons = {}
            goal_progress = {}
            benchmarks = {}
            
            if analytics_request.compare_with_peers:
                peer_comparisons = await self.comparison_analyzer.generate_peer_comparisons(
                    analytics_request.user_id, performance_metrics
                )
            
            if analytics_request.compare_with_goals:
                goal_progress = await self._analyze_goal_progress(
                    analytics_request, performance_metrics, learning_progress
                )
            
            benchmarks = await self._generate_benchmarks(
                analytics_request, performance_metrics, engagement_analytics
            )
            
            # Create dashboard
            dashboard = AnalyticsDashboard(
                user_id=analytics_request.user_id,
                dashboard_id=f"dashboard_{analytics_request.user_id}_{int(time.time())}",
                analytics_view=analytics_request.analytics_view,
                time_range=analytics_request.time_range,
                performance_metrics=performance_metrics,
                engagement_analytics=engagement_analytics,
                learning_progress=learning_progress,
                predictive_insights=predictive_insights,
                charts=charts,
                trend_visualizations=trend_visualizations,
                comparison_charts=comparison_charts,
                key_insights=key_insights,
                performance_summary=performance_summary,
                recommendations=recommendations,
                alerts=alerts,
                peer_comparisons=peer_comparisons,
                goal_progress=goal_progress,
                benchmarks=benchmarks,
                analytics_confidence=await self._calculate_analytics_confidence(
                    performance_metrics, engagement_analytics, predictive_insights
                )
            )
            
            # Cache the dashboard
            self._cache_dashboard(cache_key, dashboard)
            
            # Update metrics
            response_time = time.time() - start_time
            self.analytics_metrics['dashboards_generated'] += 1
            self._update_response_time_metric(response_time)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating analytics dashboard: {e}")
            return await self._generate_fallback_dashboard(analytics_request)
    
    async def generate_learning_insights(
        self,
        user_id: str,
        learning_dna: LearningDNA,
        recent_performance: List[Dict[str, Any]],
        behavioral_history: List[BehaviorEvent],
        time_range: TimeRange = TimeRange.LAST_WEEK
    ) -> List[LearningInsight]:
        """
        Generate personalized learning insights
        
        Args:
            user_id: User identifier
            learning_dna: User's learning DNA
            recent_performance: Recent performance data
            behavioral_history: Behavioral event history
            time_range: Time range for analysis
            
        Returns:
            List[LearningInsight]: List of personalized learning insights
        """
        try:
            insights = []
            
            # Performance insights
            performance_insights = await self._generate_performance_insights(
                user_id, recent_performance, time_range
            )
            insights.extend(performance_insights)
            
            # Engagement insights
            engagement_insights = await self._generate_engagement_insights(
                user_id, behavioral_history, time_range
            )
            insights.extend(engagement_insights)
            
            # Learning style insights
            learning_style_insights = await self._generate_learning_style_insights(
                user_id, learning_dna, recent_performance
            )
            insights.extend(learning_style_insights)
            
            # Predictive insights
            predictive_insights = await self._generate_predictive_learning_insights(
                user_id, learning_dna, recent_performance, behavioral_history
            )
            insights.extend(predictive_insights)
            
            # Sort insights by significance and confidence
            insights.sort(key=lambda x: (x.significance_level, x.confidence_score), reverse=True)
            
            # Limit number of insights
            insights = insights[:10]
            
            # Store insights history
            self.insights_history[user_id].extend(insights)
            if len(self.insights_history[user_id]) > self.max_insights_per_user:
                self.insights_history[user_id] = self.insights_history[user_id][-self.max_insights_per_user:]
            
            # Update metrics
            self.analytics_metrics['insights_created'] += len(insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return []

    # ========================================================================
    # HELPER METHODS FOR LEARNING ANALYTICS
    # ========================================================================

    def _generate_dashboard_cache_key(self, request: AnalyticsRequest) -> str:
        """Generate cache key for dashboard"""

        key_components = [
            request.user_id,
            request.analytics_view.value,
            request.time_range.value,
            str(request.include_predictions),
            str(int(request.request_timestamp.timestamp() // (self.cache_ttl_minutes * 60)))
        ]

        return "_".join(key_components)

    def _get_cached_dashboard(self, cache_key: str) -> Optional[AnalyticsDashboard]:
        """Get cached dashboard if available and valid"""

        if cache_key not in self.dashboard_cache:
            return None

        cached_data = self.dashboard_cache[cache_key]
        cache_time = cached_data['timestamp']

        # Check if cache is still valid
        if (datetime.now() - cache_time).total_seconds() > (self.cache_ttl_minutes * 60):
            del self.dashboard_cache[cache_key]
            return None

        return cached_data['dashboard']

    def _cache_dashboard(self, cache_key: str, dashboard: AnalyticsDashboard):
        """Cache dashboard"""

        self.dashboard_cache[cache_key] = {
            'dashboard': dashboard,
            'timestamp': datetime.now()
        }

        # Limit cache size
        if len(self.dashboard_cache) > 100:
            oldest_keys = sorted(
                self.dashboard_cache.keys(),
                key=lambda k: self.dashboard_cache[k]['timestamp']
            )[:20]

            for key in oldest_keys:
                del self.dashboard_cache[key]

    async def _analyze_learning_progress(
        self,
        request: AnalyticsRequest,
        learning_dna: LearningDNA,
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze learning progress"""

        if not recent_performance:
            return {
                'overall_progress': 0.5,
                'skill_mastery_levels': {},
                'learning_velocity': 0.5,
                'progress_trend': 'stable'
            }

        # Calculate overall progress
        completion_rates = [p.get('completion_rate', 0.5) for p in recent_performance]
        overall_progress = np.mean(completion_rates)

        # Calculate learning velocity
        if len(recent_performance) >= 2:
            recent_avg = np.mean([p.get('accuracy', 0.5) for p in recent_performance[-5:]])
            earlier_avg = np.mean([p.get('accuracy', 0.5) for p in recent_performance[:-5]]) if len(recent_performance) > 5 else recent_avg
            learning_velocity = max(0.0, min(1.0, recent_avg - earlier_avg + 0.5))
        else:
            learning_velocity = 0.5

        # Determine progress trend
        if len(completion_rates) >= 3:
            recent_trend = np.mean(completion_rates[-3:])
            earlier_trend = np.mean(completion_rates[:-3]) if len(completion_rates) > 3 else recent_trend

            if recent_trend > earlier_trend + 0.1:
                progress_trend = 'improving'
            elif recent_trend < earlier_trend - 0.1:
                progress_trend = 'declining'
            else:
                progress_trend = 'stable'
        else:
            progress_trend = 'stable'

        # Analyze skill mastery levels
        skill_mastery_levels = {}
        subjects = set(p.get('subject', 'general') for p in recent_performance)

        for subject in subjects:
            subject_performance = [p for p in recent_performance if p.get('subject') == subject]
            if subject_performance:
                subject_accuracy = np.mean([p.get('accuracy', 0.5) for p in subject_performance])
                skill_mastery_levels[subject] = subject_accuracy

        return {
            'overall_progress': overall_progress,
            'skill_mastery_levels': skill_mastery_levels,
            'learning_velocity': learning_velocity,
            'progress_trend': progress_trend,
            'completion_consistency': 1.0 - np.std(completion_rates) if len(completion_rates) > 1 else 0.5
        }

    async def _generate_predictive_insights(
        self,
        request: AnalyticsRequest,
        learning_dna: LearningDNA,
        recent_performance: List[Dict[str, Any]],
        behavioral_history: List[BehaviorEvent]
    ) -> Dict[str, Any]:
        """Generate predictive insights"""

        try:
            # Create prediction request
            prediction_request = PredictionRequest(
                user_id=request.user_id,
                prediction_type=PredictionType.LEARNING_OUTCOME,
                prediction_horizon=request.prediction_horizon,
                learning_dna=learning_dna,
                recent_performance=recent_performance,
                behavioral_history=behavioral_history
            )

            # Get prediction from predictive engine
            prediction_result = await self.predictive_engine.predict_learning_outcome(prediction_request)

            # Extract insights
            insights = {
                'predicted_outcome': prediction_result.predicted_outcome,
                'confidence_score': prediction_result.confidence_score,
                'risk_assessment': {
                    'risk_level': prediction_result.risk_level.value,
                    'risk_factors': prediction_result.risk_factors,
                    'protective_factors': prediction_result.protective_factors
                },
                'trajectory_forecast': prediction_result.trajectory_points,
                'milestone_predictions': prediction_result.milestone_predictions,
                'recommendations': prediction_result.recommended_actions,
                'intervention_suggestions': prediction_result.intervention_suggestions
            }

            # Update metrics
            self.analytics_metrics['predictions_integrated'] += 1

            return insights

        except Exception as e:
            logger.error(f"Error generating predictive insights: {e}")
            return {
                'predicted_outcome': {'learning_outcome': 0.5},
                'confidence_score': 0.3,
                'risk_assessment': {'risk_level': 'moderate', 'risk_factors': [], 'protective_factors': []},
                'trajectory_forecast': [],
                'milestone_predictions': [],
                'recommendations': [],
                'intervention_suggestions': []
            }

    async def _generate_charts(
        self,
        request: AnalyticsRequest,
        performance_metrics: Dict[str, Any],
        engagement_analytics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization charts"""

        charts = []

        # Performance trend chart
        if 'line' in request.chart_types:
            performance_chart = {
                'chart_id': 'performance_trend',
                'chart_type': 'line',
                'title': 'Performance Trend Over Time',
                'data': {
                    'labels': performance_metrics.get('time_labels', []),
                    'datasets': [{
                        'label': 'Accuracy',
                        'data': performance_metrics.get('accuracy_trend', []),
                        'color': '#4CAF50'
                    }, {
                        'label': 'Completion Rate',
                        'data': performance_metrics.get('completion_trend', []),
                        'color': '#2196F3'
                    }]
                },
                'options': {
                    'responsive': True,
                    'scales': {
                        'y': {'beginAtZero': True, 'max': 1.0}
                    }
                }
            }
            charts.append(performance_chart)

        # Engagement radar chart
        if 'radar' in request.chart_types:
            engagement_chart = {
                'chart_id': 'engagement_radar',
                'chart_type': 'radar',
                'title': 'Engagement Profile',
                'data': {
                    'labels': ['Focus', 'Participation', 'Consistency', 'Motivation', 'Interaction'],
                    'datasets': [{
                        'label': 'Current Level',
                        'data': [
                            engagement_analytics.get('focus_score', 0.5),
                            engagement_analytics.get('participation_score', 0.5),
                            engagement_analytics.get('consistency_score', 0.5),
                            engagement_analytics.get('motivation_score', 0.5),
                            engagement_analytics.get('interaction_score', 0.5)
                        ],
                        'color': '#FF9800'
                    }]
                },
                'options': {
                    'responsive': True,
                    'scales': {
                        'r': {'beginAtZero': True, 'max': 1.0}
                    }
                }
            }
            charts.append(engagement_chart)

        # Skill mastery bar chart
        if 'bar' in request.chart_types:
            skill_chart = {
                'chart_id': 'skill_mastery',
                'chart_type': 'bar',
                'title': 'Skill Mastery Levels',
                'data': {
                    'labels': list(performance_metrics.get('skill_levels', {}).keys()),
                    'datasets': [{
                        'label': 'Mastery Level',
                        'data': list(performance_metrics.get('skill_levels', {}).values()),
                        'color': '#9C27B0'
                    }]
                },
                'options': {
                    'responsive': True,
                    'scales': {
                        'y': {'beginAtZero': True, 'max': 1.0}
                    }
                }
            }
            charts.append(skill_chart)

        return charts

    async def _generate_key_insights(
        self,
        performance_metrics: Dict[str, Any],
        engagement_analytics: Dict[str, Any],
        predictive_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from analytics"""

        insights = []

        # Performance insights
        avg_accuracy = performance_metrics.get('average_accuracy', 0.5)
        if avg_accuracy > 0.8:
            insights.append("Excellent performance with consistently high accuracy")
        elif avg_accuracy < 0.6:
            insights.append("Performance below target - consider additional support")

        # Engagement insights
        avg_engagement = engagement_analytics.get('average_engagement', 0.5)
        if avg_engagement > 0.8:
            insights.append("High engagement levels indicate strong motivation")
        elif avg_engagement < 0.5:
            insights.append("Low engagement detected - intervention may be needed")

        # Trend insights
        performance_trend = performance_metrics.get('trend_direction', 'stable')
        if performance_trend == 'improving':
            insights.append("Performance is trending upward - maintain current approach")
        elif performance_trend == 'declining':
            insights.append("Performance decline detected - immediate attention required")

        # Predictive insights
        risk_level = predictive_insights.get('risk_assessment', {}).get('risk_level', 'moderate')
        if risk_level == 'high':
            insights.append("High risk of learning difficulties predicted")
        elif risk_level == 'low':
            insights.append("Low risk profile with positive learning trajectory")

        return insights[:5]  # Limit to top 5 insights

    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""

        current_avg = self.analytics_metrics['average_response_time']
        total_dashboards = self.analytics_metrics['dashboards_generated']

        if total_dashboards == 1:
            self.analytics_metrics['average_response_time'] = response_time
        else:
            new_avg = ((current_avg * (total_dashboards - 1)) + response_time) / total_dashboards
            self.analytics_metrics['average_response_time'] = new_avg

    async def _generate_fallback_dashboard(self, request: AnalyticsRequest) -> AnalyticsDashboard:
        """Generate fallback dashboard when main generation fails"""

        return AnalyticsDashboard(
            user_id=request.user_id,
            dashboard_id=f"fallback_{request.user_id}_{int(time.time())}",
            analytics_view=request.analytics_view,
            time_range=request.time_range,
            performance_metrics={'average_accuracy': 0.5, 'completion_rate': 0.5},
            engagement_analytics={'average_engagement': 0.5},
            learning_progress={'overall_progress': 0.5},
            predictive_insights={},
            charts=[],
            trend_visualizations=[],
            comparison_charts=[],
            key_insights=['Insufficient data for detailed analysis'],
            performance_summary={'status': 'data_limited'},
            recommendations=['Increase learning activity for better analytics'],
            alerts=[],
            peer_comparisons={},
            goal_progress={},
            benchmarks={},
            analytics_confidence=0.3
        )


class PerformanceAnalyzer:
    """
    📈 PERFORMANCE ANALYZER

    Specialized analyzer for learning performance metrics
    """

    async def analyze_performance(
        self,
        user_id: str,
        performance_data: List[Dict[str, Any]],
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Analyze learning performance metrics"""

        if not performance_data:
            return self._get_default_performance_metrics()

        # Filter data by time range
        filtered_data = self._filter_by_time_range(performance_data, time_range)

        # Calculate core metrics
        accuracy_values = [p.get('accuracy', 0.5) for p in filtered_data]
        completion_values = [p.get('completion_rate', 0.5) for p in filtered_data]

        # Performance metrics
        metrics = {
            'average_accuracy': np.mean(accuracy_values),
            'accuracy_std': np.std(accuracy_values),
            'average_completion': np.mean(completion_values),
            'completion_std': np.std(completion_values),
            'total_sessions': len(filtered_data),
            'consistency_score': 1.0 - np.std(accuracy_values) if len(accuracy_values) > 1 else 0.5
        }

        # Trend analysis
        metrics['trend_direction'] = self._calculate_trend(accuracy_values)
        metrics['accuracy_trend'] = accuracy_values
        metrics['completion_trend'] = completion_values

        # Time labels for charts
        metrics['time_labels'] = [f"Session {i+1}" for i in range(len(filtered_data))]

        # Skill-specific analysis
        metrics['skill_levels'] = self._analyze_skill_levels(filtered_data)

        # Performance categories
        metrics['performance_category'] = self._categorize_performance(metrics['average_accuracy'])

        return metrics

    def _filter_by_time_range(self, data: List[Dict[str, Any]], time_range: TimeRange) -> List[Dict[str, Any]]:
        """Filter data by time range"""

        now = datetime.now()

        if time_range == TimeRange.LAST_DAY:
            cutoff = now - timedelta(days=1)
        elif time_range == TimeRange.LAST_WEEK:
            cutoff = now - timedelta(weeks=1)
        elif time_range == TimeRange.LAST_MONTH:
            cutoff = now - timedelta(days=30)
        elif time_range == TimeRange.LAST_QUARTER:
            cutoff = now - timedelta(days=90)
        else:  # ALL_TIME
            return data

        # Filter data (assuming timestamp field exists)
        filtered = []
        for item in data:
            item_time = item.get('timestamp')
            if item_time:
                if isinstance(item_time, str):
                    try:
                        item_time = datetime.fromisoformat(item_time.replace('Z', '+00:00'))
                    except:
                        item_time = now  # Default to now if parsing fails

                if item_time >= cutoff:
                    filtered.append(item)
            else:
                filtered.append(item)  # Include items without timestamp

        return filtered

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""

        if len(values) < 3:
            return 'stable'

        # Compare recent vs earlier values
        recent_avg = np.mean(values[-3:])
        earlier_avg = np.mean(values[:-3]) if len(values) > 3 else np.mean(values)

        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'declining'
        else:
            return 'stable'

    def _analyze_skill_levels(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze skill levels by subject"""

        skill_levels = {}
        subjects = set(item.get('subject', 'general') for item in data)

        for subject in subjects:
            subject_data = [item for item in data if item.get('subject') == subject]
            if subject_data:
                accuracies = [item.get('accuracy', 0.5) for item in subject_data]
                skill_levels[subject] = np.mean(accuracies)

        return skill_levels

    def _categorize_performance(self, avg_accuracy: float) -> str:
        """Categorize performance level"""

        if avg_accuracy >= 0.9:
            return 'excellent'
        elif avg_accuracy >= 0.8:
            return 'good'
        elif avg_accuracy >= 0.7:
            return 'satisfactory'
        elif avg_accuracy >= 0.6:
            return 'needs_improvement'
        else:
            return 'requires_attention'

    def _get_default_performance_metrics(self) -> Dict[str, Any]:
        """Get default performance metrics when no data available"""

        return {
            'average_accuracy': 0.5,
            'accuracy_std': 0.0,
            'average_completion': 0.5,
            'completion_std': 0.0,
            'total_sessions': 0,
            'consistency_score': 0.5,
            'trend_direction': 'stable',
            'accuracy_trend': [],
            'completion_trend': [],
            'time_labels': [],
            'skill_levels': {},
            'performance_category': 'insufficient_data'
        }


class EngagementAnalyzer:
    """
    🎯 ENGAGEMENT ANALYZER

    Specialized analyzer for learning engagement metrics
    """

    async def analyze_engagement(
        self,
        user_id: str,
        behavioral_history: List[BehaviorEvent],
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Analyze learning engagement metrics"""

        if not behavioral_history:
            return self._get_default_engagement_metrics()

        # Filter events by time range
        filtered_events = self._filter_events_by_time_range(behavioral_history, time_range)

        # Calculate engagement metrics
        engagement_levels = [event.engagement_level for event in filtered_events]
        session_durations = [event.duration for event in filtered_events]

        metrics = {
            'average_engagement': np.mean(engagement_levels),
            'engagement_std': np.std(engagement_levels),
            'total_interactions': len(filtered_events),
            'average_session_duration': np.mean(session_durations),
            'engagement_consistency': 1.0 - np.std(engagement_levels) if len(engagement_levels) > 1 else 0.5
        }

        # Engagement components
        metrics['focus_score'] = self._calculate_focus_score(filtered_events)
        metrics['participation_score'] = self._calculate_participation_score(filtered_events)
        metrics['consistency_score'] = metrics['engagement_consistency']
        metrics['motivation_score'] = self._calculate_motivation_score(filtered_events)
        metrics['interaction_score'] = self._calculate_interaction_score(filtered_events)

        # Engagement trend
        metrics['engagement_trend'] = engagement_levels
        metrics['trend_direction'] = self._calculate_engagement_trend(engagement_levels)

        # Engagement patterns
        metrics['peak_engagement_times'] = self._identify_peak_engagement_times(filtered_events)
        metrics['engagement_patterns'] = self._analyze_engagement_patterns(filtered_events)

        return metrics

    def _filter_events_by_time_range(self, events: List[BehaviorEvent], time_range: TimeRange) -> List[BehaviorEvent]:
        """Filter behavioral events by time range"""

        now = datetime.now()

        if time_range == TimeRange.LAST_DAY:
            cutoff = now - timedelta(days=1)
        elif time_range == TimeRange.LAST_WEEK:
            cutoff = now - timedelta(weeks=1)
        elif time_range == TimeRange.LAST_MONTH:
            cutoff = now - timedelta(days=30)
        elif time_range == TimeRange.LAST_QUARTER:
            cutoff = now - timedelta(days=90)
        else:  # ALL_TIME
            return events

        return [event for event in events if event.timestamp >= cutoff]

    def _calculate_focus_score(self, events: List[BehaviorEvent]) -> float:
        """Calculate focus score from behavioral events"""

        if not events:
            return 0.5

        # Focus based on session duration and engagement consistency
        durations = [event.duration for event in events if event.duration > 0]
        if not durations:
            return 0.5

        avg_duration = np.mean(durations)
        duration_consistency = 1.0 - np.std(durations) / max(avg_duration, 1)

        # Normalize to 0-1 range
        focus_score = min(1.0, (avg_duration / 1800) * 0.7 + duration_consistency * 0.3)  # 30 min = good focus

        return max(0.0, focus_score)

    def _calculate_participation_score(self, events: List[BehaviorEvent]) -> float:
        """Calculate participation score"""

        if not events:
            return 0.5

        # Participation based on interaction frequency and success rate
        interaction_events = [e for e in events if e.event_type == BehaviorType.INTERACTION]

        if not interaction_events:
            return 0.3

        success_rate = np.mean([e.success_indicator for e in interaction_events])
        interaction_frequency = len(interaction_events) / len(events)

        participation_score = success_rate * 0.6 + interaction_frequency * 0.4

        return max(0.0, min(1.0, participation_score))

    def _calculate_motivation_score(self, events: List[BehaviorEvent]) -> float:
        """Calculate motivation score"""

        if not events:
            return 0.5

        # Motivation based on engagement levels and session completion
        engagement_levels = [e.engagement_level for e in events]
        avg_engagement = np.mean(engagement_levels)

        # Consider emotional state if available
        positive_emotions = sum(1 for e in events if e.emotional_state == 'positive')
        emotion_ratio = positive_emotions / len(events) if events else 0.5

        motivation_score = avg_engagement * 0.7 + emotion_ratio * 0.3

        return max(0.0, min(1.0, motivation_score))

    def _calculate_interaction_score(self, events: List[BehaviorEvent]) -> float:
        """Calculate interaction score"""

        if not events:
            return 0.5

        # Interaction based on variety and frequency of interactions
        interaction_types = set(e.event_type.value for e in events)
        type_diversity = len(interaction_types) / 8.0  # Normalize by max types

        interaction_frequency = len([e for e in events if e.event_type == BehaviorType.INTERACTION]) / len(events)

        interaction_score = type_diversity * 0.5 + interaction_frequency * 0.5

        return max(0.0, min(1.0, interaction_score))

    def _calculate_engagement_trend(self, engagement_levels: List[float]) -> str:
        """Calculate engagement trend direction"""

        if len(engagement_levels) < 3:
            return 'stable'

        recent_avg = np.mean(engagement_levels[-3:])
        earlier_avg = np.mean(engagement_levels[:-3]) if len(engagement_levels) > 3 else np.mean(engagement_levels)

        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'declining'
        else:
            return 'stable'

    def _identify_peak_engagement_times(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Identify peak engagement times"""

        if not events:
            return []

        # Group by hour of day
        hourly_engagement = defaultdict(list)
        for event in events:
            hour = event.timestamp.hour
            hourly_engagement[hour].append(event.engagement_level)

        # Calculate average engagement per hour
        peak_times = []
        for hour, levels in hourly_engagement.items():
            avg_engagement = np.mean(levels)
            if avg_engagement > 0.7:  # High engagement threshold
                peak_times.append({
                    'hour': hour,
                    'average_engagement': avg_engagement,
                    'session_count': len(levels)
                })

        # Sort by engagement level
        peak_times.sort(key=lambda x: x['average_engagement'], reverse=True)

        return peak_times[:3]  # Top 3 peak times

    def _analyze_engagement_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyze engagement patterns"""

        if not events:
            return {}

        # Weekly patterns
        daily_engagement = defaultdict(list)
        for event in events:
            day = event.timestamp.strftime('%A')
            daily_engagement[day].append(event.engagement_level)

        weekly_pattern = {}
        for day, levels in daily_engagement.items():
            weekly_pattern[day] = np.mean(levels)

        # Session length patterns
        duration_engagement = []
        for event in events:
            if event.duration > 0:
                duration_engagement.append({
                    'duration': event.duration,
                    'engagement': event.engagement_level
                })

        return {
            'weekly_pattern': weekly_pattern,
            'optimal_session_length': self._find_optimal_session_length(duration_engagement),
            'engagement_volatility': np.std([e.engagement_level for e in events]) if len(events) > 1 else 0
        }

    def _find_optimal_session_length(self, duration_engagement: List[Dict[str, Any]]) -> int:
        """Find optimal session length for maximum engagement"""

        if not duration_engagement:
            return 30  # Default 30 minutes

        # Group by duration ranges
        duration_ranges = {
            'short': (0, 900),      # 0-15 minutes
            'medium': (900, 1800),  # 15-30 minutes
            'long': (1800, 3600),   # 30-60 minutes
            'extended': (3600, float('inf'))  # 60+ minutes
        }

        range_engagement = {}
        for range_name, (min_dur, max_dur) in duration_ranges.items():
            matching_sessions = [
                de['engagement'] for de in duration_engagement
                if min_dur <= de['duration'] < max_dur
            ]
            if matching_sessions:
                range_engagement[range_name] = np.mean(matching_sessions)

        if not range_engagement:
            return 30

        # Find range with highest engagement
        best_range = max(range_engagement.items(), key=lambda x: x[1])[0]

        # Return midpoint of best range
        range_midpoints = {
            'short': 7,    # 7.5 minutes
            'medium': 22,  # 22.5 minutes
            'long': 45,    # 45 minutes
            'extended': 75 # 75 minutes
        }

        return range_midpoints.get(best_range, 30)

    def _get_default_engagement_metrics(self) -> Dict[str, Any]:
        """Get default engagement metrics when no data available"""

        return {
            'average_engagement': 0.5,
            'engagement_std': 0.0,
            'total_interactions': 0,
            'average_session_duration': 0,
            'engagement_consistency': 0.5,
            'focus_score': 0.5,
            'participation_score': 0.5,
            'consistency_score': 0.5,
            'motivation_score': 0.5,
            'interaction_score': 0.5,
            'engagement_trend': [],
            'trend_direction': 'stable',
            'peak_engagement_times': [],
            'engagement_patterns': {}
        }
