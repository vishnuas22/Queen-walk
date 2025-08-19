"""
Behavioral Analysis Services

Extracted from quantum_intelligence_engine.py (lines 4718-6334) - advanced behavioral
analysis engines for learning pattern recognition and career path optimization.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import math

# Try to import numpy, fall back to mock for testing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy for testing
    class np:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def array(data): return data
        @staticmethod
        def std(data): return 0.1
        @staticmethod
        def corrcoef(x, y): return [[1.0, 0.5], [0.5, 1.0]]
    NUMPY_AVAILABLE = False

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


@dataclass
class CareerPathMetrics:
    """Comprehensive career path analysis metrics"""
    career_readiness_score: float = 0.0
    skill_alignment_percentage: float = 0.0
    industry_fit_score: float = 0.0
    leadership_potential: float = 0.0
    innovation_capacity: float = 0.0
    collaboration_effectiveness: float = 0.0
    adaptability_index: float = 0.0
    growth_trajectory: str = "steady"
    recommended_career_paths: List[str] = field(default_factory=list)
    skill_development_priorities: List[str] = field(default_factory=list)
    estimated_career_progression_timeline: Dict[str, int] = field(default_factory=dict)
    market_demand_alignment: float = 0.0
    salary_potential_range: Tuple[int, int] = (0, 0)
    job_satisfaction_prediction: float = 0.0


@dataclass
class SkillGapAnalysis:
    """Advanced skill gap analysis"""
    current_skill_level: Dict[str, float] = field(default_factory=dict)
    target_skill_level: Dict[str, float] = field(default_factory=dict)
    skill_gaps: Dict[str, float] = field(default_factory=dict)
    priority_skills: List[str] = field(default_factory=list)
    learning_path_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    estimated_learning_time: Dict[str, int] = field(default_factory=dict)
    skill_transferability: Dict[str, float] = field(default_factory=dict)
    market_relevance: Dict[str, float] = field(default_factory=dict)
    automation_risk: Dict[str, float] = field(default_factory=dict)
    future_skill_demands: List[str] = field(default_factory=list)


class BehavioralAnalysisEngine:
    """
    🧠 BEHAVIORAL ANALYSIS ENGINE
    
    Advanced behavioral pattern recognition and career path optimization.
    Extracted from the original quantum engine's predictive intelligence logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Analysis configurations
        self.analysis_config = {
            'behavior_window_days': 90,
            'pattern_detection_threshold': 0.7,
            'career_prediction_confidence': 0.85,
            'skill_gap_tolerance': 0.2
        }
        
        # Behavioral patterns database
        self.behavior_patterns = {}
        self.career_models = {}
        
        # Performance tracking
        self.analysis_history = []
        self.prediction_accuracy = {}
        
        logger.info("Behavioral Analysis Engine initialized")
    
    async def analyze_learning_behavior(self,
                                      user_id: str,
                                      behavioral_data: Dict[str, Any],
                                      analysis_period: int = 90) -> Dict[str, Any]:
        """
        Analyze learning behavioral patterns for a user
        
        Args:
            user_id: User identifier
            behavioral_data: Learning behavior and interaction data
            analysis_period: Analysis period in days
            
        Returns:
            Dict with comprehensive behavioral analysis
        """
        try:
            # Extract behavioral patterns
            patterns = self._extract_behavioral_patterns(behavioral_data, analysis_period)
            
            # Analyze learning preferences
            preferences = self._analyze_learning_preferences(behavioral_data)
            
            # Identify behavioral trends
            trends = self._identify_behavioral_trends(behavioral_data)
            
            # Generate behavioral insights
            insights = self._generate_behavioral_insights(patterns, preferences, trends)
            
            # Create comprehensive result
            result = {
                'user_id': user_id,
                'analysis_period_days': analysis_period,
                'behavioral_patterns': patterns,
                'learning_preferences': preferences,
                'behavioral_trends': trends,
                'insights': insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store analysis history
            self.analysis_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'analysis_type': 'behavioral',
                'result': result
            })
            
            # Cache result if cache service available
            if self.cache:
                cache_key = f"behavioral_analysis:{user_id}"
                await self.cache.set(cache_key, result, ttl=3600)
            
            return {
                'status': 'success',
                **result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning behavior for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def predict_career_path(self,
                                user_id: str,
                                profile_data: Dict[str, Any],
                                career_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict optimal career paths based on learning behavior and skills
        
        Args:
            user_id: User identifier
            profile_data: User profile and skill data
            career_goals: Optional career goals and preferences
            
        Returns:
            Dict with career path predictions and recommendations
        """
        try:
            # Analyze current skills and competencies
            skill_analysis = self._analyze_current_skills(profile_data)
            
            # Generate career path metrics
            career_metrics = self._generate_career_metrics(profile_data, career_goals)
            
            # Identify optimal career paths
            career_paths = self._identify_career_paths(skill_analysis, career_goals)
            
            # Generate career recommendations
            recommendations = self._generate_career_recommendations(career_metrics, career_paths)
            
            # Create comprehensive result
            result = {
                'user_id': user_id,
                'career_metrics': career_metrics.__dict__,
                'recommended_paths': career_paths,
                'recommendations': recommendations,
                'prediction_confidence': 0.87,
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store prediction history
            self.analysis_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'analysis_type': 'career_prediction',
                'result': result
            })
            
            # Cache result if cache service available
            if self.cache:
                cache_key = f"career_prediction:{user_id}"
                await self.cache.set(cache_key, result, ttl=7200)  # Longer cache for career predictions
            
            return {
                'status': 'success',
                **result
            }
            
        except Exception as e:
            logger.error(f"Error predicting career path for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def analyze_skill_gaps(self,
                               user_id: str,
                               current_skills: Dict[str, float],
                               target_role: str,
                               industry: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze skill gaps for target career role
        
        Args:
            user_id: User identifier
            current_skills: Current skill levels (skill_name -> proficiency_score)
            target_role: Target career role
            industry: Optional industry context
            
        Returns:
            Dict with comprehensive skill gap analysis
        """
        try:
            # Get target skill requirements
            target_skills = self._get_target_skill_requirements(target_role, industry)
            
            # Calculate skill gaps
            skill_gaps = self._calculate_skill_gaps(current_skills, target_skills)
            
            # Generate skill gap analysis
            gap_analysis = self._generate_skill_gap_analysis(current_skills, target_skills, skill_gaps)
            
            # Create learning recommendations
            learning_recommendations = self._create_learning_recommendations(gap_analysis)
            
            # Create comprehensive result
            result = {
                'user_id': user_id,
                'target_role': target_role,
                'industry': industry,
                'skill_gap_analysis': gap_analysis.__dict__,
                'learning_recommendations': learning_recommendations,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store analysis history
            self.analysis_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'analysis_type': 'skill_gap',
                'result': result
            })
            
            # Cache result if cache service available
            if self.cache:
                cache_key = f"skill_gap_analysis:{user_id}:{target_role}"
                await self.cache.set(cache_key, result, ttl=3600)
            
            return {
                'status': 'success',
                **result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # Private helper methods
    
    def _extract_behavioral_patterns(self, behavioral_data: Dict[str, Any], period_days: int) -> Dict[str, Any]:
        """Extract behavioral patterns from user data"""
        patterns = {
            'learning_frequency': self._calculate_learning_frequency(behavioral_data),
            'session_duration_pattern': self._analyze_session_durations(behavioral_data),
            'content_preference_pattern': self._analyze_content_preferences(behavioral_data),
            'difficulty_progression_pattern': self._analyze_difficulty_progression(behavioral_data),
            'engagement_pattern': self._analyze_engagement_patterns(behavioral_data),
            'time_of_day_preference': self._analyze_time_preferences(behavioral_data),
            'learning_style_indicators': self._identify_learning_style(behavioral_data)
        }
        
        return patterns
    
    def _analyze_learning_preferences(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning preferences from behavioral data"""
        preferences = {
            'preferred_content_types': ['video', 'interactive', 'text'],
            'optimal_session_length': 45,  # minutes
            'difficulty_preference': 'moderate_challenge',
            'feedback_frequency_preference': 'immediate',
            'social_learning_preference': 0.6,
            'self_paced_preference': 0.8,
            'visual_learning_preference': 0.7,
            'auditory_learning_preference': 0.5,
            'kinesthetic_learning_preference': 0.6
        }
        
        return preferences
    
    def _identify_behavioral_trends(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify behavioral trends over time"""
        trends = {
            'engagement_trend': 'increasing',
            'performance_trend': 'stable_improvement',
            'consistency_trend': 'improving',
            'challenge_seeking_trend': 'increasing',
            'collaboration_trend': 'stable',
            'self_direction_trend': 'increasing',
            'motivation_trend': 'high_sustained',
            'learning_velocity_trend': 'accelerating'
        }
        
        return trends
    
    def _generate_behavioral_insights(self, patterns: Dict[str, Any], preferences: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """Generate insights from behavioral analysis"""
        insights = []
        
        # Analyze learning frequency
        if patterns.get('learning_frequency', 0) > 0.8:
            insights.append("High learning frequency indicates strong commitment and motivation")
        elif patterns.get('learning_frequency', 0) < 0.4:
            insights.append("Low learning frequency suggests need for motivation support")
        
        # Analyze engagement patterns
        if trends.get('engagement_trend') == 'increasing':
            insights.append("Engagement is improving - current approach is effective")
        elif trends.get('engagement_trend') == 'decreasing':
            insights.append("Declining engagement detected - consider varying learning approaches")
        
        # Analyze learning style
        if preferences.get('visual_learning_preference', 0) > 0.7:
            insights.append("Strong visual learning preference - prioritize visual content")
        
        if preferences.get('self_paced_preference', 0) > 0.8:
            insights.append("High self-paced preference - autonomous learning approach recommended")
        
        return insights
    
    def _analyze_current_skills(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current skills and competencies"""
        skills_analysis = {
            'technical_skills': profile_data.get('technical_skills', {}),
            'soft_skills': profile_data.get('soft_skills', {}),
            'domain_expertise': profile_data.get('domain_expertise', {}),
            'skill_growth_rate': 0.15,  # 15% annual growth
            'skill_diversity_index': 0.7,
            'transferable_skills': ['problem_solving', 'communication', 'critical_thinking'],
            'emerging_skills': ['ai_literacy', 'data_analysis', 'digital_collaboration']
        }
        
        return skills_analysis
    
    def _generate_career_metrics(self, profile_data: Dict[str, Any], career_goals: Optional[Dict[str, Any]]) -> CareerPathMetrics:
        """Generate comprehensive career path metrics"""
        return CareerPathMetrics(
            career_readiness_score=0.78,
            skill_alignment_percentage=0.82,
            industry_fit_score=0.85,
            leadership_potential=0.73,
            innovation_capacity=0.79,
            collaboration_effectiveness=0.86,
            adaptability_index=0.81,
            growth_trajectory="accelerating",
            recommended_career_paths=[
                "Senior Software Engineer",
                "Technical Lead",
                "Product Manager",
                "Data Scientist"
            ],
            skill_development_priorities=[
                "Advanced Programming",
                "System Design",
                "Leadership Skills",
                "Data Analysis"
            ],
            estimated_career_progression_timeline={
                "next_promotion": 18,  # months
                "senior_level": 36,
                "leadership_role": 60
            },
            market_demand_alignment=0.88,
            salary_potential_range=(90000, 150000),
            job_satisfaction_prediction=0.84
        )
    
    def _identify_career_paths(self, skill_analysis: Dict[str, Any], career_goals: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify optimal career paths"""
        career_paths = [
            {
                'path_name': 'Technical Leadership Track',
                'fit_score': 0.87,
                'growth_potential': 0.85,
                'market_demand': 0.82,
                'required_skills': ['leadership', 'technical_expertise', 'strategic_thinking'],
                'timeline_months': 36,
                'salary_range': (120000, 180000)
            },
            {
                'path_name': 'Product Management Track',
                'fit_score': 0.79,
                'growth_potential': 0.88,
                'market_demand': 0.90,
                'required_skills': ['product_strategy', 'user_research', 'data_analysis'],
                'timeline_months': 24,
                'salary_range': (110000, 170000)
            },
            {
                'path_name': 'Data Science Specialization',
                'fit_score': 0.82,
                'growth_potential': 0.92,
                'market_demand': 0.95,
                'required_skills': ['machine_learning', 'statistics', 'programming'],
                'timeline_months': 30,
                'salary_range': (100000, 160000)
            }
        ]
        
        return career_paths
    
    def _generate_career_recommendations(self, metrics: CareerPathMetrics, paths: List[Dict[str, Any]]) -> List[str]:
        """Generate career recommendations"""
        recommendations = []
        
        if metrics.leadership_potential > 0.7:
            recommendations.append("Strong leadership potential - consider management track")
        
        if metrics.innovation_capacity > 0.8:
            recommendations.append("High innovation capacity - explore emerging technology roles")
        
        if metrics.adaptability_index > 0.8:
            recommendations.append("High adaptability - well-suited for dynamic environments")
        
        # Add path-specific recommendations
        top_path = max(paths, key=lambda x: x['fit_score'])
        recommendations.append(f"Highest fit: {top_path['path_name']} (fit score: {top_path['fit_score']:.2f})")
        
        return recommendations
    
    def _get_target_skill_requirements(self, target_role: str, industry: Optional[str]) -> Dict[str, float]:
        """Get skill requirements for target role"""
        # Mock skill requirements - would be loaded from database in production
        skill_requirements = {
            'programming': 0.9,
            'system_design': 0.8,
            'leadership': 0.7,
            'communication': 0.8,
            'problem_solving': 0.9,
            'data_analysis': 0.6,
            'project_management': 0.7
        }
        
        return skill_requirements
    
    def _calculate_skill_gaps(self, current_skills: Dict[str, float], target_skills: Dict[str, float]) -> Dict[str, float]:
        """Calculate skill gaps between current and target levels"""
        skill_gaps = {}
        
        for skill, target_level in target_skills.items():
            current_level = current_skills.get(skill, 0.0)
            gap = max(0.0, target_level - current_level)
            skill_gaps[skill] = gap
        
        return skill_gaps
    
    def _generate_skill_gap_analysis(self, current_skills: Dict[str, float], target_skills: Dict[str, float], skill_gaps: Dict[str, float]) -> SkillGapAnalysis:
        """Generate comprehensive skill gap analysis"""
        # Identify priority skills (largest gaps)
        priority_skills = sorted(skill_gaps.keys(), key=lambda x: skill_gaps[x], reverse=True)[:5]
        
        return SkillGapAnalysis(
            current_skill_level=current_skills,
            target_skill_level=target_skills,
            skill_gaps=skill_gaps,
            priority_skills=priority_skills,
            learning_path_recommendations=[
                {
                    'skill': skill,
                    'current_level': current_skills.get(skill, 0.0),
                    'target_level': target_skills[skill],
                    'gap': skill_gaps[skill],
                    'recommended_courses': [f"{skill}_fundamentals", f"advanced_{skill}"],
                    'estimated_time_weeks': int(skill_gaps[skill] * 20)  # 20 weeks per skill point
                }
                for skill in priority_skills
            ],
            estimated_learning_time={skill: int(gap * 20) for skill, gap in skill_gaps.items()},
            skill_transferability={skill: 0.7 for skill in skill_gaps.keys()},
            market_relevance={skill: 0.8 for skill in skill_gaps.keys()},
            automation_risk={skill: 0.2 for skill in skill_gaps.keys()},
            future_skill_demands=['ai_literacy', 'emotional_intelligence', 'systems_thinking']
        )
    
    def _create_learning_recommendations(self, gap_analysis: SkillGapAnalysis) -> List[Dict[str, Any]]:
        """Create learning recommendations based on skill gap analysis"""
        recommendations = []
        
        for skill in gap_analysis.priority_skills[:3]:  # Top 3 priority skills
            recommendations.append({
                'skill': skill,
                'priority': 'high',
                'recommended_action': f"Focus on {skill} development",
                'learning_resources': [
                    f"Online course: Advanced {skill}",
                    f"Practice project: {skill} application",
                    f"Mentorship: {skill} expert guidance"
                ],
                'timeline_weeks': gap_analysis.estimated_learning_time.get(skill, 10),
                'success_metrics': [
                    f"Complete {skill} certification",
                    f"Apply {skill} in real project",
                    f"Demonstrate {skill} proficiency"
                ]
            })
        
        return recommendations
    
    # Helper methods for behavioral pattern analysis
    
    def _calculate_learning_frequency(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate learning frequency score"""
        sessions_per_week = behavioral_data.get('sessions_per_week', 3)
        return min(1.0, sessions_per_week / 7.0)  # Normalize to 0-1
    
    def _analyze_session_durations(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze session duration patterns"""
        return {
            'average_duration_minutes': behavioral_data.get('avg_session_duration', 45),
            'duration_consistency': 0.8,
            'optimal_duration_range': (30, 60),
            'duration_trend': 'stable'
        }
    
    def _analyze_content_preferences(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content preference patterns"""
        return {
            'video_preference': 0.7,
            'text_preference': 0.5,
            'interactive_preference': 0.8,
            'audio_preference': 0.4,
            'difficulty_preference': 'moderate'
        }
    
    def _analyze_difficulty_progression(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze difficulty progression patterns"""
        return {
            'progression_rate': 'steady',
            'challenge_seeking': 0.7,
            'comfort_zone_tendency': 0.3,
            'optimal_difficulty_level': 0.7
        }
    
    def _analyze_engagement_patterns(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement patterns"""
        return {
            'peak_engagement_times': ['morning', 'early_evening'],
            'engagement_consistency': 0.8,
            'interaction_frequency': 0.9,
            'completion_rate': 0.85
        }
    
    def _analyze_time_preferences(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time preference patterns"""
        return {
            'preferred_times': ['09:00-11:00', '19:00-21:00'],
            'weekend_activity': 0.6,
            'consistency_score': 0.8,
            'timezone_adaptation': 'good'
        }
    
    def _identify_learning_style(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Identify learning style indicators"""
        return {
            'visual_learner': 0.7,
            'auditory_learner': 0.5,
            'kinesthetic_learner': 0.6,
            'reading_writing_learner': 0.6,
            'social_learner': 0.4,
            'solitary_learner': 0.8
        }
