"""
Comprehensive tests for Gamification Services

Tests all gamification components including reward systems, achievement engines,
engagement mechanics, motivation enhancement, gamified pathways, social competition,
and orchestration systems.
"""

import pytest
import asyncio
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Import gamification services
from quantum_intelligence.services.gamification.reward_systems import (
    RewardOptimizationEngine, PsychologicalRewardAnalyzer, DynamicRewardCalculator,
    RewardPersonalizationSystem, RewardPsychology, RewardType
)
from quantum_intelligence.services.gamification.achievement_engine import (
    DynamicAchievementGenerator, AchievementTrackingSystem, BadgeDesignEngine,
    MasteryProgressTracker, AchievementType, AchievementRarity
)
from quantum_intelligence.services.gamification.engagement_mechanics import (
    EngagementMechanicsEngine, ChallengeGenerationSystem, EngagementMechanicType,
    ChallengeType
)
from quantum_intelligence.services.gamification.motivation_enhancement import (
    LearningMotivationAnalyzer, PersonalizedMotivationSystem, MotivationType,
    MotivationState
)
from quantum_intelligence.services.gamification.gamified_pathways import (
    GamifiedLearningPathways, AdaptiveDifficultyEngine, PathwayType, QuestType
)
from quantum_intelligence.services.gamification.social_competition import (
    SocialCompetitionEngine, LeaderboardSystem, CompetitiveAnalytics,
    CompetitionType, LeaderboardType
)
from quantum_intelligence.services.gamification.orchestrator import (
    AdvancedGamificationEngine, GamificationOrchestrator, GamificationMode,
    GamificationFocus
)


# Test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_timestamp():
    """Provide a consistent timestamp for testing"""
    return datetime(2024, 1, 1, 12, 0, 0).isoformat()


@pytest.fixture
def mock_learning_content():
    """Provide mock learning content for testing"""
    return {
        'videos': [
            {'id': 'video_1', 'title': 'Introduction to Python', 'duration': 30},
            {'id': 'video_2', 'title': 'Python Data Types', 'duration': 45}
        ],
        'exercises': [
            {'id': 'exercise_1', 'title': 'Basic Python Syntax', 'difficulty': 0.3},
            {'id': 'exercise_2', 'title': 'Working with Lists', 'difficulty': 0.5}
        ],
        'quizzes': [
            {'id': 'quiz_1', 'title': 'Python Basics Quiz', 'questions': 10},
            {'id': 'quiz_2', 'title': 'Data Structures Quiz', 'questions': 15}
        ]
    }


class TestRewardSystems:
    """Test reward systems components"""
    
    @pytest.fixture
    def mock_cache_service(self):
        return Mock()
    
    @pytest.fixture
    def psychological_analyzer(self, mock_cache_service):
        return PsychologicalRewardAnalyzer(mock_cache_service)
    
    @pytest.fixture
    def reward_calculator(self, psychological_analyzer):
        return DynamicRewardCalculator(psychological_analyzer)
    
    @pytest.fixture
    def reward_engine(self, mock_cache_service):
        return RewardOptimizationEngine(mock_cache_service)
    
    @pytest.fixture
    def sample_user_data(self):
        return {
            'user_id': 'test_user_123',
            'behavioral_data': {
                'engagement_patterns': {'morning': 0.8, 'afternoon': 0.6},
                'reward_responsiveness': 0.7,
                'social_sharing_frequency': 0.4,
                'choice_making_frequency': 0.6
            },
            'reward_history': [
                {
                    'type': 'points',
                    'timestamp': datetime.utcnow().isoformat(),
                    'interaction_quality': 0.8,
                    'pre_reward_engagement': 0.6,
                    'post_reward_engagement': 0.8
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_psychological_reward_analysis(self, psychological_analyzer, sample_user_data):
        """Test psychological reward pattern analysis"""
        result = await psychological_analyzer.analyze_reward_psychology(
            sample_user_data['user_id'],
            sample_user_data['behavioral_data'],
            sample_user_data['reward_history']
        )
        
        assert result['status'] == 'success'
        assert 'reward_profile' in result
        assert 'motivation_analysis' in result
        assert 'optimization_insights' in result
        
        # Verify reward profile structure
        profile = result['reward_profile']
        assert profile['user_id'] == sample_user_data['user_id']
        assert 'primary_motivation' in profile
        assert 'reward_sensitivity' in profile
        assert isinstance(profile['reward_sensitivity'], float)
        assert 0.0 <= profile['reward_sensitivity'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_dynamic_reward_calculation(self, reward_calculator, sample_user_data):
        """Test dynamic reward calculation"""
        action_context = {
            'action_type': 'task_completion',
            'difficulty_level': 0.7,
            'quality_score': 0.8,
            'improvement_factor': 1.2
        }
        
        result = await reward_calculator.calculate_dynamic_reward(
            sample_user_data['user_id'],
            'task_completion',
            action_context
        )
        
        assert result['status'] == 'success'
        assert 'dynamic_reward' in result
        
        # Verify reward calculation
        reward = result['dynamic_reward']
        assert reward['base_value'] > 0
        assert reward['total_value'] > 0
        assert reward['psychological_multiplier'] >= 1.0
        assert 'reward_message' in reward
        assert 'visual_elements' in reward
    
    @pytest.mark.asyncio
    async def test_reward_optimization_engine(self, reward_engine):
        """Test reward system optimization"""
        user_data = [
            {
                'user_id': 'user1',
                'engagement_score': 0.7,
                'reward_history': [],
                'personalization_score': 0.6
            },
            {
                'user_id': 'user2',
                'engagement_score': 0.5,
                'reward_history': [],
                'personalization_score': 0.8
            }
        ]
        
        system_metrics = {
            'user_retention_rate': 0.75,
            'task_completion_rate': 0.65,
            'average_engagement': 0.6
        }
        
        optimization_goals = {
            'target_engagement': 0.8,
            'target_retention': 0.85
        }
        
        result = await reward_engine.optimize_reward_system(
            user_data, system_metrics, optimization_goals
        )
        
        assert result['status'] == 'success'
        assert 'performance_analysis' in result
        assert 'optimization_recommendations' in result
        assert 'optimization_plan' in result
        
        # Verify optimization analysis
        performance = result['performance_analysis']
        assert 'overall_score' in performance
        assert isinstance(performance['overall_score'], float)


class TestAchievementEngine:
    """Test achievement engine components"""
    
    @pytest.fixture
    def achievement_generator(self):
        return DynamicAchievementGenerator()
    
    @pytest.fixture
    def achievement_tracker(self, achievement_generator):
        return AchievementTrackingSystem(achievement_generator)
    
    @pytest.fixture
    def badge_designer(self):
        return BadgeDesignEngine()
    
    @pytest.fixture
    def mastery_tracker(self):
        return MasteryProgressTracker()
    
    @pytest.fixture
    def sample_user_profile(self):
        return {
            'user_id': 'test_user_456',
            'skills': {'python': 0.7, 'data_science': 0.5},
            'learning_goals': ['master python', 'learn machine learning'],
            'challenge_preference': 0.8,
            'user_level': 5
        }
    
    @pytest.mark.asyncio
    async def test_achievement_generation(self, achievement_generator, sample_user_profile):
        """Test dynamic achievement generation"""
        learning_context = {
            'recent_activities': [
                {'timestamp': datetime.utcnow().isoformat(), 'type': 'coding_exercise'}
            ],
            'goal_progress': {'master python': 0.6}
        }
        
        achievement_preferences = {
            'preferred_types': ['mastery_badge', 'learning_milestone']
        }
        
        result = await achievement_generator.generate_personalized_achievement(
            sample_user_profile, learning_context, achievement_preferences
        )
        
        assert result['status'] == 'success'
        assert 'dynamic_achievement' in result
        
        # Verify achievement structure
        achievement = result['dynamic_achievement']
        assert achievement['title']
        assert achievement['description']
        assert achievement['difficulty_tier'] >= 1
        assert achievement['difficulty_tier'] <= 10
        assert achievement['points_reward'] > 0
        assert 'unlock_criteria' in achievement
    
    @pytest.mark.asyncio
    async def test_achievement_tracking(self, achievement_tracker, sample_user_profile):
        """Test achievement progress tracking"""
        # First generate an achievement
        achievement_result = await achievement_tracker.achievement_generator.generate_personalized_achievement(
            sample_user_profile, {}, {}
        )
        
        achievement_id = achievement_result['dynamic_achievement']['achievement_id']
        
        # Track progress
        activity_data = {
            'lessons_completed': 3,
            'skill_level_achieved': 0.7,
            'assessment_score': 0.85
        }
        
        result = await achievement_tracker.track_achievement_progress(
            sample_user_profile['user_id'], achievement_id, activity_data
        )
        
        assert result['status'] == 'success'
        assert 'progress' in result
        assert 'milestones_reached' in result
        
        # Verify progress tracking
        progress = result['progress']
        assert progress['user_id'] == sample_user_profile['user_id']
        assert progress['achievement_id'] == achievement_id
        assert 0.0 <= progress['current_progress'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_badge_design(self, badge_designer):
        """Test badge design generation"""
        from quantum_intelligence.services.gamification.achievement_engine import DynamicAchievement, AchievementType
        
        achievement = DynamicAchievement(
            achievement_id='test_achievement',
            title='Python Master',
            category=AchievementType.MASTERY_BADGE,
            difficulty_tier=7,
            rarity_score=0.8
        )
        
        user_preferences = {
            'badge_style': 'detailed',
            'preferred_colors': {'primary': '#FFD700'},
            'show_initials': True,
            'initials': 'TU'
        }
        
        result = await badge_designer.design_achievement_badge(achievement, user_preferences)
        
        assert result['status'] == 'success'
        assert 'badge_design' in result
        
        # Verify badge design
        design = result['badge_design']
        assert design['design_style']
        assert design['color_scheme']
        assert design['icon_elements']
        assert design['rarity_indicators']
    
    @pytest.mark.asyncio
    async def test_mastery_tracking(self, mastery_tracker):
        """Test mastery progress tracking"""
        user_id = 'test_user_789'
        skill_assessments = {
            'python': 0.85,
            'javascript': 0.65,
            'data_science': 0.75
        }
        
        learning_activities = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'activity_type': 'coding_exercise',
                'skill': 'python',
                'performance': 0.9
            }
        ]
        
        result = await mastery_tracker.track_mastery_progress(
            user_id, skill_assessments, learning_activities
        )
        
        assert result['status'] == 'success'
        assert 'mastery_profile' in result
        assert 'new_masteries' in result
        assert 'mastery_insights' in result
        
        # Verify mastery tracking
        profile = result['mastery_profile']
        assert profile['user_id'] == user_id
        assert 'skill_masteries' in profile
        assert 'expertise_areas' in profile


class TestEngagementMechanics:
    """Test engagement mechanics components"""
    
    @pytest.fixture
    def engagement_engine(self):
        return EngagementMechanicsEngine()
    
    @pytest.fixture
    def challenge_generator(self):
        return ChallengeGenerationSystem()
    
    @pytest.fixture
    def sample_user_profile(self):
        return {
            'user_id': 'test_user_engagement',
            'personality': {'extraversion': 0.7, 'openness': 0.8},
            'learning_style': {'visual': 0.8, 'kinesthetic': 0.6},
            'preferences': {'challenge_preference': 0.7, 'social_learning_preference': 0.6}
        }
    
    @pytest.mark.asyncio
    async def test_engagement_optimization(self, engagement_engine, sample_user_profile):
        """Test engagement mechanics optimization"""
        engagement_data = {
            'current_engagement_score': 0.6,
            'engagement_trend': 'stable',
            'recent_activities': [
                {'timestamp': datetime.utcnow().isoformat(), 'duration_minutes': 45}
            ]
        }
        
        learning_context = {
            'learning_objectives': ['improve engagement', 'increase motivation'],
            'current_difficulty': 0.7
        }
        
        result = await engagement_engine.optimize_engagement_mechanics(
            sample_user_profile['user_id'], sample_user_profile, engagement_data, learning_context
        )
        
        assert result['status'] == 'success'
        assert 'engagement_analysis' in result
        assert 'optimal_mechanics' in result
        assert 'implementation_plan' in result
        
        # Verify engagement optimization
        analysis = result['engagement_analysis']
        assert 'current_engagement' in analysis
        assert 'engagement_level' in analysis
        
        mechanics = result['optimal_mechanics']
        assert isinstance(mechanics, list)
        assert len(mechanics) > 0
    
    @pytest.mark.asyncio
    async def test_challenge_generation(self, challenge_generator, sample_user_profile):
        """Test dynamic challenge generation"""
        learning_context = {
            'recent_performance': 0.75,
            'current_skills': ['python', 'problem_solving'],
            'learning_goals': ['improve coding speed']
        }
        
        challenge_preferences = {
            'preferred_types': ['speed_challenge', 'skill_challenge'],
            'difficulty_preference': 0.8
        }
        
        result = await challenge_generator.generate_personalized_challenge(
            sample_user_profile, learning_context, challenge_preferences
        )
        
        assert result['status'] == 'success'
        assert 'dynamic_challenge' in result
        
        # Verify challenge generation
        challenge = result['dynamic_challenge']
        assert challenge['title']
        assert challenge['description']
        assert challenge['challenge_type']
        assert 0.0 <= challenge['difficulty_level'] <= 1.0
        assert challenge['estimated_duration_minutes'] > 0
        assert 'success_criteria' in challenge


class TestMotivationEnhancement:
    """Test motivation enhancement components"""
    
    @pytest.fixture
    def motivation_analyzer(self):
        return LearningMotivationAnalyzer()
    
    @pytest.fixture
    def motivation_system(self, motivation_analyzer):
        # Create a mock reward calculator for the motivation system
        mock_reward_calculator = Mock()
        return PersonalizedMotivationSystem(motivation_analyzer, mock_reward_calculator)
    
    @pytest.fixture
    def sample_learning_data(self):
        return {
            'learning_goals': ['master python', 'learn data science'],
            'goal_progress': {'master python': 0.7},
            'recent_achievements': ['completed python basics'],
            'recent_performance_score': 0.75,
            'skill_confidence_average': 0.6
        }
    
    @pytest.mark.asyncio
    async def test_motivation_analysis(self, motivation_analyzer, sample_learning_data):
        """Test learning motivation analysis"""
        user_id = 'test_user_motivation'
        
        behavioral_data = {
            'curiosity_driven_learning': 0.8,
            'reward_responsiveness': 0.6,
            'peer_learning_preference': 0.7,
            'self_directed_learning_ratio': 0.8
        }
        
        context_data = {
            'social_support_level': 0.6,
            'purpose_clarity_score': 0.7
        }
        
        result = await motivation_analyzer.analyze_learning_motivation(
            user_id, sample_learning_data, behavioral_data, context_data
        )
        
        assert result['status'] == 'success'
        assert 'motivation_profile' in result
        assert 'motivation_insights' in result
        
        # Verify motivation analysis
        profile = result['motivation_profile']
        assert profile['user_id'] == user_id
        assert 'primary_motivation_type' in profile
        assert 'motivation_score' in profile
        assert 0.0 <= profile['motivation_score'] <= 1.0
        assert 'motivation_factors' in profile
    
    @pytest.mark.asyncio
    async def test_personalized_motivation_plan(self, motivation_system, sample_learning_data):
        """Test personalized motivation plan creation"""
        user_id = 'test_user_plan'
        
        user_profile = {
            'user_id': user_id,
            'personality': {'conscientiousness': 0.8, 'extraversion': 0.6},
            'learning_style': {'visual': 0.7},
            'preferences': {'structure_preference': 0.8}
        }
        
        learning_context = {
            'behavioral_data': {
                'curiosity_driven_learning': 0.7,
                'reward_responsiveness': 0.6
            },
            'reward_history': []
        }
        
        motivation_goals = {
            'target_motivation_increase': 0.3,
            'focus_areas': ['goal_clarity', 'competence_feeling']
        }
        
        result = await motivation_system.create_personalized_motivation_plan(
            user_id, user_profile, learning_context, motivation_goals
        )
        
        assert result['status'] == 'success'
        assert 'motivation_plan' in result
        
        # Verify motivation plan
        plan = result['motivation_plan']
        assert plan['user_id'] == user_id
        assert 'motivation_strategies' in plan
        assert 'implementation_plan' in plan
        assert 'personalized_recommendations' in plan


class TestGamifiedPathways:
    """Test gamified pathways components"""
    
    @pytest.fixture
    def pathways_engine(self):
        return GamifiedLearningPathways()
    
    @pytest.fixture
    def difficulty_engine(self):
        return AdaptiveDifficultyEngine()
    
    @pytest.fixture
    def sample_user_profile(self):
        return {
            'user_id': 'test_user_pathways',
            'skills': {'programming': 0.6, 'problem_solving': 0.7},
            'personality': {'openness': 0.8, 'conscientiousness': 0.7},
            'interests': ['technology', 'science'],
            'challenge_preference': 0.8
        }
    
    @pytest.mark.asyncio
    async def test_pathway_creation(self, pathways_engine, sample_user_profile):
        """Test gamified pathway creation"""
        learning_objectives = ['master python programming', 'learn data structures', 'build projects']
        
        pathway_preferences = {
            'pathway_type': 'skill_tree_mastery',
            'narrative_theme': 'space_exploration',
            'team_size': 1
        }
        
        content_library = {
            'available_content': ['videos', 'exercises', 'projects'],
            'difficulty_levels': [0.3, 0.5, 0.7, 0.9]
        }
        
        result = await pathways_engine.create_gamified_pathway(
            learning_objectives, sample_user_profile, pathway_preferences, content_library
        )
        
        assert result['status'] == 'success'
        assert 'gamified_pathway' in result
        
        # Verify pathway structure
        pathway = result['gamified_pathway']
        assert pathway['title']
        assert pathway['description']
        assert pathway['pathway_type']
        assert 'pathway_nodes' in pathway
        assert len(pathway['pathway_nodes']) > 0
        assert 'progression_mechanics' in pathway
        assert 'difficulty_curve' in pathway
    
    @pytest.mark.asyncio
    async def test_adaptive_difficulty(self, difficulty_engine):
        """Test adaptive difficulty adjustment"""
        user_id = 'test_user_difficulty'
        pathway_id = 'test_pathway'
        
        performance_data = {
            'recent_scores': [0.9, 0.85, 0.95],
            'completion_times': [25, 30, 20],  # minutes
            'attempt_counts': [1, 1, 2]
        }
        
        current_difficulty = 0.6
        
        result = await difficulty_engine.adapt_pathway_difficulty(
            user_id, pathway_id, performance_data, current_difficulty
        )
        
        assert result['status'] == 'success'
        assert 'adapted_difficulty' in result
        assert 'adaptation_rationale' in result
        
        # Verify difficulty adaptation
        adapted_difficulty = result['adapted_difficulty']
        assert isinstance(adapted_difficulty, float)
        assert 0.1 <= adapted_difficulty <= 1.0
        
        # High performance should increase difficulty
        assert adapted_difficulty >= current_difficulty


class TestSocialCompetition:
    """Test social competition components"""
    
    @pytest.fixture
    def competition_engine(self):
        return SocialCompetitionEngine()
    
    @pytest.fixture
    def leaderboard_system(self):
        return LeaderboardSystem()
    
    @pytest.fixture
    def competitive_analytics(self):
        return CompetitiveAnalytics()
    
    @pytest.fixture
    def sample_participants(self):
        return [
            {
                'user_id': 'user1',
                'skill_level': 0.7,
                'social_preference': 0.8,
                'collaboration_preference': 0.6
            },
            {
                'user_id': 'user2',
                'skill_level': 0.6,
                'social_preference': 0.7,
                'collaboration_preference': 0.8
            },
            {
                'user_id': 'user3',
                'skill_level': 0.8,
                'social_preference': 0.5,
                'collaboration_preference': 0.7
            },
            {
                'user_id': 'user4',
                'skill_level': 0.5,
                'social_preference': 0.9,
                'collaboration_preference': 0.9
            },
            {
                'user_id': 'user5',
                'skill_level': 0.9,
                'social_preference': 0.6,
                'collaboration_preference': 0.5
            }
        ]
    
    @pytest.mark.asyncio
    async def test_competition_creation(self, competition_engine, sample_participants):
        """Test social competition creation"""
        competition_config = {
            'competition_type': 'team_challenge',
            'duration_days': 14,
            'team_size': 2
        }
        
        learning_objectives = ['improve coding skills', 'learn collaboration']
        
        result = await competition_engine.create_social_competition(
            competition_config, sample_participants, learning_objectives
        )
        
        assert result['status'] == 'success'
        assert 'social_competition' in result
        
        # Verify competition structure
        competition = result['social_competition']
        assert competition['title']
        assert competition['description']
        assert competition['competition_type']
        assert len(competition['participants']) == len(sample_participants)
        assert 'scoring_system' in competition
        assert 'prizes_rewards' in competition
    
    @pytest.mark.asyncio
    async def test_leaderboard_creation(self, leaderboard_system, sample_participants):
        """Test leaderboard creation and updates"""
        leaderboard_config = {
            'title': 'Test Competition Leaderboard',
            'type': 'global_ranking',
            'update_frequency': 'real_time'
        }
        
        ranking_criteria = {
            'total_score': 0.6,
            'completion_rate': 0.3,
            'collaboration_score': 0.1
        }
        
        result = await leaderboard_system.create_leaderboard(
            leaderboard_config, sample_participants, ranking_criteria
        )
        
        assert result['status'] == 'success'
        assert 'leaderboard' in result
        
        # Test leaderboard update
        leaderboard_id = result['leaderboard']['leaderboard_id']
        
        participant_scores = {
            'user1': {'score_components': {'total_score': 850, 'completion_rate': 0.9}},
            'user2': {'score_components': {'total_score': 750, 'completion_rate': 0.8}}
        }
        
        update_result = await leaderboard_system.update_leaderboard(
            leaderboard_id, participant_scores
        )
        
        assert update_result['status'] == 'success'
        assert 'updated_leaderboard' in update_result
        assert 'ranking_changes' in update_result
    
    @pytest.mark.asyncio
    async def test_competitive_analytics(self, competitive_analytics, sample_participants):
        """Test competitive analytics"""
        competition_id = 'test_competition_123'
        
        competition_data = {
            'competition_type': 'individual_leaderboard',
            'duration_days': 7,
            'participant_count': len(sample_participants)
        }
        
        participant_data = [
            {
                'user_id': 'user1',
                'total_score': 850,
                'engagement_score': 0.8,
                'completion_status': 'completed',
                'initial_score': 600,
                'final_score': 850
            },
            {
                'user_id': 'user2',
                'total_score': 750,
                'engagement_score': 0.7,
                'completion_status': 'completed',
                'initial_score': 500,
                'final_score': 750
            }
        ]
        
        result = await competitive_analytics.analyze_competition_performance(
            competition_id, competition_data, participant_data
        )
        
        assert result['status'] == 'success'
        assert 'competition_analytics' in result
        
        # Verify analytics structure
        analytics = result['competition_analytics']
        assert 'participation_metrics' in analytics
        assert 'engagement_analysis' in analytics
        assert 'performance_analysis' in analytics
        assert 'effectiveness_analysis' in analytics


class TestGamificationOrchestrator:
    """Test gamification orchestrator components"""
    
    @pytest.fixture
    def gamification_engine(self):
        return AdvancedGamificationEngine()
    
    @pytest.fixture
    def orchestrator(self, gamification_engine):
        return GamificationOrchestrator(gamification_engine)
    
    @pytest.fixture
    def comprehensive_user_profile(self):
        return {
            'user_id': 'test_user_orchestrator',
            'name': 'Test User',
            'skills': {'python': 0.7, 'data_science': 0.5, 'problem_solving': 0.8},
            'personality': {
                'conscientiousness': 0.8,
                'extraversion': 0.6,
                'openness': 0.7,
                'agreeableness': 0.7,
                'neuroticism': 0.3
            },
            'learning_style': {'visual': 0.8, 'kinesthetic': 0.6, 'auditory': 0.4},
            'preferences': {
                'challenge_preference': 0.8,
                'social_learning_preference': 0.6,
                'structure_preference': 0.7,
                'gamification_preference': 0.9
            },
            'interests': ['technology', 'science', 'problem_solving'],
            'learning_goals': ['master python', 'learn machine learning', 'build projects']
        }
    
    @pytest.mark.asyncio
    async def test_comprehensive_gamification_session(self, gamification_engine, comprehensive_user_profile):
        """Test comprehensive gamification session creation"""
        learning_context = {
            'learning_objectives': ['improve coding skills', 'increase motivation', 'build confidence'],
            'current_performance': 0.7,
            'engagement_data': {
                'current_engagement_score': 0.6,
                'engagement_trend': 'stable'
            },
            'system_metrics': {
                'user_retention_rate': 0.75,
                'average_engagement': 0.65
            }
        }
        
        session_preferences = {
            'session_type': 'comprehensive',
            'preferred_focus': 'motivation',
            'gamification_intensity': 'balanced',
            'duration_minutes': 60,
            'goals': ['increase engagement', 'improve performance']
        }
        
        result = await gamification_engine.create_comprehensive_gamification_session(
            comprehensive_user_profile, learning_context, session_preferences
        )
        
        assert result['status'] == 'success'
        assert 'gamification_session' in result
        assert 'initial_insights' in result
        
        # Verify session structure
        session = result['gamification_session']
        assert session['user_id'] == comprehensive_user_profile['user_id']
        assert session['session_type'] == 'comprehensive'
        assert 'active_components' in session
        assert len(session['active_components']) > 0
        assert 'session_data' in session
        
        # Verify insights
        insights = result['initial_insights']
        assert isinstance(insights, list)
        if insights:  # May be empty in some test scenarios
            assert 'insight_type' in insights[0]
            assert 'message' in insights[0]
    
    @pytest.mark.asyncio
    async def test_quick_gamification_session(self, orchestrator, comprehensive_user_profile):
        """Test quick gamification session creation using templates"""
        session_type = 'motivation_boost'
        learning_objectives = ['increase motivation', 'improve engagement']
        
        result = await orchestrator.create_quick_gamification_session(
            session_type, comprehensive_user_profile, learning_objectives
        )
        
        assert result['status'] == 'success'
        assert 'gamification_session' in result
        assert 'template_info' in result
        
        # Verify template usage
        template_info = result['template_info']
        assert template_info['template_used'] == session_type
        assert 'template_focus' in template_info
        assert 'template_components' in template_info
        
        # Verify session matches template
        session = result['gamification_session']
        assert session['primary_focus'] == 'motivation'
    
    @pytest.mark.asyncio
    async def test_session_adaptation(self, orchestrator, gamification_engine, comprehensive_user_profile):
        """Test real-time session adaptation"""
        # First create a session
        session_result = await orchestrator.create_quick_gamification_session(
            'achievement_focused', comprehensive_user_profile, ['improve skills']
        )
        
        session_id = session_result['gamification_session']['session_id']
        
        # Test adaptation
        performance_data = {
            'engagement_score': 0.4,  # Low engagement
            'performance_score': 0.9   # High performance
        }
        
        user_feedback = {
            'satisfaction_score': 0.5,  # Low satisfaction
            'reward_satisfaction': 0.6
        }
        
        adaptation_result = await orchestrator.adapt_session_real_time(
            session_id, performance_data, user_feedback
        )
        
        assert adaptation_result['status'] == 'success'
        assert 'adaptations_applied' in adaptation_result
        assert 'adaptation_analysis' in adaptation_result
        
        # Verify adaptations
        adaptations = adaptation_result['adaptations_applied']
        assert isinstance(adaptations, list)
        
        # Should have engagement and difficulty adaptations
        adaptation_types = [a['type'] for a in adaptations]
        assert 'engagement_boost' in adaptation_types or 'difficulty_adjustment' in adaptation_types


# Integration test
class TestGamificationIntegration:
    """Test integration between gamification components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_gamification_workflow(self):
        """Test complete gamification workflow from user profile to session completion"""
        # Create comprehensive user profile
        user_profile = {
            'user_id': 'integration_test_user',
            'name': 'Integration Test User',
            'skills': {'python': 0.6, 'javascript': 0.4},
            'personality': {'conscientiousness': 0.8, 'extraversion': 0.7},
            'learning_style': {'visual': 0.8},
            'preferences': {'gamification_preference': 0.9},
            'learning_goals': ['master web development']
        }
        
        # Initialize gamification engine
        gamification_engine = AdvancedGamificationEngine()
        
        # Create learning context
        learning_context = {
            'learning_objectives': ['improve coding skills', 'build projects'],
            'engagement_data': {'current_engagement_score': 0.6},
            'system_metrics': {'average_engagement': 0.65}
        }
        
        # Create session preferences
        session_preferences = {
            'preferred_focus': 'achievement',
            'gamification_intensity': 'balanced',
            'duration_minutes': 45
        }
        
        # Create comprehensive gamification session
        session_result = await gamification_engine.create_comprehensive_gamification_session(
            user_profile, learning_context, session_preferences
        )
        
        # Verify successful session creation
        assert session_result['status'] == 'success'
        session = session_result['gamification_session']
        
        # Verify session has multiple active components
        assert len(session['active_components']) >= 2
        
        # Verify session data contains component results
        session_data = session['session_data']
        assert 'component_results' in session_data
        assert 'integrated_experience' in session_data
        
        # Test that components work together
        component_results = session_data['component_results']
        successful_components = [
            comp for comp, result in component_results.items() 
            if result.get('status') == 'success'
        ]
        
        # Should have at least some successful components
        assert len(successful_components) >= 1
        
        # Verify integration
        integrated_experience = session_data['integrated_experience']
        assert 'estimated_engagement_boost' in integrated_experience
        assert integrated_experience['estimated_engagement_boost'] > 0


    @pytest.mark.asyncio
    async def test_component_interaction_reward_achievement(self):
        """Test interaction between reward systems and achievement engine"""
        # Initialize components
        reward_engine = RewardOptimizationEngine()
        achievement_generator = DynamicAchievementGenerator()

        user_profile = {
            'user_id': 'interaction_test_user',
            'skills': {'python': 0.7},
            'reward_history': [],
            'achievement_preferences': {'preferred_types': ['mastery_badge']}
        }

        # Generate achievement
        achievement_result = await achievement_generator.generate_personalized_achievement(
            user_profile, {}, user_profile['achievement_preferences']
        )

        assert achievement_result['status'] == 'success'
        achievement = achievement_result['dynamic_achievement']

        # Calculate reward for achievement completion
        reward_calculator = DynamicRewardCalculator(PsychologicalRewardAnalyzer())
        reward_result = await reward_calculator.calculate_dynamic_reward(
            user_profile['user_id'],
            'achievement_completion',
            {
                'achievement_difficulty': achievement['difficulty_tier'] / 10,
                'achievement_rarity': achievement['rarity_score'],
                'completion_quality': 0.9
            }
        )

        assert reward_result['status'] == 'success'
        reward = reward_result['dynamic_reward']

        # Verify reward is enhanced by achievement context
        assert reward['total_value'] > reward['base_value']
        assert reward['psychological_multiplier'] > 1.0

    @pytest.mark.asyncio
    async def test_motivation_engagement_synergy(self):
        """Test synergy between motivation enhancement and engagement mechanics"""
        # Initialize components
        motivation_analyzer = LearningMotivationAnalyzer()
        engagement_engine = EngagementMechanicsEngine()

        user_profile = {
            'user_id': 'synergy_test_user',
            'personality': {'extraversion': 0.8, 'openness': 0.7},
            'preferences': {'social_learning_preference': 0.8}
        }

        # Analyze motivation
        motivation_result = await motivation_analyzer.analyze_learning_motivation(
            user_profile['user_id'],
            {'learning_goals': ['improve skills']},
            {'social_interaction_boost': True, 'peer_learning_preference': 0.8},
            {'social_support_level': 0.7}
        )

        assert motivation_result['status'] == 'success'
        motivation_profile = motivation_result['motivation_profile']

        # Optimize engagement based on motivation profile
        engagement_data = {
            'current_engagement_score': 0.6,
            'social_activity_level': 0.7
        }

        learning_context = {
            'motivation_profile': motivation_profile,
            'social_context': 'group_learning'
        }

        engagement_result = await engagement_engine.optimize_engagement_mechanics(
            user_profile['user_id'], user_profile, engagement_data, learning_context
        )

        assert engagement_result['status'] == 'success'

        # Verify engagement mechanics align with motivation type
        optimal_mechanics = engagement_result['optimal_mechanics']
        mechanic_types = [m.get('mechanic_type', '') for m in optimal_mechanics]

        # Should include social mechanics for social motivation type
        assert any('social' in mtype for mtype in mechanic_types)

    @pytest.mark.asyncio
    async def test_pathway_difficulty_adaptation_integration(self):
        """Test integration between gamified pathways and adaptive difficulty"""
        # Initialize components
        pathways_engine = GamifiedLearningPathways()
        difficulty_engine = AdaptiveDifficultyEngine()

        user_profile = {
            'user_id': 'pathway_test_user',
            'skills': {'programming': 0.6},
            'challenge_preference': 0.7
        }

        # Create pathway
        pathway_result = await pathways_engine.create_gamified_pathway(
            ['learn programming'], user_profile, {}, {}
        )

        assert pathway_result['status'] == 'success'
        pathway = pathway_result['gamified_pathway']

        # Simulate performance data
        performance_data = {
            'recent_scores': [0.9, 0.95, 0.85],  # High performance
            'completion_times': [20, 18, 22],
            'attempt_counts': [1, 1, 1]
        }

        # Adapt difficulty
        current_difficulty = pathway['difficulty_curve']['min_difficulty']
        adaptation_result = await difficulty_engine.adapt_pathway_difficulty(
            user_profile['user_id'], pathway['pathway_id'], performance_data, current_difficulty
        )

        assert adaptation_result['status'] == 'success'

        # Verify difficulty increased due to high performance
        adapted_difficulty = adaptation_result['adapted_difficulty']
        assert adapted_difficulty > current_difficulty

    @pytest.mark.asyncio
    async def test_social_individual_balance(self):
        """Test balance between social competition and individual progress"""
        # Initialize components
        competition_engine = SocialCompetitionEngine()
        mastery_tracker = MasteryProgressTracker()

        # Create participants
        participants = [
            {'user_id': f'user_{i}', 'skill_level': 0.5 + i * 0.1}
            for i in range(5)
        ]

        # Create social competition
        competition_result = await competition_engine.create_social_competition(
            {'competition_type': 'individual_leaderboard'}, participants, ['improve skills']
        )

        assert competition_result['status'] == 'success'

        # Track individual mastery for one participant
        mastery_result = await mastery_tracker.track_mastery_progress(
            'user_0', {'programming': 0.8}, []
        )

        assert mastery_result['status'] == 'success'

        # Verify both social and individual elements are present
        competition = competition_result['social_competition']
        mastery_profile = mastery_result['mastery_profile']

        assert len(competition['participants']) > 1  # Social element
        assert mastery_profile['user_id'] == 'user_0'  # Individual element
        assert 'skill_masteries' in mastery_profile  # Individual progress


# Performance and stress tests
class TestGamificationPerformance:
    """Test performance and scalability of gamification systems"""

    @pytest.mark.asyncio
    async def test_large_scale_reward_optimization(self):
        """Test reward optimization with large user base"""
        reward_engine = RewardOptimizationEngine()

        # Create large user dataset
        user_data = [
            {
                'user_id': f'user_{i}',
                'engagement_score': 0.5 + (i % 5) * 0.1,
                'reward_history': [],
                'personalization_score': 0.6 + (i % 3) * 0.1
            }
            for i in range(100)  # 100 users
        ]

        system_metrics = {
            'user_retention_rate': 0.75,
            'average_engagement': 0.65
        }

        optimization_goals = {
            'target_engagement': 0.8
        }

        # Measure performance
        import time
        start_time = time.time()

        result = await reward_engine.optimize_reward_system(
            user_data, system_metrics, optimization_goals
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify successful processing
        assert result['status'] == 'success'

        # Verify reasonable performance (should complete within 10 seconds)
        assert processing_time < 10.0

        # Verify optimization recommendations are provided
        assert 'optimization_recommendations' in result
        assert len(result['optimization_recommendations']) > 0

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self):
        """Test concurrent gamification session creation"""
        gamification_engine = AdvancedGamificationEngine()

        # Create multiple user profiles
        user_profiles = [
            {
                'user_id': f'concurrent_user_{i}',
                'skills': {'skill_1': 0.5 + i * 0.1},
                'personality': {'conscientiousness': 0.6 + i * 0.05},
                'preferences': {'gamification_preference': 0.8}
            }
            for i in range(10)
        ]

        learning_context = {
            'learning_objectives': ['test objective'],
            'engagement_data': {'current_engagement_score': 0.6}
        }

        session_preferences = {
            'preferred_focus': 'motivation',
            'duration_minutes': 30
        }

        # Create sessions concurrently
        tasks = [
            gamification_engine.create_comprehensive_gamification_session(
                profile, learning_context, session_preferences
            )
            for profile in user_profiles
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all sessions created successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        assert len(successful_results) == len(user_profiles)

        # Verify unique session IDs
        session_ids = [r['gamification_session']['session_id'] for r in successful_results]
        assert len(set(session_ids)) == len(session_ids)  # All unique

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_competition(self):
        """Test memory efficiency with large competition"""
        competition_engine = SocialCompetitionEngine()

        # Create large participant pool
        participants = [
            {
                'user_id': f'participant_{i}',
                'skill_level': 0.3 + (i % 7) * 0.1,
                'social_preference': 0.5 + (i % 3) * 0.15,
                'collaboration_preference': 0.4 + (i % 4) * 0.15
            }
            for i in range(50)  # 50 participants
        ]

        competition_config = {
            'competition_type': 'individual_leaderboard',
            'duration_days': 7
        }

        learning_objectives = ['improve performance']

        # Create competition
        result = await competition_engine.create_social_competition(
            competition_config, participants, learning_objectives
        )

        assert result['status'] == 'success'

        # Verify competition structure is reasonable
        competition = result['social_competition']
        assert len(competition['participants']) == len(participants)
        assert 'scoring_system' in competition

        # Verify memory usage is reasonable (competition object should be manageable)
        import sys
        competition_size = sys.getsizeof(str(competition))
        assert competition_size < 1000000  # Less than 1MB when serialized


# Edge case and error handling tests
class TestGamificationEdgeCases:
    """Test edge cases and error handling in gamification systems"""

    @pytest.mark.asyncio
    async def test_empty_user_profile_handling(self):
        """Test handling of empty or minimal user profiles"""
        gamification_engine = AdvancedGamificationEngine()

        # Minimal user profile
        minimal_profile = {'user_id': 'minimal_user'}

        learning_context = {
            'learning_objectives': ['basic objective']
        }

        session_preferences = {
            'preferred_focus': 'motivation'
        }

        result = await gamification_engine.create_comprehensive_gamification_session(
            minimal_profile, learning_context, session_preferences
        )

        # Should handle gracefully and still create session
        assert result['status'] == 'success'
        session = result['gamification_session']
        assert session['user_id'] == 'minimal_user'
        assert len(session['active_components']) > 0

    @pytest.mark.asyncio
    async def test_invalid_competition_participants(self):
        """Test handling of invalid competition participant data"""
        competition_engine = SocialCompetitionEngine()

        # Too few participants
        few_participants = [
            {'user_id': 'user1', 'skill_level': 0.5},
            {'user_id': 'user2', 'skill_level': 0.6}
        ]

        result = await competition_engine.create_social_competition(
            {'competition_type': 'individual_leaderboard'}, few_participants, []
        )

        # Should return error for insufficient participants
        assert result['status'] == 'error'
        assert 'Insufficient participants' in result['error']

        # Missing required fields
        invalid_participants = [
            {'user_id': 'user1'},  # Missing skill_level
            {'skill_level': 0.6}   # Missing user_id
        ]

        result = await competition_engine.create_social_competition(
            {'competition_type': 'individual_leaderboard'}, invalid_participants, []
        )

        # Should return error for missing fields
        assert result['status'] == 'error'
        assert 'missing required field' in result['error']

    @pytest.mark.asyncio
    async def test_extreme_difficulty_values(self):
        """Test handling of extreme difficulty values"""
        difficulty_engine = AdaptiveDifficultyEngine()

        # Test with extreme performance data
        extreme_performance = {
            'recent_scores': [1.0, 1.0, 1.0],  # Perfect scores
            'completion_times': [1, 1, 1],     # Very fast
            'attempt_counts': [1, 1, 1]        # First attempt success
        }

        # Start with maximum difficulty
        max_difficulty = 1.0

        result = await difficulty_engine.adapt_pathway_difficulty(
            'test_user', 'test_pathway', extreme_performance, max_difficulty
        )

        assert result['status'] == 'success'

        # Should not exceed maximum bounds
        adapted_difficulty = result['adapted_difficulty']
        assert adapted_difficulty <= 1.0

        # Test with minimum difficulty and poor performance
        poor_performance = {
            'recent_scores': [0.1, 0.2, 0.1],  # Very low scores
            'completion_times': [120, 150, 180],  # Very slow
            'attempt_counts': [5, 6, 4]        # Multiple attempts
        }

        min_difficulty = 0.1

        result = await difficulty_engine.adapt_pathway_difficulty(
            'test_user', 'test_pathway', poor_performance, min_difficulty
        )

        assert result['status'] == 'success'

        # Should not go below minimum bounds
        adapted_difficulty = result['adapted_difficulty']
        assert adapted_difficulty >= 0.1

    @pytest.mark.asyncio
    async def test_malformed_learning_objectives(self):
        """Test handling of malformed learning objectives"""
        pathways_engine = GamifiedLearningPathways()

        user_profile = {'user_id': 'test_user'}

        # Empty objectives
        result = await pathways_engine.create_gamified_pathway(
            [], user_profile, {}, {}
        )

        # Should handle gracefully
        assert result['status'] == 'success'
        pathway = result['gamified_pathway']
        assert len(pathway['pathway_nodes']) > 0  # Should create default pathway

        # Very long objectives list
        long_objectives = [f'objective_{i}' for i in range(100)]

        result = await pathways_engine.create_gamified_pathway(
            long_objectives, user_profile, {}, {}
        )

        # Should handle gracefully and limit pathway size
        assert result['status'] == 'success'
        pathway = result['gamified_pathway']
        assert len(pathway['pathway_nodes']) <= 20  # Should be capped

    @pytest.mark.asyncio
    async def test_session_adaptation_edge_cases(self):
        """Test session adaptation with edge case data"""
        orchestrator = GamificationOrchestrator(AdvancedGamificationEngine())

        # Create a session first
        user_profile = {'user_id': 'edge_case_user'}
        session_result = await orchestrator.create_quick_gamification_session(
            'motivation_boost', user_profile, ['test objective']
        )

        session_id = session_result['gamification_session']['session_id']

        # Test with extreme performance data
        extreme_performance = {
            'engagement_score': -0.5,  # Invalid negative value
            'performance_score': 1.5   # Invalid value > 1.0
        }

        extreme_feedback = {
            'satisfaction_score': 2.0,  # Invalid value > 1.0
            'reward_satisfaction': -1.0  # Invalid negative value
        }

        result = await orchestrator.adapt_session_real_time(
            session_id, extreme_performance, extreme_feedback
        )

        # Should handle gracefully
        assert result['status'] == 'success'
        assert 'adaptations_applied' in result


# Mock and fixture utilities
class TestGamificationMocks:
    """Test gamification systems with various mock scenarios"""

    @pytest.mark.asyncio
    async def test_with_mock_cache_service(self):
        """Test gamification components with mock cache service"""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)

        # Test reward engine with mock cache
        reward_engine = RewardOptimizationEngine(mock_cache)

        user_data = [{'user_id': 'test_user', 'engagement_score': 0.7}]
        system_metrics = {'average_engagement': 0.6}
        optimization_goals = {'target_engagement': 0.8}

        result = await reward_engine.optimize_reward_system(
            user_data, system_metrics, optimization_goals
        )

        assert result['status'] == 'success'

        # Verify cache interactions
        mock_cache.get.assert_called()
        mock_cache.set.assert_called()

    @pytest.mark.asyncio
    async def test_with_mock_external_services(self):
        """Test gamification with mock external service dependencies"""
        with patch('quantum_intelligence.services.gamification.reward_systems.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = '2024-01-01T00:00:00'

            achievement_generator = DynamicAchievementGenerator()

            user_profile = {'user_id': 'mock_test_user'}
            learning_context = {}
            achievement_preferences = {}

            result = await achievement_generator.generate_personalized_achievement(
                user_profile, learning_context, achievement_preferences
            )

            assert result['status'] == 'success'
            achievement = result['dynamic_achievement']
            assert achievement['created_at'] == '2024-01-01T00:00:00'


# Configuration and setup tests
class TestGamificationConfiguration:
    """Test gamification system configuration and setup"""

    def test_reward_system_initialization(self):
        """Test reward system proper initialization"""
        reward_engine = RewardOptimizationEngine()

        # Verify default configuration
        assert hasattr(reward_engine, 'config')
        assert 'optimization_algorithms' in reward_engine.config
        assert 'performance_thresholds' in reward_engine.config

        # Verify component initialization
        assert hasattr(reward_engine, 'psychological_analyzer')
        assert hasattr(reward_engine, 'reward_calculator')
        assert hasattr(reward_engine, 'personalization_system')

    def test_achievement_engine_initialization(self):
        """Test achievement engine proper initialization"""
        achievement_generator = DynamicAchievementGenerator()

        # Verify configuration
        assert hasattr(achievement_generator, 'config')
        assert 'achievement_templates' in achievement_generator.config
        assert 'difficulty_scaling' in achievement_generator.config

        # Verify tracking systems
        achievement_tracker = AchievementTrackingSystem(achievement_generator)
        assert hasattr(achievement_tracker, 'achievement_generator')
        assert hasattr(achievement_tracker, 'user_progress')

    def test_orchestrator_initialization(self):
        """Test orchestrator proper initialization with all components"""
        gamification_engine = AdvancedGamificationEngine()

        # Verify all components are initialized
        assert hasattr(gamification_engine, 'reward_engine')
        assert hasattr(gamification_engine, 'achievement_generator')
        assert hasattr(gamification_engine, 'engagement_engine')
        assert hasattr(gamification_engine, 'motivation_analyzer')
        assert hasattr(gamification_engine, 'pathways_engine')
        assert hasattr(gamification_engine, 'competition_engine')

        # Verify configuration
        assert hasattr(gamification_engine, 'config')
        assert 'component_weights' in gamification_engine.config
        assert 'mode_configurations' in gamification_engine.config

        # Test orchestrator wrapper
        orchestrator = GamificationOrchestrator(gamification_engine)
        assert hasattr(orchestrator, 'gamification_engine')
        assert hasattr(orchestrator, 'config')
        assert 'session_templates' in orchestrator.config


# Comprehensive integration scenarios
class TestGamificationScenarios:
    """Test complete gamification scenarios and user journeys"""

    @pytest.mark.asyncio
    async def test_new_user_onboarding_scenario(self):
        """Test complete new user onboarding with gamification"""
        # Initialize system
        gamification_engine = AdvancedGamificationEngine()
        orchestrator = GamificationOrchestrator(gamification_engine)

        # New user profile (minimal data)
        new_user = {
            'user_id': 'new_user_001',
            'name': 'New User',
            'skills': {},  # No skills yet
            'learning_goals': ['learn programming basics']
        }

        # Create onboarding session
        result = await orchestrator.create_quick_gamification_session(
            'motivation_boost', new_user, ['get started with learning']
        )

        assert result['status'] == 'success'
        session = result['gamification_session']

        # Verify onboarding-appropriate configuration
        assert session['personalization_level'] >= 0.3  # Some personalization even with minimal data
        assert len(session['active_components']) >= 2  # Multiple engagement strategies

        # Simulate first learning activity
        performance_data = {
            'engagement_score': 0.8,  # High initial engagement
            'performance_score': 0.6   # Moderate performance
        }

        user_feedback = {
            'satisfaction_score': 0.7,
            'difficulty_perception': 'appropriate'
        }

        # Adapt session based on initial activity
        adaptation_result = await orchestrator.adapt_session_real_time(
            session['session_id'], performance_data, user_feedback
        )

        assert adaptation_result['status'] == 'success'

        # Should maintain engagement for new user
        adaptations = adaptation_result['adaptations_applied']
        adaptation_types = [a['type'] for a in adaptations]
        assert any('engagement' in atype for atype in adaptation_types)

    @pytest.mark.asyncio
    async def test_experienced_user_challenge_scenario(self):
        """Test gamification for experienced user seeking challenges"""
        gamification_engine = AdvancedGamificationEngine()

        # Experienced user profile
        experienced_user = {
            'user_id': 'experienced_user_001',
            'name': 'Expert User',
            'skills': {
                'python': 0.9,
                'javascript': 0.8,
                'data_science': 0.85,
                'machine_learning': 0.7
            },
            'personality': {
                'conscientiousness': 0.9,
                'openness': 0.8,
                'extraversion': 0.6
            },
            'preferences': {
                'challenge_preference': 0.95,
                'gamification_preference': 0.8
            },
            'learning_goals': ['master advanced algorithms', 'contribute to open source']
        }

        learning_context = {
            'learning_objectives': ['advanced skill development', 'expert-level challenges'],
            'current_performance': 0.85,
            'engagement_data': {'current_engagement_score': 0.7}
        }

        session_preferences = {
            'preferred_focus': 'achievement',
            'gamification_intensity': 'intensive',
            'duration_minutes': 90
        }

        # Create advanced session
        result = await gamification_engine.create_comprehensive_gamification_session(
            experienced_user, learning_context, session_preferences
        )

        assert result['status'] == 'success'
        session = result['gamification_session']

        # Verify advanced configuration
        assert session['gamification_mode'] == 'intensive'
        assert session['primary_focus'] == 'achievement'
        assert session['personalization_level'] > 0.8

        # Should have achievement and mastery components
        active_components = session['active_components']
        assert 'achievement_generation' in active_components
        assert 'mastery_tracking' in active_components

    @pytest.mark.asyncio
    async def test_team_learning_scenario(self):
        """Test gamification for team-based learning scenario"""
        competition_engine = SocialCompetitionEngine()

        # Team of learners
        team_members = [
            {
                'user_id': f'team_member_{i}',
                'skill_level': 0.5 + i * 0.1,
                'social_preference': 0.8,
                'collaboration_preference': 0.9,
                'personality': {'extraversion': 0.7 + i * 0.05}
            }
            for i in range(6)
        ]

        # Create team challenge
        team_config = {
            'competition_type': 'team_challenge',
            'team_size': 3,
            'duration_days': 14,
            'team_formation_strategy': 'skill_balanced'
        }

        learning_objectives = ['collaborative problem solving', 'peer learning', 'team communication']

        result = await competition_engine.create_social_competition(
            team_config, team_members, learning_objectives
        )

        assert result['status'] == 'success'
        competition = result['social_competition']

        # Verify team formation
        assert len(competition['teams']) == 2  # 6 members / 3 per team
        assert competition['competition_type'] == 'team_challenge'

        # Verify team balance
        for team in competition['teams']:
            assert len(team['members']) == 3
            team_skills = [member['skill_level'] for member in team['members']]
            # Teams should have balanced skill levels
            skill_variance = max(team_skills) - min(team_skills)
            assert skill_variance < 0.5  # Reasonable balance

    @pytest.mark.asyncio
    async def test_long_term_progression_scenario(self):
        """Test long-term learning progression with gamification"""
        pathways_engine = GamifiedLearningPathways()
        mastery_tracker = MasteryProgressTracker()

        # User starting learning journey
        learning_user = {
            'user_id': 'progression_user_001',
            'skills': {'programming': 0.2},  # Beginner level
            'learning_goals': ['become proficient programmer', 'build real projects'],
            'challenge_preference': 0.6
        }

        # Create long-term pathway
        learning_objectives = [
            'master programming fundamentals',
            'learn data structures and algorithms',
            'build web applications',
            'contribute to open source projects'
        ]

        pathway_preferences = {
            'pathway_type': 'skill_tree_mastery',
            'narrative_theme': 'entrepreneurial_quest'
        }

        pathway_result = await pathways_engine.create_gamified_pathway(
            learning_objectives, learning_user, pathway_preferences, {}
        )

        assert pathway_result['status'] == 'success'
        pathway = pathway_result['gamified_pathway']

        # Verify long-term structure
        assert len(pathway['pathway_nodes']) >= 8  # Substantial pathway
        assert pathway['estimated_duration_hours'] >= 40  # Long-term commitment

        # Simulate progression over time
        initial_skills = {'programming': 0.2}

        # After some learning activities
        intermediate_skills = {'programming': 0.6, 'web_development': 0.4}

        # Track mastery progression
        mastery_result = await mastery_tracker.track_mastery_progress(
            learning_user['user_id'], intermediate_skills, []
        )

        assert mastery_result['status'] == 'success'
        mastery_profile = mastery_result['mastery_profile']

        # Verify progression tracking
        assert mastery_profile['overall_mastery_level'] > 0.3
        assert 'programming' in mastery_profile['skill_masteries']

        # Should detect improvement
        improvements = mastery_result.get('mastery_insights', [])
        improvement_messages = [insight['message'] for insight in improvements]
        assert any('improvement' in msg.lower() or 'progress' in msg.lower() for msg in improvement_messages)


# Final test runner configuration
if __name__ == '__main__':
    # Configure pytest with appropriate options
    pytest_args = [
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--asyncio-mode=auto',  # Auto-detect async tests
        '--durations=10',  # Show 10 slowest tests
    ]

    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend(['--cov=quantum_intelligence.services.gamification', '--cov-report=term-missing'])
    except ImportError:
        pass

    pytest.main(pytest_args)
