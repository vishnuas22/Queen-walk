"""
Quantum Intelligence Gamification Services

Advanced gamification system for learning platforms, extracted from the quantum
intelligence engine. Provides comprehensive gamification mechanics including
reward optimization, achievement generation, engagement enhancement, motivation
analysis, gamified pathways, and social competition.

This package contains:
- Reward Systems: AI-driven reward optimization and psychological analysis
- Achievement Engine: Dynamic achievement generation and tracking
- Engagement Mechanics: Sophisticated engagement optimization
- Motivation Enhancement: Deep motivation analysis and personalized plans
- Gamified Pathways: Immersive learning journeys with adaptive difficulty
- Social Competition: Team challenges, leaderboards, and analytics
- Orchestrator: High-level integration and coordination
"""

# Version information
__version__ = "1.0.0"
__author__ = "Quantum Intelligence Team"

# Import main orchestration classes for easy access
from .orchestrator import (
    AdvancedGamificationEngine,
    GamificationOrchestrator,
    GamificationMode,
    GamificationFocus
)

# Import core component classes
from .reward_systems import (
    RewardOptimizationEngine,
    PsychologicalRewardAnalyzer,
    DynamicRewardCalculator,
    RewardPersonalizationSystem
)

from .achievement_engine import (
    DynamicAchievementGenerator,
    AchievementTrackingSystem,
    BadgeDesignEngine,
    MasteryProgressTracker
)

from .engagement_mechanics import (
    EngagementMechanicsEngine,
    ChallengeGenerationSystem
)

from .motivation_enhancement import (
    LearningMotivationAnalyzer,
    PersonalizedMotivationSystem
)

from .gamified_pathways import (
    GamifiedLearningPathways,
    AdaptiveDifficultyEngine
)

from .social_competition import (
    SocialCompetitionEngine,
    LeaderboardSystem,
    CompetitiveAnalytics
)

# Define what gets imported with "from gamification import *"
__all__ = [
    # Main orchestration
    "AdvancedGamificationEngine",
    "GamificationOrchestrator",
    "GamificationMode",
    "GamificationFocus",

    # Reward systems
    "RewardOptimizationEngine",
    "PsychologicalRewardAnalyzer",
    "DynamicRewardCalculator",
    "RewardPersonalizationSystem",

    # Achievement engine
    "DynamicAchievementGenerator",
    "AchievementTrackingSystem",
    "BadgeDesignEngine",
    "MasteryProgressTracker",

    # Engagement mechanics
    "EngagementMechanicsEngine",
    "ChallengeGenerationSystem",

    # Motivation enhancement
    "LearningMotivationAnalyzer",
    "PersonalizedMotivationSystem",

    # Gamified pathways
    "GamifiedLearningPathways",
    "AdaptiveDifficultyEngine",

    # Social competition
    "SocialCompetitionEngine",
    "LeaderboardSystem",
    "CompetitiveAnalytics",
]


# Convenience functions for quick setup
def create_basic_gamification_engine(cache_service=None):
    """
    Create a basic gamification engine with default configuration.

    Args:
        cache_service: Optional cache service for performance optimization

    Returns:
        AdvancedGamificationEngine: Configured gamification engine
    """
    return AdvancedGamificationEngine(cache_service)


def create_gamification_orchestrator(cache_service=None):
    """
    Create a complete gamification orchestrator with all components.

    Args:
        cache_service: Optional cache service for performance optimization

    Returns:
        GamificationOrchestrator: Complete orchestrator ready for use
    """
    engine = create_basic_gamification_engine(cache_service)
    return GamificationOrchestrator(engine)


async def quick_gamification_session(user_profile, learning_objectives, session_type='motivation_boost'):
    """
    Create a quick gamification session with minimal setup.

    Args:
        user_profile: User profile dictionary
        learning_objectives: List of learning objectives
        session_type: Type of session ('motivation_boost', 'achievement_focused', etc.)

    Returns:
        Dict: Gamification session result
    """
    orchestrator = create_gamification_orchestrator()
    return await orchestrator.create_quick_gamification_session(
        session_type, user_profile, learning_objectives
    )


# Add convenience functions to exports
__all__.extend([
    "create_basic_gamification_engine",
    "create_gamification_orchestrator",
    "quick_gamification_session"
])
