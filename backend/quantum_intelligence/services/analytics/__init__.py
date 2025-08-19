"""
Quantum Intelligence Research-Grade Analytics Services

Advanced analytics system for learning platforms, extracted from the quantum intelligence engine.
This comprehensive suite provides sophisticated analytics engines including learning pattern analysis,
cognitive load measurement, attention optimization, performance analytics, behavioral intelligence,
and research data pipelines.

This package contains:
- Learning Pattern Analysis: Multi-dimensional behavior pattern recognition
- Cognitive Load Measurement: Real-time cognitive load assessment
- Attention Optimization: Focus enhancement and distraction mitigation
- Performance Analytics: Comprehensive performance metrics and prediction
- Behavioral Intelligence: User behavior modeling and engagement analytics
- Research Pipeline: Academic-grade data collection and analysis

Example usage:
    from quantum_intelligence.services.analytics import (
        AnalyticsOrchestrator, LearningPatternAnalysisEngine
    )

    # Initialize analytics system
    orchestrator = AnalyticsOrchestrator()

    # Analyze learning patterns
    result = await orchestrator.create_analytics_session(
        user_data, learning_activities, preferences
    )
"""

# Version information
__version__ = "1.0.0"
__author__ = "Quantum Intelligence Research Team"

# Import main orchestration classes for easy access
from .orchestrator import (
    AnalyticsOrchestrator,
    AnalyticsSession,
    AnalyticsInsight,
    AnalyticsMode,
    AnalyticsFocus
)

# Import existing analytics engines (maintaining backward compatibility)
from .learning_patterns import LearningPatternAnalysisEngine
from .performance_prediction import PerformancePredictionEngine
from .research_analytics import ResearchAnalyticsEngine
from .cognitive_load import CognitiveLoadAssessmentEngine

# Import new analytics engines
from .attention_optimization import (
    AttentionOptimizationEngine,
    FocusEnhancementAlgorithms,
    AttentionState
)
from .behavioral_intelligence import (
    BehavioralIntelligenceSystem,
    UserBehaviorModeler,
    EngagementAnalytics,
    PersonalizationInsights,
    BehaviorState,
    EngagementLevel
)

# Import utilities
from .utils.statistical_methods import (
    StatisticalAnalyzer,
    BayesianInference,
    CausalInference,
    TimeSeriesAnalyzer,
    HypothesisTestingFramework
)
from .utils.ml_models import (
    EnsembleModelManager,
    TransformerModelWrapper,
    ReinforcementLearningAgent,
    AnomalyDetectionModels,
    FeatureEngineeringPipeline
)

# Define what gets imported with "from analytics import *"
__all__ = [
    # Main orchestration
    "AnalyticsOrchestrator",
    "AnalyticsSession",
    "AnalyticsInsight",
    "AnalyticsMode",
    "AnalyticsFocus",

    # Existing engines (backward compatibility)
    "LearningPatternAnalysisEngine",
    "PerformancePredictionEngine",
    "ResearchAnalyticsEngine",
    "CognitiveLoadAssessmentEngine",

    # New engines
    "AttentionOptimizationEngine",
    "FocusEnhancementAlgorithms",
    "BehavioralIntelligenceSystem",
    "UserBehaviorModeler",
    "EngagementAnalytics",
    "PersonalizationInsights",

    # Enums and states
    "AttentionState",
    "BehaviorState",
    "EngagementLevel",

    # Utilities
    "StatisticalAnalyzer",
    "BayesianInference",
    "CausalInference",
    "TimeSeriesAnalyzer",
    "HypothesisTestingFramework",
    "EnsembleModelManager",
    "TransformerModelWrapper",
    "ReinforcementLearningAgent",
    "AnomalyDetectionModels",
    "FeatureEngineeringPipeline",
]


# Convenience functions for quick setup
def create_analytics_orchestrator(config=None):
    """
    Create analytics orchestrator with default configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        AnalyticsOrchestrator: Configured analytics orchestrator
    """
    return AnalyticsOrchestrator(config=config)


def create_behavioral_intelligence_system(cache_service=None):
    """
    Create behavioral intelligence system with default configuration.

    Args:
        cache_service: Optional cache service for performance optimization

    Returns:
        BehavioralIntelligenceSystem: Configured behavioral intelligence system
    """
    return BehavioralIntelligenceSystem(cache_service)


def create_attention_optimization_engine(cache_service=None):
    """
    Create attention optimization engine with default configuration.

    Args:
        cache_service: Optional cache service for performance optimization

    Returns:
        AttentionOptimizationEngine: Configured attention optimization engine
    """
    return AttentionOptimizationEngine(cache_service)


async def quick_analytics_session(user_data, learning_activities, analytics_type='comprehensive'):
    """
    Create a quick analytics session with minimal setup.

    Args:
        user_data: User profile and behavioral data
        learning_activities: Learning activity data
        analytics_type: Type of analytics ('patterns', 'performance', 'comprehensive')

    Returns:
        Dict: Analytics session result
    """
    orchestrator = create_analytics_orchestrator()
    return await orchestrator.create_analytics_session(
        user_data, learning_activities, {'analytics_type': analytics_type}
    )


async def analyze_user_behavior_quick(user_id, behavioral_data, learning_activities):
    """
    Quick user behavior analysis with default settings.

    Args:
        user_id: User identifier
        behavioral_data: User behavioral data
        learning_activities: Learning activity history

    Returns:
        UserBehaviorProfile: Behavior analysis result
    """
    system = create_behavioral_intelligence_system()
    return await system.analyze_user_behavior(
        user_id, behavioral_data, learning_activities
    )


async def optimize_attention_quick(user_id, behavioral_data, environmental_data=None):
    """
    Quick attention optimization analysis.

    Args:
        user_id: User identifier
        behavioral_data: User behavioral data
        environmental_data: Optional environmental context

    Returns:
        AttentionAnalysis: Attention analysis result
    """
    engine = create_attention_optimization_engine()
    return await engine.analyze_attention_patterns(
        user_id, behavioral_data, None, environmental_data
    )


# Add convenience functions to exports
__all__.extend([
    "create_analytics_orchestrator",
    "create_behavioral_intelligence_system",
    "create_attention_optimization_engine",
    "quick_analytics_session",
    "analyze_user_behavior_quick",
    "optimize_attention_quick"
])


# Package metadata
__package_info__ = {
    'name': 'quantum_intelligence.services.analytics',
    'version': __version__,
    'description': 'Research-grade analytics services for learning platforms',
    'author': __author__,
    'components': [
        'learning_patterns',
        'cognitive_load',
        'attention_optimization',
        'performance_analytics',
        'behavioral_intelligence',
        'research_pipeline',
        'orchestrator'
    ],
    'methodologies': [
        'Deep Learning (Transformers, LSTM/GRU, Autoencoders)',
        'Statistical Methods (Bayesian inference, Causal inference, Time series)',
        'Machine Learning (Ensemble methods, Reinforcement learning, Transfer learning)',
        'Data Science (Feature engineering, Dimensionality reduction, Anomaly detection)',
        'Research Methods (Experimental design, A/B testing, Longitudinal analysis)'
    ]
}


# Logging configuration
import logging

# Create logger for the analytics package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Quantum Intelligence Analytics Services v{__version__} initialized")
logger.info(f"Available components: {', '.join(__package_info__['components'])}")


# Health check function
def health_check():
    """
    Perform a basic health check of the analytics package.

    Returns:
        Dict: Health check results
    """
    try:
        # Test basic imports and initialization
        orchestrator = create_analytics_orchestrator()
        behavior_system = create_behavioral_intelligence_system()
        attention_engine = create_attention_optimization_engine()

        return {
            'status': 'healthy',
            'version': __version__,
            'components_loaded': len(__package_info__['components']),
            'orchestrator_initialized': orchestrator is not None,
            'behavior_system_initialized': behavior_system is not None,
            'attention_engine_initialized': attention_engine is not None
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'version': __version__
        }


# Export health check
__all__.append('health_check')
