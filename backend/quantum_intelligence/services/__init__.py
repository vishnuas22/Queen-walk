"""
Services module for Quantum Intelligence Engine

This module contains all the extracted service components from the original
monolithic quantum_intelligence_engine.py file.
"""

# Personalization services
try:
    from .personalization.engine import PersonalizationEngine
    from .personalization.learning_dna import LearningDNAManager
    from .personalization.adaptive_parameters import AdaptiveParametersEngine
    from .personalization.mood_adaptation import MoodAdaptationEngine
    PERSONALIZATION_AVAILABLE = True
except ImportError:
    PERSONALIZATION_AVAILABLE = False

    class PersonalizationEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Personalization services not yet implemented")

    class LearningDNAManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("Learning DNA services not yet implemented")

    class AdaptiveParametersEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Adaptive parameters services not yet implemented")

    class MoodAdaptationEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Mood adaptation services not yet implemented")

# Analytics services
try:
    from .analytics.learning_patterns import LearningPatternAnalysisEngine
    from .analytics.performance_prediction import PerformancePredictionEngine
    from .analytics.research_analytics import ResearchAnalyticsEngine
    from .analytics.cognitive_load import CognitiveLoadAssessmentEngine
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

    class LearningPatternAnalysisEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Learning pattern analysis not yet implemented")

    class PerformancePredictionEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Performance prediction not yet implemented")

    class ResearchAnalyticsEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Research analytics not yet implemented")

    class CognitiveLoadAssessmentEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Cognitive load assessment not yet implemented")

__all__ = [
    # Personalization
    "PersonalizationEngine",
    "LearningDNAManager",
    "AdaptiveParametersEngine",
    "MoodAdaptationEngine",

    # Analytics
    "LearningPatternAnalysisEngine",
    "PerformancePredictionEngine",
    "ResearchAnalyticsEngine",
    "CognitiveLoadAssessmentEngine",
]
