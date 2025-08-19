"""
Analytics Utilities Package

Shared utilities and helper functions for the analytics services including
statistical methods, machine learning models, data processing, and visualization tools.
"""

from .statistical_methods import (
    StatisticalAnalyzer,
    BayesianInference,
    CausalInference,
    TimeSeriesAnalyzer,
    HypothesisTestingFramework
)

from .ml_models import (
    EnsembleModelManager,
    TransformerModelWrapper,
    ReinforcementLearningAgent,
    AnomalyDetectionModels,
    FeatureEngineeringPipeline
)

from .data_processing import (
    DataPreprocessor,
    FeatureExtractor,
    DimensionalityReducer,
    DataQualityAssessment,
    BiasDetectionFramework
)

from .visualization import (
    AnalyticsVisualizer,
    InteractiveDashboard,
    StatisticalPlotter,
    ResearchReportGenerator
)

__all__ = [
    # Statistical methods
    "StatisticalAnalyzer",
    "BayesianInference", 
    "CausalInference",
    "TimeSeriesAnalyzer",
    "HypothesisTestingFramework",
    
    # ML models
    "EnsembleModelManager",
    "TransformerModelWrapper",
    "ReinforcementLearningAgent",
    "AnomalyDetectionModels",
    "FeatureEngineeringPipeline",
    
    # Data processing
    "DataPreprocessor",
    "FeatureExtractor",
    "DimensionalityReducer", 
    "DataQualityAssessment",
    "BiasDetectionFramework",
    
    # Visualization
    "AnalyticsVisualizer",
    "InteractiveDashboard",
    "StatisticalPlotter",
    "ResearchReportGenerator",
]
