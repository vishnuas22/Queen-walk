"""
🚀 QUANTUM LEARNING INTELLIGENCE ENGINE 🚀
====================================================

Revolutionary unified AI system for MasterX - combining the best of all learning methodologies
into a single, powerful, quantum-caliber intelligence engine.

Version: 2.0.0 (Modularized Architecture)
"""

__version__ = "2.0.0"
__author__ = "MasterX Team"

# Core exports - these should always work
from .core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel
from .core.data_structures import QuantumLearningContext, QuantumResponse
from .core.exceptions import QuantumEngineError, ModelLoadError, ValidationError

# Try to import components that might have dependencies
try:
    from .config.settings import QuantumEngineConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    class QuantumEngineConfig:
        def __init__(self, **kwargs):
            # Provide basic configuration without pydantic
            self.app_name = "Quantum Learning Intelligence Engine"
            self.version = "2.0.0"
            self.debug = False
            self.environment = "development"

try:
    from .core.engine import QuantumLearningIntelligenceEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    class QuantumLearningIntelligenceEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Engine dependencies not available")

try:
    from .config.dependencies import get_quantum_engine, setup_dependencies
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    def get_quantum_engine():
        raise ImportError("Dependency injection not available")
    def setup_dependencies():
        raise ImportError("Dependency injection not available")

# Service exports - these will be implemented later
try:
    from .services.personalization.engine import PersonalizationEngine
    PERSONALIZATION_AVAILABLE = True
except ImportError:
    PERSONALIZATION_AVAILABLE = False
    class PersonalizationEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Personalization service not available")

# Placeholder for other services
class LearningPatternAnalysisEngine:
    def __init__(self, *args, **kwargs):
        raise ImportError("Learning pattern analysis service not implemented yet")

class MultiModalFusionEngine:
    def __init__(self, *args, **kwargs):
        raise ImportError("Multimodal fusion service not implemented yet")

class EmotionalAIWellbeingEngine:
    def __init__(self, *args, **kwargs):
        raise ImportError("Emotional AI service not implemented yet")

__all__ = [
    # Core
    "QuantumLearningIntelligenceEngine",
    "QuantumLearningMode",
    "QuantumState", 
    "IntelligenceLevel",
    "QuantumLearningContext",
    "QuantumResponse",
    
    # Exceptions
    "QuantumEngineError",
    "ModelLoadError", 
    "ValidationError",
    
    # Services
    "PersonalizationEngine",
    "LearningPatternAnalysisEngine",
    "MultiModalFusionEngine",
    "EmotionalAIWellbeingEngine",
    
    # Configuration
    "QuantumEngineConfig",
    "get_quantum_engine",
    "setup_dependencies",
]
