"""
Exception hierarchy for Quantum Intelligence Engine
"""

from typing import Optional, Dict, Any


class QuantumEngineError(Exception):
    """Base exception for all Quantum Engine errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ConfigurationError(QuantumEngineError):
    """Raised when there are configuration issues"""
    pass


class ModelLoadError(QuantumEngineError):
    """Raised when model loading fails"""
    pass


class AIProviderError(QuantumEngineError):
    """Raised when AI provider operations fail"""
    pass


class ValidationError(QuantumEngineError):
    """Raised when input validation fails"""
    pass


class DatabaseError(QuantumEngineError):
    """Raised when database operations fail"""
    pass


class CacheError(QuantumEngineError):
    """Raised when cache operations fail"""
    pass


class NeuralNetworkError(QuantumEngineError):
    """Raised when neural network operations fail"""
    pass


class PersonalizationError(QuantumEngineError):
    """Raised when personalization operations fail"""
    pass


class AnalyticsError(QuantumEngineError):
    """Raised when analytics operations fail"""
    pass


class MultiModalError(QuantumEngineError):
    """Raised when multimodal processing fails"""
    pass


class EmotionalAIError(QuantumEngineError):
    """Raised when emotional AI operations fail"""
    pass


class CollaborationError(QuantumEngineError):
    """Raised when collaboration features fail"""
    pass


class GamificationError(QuantumEngineError):
    """Raised when gamification features fail"""
    pass


class StreamingError(QuantumEngineError):
    """Raised when streaming operations fail"""
    pass


class QuantumAlgorithmError(QuantumEngineError):
    """Raised when quantum algorithm operations fail"""
    pass


class EnterpriseError(QuantumEngineError):
    """Raised when enterprise features fail"""
    pass


class SecurityError(QuantumEngineError):
    """Raised when security operations fail"""
    pass


class RateLimitError(QuantumEngineError):
    """Raised when rate limits are exceeded"""
    pass


class TimeoutError(QuantumEngineError):
    """Raised when operations timeout"""
    pass


class ResourceExhaustionError(QuantumEngineError):
    """Raised when system resources are exhausted"""
    pass


class IntelligenceAmplificationError(QuantumEngineError):
    """Raised when intelligence amplification operations fail"""
    pass


# Error code constants
class ErrorCodes:
    # Configuration errors
    MISSING_API_KEY = "CONFIG_001"
    INVALID_CONFIG = "CONFIG_002"
    MISSING_DATABASE = "CONFIG_003"
    
    # Model errors
    MODEL_LOAD_FAILED = "MODEL_001"
    MODEL_INFERENCE_FAILED = "MODEL_002"
    MODEL_TIMEOUT = "MODEL_003"
    
    # AI Provider errors
    API_KEY_INVALID = "AI_001"
    API_QUOTA_EXCEEDED = "AI_002"
    API_UNAVAILABLE = "AI_003"
    
    # Validation errors
    INVALID_INPUT = "VALID_001"
    MISSING_REQUIRED_FIELD = "VALID_002"
    INVALID_FORMAT = "VALID_003"
    
    # Database errors
    CONNECTION_FAILED = "DB_001"
    QUERY_FAILED = "DB_002"
    TRANSACTION_FAILED = "DB_003"
    
    # Cache errors
    CACHE_UNAVAILABLE = "CACHE_001"
    CACHE_WRITE_FAILED = "CACHE_002"
    CACHE_READ_FAILED = "CACHE_003"
    
    # Neural network errors
    NETWORK_INIT_FAILED = "NN_001"
    FORWARD_PASS_FAILED = "NN_002"
    TRAINING_FAILED = "NN_003"
    
    # Security errors
    UNAUTHORIZED = "SEC_001"
    FORBIDDEN = "SEC_002"
    RATE_LIMITED = "SEC_003"
    
    # Resource errors
    MEMORY_EXHAUSTED = "RES_001"
    CPU_EXHAUSTED = "RES_002"
    TIMEOUT = "RES_003"


def create_error(
    error_type: str,
    message: str,
    error_code: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> QuantumEngineError:
    """Factory function to create appropriate error types"""
    
    error_classes = {
        "configuration": ConfigurationError,
        "model": ModelLoadError,
        "ai_provider": AIProviderError,
        "validation": ValidationError,
        "database": DatabaseError,
        "cache": CacheError,
        "neural_network": NeuralNetworkError,
        "personalization": PersonalizationError,
        "analytics": AnalyticsError,
        "multimodal": MultiModalError,
        "emotional_ai": EmotionalAIError,
        "collaboration": CollaborationError,
        "gamification": GamificationError,
        "streaming": StreamingError,
        "quantum_algorithm": QuantumAlgorithmError,
        "enterprise": EnterpriseError,
        "security": SecurityError,
        "rate_limit": RateLimitError,
        "timeout": TimeoutError,
        "resource": ResourceExhaustionError,
        "intelligence": IntelligenceAmplificationError,
    }
    
    error_class = error_classes.get(error_type, QuantumEngineError)
    return error_class(message, error_code, context)
