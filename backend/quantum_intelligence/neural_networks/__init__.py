"""
Neural Networks module for Quantum Intelligence Engine
"""

# Try to import neural networks, handle missing dependencies gracefully
try:
    from .quantum_processor import QuantumResponseProcessor
    from .difficulty_network import AdaptiveDifficultyNetwork
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    NEURAL_NETWORKS_AVAILABLE = False

    class QuantumResponseProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Neural networks not available: {e}")

    class AdaptiveDifficultyNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Neural networks not available: {e}")

__all__ = [
    "QuantumResponseProcessor",
    "AdaptiveDifficultyNetwork",
]
