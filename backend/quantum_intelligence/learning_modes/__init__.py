"""
Learning Modes module for Quantum Intelligence Engine
"""

from .base_mode import BaseLearningMode
from .adaptive_quantum import AdaptiveQuantumMode
from .socratic_discovery import SocraticDiscoveryMode

__all__ = [
    "BaseLearningMode",
    "AdaptiveQuantumMode", 
    "SocraticDiscoveryMode",
]
