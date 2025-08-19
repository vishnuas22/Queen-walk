"""
Quantum Learning Data Structures

Core data structures for quantum-inspired learning algorithms including
quantum states, superposition states, entanglement patterns, and measurement results.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
import random
import cmath

# Try to import numpy
NUMPY_AVAILABLE = False
np = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Provide fallback functions
    class np:
        @staticmethod
        def random():
            class RandomModule:
                @staticmethod
                def uniform(low, high, size=None):
                    if size is None:
                        return random.uniform(low, high)
                    return [random.uniform(low, high) for _ in range(size)]
                @staticmethod
                def normal(mean, std, size=None):
                    if size is None:
                        return random.gauss(mean, std)
                    return [random.gauss(mean, std) for _ in range(size)]
                @staticmethod
                def rand(*args):
                    if len(args) == 0:
                        return random.random()
                    elif len(args) == 1:
                        return [random.random() for _ in range(args[0])]
                    else:
                        total = 1
                        for arg in args:
                            total *= arg
                        return [random.random() for _ in range(total)]
            return RandomModule()
        
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]
        
        @staticmethod
        def ones(shape):
            if isinstance(shape, int):
                return [1.0] * shape
            return [[1.0] * shape[1] for _ in range(shape[0])]
        
        @staticmethod
        def eye(n):
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                matrix[i][i] = 1.0
            return matrix
        
        @staticmethod
        def dot(a, b):
            if isinstance(a[0], list) and isinstance(b[0], list):
                # Matrix multiplication
                result = [[0.0] * len(b[0]) for _ in range(len(a))]
                for i in range(len(a)):
                    for j in range(len(b[0])):
                        for k in range(len(b)):
                            result[i][j] += a[i][k] * b[k][j]
                return result
            else:
                # Vector dot product
                return sum(a[i] * b[i] for i in range(len(a)))
        
        @staticmethod
        def linalg():
            class LinalgModule:
                @staticmethod
                def norm(vector):
                    return sum(x**2 for x in vector) ** 0.5
            return LinalgModule()
        
        @staticmethod
        def real(complex_array):
            if isinstance(complex_array, complex):
                return complex_array.real
            return [x.real if isinstance(x, complex) else x for x in complex_array]
        
        @staticmethod
        def imag(complex_array):
            if isinstance(complex_array, complex):
                return complex_array.imag
            return [x.imag if isinstance(x, complex) else 0 for x in complex_array]
        
        @staticmethod
        def abs(array):
            if isinstance(array, (int, float, complex)):
                return abs(array)
            return [abs(x) for x in array]
        
        @staticmethod
        def sum(array, axis=None):
            if axis is None:
                return sum(array)
            # Simplified for 2D arrays
            if axis == 0:
                return [sum(array[i][j] for i in range(len(array))) for j in range(len(array[0]))]
            elif axis == 1:
                return [sum(row) for row in array]
        
        @staticmethod
        def mean(array):
            return sum(array) / len(array)

        @staticmethod
        def exp(x):
            import math
            if isinstance(x, complex):
                return cmath.exp(x)
            elif isinstance(x, (int, float)):
                return math.exp(x)
            return [cmath.exp(val) if isinstance(val, complex) else math.exp(val) for val in x]

        @staticmethod
        def sqrt(x):
            import math
            if isinstance(x, (int, float)):
                return math.sqrt(x)
            return [math.sqrt(val) for val in x]

        @staticmethod
        def log2(x):
            import math
            if isinstance(x, (int, float)):
                return math.log2(x)
            return [math.log2(val) for val in x]

        # Mathematical constants
        pi = 3.141592653589793

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)


class QuantumLearningPhase(Enum):
    """Phases of quantum learning process"""
    SUPERPOSITION = "superposition"
    ENTANGLEMENT = "entanglement"
    INTERFERENCE = "interference"
    MEASUREMENT = "measurement"
    COLLAPSE = "collapse"


class MeasurementBasis(Enum):
    """Quantum measurement basis types"""
    COMPUTATIONAL = "computational"
    HADAMARD = "hadamard"
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    CUSTOM = "custom"


class EntanglementType(Enum):
    """Types of quantum entanglement in learning"""
    CONCEPT_CORRELATION = "concept_correlation"
    SKILL_DEPENDENCY = "skill_dependency"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    COLLABORATIVE_LEARNING = "collaborative_learning"
    TEMPORAL_COHERENCE = "temporal_coherence"


@dataclass
class QuantumState:
    """
    🌊 QUANTUM STATE REPRESENTATION
    
    Represents a quantum state with complex amplitudes and quantum properties.
    """
    amplitudes: Union[List[complex], Any]  # np.ndarray when numpy available
    timestamp: datetime = field(default_factory=datetime.now)
    coherence_time: float = 300.0  # seconds
    decoherence_rate: float = 0.01
    entanglement_partners: List[str] = field(default_factory=list)
    measurement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Normalize quantum state after initialization"""
        self.normalize()
    
    def normalize(self):
        """Normalize quantum state amplitudes"""
        if NUMPY_AVAILABLE:
            norm = np.linalg.norm(self.amplitudes)
            if norm > 0:
                self.amplitudes = self.amplitudes / norm
        else:
            # Fallback normalization
            norm = sum(abs(amp)**2 for amp in self.amplitudes) ** 0.5
            if norm > 0:
                self.amplitudes = [amp / norm for amp in self.amplitudes]
    
    def measure(self, basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL) -> int:
        """
        Measure quantum state and collapse to classical state
        
        Returns:
            int: Measured state index
        """
        # Calculate measurement probabilities
        probabilities = [abs(amp)**2 for amp in self.amplitudes]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        
        # Sample from probability distribution
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                # Record measurement
                self.measurement_history.append({
                    'basis': basis.value,
                    'result': i,
                    'probability': prob,
                    'timestamp': datetime.now()
                })
                return i
        
        # Fallback to last state
        return len(probabilities) - 1
    
    def get_probability_distribution(self) -> List[float]:
        """Get probability distribution from quantum amplitudes"""
        return [abs(amp)**2 for amp in self.amplitudes]
    
    def apply_decoherence(self, time_elapsed: float):
        """Apply quantum decoherence over time"""
        if NUMPY_AVAILABLE:
            decoherence_factor = np.exp(-self.decoherence_rate * time_elapsed)
            # Add random phase noise
            phase_noise = np.random.normal(0, 1 - decoherence_factor, len(self.amplitudes))
            self.amplitudes = self.amplitudes * decoherence_factor * np.exp(1j * phase_noise)
        else:
            # Simplified decoherence using math.exp
            import math
            decoherence_factor = math.exp(-self.decoherence_rate * time_elapsed)
            self.amplitudes = [amp * decoherence_factor for amp in self.amplitudes]

        self.normalize()


@dataclass
class SuperpositionState:
    """
    🌀 SUPERPOSITION LEARNING STATE
    
    Represents multiple simultaneous learning hypotheses in quantum superposition.
    """
    user_id: str
    hypotheses: List[Dict[str, Any]]
    quantum_state: QuantumState
    coherence_time: float
    creation_timestamp: datetime = field(default_factory=datetime.now)
    collapse_threshold: float = 0.8
    interference_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def get_dominant_hypothesis(self) -> Optional[Dict[str, Any]]:
        """Get the currently dominant learning hypothesis"""
        probabilities = self.quantum_state.get_probability_distribution()
        
        if not probabilities:
            return None
        
        max_prob_index = probabilities.index(max(probabilities))
        
        if max_prob_index < len(self.hypotheses):
            return self.hypotheses[max_prob_index]
        
        return None
    
    def should_collapse(self) -> bool:
        """Determine if superposition should collapse to single hypothesis"""
        probabilities = self.quantum_state.get_probability_distribution()
        
        if not probabilities:
            return True
        
        max_probability = max(probabilities)
        return max_probability >= self.collapse_threshold
    
    def calculate_entropy(self) -> float:
        """Calculate quantum entropy of superposition state"""
        probabilities = self.quantum_state.get_probability_distribution()
        
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                if NUMPY_AVAILABLE:
                    entropy -= prob * np.log2(prob)
                else:
                    # Fallback log2 approximation
                    import math
                    entropy -= prob * math.log2(prob)
        
        return entropy


@dataclass
class EntanglementPattern:
    """
    🔗 QUANTUM ENTANGLEMENT PATTERN
    
    Represents quantum entanglement between learning concepts or users.
    """
    entanglement_id: str
    entanglement_type: EntanglementType
    participants: List[str]  # User IDs or concept IDs
    entanglement_strength: float  # 0.0 to 1.0
    correlation_matrix: Union[List[List[float]], Any]  # np.ndarray when numpy available
    creation_timestamp: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.05
    measurement_correlations: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_correlation(self, participant_a: str, participant_b: str) -> float:
        """Calculate correlation strength between two participants"""
        try:
            idx_a = self.participants.index(participant_a)
            idx_b = self.participants.index(participant_b)
            
            if NUMPY_AVAILABLE:
                return float(self.correlation_matrix[idx_a][idx_b])
            else:
                return self.correlation_matrix[idx_a][idx_b]
        except (ValueError, IndexError):
            return 0.0
    
    def apply_decay(self, time_elapsed: float):
        """Apply entanglement decay over time"""
        decay_factor = np.exp(-self.decay_rate * time_elapsed) if NUMPY_AVAILABLE else (1 - self.decay_rate * time_elapsed)
        self.entanglement_strength *= max(0.0, decay_factor)
        
        # Decay correlation matrix
        if NUMPY_AVAILABLE:
            self.correlation_matrix = self.correlation_matrix * decay_factor
        else:
            for i in range(len(self.correlation_matrix)):
                for j in range(len(self.correlation_matrix[i])):
                    self.correlation_matrix[i][j] *= decay_factor


@dataclass
class InterferencePattern:
    """
    🌊 QUANTUM INTERFERENCE PATTERN
    
    Represents constructive and destructive interference in learning processes.
    """
    pattern_id: str
    user_id: str
    concept_pairs: List[Tuple[str, str]]
    interference_amplitudes: List[complex]
    constructive_regions: List[Dict[str, Any]]
    destructive_regions: List[Dict[str, Any]]
    pattern_strength: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_interference_effect(self, concept_a: str, concept_b: str) -> complex:
        """Calculate interference effect between two concepts"""
        for i, (ca, cb) in enumerate(self.concept_pairs):
            if (ca == concept_a and cb == concept_b) or (ca == concept_b and cb == concept_a):
                return self.interference_amplitudes[i]
        
        return complex(0, 0)
    
    def is_constructive(self, concept_a: str, concept_b: str) -> bool:
        """Check if interference between concepts is constructive"""
        effect = self.calculate_interference_effect(concept_a, concept_b)
        return effect.real > 0


@dataclass
class QuantumMeasurement:
    """
    📏 QUANTUM MEASUREMENT RESULT
    
    Represents the result of a quantum measurement in learning assessment.
    """
    measurement_id: str
    user_id: str
    measured_state: QuantumState
    measurement_basis: MeasurementBasis
    measurement_result: int
    probability: float
    confidence_interval: Tuple[float, float]
    measurement_timestamp: datetime = field(default_factory=datetime.now)
    pre_measurement_entropy: float = 0.0
    post_measurement_entropy: float = 0.0
    information_gain: float = 0.0
    
    def calculate_information_gain(self):
        """Calculate information gain from measurement"""
        self.information_gain = self.pre_measurement_entropy - self.post_measurement_entropy


@dataclass
class QuantumLearningPath:
    """
    🛤️ QUANTUM LEARNING PATH
    
    Represents an optimized learning path generated using quantum algorithms.
    """
    path_id: str
    user_id: str
    concepts: List[str]
    quantum_energy: float
    classical_energy: float
    quantum_advantage: float
    path_coherence: float
    learning_efficiency: float
    user_alignment: float
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    creation_timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall path quality score"""
        return (self.path_coherence + self.learning_efficiency + self.user_alignment) / 3.0


# Utility functions for quantum operations
def create_quantum_state(amplitudes: List[complex]) -> QuantumState:
    """Create a normalized quantum state"""
    return QuantumState(amplitudes=amplitudes)


def create_superposition_state(
    user_id: str,
    hypotheses: List[Dict[str, Any]],
    coherence_time: float = 300.0
) -> SuperpositionState:
    """Create a superposition state from learning hypotheses"""
    # Create equal superposition initially
    n_hypotheses = len(hypotheses)
    if n_hypotheses == 0:
        raise ValueError("Cannot create superposition state with no hypotheses")
    
    # Equal amplitude superposition
    amplitude = 1.0 / (n_hypotheses ** 0.5)
    amplitudes = [complex(amplitude, 0) for _ in range(n_hypotheses)]
    
    quantum_state = create_quantum_state(amplitudes)
    
    return SuperpositionState(
        user_id=user_id,
        hypotheses=hypotheses,
        quantum_state=quantum_state,
        coherence_time=coherence_time
    )


def create_entanglement_pattern(
    entanglement_type: EntanglementType,
    participants: List[str],
    entanglement_strength: float = 0.8
) -> EntanglementPattern:
    """Create an entanglement pattern between participants"""
    n_participants = len(participants)
    
    # Create correlation matrix
    if NUMPY_AVAILABLE:
        correlation_matrix = np.random.rand(n_participants, n_participants) * entanglement_strength
        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        # Set diagonal to 1 (perfect self-correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        # Fallback correlation matrix
        correlation_matrix = [[random.random() * entanglement_strength for _ in range(n_participants)] 
                            for _ in range(n_participants)]
        # Make symmetric and set diagonal
        for i in range(n_participants):
            correlation_matrix[i][i] = 1.0
            for j in range(i):
                correlation_matrix[i][j] = correlation_matrix[j][i]
    
    return EntanglementPattern(
        entanglement_id=f"ent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        entanglement_type=entanglement_type,
        participants=participants,
        entanglement_strength=entanglement_strength,
        correlation_matrix=correlation_matrix
    )


def measure_quantum_state(
    quantum_state: QuantumState,
    basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL
) -> QuantumMeasurement:
    """Perform quantum measurement and return result"""
    # Calculate pre-measurement entropy
    probabilities = quantum_state.get_probability_distribution()
    pre_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities) if NUMPY_AVAILABLE else 0
    
    # Perform measurement
    result = quantum_state.measure(basis)
    measured_probability = probabilities[result] if result < len(probabilities) else 0
    
    # Calculate post-measurement entropy (should be 0 for pure state)
    post_entropy = 0.0
    
    # Create measurement result
    measurement = QuantumMeasurement(
        measurement_id=f"meas_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        user_id="",  # To be set by caller
        measured_state=quantum_state,
        measurement_basis=basis,
        measurement_result=result,
        probability=measured_probability,
        confidence_interval=(max(0, measured_probability - 0.1), min(1, measured_probability + 0.1)),
        pre_measurement_entropy=pre_entropy,
        post_measurement_entropy=post_entropy
    )
    
    measurement.calculate_information_gain()
    return measurement
