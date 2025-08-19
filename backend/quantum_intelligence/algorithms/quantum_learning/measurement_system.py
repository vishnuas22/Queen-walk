"""
Quantum Measurement System for Learning Assessment

Advanced system for performing quantum measurements on learning states,
managing state collapse, and extracting classical information from quantum
superposition states for learning assessment and optimization.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import random
import cmath

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Use fallback from quantum_data_structures
    from .quantum_data_structures import np

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .quantum_data_structures import (
    QuantumState, QuantumMeasurement, MeasurementBasis, SuperpositionState,
    EntanglementPattern, InterferencePattern, measure_quantum_state
)


class QuantumMeasurementSystem:
    """
    🔬 QUANTUM MEASUREMENT SYSTEM FOR LEARNING ASSESSMENT
    
    Advanced system for performing quantum measurements on learning states,
    managing state collapse, and extracting classical information from quantum
    superposition states for learning assessment and optimization.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Measurement infrastructure
        self.measurement_apparatus: Dict[str, Dict[str, Any]] = {}
        self.measurement_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collapsed_states: Dict[str, Dict[str, Any]] = {}
        self.measurement_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Measurement basis configurations
        self.measurement_bases = {
            MeasurementBasis.COMPUTATIONAL: self._create_computational_basis,
            MeasurementBasis.HADAMARD: self._create_hadamard_basis,
            MeasurementBasis.PAULI_X: self._create_pauli_x_basis,
            MeasurementBasis.PAULI_Y: self._create_pauli_y_basis,
            MeasurementBasis.PAULI_Z: self._create_pauli_z_basis,
            MeasurementBasis.CUSTOM: self._create_custom_basis
        }
        
        # Configuration parameters
        self.measurement_precision = 0.95
        self.decoherence_threshold = 0.1
        self.collapse_fidelity = 0.98
        self.measurement_back_action = 0.05
        self.information_extraction_efficiency = 0.9
        
        # Performance tracking
        self.measurement_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.quantum_information_loss: List[float] = []
        self.measurement_efficiency_history: deque = deque(maxlen=100)
        
        logger.info("Quantum Measurement System initialized")
    
    async def perform_learning_state_measurement(
        self,
        user_id: str,
        quantum_state: QuantumState,
        measurement_basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL,
        measurement_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform quantum measurement on learning state
        
        Args:
            user_id: User identifier
            quantum_state: Quantum state to measure
            measurement_basis: Measurement basis to use
            measurement_context: Context for measurement
            
        Returns:
            Dict: Measurement result with collapsed state
        """
        try:
            # Prepare measurement apparatus
            apparatus = await self._prepare_measurement_apparatus(
                measurement_basis, measurement_context
            )
            
            # Apply pre-measurement decoherence
            await self._apply_pre_measurement_decoherence(quantum_state)
            
            # Perform quantum measurement
            measurement_result = await self._execute_quantum_measurement(
                quantum_state, apparatus, measurement_context
            )
            
            # Collapse quantum state
            collapsed_state = await self._collapse_quantum_state(
                quantum_state, measurement_result
            )
            
            # Extract classical information
            classical_info = await self._extract_classical_information(
                measurement_result, collapsed_state, measurement_context
            )
            
            # Calculate measurement metrics
            measurement_metrics = await self._calculate_measurement_metrics(
                quantum_state, measurement_result, collapsed_state
            )
            
            # Record measurement
            measurement_record = await self._record_measurement(
                user_id, quantum_state, measurement_result, collapsed_state,
                measurement_basis, measurement_context, measurement_metrics
            )
            
            # Update measurement statistics
            await self._update_measurement_statistics(
                user_id, measurement_basis, measurement_metrics
            )
            
            return {
                'status': 'success',
                'measurement_result': measurement_result,
                'collapsed_state': collapsed_state,
                'classical_information': classical_info,
                'measurement_metrics': measurement_metrics,
                'measurement_record': measurement_record,
                'quantum_information_loss': measurement_metrics.get('information_loss', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error performing learning state measurement: {e}")
            raise QuantumEngineError(f"Failed to perform measurement: {e}")
    
    async def measure_superposition_collapse(
        self,
        user_id: str,
        superposition_state: SuperpositionState,
        collapse_trigger: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Measure and collapse superposition state to single hypothesis
        
        Args:
            user_id: User identifier
            superposition_state: Superposition state to collapse
            collapse_trigger: Trigger conditions for collapse
            
        Returns:
            Dict: Collapse measurement result
        """
        try:
            # Check if collapse is warranted
            collapse_readiness = await self._assess_collapse_readiness(
                superposition_state, collapse_trigger
            )
            
            if not collapse_readiness['should_collapse']:
                return {
                    'status': 'no_collapse',
                    'reason': collapse_readiness['reason'],
                    'readiness_score': collapse_readiness['readiness_score']
                }
            
            # Perform superposition measurement
            measurement_result = measure_quantum_state(
                superposition_state.quantum_state,
                MeasurementBasis.COMPUTATIONAL
            )
            measurement_result.user_id = user_id
            
            # Select collapsed hypothesis
            collapsed_hypothesis = None
            if measurement_result.measurement_result < len(superposition_state.hypotheses):
                collapsed_hypothesis = superposition_state.hypotheses[measurement_result.measurement_result]
            
            # Calculate collapse metrics
            collapse_metrics = await self._calculate_collapse_metrics(
                superposition_state, measurement_result, collapsed_hypothesis
            )
            
            # Update superposition state
            await self._update_superposition_after_collapse(
                superposition_state, measurement_result
            )
            
            # Store collapsed state
            self.collapsed_states[user_id] = {
                'collapsed_hypothesis': collapsed_hypothesis,
                'measurement_result': measurement_result,
                'collapse_metrics': collapse_metrics,
                'collapse_timestamp': datetime.now(),
                'original_entropy': superposition_state.calculate_entropy()
            }
            
            return {
                'status': 'collapsed',
                'collapsed_hypothesis': collapsed_hypothesis,
                'measurement_result': measurement_result,
                'collapse_metrics': collapse_metrics,
                'information_gain': measurement_result.information_gain,
                'collapse_certainty': collapse_metrics.get('collapse_certainty', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error measuring superposition collapse: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def measure_entanglement_correlation(
        self,
        entanglement_pattern: EntanglementPattern,
        measurement_basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL
    ) -> Dict[str, Any]:
        """
        Measure correlation in entangled learning system
        
        Args:
            entanglement_pattern: Entanglement pattern to measure
            measurement_basis: Measurement basis
            
        Returns:
            Dict: Correlation measurement result
        """
        try:
            # Prepare correlation measurement
            correlation_apparatus = await self._prepare_correlation_measurement(
                entanglement_pattern, measurement_basis
            )
            
            # Perform correlation measurement
            correlation_result = await self._measure_quantum_correlation(
                entanglement_pattern, correlation_apparatus
            )
            
            # Check for Bell inequality violations
            bell_violation = await self._check_bell_inequality_violation(
                entanglement_pattern, correlation_result
            )
            
            # Calculate non-local correlation strength
            non_local_strength = await self._calculate_non_local_correlation(
                entanglement_pattern, correlation_result
            )
            
            # Update entanglement after measurement
            await self._update_entanglement_after_measurement(
                entanglement_pattern, correlation_result
            )
            
            return {
                'status': 'success',
                'correlation_result': correlation_result,
                'bell_violation': bell_violation,
                'non_local_strength': non_local_strength,
                'entanglement_preserved': correlation_result.get('entanglement_preserved', True),
                'measurement_disturbance': correlation_result.get('measurement_disturbance', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error measuring entanglement correlation: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def measure_interference_pattern(
        self,
        interference_pattern: InterferencePattern,
        measurement_points: List[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Measure quantum interference pattern in learning
        
        Args:
            interference_pattern: Interference pattern to measure
            measurement_points: Specific points to measure
            
        Returns:
            Dict: Interference measurement result
        """
        try:
            # Determine measurement points
            if measurement_points is None:
                measurement_points = interference_pattern.concept_pairs
            
            # Measure interference at each point
            interference_measurements = []
            for concept_a, concept_b in measurement_points:
                measurement = await self._measure_interference_at_point(
                    interference_pattern, concept_a, concept_b
                )
                interference_measurements.append(measurement)
            
            # Analyze interference pattern
            pattern_analysis = await self._analyze_interference_measurements(
                interference_measurements, interference_pattern
            )
            
            # Calculate pattern coherence
            pattern_coherence = await self._calculate_pattern_coherence(
                interference_measurements
            )
            
            # Identify constructive/destructive regions
            region_analysis = await self._analyze_interference_regions(
                interference_measurements, interference_pattern
            )
            
            return {
                'status': 'success',
                'interference_measurements': interference_measurements,
                'pattern_analysis': pattern_analysis,
                'pattern_coherence': pattern_coherence,
                'region_analysis': region_analysis,
                'measurement_points': measurement_points
            }
            
        except Exception as e:
            logger.error(f"Error measuring interference pattern: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def extract_learning_insights(
        self,
        user_id: str,
        measurement_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract learning insights from quantum measurements
        
        Args:
            user_id: User identifier
            measurement_history: Optional measurement history
            
        Returns:
            Dict: Extracted learning insights
        """
        try:
            # Get measurement history
            if measurement_history is None:
                measurement_history = list(self.measurement_history[user_id])
            
            if not measurement_history:
                return {
                    'status': 'no_data',
                    'message': 'No measurement history available'
                }
            
            # Analyze measurement patterns
            measurement_patterns = await self._analyze_measurement_patterns(
                measurement_history
            )
            
            # Extract learning trends
            learning_trends = await self._extract_learning_trends(
                measurement_history, measurement_patterns
            )
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_optimization_opportunities(
                measurement_history, learning_trends
            )
            
            # Generate learning recommendations
            learning_recommendations = await self._generate_learning_recommendations(
                measurement_patterns, learning_trends, optimization_opportunities
            )
            
            # Calculate insight confidence
            insight_confidence = await self._calculate_insight_confidence(
                measurement_history, measurement_patterns
            )
            
            return {
                'status': 'success',
                'measurement_patterns': measurement_patterns,
                'learning_trends': learning_trends,
                'optimization_opportunities': optimization_opportunities,
                'learning_recommendations': learning_recommendations,
                'insight_confidence': insight_confidence,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting learning insights: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # Measurement apparatus preparation methods
    async def _prepare_measurement_apparatus(
        self,
        measurement_basis: MeasurementBasis,
        measurement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare quantum measurement apparatus"""
        
        # Create measurement basis
        basis_creator = self.measurement_bases.get(measurement_basis)
        if not basis_creator:
            raise ValueError(f"Unsupported measurement basis: {measurement_basis}")
        
        measurement_basis_matrix = await basis_creator(measurement_context)
        
        # Configure measurement parameters
        apparatus = {
            'measurement_basis': measurement_basis,
            'basis_matrix': measurement_basis_matrix,
            'precision': self.measurement_precision,
            'decoherence_threshold': self.decoherence_threshold,
            'back_action_strength': self.measurement_back_action,
            'context': measurement_context or {}
        }
        
        return apparatus
    
    async def _execute_quantum_measurement(
        self,
        quantum_state: QuantumState,
        apparatus: Dict[str, Any],
        measurement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute quantum measurement on state"""
        
        # Apply measurement basis transformation
        transformed_state = await self._apply_basis_transformation(
            quantum_state, apparatus['basis_matrix']
        )
        
        # Calculate measurement probabilities
        probabilities = [abs(amp)**2 for amp in transformed_state.amplitudes]
        
        # Apply measurement context bias
        if measurement_context:
            probabilities = await self._apply_measurement_bias(
                probabilities, measurement_context
            )
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        
        # Perform probabilistic measurement
        measurement_outcome = await self._perform_probabilistic_measurement(
            probabilities
        )
        
        # Calculate measurement uncertainty
        measurement_uncertainty = await self._calculate_measurement_uncertainty(
            probabilities
        )
        
        return {
            'measurement_outcome': measurement_outcome,
            'measurement_probability': probabilities[measurement_outcome] if measurement_outcome < len(probabilities) else 0,
            'all_probabilities': probabilities,
            'measurement_uncertainty': measurement_uncertainty,
            'apparatus_used': apparatus['measurement_basis'].value,
            'measurement_timestamp': datetime.now()
        }
    
    async def _collapse_quantum_state(
        self,
        quantum_state: QuantumState,
        measurement_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collapse quantum state based on measurement"""
        
        measurement_outcome = measurement_result['measurement_outcome']
        
        # Create collapsed state amplitudes
        collapsed_amplitudes = [complex(0, 0)] * len(quantum_state.amplitudes)
        if measurement_outcome < len(collapsed_amplitudes):
            collapsed_amplitudes[measurement_outcome] = complex(1, 0)
        
        # Create collapsed quantum state
        collapsed_quantum_state = QuantumState(
            amplitudes=collapsed_amplitudes,
            coherence_time=quantum_state.coherence_time,
            decoherence_rate=quantum_state.decoherence_rate
        )
        
        # Calculate information loss
        original_entropy = quantum_state.superposition_entropy() if hasattr(quantum_state, 'superposition_entropy') else 0
        collapsed_entropy = 0.0  # Pure state has zero entropy
        information_loss = original_entropy - collapsed_entropy
        
        return {
            'collapsed_quantum_state': collapsed_quantum_state,
            'measurement_outcome': measurement_outcome,
            'collapse_probability': measurement_result['measurement_probability'],
            'information_loss': information_loss,
            'collapse_fidelity': self.collapse_fidelity,
            'collapse_timestamp': datetime.now()
        }
    
    # Measurement basis creation methods
    async def _create_computational_basis(self, context: Dict[str, Any]) -> Any:
        """Create computational measurement basis"""
        if NUMPY_AVAILABLE:
            return np.eye(2)  # Standard computational basis
        else:
            return [[1, 0], [0, 1]]  # Identity matrix
    
    async def _create_hadamard_basis(self, context: Dict[str, Any]) -> Any:
        """Create Hadamard measurement basis"""
        if NUMPY_AVAILABLE:
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        else:
            sqrt2_inv = 1 / (2 ** 0.5)
            return [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]]
    
    async def _create_pauli_x_basis(self, context: Dict[str, Any]) -> Any:
        """Create Pauli-X measurement basis"""
        if NUMPY_AVAILABLE:
            return np.array([[0, 1], [1, 0]])
        else:
            return [[0, 1], [1, 0]]
    
    async def _create_pauli_y_basis(self, context: Dict[str, Any]) -> Any:
        """Create Pauli-Y measurement basis"""
        if NUMPY_AVAILABLE:
            return np.array([[0, -1j], [1j, 0]])
        else:
            return [[complex(0, 0), complex(0, -1)], [complex(0, 1), complex(0, 0)]]
    
    async def _create_pauli_z_basis(self, context: Dict[str, Any]) -> Any:
        """Create Pauli-Z measurement basis"""
        if NUMPY_AVAILABLE:
            return np.array([[1, 0], [0, -1]])
        else:
            return [[1, 0], [0, -1]]
    
    async def _create_custom_basis(self, context: Dict[str, Any]) -> Any:
        """Create custom measurement basis from context"""
        if 'custom_basis' in context:
            return context['custom_basis']
        else:
            # Default to computational basis
            return await self._create_computational_basis(context)
    
    # Helper calculation methods
    async def _apply_basis_transformation(self, quantum_state: QuantumState, basis_matrix: Any) -> QuantumState:
        """Apply measurement basis transformation to quantum state"""
        # For simplicity, return the original state
        # In a full implementation, this would apply the basis transformation
        return quantum_state
    
    async def _apply_measurement_bias(
        self, probabilities: List[float], measurement_context: Dict[str, Any]
    ) -> List[float]:
        """Apply measurement context bias to probabilities"""
        
        # Apply performance-based bias
        if 'current_performance' in measurement_context:
            performance = measurement_context['current_performance']
            # Bias towards outcomes matching current performance level
            for i in range(len(probabilities)):
                performance_match = 1.0 - abs(i / len(probabilities) - performance)
                probabilities[i] *= (1.0 + performance_match * 0.2)
        
        return probabilities
    
    async def _perform_probabilistic_measurement(self, probabilities: List[float]) -> int:
        """Perform probabilistic measurement based on probabilities"""
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return i
        
        # Fallback to last outcome
        return len(probabilities) - 1
    
    async def _calculate_measurement_uncertainty(self, probabilities: List[float]) -> float:
        """Calculate measurement uncertainty (entropy)"""
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                if NUMPY_AVAILABLE:
                    entropy -= prob * np.log2(prob)
                else:
                    import math
                    entropy -= prob * math.log2(prob)
        return entropy
    
    async def _record_measurement(
        self,
        user_id: str,
        quantum_state: QuantumState,
        measurement_result: Dict[str, Any],
        collapsed_state: Dict[str, Any],
        measurement_basis: MeasurementBasis,
        measurement_context: Dict[str, Any],
        measurement_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record measurement in history"""
        
        measurement_record = {
            'user_id': user_id,
            'measurement_timestamp': datetime.now(),
            'measurement_basis': measurement_basis.value,
            'measurement_outcome': measurement_result['measurement_outcome'],
            'measurement_probability': measurement_result['measurement_probability'],
            'measurement_uncertainty': measurement_result['measurement_uncertainty'],
            'information_loss': collapsed_state['information_loss'],
            'collapse_fidelity': collapsed_state['collapse_fidelity'],
            'measurement_context': measurement_context,
            'measurement_metrics': measurement_metrics
        }
        
        # Store in history
        self.measurement_history[user_id].append(measurement_record)
        
        return measurement_record
    
    async def _calculate_measurement_metrics(
        self,
        quantum_state: QuantumState,
        measurement_result: Dict[str, Any],
        collapsed_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive measurement metrics"""
        
        return {
            'measurement_efficiency': self.information_extraction_efficiency,
            'information_loss': collapsed_state['information_loss'],
            'measurement_precision': self.measurement_precision,
            'collapse_fidelity': collapsed_state['collapse_fidelity'],
            'measurement_uncertainty': measurement_result['measurement_uncertainty'],
            'quantum_advantage': await self._calculate_quantum_advantage(quantum_state, measurement_result),
            'measurement_quality': await self._calculate_measurement_quality(measurement_result)
        }
    
    async def _calculate_quantum_advantage(
        self, quantum_state: QuantumState, measurement_result: Dict[str, Any]
    ) -> float:
        """Calculate quantum advantage of measurement"""
        # Simplified quantum advantage calculation
        uncertainty = measurement_result['measurement_uncertainty']
        max_uncertainty = len(quantum_state.amplitudes).bit_length() - 1 if len(quantum_state.amplitudes) > 1 else 1
        return uncertainty / max_uncertainty if max_uncertainty > 0 else 0.0
    
    async def _calculate_measurement_quality(self, measurement_result: Dict[str, Any]) -> float:
        """Calculate overall measurement quality"""
        uncertainty = measurement_result['measurement_uncertainty']
        probability = measurement_result['measurement_probability']
        
        # Higher probability and lower uncertainty indicate better quality
        quality = probability * (1.0 - uncertainty / 10.0)  # Normalize uncertainty
        return min(1.0, max(0.0, quality))

    # Additional helper methods
    async def _apply_pre_measurement_decoherence(self, quantum_state: QuantumState):
        """Apply pre-measurement decoherence effects"""
        # Apply small amount of decoherence before measurement
        quantum_state.apply_decoherence(0.1)

    async def _extract_classical_information(
        self, measurement_result: Dict[str, Any], collapsed_state: Dict[str, Any],
        measurement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract classical information from quantum measurement"""
        return {
            'measurement_outcome': measurement_result['measurement_outcome'],
            'confidence': measurement_result['measurement_probability'],
            'information_content': 1.0 - measurement_result['measurement_uncertainty'],
            'classical_state': collapsed_state['measurement_outcome'],
            'extraction_efficiency': self.information_extraction_efficiency
        }

    async def _assess_collapse_readiness(
        self, superposition_state: SuperpositionState, collapse_trigger: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess if superposition state is ready for collapse"""

        # Check natural collapse conditions
        should_collapse = superposition_state.should_collapse()

        # Check trigger conditions
        if collapse_trigger:
            trigger_threshold = collapse_trigger.get('threshold', 0.8)
            max_prob = max(superposition_state.quantum_state.get_probability_distribution())
            trigger_collapse = max_prob >= trigger_threshold
            should_collapse = should_collapse or trigger_collapse

        # Calculate readiness score
        probabilities = superposition_state.quantum_state.get_probability_distribution()
        max_probability = max(probabilities) if probabilities else 0
        readiness_score = max_probability

        reason = 'ready_for_collapse' if should_collapse else 'insufficient_probability'

        return {
            'should_collapse': should_collapse,
            'readiness_score': readiness_score,
            'reason': reason
        }

    async def _calculate_collapse_metrics(
        self, superposition_state: SuperpositionState, measurement_result: QuantumMeasurement,
        collapsed_hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for superposition collapse"""

        return {
            'collapse_certainty': measurement_result.probability,
            'information_gain': measurement_result.information_gain,
            'entropy_reduction': superposition_state.calculate_entropy(),
            'hypothesis_confidence': collapsed_hypothesis.get('confidence', 0) if collapsed_hypothesis else 0,
            'collapse_quality': measurement_result.probability * 0.8 + (1.0 - measurement_result.measurement_uncertainty) * 0.2
        }

    async def _update_superposition_after_collapse(
        self, superposition_state: SuperpositionState, measurement_result: QuantumMeasurement
    ):
        """Update superposition state after collapse"""
        # Create collapsed amplitudes
        new_amplitudes = [complex(0, 0)] * len(superposition_state.quantum_state.amplitudes)
        if measurement_result.measurement_result < len(new_amplitudes):
            new_amplitudes[measurement_result.measurement_result] = complex(1, 0)

        # Update quantum state
        superposition_state.quantum_state.amplitudes = new_amplitudes
        superposition_state.quantum_state.normalize()

    async def _update_measurement_statistics(
        self, user_id: str, measurement_basis: MeasurementBasis, measurement_metrics: Dict[str, Any]
    ):
        """Update measurement statistics for user"""
        if user_id not in self.measurement_statistics:
            self.measurement_statistics[user_id] = {
                'total_measurements': 0,
                'average_efficiency': 0.0,
                'average_information_gain': 0.0,
                'basis_usage': {}
            }

        stats = self.measurement_statistics[user_id]
        stats['total_measurements'] += 1

        # Update averages
        current_efficiency = measurement_metrics.get('measurement_efficiency', 0)
        current_info_gain = measurement_metrics.get('information_gain', 0)

        n = stats['total_measurements']
        stats['average_efficiency'] = ((n-1) * stats['average_efficiency'] + current_efficiency) / n
        stats['average_information_gain'] = ((n-1) * stats['average_information_gain'] + current_info_gain) / n

        # Update basis usage
        basis_name = measurement_basis.value
        if basis_name not in stats['basis_usage']:
            stats['basis_usage'][basis_name] = 0
        stats['basis_usage'][basis_name] += 1
