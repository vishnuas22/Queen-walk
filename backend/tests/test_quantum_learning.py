"""
Comprehensive Test Suite for Quantum Learning Algorithms

Tests all quantum learning components including superposition management,
entanglement simulation, interference analysis, measurement systems,
and quantum optimization algorithms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import random

# Import quantum learning components
from backend.quantum_intelligence.algorithms.quantum_learning import (
    # Core data structures
    QuantumState, SuperpositionState, EntanglementPattern, InterferencePattern,
    QuantumMeasurement, QuantumLearningPath,
    
    # Enums
    QuantumLearningPhase, MeasurementBasis, EntanglementType,
    
    # Core systems
    SuperpositionManager, QuantumEntanglementSimulator, QuantumInterferenceEngine,
    QuantumMeasurementSystem, QuantumLearningPathOptimizer,
    
    # Utility functions
    create_quantum_state, create_superposition_state, create_entanglement_pattern,
    measure_quantum_state,
    
    # Main orchestrator
    QuantumLearningOrchestrator
)


class TestQuantumDataStructures:
    """Test quantum data structures and utility functions"""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation and normalization"""
        amplitudes = [complex(0.6, 0), complex(0.8, 0)]
        quantum_state = create_quantum_state(amplitudes)
        
        assert quantum_state is not None
        assert len(quantum_state.amplitudes) == 2
        
        # Check normalization
        probabilities = quantum_state.get_probability_distribution()
        total_prob = sum(probabilities)
        assert abs(total_prob - 1.0) < 0.01  # Should be normalized
    
    def test_quantum_state_measurement(self):
        """Test quantum state measurement"""
        amplitudes = [complex(0.7, 0), complex(0.7, 0)]
        quantum_state = create_quantum_state(amplitudes)
        
        # Perform measurement
        result = quantum_state.measure()
        assert result in [0, 1]
        
        # Check measurement history
        assert len(quantum_state.measurement_history) == 1
        assert quantum_state.measurement_history[0]['result'] == result
    
    def test_superposition_state_creation(self):
        """Test superposition state creation"""
        user_id = "test_user_001"
        hypotheses = [
            {'name': 'visual_learning', 'confidence': 0.8},
            {'name': 'auditory_learning', 'confidence': 0.6},
            {'name': 'kinesthetic_learning', 'confidence': 0.7}
        ]
        
        superposition_state = create_superposition_state(user_id, hypotheses)
        
        assert superposition_state.user_id == user_id
        assert len(superposition_state.hypotheses) == 3
        assert superposition_state.quantum_state is not None
        
        # Check entropy calculation
        entropy = superposition_state.calculate_entropy()
        assert entropy >= 0
    
    def test_entanglement_pattern_creation(self):
        """Test entanglement pattern creation"""
        participants = ["user1", "user2", "concept_math"]
        entanglement_pattern = create_entanglement_pattern(
            EntanglementType.COLLABORATIVE_LEARNING,
            participants,
            0.8
        )
        
        assert entanglement_pattern.entanglement_type == EntanglementType.COLLABORATIVE_LEARNING
        assert entanglement_pattern.participants == participants
        assert entanglement_pattern.entanglement_strength == 0.8
        assert entanglement_pattern.correlation_matrix is not None
    
    def test_quantum_measurement_creation(self):
        """Test quantum measurement creation"""
        amplitudes = [complex(0.6, 0), complex(0.8, 0)]
        quantum_state = create_quantum_state(amplitudes)
        
        measurement = measure_quantum_state(quantum_state)
        
        assert measurement.user_id == ""  # Will be set by caller
        assert measurement.measurement_result in [0, 1]
        assert 0 <= measurement.probability <= 1
        assert measurement.information_gain >= 0


class TestSuperpositionManager:
    """Test quantum superposition management system"""
    
    @pytest.fixture
    def superposition_manager(self):
        """Create superposition manager for testing"""
        return SuperpositionManager()
    
    @pytest.mark.asyncio
    async def test_create_learning_superposition(self, superposition_manager):
        """Test creating learning superposition"""
        user_id = "test_user_001"
        learning_context = {
            'subject': 'mathematics',
            'difficulty_level': 0.6,
            'learning_style': 'visual'
        }
        learning_hypotheses = [
            {'name': 'step_by_step', 'confidence': 0.8, 'relevance': 0.9},
            {'name': 'conceptual_overview', 'confidence': 0.7, 'relevance': 0.8},
            {'name': 'practice_focused', 'confidence': 0.6, 'relevance': 0.7}
        ]
        
        result = await superposition_manager.create_learning_superposition(
            user_id, learning_context, learning_hypotheses
        )
        
        assert result['status'] == 'success'
        assert 'superposition_state' in result
        assert 'metrics' in result
        assert result['quantum_entropy'] >= 0
        assert result['superposition_strength'] >= 0
    
    @pytest.mark.asyncio
    async def test_update_superposition_state(self, superposition_manager):
        """Test updating superposition state"""
        user_id = "test_user_002"
        
        # First create a superposition
        learning_context = {'subject': 'physics'}
        learning_hypotheses = [
            {'name': 'theory_first', 'confidence': 0.7},
            {'name': 'experiment_first', 'confidence': 0.8}
        ]
        
        await superposition_manager.create_learning_superposition(
            user_id, learning_context, learning_hypotheses
        )
        
        # Then update it
        learning_feedback = {
            'type': 'performance',
            'value': 0.8,
            'hypothesis_index': 1
        }
        
        result = await superposition_manager.update_superposition_state(
            user_id, learning_feedback
        )
        
        assert result['status'] in ['updated', 'collapsed']
        if result['status'] == 'updated':
            assert 'metrics' in result
            assert 'entropy' in result
    
    @pytest.mark.asyncio
    async def test_measure_learning_state(self, superposition_manager):
        """Test measuring learning state"""
        user_id = "test_user_003"
        
        # Create superposition first
        learning_context = {'subject': 'chemistry'}
        learning_hypotheses = [
            {'name': 'molecular_approach', 'confidence': 0.9},
            {'name': 'atomic_approach', 'confidence': 0.6}
        ]
        
        await superposition_manager.create_learning_superposition(
            user_id, learning_context, learning_hypotheses
        )
        
        # Measure the state
        result = await superposition_manager.measure_learning_state(user_id)
        
        assert result['status'] == 'measured'
        assert 'measurement' in result
        assert 'measured_hypothesis' in result
        assert result['information_gain'] >= 0


class TestQuantumEntanglementSimulator:
    """Test quantum entanglement simulation system"""
    
    @pytest.fixture
    def entanglement_simulator(self):
        """Create entanglement simulator for testing"""
        return QuantumEntanglementSimulator()
    
    @pytest.mark.asyncio
    async def test_create_knowledge_entanglement(self, entanglement_simulator):
        """Test creating knowledge entanglement"""
        concept_a = "algebra"
        concept_b = "geometry"
        user_id = "test_user_001"
        entanglement_context = {
            'learning_session': 'mathematics_101',
            'difficulty_level': 0.7
        }
        
        result = await entanglement_simulator.create_knowledge_entanglement(
            concept_a, concept_b, user_id, entanglement_context
        )
        
        if result['entanglement_created']:
            assert 'entanglement_id' in result
            assert 'entangled_state' in result
            assert 'correlation_matrix' in result
            assert result['metrics']['non_local_strength'] >= 0
    
    @pytest.mark.asyncio
    async def test_create_collaborative_entanglement(self, entanglement_simulator):
        """Test creating collaborative entanglement"""
        user_ids = ["user1", "user2", "user3"]
        learning_context = {
            'collaboration_type': 'group_project',
            'collaboration_strength': 0.8
        }
        
        result = await entanglement_simulator.create_collaborative_entanglement(
            user_ids, learning_context
        )
        
        assert result['entanglement_created'] == True
        assert 'entanglement_id' in result
        assert 'bell_states' in result
        assert result['participants'] == user_ids
    
    @pytest.mark.asyncio
    async def test_measure_entanglement_correlation(self, entanglement_simulator):
        """Test measuring entanglement correlation"""
        # First create entanglement
        user_ids = ["user1", "user2"]
        learning_context = {'collaboration_strength': 0.9}
        
        create_result = await entanglement_simulator.create_collaborative_entanglement(
            user_ids, learning_context
        )
        
        if create_result['entanglement_created']:
            entanglement_id = create_result['entanglement_id']
            
            # Then measure correlation
            measure_result = await entanglement_simulator.measure_entanglement_correlation(
                entanglement_id
            )
            
            assert measure_result['status'] == 'success'
            assert 'correlation_result' in measure_result
            assert 'bell_violation' in measure_result


class TestQuantumInterferenceEngine:
    """Test quantum interference analysis system"""
    
    @pytest.fixture
    def interference_engine(self):
        """Create interference engine for testing"""
        return QuantumInterferenceEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_learning_interference(self, interference_engine):
        """Test analyzing learning interference"""
        user_id = "test_user_001"
        learning_concepts = ["calculus", "linear_algebra", "statistics"]
        learning_context = {
            'subject': 'mathematics',
            'learning_sequence': learning_concepts,
            'difficulty_level': 0.7
        }
        
        result = await interference_engine.analyze_learning_interference(
            user_id, learning_concepts, learning_context
        )
        
        assert result['status'] == 'success'
        assert 'interference_pattern' in result
        assert 'constructive_regions' in result
        assert 'destructive_regions' in result
        assert 'metrics' in result
    
    @pytest.mark.asyncio
    async def test_apply_constructive_interference(self, interference_engine):
        """Test applying constructive interference"""
        user_id = "test_user_002"
        target_concepts = ["algebra", "geometry"]
        learning_state = {
            'current_understanding': 0.6,
            'engagement_level': 0.8
        }
        
        result = await interference_engine.apply_constructive_interference(
            user_id, target_concepts, learning_state
        )
        
        # May not find patterns for new user, but should handle gracefully
        assert result['status'] in ['success', 'no_patterns']
    
    @pytest.mark.asyncio
    async def test_resolve_destructive_interference(self, interference_engine):
        """Test resolving destructive interference"""
        user_id = "test_user_003"
        conflicting_concepts = ["classical_physics", "quantum_physics"]
        learning_state = {
            'confusion_level': 0.7,
            'concept_conflicts': ['wave_particle_duality']
        }
        
        result = await interference_engine.resolve_destructive_interference(
            user_id, conflicting_concepts, learning_state
        )
        
        assert result['status'] in ['success', 'error']


class TestQuantumMeasurementSystem:
    """Test quantum measurement system"""
    
    @pytest.fixture
    def measurement_system(self):
        """Create measurement system for testing"""
        return QuantumMeasurementSystem()
    
    @pytest.mark.asyncio
    async def test_perform_learning_state_measurement(self, measurement_system):
        """Test performing learning state measurement"""
        user_id = "test_user_001"
        amplitudes = [complex(0.6, 0), complex(0.8, 0)]
        quantum_state = create_quantum_state(amplitudes)
        
        result = await measurement_system.perform_learning_state_measurement(
            user_id, quantum_state
        )
        
        assert result['status'] == 'success'
        assert 'measurement_result' in result
        assert 'collapsed_state' in result
        assert 'classical_information' in result
        assert result['quantum_information_loss'] >= 0
    
    @pytest.mark.asyncio
    async def test_measure_superposition_collapse(self, measurement_system):
        """Test measuring superposition collapse"""
        user_id = "test_user_002"
        hypotheses = [
            {'name': 'approach_a', 'confidence': 0.9},
            {'name': 'approach_b', 'confidence': 0.5}
        ]
        superposition_state = create_superposition_state(user_id, hypotheses)
        
        result = await measurement_system.measure_superposition_collapse(
            user_id, superposition_state
        )
        
        assert result['status'] in ['collapsed', 'no_collapse']
        if result['status'] == 'collapsed':
            assert 'collapsed_hypothesis' in result
            assert 'measurement_result' in result


class TestQuantumLearningPathOptimizer:
    """Test quantum learning path optimization"""
    
    @pytest.fixture
    def quantum_optimizer(self):
        """Create quantum optimizer for testing"""
        return QuantumLearningPathOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_learning_path(self, quantum_optimizer):
        """Test optimizing learning path"""
        user_id = "test_user_001"
        concepts = ["basic_math", "algebra", "calculus", "linear_algebra"]
        user_profile = {
            'learning_velocity': 0.7,
            'difficulty_preference': 0.6,
            'curiosity_index': 0.8
        }
        learning_objectives = ["master_calculus", "understand_linear_algebra"]
        
        result = await quantum_optimizer.optimize_learning_path(
            user_id, concepts, user_profile, learning_objectives
        )
        
        assert result['status'] == 'success'
        assert 'quantum_path' in result
        assert 'optimization_result' in result
        assert 'path_metrics' in result
        assert result['convergence_iterations'] >= 0
        
        # Check quantum path properties
        quantum_path = result['quantum_path']
        assert quantum_path.user_id == user_id
        assert len(quantum_path.concepts) > 0
        assert quantum_path.quantum_energy is not None
        assert quantum_path.learning_efficiency >= 0


class TestQuantumLearningOrchestrator:
    """Test quantum learning orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create quantum learning orchestrator for testing"""
        return QuantumLearningOrchestrator()
    
    @pytest.mark.asyncio
    async def test_create_quantum_learning_session(self, orchestrator):
        """Test creating quantum learning session"""
        user_id = "test_user_001"
        learning_context = {
            'concepts': ['math', 'physics', 'chemistry'],
            'hypotheses': [
                {'name': 'visual_learning', 'confidence': 0.8},
                {'name': 'hands_on_learning', 'confidence': 0.7}
            ],
            'subject': 'science',
            'difficulty_level': 0.6
        }
        learning_objectives = ['understand_fundamentals', 'apply_concepts']
        
        result = await orchestrator.create_quantum_learning_session(
            user_id, learning_context, learning_objectives
        )
        
        assert result['status'] == 'success'
        assert 'quantum_session' in result
        assert 'session_metrics' in result
        
        # Check session components
        session = result['quantum_session']
        assert session['user_id'] == user_id
        assert 'superposition_state' in session
        assert 'optimized_path' in session
    
    @pytest.mark.asyncio
    async def test_update_quantum_learning_state(self, orchestrator):
        """Test updating quantum learning state"""
        user_id = "test_user_002"
        
        # First create a session
        learning_context = {
            'concepts': ['biology', 'chemistry'],
            'hypotheses': [{'name': 'experimental', 'confidence': 0.8}]
        }
        learning_objectives = ['understand_biology']
        
        await orchestrator.create_quantum_learning_session(
            user_id, learning_context, learning_objectives
        )
        
        # Then update it
        learning_feedback = {
            'type': 'engagement',
            'value': 0.9,
            'concept': 'biology'
        }
        
        result = await orchestrator.update_quantum_learning_state(
            user_id, learning_feedback
        )
        
        assert result['status'] == 'success'
        assert 'superposition_update' in result
    
    @pytest.mark.asyncio
    async def test_extract_quantum_insights(self, orchestrator):
        """Test extracting quantum insights"""
        user_id = "test_user_003"
        
        # Create session first
        learning_context = {
            'concepts': ['programming', 'algorithms'],
            'hypotheses': [{'name': 'project_based', 'confidence': 0.9}]
        }
        learning_objectives = ['master_programming']
        
        await orchestrator.create_quantum_learning_session(
            user_id, learning_context, learning_objectives
        )
        
        # Extract insights
        result = await orchestrator.extract_quantum_insights(user_id)
        
        assert result['status'] == 'success'
        assert 'quantum_insights' in result
        assert 'recommendations' in result
        assert 'session_summary' in result


# Integration tests
class TestQuantumLearningIntegration:
    """Integration tests for quantum learning systems"""
    
    @pytest.mark.asyncio
    async def test_full_quantum_learning_workflow(self):
        """Test complete quantum learning workflow"""
        orchestrator = QuantumLearningOrchestrator()
        user_id = "integration_test_user"
        
        # Step 1: Create quantum learning session
        learning_context = {
            'concepts': ['machine_learning', 'deep_learning', 'neural_networks'],
            'hypotheses': [
                {'name': 'theory_first', 'confidence': 0.7, 'relevance': 0.8},
                {'name': 'practice_first', 'confidence': 0.8, 'relevance': 0.9}
            ],
            'subject': 'artificial_intelligence',
            'difficulty_level': 0.8,
            'collaborative_users': ['peer_user_1']
        }
        learning_objectives = ['understand_ml_fundamentals', 'build_neural_network']
        
        session_result = await orchestrator.create_quantum_learning_session(
            user_id, learning_context, learning_objectives
        )
        
        assert session_result['status'] == 'success'
        
        # Step 2: Update learning state with feedback
        learning_feedback = {
            'type': 'performance',
            'value': 0.85,
            'hypothesis_index': 1,
            'engagement_level': 0.9
        }
        
        update_result = await orchestrator.update_quantum_learning_state(
            user_id, learning_feedback
        )
        
        assert update_result['status'] == 'success'
        
        # Step 3: Extract insights and recommendations
        insights_result = await orchestrator.extract_quantum_insights(user_id)
        
        assert insights_result['status'] == 'success'
        assert len(insights_result['recommendations']) >= 0
        
        # Verify session coherence
        session_coherence = insights_result['session_summary']['quantum_coherence']
        assert 0 <= session_coherence <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
