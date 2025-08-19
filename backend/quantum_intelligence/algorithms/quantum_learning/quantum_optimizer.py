"""
Quantum-Inspired Learning Path Optimization

Advanced system for optimizing learning paths using quantum-inspired algorithms
including quantum annealing, genetic algorithms with quantum operators, and
variational quantum optimization for personalized learning experiences.
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

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .quantum_data_structures import (
    QuantumState, QuantumLearningPath, create_quantum_state
)


class QuantumLearningPathOptimizer:
    """
    🚀 QUANTUM-INSPIRED LEARNING PATH OPTIMIZATION
    
    Advanced system for optimizing learning paths using quantum-inspired algorithms
    including quantum annealing, genetic algorithms with quantum operators, and
    variational quantum optimization for personalized learning experiences.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Optimization algorithms
        self.optimization_algorithms = {
            'quantum_annealing': self._quantum_annealing_optimization,
            'genetic_quantum': self._genetic_quantum_optimization,
            'variational_quantum': self._variational_quantum_optimization,
            'simulated_annealing': self._simulated_annealing_optimization,
            'particle_swarm': self._particle_swarm_optimization
        }
        
        # Configuration parameters
        self.max_iterations = 1000
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.temperature_schedule = self._generate_temperature_schedule()
        
        # Quantum annealing parameters
        self.initial_hamiltonian_strength = 10.0
        self.final_hamiltonian_strength = 0.1
        self.tunneling_strength = 5.0
        self.annealing_time = 100.0
        
        # Optimization tracking
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.best_solutions: Dict[str, Dict[str, Any]] = {}
        self.convergence_data: Dict[str, List[float]] = defaultdict(list)
        
        # Performance metrics
        self.optimization_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        logger.info("Quantum Learning Path Optimizer initialized")
    
    async def optimize_learning_path(
        self,
        user_id: str,
        concepts: List[str],
        user_profile: Dict[str, Any],
        learning_objectives: List[str],
        constraints: Dict[str, Any] = None,
        optimization_method: str = 'quantum_annealing'
    ) -> Dict[str, Any]:
        """
        Optimize learning path using quantum-inspired algorithms
        
        Args:
            user_id: User identifier
            concepts: Available learning concepts
            user_profile: User learning profile
            learning_objectives: Target learning objectives
            constraints: Optimization constraints
            optimization_method: Optimization algorithm to use
            
        Returns:
            Dict: Optimization result with optimal path
        """
        try:
            # Validate optimization method
            if optimization_method not in self.optimization_algorithms:
                optimization_method = 'quantum_annealing'
            
            # Encode optimization problem
            problem_encoding = await self._encode_optimization_problem(
                concepts, user_profile, learning_objectives, constraints
            )
            
            # Run optimization algorithm
            optimization_algorithm = self.optimization_algorithms[optimization_method]
            optimization_result = await optimization_algorithm(
                problem_encoding, user_profile
            )
            
            # Decode optimal path
            optimal_path = await self._decode_optimal_path(
                optimization_result, concepts, problem_encoding
            )
            
            # Calculate path metrics
            path_metrics = await self._calculate_path_metrics(
                optimal_path, problem_encoding, user_profile
            )
            
            # Create quantum learning path
            quantum_path = QuantumLearningPath(
                path_id=f"path_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                concepts=optimal_path,
                quantum_energy=optimization_result.get('best_energy', 0.0),
                classical_energy=optimization_result.get('classical_energy', 0.0),
                quantum_advantage=path_metrics.get('quantum_advantage', 0.0),
                path_coherence=path_metrics.get('coherence', 0.0),
                learning_efficiency=path_metrics.get('efficiency', 0.0),
                user_alignment=path_metrics.get('user_alignment', 0.0),
                optimization_history=optimization_result.get('history', [])
            )
            
            # Store optimization results
            self.best_solutions[user_id] = {
                'quantum_path': quantum_path,
                'optimization_result': optimization_result,
                'path_metrics': path_metrics,
                'optimization_method': optimization_method,
                'timestamp': datetime.now()
            }
            
            return {
                'status': 'success',
                'quantum_path': quantum_path,
                'optimization_result': optimization_result,
                'path_metrics': path_metrics,
                'optimization_method': optimization_method,
                'convergence_iterations': optimization_result.get('iterations', 0)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing learning path: {e}")
            raise QuantumEngineError(f"Failed to optimize learning path: {e}")
    
    async def _quantum_annealing_optimization(
        self,
        problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quantum annealing optimization algorithm"""
        
        # Initialize quantum state in superposition
        num_concepts = problem_encoding['num_concepts']
        initial_amplitudes = [complex(1.0 / np.sqrt(num_concepts), 0) for _ in range(num_concepts)]
        current_state = create_quantum_state(initial_amplitudes)
        
        best_energy = float('inf')
        best_path = None
        optimization_history = []
        
        for iteration in range(self.max_iterations):
            # Calculate current temperature
            temperature = self.temperature_schedule[min(iteration, len(self.temperature_schedule) - 1)]
            
            # Calculate transverse field strength (decreases over time)
            transverse_field = self.initial_hamiltonian_strength * (
                1 - iteration / self.max_iterations
            )
            
            # Apply quantum evolution
            evolved_state = await self._apply_quantum_evolution(
                current_state, problem_encoding, transverse_field, temperature
            )
            
            # Measure quantum state
            measured_path = await self._measure_quantum_path(
                evolved_state, problem_encoding
            )
            
            # Calculate energy of measured path
            energy = await self._calculate_path_energy(measured_path, problem_encoding)
            
            # Update best solution
            if energy < best_energy:
                best_energy = energy
                best_path = measured_path
            
            # Record optimization step
            optimization_history.append({
                'iteration': iteration,
                'energy': energy,
                'temperature': temperature,
                'transverse_field': transverse_field,
                'path': measured_path
            })
            
            current_state = evolved_state
            
            # Check convergence
            if iteration > 100 and len(optimization_history) >= 50:
                recent_energies = [h['energy'] for h in optimization_history[-50:]]
                if max(recent_energies) - min(recent_energies) < 0.01:
                    break
        
        return {
            'best_path': best_path,
            'best_energy': best_energy,
            'classical_energy': await self._calculate_classical_energy(best_path, problem_encoding),
            'iterations': len(optimization_history),
            'history': optimization_history,
            'convergence_achieved': True
        }
    
    async def _genetic_quantum_optimization(
        self,
        problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genetic algorithm with quantum operators"""
        
        # Initialize population
        population = await self._initialize_quantum_population(
            self.population_size, problem_encoding
        )
        
        best_individual = None
        best_fitness = float('-inf')
        optimization_history = []
        
        for generation in range(self.max_iterations // 10):  # Fewer generations for GA
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = await self._calculate_fitness(individual, problem_encoding)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection
            selected_population = await self._quantum_selection(
                population, fitness_scores
            )
            
            # Crossover
            offspring = await self._quantum_crossover(
                selected_population, problem_encoding
            )
            
            # Mutation
            mutated_offspring = await self._quantum_mutation(
                offspring, problem_encoding
            )
            
            # Replace population
            population = mutated_offspring
            
            # Record generation
            optimization_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'average_fitness': np.mean(fitness_scores) if fitness_scores else 0,
                'population_diversity': await self._calculate_population_diversity(population)
            })
        
        return {
            'best_path': best_individual,
            'best_energy': -best_fitness,  # Convert fitness to energy
            'classical_energy': await self._calculate_classical_energy(best_individual, problem_encoding),
            'iterations': len(optimization_history),
            'history': optimization_history,
            'convergence_achieved': True
        }
    
    async def _variational_quantum_optimization(
        self,
        problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Variational quantum optimization algorithm"""
        
        # Initialize variational parameters
        num_params = problem_encoding['num_concepts'] * 2  # Amplitude and phase for each concept
        initial_params = np.random.uniform(0, 2*np.pi, num_params) if NUMPY_AVAILABLE else [random.uniform(0, 2*3.14159) for _ in range(num_params)]
        
        best_params = initial_params.copy() if NUMPY_AVAILABLE else initial_params[:]
        best_energy = float('inf')
        optimization_history = []
        
        # Optimization loop
        for iteration in range(self.max_iterations // 5):  # Fewer iterations for VQE
            # Create quantum state from parameters
            quantum_state = await self._create_variational_state(
                initial_params if iteration == 0 else best_params, problem_encoding
            )
            
            # Calculate expectation value (energy)
            energy = await self._calculate_expectation_value(
                quantum_state, problem_encoding
            )
            
            # Update best solution
            if energy < best_energy:
                best_energy = energy
                best_params = (initial_params if iteration == 0 else best_params).copy() if NUMPY_AVAILABLE else (initial_params if iteration == 0 else best_params)[:]
            
            # Parameter update (simplified gradient descent)
            if iteration < self.max_iterations // 5 - 1:
                gradient = await self._calculate_parameter_gradient(
                    initial_params if iteration == 0 else best_params, problem_encoding
                )
                learning_rate = 0.1 * (0.95 ** iteration)
                
                if NUMPY_AVAILABLE:
                    initial_params = initial_params - learning_rate * gradient
                else:
                    initial_params = [p - learning_rate * g for p, g in zip(initial_params, gradient)]
            
            # Record optimization step
            optimization_history.append({
                'iteration': iteration,
                'energy': energy,
                'parameters': (initial_params if iteration == 0 else best_params).copy() if NUMPY_AVAILABLE else (initial_params if iteration == 0 else best_params)[:]
            })
        
        # Decode best path from parameters
        best_path = await self._decode_variational_path(best_params, problem_encoding)
        
        return {
            'best_path': best_path,
            'best_energy': best_energy,
            'classical_energy': await self._calculate_classical_energy(best_path, problem_encoding),
            'iterations': len(optimization_history),
            'history': optimization_history,
            'convergence_achieved': True
        }
    
    async def _simulated_annealing_optimization(
        self,
        problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulated annealing optimization algorithm"""
        
        # Initialize random solution
        concepts = problem_encoding['concept_names']
        current_path = random.sample(concepts, len(concepts))
        current_energy = await self._calculate_path_energy(current_path, problem_encoding)
        
        best_path = current_path[:]
        best_energy = current_energy
        optimization_history = []
        
        for iteration in range(self.max_iterations):
            # Calculate temperature
            temperature = self.temperature_schedule[min(iteration, len(self.temperature_schedule) - 1)]
            
            # Generate neighbor solution
            neighbor_path = await self._generate_neighbor_solution(current_path)
            neighbor_energy = await self._calculate_path_energy(neighbor_path, problem_encoding)
            
            # Accept or reject neighbor
            if neighbor_energy < current_energy:
                # Accept better solution
                current_path = neighbor_path
                current_energy = neighbor_energy
            else:
                # Accept worse solution with probability
                delta_energy = neighbor_energy - current_energy
                acceptance_probability = np.exp(-delta_energy / temperature) if temperature > 0 else 0
                
                if random.random() < acceptance_probability:
                    current_path = neighbor_path
                    current_energy = neighbor_energy
            
            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_path = current_path[:]
            
            # Record optimization step
            optimization_history.append({
                'iteration': iteration,
                'energy': current_energy,
                'best_energy': best_energy,
                'temperature': temperature,
                'path': current_path[:]
            })
        
        return {
            'best_path': best_path,
            'best_energy': best_energy,
            'classical_energy': best_energy,  # SA is classical
            'iterations': len(optimization_history),
            'history': optimization_history,
            'convergence_achieved': True
        }
    
    async def _particle_swarm_optimization(
        self,
        problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Particle swarm optimization algorithm"""
        
        # Initialize swarm
        swarm_size = min(self.population_size, 30)
        particles = await self._initialize_particle_swarm(swarm_size, problem_encoding)
        
        global_best_position = None
        global_best_fitness = float('-inf')
        optimization_history = []
        
        for iteration in range(self.max_iterations // 20):  # Fewer iterations for PSO
            for particle in particles:
                # Evaluate fitness
                fitness = await self._calculate_fitness(particle['position'], problem_encoding)
                
                # Update personal best
                if fitness > particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'][:]
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle['position'][:]
            
            # Update particle velocities and positions
            for particle in particles:
                await self._update_particle_velocity(particle, global_best_position)
                await self._update_particle_position(particle, problem_encoding)
            
            # Record iteration
            optimization_history.append({
                'iteration': iteration,
                'global_best_fitness': global_best_fitness,
                'swarm_diversity': await self._calculate_swarm_diversity(particles)
            })
        
        return {
            'best_path': global_best_position,
            'best_energy': -global_best_fitness,  # Convert fitness to energy
            'classical_energy': await self._calculate_classical_energy(global_best_position, problem_encoding),
            'iterations': len(optimization_history),
            'history': optimization_history,
            'convergence_achieved': True
        }
    
    # Helper methods for optimization algorithms
    def _generate_temperature_schedule(self) -> List[float]:
        """Generate temperature schedule for annealing"""
        return [10.0 * (0.95 ** i) for i in range(self.max_iterations)]
    
    async def _encode_optimization_problem(
        self,
        concepts: List[str],
        user_profile: Dict[str, Any],
        learning_objectives: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode learning optimization problem"""
        
        # Create Hamiltonian matrix for concept relationships
        num_concepts = len(concepts)
        if NUMPY_AVAILABLE:
            hamiltonian = np.zeros((num_concepts, num_concepts), dtype=complex)
        else:
            hamiltonian = [[complex(0, 0) for _ in range(num_concepts)] for _ in range(num_concepts)]
        
        # Encode concept dependencies and relationships
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts):
                if i != j:
                    # Calculate relationship strength
                    relationship_strength = await self._calculate_concept_relationship(
                        concept_a, concept_b, user_profile
                    )
                    
                    if NUMPY_AVAILABLE:
                        hamiltonian[i, j] = relationship_strength
                    else:
                        hamiltonian[i][j] = complex(relationship_strength, 0)
        
        # Encode user preferences
        preference_vector = await self._encode_user_preferences(user_profile, concepts)
        
        # Encode learning objectives
        objective_matrix = await self._encode_learning_objectives(learning_objectives, concepts)
        
        return {
            'hamiltonian': hamiltonian,
            'preference_vector': preference_vector,
            'objective_matrix': objective_matrix,
            'constraints': constraints or {},
            'num_concepts': num_concepts,
            'concept_names': concepts
        }
    
    async def _calculate_concept_relationship(
        self, concept_a: str, concept_b: str, user_profile: Dict[str, Any]
    ) -> float:
        """Calculate relationship strength between concepts"""
        # Simplified relationship calculation
        base_relationship = random.uniform(0.1, 0.9)
        user_modifier = user_profile.get('learning_velocity', 0.7)
        return base_relationship * user_modifier
    
    async def _encode_user_preferences(
        self, user_profile: Dict[str, Any], concepts: List[str]
    ) -> List[float]:
        """Encode user preferences as preference vector"""
        preferences = []
        for concept in concepts:
            preference_score = (
                user_profile.get('curiosity_index', 0.7) * 0.3 +
                user_profile.get('difficulty_preference', 0.5) * 0.4 +
                user_profile.get('learning_velocity', 0.6) * 0.3
            )
            preferences.append(preference_score)
        return preferences
    
    async def _encode_learning_objectives(
        self, learning_objectives: List[str], concepts: List[str]
    ) -> List[List[float]]:
        """Encode learning objectives as objective matrix"""
        objective_matrix = []
        for objective in learning_objectives:
            objective_row = []
            for concept in concepts:
                relevance = await self._calculate_objective_relevance(objective, concept)
                objective_row.append(relevance)
            objective_matrix.append(objective_row)
        return objective_matrix
    
    async def _calculate_objective_relevance(self, objective: str, concept: str) -> float:
        """Calculate relevance between learning objective and concept"""
        return random.uniform(0.2, 1.0)
    
    async def _calculate_path_energy(
        self, path: List[str], problem_encoding: Dict[str, Any]
    ) -> float:
        """Calculate energy (cost) of learning path"""
        total_energy = 0.0
        hamiltonian = problem_encoding['hamiltonian']
        concept_names = problem_encoding['concept_names']
        
        for i, concept_a in enumerate(path):
            for j, concept_b in enumerate(path):
                if i != j and concept_a in concept_names and concept_b in concept_names:
                    idx_a = concept_names.index(concept_a)
                    idx_b = concept_names.index(concept_b)
                    
                    if NUMPY_AVAILABLE:
                        total_energy += np.real(hamiltonian[idx_a, idx_b])
                    else:
                        total_energy += hamiltonian[idx_a][idx_b].real
        
        # Add preference penalty
        preference_vector = problem_encoding['preference_vector']
        for concept in path:
            if concept in concept_names:
                idx = concept_names.index(concept)
                total_energy -= preference_vector[idx]  # Negative because we minimize
        
        return total_energy
    
    async def _calculate_classical_energy(
        self, path: List[str], problem_encoding: Dict[str, Any]
    ) -> float:
        """Calculate classical energy for comparison"""
        # For classical comparison, use random path
        concepts = problem_encoding['concept_names']
        random_path = random.sample(concepts, len(concepts))
        return await self._calculate_path_energy(random_path, problem_encoding)

    async def _apply_quantum_evolution(
        self, quantum_state: QuantumState, problem_encoding: Dict[str, Any],
        transverse_field: float, temperature: float
    ) -> QuantumState:
        """Apply quantum evolution to state"""

        # Apply transverse field (quantum tunneling)
        evolved_amplitudes = quantum_state.amplitudes[:]

        for i in range(len(evolved_amplitudes)):
            # Add quantum tunneling effects
            tunneling_amplitude = transverse_field * 0.1 * cmath.exp(1j * random.uniform(0, 2*np.pi))
            evolved_amplitudes[i] += tunneling_amplitude

            # Add thermal fluctuations
            thermal_noise = temperature * 0.01 * (random.random() - 0.5)
            evolved_amplitudes[i] += complex(thermal_noise, thermal_noise)

        # Create evolved state
        evolved_state = QuantumState(
            amplitudes=evolved_amplitudes,
            coherence_time=quantum_state.coherence_time,
            decoherence_rate=quantum_state.decoherence_rate
        )

        return evolved_state

    async def _measure_quantum_path(
        self, quantum_state: QuantumState, problem_encoding: Dict[str, Any]
    ) -> List[str]:
        """Measure quantum state to get classical learning path"""

        # Get measurement probabilities
        probabilities = quantum_state.get_probability_distribution()
        concept_names = problem_encoding['concept_names']

        # Sample path based on probabilities
        measured_path = []
        available_concepts = concept_names[:]

        while available_concepts and len(measured_path) < len(concept_names):
            # Calculate selection probabilities
            if len(probabilities) > len(available_concepts):
                selection_probs = probabilities[:len(available_concepts)]
            else:
                selection_probs = probabilities + [0.1] * (len(available_concepts) - len(probabilities))

            # Normalize probabilities
            total_prob = sum(selection_probs)
            if total_prob > 0:
                selection_probs = [p / total_prob for p in selection_probs]
            else:
                selection_probs = [1.0 / len(available_concepts)] * len(available_concepts)

            # Sample concept
            rand_val = random.random()
            cumulative_prob = 0.0
            selected_idx = 0

            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_idx = i
                    break

            # Add selected concept to path
            if selected_idx < len(available_concepts):
                selected_concept = available_concepts[selected_idx]
                measured_path.append(selected_concept)
                available_concepts.remove(selected_concept)

        return measured_path

    async def _decode_optimal_path(
        self, optimization_result: Dict[str, Any], concepts: List[str],
        problem_encoding: Dict[str, Any]
    ) -> List[str]:
        """Decode optimal path from optimization result"""
        return optimization_result.get('best_path', concepts[:])

    async def _calculate_path_metrics(
        self, optimal_path: List[str], problem_encoding: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive path metrics"""

        # Calculate path coherence
        coherence = await self._calculate_path_coherence(optimal_path, problem_encoding)

        # Calculate learning efficiency
        efficiency = await self._calculate_learning_efficiency(optimal_path, user_profile)

        # Calculate user alignment
        user_alignment = await self._calculate_user_alignment(optimal_path, user_profile)

        # Calculate quantum advantage
        quantum_advantage = await self._calculate_quantum_advantage_metric(
            optimal_path, problem_encoding
        )

        return {
            'coherence': coherence,
            'efficiency': efficiency,
            'user_alignment': user_alignment,
            'quantum_advantage': quantum_advantage,
            'path_length': len(optimal_path),
            'complexity_score': await self._calculate_complexity_score(optimal_path)
        }

    async def _calculate_path_coherence(
        self, path: List[str], problem_encoding: Dict[str, Any]
    ) -> float:
        """Calculate coherence of learning path"""
        if len(path) < 2:
            return 1.0

        total_coherence = 0.0
        for i in range(len(path) - 1):
            concept_a = path[i]
            concept_b = path[i + 1]
            relationship = await self._calculate_concept_relationship(
                concept_a, concept_b, {}
            )
            total_coherence += relationship

        return total_coherence / (len(path) - 1)

    async def _calculate_learning_efficiency(
        self, path: List[str], user_profile: Dict[str, Any]
    ) -> float:
        """Calculate learning efficiency of path"""
        base_efficiency = 0.7

        # Adjust based on user profile
        learning_velocity = user_profile.get('learning_velocity', 0.6)
        difficulty_preference = user_profile.get('difficulty_preference', 0.5)

        efficiency = base_efficiency * learning_velocity + difficulty_preference * 0.2
        return min(1.0, max(0.0, efficiency))

    async def _calculate_user_alignment(
        self, path: List[str], user_profile: Dict[str, Any]
    ) -> float:
        """Calculate alignment with user preferences"""
        alignment_score = 0.0

        for concept in path:
            # Calculate concept alignment with user preferences
            concept_alignment = (
                user_profile.get('curiosity_index', 0.7) * 0.4 +
                user_profile.get('engagement_preference', 0.6) * 0.3 +
                user_profile.get('learning_style_match', 0.5) * 0.3
            )
            alignment_score += concept_alignment

        return alignment_score / len(path) if path else 0.0

    async def _calculate_quantum_advantage_metric(
        self, path: List[str], problem_encoding: Dict[str, Any]
    ) -> float:
        """Calculate quantum advantage metric"""
        # Compare quantum path energy to classical random path
        quantum_energy = await self._calculate_path_energy(path, problem_encoding)
        classical_energy = await self._calculate_classical_energy(path, problem_encoding)

        if classical_energy != 0:
            advantage = (classical_energy - quantum_energy) / abs(classical_energy)
        else:
            advantage = 0.0

        return max(0.0, advantage)

    async def _calculate_complexity_score(self, path: List[str]) -> float:
        """Calculate complexity score of learning path"""
        # Simplified complexity based on path length and concept diversity
        base_complexity = len(path) / 20.0  # Normalize by typical path length
        diversity_bonus = len(set(path)) / len(path) if path else 0

        return min(1.0, base_complexity + diversity_bonus * 0.2)
