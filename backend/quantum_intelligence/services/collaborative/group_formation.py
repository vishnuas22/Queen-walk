"""
Group Formation Services

Extracted from quantum_intelligence_engine.py (lines 10289-12523) - advanced group formation
algorithms, learning group management, team dynamics analysis, and collaborative project management.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import math
import random

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


@dataclass
class LearningGroup:
    """Learning group data structure"""
    group_id: str = ""
    group_name: str = ""
    members: List[Dict[str, Any]] = field(default_factory=list)
    group_size: int = 0
    formation_strategy: str = ""
    compatibility_score: float = 0.0
    diversity_score: float = 0.0
    learning_objectives: List[str] = field(default_factory=list)
    group_dynamics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    formation_timestamp: str = ""
    status: str = "active"


@dataclass
class TeamDynamicsAnalysis:
    """Team dynamics analysis results"""
    group_id: str = ""
    cohesion_score: float = 0.0
    communication_effectiveness: float = 0.0
    role_distribution: Dict[str, List[str]] = field(default_factory=dict)
    conflict_indicators: List[str] = field(default_factory=list)
    collaboration_patterns: Dict[str, Any] = field(default_factory=dict)
    leadership_emergence: Dict[str, float] = field(default_factory=dict)
    group_satisfaction: float = 0.0
    productivity_score: float = 0.0
    improvement_recommendations: List[str] = field(default_factory=list)


@dataclass
class CollaborativeProject:
    """Collaborative project data structure"""
    project_id: str = ""
    project_name: str = ""
    group_id: str = ""
    project_type: str = ""
    learning_objectives: List[str] = field(default_factory=list)
    project_phases: List[Dict[str, Any]] = field(default_factory=list)
    role_assignments: Dict[str, str] = field(default_factory=dict)
    timeline: Dict[str, str] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    progress_tracking: Dict[str, Any] = field(default_factory=dict)
    collaboration_tools: List[str] = field(default_factory=list)


class GroupFormationEngine:
    """
    👥 GROUP FORMATION ENGINE
    
    Advanced group formation algorithms for optimal learning group creation.
    Extracted from the original quantum engine's collaborative intelligence logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Group formation configuration
        self.config = {
            'default_group_size': 4,
            'min_group_size': 2,
            'max_group_size': 6,
            'formation_algorithms': ['compatibility_based', 'diversity_based', 'skill_balanced', 'random'],
            'optimization_iterations': 100,
            'compatibility_weight': 0.4,
            'diversity_weight': 0.3,
            'skill_balance_weight': 0.3
        }
        
        # Formation tracking
        self.formation_history = []
        self.group_performance_data = {}
        
        logger.info("Group Formation Engine initialized")
    
    async def form_learning_groups(self,
                                 participants: List[Dict[str, Any]],
                                 formation_criteria: Dict[str, Any],
                                 constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Form optimal learning groups from a pool of participants
        
        Args:
            participants: List of participant profiles
            formation_criteria: Criteria for group formation
            constraints: Optional constraints (group size, specific requirements)
            
        Returns:
            Dict with formed groups and formation analysis
        """
        try:
            # Validate inputs
            if len(participants) < 2:
                return {'status': 'error', 'error': 'Insufficient participants for group formation'}
            
            # Determine formation strategy
            formation_strategy = formation_criteria.get('strategy', 'compatibility_based')
            
            # Apply formation algorithm
            if formation_strategy == 'compatibility_based':
                groups = await self._form_compatibility_based_groups(participants, formation_criteria, constraints)
            elif formation_strategy == 'diversity_based':
                groups = await self._form_diversity_based_groups(participants, formation_criteria, constraints)
            elif formation_strategy == 'skill_balanced':
                groups = await self._form_skill_balanced_groups(participants, formation_criteria, constraints)
            elif formation_strategy == 'hybrid_optimized':
                groups = await self._form_hybrid_optimized_groups(participants, formation_criteria, constraints)
            else:
                groups = await self._form_random_groups(participants, formation_criteria, constraints)
            
            # Analyze formation quality
            formation_analysis = await self._analyze_formation_quality(groups, participants, formation_criteria)
            
            # Generate formation insights
            formation_insights = await self._generate_formation_insights(groups, formation_analysis)
            
            # Store formation history
            self.formation_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'strategy': formation_strategy,
                'participant_count': len(participants),
                'groups_formed': len(groups),
                'formation_quality': formation_analysis.get('overall_quality', 0.0)
            })
            
            return {
                'status': 'success',
                'groups': [group.__dict__ for group in groups],
                'formation_analysis': formation_analysis,
                'formation_insights': formation_insights,
                'formation_strategy': formation_strategy,
                'formation_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error forming learning groups: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _form_compatibility_based_groups(self,
                                             participants: List[Dict[str, Any]],
                                             criteria: Dict[str, Any],
                                             constraints: Optional[Dict[str, Any]]) -> List[LearningGroup]:
        """Form groups based on participant compatibility"""
        target_group_size = constraints.get('group_size', self.config['default_group_size']) if constraints else self.config['default_group_size']
        
        # Calculate compatibility matrix
        compatibility_matrix = await self._calculate_compatibility_matrix(participants)
        
        # Use greedy algorithm to form compatible groups
        groups = []
        remaining_participants = participants.copy()
        group_counter = 1
        
        while len(remaining_participants) >= self.config['min_group_size']:
            # Start with highest compatibility pair
            best_pair = self._find_best_compatibility_pair(remaining_participants, compatibility_matrix)
            
            if not best_pair:
                break
            
            # Build group around this pair
            group_members = [best_pair[0], best_pair[1]]
            remaining_participants.remove(best_pair[0])
            remaining_participants.remove(best_pair[1])
            
            # Add more members to reach target size
            while len(group_members) < target_group_size and remaining_participants:
                best_addition = self._find_best_group_addition(
                    group_members, remaining_participants, compatibility_matrix
                )
                
                if best_addition and self._calculate_group_compatibility(group_members + [best_addition], compatibility_matrix) > 0.6:
                    group_members.append(best_addition)
                    remaining_participants.remove(best_addition)
                else:
                    break
            
            # Create group
            group = await self._create_learning_group(
                f"group_{group_counter}",
                group_members,
                "compatibility_based",
                criteria.get('learning_objectives', [])
            )
            groups.append(group)
            group_counter += 1
        
        # Handle remaining participants
        if remaining_participants:
            if len(remaining_participants) >= self.config['min_group_size']:
                # Form additional group
                group = await self._create_learning_group(
                    f"group_{group_counter}",
                    remaining_participants,
                    "compatibility_based",
                    criteria.get('learning_objectives', [])
                )
                groups.append(group)
            else:
                # Distribute to existing groups
                await self._distribute_remaining_participants(groups, remaining_participants, compatibility_matrix)
        
        return groups
    
    async def _form_diversity_based_groups(self,
                                         participants: List[Dict[str, Any]],
                                         criteria: Dict[str, Any],
                                         constraints: Optional[Dict[str, Any]]) -> List[LearningGroup]:
        """Form groups to maximize diversity"""
        target_group_size = constraints.get('group_size', self.config['default_group_size']) if constraints else self.config['default_group_size']
        
        # Calculate diversity metrics for each participant
        diversity_profiles = await self._calculate_diversity_profiles(participants)
        
        # Use round-robin assignment to maximize diversity
        groups = []
        num_groups = len(participants) // target_group_size
        
        if num_groups == 0:
            num_groups = 1
        
        # Initialize groups
        for i in range(num_groups):
            groups.append([])
        
        # Sort participants by diversity potential
        sorted_participants = sorted(
            participants,
            key=lambda p: self._calculate_diversity_potential(p, diversity_profiles),
            reverse=True
        )
        
        # Distribute participants round-robin style
        for i, participant in enumerate(sorted_participants):
            group_index = i % num_groups
            groups[group_index].append(participant)
        
        # Convert to LearningGroup objects
        learning_groups = []
        for i, group_members in enumerate(groups):
            if len(group_members) >= self.config['min_group_size']:
                group = await self._create_learning_group(
                    f"group_{i+1}",
                    group_members,
                    "diversity_based",
                    criteria.get('learning_objectives', [])
                )
                learning_groups.append(group)
        
        return learning_groups
    
    async def _form_skill_balanced_groups(self,
                                        participants: List[Dict[str, Any]],
                                        criteria: Dict[str, Any],
                                        constraints: Optional[Dict[str, Any]]) -> List[LearningGroup]:
        """Form groups with balanced skill distributions"""
        target_group_size = constraints.get('group_size', self.config['default_group_size']) if constraints else self.config['default_group_size']
        
        # Analyze skill distributions
        skill_analysis = await self._analyze_skill_distributions(participants)
        
        # Create balanced groups using skill-based assignment
        groups = []
        num_groups = len(participants) // target_group_size
        
        if num_groups == 0:
            num_groups = 1
        
        # Initialize groups with skill tracking
        group_skill_levels = [defaultdict(list) for _ in range(num_groups)]
        group_members = [[] for _ in range(num_groups)]
        
        # Sort participants by overall skill level
        sorted_participants = sorted(
            participants,
            key=lambda p: sum(p.get('skills', {}).values()),
            reverse=True
        )
        
        # Assign participants to balance skills
        for participant in sorted_participants:
            best_group_index = self._find_best_skill_balanced_group(
                participant, group_skill_levels, group_members, target_group_size
            )
            
            if best_group_index is not None:
                group_members[best_group_index].append(participant)
                
                # Update skill tracking
                participant_skills = participant.get('skills', {})
                for skill, level in participant_skills.items():
                    group_skill_levels[best_group_index][skill].append(level)
        
        # Convert to LearningGroup objects
        learning_groups = []
        for i, members in enumerate(group_members):
            if len(members) >= self.config['min_group_size']:
                group = await self._create_learning_group(
                    f"group_{i+1}",
                    members,
                    "skill_balanced",
                    criteria.get('learning_objectives', [])
                )
                learning_groups.append(group)
        
        return learning_groups
    
    async def _form_hybrid_optimized_groups(self,
                                          participants: List[Dict[str, Any]],
                                          criteria: Dict[str, Any],
                                          constraints: Optional[Dict[str, Any]]) -> List[LearningGroup]:
        """Form groups using hybrid optimization combining multiple factors"""
        target_group_size = constraints.get('group_size', self.config['default_group_size']) if constraints else self.config['default_group_size']
        
        # Use genetic algorithm approach for optimization
        best_grouping = await self._genetic_algorithm_grouping(
            participants, criteria, target_group_size
        )
        
        # Convert to LearningGroup objects
        learning_groups = []
        for i, group_members in enumerate(best_grouping):
            if len(group_members) >= self.config['min_group_size']:
                group = await self._create_learning_group(
                    f"group_{i+1}",
                    group_members,
                    "hybrid_optimized",
                    criteria.get('learning_objectives', [])
                )
                learning_groups.append(group)
        
        return learning_groups
    
    async def _form_random_groups(self,
                                participants: List[Dict[str, Any]],
                                criteria: Dict[str, Any],
                                constraints: Optional[Dict[str, Any]]) -> List[LearningGroup]:
        """Form random groups for baseline comparison"""
        target_group_size = constraints.get('group_size', self.config['default_group_size']) if constraints else self.config['default_group_size']
        
        # Shuffle participants
        shuffled_participants = participants.copy()
        random.shuffle(shuffled_participants)
        
        # Create groups
        groups = []
        for i in range(0, len(shuffled_participants), target_group_size):
            group_members = shuffled_participants[i:i + target_group_size]
            
            if len(group_members) >= self.config['min_group_size']:
                group = await self._create_learning_group(
                    f"group_{len(groups)+1}",
                    group_members,
                    "random",
                    criteria.get('learning_objectives', [])
                )
                groups.append(group)
        
        return groups
    
    async def _calculate_compatibility_matrix(self, participants: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate compatibility matrix for all participant pairs"""
        compatibility_matrix = {}
        
        for i, participant1 in enumerate(participants):
            for j, participant2 in enumerate(participants):
                if i != j:
                    user1_id = participant1.get('user_id', f'user_{i}')
                    user2_id = participant2.get('user_id', f'user_{j}')
                    
                    compatibility = await self._calculate_pairwise_compatibility(participant1, participant2)
                    compatibility_matrix[f"{user1_id}_{user2_id}"] = compatibility
        
        return compatibility_matrix
    
    async def _calculate_pairwise_compatibility(self,
                                             participant1: Dict[str, Any],
                                             participant2: Dict[str, Any]) -> float:
        """Calculate compatibility between two participants"""
        # Learning style compatibility
        style1 = participant1.get('learning_style', {})
        style2 = participant2.get('learning_style', {})
        
        style_compatibility = 1.0 - sum(
            abs(style1.get(dim, 0.5) - style2.get(dim, 0.5))
            for dim in ['visual', 'auditory', 'kinesthetic', 'reading_writing']
        ) / 4
        
        # Personality compatibility
        personality1 = participant1.get('personality', {})
        personality2 = participant2.get('personality', {})
        
        personality_compatibility = 1.0 - sum(
            abs(personality1.get(dim, 0.5) - personality2.get(dim, 0.5))
            for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        ) / 5
        
        # Schedule compatibility
        schedule1 = participant1.get('availability', {})
        schedule2 = participant2.get('availability', {})
        
        common_slots = sum(
            1 for slot in schedule1.keys()
            if schedule1.get(slot, False) and schedule2.get(slot, False)
        )
        total_slots = len(schedule1) if schedule1 else 24
        schedule_compatibility = common_slots / total_slots if total_slots > 0 else 0.5
        
        # Overall compatibility
        overall_compatibility = (
            style_compatibility * 0.4 +
            personality_compatibility * 0.4 +
            schedule_compatibility * 0.2
        )
        
        return overall_compatibility
    
    def _find_best_compatibility_pair(self,
                                    participants: List[Dict[str, Any]],
                                    compatibility_matrix: Dict[str, float]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Find the pair with highest compatibility"""
        best_compatibility = 0.0
        best_pair = None
        
        for i, participant1 in enumerate(participants):
            for j, participant2 in enumerate(participants):
                if i < j:
                    user1_id = participant1.get('user_id', f'user_{i}')
                    user2_id = participant2.get('user_id', f'user_{j}')
                    
                    compatibility_key = f"{user1_id}_{user2_id}"
                    compatibility = compatibility_matrix.get(compatibility_key, 0.0)
                    
                    if compatibility > best_compatibility:
                        best_compatibility = compatibility
                        best_pair = (participant1, participant2)
        
        return best_pair
    
    def _find_best_group_addition(self,
                                current_group: List[Dict[str, Any]],
                                candidates: List[Dict[str, Any]],
                                compatibility_matrix: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Find the best candidate to add to current group"""
        best_score = 0.0
        best_candidate = None
        
        for candidate in candidates:
            # Calculate average compatibility with current group members
            compatibility_scores = []
            
            for member in current_group:
                member_id = member.get('user_id', 'member')
                candidate_id = candidate.get('user_id', 'candidate')
                
                compatibility_key = f"{member_id}_{candidate_id}"
                reverse_key = f"{candidate_id}_{member_id}"
                
                compatibility = compatibility_matrix.get(compatibility_key, 
                                                       compatibility_matrix.get(reverse_key, 0.5))
                compatibility_scores.append(compatibility)
            
            avg_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0
            
            if avg_compatibility > best_score:
                best_score = avg_compatibility
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_group_compatibility(self,
                                     group_members: List[Dict[str, Any]],
                                     compatibility_matrix: Dict[str, float]) -> float:
        """Calculate overall compatibility for a group"""
        if len(group_members) < 2:
            return 1.0
        
        compatibility_scores = []
        
        for i, member1 in enumerate(group_members):
            for j, member2 in enumerate(group_members):
                if i < j:
                    member1_id = member1.get('user_id', f'member_{i}')
                    member2_id = member2.get('user_id', f'member_{j}')
                    
                    compatibility_key = f"{member1_id}_{member2_id}"
                    reverse_key = f"{member2_id}_{member1_id}"
                    
                    compatibility = compatibility_matrix.get(compatibility_key,
                                                           compatibility_matrix.get(reverse_key, 0.5))
                    compatibility_scores.append(compatibility)
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.5
    
    async def _distribute_remaining_participants(self,
                                               groups: List[LearningGroup],
                                               remaining: List[Dict[str, Any]],
                                               compatibility_matrix: Dict[str, float]):
        """Distribute remaining participants to existing groups"""
        for participant in remaining:
            best_group = None
            best_compatibility = 0.0
            
            for group in groups:
                if len(group.members) < self.config['max_group_size']:
                    # Calculate compatibility with group
                    compatibility_scores = []
                    
                    for member in group.members:
                        participant_id = participant.get('user_id', 'participant')
                        member_id = member.get('user_id', 'member')
                        
                        compatibility_key = f"{participant_id}_{member_id}"
                        reverse_key = f"{member_id}_{participant_id}"
                        
                        compatibility = compatibility_matrix.get(compatibility_key,
                                                               compatibility_matrix.get(reverse_key, 0.5))
                        compatibility_scores.append(compatibility)
                    
                    avg_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0
                    
                    if avg_compatibility > best_compatibility:
                        best_compatibility = avg_compatibility
                        best_group = group
            
            if best_group:
                best_group.members.append(participant)
                best_group.group_size += 1
    
    async def _calculate_diversity_profiles(self, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diversity profiles for participants"""
        diversity_dimensions = {
            'background_field': defaultdict(int),
            'experience_level': defaultdict(int),
            'learning_style_primary': defaultdict(int),
            'personality_type': defaultdict(int)
        }
        
        for participant in participants:
            # Background field
            field = participant.get('background', {}).get('field', 'unknown')
            diversity_dimensions['background_field'][field] += 1
            
            # Experience level
            experience = participant.get('background', {}).get('experience_level', 'beginner')
            diversity_dimensions['experience_level'][experience] += 1
            
            # Primary learning style
            learning_style = participant.get('learning_style', {})
            primary_style = max(learning_style.keys(), key=lambda k: learning_style[k]) if learning_style else 'unknown'
            diversity_dimensions['learning_style_primary'][primary_style] += 1
            
            # Personality type (simplified)
            personality = participant.get('personality', {})
            if personality:
                extraversion = personality.get('extraversion', 0.5)
                personality_type = 'extraverted' if extraversion > 0.6 else 'introverted'
                diversity_dimensions['personality_type'][personality_type] += 1
        
        return diversity_dimensions
    
    def _calculate_diversity_potential(self,
                                     participant: Dict[str, Any],
                                     diversity_profiles: Dict[str, Any]) -> float:
        """Calculate diversity potential for a participant"""
        diversity_score = 0.0
        
        # Background field diversity
        field = participant.get('background', {}).get('field', 'unknown')
        field_rarity = 1.0 / (diversity_profiles['background_field'][field] + 1)
        diversity_score += field_rarity
        
        # Experience level diversity
        experience = participant.get('background', {}).get('experience_level', 'beginner')
        experience_rarity = 1.0 / (diversity_profiles['experience_level'][experience] + 1)
        diversity_score += experience_rarity
        
        # Learning style diversity
        learning_style = participant.get('learning_style', {})
        primary_style = max(learning_style.keys(), key=lambda k: learning_style[k]) if learning_style else 'unknown'
        style_rarity = 1.0 / (diversity_profiles['learning_style_primary'][primary_style] + 1)
        diversity_score += style_rarity
        
        return diversity_score / 3  # Average across dimensions
    
    async def _analyze_skill_distributions(self, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze skill distributions across participants"""
        all_skills = set()
        skill_levels = defaultdict(list)
        
        for participant in participants:
            skills = participant.get('skills', {})
            all_skills.update(skills.keys())
            
            for skill, level in skills.items():
                skill_levels[skill].append(level)
        
        skill_stats = {}
        for skill in all_skills:
            levels = skill_levels[skill]
            if levels:
                skill_stats[skill] = {
                    'mean': sum(levels) / len(levels),
                    'min': min(levels),
                    'max': max(levels),
                    'count': len(levels)
                }
        
        return {
            'all_skills': list(all_skills),
            'skill_statistics': skill_stats,
            'total_participants': len(participants)
        }
    
    def _find_best_skill_balanced_group(self,
                                      participant: Dict[str, Any],
                                      group_skill_levels: List[Dict[str, List[float]]],
                                      group_members: List[List[Dict[str, Any]]],
                                      target_group_size: int) -> Optional[int]:
        """Find the best group for skill balance"""
        participant_skills = participant.get('skills', {})
        best_group_index = None
        best_balance_score = float('-inf')
        
        for i, (skill_levels, members) in enumerate(zip(group_skill_levels, group_members)):
            if len(members) >= target_group_size:
                continue  # Group is full
            
            # Calculate balance score if participant is added
            balance_score = 0.0
            
            for skill, level in participant_skills.items():
                current_levels = skill_levels[skill]
                
                if not current_levels:
                    # New skill adds diversity
                    balance_score += 1.0
                else:
                    # Calculate how this addition affects balance
                    current_mean = sum(current_levels) / len(current_levels)
                    new_levels = current_levels + [level]
                    new_mean = sum(new_levels) / len(new_levels)
                    
                    # Prefer additions that bring levels closer to overall mean
                    balance_improvement = abs(0.5 - current_mean) - abs(0.5 - new_mean)
                    balance_score += balance_improvement
            
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                best_group_index = i
        
        return best_group_index
    
    async def _genetic_algorithm_grouping(self,
                                        participants: List[Dict[str, Any]],
                                        criteria: Dict[str, Any],
                                        target_group_size: int) -> List[List[Dict[str, Any]]]:
        """Use genetic algorithm for optimal grouping"""
        # Simplified genetic algorithm implementation
        population_size = min(50, len(participants) * 2)
        generations = 20
        
        # Initialize population
        population = []
        for _ in range(population_size):
            grouping = self._create_random_grouping(participants, target_group_size)
            population.append(grouping)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for grouping in population:
                fitness = await self._evaluate_grouping_fitness(grouping, criteria)
                fitness_scores.append(fitness)
            
            # Selection and reproduction
            new_population = []
            
            # Keep best solutions (elitism)
            best_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            for i in best_indices[:population_size // 4]:
                new_population.append(population[i])
            
            # Generate new solutions
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover and mutation
                child = self._crossover_groupings(parent1, parent2, participants)
                child = self._mutate_grouping(child, participants, target_group_size)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        final_fitness_scores = []
        for grouping in population:
            fitness = await self._evaluate_grouping_fitness(grouping, criteria)
            final_fitness_scores.append(fitness)
        
        best_index = max(range(len(final_fitness_scores)), key=lambda i: final_fitness_scores[i])
        return population[best_index]
    
    def _create_random_grouping(self,
                              participants: List[Dict[str, Any]],
                              target_group_size: int) -> List[List[Dict[str, Any]]]:
        """Create a random grouping of participants"""
        shuffled = participants.copy()
        random.shuffle(shuffled)
        
        groups = []
        for i in range(0, len(shuffled), target_group_size):
            group = shuffled[i:i + target_group_size]
            if len(group) >= self.config['min_group_size']:
                groups.append(group)
        
        return groups
    
    async def _evaluate_grouping_fitness(self,
                                       grouping: List[List[Dict[str, Any]]],
                                       criteria: Dict[str, Any]) -> float:
        """Evaluate fitness of a grouping solution"""
        if not grouping:
            return 0.0
        
        total_fitness = 0.0
        
        for group in grouping:
            if len(group) < self.config['min_group_size']:
                continue
            
            # Compatibility fitness
            compatibility_matrix = await self._calculate_compatibility_matrix(group)
            compatibility_score = self._calculate_group_compatibility(group, compatibility_matrix)
            
            # Diversity fitness
            diversity_score = self._calculate_group_diversity(group)
            
            # Skill balance fitness
            skill_balance_score = self._calculate_skill_balance(group)
            
            # Combined fitness
            group_fitness = (
                compatibility_score * self.config['compatibility_weight'] +
                diversity_score * self.config['diversity_weight'] +
                skill_balance_score * self.config['skill_balance_weight']
            )
            
            total_fitness += group_fitness
        
        return total_fitness / len(grouping) if grouping else 0.0
    
    def _calculate_group_diversity(self, group: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a group"""
        if len(group) < 2:
            return 0.0
        
        diversity_factors = []
        
        # Background diversity
        backgrounds = [member.get('background', {}).get('field', 'unknown') for member in group]
        unique_backgrounds = len(set(backgrounds))
        background_diversity = unique_backgrounds / len(group)
        diversity_factors.append(background_diversity)
        
        # Experience diversity
        experiences = [member.get('background', {}).get('experience_level', 'beginner') for member in group]
        unique_experiences = len(set(experiences))
        experience_diversity = unique_experiences / len(group)
        diversity_factors.append(experience_diversity)
        
        # Learning style diversity
        primary_styles = []
        for member in group:
            learning_style = member.get('learning_style', {})
            if learning_style:
                primary_style = max(learning_style.keys(), key=lambda k: learning_style[k])
                primary_styles.append(primary_style)
        
        unique_styles = len(set(primary_styles))
        style_diversity = unique_styles / len(group) if primary_styles else 0.0
        diversity_factors.append(style_diversity)
        
        return sum(diversity_factors) / len(diversity_factors)
    
    def _calculate_skill_balance(self, group: List[Dict[str, Any]]) -> float:
        """Calculate skill balance score for a group"""
        if len(group) < 2:
            return 1.0
        
        all_skills = set()
        for member in group:
            all_skills.update(member.get('skills', {}).keys())
        
        if not all_skills:
            return 0.5
        
        balance_scores = []
        
        for skill in all_skills:
            skill_levels = [member.get('skills', {}).get(skill, 0.0) for member in group]
            
            # Calculate variance (lower variance = better balance)
            mean_level = sum(skill_levels) / len(skill_levels)
            variance = sum((level - mean_level) ** 2 for level in skill_levels) / len(skill_levels)
            
            # Convert variance to balance score (0 variance = 1.0 balance)
            balance_score = 1.0 / (1.0 + variance)
            balance_scores.append(balance_score)
        
        return sum(balance_scores) / len(balance_scores)
    
    def _tournament_selection(self,
                            population: List[List[List[Dict[str, Any]]]],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List[List[Dict[str, Any]]]:
        """Tournament selection for genetic algorithm"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index]
    
    def _crossover_groupings(self,
                           parent1: List[List[Dict[str, Any]]],
                           parent2: List[List[Dict[str, Any]]],
                           all_participants: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Crossover operation for groupings"""
        # Simple crossover: take some groups from parent1, some from parent2
        child_groups = []
        used_participants = set()
        
        # Add groups from parent1
        for group in parent1[:len(parent1)//2]:
            group_participant_ids = {p.get('user_id', id(p)) for p in group}
            if not (group_participant_ids & used_participants):
                child_groups.append(group)
                used_participants.update(group_participant_ids)
        
        # Add non-conflicting groups from parent2
        for group in parent2:
            group_participant_ids = {p.get('user_id', id(p)) for p in group}
            if not (group_participant_ids & used_participants):
                child_groups.append(group)
                used_participants.update(group_participant_ids)
        
        # Handle remaining participants
        remaining_participants = [
            p for p in all_participants
            if p.get('user_id', id(p)) not in used_participants
        ]
        
        if remaining_participants:
            child_groups.append(remaining_participants)
        
        return child_groups
    
    def _mutate_grouping(self,
                       grouping: List[List[Dict[str, Any]]],
                       all_participants: List[Dict[str, Any]],
                       target_group_size: int,
                       mutation_rate: float = 0.1) -> List[List[Dict[str, Any]]]:
        """Mutation operation for groupings"""
        if random.random() > mutation_rate:
            return grouping
        
        # Simple mutation: swap two random participants between groups
        if len(grouping) >= 2:
            group1_idx = random.randint(0, len(grouping) - 1)
            group2_idx = random.randint(0, len(grouping) - 1)
            
            if group1_idx != group2_idx and grouping[group1_idx] and grouping[group2_idx]:
                participant1_idx = random.randint(0, len(grouping[group1_idx]) - 1)
                participant2_idx = random.randint(0, len(grouping[group2_idx]) - 1)
                
                # Swap participants
                grouping[group1_idx][participant1_idx], grouping[group2_idx][participant2_idx] = \
                    grouping[group2_idx][participant2_idx], grouping[group1_idx][participant1_idx]
        
        return grouping
    
    async def _create_learning_group(self,
                                   group_id: str,
                                   members: List[Dict[str, Any]],
                                   formation_strategy: str,
                                   learning_objectives: List[str]) -> LearningGroup:
        """Create a LearningGroup object"""
        # Calculate group metrics
        compatibility_matrix = await self._calculate_compatibility_matrix(members)
        compatibility_score = self._calculate_group_compatibility(members, compatibility_matrix)
        diversity_score = self._calculate_group_diversity(members)
        
        return LearningGroup(
            group_id=group_id,
            group_name=f"Learning Group {group_id.split('_')[-1]}",
            members=members,
            group_size=len(members),
            formation_strategy=formation_strategy,
            compatibility_score=compatibility_score,
            diversity_score=diversity_score,
            learning_objectives=learning_objectives,
            group_dynamics={},
            performance_metrics={},
            formation_timestamp=datetime.utcnow().isoformat(),
            status="active"
        )
    
    async def _analyze_formation_quality(self,
                                       groups: List[LearningGroup],
                                       participants: List[Dict[str, Any]],
                                       criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of group formation"""
        if not groups:
            return {'overall_quality': 0.0}
        
        # Calculate metrics
        avg_compatibility = sum(group.compatibility_score for group in groups) / len(groups)
        avg_diversity = sum(group.diversity_score for group in groups) / len(groups)
        
        # Group size distribution
        group_sizes = [group.group_size for group in groups]
        avg_group_size = sum(group_sizes) / len(group_sizes)
        size_variance = sum((size - avg_group_size) ** 2 for size in group_sizes) / len(group_sizes)
        
        # Coverage (percentage of participants placed)
        total_placed = sum(group.group_size for group in groups)
        coverage = total_placed / len(participants) if participants else 0.0
        
        # Overall quality score
        overall_quality = (
            avg_compatibility * 0.4 +
            avg_diversity * 0.3 +
            coverage * 0.2 +
            (1.0 / (1.0 + size_variance)) * 0.1
        )
        
        return {
            'overall_quality': overall_quality,
            'average_compatibility': avg_compatibility,
            'average_diversity': avg_diversity,
            'coverage': coverage,
            'average_group_size': avg_group_size,
            'size_variance': size_variance,
            'total_groups': len(groups),
            'total_participants_placed': total_placed
        }
    
    async def _generate_formation_insights(self,
                                         groups: List[LearningGroup],
                                         formation_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about group formation"""
        insights = []
        
        overall_quality = formation_analysis.get('overall_quality', 0.0)
        
        if overall_quality > 0.8:
            insights.append("Excellent group formation quality achieved")
        elif overall_quality > 0.6:
            insights.append("Good group formation with room for optimization")
        else:
            insights.append("Group formation could be improved")
        
        # Compatibility insights
        avg_compatibility = formation_analysis.get('average_compatibility', 0.0)
        if avg_compatibility > 0.8:
            insights.append("High compatibility across all groups")
        elif avg_compatibility < 0.5:
            insights.append("Low compatibility detected - consider alternative formation strategy")
        
        # Diversity insights
        avg_diversity = formation_analysis.get('average_diversity', 0.0)
        if avg_diversity > 0.7:
            insights.append("Good diversity achieved across groups")
        elif avg_diversity < 0.3:
            insights.append("Limited diversity - consider diversity-focused formation")
        
        # Coverage insights
        coverage = formation_analysis.get('coverage', 0.0)
        if coverage < 1.0:
            insights.append(f"Some participants not placed in groups ({coverage:.1%} coverage)")
        
        # Group size insights
        avg_size = formation_analysis.get('average_group_size', 0.0)
        if avg_size < 3:
            insights.append("Small group sizes promote intensive collaboration")
        elif avg_size > 5:
            insights.append("Large groups may benefit from sub-group activities")
        
        return insights
