"""
Collective Intelligence Services

Extracted from quantum_intelligence_engine.py (lines 10289-12523) - advanced collective
intelligence harvesting, wisdom aggregation, group problem solving, and knowledge graph building.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import math

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


@dataclass
class CollectiveWisdom:
    """Collective wisdom aggregation results"""
    topic: str = ""
    aggregated_knowledge: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    contributor_count: int = 0
    knowledge_sources: List[str] = field(default_factory=list)
    consensus_level: float = 0.0
    diversity_index: float = 0.0
    wisdom_quality_score: float = 0.0
    emergent_insights: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)


@dataclass
class GroupProblemSolution:
    """Group problem solving results"""
    problem_id: str = ""
    problem_description: str = ""
    solution_approaches: List[Dict[str, Any]] = field(default_factory=list)
    best_solution: Dict[str, Any] = field(default_factory=dict)
    solution_confidence: float = 0.0
    collaboration_quality: float = 0.0
    innovation_score: float = 0.0
    problem_complexity: float = 0.0
    solving_process: List[Dict[str, Any]] = field(default_factory=list)
    group_dynamics_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class KnowledgeNode:
    """Knowledge graph node"""
    node_id: str = ""
    concept: str = ""
    knowledge_type: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Collective knowledge graph"""
    graph_id: str = ""
    domain: str = ""
    nodes: Dict[str, KnowledgeNode] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    graph_metrics: Dict[str, float] = field(default_factory=dict)
    creation_timestamp: str = ""
    last_updated: str = ""


class CollectiveIntelligenceHarvester:
    """
    🧠 COLLECTIVE INTELLIGENCE HARVESTER

    Advanced system for harvesting and aggregating collective intelligence from groups.
    Extracted from the original quantum engine's collaborative intelligence logic.
    """

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service

        # Harvesting configuration
        self.config = {
            'min_contributors': 3,
            'confidence_threshold': 0.7,
            'consensus_threshold': 0.6,
            'diversity_weight': 0.3,
            'quality_weight': 0.4,
            'consensus_weight': 0.3
        }

        # Intelligence tracking
        self.harvesting_history = []
        self.collective_knowledge_base = {}

        logger.info("Collective Intelligence Harvester initialized")

    async def harvest_collective_intelligence(self,
                                            topic: str,
                                            group_contributions: List[Dict[str, Any]],
                                            harvesting_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Harvest collective intelligence from group contributions

        Args:
            topic: Topic or domain for intelligence harvesting
            group_contributions: List of individual contributions from group members
            harvesting_context: Optional context for harvesting process

        Returns:
            Dict with harvested collective intelligence
        """
        try:
            if len(group_contributions) < self.config['min_contributors']:
                return {
                    'status': 'error',
                    'error': f'Insufficient contributors (minimum {self.config["min_contributors"]} required)'
                }

            # Analyze individual contributions
            contribution_analysis = await self._analyze_contributions(group_contributions)

            # Aggregate knowledge
            aggregated_knowledge = await self._aggregate_knowledge(group_contributions, contribution_analysis)

            # Calculate consensus and diversity
            consensus_analysis = await self._analyze_consensus(group_contributions, aggregated_knowledge)
            diversity_analysis = await self._analyze_diversity(group_contributions)

            # Generate emergent insights
            emergent_insights = await self._generate_emergent_insights(
                aggregated_knowledge, contribution_analysis, consensus_analysis
            )

            # Identify knowledge gaps
            knowledge_gaps = await self._identify_knowledge_gaps(
                aggregated_knowledge, group_contributions
            )

            # Calculate wisdom quality
            wisdom_quality = await self._calculate_wisdom_quality(
                aggregated_knowledge, consensus_analysis, diversity_analysis, emergent_insights
            )

            # Create collective wisdom result
            collective_wisdom = CollectiveWisdom(
                topic=topic,
                aggregated_knowledge=aggregated_knowledge,
                confidence_score=wisdom_quality.get('confidence', 0.0),
                contributor_count=len(group_contributions),
                knowledge_sources=[contrib.get('contributor_id', 'unknown') for contrib in group_contributions],
                consensus_level=consensus_analysis.get('consensus_score', 0.0),
                diversity_index=diversity_analysis.get('diversity_score', 0.0),
                wisdom_quality_score=wisdom_quality.get('overall_quality', 0.0),
                emergent_insights=emergent_insights,
                knowledge_gaps=knowledge_gaps
            )

            # Store in collective knowledge base
            self.collective_knowledge_base[topic] = collective_wisdom

            # Track harvesting history
            self.harvesting_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'topic': topic,
                'contributor_count': len(group_contributions),
                'wisdom_quality': wisdom_quality.get('overall_quality', 0.0)
            })

            return {
                'status': 'success',
                'collective_wisdom': collective_wisdom.__dict__,
                'harvesting_analysis': {
                    'contribution_analysis': contribution_analysis,
                    'consensus_analysis': consensus_analysis,
                    'diversity_analysis': diversity_analysis,
                    'wisdom_quality': wisdom_quality
                },
                'harvesting_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error harvesting collective intelligence: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _analyze_contributions(self, contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual contributions for quality and relevance"""
        analysis = {
            'contribution_scores': [],
            'quality_distribution': {},
            'relevance_scores': [],
            'expertise_indicators': [],
            'contribution_types': defaultdict(int)
        }

        for contrib in contributions:
            # Analyze contribution quality
            quality_score = await self._assess_contribution_quality(contrib)
            analysis['contribution_scores'].append(quality_score)

            # Analyze relevance
            relevance_score = await self._assess_contribution_relevance(contrib)
            analysis['relevance_scores'].append(relevance_score)

            # Identify expertise indicators
            expertise = await self._identify_expertise_indicators(contrib)
            analysis['expertise_indicators'].append(expertise)

            # Categorize contribution type
            contrib_type = contrib.get('contribution_type', 'knowledge_sharing')
            analysis['contribution_types'][contrib_type] += 1

        # Calculate distribution statistics
        if analysis['contribution_scores']:
            analysis['quality_distribution'] = {
                'mean': sum(analysis['contribution_scores']) / len(analysis['contribution_scores']),
                'min': min(analysis['contribution_scores']),
                'max': max(analysis['contribution_scores']),
                'std': self._calculate_std(analysis['contribution_scores'])
            }

        return analysis

    async def _assess_contribution_quality(self, contribution: Dict[str, Any]) -> float:
        """Assess the quality of an individual contribution"""
        quality_factors = []

        # Content depth
        content = contribution.get('content', '')
        content_depth = min(1.0, len(content.split()) / 100)  # Normalize by word count
        quality_factors.append(content_depth)

        # Evidence provided
        evidence_count = len(contribution.get('evidence', []))
        evidence_score = min(1.0, evidence_count / 3)  # Up to 3 pieces of evidence
        quality_factors.append(evidence_score)

        # Clarity and structure
        clarity_score = contribution.get('clarity_score', 0.7)  # Would be assessed by NLP
        quality_factors.append(clarity_score)

        # Originality
        originality_score = contribution.get('originality_score', 0.6)  # Would be assessed by comparison
        quality_factors.append(originality_score)

        return sum(quality_factors) / len(quality_factors)

    async def _assess_contribution_relevance(self, contribution: Dict[str, Any]) -> float:
        """Assess relevance of contribution to the topic"""
        # Mock relevance assessment - would use NLP/semantic analysis in production
        topic_keywords = contribution.get('topic_keywords', [])
        content = contribution.get('content', '')

        # Simple keyword matching
        keyword_matches = sum(1 for keyword in topic_keywords if keyword.lower() in content.lower())
        relevance_score = min(1.0, keyword_matches / max(1, len(topic_keywords)))

        # Adjust based on contribution metadata
        if contribution.get('directly_addresses_topic', False):
            relevance_score = min(1.0, relevance_score + 0.3)

        return relevance_score

    async def _identify_expertise_indicators(self, contribution: Dict[str, Any]) -> Dict[str, Any]:
        """Identify indicators of expertise in contribution"""
        return {
            'domain_experience': contribution.get('contributor_experience_years', 0),
            'credentials': contribution.get('contributor_credentials', []),
            'citation_quality': len(contribution.get('references', [])),
            'technical_depth': contribution.get('technical_depth_score', 0.5),
            'practical_experience': contribution.get('practical_examples_count', 0)
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    async def _aggregate_knowledge(self,
                                 contributions: List[Dict[str, Any]],
                                 contribution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate knowledge from multiple contributions"""
        aggregated = {
            'core_concepts': {},
            'key_insights': [],
            'best_practices': [],
            'common_patterns': [],
            'conflicting_views': [],
            'supporting_evidence': []
        }

        # Weight contributions by quality
        quality_scores = contribution_analysis['contribution_scores']

        for i, contrib in enumerate(contributions):
            weight = quality_scores[i] if i < len(quality_scores) else 0.5

            # Aggregate core concepts
            concepts = contrib.get('concepts', {})
            for concept, value in concepts.items():
                if concept not in aggregated['core_concepts']:
                    aggregated['core_concepts'][concept] = {'values': [], 'weights': []}

                aggregated['core_concepts'][concept]['values'].append(value)
                aggregated['core_concepts'][concept]['weights'].append(weight)

            # Collect insights
            insights = contrib.get('insights', [])
            for insight in insights:
                aggregated['key_insights'].append({
                    'insight': insight,
                    'weight': weight,
                    'contributor': contrib.get('contributor_id', 'unknown')
                })

            # Collect best practices
            practices = contrib.get('best_practices', [])
            for practice in practices:
                aggregated['best_practices'].append({
                    'practice': practice,
                    'weight': weight,
                    'contributor': contrib.get('contributor_id', 'unknown')
                })

            # Collect evidence
            evidence = contrib.get('evidence', [])
            for evidence_item in evidence:
                aggregated['supporting_evidence'].append({
                    'evidence': evidence_item,
                    'weight': weight,
                    'contributor': contrib.get('contributor_id', 'unknown')
                })

        # Calculate weighted averages for core concepts
        for concept, data in aggregated['core_concepts'].items():
            values = data['values']
            weights = data['weights']

            if values and weights:
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                aggregated['core_concepts'][concept] = {
                    'weighted_average': weighted_sum / weight_sum if weight_sum > 0 else 0,
                    'confidence': weight_sum / len(contributions),
                    'agreement_level': 1.0 - (self._calculate_std(values) / max(values) if max(values) > 0 else 0)
                }

        return aggregated

    async def _analyze_consensus(self,
                               contributions: List[Dict[str, Any]],
                               aggregated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus level across contributions"""
        consensus_analysis = {
            'consensus_score': 0.0,
            'agreement_areas': [],
            'disagreement_areas': [],
            'consensus_distribution': {}
        }

        # Analyze concept consensus
        core_concepts = aggregated_knowledge.get('core_concepts', {})
        agreement_scores = []

        for concept, data in core_concepts.items():
            agreement_level = data.get('agreement_level', 0.0)
            agreement_scores.append(agreement_level)

            if agreement_level > 0.8:
                consensus_analysis['agreement_areas'].append(concept)
            elif agreement_level < 0.4:
                consensus_analysis['disagreement_areas'].append(concept)

        # Calculate overall consensus
        if agreement_scores:
            consensus_analysis['consensus_score'] = sum(agreement_scores) / len(agreement_scores)

        # Analyze insight consensus
        insights = aggregated_knowledge.get('key_insights', [])
        insight_consensus = await self._analyze_insight_consensus(insights)
        consensus_analysis['insight_consensus'] = insight_consensus

        return consensus_analysis

    async def _analyze_insight_consensus(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus among insights"""
        # Group similar insights
        insight_groups = defaultdict(list)

        for insight in insights:
            insight_text = insight.get('insight', '')
            # Simple grouping by first few words (would use semantic similarity in production)
            key = ' '.join(insight_text.split()[:3]).lower()
            insight_groups[key].append(insight)

        # Calculate consensus for each group
        group_consensus = {}
        for group_key, group_insights in insight_groups.items():
            if len(group_insights) > 1:
                weights = [insight.get('weight', 0.5) for insight in group_insights]
                avg_weight = sum(weights) / len(weights)
                group_consensus[group_key] = {
                    'support_count': len(group_insights),
                    'average_weight': avg_weight,
                    'consensus_strength': avg_weight * len(group_insights)
                }

        return group_consensus

    async def _analyze_diversity(self, contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversity of perspectives in contributions"""
        diversity_analysis = {
            'diversity_score': 0.0,
            'perspective_types': [],
            'background_diversity': 0.0,
            'approach_diversity': 0.0,
            'experience_diversity': 0.0
        }

        # Analyze background diversity
        backgrounds = [contrib.get('contributor_background', 'unknown') for contrib in contributions]
        unique_backgrounds = len(set(backgrounds))
        diversity_analysis['background_diversity'] = unique_backgrounds / len(contributions) if contributions else 0

        # Analyze approach diversity
        approaches = [contrib.get('approach_type', 'general') for contrib in contributions]
        unique_approaches = len(set(approaches))
        diversity_analysis['approach_diversity'] = unique_approaches / len(contributions) if contributions else 0

        # Analyze experience diversity
        experience_levels = [contrib.get('contributor_experience_years', 0) for contrib in contributions]
        if experience_levels:
            exp_std = self._calculate_std(experience_levels)
            max_exp = max(experience_levels)
            diversity_analysis['experience_diversity'] = exp_std / max_exp if max_exp > 0 else 0

        # Calculate overall diversity score
        diversity_analysis['diversity_score'] = (
            diversity_analysis['background_diversity'] * 0.4 +
            diversity_analysis['approach_diversity'] * 0.4 +
            diversity_analysis['experience_diversity'] * 0.2
        )

        return diversity_analysis

    async def _generate_emergent_insights(self,
                                        aggregated_knowledge: Dict[str, Any],
                                        contribution_analysis: Dict[str, Any],
                                        consensus_analysis: Dict[str, Any]) -> List[str]:
        """Generate emergent insights from collective intelligence"""
        emergent_insights = []

        # Insights from high consensus areas
        agreement_areas = consensus_analysis.get('agreement_areas', [])
        if len(agreement_areas) > 2:
            emergent_insights.append(f"Strong consensus achieved on {len(agreement_areas)} key concepts")

        # Insights from diversity
        quality_dist = contribution_analysis.get('quality_distribution', {})
        if quality_dist.get('std', 0) < 0.2:
            emergent_insights.append("Consistent high-quality contributions across all participants")

        # Insights from knowledge patterns
        core_concepts = aggregated_knowledge.get('core_concepts', {})
        high_confidence_concepts = [
            concept for concept, data in core_concepts.items()
            if data.get('confidence', 0) > 0.8
        ]

        if len(high_confidence_concepts) > 3:
            emergent_insights.append(f"High confidence established in {len(high_confidence_concepts)} core areas")

        # Insights from evidence
        evidence_count = len(aggregated_knowledge.get('supporting_evidence', []))
        if evidence_count > len(contribution_analysis.get('contribution_scores', [])) * 2:
            emergent_insights.append("Rich evidence base supports collective conclusions")

        return emergent_insights

    async def _identify_knowledge_gaps(self,
                                     aggregated_knowledge: Dict[str, Any],
                                     contributions: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps in collective knowledge"""
        knowledge_gaps = []

        # Gaps from low consensus areas
        core_concepts = aggregated_knowledge.get('core_concepts', {})
        low_agreement_concepts = [
            concept for concept, data in core_concepts.items()
            if data.get('agreement_level', 1.0) < 0.4
        ]

        if low_agreement_concepts:
            knowledge_gaps.extend([f"Unclear consensus on {concept}" for concept in low_agreement_concepts])

        # Gaps from missing evidence
        concepts_without_evidence = []
        for concept in core_concepts.keys():
            # Check if concept has supporting evidence
            evidence_items = aggregated_knowledge.get('supporting_evidence', [])
            concept_evidence = [e for e in evidence_items if concept.lower() in str(e).lower()]

            if not concept_evidence:
                concepts_without_evidence.append(concept)

        if concepts_without_evidence:
            knowledge_gaps.extend([f"Limited evidence for {concept}" for concept in concepts_without_evidence])

        # Gaps from contribution analysis
        contrib_types = set()
        for contrib in contributions:
            contrib_types.add(contrib.get('contribution_type', 'general'))

        expected_types = {'theoretical', 'practical', 'empirical', 'experiential'}
        missing_types = expected_types - contrib_types

        if missing_types:
            knowledge_gaps.extend([f"Missing {contrib_type} perspectives" for contrib_type in missing_types])

        return knowledge_gaps

    async def _calculate_wisdom_quality(self,
                                      aggregated_knowledge: Dict[str, Any],
                                      consensus_analysis: Dict[str, Any],
                                      diversity_analysis: Dict[str, Any],
                                      emergent_insights: List[str]) -> Dict[str, Any]:
        """Calculate overall wisdom quality score"""
        # Quality factors
        consensus_score = consensus_analysis.get('consensus_score', 0.0)
        diversity_score = diversity_analysis.get('diversity_score', 0.0)

        # Evidence quality
        evidence_count = len(aggregated_knowledge.get('supporting_evidence', []))
        evidence_quality = min(1.0, evidence_count / 10)  # Normalize to max 10 pieces

        # Insight emergence
        insight_quality = min(1.0, len(emergent_insights) / 5)  # Normalize to max 5 insights

        # Concept coverage
        concept_count = len(aggregated_knowledge.get('core_concepts', {}))
        concept_coverage = min(1.0, concept_count / 8)  # Normalize to max 8 concepts

        # Calculate weighted quality score
        overall_quality = (
            consensus_score * self.config['consensus_weight'] +
            diversity_score * self.config['diversity_weight'] +
            evidence_quality * 0.2 +
            insight_quality * 0.1 +
            concept_coverage * 0.1
        )

        # Confidence calculation
        confidence = min(1.0, (consensus_score + evidence_quality) / 2)

        return {
            'overall_quality': overall_quality,
            'confidence': confidence,
            'consensus_contribution': consensus_score * self.config['consensus_weight'],
            'diversity_contribution': diversity_score * self.config['diversity_weight'],
            'evidence_quality': evidence_quality,
            'insight_quality': insight_quality,
            'concept_coverage': concept_coverage
        }


class WisdomAggregationEngine:
    """
    🧩 WISDOM AGGREGATION ENGINE

    Advanced engine for aggregating wisdom from multiple sources and perspectives.
    """

    def __init__(self, harvester: CollectiveIntelligenceHarvester):
        self.harvester = harvester

        # Aggregation configuration
        self.config = {
            'aggregation_methods': ['weighted_average', 'consensus_based', 'expertise_weighted'],
            'quality_threshold': 0.6,
            'temporal_decay_factor': 0.95,
            'source_credibility_weight': 0.3
        }

        # Wisdom tracking
        self.aggregated_wisdom = {}
        self.aggregation_history = []

        logger.info("Wisdom Aggregation Engine initialized")

    async def aggregate_wisdom_across_topics(self,
                                           topics: List[str],
                                           aggregation_method: str = "weighted_average") -> Dict[str, Any]:
        """
        Aggregate wisdom across multiple related topics

        Args:
            topics: List of topics to aggregate wisdom from
            aggregation_method: Method for aggregation

        Returns:
            Dict with cross-topic aggregated wisdom
        """
        try:
            # Collect wisdom from all topics
            topic_wisdom = {}
            for topic in topics:
                if topic in self.harvester.collective_knowledge_base:
                    topic_wisdom[topic] = self.harvester.collective_knowledge_base[topic]

            if not topic_wisdom:
                return {'status': 'error', 'error': 'No wisdom found for specified topics'}

            # Apply aggregation method
            if aggregation_method == "weighted_average":
                aggregated = await self._weighted_average_aggregation(topic_wisdom)
            elif aggregation_method == "consensus_based":
                aggregated = await self._consensus_based_aggregation(topic_wisdom)
            elif aggregation_method == "expertise_weighted":
                aggregated = await self._expertise_weighted_aggregation(topic_wisdom)
            else:
                aggregated = await self._weighted_average_aggregation(topic_wisdom)

            # Generate cross-topic insights
            cross_topic_insights = await self._generate_cross_topic_insights(topic_wisdom, aggregated)

            return {
                'status': 'success',
                'aggregated_wisdom': aggregated,
                'cross_topic_insights': cross_topic_insights,
                'topics_included': topics,
                'aggregation_method': aggregation_method,
                'aggregation_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error aggregating wisdom across topics: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _weighted_average_aggregation(self, topic_wisdom: Dict[str, CollectiveWisdom]) -> Dict[str, Any]:
        """Aggregate using weighted averages"""
        aggregated = {
            'core_concepts': {},
            'key_insights': [],
            'confidence_score': 0.0,
            'consensus_level': 0.0,
            'diversity_index': 0.0
        }

        total_weight = 0
        weighted_confidence = 0
        weighted_consensus = 0
        weighted_diversity = 0

        for topic, wisdom in topic_wisdom.items():
            weight = wisdom.wisdom_quality_score
            total_weight += weight

            # Aggregate confidence, consensus, diversity
            weighted_confidence += wisdom.confidence_score * weight
            weighted_consensus += wisdom.consensus_level * weight
            weighted_diversity += wisdom.diversity_index * weight

            # Aggregate concepts
            for concept, value in wisdom.aggregated_knowledge.get('core_concepts', {}).items():
                if concept not in aggregated['core_concepts']:
                    aggregated['core_concepts'][concept] = {'values': [], 'weights': []}

                aggregated['core_concepts'][concept]['values'].append(value)
                aggregated['core_concepts'][concept]['weights'].append(weight)

            # Aggregate insights
            for insight in wisdom.emergent_insights:
                aggregated['key_insights'].append({
                    'insight': insight,
                    'source_topic': topic,
                    'weight': weight
                })

        # Calculate final aggregated values
        if total_weight > 0:
            aggregated['confidence_score'] = weighted_confidence / total_weight
            aggregated['consensus_level'] = weighted_consensus / total_weight
            aggregated['diversity_index'] = weighted_diversity / total_weight

        # Process aggregated concepts
        for concept, data in aggregated['core_concepts'].items():
            values = data['values']
            weights = data['weights']

            if values and weights:
                weighted_sum = sum(v.get('weighted_average', 0) * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                aggregated['core_concepts'][concept] = {
                    'aggregated_value': weighted_sum / weight_sum if weight_sum > 0 else 0,
                    'confidence': weight_sum / len(topic_wisdom),
                    'source_count': len(values)
                }

        return aggregated

    async def _consensus_based_aggregation(self, topic_wisdom: Dict[str, CollectiveWisdom]) -> Dict[str, Any]:
        """Aggregate based on consensus levels"""
        # Filter by high consensus items only
        high_consensus_items = {}

        for topic, wisdom in topic_wisdom.items():
            if wisdom.consensus_level > self.config['quality_threshold']:
                high_consensus_items[topic] = wisdom

        if not high_consensus_items:
            # Fall back to weighted average if no high consensus
            return await self._weighted_average_aggregation(topic_wisdom)

        return await self._weighted_average_aggregation(high_consensus_items)

    async def _expertise_weighted_aggregation(self, topic_wisdom: Dict[str, CollectiveWisdom]) -> Dict[str, Any]:
        """Aggregate with expertise weighting"""
        # Weight by contributor expertise (simplified)
        expertise_weighted = {}

        for topic, wisdom in topic_wisdom.items():
            # Calculate expertise weight based on contributor count and quality
            expertise_weight = wisdom.contributor_count * wisdom.wisdom_quality_score
            expertise_weighted[topic] = wisdom
            # Modify wisdom quality score to reflect expertise
            wisdom.wisdom_quality_score = expertise_weight

        return await self._weighted_average_aggregation(expertise_weighted)

    async def _generate_cross_topic_insights(self,
                                           topic_wisdom: Dict[str, CollectiveWisdom],
                                           aggregated: Dict[str, Any]) -> List[str]:
        """Generate insights from cross-topic analysis"""
        insights = []

        # Analyze concept overlap
        all_concepts = set()
        topic_concepts = {}

        for topic, wisdom in topic_wisdom.items():
            concepts = set(wisdom.aggregated_knowledge.get('core_concepts', {}).keys())
            all_concepts.update(concepts)
            topic_concepts[topic] = concepts

        # Find common concepts
        common_concepts = all_concepts.copy()
        for concepts in topic_concepts.values():
            common_concepts &= concepts

        if len(common_concepts) > 2:
            insights.append(f"Strong conceptual overlap across topics: {', '.join(list(common_concepts)[:3])}")

        # Analyze quality consistency
        quality_scores = [wisdom.wisdom_quality_score for wisdom in topic_wisdom.values()]
        if quality_scores:
            quality_std = self.harvester._calculate_std(quality_scores)
            if quality_std < 0.1:
                insights.append("Consistent high quality across all topics")

        # Analyze emergent patterns
        all_insights = []
        for wisdom in topic_wisdom.values():
            all_insights.extend(wisdom.emergent_insights)

        if len(all_insights) > len(topic_wisdom) * 2:
            insights.append("Rich emergent insights across topic domains")

        return insights


class GroupProblemSolvingAnalyzer:
    """
    🔍 GROUP PROBLEM SOLVING ANALYZER

    Advanced analyzer for group problem-solving processes and outcomes.
    """

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service

        # Analysis configuration
        self.config = {
            'solution_evaluation_criteria': ['feasibility', 'innovation', 'completeness', 'efficiency'],
            'collaboration_metrics': ['participation', 'communication', 'coordination', 'conflict_resolution'],
            'innovation_threshold': 0.7,
            'solution_confidence_threshold': 0.6
        }

        # Problem solving tracking
        self.problem_solving_history = []
        self.solution_database = {}

        logger.info("Group Problem Solving Analyzer initialized")

    async def analyze_group_problem_solving(self,
                                          problem_description: str,
                                          group_process_data: Dict[str, Any],
                                          solution_proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze group problem-solving process and outcomes

        Args:
            problem_description: Description of the problem being solved
            group_process_data: Data about the group problem-solving process
            solution_proposals: List of proposed solutions from the group

        Returns:
            Dict with comprehensive problem-solving analysis
        """
        try:
            problem_id = f"problem_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Analyze problem complexity
            problem_analysis = await self._analyze_problem_complexity(problem_description)

            # Analyze solution proposals
            solution_analysis = await self._analyze_solution_proposals(solution_proposals, problem_analysis)

            # Analyze group collaboration process
            collaboration_analysis = await self._analyze_collaboration_process(group_process_data)

            # Evaluate solution quality
            solution_evaluation = await self._evaluate_solution_quality(
                solution_proposals, problem_analysis, collaboration_analysis
            )

            # Identify best solution
            best_solution = await self._identify_best_solution(solution_proposals, solution_evaluation)

            # Analyze innovation level
            innovation_analysis = await self._analyze_innovation_level(solution_proposals, problem_analysis)

            # Create problem solution result
            problem_solution = GroupProblemSolution(
                problem_id=problem_id,
                problem_description=problem_description,
                solution_approaches=[sol for sol in solution_proposals],
                best_solution=best_solution,
                solution_confidence=solution_evaluation.get('overall_confidence', 0.0),
                collaboration_quality=collaboration_analysis.get('overall_quality', 0.0),
                innovation_score=innovation_analysis.get('innovation_score', 0.0),
                problem_complexity=problem_analysis.get('complexity_score', 0.0),
                solving_process=group_process_data.get('process_steps', []),
                group_dynamics_impact=collaboration_analysis.get('dynamics_impact', {})
            )

            # Store in solution database
            self.solution_database[problem_id] = problem_solution

            # Track problem solving history
            self.problem_solving_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'problem_id': problem_id,
                'solution_count': len(solution_proposals),
                'collaboration_quality': collaboration_analysis.get('overall_quality', 0.0),
                'innovation_score': innovation_analysis.get('innovation_score', 0.0)
            })

            return {
                'status': 'success',
                'problem_solution': problem_solution.__dict__,
                'analysis_details': {
                    'problem_analysis': problem_analysis,
                    'solution_analysis': solution_analysis,
                    'collaboration_analysis': collaboration_analysis,
                    'innovation_analysis': innovation_analysis
                },
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing group problem solving: {e}")
            return {'status': 'error', 'error': str(e)}