"""
Social Learning Networks Services

Extracted from quantum_intelligence_engine.py (lines 10289-12523) - advanced social learning
network analysis, influence propagation, learning community management, and knowledge sharing incentives.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
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
class SocialLearningNetwork:
    """Social learning network structure"""
    network_id: str = ""
    network_name: str = ""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    network_metrics: Dict[str, float] = field(default_factory=dict)
    community_structure: Dict[str, List[str]] = field(default_factory=dict)
    influence_patterns: Dict[str, Any] = field(default_factory=dict)
    knowledge_flow: Dict[str, Any] = field(default_factory=dict)
    creation_timestamp: str = ""


@dataclass
class InfluencePropagation:
    """Influence propagation analysis results"""
    source_node: str = ""
    influence_type: str = ""
    propagation_path: List[str] = field(default_factory=list)
    influence_strength: float = 0.0
    propagation_speed: float = 0.0
    affected_nodes: Set[str] = field(default_factory=set)
    influence_decay: float = 0.0
    propagation_barriers: List[str] = field(default_factory=list)
    amplification_nodes: List[str] = field(default_factory=list)


@dataclass
class LearningCommunity:
    """Learning community data structure"""
    community_id: str = ""
    community_name: str = ""
    members: List[str] = field(default_factory=list)
    community_type: str = ""
    learning_focus: List[str] = field(default_factory=list)
    activity_level: float = 0.0
    knowledge_sharing_rate: float = 0.0
    community_health: float = 0.0
    growth_rate: float = 0.0
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    community_leaders: List[str] = field(default_factory=list)


class SocialLearningNetworkAnalyzer:
    """
    🌐 SOCIAL LEARNING NETWORK ANALYZER
    
    Advanced analyzer for social learning networks and knowledge propagation.
    Extracted from the original quantum engine's collaborative intelligence logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Network analysis configuration
        self.config = {
            'centrality_measures': ['degree', 'betweenness', 'closeness', 'eigenvector'],
            'community_detection_algorithm': 'modularity_optimization',
            'influence_threshold': 0.3,
            'knowledge_flow_window_days': 30,
            'network_health_threshold': 0.7
        }
        
        # Network tracking
        self.networks = {}
        self.analysis_history = []
        
        logger.info("Social Learning Network Analyzer initialized")
    
    async def analyze_social_learning_network(self,
                                            network_data: Dict[str, Any],
                                            analysis_scope: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze social learning network structure and dynamics
        
        Args:
            network_data: Network structure data (nodes, edges, interactions)
            analysis_scope: Optional scope for analysis (time range, specific metrics)
            
        Returns:
            Dict with comprehensive network analysis
        """
        try:
            network_id = network_data.get('network_id', f"network_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            
            # Build network structure
            network_structure = await self._build_network_structure(network_data)
            
            # Calculate network metrics
            network_metrics = await self._calculate_network_metrics(network_structure)
            
            # Detect communities
            community_structure = await self._detect_communities(network_structure)
            
            # Analyze influence patterns
            influence_patterns = await self._analyze_influence_patterns(network_structure, network_data)
            
            # Analyze knowledge flow
            knowledge_flow = await self._analyze_knowledge_flow(network_structure, network_data)
            
            # Create social learning network
            social_network = SocialLearningNetwork(
                network_id=network_id,
                network_name=network_data.get('network_name', f'Learning Network {network_id}'),
                nodes=network_structure['nodes'],
                edges=network_structure['edges'],
                network_metrics=network_metrics,
                community_structure=community_structure,
                influence_patterns=influence_patterns,
                knowledge_flow=knowledge_flow,
                creation_timestamp=datetime.utcnow().isoformat()
            )
            
            # Store network
            self.networks[network_id] = social_network
            
            # Generate network insights
            network_insights = await self._generate_network_insights(social_network)
            
            # Track analysis history
            self.analysis_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'network_id': network_id,
                'node_count': len(network_structure['nodes']),
                'edge_count': len(network_structure['edges']),
                'network_health': network_metrics.get('network_health', 0.0)
            })
            
            return {
                'status': 'success',
                'social_network': social_network.__dict__,
                'network_insights': network_insights,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social learning network: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _build_network_structure(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build network structure from input data"""
        nodes = {}
        edges = []
        
        # Process nodes
        raw_nodes = network_data.get('nodes', [])
        for node_data in raw_nodes:
            node_id = node_data.get('user_id', node_data.get('id', 'unknown'))
            nodes[node_id] = {
                'id': node_id,
                'name': node_data.get('name', f'User {node_id}'),
                'learning_profile': node_data.get('learning_profile', {}),
                'activity_level': node_data.get('activity_level', 0.5),
                'expertise_areas': node_data.get('expertise_areas', []),
                'connection_count': 0,
                'influence_score': 0.0
            }
        
        # Process edges
        raw_edges = network_data.get('edges', [])
        for edge_data in raw_edges:
            source = edge_data.get('source', edge_data.get('from'))
            target = edge_data.get('target', edge_data.get('to'))
            
            if source and target and source in nodes and target in nodes:
                edge = {
                    'source': source,
                    'target': target,
                    'weight': edge_data.get('weight', 1.0),
                    'interaction_type': edge_data.get('interaction_type', 'general'),
                    'frequency': edge_data.get('frequency', 1),
                    'strength': edge_data.get('strength', 0.5),
                    'knowledge_exchange': edge_data.get('knowledge_exchange', False)
                }
                edges.append(edge)
                
                # Update node connection counts
                nodes[source]['connection_count'] += 1
                nodes[target]['connection_count'] += 1
        
        return {'nodes': nodes, 'edges': edges}
    
    async def _calculate_network_metrics(self, network_structure: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive network metrics"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        if not nodes:
            return {'network_health': 0.0}
        
        # Basic metrics
        node_count = len(nodes)
        edge_count = len(edges)
        density = (2 * edge_count) / (node_count * (node_count - 1)) if node_count > 1 else 0
        
        # Calculate centrality measures
        centrality_metrics = await self._calculate_centrality_measures(network_structure)
        
        # Calculate clustering coefficient
        clustering_coefficient = await self._calculate_clustering_coefficient(network_structure)
        
        # Calculate path lengths
        path_metrics = await self._calculate_path_metrics(network_structure)
        
        # Calculate network health
        network_health = await self._calculate_network_health(network_structure, centrality_metrics)
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': density,
            'clustering_coefficient': clustering_coefficient,
            'average_path_length': path_metrics.get('average_path_length', 0.0),
            'diameter': path_metrics.get('diameter', 0.0),
            'network_health': network_health,
            **centrality_metrics
        }
    
    async def _calculate_centrality_measures(self, network_structure: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various centrality measures"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        # Build adjacency structure
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        # Degree centrality
        degree_centrality = {}
        max_degree = 0
        for node_id in nodes:
            degree = len(adjacency[node_id])
            degree_centrality[node_id] = degree
            max_degree = max(max_degree, degree)
        
        # Normalize degree centrality
        if max_degree > 0:
            for node_id in degree_centrality:
                degree_centrality[node_id] /= max_degree
        
        # Betweenness centrality (simplified)
        betweenness_centrality = await self._calculate_betweenness_centrality(adjacency, nodes)
        
        # Calculate average centralities
        avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0
        avg_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0
        
        return {
            'average_degree_centrality': avg_degree_centrality,
            'average_betweenness_centrality': avg_betweenness_centrality,
            'max_degree_centrality': max(degree_centrality.values()) if degree_centrality else 0,
            'centrality_variance': self._calculate_variance(list(degree_centrality.values()))
        }
    
    async def _calculate_betweenness_centrality(self, adjacency: Dict[str, Set[str]], nodes: Dict[str, Any]) -> Dict[str, float]:
        """Calculate betweenness centrality using simplified algorithm"""
        betweenness = {node_id: 0.0 for node_id in nodes}
        
        # Simplified betweenness calculation
        for source in nodes:
            for target in nodes:
                if source != target:
                    # Find shortest path (BFS)
                    path = self._find_shortest_path(adjacency, source, target)
                    if path and len(path) > 2:
                        # Add to betweenness for intermediate nodes
                        for intermediate in path[1:-1]:
                            betweenness[intermediate] += 1.0
        
        # Normalize
        max_betweenness = max(betweenness.values()) if betweenness.values() else 1
        if max_betweenness > 0:
            for node_id in betweenness:
                betweenness[node_id] /= max_betweenness
        
        return betweenness
    
    def _find_shortest_path(self, adjacency: Dict[str, Set[str]], source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS"""
        if source == target:
            return [source]
        
        visited = set()
        queue = deque([(source, [source])])
        
        while queue:
            current, path = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in adjacency.get(current, set()):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    async def _calculate_clustering_coefficient(self, network_structure: Dict[str, Any]) -> float:
        """Calculate global clustering coefficient"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        # Build adjacency structure
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        total_clustering = 0.0
        valid_nodes = 0
        
        for node_id in nodes:
            neighbors = adjacency[node_id]
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 < neighbor2 and neighbor2 in adjacency[neighbor1]:
                        triangles += 1
            
            if possible_triangles > 0:
                local_clustering = triangles / possible_triangles
                total_clustering += local_clustering
                valid_nodes += 1
        
        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0
    
    async def _calculate_path_metrics(self, network_structure: Dict[str, Any]) -> Dict[str, float]:
        """Calculate path-related metrics"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        # Build adjacency structure
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        path_lengths = []
        max_path_length = 0
        
        node_list = list(nodes.keys())
        for i, source in enumerate(node_list):
            for target in node_list[i+1:]:
                path = self._find_shortest_path(adjacency, source, target)
                if path:
                    path_length = len(path) - 1
                    path_lengths.append(path_length)
                    max_path_length = max(max_path_length, path_length)
        
        average_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        
        return {
            'average_path_length': average_path_length,
            'diameter': max_path_length
        }
    
    async def _calculate_network_health(self,
                                      network_structure: Dict[str, Any],
                                      centrality_metrics: Dict[str, float]) -> float:
        """Calculate overall network health score"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        if not nodes:
            return 0.0
        
        # Connectivity health
        connectivity_score = min(1.0, len(edges) / len(nodes))
        
        # Activity health
        activity_scores = [node.get('activity_level', 0.5) for node in nodes.values()]
        activity_health = sum(activity_scores) / len(activity_scores)
        
        # Centrality distribution health (avoid over-centralization)
        centrality_variance = centrality_metrics.get('centrality_variance', 0.5)
        centrality_health = 1.0 - min(1.0, centrality_variance)
        
        # Knowledge exchange health
        knowledge_edges = sum(1 for edge in edges if edge.get('knowledge_exchange', False))
        knowledge_health = knowledge_edges / len(edges) if edges else 0
        
        # Overall health
        health_score = (
            connectivity_score * 0.3 +
            activity_health * 0.3 +
            centrality_health * 0.2 +
            knowledge_health * 0.2
        )
        
        return health_score
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    async def _detect_communities(self, network_structure: Dict[str, Any]) -> Dict[str, List[str]]:
        """Detect communities in the network"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        # Simple community detection based on modularity (simplified)
        communities = {}
        
        # Build adjacency structure
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        # Use connected components as basic communities
        visited = set()
        community_id = 0
        
        for node_id in nodes:
            if node_id not in visited:
                # BFS to find connected component
                component = []
                queue = deque([node_id])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if component:
                    communities[f'community_{community_id}'] = component
                    community_id += 1
        
        return communities
    
    async def _analyze_influence_patterns(self,
                                        network_structure: Dict[str, Any],
                                        network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze influence patterns in the network"""
        nodes = network_structure['nodes']
        edges = network_structure['edges']
        
        # Calculate influence scores
        influence_scores = {}
        
        for node_id, node_data in nodes.items():
            # Base influence from connections
            connection_influence = node_data['connection_count'] / len(nodes) if nodes else 0
            
            # Activity influence
            activity_influence = node_data.get('activity_level', 0.5)
            
            # Expertise influence
            expertise_count = len(node_data.get('expertise_areas', []))
            expertise_influence = min(1.0, expertise_count / 5)
            
            # Combined influence
            total_influence = (
                connection_influence * 0.4 +
                activity_influence * 0.3 +
                expertise_influence * 0.3
            )
            
            influence_scores[node_id] = total_influence
            nodes[node_id]['influence_score'] = total_influence
        
        # Identify influence hubs
        influence_threshold = self.config['influence_threshold']
        influence_hubs = [
            node_id for node_id, score in influence_scores.items()
            if score > influence_threshold
        ]
        
        # Analyze influence propagation paths
        propagation_paths = await self._analyze_propagation_paths(network_structure, influence_hubs)
        
        return {
            'influence_scores': influence_scores,
            'influence_hubs': influence_hubs,
            'propagation_paths': propagation_paths,
            'average_influence': sum(influence_scores.values()) / len(influence_scores) if influence_scores else 0
        }
    
    async def _analyze_propagation_paths(self,
                                       network_structure: Dict[str, Any],
                                       influence_hubs: List[str]) -> List[Dict[str, Any]]:
        """Analyze influence propagation paths from hubs"""
        edges = network_structure['edges']
        
        # Build adjacency structure
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
        
        propagation_paths = []
        
        for hub in influence_hubs:
            # Find paths from this hub to other nodes
            visited = set()
            queue = deque([(hub, [hub], 1.0)])  # (node, path, influence_strength)
            
            while queue:
                current, path, strength = queue.popleft()
                
                if current in visited or len(path) > 4:  # Limit path length
                    continue
                
                visited.add(current)
                
                if len(path) > 1:  # Don't include the hub itself
                    propagation_paths.append({
                        'source_hub': hub,
                        'target_node': current,
                        'path': path,
                        'path_length': len(path) - 1,
                        'influence_strength': strength
                    })
                
                # Continue propagation to neighbors
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        # Decay influence with distance
                        new_strength = strength * 0.8
                        queue.append((neighbor, path + [neighbor], new_strength))
        
        return propagation_paths
    
    async def _analyze_knowledge_flow(self,
                                    network_structure: Dict[str, Any],
                                    network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze knowledge flow patterns in the network"""
        edges = network_structure['edges']
        
        # Analyze knowledge exchange edges
        knowledge_edges = [edge for edge in edges if edge.get('knowledge_exchange', False)]
        
        # Calculate knowledge flow metrics
        total_knowledge_flow = sum(edge.get('weight', 1.0) for edge in knowledge_edges)
        knowledge_flow_density = len(knowledge_edges) / len(edges) if edges else 0
        
        # Identify knowledge sources and sinks
        knowledge_out = defaultdict(float)
        knowledge_in = defaultdict(float)
        
        for edge in knowledge_edges:
            weight = edge.get('weight', 1.0)
            knowledge_out[edge['source']] += weight
            knowledge_in[edge['target']] += weight
        
        # Identify top knowledge contributors and receivers
        top_contributors = sorted(knowledge_out.items(), key=lambda x: x[1], reverse=True)[:5]
        top_receivers = sorted(knowledge_in.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_knowledge_flow': total_knowledge_flow,
            'knowledge_flow_density': knowledge_flow_density,
            'knowledge_exchange_count': len(knowledge_edges),
            'top_knowledge_contributors': [{'node': node, 'flow_out': flow} for node, flow in top_contributors],
            'top_knowledge_receivers': [{'node': node, 'flow_in': flow} for node, flow in top_receivers],
            'knowledge_balance': self._calculate_knowledge_balance(knowledge_out, knowledge_in)
        }
    
    def _calculate_knowledge_balance(self,
                                   knowledge_out: Dict[str, float],
                                   knowledge_in: Dict[str, float]) -> Dict[str, float]:
        """Calculate knowledge flow balance for nodes"""
        balance = {}
        all_nodes = set(knowledge_out.keys()) | set(knowledge_in.keys())
        
        for node in all_nodes:
            out_flow = knowledge_out.get(node, 0.0)
            in_flow = knowledge_in.get(node, 0.0)
            
            if out_flow + in_flow > 0:
                balance[node] = (out_flow - in_flow) / (out_flow + in_flow)
            else:
                balance[node] = 0.0
        
        return balance
    
    async def _generate_network_insights(self, social_network: SocialLearningNetwork) -> List[str]:
        """Generate insights about the social learning network"""
        insights = []
        
        metrics = social_network.network_metrics
        
        # Network health insights
        health = metrics.get('network_health', 0.0)
        if health > 0.8:
            insights.append("Excellent network health - strong connectivity and engagement")
        elif health < 0.5:
            insights.append("Network health needs improvement - consider engagement initiatives")
        
        # Density insights
        density = metrics.get('density', 0.0)
        if density > 0.3:
            insights.append("High network density promotes knowledge sharing")
        elif density < 0.1:
            insights.append("Low network density - opportunities for new connections")
        
        # Community insights
        community_count = len(social_network.community_structure)
        if community_count > 1:
            insights.append(f"Network has {community_count} distinct learning communities")
        
        # Influence insights
        influence_hubs = social_network.influence_patterns.get('influence_hubs', [])
        if len(influence_hubs) > 3:
            insights.append("Multiple influence hubs support distributed leadership")
        elif len(influence_hubs) == 0:
            insights.append("No clear influence hubs - consider leadership development")
        
        # Knowledge flow insights
        knowledge_flow_density = social_network.knowledge_flow.get('knowledge_flow_density', 0.0)
        if knowledge_flow_density > 0.5:
            insights.append("Strong knowledge sharing culture detected")
        elif knowledge_flow_density < 0.2:
            insights.append("Limited knowledge sharing - consider incentive programs")
        
        return insights
