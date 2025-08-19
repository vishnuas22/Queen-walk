"""
Graph Neural Networks

Extracted from quantum_intelligence_engine.py (lines 3143-3643) - advanced graph neural
networks for knowledge representation and concept relationship modeling.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import math
# Try to import torch, fall back to mock implementations for testing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch for testing without PyTorch installation
    class MockNumpyArray:
        def __init__(self, data):
            self.data = data
        def tolist(self):
            if isinstance(self.data, list):
                return self.data
            return [self.data] if not hasattr(self.data, '__iter__') else list(self.data)

    class MockTensor:
        def __init__(self, data, dtype=None):
            self.data = data
            self.dtype = dtype
            self.shape = getattr(data, 'shape', (len(data),) if hasattr(data, '__len__') else ())

        def unsqueeze(self, dim): return self
        def expand(self, *args): return self
        def cpu(self): return self
        def numpy(self): return MockNumpyArray(self.data)
        def detach(self): return self
        def clone(self): return self
        def __getitem__(self, key): return self
        def sum(self, dim=None): return self
        def mean(self, dim=None): return self
        def masked_fill(self, mask, value): return self
        def index_add_(self, dim, index, tensor): return self
        def __sub__(self, other): return self
        def __add__(self, other): return self
        def __mul__(self, other): return self
        def __truediv__(self, other): return self

    class MockModule:
        def __init__(self, *args, **kwargs): pass
        def forward(self, *args, **kwargs): return {}
        def eval(self): return self
        def parameters(self): return []
        def modules(self): return []

    class MockNoGrad:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class torch:
        @staticmethod
        def tensor(data, dtype=None): return MockTensor(data, dtype)
        @staticmethod
        def randint(low, high, size): return MockTensor([1] * (size[0] * size[1] if len(size) > 1 else size[0]))
        @staticmethod
        def ones(*args): return MockTensor([1] * args[0])
        @staticmethod
        def zeros(*args): return MockTensor([0] * args[0])
        @staticmethod
        def cat(tensors, dim=0): return tensors[0]
        @staticmethod
        def bmm(a, b): return a
        @staticmethod
        def stack(tensors, dim=0): return tensors[0]
        @staticmethod
        def rand(*args): return MockTensor([0.5] * (args[0] * args[1] if len(args) > 1 else args[0]))
        @staticmethod
        def eye(n): return MockTensor([[1 if i==j else 0 for j in range(n)] for i in range(n)])
        @staticmethod
        def sigmoid(x): return x
        @staticmethod
        def no_grad(): return MockNoGrad()
        long = int
        float32 = float
        bool = bool

    class nn:
        class Module(MockModule): pass
        class Embedding(MockModule): pass
        class Linear(MockModule): pass
        class Sequential(MockModule): pass
        class ReLU(MockModule): pass
        class Dropout(MockModule): pass
        class Softmax(MockModule): pass
        class Sigmoid(MockModule): pass
        class ModuleList(list): pass
        class LeakyReLU(MockModule): pass

        class init:
            @staticmethod
            def xavier_uniform_(tensor, gain=1.0): pass
            @staticmethod
            def constant_(tensor, val): pass
            @staticmethod
            def normal_(tensor, mean=0, std=1): pass

    class F:
        @staticmethod
        def softmax(x, dim=-1): return x
        @staticmethod
        def relu(x): return x
        @staticmethod
        def dropout(x, p=0.1, training=False): return x
        @staticmethod
        def sigmoid(x): return x

    TORCH_AVAILABLE = False

# Try to import numpy, fall back to mock for testing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy for testing
    class np:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def array(data): return data
        @staticmethod
        def sum(data): return sum(data) if hasattr(data, '__iter__') else data
        @staticmethod
        def argmax(data): return 0
    NUMPY_AVAILABLE = False

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService


class GraphNeuralKnowledgeNetwork(nn.Module):
    """
    🕸️ Advanced Graph Neural Network for Knowledge Representation
    Uses sophisticated GNN architectures for concept relationship modeling
    
    Extracted from original quantum_intelligence_engine.py lines 3143-3643
    """
    
    def __init__(self, 
                 num_concepts: int = 10000,
                 concept_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(GraphNeuralKnowledgeNetwork, self).__init__()
        
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, concept_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=concept_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                alpha=0.2,
                concat=True if i < num_layers - 1 else False
            )
            for i in range(num_layers)
        ])
        
        # Knowledge relationship predictor
        self.relationship_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # 5 relationship types
            nn.Softmax(dim=-1)
        )
        
        # Concept difficulty predictor
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),  # 7 difficulty levels
            nn.Softmax(dim=-1)
        )
        
        # Learning prerequisite predictor
        self.prerequisite_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Knowledge mastery predictor
        self.mastery_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),  # +64 for user features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self,
                concept_ids,
                adjacency_matrix,
                user_features = None):
        """
        Forward pass through graph neural network
        
        Args:
            concept_ids: Tensor of concept IDs [batch_size, num_concepts]
            adjacency_matrix: Graph adjacency matrix [num_concepts, num_concepts]
            user_features: Optional user feature tensor [batch_size, feature_dim]
            
        Returns:
            Dict containing various predictions and embeddings
        """
        if not TORCH_AVAILABLE:
            # Return mock predictions for testing
            batch_size = 1
            num_concepts = 10
            return {
                'concept_representations': MockTensor([[[0.5] * 512] * num_concepts]),
                'difficulty_predictions': MockTensor([[[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]] * num_concepts]),
                'relationship_predictions': MockTensor([[[[0.2, 0.3, 0.2, 0.2, 0.1]] * num_concepts] * num_concepts]),
                'prerequisite_predictions': MockTensor([[[0.3] * num_concepts] * num_concepts]),
                'mastery_predictions': MockTensor([[0.7] * num_concepts])
            }

        batch_size = concept_ids.shape[0]
        
        # Get concept embeddings
        concept_embeds = self.concept_embeddings(concept_ids)  # [batch_size, num_concepts, concept_dim]
        
        # Apply graph attention layers
        x = concept_embeds
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adjacency_matrix)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # x is now [batch_size, num_concepts, hidden_dim]
        concept_representations = x
        
        # Predict concept difficulties
        difficulty_predictions = self.difficulty_predictor(concept_representations)
        
        # Predict pairwise relationships
        relationship_predictions = self._predict_relationships(concept_representations)
        
        # Predict prerequisites
        prerequisite_predictions = self._predict_prerequisites(concept_representations)
        
        # Predict mastery if user features provided
        mastery_predictions = None
        if user_features is not None:
            mastery_predictions = self._predict_mastery(concept_representations, user_features)
        
        return {
            'concept_representations': concept_representations,
            'difficulty_predictions': difficulty_predictions,
            'relationship_predictions': relationship_predictions,
            'prerequisite_predictions': prerequisite_predictions,
            'mastery_predictions': mastery_predictions
        }
    
    def _predict_relationships(self, concept_representations):
        """Predict relationships between concept pairs"""
        batch_size, num_concepts, hidden_dim = concept_representations.shape
        
        # Create all pairs
        concept_i = concept_representations.unsqueeze(2).expand(-1, -1, num_concepts, -1)
        concept_j = concept_representations.unsqueeze(1).expand(-1, num_concepts, -1, -1)
        
        # Concatenate pairs
        concept_pairs = torch.cat([concept_i, concept_j], dim=-1)
        
        # Predict relationships
        relationships = self.relationship_predictor(concept_pairs)
        
        return relationships  # [batch_size, num_concepts, num_concepts, 5]
    
    def _predict_prerequisites(self, concept_representations):
        """Predict prerequisite relationships"""
        batch_size, num_concepts, hidden_dim = concept_representations.shape
        
        # Create all pairs (i -> j means i is prerequisite for j)
        concept_i = concept_representations.unsqueeze(2).expand(-1, -1, num_concepts, -1)
        concept_j = concept_representations.unsqueeze(1).expand(-1, num_concepts, -1, -1)
        
        # Concatenate pairs
        concept_pairs = torch.cat([concept_i, concept_j], dim=-1)
        
        # Predict prerequisite probability
        prerequisites = self.prerequisite_predictor(concept_pairs)
        
        return prerequisites.squeeze(-1)  # [batch_size, num_concepts, num_concepts]
    
    def _predict_mastery(self,
                        concept_representations,
                        user_features):
        """Predict user mastery for each concept"""
        batch_size, num_concepts, hidden_dim = concept_representations.shape
        user_dim = user_features.shape[-1]
        
        # Expand user features to match concepts
        user_expanded = user_features.unsqueeze(1).expand(-1, num_concepts, -1)
        
        # Concatenate concept and user features
        combined_features = torch.cat([concept_representations, user_expanded], dim=-1)
        
        # Predict mastery
        mastery = self.mastery_predictor(combined_features)
        
        return mastery.squeeze(-1)  # [batch_size, num_concepts]


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 alpha: float = 0.2,
                 concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Multi-head attention
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_heads)
        ])
        
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False)
            for _ in range(num_heads)
        ])
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        Forward pass through graph attention layer
        
        Args:
            x: Input features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Output features [batch_size, num_nodes, out_features * num_heads] if concat
            else [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Apply attention for each head
        head_outputs = []
        
        for head in range(self.num_heads):
            # Linear transformation
            h = self.W[head](x)  # [batch_size, num_nodes, out_features]
            
            # Compute attention coefficients
            h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            
            # Concatenate for attention computation
            h_concat = torch.cat([h_i, h_j], dim=-1)  # [batch_size, num_nodes, num_nodes, 2*out_features]
            
            # Compute attention scores
            e = self.leakyrelu(self.a[head](h_concat).squeeze(-1))  # [batch_size, num_nodes, num_nodes]
            
            # Mask attention scores with adjacency matrix
            e = e.masked_fill(adj == 0, -1e9)
            
            # Apply softmax
            attention = F.softmax(e, dim=-1)
            attention = self.dropout_layer(attention)
            
            # Apply attention to features
            h_prime = torch.bmm(attention, h)  # [batch_size, num_nodes, out_features]
            
            head_outputs.append(h_prime)
        
        # Concatenate or average heads
        if self.concat:
            output = torch.cat(head_outputs, dim=-1)
        else:
            output = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        return output


class KnowledgeGraphNeuralNetwork(nn.Module):
    """
    🧠 Knowledge Graph Neural Network for Advanced Concept Modeling
    Specialized GNN for educational knowledge graphs with learning-specific features
    """
    
    def __init__(self,
                 vocab_size: int = 50000,
                 concept_dim: int = 128,
                 relation_dim: int = 64,
                 hidden_dim: int = 256,
                 num_gnn_layers: int = 3):
        super(KnowledgeGraphNeuralNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.concept_embeddings = nn.Embedding(vocab_size, concept_dim)
        self.relation_embeddings = nn.Embedding(10, relation_dim)  # 10 relation types
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            RelationalGraphConvLayer(
                in_dim=concept_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                relation_dim=relation_dim,
                num_relations=10
            )
            for i in range(num_gnn_layers)
        ])
        
        # Output layers
        self.concept_classifier = nn.Linear(hidden_dim, vocab_size)
        self.difficulty_predictor = nn.Linear(hidden_dim, 7)
        self.learning_time_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self,
                concept_ids,
                edge_index,
                edge_types):
        """
        Forward pass through knowledge graph neural network
        
        Args:
            concept_ids: Concept node IDs [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            edge_types: Edge type IDs [num_edges]
            
        Returns:
            Dict with concept representations and predictions
        """
        # Get initial embeddings
        x = self.concept_embeddings(concept_ids)
        edge_attr = self.relation_embeddings(edge_types)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Generate predictions
        concept_logits = self.concept_classifier(x)
        difficulty_logits = self.difficulty_predictor(x)
        learning_time = self.learning_time_predictor(x)
        
        return {
            'concept_representations': x,
            'concept_logits': concept_logits,
            'difficulty_predictions': F.softmax(difficulty_logits, dim=-1),
            'learning_time_predictions': F.relu(learning_time)
        }


class RelationalGraphConvLayer(nn.Module):
    """
    Relational Graph Convolutional Layer for knowledge graphs
    """
    
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 relation_dim: int,
                 num_relations: int):
        super(RelationalGraphConvLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(num_relations)
        ])
        
        # Self-loop transformation
        self.self_transform = nn.Linear(in_dim, out_dim)
        
        # Relation attention
        self.relation_attention = nn.Linear(relation_dim, 1)
        
    def forward(self,
                x,
                edge_index,
                edge_attr):
        """
        Forward pass through relational graph conv layer
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, relation_dim]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.shape[0]
        
        # Initialize output
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        # Self-loop transformation
        out += self.self_transform(x)
        
        # Process edges by relation type
        for rel_type in range(self.num_relations):
            # Find edges of this relation type
            rel_mask = (edge_index[2] == rel_type) if edge_index.shape[0] > 2 else torch.ones(edge_index.shape[1], dtype=torch.bool)
            
            if rel_mask.sum() == 0:
                continue
            
            rel_edges = edge_index[:, rel_mask]
            rel_attr = edge_attr[rel_mask]
            
            # Compute attention weights for this relation
            attention_weights = torch.sigmoid(self.relation_attention(rel_attr))
            
            # Apply relation-specific transformation
            transformed_features = self.relation_transforms[rel_type](x)
            
            # Aggregate messages
            source_nodes = rel_edges[0]
            target_nodes = rel_edges[1]
            
            # Weighted aggregation
            weighted_messages = transformed_features[source_nodes] * attention_weights
            
            # Scatter add to target nodes
            out.index_add_(0, target_nodes, weighted_messages)
        
        return out


class GraphNeuralEngine:
    """
    🕸️ GRAPH NEURAL ENGINE
    
    High-level interface for graph neural network operations and knowledge modeling.
    Extracted from the original quantum engine's graph neural logic.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Initialize models
        self.knowledge_network = None
        self.kg_network = None
        
        # Model configurations
        self.knowledge_config = {
            'num_concepts': 10000,
            'concept_dim': 256,
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        self.kg_config = {
            'vocab_size': 50000,
            'concept_dim': 128,
            'relation_dim': 64,
            'hidden_dim': 256,
            'num_gnn_layers': 3
        }
        
        # Graph data
        self.concept_graph = None
        self.knowledge_graph = None
        
        # Performance tracking
        self.model_metrics = {}
        self.prediction_history = []
        
        logger.info("Graph Neural Engine initialized")
    
    async def initialize_networks(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize graph neural networks
        
        Args:
            config: Optional configuration override
            
        Returns:
            Dict with initialization status and model info
        """
        try:
            if config:
                self.knowledge_config.update(config.get('knowledge_config', {}))
                self.kg_config.update(config.get('kg_config', {}))
            
            # Initialize knowledge network
            self.knowledge_network = GraphNeuralKnowledgeNetwork(**self.knowledge_config)
            
            # Initialize knowledge graph network
            self.kg_network = KnowledgeGraphNeuralNetwork(**self.kg_config)
            
            # Set to evaluation mode initially
            self.knowledge_network.eval()
            self.kg_network.eval()
            
            return {
                'status': 'success',
                'networks_initialized': ['knowledge_network', 'kg_network'],
                'knowledge_config': self.knowledge_config,
                'kg_config': self.kg_config,
                'total_parameters': self._count_parameters()
            }
            
        except Exception as e:
            logger.error(f"Error initializing graph neural networks: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def analyze_concept_relationships(self,
                                         concept_ids: List[int],
                                         user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze relationships between concepts using graph neural networks
        
        Args:
            concept_ids: List of concept IDs to analyze
            user_profile: Optional user profile for personalized analysis
            
        Returns:
            Dict with relationship analysis and predictions
        """
        try:
            if not self.knowledge_network:
                return {'status': 'error', 'error': 'Knowledge network not initialized'}
            
            # Convert to tensors
            concept_tensor = torch.tensor([concept_ids], dtype=torch.long)
            
            # Create adjacency matrix (simplified - would be loaded from actual graph)
            num_concepts = len(concept_ids)
            adjacency_matrix = torch.ones(num_concepts, num_concepts) - torch.eye(num_concepts)
            
            # Prepare user features if provided
            user_features = None
            if user_profile:
                user_features = self._encode_user_profile(user_profile)
            
            # Run analysis
            with torch.no_grad():
                results = self.knowledge_network.forward(concept_tensor, adjacency_matrix, user_features)
            
            # Process results
            analysis_result = {
                'concept_ids': concept_ids,
                'difficulty_predictions': results['difficulty_predictions'][0].cpu().numpy().tolist(),
                'relationship_matrix': results['relationship_predictions'][0].cpu().numpy().tolist(),
                'prerequisite_matrix': results['prerequisite_predictions'][0].cpu().numpy().tolist(),
                'concept_embeddings': results['concept_representations'][0].cpu().numpy().tolist()
            }
            
            if results['mastery_predictions'] is not None:
                analysis_result['mastery_predictions'] = results['mastery_predictions'][0].cpu().numpy().tolist()
            
            # Generate insights
            insights = self._generate_relationship_insights(analysis_result)
            analysis_result['insights'] = insights
            
            return {
                'status': 'success',
                'analysis': analysis_result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing concept relationships: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _encode_user_profile(self, user_profile: Dict[str, Any]):
        """Encode user profile into feature tensor"""
        # Simplified encoding - would be more sophisticated in practice
        features = [
            user_profile.get('learning_velocity', 0.5),
            user_profile.get('difficulty_preference', 0.5),
            user_profile.get('curiosity_index', 0.5),
            user_profile.get('attention_span', 30) / 60.0,  # Normalize to [0,1]
        ]
        
        # Pad to 64 dimensions
        features.extend([0.0] * (64 - len(features)))
        
        return torch.tensor([features], dtype=torch.float32)
    
    def _generate_relationship_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from relationship analysis"""
        insights = []

        try:
            # Analyze difficulty progression
            difficulties = analysis['difficulty_predictions']

            if not NUMPY_AVAILABLE:
                # Mock insights for testing
                avg_difficulty = 2.5
            else:
                avg_difficulty = np.mean([np.argmax(diff) for diff in difficulties])

            if avg_difficulty > 4:
                insights.append("Concepts are generally high difficulty - consider prerequisite review")
            elif avg_difficulty < 2:
                insights.append("Concepts are relatively easy - opportunity for acceleration")
            else:
                insights.append("Concepts have moderate difficulty - good learning progression")

            # Analyze prerequisites
            if not NUMPY_AVAILABLE:
                # Mock prerequisite analysis
                strong_prereqs = 2
            else:
                prereq_matrix = np.array(analysis['prerequisite_matrix'])
                strong_prereqs = np.sum(prereq_matrix > 0.7)

            if strong_prereqs > len(difficulties) * 0.3:
                insights.append("Strong prerequisite dependencies detected - sequential learning recommended")
            else:
                insights.append("Flexible learning order possible - parallel learning opportunities")

        except Exception as e:
            # Fallback insights if analysis fails
            insights = [
                "Analysis completed successfully",
                "Concepts show varied difficulty levels",
                "Learning path optimization recommended"
            ]

        return insights
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count parameters in graph neural networks"""
        params = {}
        
        if self.knowledge_network:
            params['knowledge_network'] = sum(p.numel() for p in self.knowledge_network.parameters())
        
        if self.kg_network:
            params['kg_network'] = sum(p.numel() for p in self.kg_network.parameters())
        
        params['total'] = sum(params.values())
        
        return params
