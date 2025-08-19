#!/usr/bin/env python3
"""
Test Phase 3 Neural Architectures - Service Extraction Validation

This script validates that Phase 3 neural architecture service extraction is proceeding correctly.
"""

import sys
import asyncio
from pathlib import Path

# Try to import torch and numpy, use mocks if not available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    # Mock implementations for testing
    class MockTensor:
        def __init__(self, data, dtype=None):
            self.data = data
            self.dtype = dtype
            self.shape = getattr(data, 'shape', (len(data),) if hasattr(data, '__len__') else ())

        def cpu(self): return self
        def numpy(self): return self.data

    class torch:
        @staticmethod
        def randint(low, high, size): return MockTensor([1] * (size[0] * size[1] if len(size) > 1 else size[0]))
        @staticmethod
        def ones(*args): return MockTensor([1] * args[0])
        @staticmethod
        def rand(*args): return MockTensor([0.5] * (args[0] * args[1] if len(args) > 1 else args[0]))
        long = int
        float32 = float

    class np:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def array(data): return data

    TORCH_AVAILABLE = False

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_neural_services_import():
    """Test neural services can be imported"""
    print("🧪 Testing neural services import...")
    
    try:
        # Test neural services
        from quantum_intelligence.services.neural.transformers import (
            QuantumTransformerLearningPathOptimizer,
            TransformerLearningPathOptimizer,
            QuantumTransformerEngine
        )
        from quantum_intelligence.services.neural.graph_networks import (
            GraphNeuralKnowledgeNetwork,
            KnowledgeGraphNeuralNetwork,
            GraphNeuralEngine
        )
        
        print("✅ Neural service classes can be imported")
        
        # Test instantiation
        quantum_transformer = QuantumTransformerLearningPathOptimizer()
        transformer_optimizer = TransformerLearningPathOptimizer()
        transformer_engine = QuantumTransformerEngine()
        
        graph_knowledge_net = GraphNeuralKnowledgeNetwork()
        kg_network = KnowledgeGraphNeuralNetwork()
        graph_engine = GraphNeuralEngine()
        
        print("✅ Neural services can be instantiated")
        
        # Test that they have the expected methods
        assert hasattr(quantum_transformer, 'forward')
        assert hasattr(quantum_transformer, 'predict_optimal_path')
        assert hasattr(transformer_optimizer, 'generate_learning_path')
        assert hasattr(transformer_engine, 'optimize_learning_path')
        assert hasattr(graph_knowledge_net, 'forward')
        assert hasattr(graph_engine, 'analyze_concept_relationships')
        
        print("✅ Neural services have expected methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural services import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_quantum_transformer_functionality():
    """Test quantum transformer functionality"""
    print("\n🧪 Testing quantum transformer functionality...")
    
    try:
        from quantum_intelligence.services.neural.transformers import QuantumTransformerLearningPathOptimizer
        
        # Create model
        model = QuantumTransformerLearningPathOptimizer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            max_seq_length=50
        )
        
        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        # Validate outputs
        assert 'difficulty_prediction' in outputs
        assert 'engagement_prediction' in outputs
        assert 'learning_velocity_prediction' in outputs
        assert 'next_content_prediction' in outputs
        
        # Check output shapes
        assert outputs['difficulty_prediction'].shape == (batch_size, 7)
        assert outputs['engagement_prediction'].shape == (batch_size, 1)
        assert outputs['next_content_prediction'].shape == (batch_size, 1000)
        
        print("✅ Quantum transformer forward pass works")
        
        # Test path optimization
        user_profile = {
            'exploration_factor': 1.2,
            'difficulty_preference': 0.6
        }
        
        path_result = model.predict_optimal_path(
            input_ids[:1],  # Single sequence
            user_profile,
            path_length=5
        )
        
        assert 'optimal_path' in path_result
        assert 'path_metrics' in path_result
        assert 'confidence_score' in path_result
        assert len(path_result['optimal_path']) == 5
        
        print("✅ Quantum transformer path optimization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum transformer functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_transformer_engine():
    """Test transformer engine functionality"""
    print("\n🧪 Testing transformer engine...")
    
    try:
        from quantum_intelligence.services.neural.transformers import QuantumTransformerEngine
        
        engine = QuantumTransformerEngine()
        
        # Test initialization
        init_result = await engine.initialize_models()
        
        assert init_result['status'] == 'success'
        assert 'models_initialized' in init_result
        assert 'total_parameters' in init_result
        
        print("✅ Transformer engine initialization works")
        
        # Test learning path optimization
        user_profile = {
            'path_length': 8,
            'difficulty_preference': 2,
            'exploration_factor': 1.0
        }
        
        current_sequence = [1, 5, 10, 15, 20]
        
        optimization_result = await engine.optimize_learning_path(
            user_id="test_user",
            current_sequence=current_sequence,
            user_profile=user_profile,
            optimization_type='quantum'
        )
        
        assert optimization_result['status'] == 'success'
        assert 'optimal_path' in optimization_result
        assert 'path_metrics' in optimization_result
        assert optimization_result['user_id'] == "test_user"
        
        print("✅ Transformer engine optimization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graph_neural_networks():
    """Test graph neural network functionality"""
    print("\n🧪 Testing graph neural networks...")
    
    try:
        from quantum_intelligence.services.neural.graph_networks import GraphNeuralKnowledgeNetwork
        
        # Create model
        model = GraphNeuralKnowledgeNetwork(
            num_concepts=100,
            concept_dim=64,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )
        
        # Test forward pass
        batch_size = 2
        num_concepts = 10
        concept_ids = torch.randint(0, 100, (batch_size, num_concepts))
        adjacency_matrix = torch.rand(num_concepts, num_concepts)
        adjacency_matrix = (adjacency_matrix > 0.5).float()  # Binary adjacency
        
        # Optional user features
        user_features = torch.rand(batch_size, 64)
        
        with torch.no_grad():
            outputs = model(concept_ids, adjacency_matrix, user_features)
        
        # Validate outputs
        assert 'concept_representations' in outputs
        assert 'difficulty_predictions' in outputs
        assert 'relationship_predictions' in outputs
        assert 'prerequisite_predictions' in outputs
        assert 'mastery_predictions' in outputs
        
        # Check output shapes
        assert outputs['concept_representations'].shape == (batch_size, num_concepts, 128)
        assert outputs['difficulty_predictions'].shape == (batch_size, num_concepts, 7)
        assert outputs['mastery_predictions'].shape == (batch_size, num_concepts)
        
        print("✅ Graph neural network forward pass works")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graph_neural_engine():
    """Test graph neural engine functionality"""
    print("\n🧪 Testing graph neural engine...")
    
    try:
        from quantum_intelligence.services.neural.graph_networks import GraphNeuralEngine
        
        engine = GraphNeuralEngine()
        
        # Test initialization
        init_result = await engine.initialize_networks()
        
        assert init_result['status'] == 'success'
        assert 'networks_initialized' in init_result
        assert 'total_parameters' in init_result
        
        print("✅ Graph neural engine initialization works")
        
        # Test concept relationship analysis
        concept_ids = [1, 5, 10, 15, 20]
        user_profile = {
            'learning_velocity': 0.7,
            'difficulty_preference': 0.6,
            'curiosity_index': 0.8,
            'attention_span': 45
        }
        
        analysis_result = await engine.analyze_concept_relationships(
            concept_ids=concept_ids,
            user_profile=user_profile
        )
        
        assert analysis_result['status'] == 'success'
        assert 'analysis' in analysis_result
        
        analysis = analysis_result['analysis']
        assert 'concept_ids' in analysis
        assert 'difficulty_predictions' in analysis
        assert 'relationship_matrix' in analysis
        assert 'prerequisite_matrix' in analysis
        assert 'insights' in analysis
        
        assert len(analysis['difficulty_predictions']) == len(concept_ids)
        assert len(analysis['relationship_matrix']) == len(concept_ids)
        
        print("✅ Graph neural engine concept analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph neural engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_integration():
    """Test neural services integration"""
    print("\n🧪 Testing neural services integration...")
    
    try:
        # Test importing from main services module
        from quantum_intelligence.services.neural import (
            QuantumTransformerEngine,
            GraphNeuralEngine,
            QuantumTransformerLearningPathOptimizer,
            GraphNeuralKnowledgeNetwork
        )
        
        print("✅ Neural services can be imported from main neural module")
        
        # Test that core components still work
        from quantum_intelligence import (
            QuantumLearningMode,
            QuantumState,
            IntelligenceLevel
        )
        
        print("✅ Core components still work with neural extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_combined_neural_workflow():
    """Test combined neural workflow"""
    print("\n🧪 Testing combined neural workflow...")
    
    try:
        from quantum_intelligence.services.neural.transformers import QuantumTransformerEngine
        from quantum_intelligence.services.neural.graph_networks import GraphNeuralEngine
        
        # Initialize engines
        transformer_engine = QuantumTransformerEngine()
        graph_engine = GraphNeuralEngine()
        
        # Initialize models
        transformer_init = await transformer_engine.initialize_models()
        graph_init = await graph_engine.initialize_networks()
        
        assert transformer_init['status'] == 'success'
        assert graph_init['status'] == 'success'
        
        # Test workflow: graph analysis -> transformer optimization
        
        # 1. Analyze concept relationships
        concept_ids = [1, 2, 3, 4, 5]
        user_profile = {
            'learning_velocity': 0.7,
            'difficulty_preference': 0.5,
            'path_length': 6
        }
        
        graph_analysis = await graph_engine.analyze_concept_relationships(
            concept_ids=concept_ids,
            user_profile=user_profile
        )
        
        assert graph_analysis['status'] == 'success'
        
        # 2. Use graph insights for transformer optimization
        transformer_optimization = await transformer_engine.optimize_learning_path(
            user_id="test_user",
            current_sequence=concept_ids,
            user_profile=user_profile,
            optimization_type='quantum'
        )
        
        assert transformer_optimization['status'] == 'success'
        
        # Verify the workflow produced meaningful results
        assert 'optimal_path' in transformer_optimization
        assert 'insights' in graph_analysis['analysis']
        
        print("✅ Combined neural workflow works")
        
        return True
        
    except Exception as e:
        print(f"❌ Combined neural workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 3 neural architecture tests"""
    print("🚀 PHASE 3 NEURAL ARCHITECTURES SERVICE EXTRACTION - VALIDATION")
    print("=" * 75)
    
    tests = [
        ("Neural Services Import", test_neural_services_import),
        ("Quantum Transformer Functionality", test_quantum_transformer_functionality),
        ("Transformer Engine", test_transformer_engine),
        ("Graph Neural Networks", test_graph_neural_networks),
        ("Graph Neural Engine", test_graph_neural_engine),
        ("Neural Integration", test_neural_integration),
        ("Combined Neural Workflow", test_combined_neural_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*75}")
        print(f"🧪 {test_name}")
        print('='*75)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*75}")
    print(f"🏆 PHASE 3 NEURAL RESULTS: {passed}/{total} tests passed")
    print('='*75)
    
    if passed == total:
        print("🎉 PHASE 3 NEURAL ARCHITECTURES SERVICE EXTRACTION SUCCESSFUL!")
        print("""
✅ ACHIEVEMENTS:
• Quantum Transformer Learning Path Optimizer extracted and functional
• Advanced Graph Neural Networks for knowledge representation
• Transformer Engine with quantum-enhanced processing
• Graph Neural Engine with concept relationship analysis
• Full integration with existing personalization and analytics services

🚀 NEXT STEPS:
• Continue with remaining neural components (Memory Networks, RL, etc.)
• Extract Predictive Intelligence Service
• Implement comprehensive integration tests
• Validate production readiness
""")
        return True
    else:
        print("⚠️  Some Phase 3 neural tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
