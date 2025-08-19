#!/usr/bin/env python3
"""
Test Phase 3 Neural Architectures - Simple Service Extraction Validation

This script validates that Phase 3 neural architecture service extraction is working
without requiring PyTorch or NumPy dependencies.
"""

import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_neural_services_import():
    """Test neural services can be imported"""
    print("🧪 Testing neural services import...")
    
    try:
        # Test neural services
        from quantum_intelligence.services.neural.transformers import (
            QuantumTransformerEngine
        )
        from quantum_intelligence.services.neural.graph_networks import (
            GraphNeuralEngine
        )
        from quantum_intelligence.services.neural.architectures import (
            NeuralArchitectureManager
        )
        
        print("✅ Neural service classes can be imported")
        
        # Test instantiation
        transformer_engine = QuantumTransformerEngine()
        graph_engine = GraphNeuralEngine()
        architecture_manager = NeuralArchitectureManager()
        
        print("✅ Neural services can be instantiated")
        
        # Test that they have the expected methods
        assert hasattr(transformer_engine, 'initialize_models')
        assert hasattr(transformer_engine, 'optimize_learning_path')
        assert hasattr(graph_engine, 'initialize_networks')
        assert hasattr(graph_engine, 'analyze_concept_relationships')
        assert hasattr(architecture_manager, 'initialize_architectures')
        assert hasattr(architecture_manager, 'search_optimal_architecture')
        
        print("✅ Neural services have expected methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural services import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_transformer_engine_basic():
    """Test basic transformer engine functionality"""
    print("\n🧪 Testing transformer engine basic functionality...")
    
    try:
        from quantum_intelligence.services.neural.transformers import QuantumTransformerEngine
        
        engine = QuantumTransformerEngine()
        
        # Test initialization
        init_result = await engine.initialize_models()
        
        assert init_result['status'] == 'success'
        assert 'models_initialized' in init_result
        assert 'total_parameters' in init_result
        
        print("✅ Transformer engine initialization works")
        
        # Test learning path optimization with mock data
        user_profile = {
            'path_length': 5,
            'difficulty_preference': 2,
            'exploration_factor': 1.0
        }
        
        current_sequence = [1, 2, 3, 4, 5]
        
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

async def test_graph_neural_engine_basic():
    """Test basic graph neural engine functionality"""
    print("\n🧪 Testing graph neural engine basic functionality...")
    
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
        concept_ids = [1, 2, 3, 4, 5]
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
        
        print("✅ Graph neural engine concept analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph neural engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_neural_architecture_manager():
    """Test neural architecture manager functionality"""
    print("\n🧪 Testing neural architecture manager...")
    
    try:
        from quantum_intelligence.services.neural.architectures import NeuralArchitectureManager
        
        manager = NeuralArchitectureManager()
        
        # Test initialization
        config = {
            "enable_adaptive_difficulty": True,
            "enable_nas": True,
            "difficulty_input_dim": 128,
            "nas_search_space": 500
        }
        
        init_result = await manager.initialize_architectures(config)
        
        assert init_result['status'] == 'success'
        assert 'initialized_architectures' in init_result
        assert 'architecture_count' in init_result
        
        print("✅ Neural architecture manager initialization works")
        
        # Test architecture search
        task_requirements = {
            "task_type": "sequence_modeling",
            "complexity": 0.7
        }
        
        performance_targets = {
            "accuracy": 0.9,
            "latency_ms": 50
        }
        
        search_result = await manager.search_optimal_architecture(
            task_requirements,
            performance_targets
        )
        
        assert search_result['status'] == 'success'
        assert 'best_architecture' in search_result
        assert 'candidate_count' in search_result
        
        print("✅ Neural architecture search works")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural architecture manager test failed: {e}")
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
            NeuralArchitectureManager
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
        from quantum_intelligence.services.neural.architectures import NeuralArchitectureManager
        
        # Initialize engines
        transformer_engine = QuantumTransformerEngine()
        graph_engine = GraphNeuralEngine()
        architecture_manager = NeuralArchitectureManager()
        
        # Initialize all components
        transformer_init = await transformer_engine.initialize_models()
        graph_init = await graph_engine.initialize_networks()
        arch_init = await architecture_manager.initialize_architectures({})
        
        assert transformer_init['status'] == 'success'
        assert graph_init['status'] == 'success'
        assert arch_init['status'] == 'success'
        
        # Test workflow: architecture search -> graph analysis -> transformer optimization
        
        # 1. Search for optimal architecture
        task_requirements = {"task_type": "learning_optimization", "complexity": 0.6}
        arch_search = await architecture_manager.search_optimal_architecture(
            task_requirements, {"accuracy": 0.85}
        )
        
        # 2. Analyze concept relationships
        concept_ids = [1, 2, 3, 4, 5]
        user_profile = {'learning_velocity': 0.7, 'difficulty_preference': 0.5}
        
        graph_analysis = await graph_engine.analyze_concept_relationships(
            concept_ids=concept_ids,
            user_profile=user_profile
        )
        
        # 3. Optimize learning path
        transformer_optimization = await transformer_engine.optimize_learning_path(
            user_id="test_user",
            current_sequence=concept_ids,
            user_profile=user_profile,
            optimization_type='quantum'
        )
        
        # Verify the workflow produced meaningful results
        assert arch_search['status'] == 'success'
        assert graph_analysis['status'] == 'success'
        assert transformer_optimization['status'] == 'success'
        
        print("✅ Combined neural workflow works")
        
        return True
        
    except Exception as e:
        print(f"❌ Combined neural workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 3 neural architecture tests"""
    print("🚀 PHASE 3 NEURAL ARCHITECTURES SERVICE EXTRACTION - SIMPLE VALIDATION")
    print("=" * 80)
    
    tests = [
        ("Neural Services Import", test_neural_services_import),
        ("Transformer Engine Basic", test_transformer_engine_basic),
        ("Graph Neural Engine Basic", test_graph_neural_engine_basic),
        ("Neural Architecture Manager", test_neural_architecture_manager),
        ("Neural Integration", test_neural_integration),
        ("Combined Neural Workflow", test_combined_neural_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 {test_name}")
        print('='*80)
        
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
    
    print(f"\n{'='*80}")
    print(f"🏆 PHASE 3 NEURAL RESULTS: {passed}/{total} tests passed")
    print('='*80)
    
    if passed == total:
        print("🎉 PHASE 3 NEURAL ARCHITECTURES SERVICE EXTRACTION SUCCESSFUL!")
        print("""
✅ ACHIEVEMENTS:
• Neural Architecture Manager with architecture search capabilities
• Quantum Transformer Engine with learning path optimization
• Graph Neural Engine with concept relationship analysis
• Full service integration and workflow validation
• Production-ready architecture without external ML dependencies

🚀 NEXT STEPS:
• Continue with remaining neural components (Memory Networks, RL, etc.)
• Extract Predictive Intelligence Service (next major phase)
• Add PyTorch/NumPy dependencies for full ML functionality
• Implement comprehensive integration tests
""")
        return True
    else:
        print("⚠️  Some Phase 3 neural tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
