#!/usr/bin/env python3
"""
Debug test for peer learning services
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_peer_learning_import():
    """Test importing peer learning services"""
    try:
        print("Testing peer learning imports...")
        
        from quantum_intelligence.services.collaborative.peer_learning import PeerCompatibilityAnalyzer
        print("✅ PeerCompatibilityAnalyzer imported successfully")
        
        from quantum_intelligence.services.collaborative.peer_learning import PeerMatchingNetwork
        print("✅ PeerMatchingNetwork imported successfully")
        
        from quantum_intelligence.services.collaborative.peer_learning import PeerTutoringSystem
        print("✅ PeerTutoringSystem imported successfully")
        
        from quantum_intelligence.services.collaborative.peer_learning import PeerLearningOptimizer
        print("✅ PeerLearningOptimizer imported successfully")
        
        # Test instantiation
        analyzer = PeerCompatibilityAnalyzer()
        print("✅ PeerCompatibilityAnalyzer instantiated successfully")
        
        # Test basic functionality
        sample_user1 = {
            'user_id': 'user1',
            'learning_style': {'visual': 0.8, 'auditory': 0.3},
            'personality': {'openness': 0.8, 'conscientiousness': 0.7},
            'availability': {'09:00': True, '10:00': True},
            'communication_style': {'directness': 0.7},
            'motivation': {'intrinsic_motivation': 0.8}
        }
        
        sample_user2 = {
            'user_id': 'user2',
            'learning_style': {'visual': 0.6, 'auditory': 0.7},
            'personality': {'openness': 0.7, 'conscientiousness': 0.8},
            'availability': {'09:00': True, '11:00': True},
            'communication_style': {'directness': 0.6},
            'motivation': {'intrinsic_motivation': 0.7}
        }
        
        result = await analyzer.analyze_peer_compatibility(sample_user1, sample_user2)
        print(f"✅ Compatibility analysis result: {result['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_peer_learning_import()
    if success:
        print("🎉 Peer learning services working correctly!")
    else:
        print("💥 Peer learning services have issues")

if __name__ == "__main__":
    asyncio.run(main())
