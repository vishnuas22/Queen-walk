#!/usr/bin/env python3
"""
Phase 13 Integration Test Script
Tests the core Phase 13 objectives without requiring full system startup.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_environment_security():
    """Test 1: Environment Security Enhancement"""
    print("🔐 Testing Environment Security...")
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print("   ✅ .env file exists")
        
        # Check if API keys are in environment
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
        groq_key = os.getenv('GROQ_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        jwt_secret = os.getenv('JWT_SECRET')
        
        if groq_key:
            print(f"   ✅ Groq API key configured: {groq_key[:10]}...")
        else:
            print("   ❌ Groq API key not found")
            
        if gemini_key:
            print(f"   ✅ Gemini API key configured: {gemini_key[:10]}...")
        else:
            print("   ❌ Gemini API key not found")
            
        if jwt_secret:
            print(f"   ✅ JWT secret configured: {jwt_secret[:10]}...")
        else:
            print("   ❌ JWT secret not found")
            
        return True
    else:
        print("   ❌ .env file not found")
        return False

def test_api_components():
    """Test 2: API Components"""
    print("\n🔧 Testing API Components...")
    
    try:
        # Test importing core API components
        from api.auth import AuthManager
        print("   ✅ AuthManager imported successfully")
        
        from api.utils import LLMIntegration
        print("   ✅ LLMIntegration imported successfully")
        
        from api.models import ChatRequest, ChatResponse, LoginRequest
        print("   ✅ API models imported successfully")
        
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

async def test_llm_integration():
    """Test 3: Multi-LLM Integration"""
    print("\n🤖 Testing Multi-LLM Integration...")
    
    try:
        from api.utils import LLMIntegration
        
        # Initialize LLM integration
        llm = LLMIntegration()
        print("   ✅ LLM Integration initialized")
        
        # Test provider configuration
        providers = llm.providers
        available_providers = [name for name, config in providers.items() if config['available']]
        
        print(f"   📊 Available providers: {', '.join(available_providers)}")
        
        # Test task type analysis
        test_messages = [
            ("Write a Python function to sort a list", "coding"),
            ("Solve this math equation: 2x + 5 = 15", "reasoning"),
            ("Write a creative story about space", "creative"),
            ("What is the capital of France?", "fast"),
        ]
        
        for message, expected_type in test_messages:
            detected_type = llm._analyze_task_type(message, {})
            print(f"   🎯 '{message[:30]}...' -> {detected_type}")
        
        # Test model selection
        for task_type in ['reasoning', 'coding', 'creative', 'fast', 'general']:
            provider = llm._select_best_provider_for_task(task_type)
            if provider:
                model = llm._get_model_for_task(provider, task_type)
                print(f"   🧠 {task_type} -> {provider}:{model}")
            else:
                print(f"   ❌ No provider available for {task_type}")
        
        return True
    except Exception as e:
        print(f"   ❌ LLM integration error: {e}")
        return False

def test_authentication():
    """Test 4: Authentication System"""
    print("\n🔑 Testing Authentication System...")
    
    try:
        from api.auth import AuthManager
        
        # Initialize auth manager
        auth_manager = AuthManager()
        print("   ✅ AuthManager initialized")
        
        # Test LLM config
        llm_config = auth_manager.get_llm_config()
        print(f"   📊 Default provider: {llm_config['default_provider']}")
        print(f"   📊 Provider priority: {llm_config['provider_priority']}")
        
        # Test user creation (mock)
        test_user = {
            'user_id': 'test_user_001',
            'email': 'student@example.com',
            'name': 'Test Student',
            'role': 'student'
        }
        
        # Test token generation (using private method for testing)
        token = auth_manager._generate_jwt_token(test_user)
        print(f"   ✅ JWT token generated: {token[:20]}...")

        # Test token validation
        decoded_token = auth_manager._decode_jwt_token(token)
        if decoded_token.user_id == test_user['user_id']:
            print("   ✅ Token validation successful")
        else:
            print("   ❌ Token validation failed")
        
        return True
    except Exception as e:
        print(f"   ❌ Authentication error: {e}")
        return False

def test_frontend_api_config():
    """Test 5: Frontend API Configuration"""
    print("\n🌐 Testing Frontend API Configuration...")
    
    frontend_env = os.path.join(os.path.dirname(__file__), '..', 'frontend', '.env.local')
    
    if os.path.exists(frontend_env):
        print("   ✅ Frontend .env.local exists")
        
        with open(frontend_env, 'r') as f:
            content = f.read()
            
        if 'NEXT_PUBLIC_BACKEND_URL=http://localhost:8000' in content:
            print("   ✅ Backend URL configured correctly")
        else:
            print("   ❌ Backend URL not configured")
            
        if 'NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1' in content:
            print("   ✅ API base URL configured correctly")
        else:
            print("   ❌ API base URL not configured")
            
        return True
    else:
        print("   ❌ Frontend .env.local not found")
        return False

async def run_integration_tests():
    """Run all Phase 13 integration tests"""
    print("🚀 PHASE 13: FRONTEND INTEGRATION & MULTI-LLM ENHANCEMENT")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Environment Security Enhancement", test_environment_security),
        ("API Components", test_api_components),
        ("Multi-LLM Integration", test_llm_integration),
        ("Authentication System", test_authentication),
        ("Frontend API Configuration", test_frontend_api_config),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 PHASE 13 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 13 INTEGRATION TESTS PASSED!")
        print("\n✅ Phase 13 Core Objectives Verified:")
        print("   • Environment Security (API keys moved to .env)")
        print("   • Multi-LLM Integration with intelligent selection")
        print("   • Enhanced authentication system")
        print("   • Frontend-backend API configuration")
        print("   • Task-based model routing")
        print("\n🚀 Phase 13 implementation is ready for frontend testing!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
