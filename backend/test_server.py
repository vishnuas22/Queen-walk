#!/usr/bin/env python3
"""
Phase 13 Test Server
Minimal FastAPI server for testing Phase 13 integration without full dependencies.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(__file__))

# Load environment variables from the correct path
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Import our API components
from api.auth import AuthManager
from api.utils import LLMIntegration
from api.models import LoginRequest, ChatRequest

# Initialize FastAPI app
app = FastAPI(
    title="MasterX Phase 13 Test API",
    description="Test server for Phase 13 Frontend Integration & Multi-LLM Enhancement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
auth_manager = AuthManager()
llm_integration = LLMIntegration()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MasterX Phase 13 Test API",
        "version": "1.0.0",
        "status": "operational",
        "phase": "Phase 13: Frontend Integration & Multi-LLM Enhancement"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Check LLM providers
    available_providers = [
        name for name, config in llm_integration.providers.items() 
        if config['available']
    ]
    
    return {
        "status": "healthy",
        "timestamp": "2025-07-18T22:55:00Z",
        "services": {
            "authentication": "operational",
            "llm_integration": "operational",
            "available_providers": available_providers
        },
        "phase13_features": {
            "environment_security": True,
            "multi_llm_integration": True,
            "intelligent_model_selection": True,
            "task_based_routing": True
        }
    }

@app.post("/api/v1/auth/login")
async def login(request: LoginRequest):
    """Login endpoint"""
    
    try:
        # Authenticate user (simplified for testing)
        if request.email == "student@example.com" and request.password == "student123":
            user_data = {
                "user_id": "student_001",
                "email": "student@example.com",
                "name": "Test Student",
                "role": "student"
            }
        elif request.email == "teacher@example.com" and request.password == "teacher123":
            user_data = {
                "user_id": "teacher_001",
                "email": "teacher@example.com",
                "name": "Test Teacher",
                "role": "teacher"
            }
        elif request.email == "admin@masterx.ai" and request.password == "admin123":
            user_data = {
                "user_id": "admin_001",
                "email": "admin@masterx.ai",
                "name": "Test Admin",
                "role": "admin"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate tokens
        access_token = auth_manager._generate_jwt_token(user_data)
        refresh_token = auth_manager._generate_refresh_token(user_data)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 86400,
            "user_info": user_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")

@app.post("/api/v1/chat/message")
async def send_chat_message(request: ChatRequest):
    """Send chat message endpoint"""
    
    try:
        # Analyze task type
        task_type = request.task_type or llm_integration._analyze_task_type(request.message, {})
        
        # Select best provider
        provider = request.provider or llm_integration._select_best_provider_for_task(task_type)
        
        if not provider:
            raise HTTPException(status_code=503, detail="No LLM providers available")
        
        # Get optimal model
        model = llm_integration._get_model_for_task(provider, task_type)
        
        # Generate mock response (for testing - replace with actual LLM call)
        mock_responses = {
            "reasoning": f"Let me analyze this step by step. For the question '{request.message}', I'll use logical reasoning to provide a comprehensive answer.",
            "coding": f"Here's a code solution for '{request.message}':\n\n```python\n# Example code\ndef solution():\n    return 'Hello, World!'\n```",
            "creative": f"Here's a creative response to '{request.message}': Once upon a time, in a world of infinite possibilities...",
            "fast": f"Quick answer: {request.message}",
            "multimodal": f"I can analyze various types of content for '{request.message}'. Please provide the media you'd like me to examine.",
            "general": f"Thank you for your question: '{request.message}'. I'm here to help with comprehensive, intelligent responses."
        }
        
        response_content = mock_responses.get(task_type, mock_responses["general"])
        
        # Generate suggestions based on task type
        suggestions = llm_integration._generate_suggestions(response_content, task_type)
        
        return {
            "session_id": request.session_id or f"session_{int(os.urandom(4).hex(), 16)}",
            "message_id": f"msg_{int(os.urandom(4).hex(), 16)}",
            "response": response_content,
            "suggestions": suggestions,
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "metadata": {
                "response_time": 0.5,
                "tokens_used": len(response_content.split()),
                "task_optimization": f"Optimized for {task_type} tasks using {provider}:{model}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/api/v1/chat/sessions")
async def get_chat_sessions():
    """Get chat sessions endpoint"""
    return {
        "sessions": [
            {
                "session_id": "session_001",
                "created_at": "2025-07-18T22:00:00Z",
                "last_message": "Hello, this is a test session",
                "message_count": 5
            }
        ]
    }

@app.get("/api/v1/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard endpoint"""
    return {
        "user_id": "test_user",
        "overview": {
            "total_study_time": 120.5,
            "goals_completed": 3,
            "current_streak": 7,
            "overall_progress": 68.5
        },
        "llm_usage": {
            "total_requests": 150,
            "provider_breakdown": {
                "groq": 80,
                "gemini": 70
            },
            "task_breakdown": {
                "reasoning": 40,
                "coding": 35,
                "creative": 25,
                "fast": 30,
                "general": 20
            }
        },
        "phase13_metrics": {
            "intelligent_routing_accuracy": 0.92,
            "average_response_time": 1.2,
            "provider_availability": 0.98
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MasterX Phase 13 Test Server...")
    print("📊 Available at: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
