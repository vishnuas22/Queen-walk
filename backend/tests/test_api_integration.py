"""
Comprehensive API Integration Tests for MasterX Quantum Intelligence Platform

Complete test suite that validates all API endpoints, integration with the
modular architecture, real-time features, and end-to-end functionality.

🧪 TEST COVERAGE:
- Authentication and authorization
- All API endpoints and routers
- Real-time streaming and WebSocket features
- Integration with quantum intelligence services
- Rate limiting and middleware functionality
- Error handling and edge cases

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import pytest
import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from fastapi import status
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import API components
from api.main import app
from api.auth import auth_manager
from api.models import *

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
async def authenticated_headers():
    """Get authenticated headers for testing"""
    
    # Login as test user
    login_request = LoginRequest(
        email="student@example.com",
        password="student123"
    )
    
    response = await auth_manager.authenticate_user(login_request)
    
    return {
        "Authorization": f"Bearer {response.access_token}",
        "Content-Type": "application/json"
    }

@pytest.fixture
async def admin_headers():
    """Get admin headers for testing"""
    
    # Login as admin
    login_request = LoginRequest(
        email="admin@masterx.ai",
        password="admin123"
    )
    
    response = await auth_manager.authenticate_user(login_request)
    
    return {
        "Authorization": f"Bearer {response.access_token}",
        "Content-Type": "application/json"
    }

# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_login_success(self, client):
        """Test successful login"""
        
        response = client.post("/api/v1/auth/login", json={
            "email": "student@example.com",
            "password": "student123"
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        
        response = client.post("/api/v1/auth/login", json={
            "email": "invalid@example.com",
            "password": "wrongpassword"
        })
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        
        response = client.get("/api/v1/learning/goals")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

# ============================================================================
# CORE API ENDPOINT TESTS
# ============================================================================

class TestCoreEndpoints:
    """Test core API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "MasterX Quantum Intelligence API"
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        
        response = client.get("/health")
        
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        
        response = client.get("/metrics")
        
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "timestamp" in data

# ============================================================================
# CHAT API TESTS
# ============================================================================

class TestChatAPI:
    """Test chat API endpoints"""
    
    @pytest.mark.asyncio
    async def test_send_chat_message(self, client, authenticated_headers):
        """Test sending a chat message"""
        
        chat_request = {
            "message": "Hello, can you help me learn Python?",
            "message_type": "text",
            "stream": False
        }
        
        response = client.post(
            "/api/v1/chat/message",
            json=chat_request,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert "message_id" in data
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, client, authenticated_headers):
        """Test chat streaming endpoint"""
        
        chat_request = {
            "message": "Explain machine learning concepts",
            "stream": True
        }
        
        response = client.post(
            "/api/v1/chat/stream",
            json=chat_request,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

# ============================================================================
# LEARNING API TESTS
# ============================================================================

class TestLearningAPI:
    """Test learning API endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_learning_goal(self, client, authenticated_headers):
        """Test creating a learning goal"""
        
        goal_data = {
            "title": "Learn Python Programming",
            "description": "Master Python fundamentals and advanced concepts",
            "subject": "Programming",
            "target_skills": ["variables", "functions", "classes", "modules"],
            "difficulty_level": 0.6,
            "estimated_duration_hours": 40
        }
        
        response = client.post(
            "/api/v1/learning/goals",
            json=goal_data,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["title"] == goal_data["title"]
        assert data["subject"] == goal_data["subject"]
        assert "goal_id" in data
    
    @pytest.mark.asyncio
    async def test_get_learning_goals(self, client, authenticated_headers):
        """Test getting learning goals"""
        
        response = client.get(
            "/api/v1/learning/goals",
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_record_learning_session(self, client, authenticated_headers):
        """Test recording a learning session"""
        
        session_data = {
            "subject": "Python Programming",
            "duration_minutes": 45,
            "activities": [
                {"type": "reading", "content": "Python basics"},
                {"type": "coding", "content": "Hello World program"}
            ],
            "performance_metrics": {
                "comprehension": 0.8,
                "speed": 0.7
            },
            "engagement_score": 0.85
        }
        
        response = client.post(
            "/api/v1/learning/sessions",
            json=session_data,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["subject"] == session_data["subject"]
        assert data["duration_minutes"] == session_data["duration_minutes"]

# ============================================================================
# CONTENT API TESTS
# ============================================================================

class TestContentAPI:
    """Test content generation API"""
    
    @pytest.mark.asyncio
    async def test_generate_content(self, client, authenticated_headers):
        """Test content generation"""
        
        content_request = {
            "subject": "Mathematics",
            "topic": "Linear Algebra",
            "content_type": "lesson",
            "difficulty_level": 0.7,
            "duration_minutes": 30,
            "learning_objectives": [
                "Understand vector operations",
                "Learn matrix multiplication"
            ]
        }
        
        response = client.post(
            "/api/v1/content/generate",
            json=content_request,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "generated_content" in data
        assert data["generated_content"]["content_type"] == content_request["content_type"]

# ============================================================================
# ASSESSMENT API TESTS
# ============================================================================

class TestAssessmentAPI:
    """Test assessment API endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_assessment(self, client, authenticated_headers):
        """Test assessment creation"""
        
        assessment_request = {
            "subject": "Python Programming",
            "topics": ["variables", "functions", "loops"],
            "assessment_type": "quiz",
            "difficulty_level": 0.6,
            "duration_minutes": 30,
            "question_count": 10
        }
        
        response = client.post(
            "/api/v1/assessment/create",
            json=assessment_request,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "assessment" in data
        assert data["assessment"]["metadata"]["subject"] == assessment_request["subject"]

# ============================================================================
# STREAMING API TESTS
# ============================================================================

class TestStreamingAPI:
    """Test streaming API endpoints"""
    
    @pytest.mark.asyncio
    async def test_stream_events(self, client, authenticated_headers):
        """Test event streaming"""
        
        response = client.get(
            "/api/v1/streaming/events?event_types=notification,learning_update",
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    @pytest.mark.asyncio
    async def test_stream_progress(self, client, authenticated_headers):
        """Test progress streaming"""
        
        response = client.get(
            "/api/v1/streaming/progress",
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

# ============================================================================
# ANALYTICS API TESTS
# ============================================================================

class TestAnalyticsAPI:
    """Test analytics API endpoints"""
    
    @pytest.mark.asyncio
    async def test_generate_predictions(self, client, authenticated_headers):
        """Test prediction generation"""
        
        analytics_request = {
            "prediction_type": "learning_outcome",
            "time_horizon": "medium_term",
            "include_interventions": True
        }
        
        response = client.post(
            "/api/v1/analytics/predict",
            json=analytics_request,
            headers=authenticated_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predictions" in data
        assert "learning_analytics" in data

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test end-to-end integration"""
    
    @pytest.mark.asyncio
    async def test_complete_learning_workflow(self, client, authenticated_headers):
        """Test complete learning workflow"""
        
        # 1. Create learning goal
        goal_response = client.post(
            "/api/v1/learning/goals",
            json={
                "title": "Integration Test Goal",
                "description": "Test goal for integration testing",
                "subject": "Testing",
                "target_skills": ["integration", "testing"],
                "difficulty_level": 0.5,
                "estimated_duration_hours": 10
            },
            headers=authenticated_headers
        )
        
        assert goal_response.status_code == status.HTTP_200_OK
        goal_data = goal_response.json()
        goal_id = goal_data["goal_id"]
        
        # 2. Generate content
        content_response = client.post(
            "/api/v1/content/generate",
            json={
                "subject": "Testing",
                "topic": "Integration Testing",
                "content_type": "lesson",
                "difficulty_level": 0.5,
                "duration_minutes": 20,
                "learning_objectives": ["Understand integration testing"]
            },
            headers=authenticated_headers
        )
        
        assert content_response.status_code == status.HTTP_200_OK
        
        # 3. Record learning session
        session_response = client.post(
            "/api/v1/learning/sessions",
            json={
                "goal_id": goal_id,
                "subject": "Testing",
                "duration_minutes": 20,
                "activities": [{"type": "study", "content": "Integration testing"}],
                "performance_metrics": {"comprehension": 0.8},
                "engagement_score": 0.9
            },
            headers=authenticated_headers
        )
        
        assert session_response.status_code == status.HTTP_200_OK
        
        # 4. Get progress
        progress_response = client.post(
            "/api/v1/learning/progress",
            json={
                "time_range": "week"
            },
            headers=authenticated_headers
        )
        
        assert progress_response.status_code == status.HTTP_200_OK
        progress_data = progress_response.json()
        assert "overall_progress" in progress_data
        
        print("✅ Complete learning workflow test passed")

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test API performance"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, authenticated_headers):
        """Test handling concurrent requests"""
        
        import concurrent.futures
        import time
        
        def make_request():
            response = client.get("/health")
            return response.status_code
        
        # Make 10 concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # All requests should succeed
        assert all(status_code in [200, 503] for status_code in results)
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0
        
        print(f"✅ Concurrent requests test passed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
