"""
MasterX Quantum Intelligence Platform - Main API Router

Comprehensive REST API that exposes all quantum intelligence services implemented
in Phases 1-11, providing seamless integration with the modular architecture
through the master orchestrator and API gateway.

🌐 API INTEGRATION CAPABILITIES:
- Complete integration with all quantum intelligence services
- Real-time streaming support with WebSocket connections
- User session management with personalization engine
- Learning progress tracking with predictive analytics
- Authentication and rate limiting through API gateway
- Comprehensive monitoring and logging

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import quantum intelligence orchestration
from quantum_intelligence.orchestration import (
    MasterXOrchestrationPlatform,
    ServiceType,
    ServiceRequest,
    IntegrationMessage,
    MessageType,
    CommunicationPattern,
    start_masterx_platform
)

# Import API routers
from .routers import (
    chat_router,
    learning_router,
    personalization_router,
    analytics_router,
    content_router,
    assessment_router,
    streaming_router,
    websocket_router
)

# Import authentication and middleware
from .auth import get_current_user  # AuthManager removed for development
from .middleware import RequestLoggingMiddleware, RateLimitingMiddleware
from .models import *
from .utils import APIResponseHandler, WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestration platform instance
orchestration_platform: Optional[MasterXOrchestrationPlatform] = None

# ============================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestration_platform
    
    try:
        logger.info("🚀 Starting MasterX Quantum Intelligence API Platform...")
        
        # Initialize orchestration platform
        orchestration_platform = await start_masterx_platform(
            config={
                'api_gateway': {
                    'authentication': {
                        'jwt_secret': os.getenv('JWT_SECRET', 'masterx-secret-key'),
                        'jwt_expiration_hours': 24
                    },
                    'rate_limiting': {
                        'default_requests_per_minute': 1000,
                        'burst_size': 50
                    }
                },
                'monitoring': {
                    'health_checks': {
                        'default_timeout_seconds': 5.0,
                        'default_interval_seconds': 30
                    }
                }
            },
            api_gateway_port=8001  # Internal orchestration port
        )
        
        # Register quantum intelligence services
        await _register_quantum_services()
        
        logger.info("✅ MasterX Quantum Intelligence API Platform started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start API platform: {e}")
        raise
    finally:
        # Cleanup
        if orchestration_platform:
            await orchestration_platform.shutdown()
        logger.info("🛑 MasterX Quantum Intelligence API Platform shutdown complete")

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="MasterX Quantum Intelligence API",
    description="Comprehensive REST API for the MasterX Quantum Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitingMiddleware)

# ============================================================================
# AUTHENTICATION SETUP
# ============================================================================

# Authentication disabled for development
# auth_manager = AuthManager()
# security = HTTPBearer()

# ============================================================================
# API ROUTERS
# ============================================================================

# Include all API routers
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(learning_router.router, prefix="/api/v1/learning", tags=["Learning"])
app.include_router(personalization_router.router, prefix="/api/v1/personalization", tags=["Personalization"])
app.include_router(analytics_router.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(content_router.router, prefix="/api/v1/content", tags=["Content"])
app.include_router(assessment_router.router, prefix="/api/v1/assessment", tags=["Assessment"])
app.include_router(streaming_router.router, prefix="/api/v1/streaming", tags=["Streaming"])
app.include_router(websocket_router.router, prefix="/api/v1/ws", tags=["WebSocket"])

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MasterX Quantum Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        if not orchestration_platform:
            raise HTTPException(status_code=503, detail="Orchestration platform not available")
        
        # Get platform status
        platform_status = await orchestration_platform.get_platform_status()
        
        # Check if all critical services are running
        components = platform_status.get('components', {})
        all_healthy = all(
            comp.get('status') == 'active' 
            for comp in components.values()
        )
        
        health_status = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "running": platform_status.get('platform', {}).get('is_running', False),
                "uptime_seconds": platform_status.get('platform', {}).get('uptime_seconds', 0),
                "services_registered": platform_status.get('platform', {}).get('metrics', {}).get('total_services_registered', 0)
            },
            "components": {
                name: {
                    "status": comp.get('status', 'unknown'),
                    "services": comp.get('services_registered', 0) if 'services_registered' in comp else comp.get('endpoints_registered', 0)
                }
                for name, comp in components.items()
            },
            "system_health": platform_status.get('system_health', {}).get('overall_health', 'unknown')
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive platform metrics"""
    try:
        if not orchestration_platform:
            raise HTTPException(status_code=503, detail="Orchestration platform not available")
        
        platform_status = await orchestration_platform.get_platform_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "platform_metrics": platform_status.get('platform', {}).get('metrics', {}),
            "component_metrics": {
                name: {
                    "requests_processed": comp.get('requests_processed', 0),
                    "services_registered": comp.get('services_registered', 0),
                    "status": comp.get('status', 'unknown')
                }
                for name, comp in platform_status.get('components', {}).items()
            },
            "system_resources": platform_status.get('system_health', {}).get('system_resources', {})
        }
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get detailed platform status"""
    try:
        if not orchestration_platform:
            return {
                "status": "initializing",
                "message": "Orchestration platform is starting up",
                "timestamp": datetime.now().isoformat()
            }
        
        platform_status = await orchestration_platform.get_platform_status()
        
        return {
            "status": "operational" if platform_status.get('platform', {}).get('is_running') else "degraded",
            "platform": platform_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _register_quantum_services():
    """Register all quantum intelligence services with the orchestration platform"""
    try:
        # Register Quantum Intelligence Engine
        await orchestration_platform.register_quantum_service(
            service_name="quantum_intelligence_engine",
            service_type=ServiceType.QUANTUM_INTELLIGENCE,
            host="localhost",
            port=8002,
            capabilities=[
                "learning_path_optimization",
                "content_generation",
                "assessment_creation",
                "quantum_processing"
            ]
        )
        
        # Register Personalization Engine
        await orchestration_platform.register_quantum_service(
            service_name="personalization_engine",
            service_type=ServiceType.PERSONALIZATION,
            host="localhost",
            port=8003,
            capabilities=[
                "user_profiling",
                "learning_dna_analysis",
                "adaptive_content",
                "behavior_tracking"
            ]
        )
        
        # Register Predictive Analytics Engine
        await orchestration_platform.register_quantum_service(
            service_name="predictive_analytics_engine",
            service_type=ServiceType.PREDICTIVE_ANALYTICS,
            host="localhost",
            port=8004,
            capabilities=[
                "outcome_prediction",
                "intervention_detection",
                "learning_analytics",
                "performance_forecasting"
            ]
        )
        
        logger.info("✅ All quantum intelligence services registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register quantum services: {e}")
        raise

def get_orchestration_platform() -> MasterXOrchestrationPlatform:
    """Dependency to get the orchestration platform"""
    if not orchestration_platform:
        raise HTTPException(status_code=503, detail="Orchestration platform not available")
    return orchestration_platform

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
