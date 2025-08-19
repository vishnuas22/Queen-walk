"""
API Routers for MasterX Quantum Intelligence Platform

Collection of API routers that provide comprehensive REST endpoints
for all quantum intelligence services and platform functionality.

🌐 AVAILABLE ROUTERS:
- chat_router: Intelligent chat and conversation management
- learning_router: Learning goals and progress tracking
- personalization_router: User profiling and personalization
- analytics_router: Predictive analytics and insights
- content_router: Content generation and management
- assessment_router: Assessment creation and evaluation
- streaming_router: Real-time streaming capabilities
- websocket_router: WebSocket connections and real-time features

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

from . import (
    chat_router,
    learning_router,
    personalization_router,
    analytics_router,
    content_router,
    assessment_router,
    streaming_router,
    websocket_router
)

__all__ = [
    "chat_router",
    "learning_router", 
    "personalization_router",
    "analytics_router",
    "content_router",
    "assessment_router",
    "streaming_router",
    "websocket_router"
]
