"""
Simplified Authentication Manager for MasterX Quantum Intelligence Platform

DEVELOPMENT MODE: Authentication completely disabled for development purposes.
All authentication checks are bypassed to allow direct API access.

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Development Mode
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

# Configure logging
logger = logging.getLogger(__name__)

# Development mode - authentication disabled
logger.info("🔐 Authentication Manager initialized (DEVELOPMENT MODE - AUTH DISABLED)")

# Mock security scheme (not used but needed for compatibility)
security = HTTPBearer(auto_error=False)

def get_current_user(token: Optional[str] = Depends(security)):
    """
    Development mode: Always return a mock user
    In production, this would validate the JWT token and return the actual user
    """
    from .models import UserProfile, UserRole
    
    # Return a mock user for development
    return UserProfile(
        user_id="dev_user_001",
        name="Developer User",
        username="developer",
        email="dev@masterx.ai",
        role=UserRole.STUDENT,
        created_at="2024-01-01T00:00:00Z"
    )

def require_permission(permission: str):
    """
    Development mode: Always allow access
    In production, this would check user permissions
    """
    def permission_checker(current_user = Depends(get_current_user)):
        # In development mode, always allow access
        return current_user
    
    return permission_checker

# Export the functions that other modules expect
__all__ = ["get_current_user", "require_permission"]
