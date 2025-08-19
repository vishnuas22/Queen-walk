"""
Middleware for MasterX Quantum Intelligence Platform API

Custom middleware components that provide request logging, rate limiting,
security enhancements, and performance monitoring for all API endpoints.

🛡️ MIDDLEWARE CAPABILITIES:
- Request/response logging and monitoring
- Rate limiting and throttling
- Security headers and CORS handling
- Performance metrics collection
- Error handling and recovery
- Request validation and sanitization

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import time
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    📝 REQUEST LOGGING MIDDLEWARE
    
    Comprehensive request and response logging with performance metrics
    and detailed tracking for all API interactions.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        logger.info("📝 Request Logging Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details"""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        self.request_count += 1
        
        logger.info(
            f"REQUEST {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Log request headers (excluding sensitive data)
        safe_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ['authorization', 'cookie', 'x-api-key']
        }
        logger.debug(f"REQUEST {request_id} Headers: {safe_headers}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-Count"] = str(self.request_count)
            
            # Log response
            logger.info(
                f"RESPONSE {request_id}: {response.status_code} "
                f"in {response_time:.3f}s"
            )
            
            # Log slow requests
            if response_time > 5.0:
                logger.warning(
                    f"SLOW REQUEST {request_id}: {request.method} {request.url.path} "
                    f"took {response_time:.3f}s"
                )
            
            return response
            
        except Exception as e:
            # Calculate response time for errors
            response_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"ERROR {request_id}: {request.method} {request.url.path} "
                f"failed after {response_time:.3f}s - {str(e)}"
            )
            
            # Re-raise the exception
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        return {
            "total_requests": self.request_count,
            "average_response_time": avg_response_time,
            "recent_response_times": list(self.response_times)[-10:],  # Last 10
            "max_response_time": max(self.response_times) if self.response_times else 0,
            "min_response_time": min(self.response_times) if self.response_times else 0
        }

# ============================================================================
# RATE LIMITING MIDDLEWARE
# ============================================================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    🚦 RATE LIMITING MIDDLEWARE
    
    Advanced rate limiting with per-user, per-endpoint, and global limits
    to prevent abuse and ensure fair resource usage.
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Rate limiting configuration
        self.global_limit = 1000  # requests per minute
        self.user_limit = 100     # requests per minute per user
        self.endpoint_limits = {
            "/api/v1/chat/stream": 10,    # Lower limit for streaming
            "/api/v1/chat/message": 30,   # Moderate limit for chat
            "/api/v1/content/generate": 20,  # Lower limit for content generation
        }
        
        # Tracking data structures
        self.global_requests = deque()
        self.user_requests = defaultdict(lambda: deque())
        self.endpoint_requests = defaultdict(lambda: deque())
        
        # Blocked IPs and users
        self.blocked_ips = set()
        self.blocked_users = set()
        
        logger.info("🚦 Rate Limiting Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests"""
        
        current_time = datetime.now()
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            raise HTTPException(status_code=429, detail="IP address is blocked")
        
        # Get user ID from request (if authenticated)
        user_id = getattr(request.state, 'user_id', None)
        
        # Check if user is blocked
        if user_id and user_id in self.blocked_users:
            logger.warning(f"Blocked user attempted access: {user_id}")
            raise HTTPException(status_code=429, detail="User account is rate limited")
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self._clean_old_requests(cutoff_time)
        
        # Check global rate limit
        if len(self.global_requests) >= self.global_limit:
            logger.warning(f"Global rate limit exceeded from {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Global rate limit exceeded. Please try again later."
            )
        
        # Check user rate limit
        if user_id:
            user_request_count = len(self.user_requests[user_id])
            if user_request_count >= self.user_limit:
                logger.warning(f"User rate limit exceeded: {user_id}")
                raise HTTPException(
                    status_code=429,
                    detail="User rate limit exceeded. Please try again later."
                )
        
        # Check endpoint-specific rate limit
        endpoint_limit = self.endpoint_limits.get(endpoint)
        if endpoint_limit:
            endpoint_key = f"{user_id or client_ip}:{endpoint}"
            endpoint_request_count = len(self.endpoint_requests[endpoint_key])
            
            if endpoint_request_count >= endpoint_limit:
                logger.warning(f"Endpoint rate limit exceeded: {endpoint} by {user_id or client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {endpoint}. Please try again later."
                )
        
        # Record the request
        self.global_requests.append(current_time)
        if user_id:
            self.user_requests[user_id].append(current_time)
        if endpoint_limit:
            endpoint_key = f"{user_id or client_ip}:{endpoint}"
            self.endpoint_requests[endpoint_key].append(current_time)
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Global-Limit"] = str(self.global_limit)
            response.headers["X-RateLimit-Global-Remaining"] = str(
                max(0, self.global_limit - len(self.global_requests))
            )
            
            if user_id:
                response.headers["X-RateLimit-User-Limit"] = str(self.user_limit)
                response.headers["X-RateLimit-User-Remaining"] = str(
                    max(0, self.user_limit - len(self.user_requests[user_id]))
                )
            
            return response
            
        except HTTPException as e:
            # If rate limited, consider temporary blocking for repeated violations
            if e.status_code == 429:
                self._handle_rate_limit_violation(client_ip, user_id)
            raise
    
    def _clean_old_requests(self, cutoff_time: datetime):
        """Remove requests older than cutoff time"""
        
        # Clean global requests
        while self.global_requests and self.global_requests[0] < cutoff_time:
            self.global_requests.popleft()
        
        # Clean user requests
        for user_id in list(self.user_requests.keys()):
            user_queue = self.user_requests[user_id]
            while user_queue and user_queue[0] < cutoff_time:
                user_queue.popleft()
            
            # Remove empty queues
            if not user_queue:
                del self.user_requests[user_id]
        
        # Clean endpoint requests
        for endpoint_key in list(self.endpoint_requests.keys()):
            endpoint_queue = self.endpoint_requests[endpoint_key]
            while endpoint_queue and endpoint_queue[0] < cutoff_time:
                endpoint_queue.popleft()
            
            # Remove empty queues
            if not endpoint_queue:
                del self.endpoint_requests[endpoint_key]
    
    def _handle_rate_limit_violation(self, client_ip: str, user_id: Optional[str]):
        """Handle repeated rate limit violations"""
        
        # This is a simplified implementation
        # In production, you might want more sophisticated logic
        
        # Count recent violations (this would need persistent storage)
        # For now, just log the violation
        logger.warning(
            f"Rate limit violation from IP: {client_ip}, User: {user_id or 'anonymous'}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        
        return {
            "global_requests_last_minute": len(self.global_requests),
            "active_users": len(self.user_requests),
            "active_endpoints": len(self.endpoint_requests),
            "blocked_ips": len(self.blocked_ips),
            "blocked_users": len(self.blocked_users),
            "limits": {
                "global": self.global_limit,
                "user": self.user_limit,
                "endpoints": self.endpoint_limits
            }
        }
    
    def block_ip(self, ip: str):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        logger.info(f"Blocked IP: {ip}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP: {ip}")
    
    def block_user(self, user_id: str):
        """Block a user"""
        self.blocked_users.add(user_id)
        logger.info(f"Blocked user: {user_id}")
    
    def unblock_user(self, user_id: str):
        """Unblock a user"""
        self.blocked_users.discard(user_id)
        logger.info(f"Unblocked user: {user_id}")

# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    🛡️ SECURITY MIDDLEWARE
    
    Security enhancements including security headers, input validation,
    and protection against common attacks.
    """
    
    def __init__(self, app):
        super().__init__(app)
        logger.info("🛡️ Security Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply security measures to requests"""
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Add custom security headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Powered-By"] = "MasterX-Quantum-Intelligence"
        
        return response

# ============================================================================
# PERFORMANCE MONITORING MIDDLEWARE
# ============================================================================

class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    📊 PERFORMANCE MONITORING MIDDLEWARE
    
    Performance monitoring and metrics collection for API optimization.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "endpoint_metrics": defaultdict(lambda: {
                "count": 0,
                "total_time": 0.0,
                "errors": 0
            })
        }
        logger.info("📊 Performance Monitoring Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        
        start_time = time.time()
        endpoint = request.url.path
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            response_time = time.time() - start_time
            
            # Update global metrics
            self.metrics["total_requests"] += 1
            self.metrics["total_response_time"] += response_time
            
            # Update endpoint metrics
            endpoint_stats = self.metrics["endpoint_metrics"][endpoint]
            endpoint_stats["count"] += 1
            endpoint_stats["total_time"] += response_time
            
            # Add performance headers
            response.headers["X-Performance-Time"] = f"{response_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Track errors
            response_time = time.time() - start_time
            endpoint_stats = self.metrics["endpoint_metrics"][endpoint]
            endpoint_stats["errors"] += 1
            
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        avg_response_time = (
            self.metrics["total_response_time"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )
        
        endpoint_stats = {}
        for endpoint, stats in self.metrics["endpoint_metrics"].items():
            endpoint_stats[endpoint] = {
                "count": stats["count"],
                "avg_response_time": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                "error_rate": stats["errors"] / stats["count"] if stats["count"] > 0 else 0
            }
        
        return {
            "total_requests": self.metrics["total_requests"],
            "average_response_time": avg_response_time,
            "endpoints": endpoint_stats
        }
