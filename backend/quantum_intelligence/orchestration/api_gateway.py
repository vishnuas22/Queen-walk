"""
API Gateway for MasterX Quantum Intelligence Platform

Advanced API Gateway that provides unified endpoint management, authentication,
authorization, rate limiting, request routing, load balancing, and comprehensive
API management for all quantum intelligence services.

🌐 API GATEWAY CAPABILITIES:
- Unified API endpoint management and routing
- Advanced authentication and authorization systems
- Intelligent rate limiting and throttling mechanisms
- Request routing and load balancing across services
- API versioning and backward compatibility
- Comprehensive monitoring and analytics

Author: MasterX AI Team - Integration & Orchestration Division
Version: 1.0 - Phase 11 Integration & Orchestration
"""

import asyncio
import json
import time
import uuid
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

# Try to import advanced libraries with fallbacks
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available - using basic HTTP implementation")

# ============================================================================
# API GATEWAY ENUMS & DATA STRUCTURES
# ============================================================================

class AuthenticationType(Enum):
    """Authentication type enumeration"""
    NONE = "none"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"

class RateLimitType(Enum):
    """Rate limit type enumeration"""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithm enumeration"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    HEALTH_BASED = "health_based"

@dataclass
class APIEndpoint:
    """
    🔌 API ENDPOINT
    
    Represents an API endpoint configuration
    """
    endpoint_id: str
    path: str
    method: str
    service_name: str
    
    # Routing configuration
    upstream_path: str
    upstream_hosts: List[str]
    load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    
    # Authentication and authorization
    authentication_type: AuthenticationType = AuthenticationType.JWT_TOKEN
    required_roles: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_type: RateLimitType = RateLimitType.REQUESTS_PER_MINUTE
    rate_limit_value: int = 100
    rate_limit_burst: int = 10
    
    # Caching
    cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    cache_key_pattern: Optional[str] = None
    
    # Timeouts and retries
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    
    # Monitoring and logging
    logging_enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    
    # Metadata
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False

@dataclass
class APIRequest:
    """
    📨 API REQUEST
    
    Represents an incoming API request
    """
    request_id: str
    method: str
    path: str
    
    # Request data
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    
    # Client information
    client_ip: str = ""
    user_agent: str = ""
    
    # Authentication
    auth_token: Optional[str] = None
    user_id: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    
    # Routing
    matched_endpoint: Optional[APIEndpoint] = None
    upstream_host: Optional[str] = None
    
    # Timing
    received_at: datetime = field(default_factory=datetime.now)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None

@dataclass
class APIResponse:
    """
    📬 API RESPONSE
    
    Represents an API response
    """
    request_id: str
    status_code: int
    
    # Response data
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    
    # Performance metrics
    processing_time_ms: float = 0.0
    upstream_time_ms: float = 0.0
    cache_hit: bool = False
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metadata
    response_timestamp: datetime = field(default_factory=datetime.now)


class APIGateway:
    """
    🌐 API GATEWAY
    
    Advanced API Gateway that provides unified endpoint management, authentication,
    authorization, rate limiting, request routing, and comprehensive API management
    for all quantum intelligence services.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API Gateway"""
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Core components
        self.endpoint_registry: Dict[str, APIEndpoint] = {}
        self.route_matcher = RouteMatcher()
        self.authenticator = Authenticator(self.config.get('authentication', {}))
        self.rate_limiter = RateLimiter(self.config.get('rate_limiting', {}))
        self.load_balancer = APILoadBalancer(self.config.get('load_balancing', {}))
        self.cache_manager = CacheManager(self.config.get('caching', {}))
        
        # Monitoring and analytics
        self.request_logger = RequestLogger()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        
        # Gateway state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # FastAPI application (if available)
        self.app = None
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        
        # Performance metrics
        self.gateway_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'rate_limited_requests': 0
        }
        
        logger.info("🌐 API Gateway initialized")
    
    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> bool:
        """
        Start the API Gateway
        
        Args:
            host: Host to bind to
            port: Port to bind to
            
        Returns:
            bool: True if startup successful, False otherwise
        """
        try:
            logger.info(f"🚀 Starting API Gateway on {host}:{port}...")
            
            # Start core components
            await self.authenticator.start()
            await self.rate_limiter.start()
            await self.cache_manager.start()
            await self.health_checker.start()
            
            # Register default endpoints
            await self._register_default_endpoints()
            
            # Start FastAPI server if available
            if FASTAPI_AVAILABLE and self.app:
                # Configure CORS
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=self.config.get('cors', {}).get('allowed_origins', ["*"]),
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
                
                # Start server in background
                config = uvicorn.Config(
                    app=self.app,
                    host=host,
                    port=port,
                    log_level="info"
                )
                server = uvicorn.Server(config)
                asyncio.create_task(server.serve())
            
            self.is_running = True
            
            logger.info("✅ API Gateway started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start API Gateway: {e}")
            await self.shutdown()
            return False
    
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the API Gateway
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            logger.info("🛑 Shutting down API Gateway...")
            
            self.is_running = False
            
            # Shutdown components
            await self.health_checker.shutdown()
            await self.cache_manager.shutdown()
            await self.rate_limiter.shutdown()
            await self.authenticator.shutdown()
            
            logger.info("✅ API Gateway shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during API Gateway shutdown: {e}")
            return False
    
    async def register_endpoint(self, endpoint: APIEndpoint) -> bool:
        """
        Register an API endpoint
        
        Args:
            endpoint: API endpoint to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Validate endpoint
            if not await self._validate_endpoint(endpoint):
                return False
            
            # Register endpoint
            endpoint_key = f"{endpoint.method}:{endpoint.path}"
            self.endpoint_registry[endpoint_key] = endpoint
            
            # Register with route matcher
            self.route_matcher.add_route(endpoint)
            
            # Register with FastAPI if available
            if FASTAPI_AVAILABLE and self.app:
                await self._register_fastapi_route(endpoint)
            
            logger.info(f"✅ API endpoint registered: {endpoint_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register API endpoint: {e}")
            return False
    
    async def process_request(self, request: APIRequest) -> APIResponse:
        """
        Process an incoming API request
        
        Args:
            request: API request to process
            
        Returns:
            APIResponse: Processed response
        """
        try:
            start_time = time.time()
            request.processing_started_at = datetime.now()
            
            # Update metrics
            self.gateway_metrics['total_requests'] += 1
            
            # Match route
            endpoint = await self.route_matcher.match_route(request.method, request.path)
            if not endpoint:
                return self._create_error_response(request.request_id, 404, "Endpoint not found")
            
            request.matched_endpoint = endpoint
            
            # Authenticate request
            auth_result = await self.authenticator.authenticate(request, endpoint)
            if not auth_result.success:
                return self._create_error_response(request.request_id, 401, auth_result.error_message)
            
            request.user_id = auth_result.user_id
            request.user_roles = auth_result.user_roles
            
            # Check rate limits
            rate_limit_result = await self.rate_limiter.check_rate_limit(request, endpoint)
            if not rate_limit_result.allowed:
                self.gateway_metrics['rate_limited_requests'] += 1
                return self._create_error_response(request.request_id, 429, "Rate limit exceeded")
            
            # Check cache
            cache_key = self._generate_cache_key(request, endpoint)
            cached_response = await self.cache_manager.get(cache_key)
            if cached_response:
                cached_response.cache_hit = True
                self._update_cache_hit_rate(True)
                return cached_response
            
            # Route to upstream service
            upstream_response = await self._route_to_upstream(request, endpoint)
            
            # Cache response if enabled
            if endpoint.cache_enabled and upstream_response.status_code == 200:
                await self.cache_manager.set(cache_key, upstream_response, endpoint.cache_ttl_seconds)
            
            # Update metrics
            self.gateway_metrics['successful_requests'] += 1
            self._update_cache_hit_rate(False)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            upstream_response.processing_time_ms = processing_time
            
            # Update average response time
            self._update_average_response_time(processing_time)
            
            # Log request
            await self.request_logger.log_request(request, upstream_response)
            
            return upstream_response
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            self.gateway_metrics['failed_requests'] += 1
            return self._create_error_response(request.request_id, 500, "Internal server error")
        
        finally:
            request.processing_completed_at = datetime.now()
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive API Gateway status"""
        
        return {
            'is_running': self.is_running,
            'startup_time': self.startup_time,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'metrics': self.gateway_metrics,
            'endpoints': {
                'total_registered': len(self.endpoint_registry),
                'endpoints': list(self.endpoint_registry.keys())
            },
            'components': {
                'authenticator': 'active' if self.authenticator else 'inactive',
                'rate_limiter': 'active' if self.rate_limiter else 'inactive',
                'cache_manager': 'active' if self.cache_manager else 'inactive',
                'health_checker': 'active' if self.health_checker else 'inactive'
            },
            'fastapi_enabled': FASTAPI_AVAILABLE and self.app is not None
        }

    # ========================================================================
    # HELPER METHODS FOR API GATEWAY
    # ========================================================================

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default API Gateway configuration"""

        return {
            'authentication': {
                'jwt_secret': 'your-secret-key',
                'jwt_algorithm': 'HS256',
                'jwt_expiration_hours': 24,
                'api_key_header': 'X-API-Key'
            },
            'rate_limiting': {
                'default_requests_per_minute': 100,
                'burst_size': 10,
                'cleanup_interval_seconds': 60
            },
            'load_balancing': {
                'default_algorithm': 'round_robin',
                'health_check_interval_seconds': 30,
                'unhealthy_threshold': 3
            },
            'caching': {
                'enabled': True,
                'default_ttl_seconds': 300,
                'max_cache_size': 1000
            },
            'cors': {
                'allowed_origins': ["*"],
                'allowed_methods': ["GET", "POST", "PUT", "DELETE"],
                'allowed_headers': ["*"]
            },
            'monitoring': {
                'metrics_enabled': True,
                'logging_enabled': True,
                'tracing_enabled': True
            }
        }

    def _create_fastapi_app(self):
        """Create FastAPI application"""

        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available - API Gateway will use basic HTTP implementation")
            return None

        app = FastAPI(
            title="MasterX Quantum Intelligence API Gateway",
            description="Advanced API Gateway for Quantum Intelligence Services",
            version="1.0.0"
        )

        # Add middleware for request processing
        @app.middleware("http")
        async def process_request_middleware(request: Request, call_next):
            # Convert FastAPI request to APIRequest
            api_request = await self._convert_fastapi_request(request)

            # Process through gateway
            api_response = await self.process_request(api_request)

            # Convert APIResponse to FastAPI response
            return await self._convert_to_fastapi_response(api_response)

        return app

    async def _register_default_endpoints(self):
        """Register default API endpoints"""

        # Health check endpoint
        health_endpoint = APIEndpoint(
            endpoint_id="health_check",
            path="/health",
            method="GET",
            service_name="api_gateway",
            upstream_path="/health",
            upstream_hosts=["localhost:8000"],
            authentication_type=AuthenticationType.NONE,
            rate_limit_value=1000,
            description="Health check endpoint"
        )
        await self.register_endpoint(health_endpoint)

        # Metrics endpoint
        metrics_endpoint = APIEndpoint(
            endpoint_id="metrics",
            path="/metrics",
            method="GET",
            service_name="api_gateway",
            upstream_path="/metrics",
            upstream_hosts=["localhost:8000"],
            authentication_type=AuthenticationType.API_KEY,
            rate_limit_value=100,
            description="Metrics endpoint"
        )
        await self.register_endpoint(metrics_endpoint)

    async def _validate_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Validate an API endpoint"""

        if not all([endpoint.endpoint_id, endpoint.path, endpoint.method, endpoint.service_name]):
            logger.error("Invalid endpoint: missing required fields")
            return False

        if not endpoint.upstream_hosts:
            logger.error("Invalid endpoint: no upstream hosts specified")
            return False

        return True

    async def _register_fastapi_route(self, endpoint: APIEndpoint):
        """Register endpoint with FastAPI"""

        if not self.app:
            return

        # This would register the route with FastAPI
        # Implementation depends on dynamic route registration
        pass

    async def _convert_fastapi_request(self, request) -> APIRequest:
        """Convert FastAPI request to APIRequest"""

        # Extract headers
        headers = dict(request.headers)

        # Extract query parameters
        query_params = dict(request.query_params)

        # Read body
        body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None

        # Extract auth token
        auth_token = headers.get('authorization', '').replace('Bearer ', '') if 'authorization' in headers else None

        return APIRequest(
            request_id=str(uuid.uuid4()),
            method=request.method,
            path=request.url.path,
            headers=headers,
            query_params=query_params,
            body=body,
            client_ip=request.client.host if request.client else "",
            user_agent=headers.get('user-agent', ''),
            auth_token=auth_token
        )

    async def _convert_to_fastapi_response(self, api_response: APIResponse):
        """Convert APIResponse to FastAPI Response"""

        if not FASTAPI_AVAILABLE:
            return api_response

        return Response(
            content=api_response.body,
            status_code=api_response.status_code,
            headers=api_response.headers
        )

    def _create_error_response(self, request_id: str, status_code: int, message: str) -> APIResponse:
        """Create an error response"""

        error_body = json.dumps({
            'error': {
                'code': status_code,
                'message': message,
                'request_id': request_id,
                'timestamp': datetime.now().isoformat()
            }
        }).encode('utf-8')

        return APIResponse(
            request_id=request_id,
            status_code=status_code,
            headers={'Content-Type': 'application/json'},
            body=error_body,
            error_message=message
        )

    def _generate_cache_key(self, request: APIRequest, endpoint: APIEndpoint) -> str:
        """Generate cache key for request"""

        if endpoint.cache_key_pattern:
            # Use custom cache key pattern
            return endpoint.cache_key_pattern.format(
                method=request.method,
                path=request.path,
                user_id=request.user_id or 'anonymous'
            )
        else:
            # Default cache key
            key_parts = [request.method, request.path]
            if request.query_params:
                query_string = '&'.join(f"{k}={v}" for k, v in sorted(request.query_params.items()))
                key_parts.append(query_string)

            key = ':'.join(key_parts)
            return hashlib.md5(key.encode()).hexdigest()

    async def _route_to_upstream(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
        """Route request to upstream service"""

        try:
            # Select upstream host
            upstream_host = await self.load_balancer.select_host(
                endpoint.upstream_hosts,
                endpoint.load_balancing_algorithm
            )

            if not upstream_host:
                return self._create_error_response(request.request_id, 503, "No healthy upstream hosts")

            request.upstream_host = upstream_host

            # Simulate upstream call (in production, this would be an actual HTTP call)
            await asyncio.sleep(0.01)  # Simulate network latency

            # Create mock response
            response_body = json.dumps({
                'message': 'Success',
                'data': request.body.decode('utf-8') if request.body else None,
                'upstream_host': upstream_host,
                'processed_at': datetime.now().isoformat()
            }).encode('utf-8')

            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                headers={'Content-Type': 'application/json'},
                body=response_body
            )

        except Exception as e:
            logger.error(f"Error routing to upstream: {e}")
            return self._create_error_response(request.request_id, 502, "Bad gateway")

    def _update_average_response_time(self, response_time_ms: float):
        """Update average response time metric"""

        current_avg = self.gateway_metrics['average_response_time']
        total_requests = self.gateway_metrics['total_requests']

        if total_requests == 1:
            self.gateway_metrics['average_response_time'] = response_time_ms
        else:
            new_avg = ((current_avg * (total_requests - 1)) + response_time_ms) / total_requests
            self.gateway_metrics['average_response_time'] = new_avg

    def _update_cache_hit_rate(self, cache_hit: bool):
        """Update cache hit rate metric"""

        total_requests = self.gateway_metrics['total_requests']
        current_hits = self.gateway_metrics['cache_hit_rate'] * (total_requests - 1)

        if cache_hit:
            current_hits += 1

        self.gateway_metrics['cache_hit_rate'] = current_hits / total_requests


# ============================================================================
# HELPER CLASSES FOR API GATEWAY
# ============================================================================

class RouteMatcher:
    """Route matcher for API endpoints"""

    def __init__(self):
        self.routes = {}

    def add_route(self, endpoint: APIEndpoint):
        """Add a route to the matcher"""
        route_key = f"{endpoint.method}:{endpoint.path}"
        self.routes[route_key] = endpoint

    async def match_route(self, method: str, path: str) -> Optional[APIEndpoint]:
        """Match a route based on method and path"""
        route_key = f"{method}:{path}"
        return self.routes.get(route_key)


@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class Authenticator:
    """Authentication manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jwt_secret = config.get('jwt_secret', 'default-secret')
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        self.api_keys = {}  # In production, this would be a database

    async def start(self):
        """Start the authenticator"""
        # Load API keys, initialize JWT validation, etc.
        pass

    async def shutdown(self):
        """Shutdown the authenticator"""
        pass

    async def authenticate(self, request: APIRequest, endpoint: APIEndpoint) -> AuthenticationResult:
        """Authenticate a request"""

        if endpoint.authentication_type == AuthenticationType.NONE:
            return AuthenticationResult(success=True)

        elif endpoint.authentication_type == AuthenticationType.JWT_TOKEN:
            return await self._authenticate_jwt(request)

        elif endpoint.authentication_type == AuthenticationType.API_KEY:
            return await self._authenticate_api_key(request)

        else:
            return AuthenticationResult(success=False, error_message="Unsupported authentication type")

    async def _authenticate_jwt(self, request: APIRequest) -> AuthenticationResult:
        """Authenticate using JWT token"""

        if not request.auth_token:
            return AuthenticationResult(success=False, error_message="Missing JWT token")

        try:
            # Decode JWT token
            payload = jwt.decode(request.auth_token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            return AuthenticationResult(
                success=True,
                user_id=payload.get('user_id'),
                user_roles=payload.get('roles', [])
            )

        except jwt.ExpiredSignatureError:
            return AuthenticationResult(success=False, error_message="Token expired")
        except jwt.InvalidTokenError:
            return AuthenticationResult(success=False, error_message="Invalid token")

    async def _authenticate_api_key(self, request: APIRequest) -> AuthenticationResult:
        """Authenticate using API key"""

        api_key = request.headers.get('x-api-key')
        if not api_key:
            return AuthenticationResult(success=False, error_message="Missing API key")

        # In production, validate against database
        if api_key == "valid-api-key":
            return AuthenticationResult(success=True, user_id="api_user")
        else:
            return AuthenticationResult(success=False, error_message="Invalid API key")


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining_requests: int = 0
    reset_time: Optional[datetime] = None


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.request_timestamps = defaultdict(lambda: defaultdict(deque))

    async def start(self):
        """Start the rate limiter"""
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_requests())

    async def shutdown(self):
        """Shutdown the rate limiter"""
        pass

    async def check_rate_limit(self, request: APIRequest, endpoint: APIEndpoint) -> RateLimitResult:
        """Check if request is within rate limits"""

        # Generate rate limit key
        rate_limit_key = f"{request.client_ip}:{endpoint.endpoint_id}"

        # Get current time window
        now = datetime.now()
        window_start = self._get_window_start(now, endpoint.rate_limit_type)

        # Clean old requests
        self._clean_old_requests(rate_limit_key, window_start, endpoint.rate_limit_type)

        # Check current count
        current_count = len(self.request_timestamps[rate_limit_key][endpoint.rate_limit_type])

        if current_count >= endpoint.rate_limit_value:
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=self._get_next_window_start(now, endpoint.rate_limit_type)
            )

        # Record this request
        self.request_timestamps[rate_limit_key][endpoint.rate_limit_type].append(now)

        return RateLimitResult(
            allowed=True,
            remaining_requests=endpoint.rate_limit_value - current_count - 1
        )

    def _get_window_start(self, now: datetime, rate_limit_type: RateLimitType) -> datetime:
        """Get the start of the current time window"""

        if rate_limit_type == RateLimitType.REQUESTS_PER_SECOND:
            return now.replace(microsecond=0)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_MINUTE:
            return now.replace(second=0, microsecond=0)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_HOUR:
            return now.replace(minute=0, second=0, microsecond=0)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return now.replace(second=0, microsecond=0)

    def _get_next_window_start(self, now: datetime, rate_limit_type: RateLimitType) -> datetime:
        """Get the start of the next time window"""

        if rate_limit_type == RateLimitType.REQUESTS_PER_SECOND:
            return now.replace(microsecond=0) + timedelta(seconds=1)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_MINUTE:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_HOUR:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif rate_limit_type == RateLimitType.REQUESTS_PER_DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=1)

    def _clean_old_requests(self, key: str, window_start: datetime, rate_limit_type: RateLimitType):
        """Clean old requests outside the current window"""

        timestamps = self.request_timestamps[key][rate_limit_type]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

    async def _cleanup_old_requests(self):
        """Background task to cleanup old request data"""

        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                # Remove old data
                cutoff_time = datetime.now() - timedelta(days=1)

                for key in list(self.request_timestamps.keys()):
                    for rate_type in list(self.request_timestamps[key].keys()):
                        timestamps = self.request_timestamps[key][rate_type]
                        while timestamps and timestamps[0] < cutoff_time:
                            timestamps.popleft()

                        # Remove empty entries
                        if not timestamps:
                            del self.request_timestamps[key][rate_type]

                    if not self.request_timestamps[key]:
                        del self.request_timestamps[key]

            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")


class APILoadBalancer:
    """Load balancer for upstream services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.round_robin_counters = defaultdict(int)
        self.host_health = defaultdict(lambda: True)

    async def select_host(self, hosts: List[str], algorithm: LoadBalancingAlgorithm) -> Optional[str]:
        """Select a host based on load balancing algorithm"""

        # Filter healthy hosts
        healthy_hosts = [host for host in hosts if self.host_health[host]]

        if not healthy_hosts:
            return None

        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_selection(healthy_hosts)
        elif algorithm == LoadBalancingAlgorithm.IP_HASH:
            return self._ip_hash_selection(healthy_hosts, "default_ip")
        else:
            return healthy_hosts[0]  # Default to first healthy host

    def _round_robin_selection(self, hosts: List[str]) -> str:
        """Round robin selection"""
        hosts_key = ':'.join(sorted(hosts))
        counter = self.round_robin_counters[hosts_key]
        selected = hosts[counter % len(hosts)]
        self.round_robin_counters[hosts_key] = (counter + 1) % len(hosts)
        return selected

    def _ip_hash_selection(self, hosts: List[str], client_ip: str) -> str:
        """IP hash selection for session affinity"""
        hash_value = hash(client_ip)
        return hosts[hash_value % len(hosts)]


class CacheManager:
    """Cache manager for API responses"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        self.max_size = config.get('max_cache_size', 1000)

    async def start(self):
        """Start the cache manager"""
        asyncio.create_task(self._cleanup_expired_cache())

    async def shutdown(self):
        """Shutdown the cache manager"""
        pass

    async def get(self, key: str) -> Optional[APIResponse]:
        """Get cached response"""

        if key not in self.cache:
            return None

        # Check if expired
        if key in self.cache_timestamps:
            if datetime.now() > self.cache_timestamps[key]:
                del self.cache[key]
                del self.cache_timestamps[key]
                return None

        return self.cache[key]

    async def set(self, key: str, response: APIResponse, ttl_seconds: int):
        """Set cached response"""

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]

        self.cache[key] = response
        self.cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl_seconds)

    async def _cleanup_expired_cache(self):
        """Background task to cleanup expired cache entries"""

        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                now = datetime.now()
                expired_keys = [
                    key for key, expiry in self.cache_timestamps.items()
                    if now > expiry
                ]

                for key in expired_keys:
                    del self.cache[key]
                    del self.cache_timestamps[key]

            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")


class RequestLogger:
    """Request logger for API Gateway"""

    async def log_request(self, request: APIRequest, response: APIResponse):
        """Log API request and response"""

        log_entry = {
            'request_id': request.request_id,
            'method': request.method,
            'path': request.path,
            'status_code': response.status_code,
            'processing_time_ms': response.processing_time_ms,
            'client_ip': request.client_ip,
            'user_id': request.user_id,
            'timestamp': request.received_at.isoformat()
        }

        logger.info(f"API Request: {json.dumps(log_entry)}")


class MetricsCollector:
    """Metrics collector for API Gateway"""

    def __init__(self):
        self.metrics = defaultdict(int)

    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric"""
        self.metrics[metric_name] += value

    def get_metrics(self) -> Dict[str, int]:
        """Get all collected metrics"""
        return dict(self.metrics)


class HealthChecker:
    """Health checker for upstream services"""

    def __init__(self):
        self.health_status = defaultdict(lambda: True)

    async def start(self):
        """Start health checking"""
        asyncio.create_task(self._health_check_loop())

    async def shutdown(self):
        """Shutdown health checker"""
        pass

    async def _health_check_loop(self):
        """Background health check loop"""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                # Perform health checks on upstream services

            except Exception as e:
                logger.error(f"Error in health check: {e}")
