"""
Main Quantum Learning Intelligence Engine with improved architecture
"""

import asyncio
import weakref
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ..config.settings import QuantumEngineConfig
from ..utils.caching import CacheService, cached
from ..utils.monitoring import MetricsService, HealthCheckService, timed, counted
from .exceptions import QuantumEngineError, ModelLoadError, AIProviderError, ErrorCodes
from .enums import QuantumLearningMode, QuantumState, IntelligenceLevel
from .data_structures import QuantumLearningContext, QuantumResponse


class QuantumLearningIntelligenceEngine:
    """
    🚀 REVOLUTIONARY QUANTUM LEARNING INTELLIGENCE ENGINE 🚀
    
    Modularized, production-ready AI learning system with:
    - Dependency injection architecture
    - Proper error handling and monitoring
    - Memory management and caching
    - Backward compatibility
    """
    
    def __init__(
        self,
        config: QuantumEngineConfig,
        cache_service: CacheService,
        metrics_service: MetricsService,
        health_service: HealthCheckService
    ):
        self.config = config
        self.cache = cache_service
        self.metrics = metrics_service
        self.health = health_service
        
        # Initialize core components
        self._ai_providers: Dict[str, Any] = {}
        self._neural_networks: Dict[str, Any] = {}
        self._learning_modes: Dict[str, Any] = {}
        self._services: Dict[str, Any] = {}
        
        # Weak references for memory management
        self._model_cache = weakref.WeakValueDictionary()
        self._session_cache = weakref.WeakValueDictionary()
        
        # Initialize systems
        self._initialize_ai_providers()
        self._initialize_neural_networks()
        self._initialize_learning_modes()
        self._initialize_services()
        
        # Register health checks
        self._register_health_checks()
        
        logger.info("Quantum Learning Intelligence Engine initialized")
    
    def _initialize_ai_providers(self) -> None:
        """Initialize AI providers with fallback support"""
        try:
            # Primary provider (Groq)
            if self.config.groq_api_key:
                from groq import AsyncGroq
                self._ai_providers['groq'] = AsyncGroq(api_key=self.config.groq_api_key)
                logger.info("Groq AI provider initialized")
            
            # Fallback providers
            if self.config.openai_api_key:
                try:
                    import openai
                    self._ai_providers['openai'] = openai.AsyncOpenAI(api_key=self.config.openai_api_key)
                    logger.info("OpenAI provider initialized")
                except ImportError:
                    logger.warning("OpenAI package not available")
            
            if self.config.anthropic_api_key:
                try:
                    import anthropic
                    self._ai_providers['anthropic'] = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
                    logger.info("Anthropic provider initialized")
                except ImportError:
                    logger.warning("Anthropic package not available")
            
            if not self._ai_providers:
                raise ModelLoadError(
                    "No AI providers available",
                    ErrorCodes.MODEL_LOAD_FAILED,
                    {"configured_keys": bool(self.config.groq_api_key or self.config.openai_api_key or self.config.anthropic_api_key)}
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize AI providers: {e}")
            raise ModelLoadError(f"AI provider initialization failed: {e}", ErrorCodes.MODEL_LOAD_FAILED)
    
    def _initialize_neural_networks(self) -> None:
        """Initialize neural networks with lazy loading"""
        if not self.config.enable_neural_networks:
            logger.info("Neural networks disabled in configuration")
            return
        
        try:
            # Lazy loading of neural networks
            self._neural_networks = {
                'quantum_processor': None,  # Loaded on demand
                'difficulty_network': None,  # Loaded on demand
                'transformer_optimizer': None,  # Loaded on demand
            }
            logger.info("Neural network placeholders initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize neural networks: {e}")
            # Don't fail completely - continue without neural networks
            self._neural_networks = {}
    
    def _get_neural_network(self, network_name: str):
        """Lazy load neural networks"""
        if network_name not in self._neural_networks:
            return None
        
        if self._neural_networks[network_name] is None:
            try:
                if network_name == 'quantum_processor':
                    from ..neural_networks.quantum_processor import QuantumResponseProcessor
                    self._neural_networks[network_name] = QuantumResponseProcessor()
                elif network_name == 'difficulty_network':
                    from ..neural_networks.difficulty_network import AdaptiveDifficultyNetwork
                    self._neural_networks[network_name] = AdaptiveDifficultyNetwork()
                # Add other networks as needed
                
                logger.info(f"Loaded neural network: {network_name}")
                
            except Exception as e:
                logger.error(f"Failed to load neural network {network_name}: {e}")
                return None
        
        return self._neural_networks[network_name]
    
    def _initialize_learning_modes(self) -> None:
        """Initialize learning mode handlers"""
        try:
            # Import learning modes
            from ..learning_modes.adaptive_quantum import AdaptiveQuantumMode
            from ..learning_modes.socratic_discovery import SocraticDiscoveryMode

            self._learning_modes = {
                QuantumLearningMode.ADAPTIVE_QUANTUM: AdaptiveQuantumMode(),
                QuantumLearningMode.SOCRATIC_DISCOVERY: SocraticDiscoveryMode(),
            }
            logger.info(f"Learning modes initialized: {list(self._learning_modes.keys())}")

        except ImportError as e:
            logger.warning(f"Some learning modes not available: {e}")
            # Use fallback implementations
            self._learning_modes = {}
    
    def _initialize_services(self) -> None:
        """Initialize service modules"""
        try:
            # Initialize services with dependency injection
            from ..services.personalization.engine import PersonalizationEngine
            from ..services.analytics.learning_patterns import LearningPatternAnalysisEngine
            
            self._services = {
                'personalization': PersonalizationEngine(self.cache),
                'analytics': LearningPatternAnalysisEngine(self.cache),
                # Add other services as they're implemented
            }
            logger.info("Services initialized")
            
        except ImportError as e:
            logger.warning(f"Some services not available: {e}")
            self._services = {}
    
    def _register_health_checks(self) -> None:
        """Register health checks for the engine"""
        try:
            # Only register health checks if the service supports it
            if hasattr(self.health, 'register_check'):
                self.health.register_check("ai_providers", self._check_ai_providers, critical=True)
                self.health.register_check("neural_networks", self._check_neural_networks, critical=False)
                self.health.register_check("cache_service", self._check_cache_service, critical=True)
            else:
                logger.info("Health service doesn't support check registration - using simple health service")
        except Exception as e:
            logger.warning(f"Failed to register health checks: {e}")
    
    def _check_ai_providers(self) -> bool:
        """Health check for AI providers"""
        return len(self._ai_providers) > 0
    
    def _check_neural_networks(self) -> bool:
        """Health check for neural networks"""
        if not self.config.enable_neural_networks:
            return True
        return len(self._neural_networks) > 0
    
    async def _check_cache_service(self) -> bool:
        """Health check for cache service"""
        try:
            await self.cache.set("health_check", "ok", ttl=60)
            result = await self.cache.get("health_check")
            return result == "ok"
        except Exception:
            return False
    
    @timed("quantum_engine.get_quantum_response.duration")
    @counted("quantum_engine.get_quantum_response.calls")
    async def get_quantum_response(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        learning_dna: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QuantumResponse:
        """
        Generate quantum AI response with full intelligence
        
        This is the main entry point that maintains backward compatibility
        while using the new modular architecture.
        """
        try:
            # Create quantum learning context
            quantum_context = await self._create_quantum_context(
                user_message, user_id, session_id, learning_dna, context
            )
            
            # Determine optimal learning mode
            learning_mode = await self._determine_learning_mode(user_message, quantum_context)
            
            # Generate response using appropriate mode
            if learning_mode in self._learning_modes:
                response = await self._learning_modes[learning_mode].generate_response(
                    user_message, quantum_context
                )
            else:
                # Fallback to direct AI generation
                response = await self._generate_fallback_response(user_message, quantum_context)
            
            # Record metrics
            self.metrics.increment_counter("quantum_responses.generated")
            self.metrics.set_gauge("quantum_responses.last_generation_time", response.processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating quantum response: {e}")
            self.metrics.increment_counter("quantum_responses.errors")
            
            # Return fallback response
            return await self._create_fallback_response(user_message, user_id, session_id)
    
    @cached(ttl=300, key_prefix="quantum_context")
    async def _create_quantum_context(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        learning_dna: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QuantumLearningContext:
        """Create quantum learning context with caching"""
        
        # Get personalization data
        if 'personalization' in self._services:
            personalization_data = await self._services['personalization'].get_user_profile(user_id)
        else:
            personalization_data = learning_dna or {}
        
        # Create context
        return QuantumLearningContext(
            user_id=user_id,
            session_id=session_id,
            message=user_message,
            learning_dna=personalization_data,
            context=context or {},
            timestamp=datetime.utcnow()
        )
    
    async def _determine_learning_mode(
        self,
        user_message: str,
        quantum_context: QuantumLearningContext
    ) -> QuantumLearningMode:
        """Determine optimal learning mode using AI"""
        
        # Use cached result if available
        cache_key = f"learning_mode:{hash(user_message)}"
        cached_mode = await self.cache.get(cache_key)
        if cached_mode:
            return QuantumLearningMode(cached_mode)
        
        # Analyze message to determine mode
        message_lower = user_message.lower()
        
        # Simple rule-based determination (can be enhanced with ML)
        if any(word in message_lower for word in ["why", "how", "explain", "understand"]):
            mode = QuantumLearningMode.SOCRATIC_DISCOVERY
        elif any(word in message_lower for word in ["debug", "error", "fix", "problem"]):
            mode = QuantumLearningMode.DEBUG_MASTERY
        elif any(word in message_lower for word in ["challenge", "test", "quiz", "practice"]):
            mode = QuantumLearningMode.CHALLENGE_MODE
        else:
            mode = QuantumLearningMode.ADAPTIVE_QUANTUM
        
        # Cache the result
        await self.cache.set(cache_key, mode.value, ttl=3600)
        
        return mode
    
    async def _generate_fallback_response(
        self,
        user_message: str,
        quantum_context: QuantumLearningContext
    ) -> QuantumResponse:
        """Generate fallback response using AI providers"""
        
        for provider_name, provider in self._ai_providers.items():
            try:
                # Generate response using provider
                if provider_name == 'groq':
                    response = await self._generate_groq_response(user_message, provider)
                elif provider_name == 'openai':
                    response = await self._generate_openai_response(user_message, provider)
                elif provider_name == 'anthropic':
                    response = await self._generate_anthropic_response(user_message, provider)
                else:
                    continue
                
                return QuantumResponse(
                    content=response,
                    quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
                    quantum_state=QuantumState.DISCOVERY,
                    intelligence_level=IntelligenceLevel.ENHANCED,
                    processing_time=0.5,  # Placeholder
                    confidence=0.8,
                    metadata={"provider": provider_name}
                )
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # If all providers fail, return basic fallback
        return await self._create_fallback_response(
            user_message, 
            quantum_context.user_id, 
            quantum_context.session_id
        )
    
    async def _generate_groq_response(self, message: str, provider) -> str:
        """Generate response using Groq"""
        response = await provider.chat.completions.create(
            model=self.config.primary_model,
            messages=[{"role": "user", "content": message}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    async def _generate_openai_response(self, message: str, provider) -> str:
        """Generate response using OpenAI"""
        response = await provider.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    async def _generate_anthropic_response(self, message: str, provider) -> str:
        """Generate response using Anthropic"""
        response = await provider.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text
    
    async def _create_fallback_response(
        self,
        user_message: str,
        user_id: str,
        session_id: str
    ) -> QuantumResponse:
        """Create basic fallback response when all else fails"""
        
        return QuantumResponse(
            content="I'm here to help you learn! While my quantum intelligence is optimizing, I can still provide learning support. Could you please rephrase your question?",
            quantum_mode=QuantumLearningMode.ADAPTIVE_QUANTUM,
            quantum_state=QuantumState.DISCOVERY,
            intelligence_level=IntelligenceLevel.BASIC,
            processing_time=0.1,
            confidence=0.5,
            metadata={
                "fallback": True,
                "user_id": user_id,
                "session_id": session_id
            }
        )
    
    async def close(self) -> None:
        """Cleanup resources"""
        logger.info("Closing Quantum Learning Intelligence Engine")
        
        # Close AI provider connections
        for provider in self._ai_providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        
        # Clear caches
        self._model_cache.clear()
        self._session_cache.clear()
        
        logger.info("Quantum engine closed successfully")
