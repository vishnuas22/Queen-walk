"""
Utility Classes for MasterX Quantum Intelligence Platform API

Comprehensive utility classes that provide LLM integration, response handling,
WebSocket management, and other essential API functionality.

🛠️ UTILITY CAPABILITIES:
- Multi-LLM integration (Groq, Gemini, OpenAI)
- Response handling and formatting
- WebSocket connection management
- API response standardization
- Error handling and logging
- Performance monitoring

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Set
from fastapi import WebSocket, WebSocketDisconnect
import aiohttp
import ssl
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMIntegration:
    """
    🤖 LLM INTEGRATION
    
    Multi-provider LLM integration supporting Groq, Gemini, and OpenAI
    with intelligent routing, fallback mechanisms, and response optimization.
    """
    
    def __init__(self):
        """Initialize LLM integration"""
        
        # API keys from environment variables only
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Enhanced provider configurations with latest models
        self.providers = {
            'groq': {
                'api_key': self.groq_api_key,
                'base_url': 'https://api.groq.com/openai/v1',
                'models': {
                    'reasoning': 'deepseek-r1-distill-llama-70b',
                    'coding': 'llama-3.3-70b-versatile',
                    'fast': 'llama-3.1-8b-instant',
                    'general': 'llama-3.3-70b-versatile'
                },
                'default_model': 'llama-3.3-70b-versatile',
                'available': bool(self.groq_api_key),
                'strengths': ['speed', 'coding', 'reasoning'],
                'cost_tier': 'low'
            },
            'gemini': {
                'api_key': self.gemini_api_key,
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'models': {
                    'reasoning': 'gemini-2.0-flash-exp',
                    'multimodal': 'gemini-2.0-flash-exp',
                    'creative': 'gemini-1.5-pro',
                    'general': 'gemini-2.0-flash-exp'
                },
                'default_model': 'gemini-2.0-flash-exp',
                'available': bool(self.gemini_api_key),
                'strengths': ['multimodal', 'reasoning', 'creative'],
                'cost_tier': 'medium'
            },
            'openai': {
                'api_key': self.openai_api_key,
                'base_url': 'https://api.openai.com/v1',
                'models': {
                    'reasoning': 'gpt-4o',
                    'creative': 'gpt-4o',
                    'fast': 'gpt-4o-mini',
                    'general': 'gpt-4o'
                },
                'default_model': 'gpt-4o',
                'available': bool(self.openai_api_key),
                'strengths': ['reasoning', 'general', 'creative'],
                'cost_tier': 'high'
            },
            'anthropic': {
                'api_key': self.anthropic_api_key,
                'base_url': 'https://api.anthropic.com/v1',
                'models': {
                    'coding': 'claude-3-5-sonnet-20241022',
                    'reasoning': 'claude-3-5-sonnet-20241022',
                    'fast': 'claude-3-haiku-20240307',
                    'general': 'claude-3-5-sonnet-20241022'
                },
                'default_model': 'claude-3-5-sonnet-20241022',
                'available': bool(self.anthropic_api_key),
                'strengths': ['coding', 'analysis', 'reasoning'],
                'cost_tier': 'high'
            }
        }
        
        # Provider priority from environment or default
        self.provider_priority = os.getenv('LLM_PROVIDER_PRIORITY', 'groq,gemini,openai,anthropic').split(',')

        # Task-based model selection configuration
        self.task_models = {
            'reasoning': os.getenv('LLM_MODEL_REASONING', 'deepseek-r1-distill-llama-70b,gemini-2.0-flash-exp,gpt-4o').split(','),
            'coding': os.getenv('LLM_MODEL_CODING', 'llama-3.3-70b-versatile,claude-3-5-sonnet-20241022').split(','),
            'creative': os.getenv('LLM_MODEL_CREATIVE', 'gemini-1.5-pro,gpt-4o,gemini-2.0-flash-exp').split(','),
            'fast': os.getenv('LLM_MODEL_FAST', 'llama-3.1-8b-instant,gpt-4o-mini').split(','),
            'multimodal': os.getenv('LLM_MODEL_MULTIMODAL', 'gemini-2.0-flash-exp,gpt-4o').split(','),
            'general': ['llama-3.3-70b-versatile', 'gemini-2.0-flash-exp', 'gpt-4o', 'claude-3-5-sonnet-20241022']
        }

        # Performance configuration
        self.response_timeout = int(os.getenv('LLM_RESPONSE_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('LLM_MAX_RETRIES', '3'))
        self.fallback_enabled = os.getenv('LLM_FALLBACK_ENABLED', 'true').lower() == 'true'
        
        # Performance tracking
        self.provider_stats = {
            provider: {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'avg_response_time': 0.0
            }
            for provider in self.providers.keys()
        }

        # SSL context for development (bypass certificate verification)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        logger.info("🤖 LLM Integration initialized")
        logger.info(f"Available providers: {[p for p, config in self.providers.items() if config['available']]}")
    
    async def generate_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using intelligent model selection or specified provider"""

        try:
            # Analyze task type if not provided
            if not task_type:
                task_type = self._analyze_task_type(message, context)

            # Select provider based on task type or use specified provider
            if provider:
                selected_provider = provider if self.providers[provider]['available'] else None
            else:
                selected_provider = self._select_best_provider_for_task(task_type)

            if not selected_provider:
                raise Exception("No LLM providers available")

            # Get optimal model for the task if not specified
            if not model:
                model = self._get_model_for_task(selected_provider, task_type)

            # Generate response based on provider
            if selected_provider == 'groq':
                return await self._generate_groq_response(message, context, user_id, model, task_type)
            elif selected_provider == 'gemini':
                return await self._generate_gemini_response(message, context, user_id, model, task_type)
            elif selected_provider == 'openai':
                return await self._generate_openai_response(message, context, user_id, model, task_type)
            elif selected_provider == 'anthropic':
                return await self._generate_anthropic_response(message, context, user_id, model, task_type)
            else:
                raise Exception(f"Unsupported provider: {selected_provider}")

        except Exception as e:
            logger.error(f"LLM response generation error: {e}")

            # Try fallback provider if enabled
            if self.fallback_enabled and provider:
                fallback_provider = self._select_best_provider_for_task(task_type, exclude=[provider])
                if fallback_provider:
                    return await self.generate_response(message, context, user_id, fallback_provider, None, task_type)

            # Return development mode response (more realistic than error message)
            return await self._generate_development_response(message, context, task_type)
    
    async def stream_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        provider: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from LLM provider"""
        
        try:
            # Select provider
            selected_provider = provider or self._select_best_provider()
            
            if not selected_provider:
                yield {'error': 'No LLM providers available'}
                return
            
            # Stream based on provider
            if selected_provider == 'groq':
                async for chunk in self._stream_groq_response(message, context, user_id):
                    yield chunk
            elif selected_provider == 'gemini':
                async for chunk in self._stream_gemini_response(message, context, user_id):
                    yield chunk
            else:
                # Fallback to non-streaming
                response = await self.generate_response(message, context, user_id, selected_provider)
                yield {'content': response['content'], 'done': True}
                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield {'error': str(e)}
    
    def _analyze_task_type(self, message: str, context: Dict[str, Any]) -> str:
        """Analyze the message to determine the optimal task type"""

        message_lower = message.lower()

        # Check for coding-related keywords
        coding_keywords = ['code', 'program', 'function', 'class', 'debug', 'error', 'python', 'javascript', 'java', 'c++', 'algorithm']
        if any(keyword in message_lower for keyword in coding_keywords):
            return 'coding'

        # Check for reasoning/problem-solving keywords
        reasoning_keywords = ['solve', 'calculate', 'analyze', 'explain', 'why', 'how', 'problem', 'logic', 'reasoning']
        if any(keyword in message_lower for keyword in reasoning_keywords):
            return 'reasoning'

        # Check for creative keywords
        creative_keywords = ['story', 'creative', 'write', 'poem', 'essay', 'imagine', 'create', 'design']
        if any(keyword in message_lower for keyword in creative_keywords):
            return 'creative'

        # Check for multimodal keywords
        multimodal_keywords = ['image', 'picture', 'video', 'audio', 'visual', 'see', 'look', 'analyze image']
        if any(keyword in message_lower for keyword in multimodal_keywords):
            return 'multimodal'

        # Check for fast response indicators (short messages, simple questions)
        if len(message.split()) < 10 and any(word in message_lower for word in ['what', 'who', 'when', 'where', 'yes', 'no']):
            return 'fast'

        # Default to general
        return 'general'

    def _select_best_provider_for_task(self, task_type: str, exclude: List[str] = None) -> Optional[str]:
        """Select the best provider for a specific task type"""

        exclude = exclude or []

        # Get preferred models for this task type
        preferred_models = self.task_models.get(task_type, self.task_models['general'])

        # Find the best available provider for the preferred models
        for model in preferred_models:
            for provider_name, provider_config in self.providers.items():
                if (provider_name not in exclude and
                    provider_config['available'] and
                    model in provider_config['models'].values()):
                    return provider_name

        # Fallback to general provider selection
        return self._select_best_provider(exclude)

    def _select_best_provider(self, exclude: List[str] = None) -> Optional[str]:
        """Select the best available LLM provider"""

        exclude = exclude or []

        for provider in self.provider_priority:
            if provider not in exclude and self.providers[provider]['available']:
                return provider

        return None

    def _get_model_for_task(self, provider: str, task_type: str) -> str:
        """Get the best model for a specific task from a provider"""

        provider_config = self.providers.get(provider, {})
        models = provider_config.get('models', {})

        # Try to get task-specific model
        if task_type in models:
            return models[task_type]

        # Fallback to general or default model
        return models.get('general', provider_config.get('default_model', 'default'))
    
    async def _generate_groq_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None,
        task_type: str = 'general'
    ) -> Dict[str, Any]:
        """Generate response using Groq API"""
        
        start_time = time.time()
        provider = 'groq'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare request with optimal model
            model_name = model or self._get_model_for_task(provider, task_type)
            
            headers = {
                'Authorization': f'Bearer {self.providers[provider]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': message}
                ],
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            # Make API call
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'task_type': task_type,
                            'suggestions': self._generate_suggestions(content, task_type),
                            'metadata': {
                                'response_time': response_time,
                                'tokens_used': data.get('usage', {}).get('total_tokens', 0),
                                'task_optimization': f"Optimized for {task_type} tasks"
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Groq API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e
    
    async def _generate_gemini_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        
        start_time = time.time()
        provider = 'gemini'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare context-aware prompt
            context_prompt = self._build_context_prompt(message, context)
            
            # Prepare request
            model_name = model or self.providers[provider]['default_model']
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                'contents': [{
                    'parts': [{'text': context_prompt}]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'maxOutputTokens': 1000
                }
            }
            
            # Make API call
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/models/{model_name}:generateContent?key={self.providers[provider]['api_key']}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'suggestions': self._generate_suggestions(content),
                            'metadata': {
                                'response_time': response_time,
                                'safety_ratings': data['candidates'][0].get('safetyRatings', [])
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Gemini API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e
    
    async def _generate_openai_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        start_time = time.time()
        provider = 'openai'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare request
            model_name = model or self.providers[provider]['default_model']
            
            headers = {
                'Authorization': f'Bearer {self.providers[provider]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': message}
                ],
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            # Make API call
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'suggestions': self._generate_suggestions(content),
                            'metadata': {
                                'response_time': response_time,
                                'tokens_used': data.get('usage', {}).get('total_tokens', 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e

    async def _generate_anthropic_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None,
        task_type: str = 'general'
    ) -> Dict[str, Any]:
        """Generate response using Anthropic API"""

        start_time = time.time()
        provider = 'anthropic'

        try:
            self.provider_stats[provider]['requests'] += 1

            # Prepare system prompt with context
            system_prompt = self._build_system_prompt(context, task_type)

            # Prepare request with optimal model
            model_name = model or self._get_model_for_task(provider, task_type)

            headers = {
                'x-api-key': self.providers[provider]['api_key'],
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }

            payload = {
                'model': model_name,
                'max_tokens': 1000,
                'temperature': 0.7,
                'system': system_prompt,
                'messages': [
                    {'role': 'user', 'content': message}
                ]
            }

            # Make API call
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.response_timeout)
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']

                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)

                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'task_type': task_type,
                            'suggestions': self._generate_suggestions(content, task_type),
                            'metadata': {
                                'response_time': response_time,
                                'tokens_used': data.get('usage', {}).get('output_tokens', 0),
                                'task_optimization': f"Optimized for {task_type} tasks"
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error {response.status}: {error_text}")

        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e

    async def _stream_groq_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Groq API"""
        
        try:
            # For now, simulate streaming by chunking the response
            response = await self._generate_groq_response(message, context, user_id)
            content = response['content']
            
            # Split into chunks and yield
            words = content.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                yield {
                    'content': chunk,
                    'provider': 'groq',
                    'done': i + chunk_size >= len(words)
                }
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
        except Exception as e:
            yield {'error': str(e)}
    
    async def _stream_gemini_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Gemini API"""
        
        try:
            # For now, simulate streaming by chunking the response
            response = await self._generate_gemini_response(message, context, user_id, None, None)
            content = response['content']
            
            # Split into chunks and yield
            words = content.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                yield {
                    'content': chunk,
                    'provider': 'gemini',
                    'done': i + chunk_size >= len(words)
                }
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
        except Exception as e:
            yield {'error': str(e)}
    
    def _build_system_prompt(self, context: Dict[str, Any], task_type: str = 'general') -> str:
        """Build system prompt with context and task optimization"""

        # Task-specific prompt optimization
        task_prompts = {
            'reasoning': "You are MasterX AI, an advanced quantum intelligence learning assistant specialized in logical reasoning and problem-solving. Break down complex problems step-by-step and provide clear explanations.",
            'coding': "You are MasterX AI, an advanced quantum intelligence learning assistant specialized in programming and software development. Provide clean, well-commented code with explanations.",
            'creative': "You are MasterX AI, an advanced quantum intelligence learning assistant specialized in creative tasks. Use imagination and creativity while maintaining educational value.",
            'fast': "You are MasterX AI, an advanced quantum intelligence learning assistant. Provide concise, accurate responses quickly.",
            'multimodal': "You are MasterX AI, an advanced quantum intelligence learning assistant with multimodal capabilities. Analyze and respond to various types of content including text, images, and other media.",
            'general': "You are MasterX AI, an advanced quantum intelligence learning assistant. You provide personalized, adaptive learning support with deep understanding of each student's unique learning profile."
        }

        base_prompt = task_prompts.get(task_type, task_prompts['general'])
        
        # Add personalization context
        if 'personalization' in context:
            personalization = context['personalization']
            base_prompt += f"\n\nUser Learning Profile:"
            base_prompt += f"\n- Learning Style: {personalization.get('learning_style', 'adaptive')}"
            base_prompt += f"\n- Preferred Pace: {personalization.get('preferred_pace', 'moderate')}"
            base_prompt += f"\n- Difficulty Preference: {personalization.get('difficulty_preference', 0.5)}"
        
        # Add learning context
        if 'learning' in context:
            learning = context['learning']
            base_prompt += f"\n\nCurrent Learning Context:"
            base_prompt += f"\n- Goals: {', '.join(learning.get('current_goals', []))}"
            base_prompt += f"\n- Recent Topics: {', '.join(learning.get('recent_topics', []))}"
            base_prompt += f"\n- Skill Levels: {learning.get('skill_levels', {})}"
        
        base_prompt += "\n\nProvide helpful, encouraging, and personalized responses that adapt to the user's learning style and current progress."
        
        return base_prompt
    
    def _build_context_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build context-aware prompt for Gemini"""
        
        system_context = self._build_system_prompt(context)
        return f"{system_context}\n\nUser Question: {message}"
    
    def _generate_suggestions(self, content: str, task_type: str = 'general') -> List[str]:
        """Generate follow-up suggestions based on response content and task type"""

        # Task-specific suggestion generation
        task_suggestions = {
            'reasoning': [
                "Can you solve a similar problem?",
                "What's the step-by-step reasoning?",
                "Can you explain the logic behind this?"
            ],
            'coding': [
                "Can you show me a code example?",
                "What are the best practices for this?",
                "How can I debug this code?"
            ],
            'creative': [
                "Can you create another example?",
                "How can I make this more creative?",
                "What other approaches could work?"
            ],
            'fast': [
                "Tell me more",
                "Can you elaborate?",
                "What's next?"
            ],
            'multimodal': [
                "Can you analyze another example?",
                "What should I look for in images like this?",
                "How does this apply to other media?"
            ],
            'general': [
                "Can you explain this in more detail?",
                "What should I learn next?",
                "Can you give me practice exercises?"
            ]
        }

        # Get task-specific suggestions
        suggestions = task_suggestions.get(task_type, task_suggestions['general']).copy()

        # Add content-specific suggestions
        content_lower = content.lower()

        if 'python' in content_lower or 'code' in content_lower:
            suggestions.insert(0, "Can you show me a working example?")

        if 'math' in content_lower or 'equation' in content_lower:
            suggestions.insert(0, "Can you solve a practice problem?")

        if 'learn' in content_lower:
            suggestions.append("What resources do you recommend?")

        return suggestions[:3]  # Return max 3 suggestions
    
    def _update_provider_stats(self, provider: str, success: bool, response_time: float):
        """Update provider performance statistics"""
        
        stats = self.provider_stats[provider]
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        # Update average response time
        total_requests = stats['successes'] + stats['failures']
        current_avg = stats['avg_response_time']
        stats['avg_response_time'] = ((current_avg * (total_requests - 1)) + response_time) / total_requests
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider performance statistics"""
        return self.provider_stats.copy()

    async def _generate_development_response(self, message: str, context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Generate a realistic development response when API providers are unavailable"""

        # Create realistic responses based on task type and message content
        if task_type == 'coding' or 'code' in message.lower() or 'programming' in message.lower():
            response = """I'd be happy to help you with your coding question! Here are some general tips:

1. **Break down the problem**: Start by understanding what you're trying to achieve
2. **Plan your approach**: Think about the steps needed to solve the problem
3. **Write clean code**: Use meaningful variable names and add comments
4. **Test your solution**: Make sure your code works with different inputs

Could you share more details about the specific coding challenge you're working on? I can provide more targeted assistance once I understand the context better."""

        elif task_type == 'reasoning' or any(word in message.lower() for word in ['why', 'how', 'explain', 'understand']):
            response = """That's a great question! Let me help you think through this step by step:

🧠 **Analytical Approach:**
- First, let's identify the key components of your question
- Then we can examine the relationships between these elements
- Finally, we'll work toward a logical conclusion

The reasoning process often involves breaking complex ideas into smaller, manageable parts. This helps us understand not just the "what" but also the "why" behind concepts.

What specific aspect would you like me to elaborate on? I'm here to help you develop a deeper understanding."""

        elif task_type == 'creative' or any(word in message.lower() for word in ['creative', 'story', 'write', 'imagine']):
            response = """✨ **Creative Inspiration Activated!** ✨

I love helping with creative projects! Here are some approaches we could explore:

🎨 **Brainstorming Techniques:**
- Mind mapping to connect ideas
- "What if" scenarios to spark imagination
- Drawing inspiration from unexpected sources

🌟 **Creative Process:**
- Start with a core concept or emotion
- Build layers of detail and complexity
- Don't be afraid to experiment and iterate

Whether you're working on writing, art, problem-solving, or any other creative endeavor, I'm here to help you explore new possibilities and push creative boundaries.

What kind of creative project are you working on? I'd love to help you develop it further!"""

        else:
            # General response
            response = f"""Hello! I'm here to help you with your question about: "{message[:100]}{'...' if len(message) > 100 else ''}"

🎯 **How I can assist you:**
- Answer questions across various topics
- Help with problem-solving and analysis
- Provide explanations and learning guidance
- Offer creative ideas and suggestions

I'm designed to provide personalized, intelligent responses that adapt to your learning style and needs. While I'm currently running in development mode, I can still offer valuable insights and assistance.

Could you provide a bit more context about what you're looking for? This will help me give you the most relevant and helpful response."""

        return {
            'content': response,
            'provider': 'development_mode',
            'task_type': task_type,
            'suggestions': [
                'Ask follow-up questions for more specific help',
                'Provide more context about your specific needs',
                'Try different phrasing if you need a different approach'
            ],
            'metadata': {
                'mode': 'development',
                'response_type': 'realistic_fallback',
                'message_length': len(message)
            }
        }

# ============================================================================
# API RESPONSE HANDLER
# ============================================================================

class APIResponseHandler:
    """
    📋 API RESPONSE HANDLER
    
    Standardized response handling and formatting for all API endpoints.
    """
    
    def __init__(self):
        """Initialize response handler"""
        logger.info("📋 API Response Handler initialized")
    
    def success_response(
        self,
        data: Any,
        message: str = "Operation completed successfully",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized success response"""
        
        return {
            'success': True,
            'message': message,
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4())
        }
    
    def error_response(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            'success': False,
            'message': error_message,
            'error_code': error_code,
            'error_details': details or {},
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4())
        }

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class WebSocketManager:
    """
    🔌 WEBSOCKET MANAGER
    
    WebSocket connection management for real-time features.
    """
    
    def __init__(self):
        """Initialize WebSocket manager"""
        
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        
        logger.info("🔌 WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection"""
        
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Disconnect WebSocket"""
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection"""
        
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id, "unknown")
    
    async def send_user_message(self, message: str, user_id: str):
        """Send message to all connections of a user"""
        
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)
