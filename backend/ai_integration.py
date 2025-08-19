"""
MasterX AI Integration Module
Supports multiple AI providers with fallback mechanisms
"""

import os
import asyncio
import aiohttp
import json
import ssl
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create SSL context that doesn't verify certificates (for development)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    confidence: float

class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = ""
        self.headers = {}
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        raise NotImplementedError

class GroqProvider(AIProvider):
    """Groq AI Provider"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        super().__init__(api_key, model)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False
        }
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        tokens_used = data["usage"]["total_tokens"]
                        
                        return AIResponse(
                            content=content,
                            model=self.model,
                            provider="groq",
                            tokens_used=tokens_used,
                            response_time=time.time() - start_time,
                            confidence=0.95
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Groq provider error: {str(e)}")
            raise

class GeminiProvider(AIProvider):
    """Google Gemini AI Provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, model)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        start_time = time.time()
        
        # Convert OpenAI format to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            }
        }
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.base_url}?key={self.api_key}",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Estimate tokens (Gemini doesn't always return usage)
                        tokens_used = len(content.split()) * 1.3  # Rough estimation
                        
                        return AIResponse(
                            content=content,
                            model=self.model,
                            provider="gemini",
                            tokens_used=int(tokens_used),
                            response_time=time.time() - start_time,
                            confidence=0.92
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        raise Exception(f"Gemini API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Gemini provider error: {str(e)}")
            raise

class OpenAIProvider(AIProvider):
    """OpenAI Provider (optional)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        tokens_used = data["usage"]["total_tokens"]
                        
                        return AIResponse(
                            content=content,
                            model=self.model,
                            provider="openai",
                            tokens_used=tokens_used,
                            response_time=time.time() - start_time,
                            confidence=0.98
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise Exception(f"OpenAI API error: {response.status}")
        
        except Exception as e:
            logger.error(f"OpenAI provider error: {str(e)}")
            raise

class AnthropicProvider(AIProvider):
    """Anthropic Claude Provider (optional)"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> AIResponse:
        start_time = time.time()
        
        # Convert to Anthropic format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": anthropic_messages
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["content"][0]["text"]
                        tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                        
                        return AIResponse(
                            content=content,
                            model=self.model,
                            provider="anthropic",
                            tokens_used=tokens_used,
                            response_time=time.time() - start_time,
                            confidence=0.96
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {response.status} - {error_text}")
                        raise Exception(f"Anthropic API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Anthropic provider error: {str(e)}")
            raise

class AIManager:
    """Manages multiple AI providers with fallback"""
    
    def __init__(self):
        self.providers = []
        self.setup_providers()
    
    def setup_providers(self):
        """Initialize available providers based on environment variables"""
        
        # Primary: Gemini 2.0 Flash (your provided key)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.providers.append(GeminiProvider(gemini_key, "gemini-2.0-flash-exp"))
            logger.info("Gemini 2.0 Flash provider initialized")
        
        # Fallback: Groq Llama 3.3 70B
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.providers.append(GroqProvider(groq_key, "llama-3.3-70b-versatile"))
            logger.info("Groq Llama 3.3 70B provider initialized")
        
        # Optional: OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.providers.append(OpenAIProvider(openai_key, "gpt-4"))
            logger.info("OpenAI GPT-4 provider initialized")
        
        # Optional: Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers.append(AnthropicProvider(anthropic_key, "claude-3-sonnet-20240229"))
            logger.info("Anthropic Claude provider initialized")
        
        if not self.providers:
            logger.error("No AI providers available! Check your API keys.")
    
    async def generate_response(self, user_message: str, context: Optional[str] = None) -> AIResponse:
        """Generate AI response with fallback mechanism"""
        
        # Prepare messages
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"You are MasterX, an advanced quantum intelligence AI assistant. {context}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are MasterX, an advanced quantum intelligence AI assistant. Provide helpful, accurate, and engaging responses."
            })
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Try providers in order
        for i, provider in enumerate(self.providers):
            try:
                logger.info(f"Attempting response with {provider.__class__.__name__}")
                response = await provider.generate_response(
                    messages,
                    max_tokens=int(os.getenv("MAX_TOKENS", 4096)),
                    temperature=float(os.getenv("TEMPERATURE", 0.7)),
                    top_p=float(os.getenv("TOP_P", 0.9))
                )
                logger.info(f"Success with {provider.__class__.__name__}")
                return response
            
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed: {str(e)}")
                if i == len(self.providers) - 1:  # Last provider
                    # Return fallback response
                    return AIResponse(
                        content="I apologize, but I'm experiencing technical difficulties with my AI systems. Please try again in a moment.",
                        model="fallback",
                        provider="system",
                        tokens_used=0,
                        response_time=0.1,
                        confidence=0.0
                    )
                continue
        
        # Should never reach here, but just in case
        return AIResponse(
            content="System error: No providers available.",
            model="error",
            provider="system",
            tokens_used=0,
            response_time=0.0,
            confidence=0.0
        )

# Global AI manager instance
ai_manager = AIManager()
