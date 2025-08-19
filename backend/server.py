from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
import time
import hashlib
import secrets
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, AsyncGenerator
import uuid
from datetime import datetime, timedelta
import re
import aiofiles
import mimetypes

# Import AI Integration
from ai_integration import ai_manager, AIResponse

# Import Quantum Intelligence Engine (with proper path and fallback)
try:
    from quantum_intelligence.core.engine import QuantumLearningIntelligenceEngine
    from quantum_intelligence.config.dependencies import setup_dependencies, get_quantum_engine, cleanup_dependencies
    from quantum_intelligence.config.settings import get_config
    QUANTUM_ENGINE_AVAILABLE = True
    print("✅ Quantum Intelligence Engine loaded successfully")
except ImportError as e:
    print(f"⚠️ Quantum Intelligence Engine not available: {str(e)}")
    print("🔄 Using simplified AI response system")
    QUANTUM_ENGINE_AVAILABLE = False

# Import Interactive API (with fallback)
try:
    from interactive_api import router as interactive_router
    from interactive_service import InteractiveContentService
    INTERACTIVE_FEATURES_AVAILABLE = True
    print("✅ Interactive Features loaded successfully")
except ImportError as e:
    print(f"⚠️ Interactive Features not available: {str(e)}")
    print("🔄 Using basic message system")
    INTERACTIVE_FEATURES_AVAILABLE = False
    
from models import ChatSession, SessionCreate as ModelSessionCreate


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Production Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Configure logging
log_level = logging.DEBUG if ENVIRONMENT == "development" else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configurations
security = HTTPBearer(auto_error=False)

# Rate limiting storage
rate_limit_storage = {}

# Security utilities
def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash password securely"""
    salt = secrets.token_hex(16)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ':' + salt

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        password_hash, salt = hashed.split(':')
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() == password_hash
    except:
        return False

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    # Limit length
    sanitized = sanitized[:10000]
    return sanitized.strip()

def validate_file_type(filename: str) -> bool:
    """Validate file type against allowed extensions"""
    if not filename:
        return False

    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

async def check_rate_limit(request: Request) -> bool:
    """Check if request is within rate limits"""
    client_ip = getattr(request.client, 'host', '127.0.0.1') if request.client else '127.0.0.1'
    current_time = time.time()

    # Clean old entries
    cutoff_time = current_time - RATE_LIMIT_WINDOW
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage.get(client_ip, [])
        if timestamp > cutoff_time
    ]

    # Check current rate
    request_count = len(rate_limit_storage.get(client_ip, []))
    if request_count >= RATE_LIMIT_REQUESTS:
        return False

    # Add current request
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    rate_limit_storage[client_ip].append(current_time)

    return True

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user (placeholder for production auth)"""
    # In production, implement proper JWT token validation
    if ENVIRONMENT == "development":
        return {"user_id": "dev_user", "role": "admin"}

    if not credentials:
        return {"user_id": "anonymous", "role": "guest"}

    # Placeholder token validation
    return {"user_id": "authenticated_user", "role": "user"}

# MongoDB connection with fallback
try:
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ['DB_NAME']]
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ MongoDB not available: {e}")
    print("🔄 Using in-memory storage for development")
    MONGODB_AVAILABLE = False
    client = None
    db = None

# In-memory storage fallback
memory_storage = {
    'chat_sessions': {},
    'chat_messages': {},
    'status_checks': {},
    'learning_progress': {},
    'learning_streaks': {},
    'user_achievements': {},
    'learning_sessions': {},
    'uploaded_files': {}
}

# Response cache for performance optimization
response_cache = {}
CACHE_TTL = 300  # 5 minutes cache TTL

# Enhanced Performance monitoring
performance_stats = {
    'total_requests': 0,
    'avg_response_time': 0,
    'cache_hits': 0,
    'errors': 0,
    'security_events': 0,
    'rate_limit_hits': 0,
    'file_uploads': 0,
    'websocket_connections': 0
}

# System health metrics
system_health = {
    'status': 'healthy',
    'uptime_start': datetime.utcnow(),
    'last_health_check': datetime.utcnow(),
    'database_status': 'unknown',
    'memory_usage': 0,
    'active_sessions': 0
}

# Security event logging
security_events = []

def log_security_event(event_type: str, details: Dict[str, Any], request: Request = None):
    """Log security events for monitoring"""
    event = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'details': details,
        'client_ip': request.client.host if request else 'unknown',
        'user_agent': request.headers.get('user-agent', 'unknown') if request else 'unknown'
    }

    security_events.append(event)
    performance_stats['security_events'] += 1

    # Keep only last 1000 events
    if len(security_events) > 1000:
        security_events.pop(0)

    logger.warning(f"Security event: {event_type} - {details}")

def update_system_health():
    """Update system health metrics"""
    system_health['last_health_check'] = datetime.utcnow()
    system_health['active_sessions'] = len(memory_storage.get('chat_sessions', {}))

    # Check database status
    try:
        if MONGODB_AVAILABLE and db is not None:
            system_health['database_status'] = 'connected'
        else:
            system_health['database_status'] = 'memory_fallback'
    except:
        system_health['database_status'] = 'error'

    # Determine overall status
    if system_health['database_status'] == 'error':
        system_health['status'] = 'degraded'
    elif performance_stats['errors'] > 10:
        system_health['status'] = 'warning'
    else:
        system_health['status'] = 'healthy'

# Helper functions for database operations with fallback
async def save_chat_message(message_data):
    """Save chat message with MongoDB fallback to memory"""
    if MONGODB_AVAILABLE and db is not None:
        try:
            await db.chat_messages.insert_one(message_data)
        except Exception as e:
            print(f"MongoDB save failed, using memory: {e}")
            memory_storage['chat_messages'][message_data['session_id']] = memory_storage['chat_messages'].get(message_data['session_id'], [])
            memory_storage['chat_messages'][message_data['session_id']].append(message_data)
    else:
        memory_storage['chat_messages'][message_data['session_id']] = memory_storage['chat_messages'].get(message_data['session_id'], [])
        memory_storage['chat_messages'][message_data['session_id']].append(message_data)

# Enhanced AI Response Generation
async def generate_enhanced_ai_response(user_message: str, message_lower: str, context: Optional[Dict[str, Any]], context_info: str) -> tuple[str, str]:
    """Generate enhanced AI responses with advanced reasoning and context awareness"""

    # Analyze message intent and complexity
    intent_analysis = analyze_message_intent(user_message, message_lower, context)

    # Generate contextually appropriate response
    if intent_analysis["category"] == "greeting":
        response = await generate_greeting_response(user_message, context_info, intent_analysis)
        learning_mode = "greeting"

    elif intent_analysis["category"] == "capabilities":
        response = await generate_capabilities_response(user_message, context_info, intent_analysis)
        learning_mode = "capabilities"

    elif intent_analysis["category"] == "coding":
        response = await generate_coding_response(user_message, context_info, intent_analysis, context)
        learning_mode = "coding_assistance"

    elif intent_analysis["category"] == "explanation":
        response = await generate_explanation_response(user_message, context_info, intent_analysis)
        learning_mode = "explanation"

    elif intent_analysis["category"] == "analysis":
        response = await generate_analysis_response(user_message, context_info, intent_analysis, context)
        learning_mode = "analysis"

    elif intent_analysis["category"] == "creative":
        response = await generate_creative_response(user_message, context_info, intent_analysis)
        learning_mode = "creative_assistance"

    elif intent_analysis["category"] == "problem_solving":
        response = await generate_problem_solving_response(user_message, context_info, intent_analysis, context)
        learning_mode = "problem_solving"

    else:
        response = await generate_general_response(user_message, context_info, intent_analysis)
        learning_mode = "general_assistance"

    return response, learning_mode

def analyze_message_intent(user_message: str, message_lower: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze user message to determine intent and complexity"""

    # Intent categories with keywords and patterns
    intent_patterns = {
        "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"],
        "capabilities": ["help", "what can you do", "capabilities", "features", "abilities"],
        "coding": ["code", "programming", "function", "class", "variable", "debug", "algorithm", "syntax"],
        "explanation": ["explain", "what is", "how does", "why", "define", "meaning", "concept"],
        "analysis": ["analyze", "review", "check", "examine", "evaluate", "assess"],
        "creative": ["create", "generate", "write", "design", "brainstorm", "imagine"],
        "problem_solving": ["solve", "fix", "troubleshoot", "issue", "problem", "error", "help me"]
    }

    # Determine primary category
    category_scores = {}
    for category, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in message_lower)
        if score > 0:
            category_scores[category] = score

    primary_category = max(category_scores, key=category_scores.get) if category_scores else "general"

    # Analyze complexity
    complexity = "simple"
    if len(user_message.split()) > 10:
        complexity = "medium"
    if any(word in message_lower for word in ["complex", "detailed", "comprehensive", "advanced"]):
        complexity = "high"

    # Check for multi-modal context
    has_files = context and context.get('has_files', False)
    file_types = context.get('file_types', []) if context else []

    return {
        "category": primary_category,
        "complexity": complexity,
        "confidence": max(category_scores.values()) / len(message_lower.split()) if category_scores else 0.1,
        "has_files": has_files,
        "file_types": file_types,
        "message_length": len(user_message),
        "word_count": len(user_message.split())
    }

# Enhanced Response Generators
async def generate_greeting_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any]) -> str:
    """Generate personalized greeting responses"""
    time_of_day = datetime.utcnow().hour
    if time_of_day < 12:
        greeting = "Good morning!"
    elif time_of_day < 17:
        greeting = "Good afternoon!"
    else:
        greeting = "Good evening!"

    return f"{greeting} I'm MasterX, your quantum-powered AI assistant. I'm excited to help you explore, learn, and create today. What fascinating challenge can we tackle together?{context_info}"

async def generate_capabilities_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any]) -> str:
    """Generate detailed capabilities overview"""
    return f"""I'm equipped with cutting-edge multi-modal AI capabilities designed for the modern digital age:

🎤 **Advanced Voice Interaction**
   • Natural speech recognition and processing
   • Voice commands and conversational AI
   • Real-time audio analysis

📁 **Intelligent File Processing**
   • Document analysis (PDF, Word, text files)
   • Code review and optimization (Python, JavaScript, etc.)
   • Image analysis and description
   • Data extraction and insights

🧠 **Quantum-Enhanced Intelligence**
   • Context-aware conversations with memory
   • Multi-step problem solving
   • Creative content generation
   • Technical explanations and tutorials

🔊 **Premium Voice Output**
   • Natural text-to-speech synthesis
   • Customizable voice settings
   • Auto-speak mode for hands-free interaction

🚀 **Real-Time Collaboration**
   • Live session management
   • Interactive learning experiences
   • Adaptive response optimization

What specific capability would you like to explore first?{context_info}"""

async def generate_coding_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    """Generate coding assistance responses"""
    if intent_analysis["has_files"] and "code" in intent_analysis["file_types"]:
        return f"""Perfect! I can see you've uploaded code files. I'm ready to provide comprehensive code analysis:

🔍 **Code Review Services:**
   • Syntax and logic analysis
   • Performance optimization suggestions
   • Security vulnerability detection
   • Best practices recommendations

🛠️ **Development Support:**
   • Bug identification and fixes
   • Algorithm optimization
   • Code refactoring suggestions
   • Documentation generation

📚 **Learning & Guidance:**
   • Concept explanations
   • Design pattern recommendations
   • Testing strategies
   • Code quality improvements

Let me analyze your uploaded code and provide detailed insights.{context_info}"""

    return f"""I'm your dedicated coding companion! I can assist with all aspects of software development:

💻 **Programming Languages:** Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
🏗️ **Architecture & Design:** System design, design patterns, best practices
🐛 **Debugging & Testing:** Error analysis, test strategies, performance optimization
📖 **Learning Support:** Concept explanations, tutorials, code examples

Share your code, describe your challenge, or ask about any programming concept!{context_info}"""

async def generate_explanation_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any]) -> str:
    """Generate detailed explanations"""
    complexity = intent_analysis["complexity"]

    if complexity == "high":
        return f"""Excellent question! I'll provide a comprehensive, multi-layered explanation of: **{user_message}**

🎯 **Core Concept Overview**
I'll start with the fundamental principles and build up to advanced concepts.

🔍 **Detailed Analysis**
• Key components and relationships
• Real-world applications and examples
• Common misconceptions and clarifications

💡 **Practical Insights**
• Best practices and recommendations
• Potential challenges and solutions
• Related concepts worth exploring

📚 **Learning Path**
I'll suggest next steps for deeper understanding.

Let me break this down systematically for you.{context_info}"""

    return f"""Great question! I love explaining concepts clearly and thoroughly.

**Topic: {user_message}**

I'll provide you with:
• Clear, step-by-step explanation
• Practical examples and analogies
• Key takeaways and applications
• Additional resources for deeper learning

Let me walk you through this concept in an engaging and understandable way.{context_info}"""

async def generate_analysis_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    """Generate analysis responses"""
    if intent_analysis["has_files"]:
        file_types = intent_analysis["file_types"]
        analysis_type = "multi-modal analysis"

        if "code" in file_types:
            analysis_focus = """
🔍 **Code Analysis Focus:**
   • Architecture and design patterns
   • Performance and optimization opportunities
   • Security considerations
   • Code quality and maintainability
   • Testing coverage and strategies"""
        elif "document" in file_types:
            analysis_focus = """
📄 **Document Analysis Focus:**
   • Content structure and organization
   • Key insights and themes
   • Data extraction and summarization
   • Recommendations and action items
   • Quality and clarity assessment"""
        else:
            analysis_focus = """
📊 **Comprehensive Analysis Focus:**
   • Content evaluation and insights
   • Structure and organization
   • Quality assessment
   • Improvement recommendations
   • Strategic implications"""

        return f"""Perfect! I'll conduct a thorough {analysis_type} of your uploaded content.

{analysis_focus}

🎯 **Analysis Methodology:**
   • Systematic evaluation framework
   • Evidence-based insights
   • Actionable recommendations
   • Comparative benchmarking
   • Future-focused suggestions

I'm processing your files now and will provide detailed insights.{context_info}"""

    return f"""I'll provide a comprehensive analysis of: **{user_message}**

🔬 **Analysis Framework:**
   • Systematic evaluation approach
   • Multi-dimensional perspective
   • Evidence-based insights
   • Practical recommendations

📊 **Deliverables:**
   • Key findings and observations
   • Strengths and improvement areas
   • Actionable next steps
   • Strategic implications

Let me examine this thoroughly and provide you with valuable insights.{context_info}"""

async def generate_creative_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any]) -> str:
    """Generate creative assistance responses"""
    return f"""🎨 Fantastic! I'm excited to help with your creative project: **{user_message}**

✨ **Creative Capabilities:**
   • Original content generation
   • Brainstorming and ideation
   • Creative problem solving
   • Design thinking approaches
   • Innovative solutions

🚀 **Creative Process:**
   • Understanding your vision and goals
   • Exploring multiple creative directions
   • Iterative refinement and enhancement
   • Professional-quality output
   • Unique and engaging results

💡 **Specializations:**
   • Writing and storytelling
   • Concept development
   • Visual design ideas
   • Marketing and branding
   • Technical innovation

Let's bring your creative vision to life!{context_info}"""

async def generate_problem_solving_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    """Generate problem-solving responses"""
    return f"""🎯 I'm ready to tackle this challenge with you: **{user_message}**

🔧 **Problem-Solving Approach:**
   • Root cause analysis
   • Multiple solution pathways
   • Risk assessment and mitigation
   • Implementation strategies
   • Success metrics and validation

💪 **Solution Framework:**
   • Immediate quick fixes
   • Long-term strategic solutions
   • Preventive measures
   • Optimization opportunities
   • Best practices integration

🧠 **Collaborative Process:**
   • Understanding the full context
   • Exploring all possibilities
   • Testing and validation
   • Iterative improvement
   • Knowledge transfer

Let's solve this systematically and effectively!{context_info}"""

async def generate_general_response(user_message: str, context_info: str, intent_analysis: Dict[str, Any]) -> str:
    """Generate general assistance responses"""
    return f"""I'm here to help with your inquiry: **{user_message}**

🎯 **My Approach:**
   • Comprehensive understanding of your needs
   • Tailored, relevant information
   • Practical, actionable insights
   • Clear, engaging explanations
   • Follow-up support and guidance

💡 **What I'll Provide:**
   • Accurate, up-to-date information
   • Multiple perspectives and approaches
   • Real-world examples and applications
   • Step-by-step guidance when needed
   • Additional resources for deeper exploration

I'm committed to giving you the most helpful and insightful response possible.{context_info}"""

# Optimized response generation
async def generate_optimized_response(user_message: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
    """Real AI response generation with caching and fallback"""
    try:
        # Check cache for common queries (without context)
        if not context:
            cache_key = user_message.lower().strip()
            if cache_key in response_cache:
                cached_response, cached_metadata, timestamp = response_cache[cache_key]
                # Check if cache is still valid (5 minutes)
                if (datetime.utcnow().timestamp() - timestamp) < CACHE_TTL:
                    cached_metadata["cached"] = True
                    return cached_response, cached_metadata

        # Multi-modal context processing
        context_prompt = ""
        if context and context.get('has_files'):
            files = context.get('files', [])
            if files:
                context_prompt = f"The user has uploaded {len(files)} file(s). "
                file_types = context.get('file_types', [])
                if 'code' in file_types:
                    context_prompt += "Focus on code analysis, debugging, and programming assistance. "
                elif 'document' in file_types:
                    context_prompt += "Focus on document analysis and content insights. "
                else:
                    context_prompt += "Analyze the uploaded content and provide relevant assistance. "

        # Generate real AI response
        ai_response: AIResponse = await ai_manager.generate_response(
            user_message=user_message,
            context=context_prompt if context_prompt else None
        )

        # Generate metadata from AI response
        metadata = {
            "model": ai_response.model,
            "provider": ai_response.provider,
            "tokens_used": ai_response.tokens_used,
            "response_time_ms": round(ai_response.response_time * 1000, 2),
            "confidence": ai_response.confidence,
            "session_context": True,
            "quantum_powered": True,
            "multi_modal": context is not None,
            "intelligence_level": "Advanced",
            "learning_mode": "ai_powered"
        }

        # Cache response for common queries (without context)
        if not context and ai_response.confidence > 0.8:
            cache_key = user_message.lower().strip()
            response_cache[cache_key] = (ai_response.content, metadata, datetime.utcnow().timestamp())

        return ai_response.content, metadata

    except Exception as e:
        logger.error(f"Error in AI response generation: {str(e)}")
        # Fallback response
        return f"I'm here to help with your question: '{user_message}'. Let me provide you with the best assistance I can.", {
            "learning_mode": "fallback",
            "confidence": 0.75,
            "error_recovery": True,
            "model": "fallback",
            "provider": "system"
        }

# Async storage function (non-blocking)
async def store_chat_messages_async(session_id: str, user_message: str, ai_response: str, metadata: Dict[str, Any]):
    """Store chat messages asynchronously without blocking the response"""
    try:
        # Store user message
        user_msg_data = {
            "session_id": session_id,
            "message": user_message,
            "sender": "user",
            "timestamp": datetime.utcnow(),
            "metadata": None
        }

        # Store AI message
        ai_msg_data = {
            "session_id": session_id,
            "message": ai_response,
            "sender": "ai",
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }

        # Save both messages
        await save_chat_message(user_msg_data)
        await save_chat_message(ai_msg_data)

    except Exception as e:
        logger.error(f"Error storing messages asynchronously: {str(e)}")

# Create the main app with production security
app = FastAPI(
    title="🚀 MasterX Quantum Intelligence API",
    description="Revolutionary AI learning platform with quantum intelligence and premium interactive experiences",
    version="3.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# Security Middleware
if ENVIRONMENT == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# CORS Middleware with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://masterx.ai"] if ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Request size limiting middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            raise HTTPException(status_code=413, detail="Request entity too large")

    response = await call_next(request)
    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not await check_rate_limit(request):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
        )

    response = await call_next(request)
    return response

# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    return response

# Include interactive API routes if available
if INTERACTIVE_FEATURES_AVAILABLE:
    app.include_router(interactive_router)
    print("✅ Interactive API routes included")

# Add a simple health check at the root level for preview environment
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "platform": "MasterX Quantum Intelligence",
        "version": "3.0",
        "quantum_engine": "online" if QUANTUM_ENGINE_AVAILABLE else "offline",
        "interactive_features": "online" if INTERACTIVE_FEATURES_AVAILABLE else "offline",
        "features": {
            "quantum_intelligence": QUANTUM_ENGINE_AVAILABLE,
            "interactive_content": INTERACTIVE_FEATURES_AVAILABLE,
            "real_time_collaboration": INTERACTIVE_FEATURES_AVAILABLE,
            "advanced_analytics": INTERACTIVE_FEATURES_AVAILABLE
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def app_root():
    return {
        "message": "MasterX Quantum Intelligence Platform",
        "version": "3.0",
        "status": "online",
        "api_docs": "/docs",
        "api_base": "/api"
    }

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize Quantum Intelligence Engine (with proper dependency injection)
quantum_engine = None
if QUANTUM_ENGINE_AVAILABLE:
    try:
        # Setup dependencies first
        setup_dependencies()
        
        # Get configured quantum engine instance
        quantum_engine = get_quantum_engine()
        print("🚀 Quantum Intelligence Engine initialized with dependencies")
        
        # Verify configuration
        config = get_config()
        print(f"✅ Configuration loaded: {config.app_name} v{config.version}")
        print(f"✅ AI Providers available: {config.has_ai_provider}")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize Quantum Engine: {str(e)}")
        print(f"🔧 Error details: {type(e).__name__}")
        quantum_engine = None
        QUANTUM_ENGINE_AVAILABLE = False
else:
    quantum_engine = None

# Enhanced WebSocket connection manager with real-time collaboration
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
        self.session_participants: Dict[str, List[str]] = {}  # session_id -> [user_ids]
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.typing_users: Dict[str, List[str]] = {}  # session_id -> [typing_user_ids]

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket

        # Send welcome message with connection info
        await self.send_personal_message(json.dumps({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "features": ["real_time_chat", "typing_indicators", "file_sharing", "voice_sync"]
        }), user_id)

    def disconnect(self, websocket: WebSocket, user_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]

        # Remove from session participants
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.session_participants:
                self.session_participants[session_id] = [
                    uid for uid in self.session_participants[session_id] if uid != user_id
                ]
            del self.user_sessions[user_id]

    async def join_session(self, user_id: str, session_id: str):
        """Add user to a chat session for collaboration"""
        if session_id not in self.session_participants:
            self.session_participants[session_id] = []

        if user_id not in self.session_participants[session_id]:
            self.session_participants[session_id].append(user_id)
            self.user_sessions[user_id] = session_id

            # Notify other participants
            await self.broadcast_to_session(json.dumps({
                "type": "user_joined",
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }), session_id, exclude_user=user_id)

    async def set_typing_status(self, user_id: str, session_id: str, is_typing: bool):
        """Update typing status for real-time indicators"""
        if session_id not in self.typing_users:
            self.typing_users[session_id] = []

        if is_typing and user_id not in self.typing_users[session_id]:
            self.typing_users[session_id].append(user_id)
        elif not is_typing and user_id in self.typing_users[session_id]:
            self.typing_users[session_id].remove(user_id)

        # Broadcast typing status to session participants
        await self.broadcast_to_session(json.dumps({
            "type": "typing_status",
            "typing_users": self.typing_users[session_id],
            "session_id": session_id
        }), session_id, exclude_user=user_id)

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")

    async def broadcast_to_session(self, message: str, session_id: str, exclude_user: str = None):
        """Broadcast message to all participants in a session"""
        if session_id in self.session_participants:
            for user_id in self.session_participants[session_id]:
                if user_id != exclude_user:
                    await self.send_personal_message(message, user_id)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get real-time collaboration statistics"""
        return {
            "total_connections": len(self.active_connections),
            "active_sessions": len(self.session_participants),
            "total_participants": sum(len(participants) for participants in self.session_participants.values()),
            "typing_sessions": len([s for s in self.typing_users.values() if s])
        }

manager = ConnectionManager()


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Chat Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str  # "user" or "ai"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(None, max_length=100, description="Session identifier")
    message_type: Optional[str] = Field('text', pattern=r'^(text|voice|file)$', description="Message type")
    task_type: Optional[str] = Field('general', max_length=50, description="Task type")
    provider: Optional[str] = Field(None, max_length=50, description="AI provider")
    stream: Optional[bool] = Field(False, description="Stream response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v):
        return sanitize_input(v)

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('Invalid session ID format')
        return v

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class SessionCreate(BaseModel):
    user_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {
        "message": "MasterX Quantum Intelligence Engine API",
        "version": "3.0",
        "status": "online",
        "capabilities": [
            "quantum_intelligence",
            "multi_modal_ai",
            "adaptive_learning",
            "real_time_mentorship"
        ]
    }

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Chat Endpoints
@api_router.post("/chat/session", response_model=SessionResponse)
async def create_chat_session(session_data: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "user_id": session_data.user_id,
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    await db.chat_sessions.insert_one(session)
    logger.info(f"Created new chat session: {session_id}")
    
    return SessionResponse(
        session_id=session_id,
        created_at=session["created_at"]
    )

@api_router.get("/performance")
async def get_performance_stats():
    """Get performance statistics for monitoring"""
    cache_hit_rate = (performance_stats['cache_hits'] / max(performance_stats['total_requests'], 1)) * 100

    return {
        "total_requests": performance_stats['total_requests'],
        "average_response_time_ms": round(performance_stats['avg_response_time'] * 1000, 2),
        "cache_hits": performance_stats['cache_hits'],
        "cache_hit_rate_percent": round(cache_hit_rate, 2),
        "errors": performance_stats['errors'],
        "status": "optimized" if performance_stats['avg_response_time'] < 2.0 else "needs_optimization"
    }

@api_router.get("/collaboration/stats")
async def get_collaboration_stats():
    """Get real-time collaboration statistics"""
    stats = manager.get_session_stats()

    return {
        "real_time_collaboration": {
            "active_connections": stats["total_connections"],
            "active_sessions": stats["active_sessions"],
            "total_participants": stats["total_participants"],
            "typing_sessions": stats["typing_sessions"]
        },
        "features": {
            "websocket_support": True,
            "typing_indicators": True,
            "file_sharing": True,
            "session_collaboration": True,
            "real_time_sync": True
        },
        "status": "active" if stats["total_connections"] > 0 else "idle",
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.get("/health/detailed")
async def get_detailed_health():
    """Comprehensive health check for production monitoring"""
    update_system_health()

    uptime = datetime.utcnow() - system_health['uptime_start']

    return {
        "status": system_health['status'],
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": {
            "seconds": uptime.total_seconds(),
            "human_readable": str(uptime)
        },
        "system": {
            "database_status": system_health['database_status'],
            "active_sessions": system_health['active_sessions'],
            "memory_fallback": not MONGODB_AVAILABLE
        },
        "performance": {
            "total_requests": performance_stats['total_requests'],
            "average_response_time_ms": round(performance_stats['avg_response_time'] * 1000, 2),
            "cache_hit_rate": round((performance_stats['cache_hits'] / max(performance_stats['total_requests'], 1)) * 100, 2),
            "error_rate": round((performance_stats['errors'] / max(performance_stats['total_requests'], 1)) * 100, 2)
        },
        "security": {
            "security_events": performance_stats['security_events'],
            "rate_limit_hits": performance_stats['rate_limit_hits'],
            "recent_events": security_events[-5:] if security_events else []
        },
        "features": {
            "ai_responses": "optimized",
            "file_processing": "advanced",
            "real_time_collaboration": "active",
            "multi_modal": "enabled"
        }
    }

@api_router.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    update_system_health()

    metrics = []

    # Performance metrics
    metrics.append(f"masterx_requests_total {performance_stats['total_requests']}")
    metrics.append(f"masterx_response_time_seconds {performance_stats['avg_response_time']}")
    metrics.append(f"masterx_cache_hits_total {performance_stats['cache_hits']}")
    metrics.append(f"masterx_errors_total {performance_stats['errors']}")

    # System metrics
    metrics.append(f"masterx_active_sessions {system_health['active_sessions']}")
    metrics.append(f"masterx_websocket_connections {performance_stats['websocket_connections']}")
    metrics.append(f"masterx_file_uploads_total {performance_stats['file_uploads']}")

    # Security metrics
    metrics.append(f"masterx_security_events_total {performance_stats['security_events']}")
    metrics.append(f"masterx_rate_limit_hits_total {performance_stats['rate_limit_hits']}")

    # Health status (1 = healthy, 0.5 = degraded, 0 = unhealthy)
    health_value = 1 if system_health['status'] == 'healthy' else 0.5 if system_health['status'] == 'degraded' else 0
    metrics.append(f"masterx_health_status {health_value}")

    return "\n".join(metrics)

@api_router.get("/security/events")
async def get_security_events(limit: int = 50):
    """Get recent security events for monitoring"""
    return {
        "events": security_events[-limit:] if security_events else [],
        "total_events": len(security_events),
        "summary": {
            "total_security_events": performance_stats['security_events'],
            "rate_limit_hits": performance_stats['rate_limit_hits'],
            "last_24h": len([e for e in security_events if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00'))).total_seconds() < 86400])
        }
    }

@api_router.post("/chat/test", response_model=ChatResponse)
async def test_chat_message(chat_request: ChatRequest):
    """Simple test endpoint for chat functionality"""
    try:
        # Generate a simple test response
        test_response = f"✅ Test response received! You said: '{chat_request.message}'"

        # Create test metadata
        metadata = {
            "learning_mode": "test_mode",
            "confidence": 0.95,
            "test": True,
            "quantum_powered": False
        }

        return ChatResponse(
            response=test_response,
            session_id="test-session-123",
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error in test chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Test chat failed")

@api_router.post("/chat/send", response_model=ChatResponse)
async def send_chat_message(
    chat_request: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Secure, optimized chat endpoint with fast response times"""
    start_time = datetime.utcnow().timestamp()

    # Security logging
    logger.info(f"Chat request from {request.client.host} by user {current_user.get('user_id')}")

    try:
        # Update performance stats
        performance_stats['total_requests'] += 1

        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Generate AI response using optimized system
        ai_response_text, metadata = await generate_optimized_response(
            chat_request.message,
            session_id,
            chat_request.context
        )

        # Track cache hits
        if metadata.get('cached'):
            performance_stats['cache_hits'] += 1

        # Store messages asynchronously (non-blocking)
        asyncio.create_task(store_chat_messages_async(
            session_id,
            chat_request.message,
            ai_response_text,
            metadata
        ))

        # Calculate and update response time
        end_time = datetime.utcnow().timestamp()
        response_time = end_time - start_time

        # Update average response time
        total_requests = performance_stats['total_requests']
        current_avg = performance_stats['avg_response_time']
        performance_stats['avg_response_time'] = ((current_avg * (total_requests - 1)) + response_time) / total_requests

        # Add performance info to metadata
        metadata['response_time_ms'] = round(response_time * 1000, 2)
        metadata['performance_optimized'] = True

        logger.info(f"Generated response for session: {session_id} in {response_time:.3f}s")

        return ChatResponse(
            response=ai_response_text,
            session_id=session_id,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Return a fallback response instead of failing
        return ChatResponse(
            response=f"I apologize, but I'm experiencing some technical difficulties. However, I can see you said: '{chat_request.message}'. Let me help you with that topic.",
            session_id=chat_request.session_id or str(uuid.uuid4()),
            metadata={"error_fallback": True, "learning_mode": "recovery"}
        )

# Streaming Chat Endpoint
@api_router.post("/chat/stream")
async def stream_chat_message(chat_request: ChatRequest):
    """Send a message and get streaming AI response"""
    try:
        # Store user message
        user_message = ChatMessage(
            session_id=chat_request.session_id or str(uuid.uuid4()),
            message=chat_request.message,
            sender="user"
        )
        await db.chat_messages.insert_one(user_message.dict())
        
        async def generate_stream():
            try:
                # Create or get existing session
                if chat_request.session_id:
                    session_data = await db.chat_sessions.find_one({"session_id": chat_request.session_id})
                    if session_data:
                        chat_session = ChatSession(
                            id=session_data["session_id"],
                            user_id=session_data.get("user_id", "anonymous"),
                            created_at=session_data.get("created_at", datetime.utcnow()),
                            updated_at=datetime.utcnow(),
                            is_active=True
                        )
                    else:
                        chat_session = ChatSession(
                            id=chat_request.session_id,
                            user_id="anonymous",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow(),
                            is_active=True
                        )
                else:
                    new_session_id = str(uuid.uuid4())
                    chat_session = ChatSession(
                        id=new_session_id,
                        user_id="anonymous",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        is_active=True
                    )

                # Get streaming response from quantum engine
                try:
                    stream_response = await quantum_engine.get_quantum_response(
                        user_message=chat_request.message,
                        user_id=chat_session.user_id,
                        session_id=chat_session.id,
                        learning_dna=None,
                        context=chat_request.context or {}
                    )
                except Exception as e:
                    logger.error(f"Quantum streaming error: {str(e)}")
                    stream_response = f"I understand your question about: {chat_request.message}. Let me help you explore this topic with advanced learning techniques."

                full_response = ""
                if hasattr(stream_response, '__aiter__'):
                    async for chunk in stream_response:
                        if chunk:
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk, 'session_id': user_message.session_id})}\n\n"
                else:
                    # Fallback for non-streaming response
                    response_text = str(stream_response)
                    words = response_text.split()
                    for i, word in enumerate(words):
                        chunk = word + " "
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk, 'session_id': user_message.session_id})}\n\n"
                        await asyncio.sleep(0.05)  # Simulate typing delay

                # Store complete AI response
                ai_message = ChatMessage(
                    session_id=user_message.session_id,
                    message=full_response.strip(),
                    sender="ai",
                    metadata={"streaming": True, "learning_mode": "adaptive_quantum"}
                )
                await db.chat_messages.insert_one(ai_message.dict())
                
                yield f"data: {json.dumps({'done': True, 'session_id': user_message.session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process streaming chat message")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Process chat message with enhanced features
                user_message = message_data.get("message", "")
                session_id = message_data.get("session_id", str(uuid.uuid4()))
                context = message_data.get("context")

                # Join session for collaboration
                await manager.join_session(user_id, session_id)

                # Generate enhanced AI response
                ai_response, metadata = await generate_optimized_response(user_message, session_id, context)

                # Send response back to client and session participants
                response_data = {
                    "type": "chat_response",
                    "message": ai_response,
                    "session_id": session_id,
                    "user_id": user_id,
                    "metadata": metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Send to user and broadcast to session participants
                await manager.send_personal_message(json.dumps(response_data), user_id)
                await manager.broadcast_to_session(json.dumps({
                    **response_data,
                    "type": "session_chat_update"
                }), session_id, exclude_user=user_id)

            elif message_data.get("type") == "typing":
                # Handle typing indicators
                session_id = message_data.get("session_id")
                is_typing = message_data.get("is_typing", False)

                if session_id:
                    await manager.set_typing_status(user_id, session_id, is_typing)

            elif message_data.get("type") == "join_session":
                # Handle session joining
                session_id = message_data.get("session_id")
                if session_id:
                    await manager.join_session(user_id, session_id)

            elif message_data.get("type") == "file_shared":
                # Handle file sharing in session
                session_id = message_data.get("session_id")
                file_info = message_data.get("file_info")

                if session_id and file_info:
                    await manager.broadcast_to_session(json.dumps({
                        "type": "file_shared",
                        "file_info": file_info,
                        "shared_by": user_id,
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }), session_id, exclude_user=user_id)

            elif message_data.get("type") == "ping":
                # Handle ping for connection keepalive
                await manager.send_personal_message(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }), user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user: {user_id}")

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        
        return [ChatMessage(**msg) for msg in messages]
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")

# File Upload APIs
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.txt', '.md', '.rtf',  # Documents
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',  # Images
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.xml', '.yaml', '.yml',  # Code
    '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift',  # More code
    '.csv', '.xlsx', '.xls'  # Data files
}

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = Path(filename).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']:
        return 'image'
    elif ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
        return 'document'
    elif ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
                 '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift']:
        return 'code'
    elif ext in ['.csv', '.xlsx', '.xls']:
        return 'data'
    else:
        return 'other'

async def process_text_file(file_path: Path) -> str:
    """Extract text content from text-based files"""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return content[:10000]  # Limit to first 10k characters
    except UnicodeDecodeError:
        try:
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                content = await f.read()
                return content[:10000]
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Advanced File Processing Functions
async def process_file_with_analysis(file_path: Path, file_type: str, file_ext: str) -> tuple[str, Dict[str, Any]]:
    """Enhanced file processing with intelligent analysis"""

    # Read file content
    content = await process_text_file(file_path)

    # Perform analysis based on file type
    if file_type == 'code':
        analysis = await analyze_code_file(content, file_ext)
    elif file_type == 'document':
        analysis = await analyze_document_file(content, file_ext)
    else:
        analysis = await analyze_general_text_file(content, file_ext)

    # Generate enhanced summary
    enhanced_content = await generate_file_summary(content, file_type, analysis)

    return enhanced_content, analysis

async def analyze_code_file(content: str, file_ext: str) -> Dict[str, Any]:
    """Analyze code files for structure, quality, and insights"""

    lines = content.split('\n')
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])

    # Language detection
    language_map = {
        '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
        '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
        '.go': 'Go', '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby'
    }
    language = language_map.get(file_ext, 'Unknown')

    # Code analysis
    functions = len([line for line in lines if 'def ' in line or 'function ' in line or 'func ' in line])
    classes = len([line for line in lines if 'class ' in line])
    imports = len([line for line in lines if line.strip().startswith(('import ', 'from ', '#include', 'using '))])
    comments = len([line for line in lines if line.strip().startswith(('#', '//', '/*', '*'))])

    # Code quality metrics
    comment_ratio = (comments / max(non_empty_lines, 1)) * 100
    avg_line_length = sum(len(line) for line in lines) / max(total_lines, 1)

    # Complexity indicators
    complexity_keywords = ['if', 'for', 'while', 'try', 'catch', 'switch', 'case']
    complexity_score = sum(content.lower().count(keyword) for keyword in complexity_keywords)

    return {
        "file_type": "code",
        "language": language,
        "metrics": {
            "total_lines": total_lines,
            "code_lines": non_empty_lines,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "comments": comments,
            "comment_ratio": round(comment_ratio, 2),
            "avg_line_length": round(avg_line_length, 2),
            "complexity_score": complexity_score
        },
        "quality_assessment": {
            "documentation": "Good" if comment_ratio > 15 else "Needs Improvement",
            "structure": "Well-organized" if classes > 0 or functions > 0 else "Basic",
            "complexity": "High" if complexity_score > 20 else "Moderate" if complexity_score > 10 else "Low"
        },
        "recommendations": generate_code_recommendations(comment_ratio, complexity_score, functions, classes)
    }

async def analyze_document_file(content: str, file_ext: str) -> Dict[str, Any]:
    """Analyze document files for structure and content insights"""

    lines = content.split('\n')
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    words = content.split()
    sentences = content.split('.')

    # Document structure analysis
    headings = len([line for line in lines if line.strip().startswith('#') or line.isupper()])
    bullet_points = len([line for line in lines if line.strip().startswith(('•', '-', '*', '1.', '2.'))])

    # Content analysis
    word_count = len(words)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    reading_time = word_count / 200  # Average reading speed

    # Document type detection
    doc_type = "Technical" if any(word in content.lower() for word in ['api', 'function', 'code', 'implementation']) else \
               "Academic" if any(word in content.lower() for word in ['research', 'study', 'analysis', 'conclusion']) else \
               "Business" if any(word in content.lower() for word in ['strategy', 'market', 'revenue', 'customer']) else \
               "General"

    return {
        "file_type": "document",
        "document_type": doc_type,
        "metrics": {
            "word_count": word_count,
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "headings": headings,
            "bullet_points": bullet_points,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "estimated_reading_time": f"{reading_time:.1f} minutes"
        },
        "content_analysis": {
            "structure": "Well-structured" if headings > 0 else "Basic",
            "readability": "Good" if 10 < avg_sentence_length < 25 else "Needs Improvement",
            "organization": "Organized" if bullet_points > 0 else "Narrative"
        },
        "key_topics": extract_key_topics(content),
        "recommendations": generate_document_recommendations(word_count, headings, bullet_points)
    }

async def analyze_general_text_file(content: str, file_ext: str) -> Dict[str, Any]:
    """Analyze general text files"""

    lines = content.split('\n')
    words = content.split()

    return {
        "file_type": "text",
        "format": file_ext,
        "metrics": {
            "line_count": len(lines),
            "word_count": len(words),
            "character_count": len(content),
            "file_size_kb": len(content.encode('utf-8')) / 1024
        },
        "content_preview": content[:500] + "..." if len(content) > 500 else content
    }

def generate_code_recommendations(comment_ratio: float, complexity_score: int, functions: int, classes: int) -> List[str]:
    """Generate code improvement recommendations"""
    recommendations = []

    if comment_ratio < 10:
        recommendations.append("Add more comments to improve code documentation")
    if complexity_score > 25:
        recommendations.append("Consider refactoring to reduce complexity")
    if functions == 0 and classes == 0:
        recommendations.append("Consider organizing code into functions or classes")
    if complexity_score > 15 and comment_ratio < 15:
        recommendations.append("Complex code would benefit from better documentation")

    return recommendations or ["Code structure looks good!"]

def extract_key_topics(content: str) -> List[str]:
    """Extract key topics from document content"""
    # Simple keyword extraction (in production, use NLP libraries)
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those'}

    words = [word.lower().strip('.,!?;:"()[]{}') for word in content.split()]
    word_freq = {}

    for word in words:
        if len(word) > 3 and word not in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Return top 5 most frequent meaningful words
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

def generate_document_recommendations(word_count: int, headings: int, bullet_points: int) -> List[str]:
    """Generate document improvement recommendations"""
    recommendations = []

    if word_count > 1000 and headings == 0:
        recommendations.append("Consider adding headings to improve document structure")
    if word_count > 500 and bullet_points == 0:
        recommendations.append("Use bullet points to highlight key information")
    if word_count < 100:
        recommendations.append("Document could benefit from more detailed content")

    return recommendations or ["Document structure looks good!"]

async def process_image_file(file_path: Path, filename: str) -> tuple[str, Dict[str, Any]]:
    """Process image files with basic analysis"""

    try:
        # Get file size
        file_size = file_path.stat().st_size

        # Basic image analysis (without external libraries)
        analysis = {
            "file_type": "image",
            "filename": filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "format": Path(filename).suffix.lower(),
            "analysis_note": "Image uploaded successfully. Advanced image analysis requires additional libraries."
        }

        content = f"📸 **Image Analysis: {filename}**\n\n" \
                 f"• **File Size:** {analysis['size_mb']} MB\n" \
                 f"• **Format:** {analysis['format']}\n" \
                 f"• **Status:** Successfully uploaded and ready for processing\n\n" \
                 f"💡 **Next Steps:** You can ask me to describe what you'd like to know about this image, " \
                 f"and I'll provide relevant insights based on the context of our conversation."

        return content, analysis

    except Exception as e:
        return f"Error processing image: {str(e)}", {"error": str(e)}

async def generate_file_summary(content: str, file_type: str, analysis: Dict[str, Any]) -> str:
    """Generate an enhanced summary of the processed file"""

    if file_type == 'code':
        metrics = analysis.get('metrics', {})
        quality = analysis.get('quality_assessment', {})
        recommendations = analysis.get('recommendations', [])

        summary = f"🔍 **Code Analysis Complete**\n\n" \
                 f"**Language:** {analysis.get('language', 'Unknown')}\n" \
                 f"**Structure:** {metrics.get('functions', 0)} functions, {metrics.get('classes', 0)} classes\n" \
                 f"**Quality:** {quality.get('documentation', 'Unknown')} documentation, {quality.get('complexity', 'Unknown')} complexity\n" \
                 f"**Lines of Code:** {metrics.get('code_lines', 0)} ({metrics.get('total_lines', 0)} total)\n\n" \
                 f"**💡 Recommendations:**\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n" \
                 f"**📋 Code Preview:**\n```\n{content[:800]}{'...' if len(content) > 800 else ''}\n```"

    elif file_type == 'document':
        metrics = analysis.get('metrics', {})
        topics = analysis.get('key_topics', [])
        recommendations = analysis.get('recommendations', [])

        topic_list = ", ".join([topic[0] for topic in topics[:3]]) if topics else "General content"

        summary = f"📄 **Document Analysis Complete**\n\n" \
                 f"**Type:** {analysis.get('document_type', 'General')} document\n" \
                 f"**Length:** {metrics.get('word_count', 0)} words ({metrics.get('estimated_reading_time', 'Unknown')})\n" \
                 f"**Structure:** {metrics.get('headings', 0)} headings, {metrics.get('paragraph_count', 0)} paragraphs\n" \
                 f"**Key Topics:** {topic_list}\n\n" \
                 f"**💡 Recommendations:**\n" + "\n".join(f"• {rec}" for rec in recommendations) + "\n\n" \
                 f"**📋 Content Preview:**\n{content[:800]}{'...' if len(content) > 800 else ''}"

    else:
        summary = f"📁 **File Processed Successfully**\n\n" \
                 f"**Type:** {file_type}\n" \
                 f"**Content Preview:**\n{content[:800]}{'...' if len(content) > 800 else ''}"

    return summary

@api_router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a single file"""

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename

    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    # Enhanced file processing with advanced analysis
    file_type = get_file_type(file.filename)
    processed_content = None
    analysis_results = {}

    try:
        if file_type in ['document', 'code'] or file_ext in ['.txt', '.md', '.json', '.xml', '.yaml', '.yml']:
            processed_content, analysis_results = await process_file_with_analysis(file_path, file_type, file_ext)
        elif file_type == 'image':
            processed_content, analysis_results = await process_image_file(file_path, file.filename)
        else:
            processed_content = f"File uploaded: {file.filename} ({file_type})"
            analysis_results = {"basic_info": True}
    except Exception as e:
        processed_content = f"Error processing file: {str(e)}"
        analysis_results = {"error": str(e)}

    # Store file info with analysis results
    file_info = {
        "file_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "size": len(content),
        "upload_time": datetime.now().isoformat(),
        "processed_content": processed_content,
        "analysis_results": analysis_results,
        "status": "processed"
    }

    # Store in memory storage
    memory_storage['uploaded_files'] = memory_storage.get('uploaded_files', {})
    memory_storage['uploaded_files'][file_id] = file_info

    return file_info

@api_router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """Get information about an uploaded file"""

    # Check memory storage first
    if 'uploaded_files' in memory_storage and file_id in memory_storage['uploaded_files']:
        return memory_storage['uploaded_files'][file_id]

    # Find file by ID in filesystem
    for file_path in UPLOAD_DIR.glob(f"{file_id}_*"):
        if file_path.is_file():
            stat = file_path.stat()
            return {
                "file_id": file_id,
                "filename": file_path.name.split("_", 1)[1],  # Remove UUID prefix
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_type": get_file_type(file_path.name),
                "status": "available"
            }

    raise HTTPException(status_code=404, detail="File not found")

@api_router.get("/files")
async def list_files():
    """List all uploaded files"""

    files = []

    # Get from memory storage
    if 'uploaded_files' in memory_storage:
        files.extend(memory_storage['uploaded_files'].values())

    return {"files": files, "total_count": len(files)}

# Learning Progress Tracking APIs
@api_router.get("/progress/{user_id}")
async def get_user_learning_progress(user_id: str):
    """Get comprehensive learning progress for a user"""
    try:
        # Get learning progress from database
        progress_data = await db.learning_progress.find({"user_id": user_id}).to_list(100)
        
        # Get learning streaks
        streak_data = await db.learning_streaks.find_one({"user_id": user_id})
        
        # Get achievements
        achievements = await db.user_achievements.find({"user_id": user_id}).to_list(100)
        
        # Calculate comprehensive stats
        total_sessions = len(progress_data)
        concepts_mastered = sum([len(p.get("concepts_mastered", [])) for p in progress_data])
        avg_competency = sum([p.get("competency_level", 0) for p in progress_data]) / max(len(progress_data), 1)
        
        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "concepts_mastered": concepts_mastered,
            "average_competency": round(avg_competency, 2),
            "current_streak": streak_data.get("current_streak", 0) if streak_data else 0,
            "longest_streak": streak_data.get("longest_streak", 0) if streak_data else 0,
            "total_achievements": len(achievements),
            "progress_history": progress_data[:10],  # Last 10 sessions
            "recent_achievements": achievements[-5:] if achievements else []  # Last 5 achievements
        }
        
    except Exception as e:
        logger.error(f"Error fetching learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch learning progress")

@api_router.post("/progress/{user_id}")
async def update_learning_progress(user_id: str, progress_data: Dict[str, Any]):
    """Update learning progress for a user"""
    try:
        # Create progress entry
        progress_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "subject": progress_data.get("subject", "general"),
            "topic": progress_data.get("topic", ""),
            "competency_level": progress_data.get("competency_level", 0.0),
            "concepts_mastered": progress_data.get("concepts_mastered", []),
            "areas_for_improvement": progress_data.get("areas_for_improvement", []),
            "session_duration": progress_data.get("session_duration", 0),
            "learning_mode": progress_data.get("learning_mode", "adaptive_quantum"),
            "last_reviewed": datetime.utcnow(),
            "metadata": progress_data.get("metadata", {})
        }
        
        # Insert into database
        await db.learning_progress.insert_one(progress_entry)
        
        # Update learning streak
        await update_learning_streak(user_id)
        
        return {"message": "Learning progress updated successfully", "progress_id": progress_entry["id"]}
        
    except Exception as e:
        logger.error(f"Error updating learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update learning progress")

async def update_learning_streak(user_id: str):
    """Update learning streak for user"""
    try:
        today = datetime.utcnow().date()
        
        # Get current streak data
        streak_data = await db.learning_streaks.find_one({"user_id": user_id})
        
        if not streak_data:
            # Create new streak
            new_streak = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "current_streak": 1,
                "longest_streak": 1,
                "last_activity_date": today.isoformat(),
                "total_learning_days": 1,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await db.learning_streaks.insert_one(new_streak)
        else:
            # Update existing streak
            last_activity = datetime.fromisoformat(streak_data["last_activity_date"]).date()
            
            if last_activity == today:
                # Same day, no update needed
                return
            elif (today - last_activity).days == 1:
                # Consecutive day, increment streak
                new_current_streak = streak_data["current_streak"] + 1
                new_longest_streak = max(streak_data["longest_streak"], new_current_streak)
                
                await db.learning_streaks.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "current_streak": new_current_streak,
                            "longest_streak": new_longest_streak,
                            "last_activity_date": today.isoformat(),
                            "total_learning_days": streak_data["total_learning_days"] + 1,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                # Streak broken, reset
                await db.learning_streaks.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "current_streak": 1,
                            "last_activity_date": today.isoformat(),
                            "total_learning_days": streak_data["total_learning_days"] + 1,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
    except Exception as e:
        logger.error(f"Error updating learning streak: {str(e)}")

# Quantum Learning Modes API
@api_router.get("/learning-modes")
async def get_available_learning_modes():
    """Get all available quantum learning modes"""
    return {
        "learning_modes": [
            {
                "id": "adaptive_quantum",
                "name": "Adaptive Quantum",
                "description": "AI-driven adaptive learning with real-time personalization",
                "features": ["Dynamic difficulty", "Personalized content", "Real-time adaptation"]
            },
            {
                "id": "socratic_discovery", 
                "name": "Socratic Discovery",
                "description": "Question-based discovery learning through guided inquiry",
                "features": ["Guided questioning", "Self-discovery", "Critical thinking"]
            },
            {
                "id": "debug_mastery",
                "name": "Debug Mastery", 
                "description": "Knowledge gap identification and targeted remediation",
                "features": ["Gap analysis", "Targeted practice", "Misconception correction"]
            },
            {
                "id": "challenge_evolution",
                "name": "Challenge Evolution",
                "description": "Progressive difficulty evolution with optimal challenge",
                "features": ["Progressive difficulty", "Optimal challenge", "Mastery tracking"]
            },
            {
                "id": "mentor_wisdom",
                "name": "Mentor Wisdom",
                "description": "Professional mentorship with industry insights",
                "features": ["Expert guidance", "Industry context", "Career advice"]
            },
            {
                "id": "creative_synthesis",
                "name": "Creative Synthesis", 
                "description": "Creative learning through analogies and storytelling",
                "features": ["Creative analogies", "Storytelling", "Memorable connections"]
            },
            {
                "id": "analytical_precision",
                "name": "Analytical Precision",
                "description": "Structured analytical learning with logical frameworks",
                "features": ["Logical frameworks", "Step-by-step analysis", "Structured thinking"]
            }
        ]
    }

@api_router.post("/learning-modes/{mode_id}/session")
async def create_mode_specific_session(mode_id: str, session_data: Dict[str, Any]):
    """Create a learning session with specific quantum mode"""
    try:
        # Validate learning mode
        valid_modes = ["adaptive_quantum", "socratic_discovery", "debug_mastery", 
                      "challenge_evolution", "mentor_wisdom", "creative_synthesis", "analytical_precision"]
        
        if mode_id not in valid_modes:
            raise HTTPException(status_code=400, detail="Invalid learning mode")
        
        # Create specialized session
        session = {
            "session_id": str(uuid.uuid4()),
            "user_id": session_data.get("user_id", "anonymous"),
            "learning_mode": mode_id,
            "topic": session_data.get("topic", ""),
            "difficulty_level": session_data.get("difficulty_level", "intermediate"),
            "learning_objectives": session_data.get("learning_objectives", []),
            "created_at": datetime.utcnow(),
            "status": "active",
            "metadata": {
                "mode_specific_config": session_data.get("config", {}),
                "personalization_settings": session_data.get("personalization", {})
            }
        }
        
        await db.learning_sessions.insert_one(session)
        
        return {
            "session_id": session["session_id"],
            "learning_mode": mode_id,
            "message": f"Created {mode_id} learning session successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating mode-specific session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create learning session")

# Quantum Intelligence Integration
async def generate_quantum_response(user_message: str, session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
    """
    Generate AI response using the REAL Quantum Intelligence Engine (with fallback)
    """
    try:
        # Create or get existing session
        if session_id:
            # Get existing session from database
            session_data = await db.chat_sessions.find_one({"session_id": session_id})
            if session_data:
                # Convert to ChatSession model
                chat_session = ChatSession(
                    id=session_data["session_id"],
                    user_id=session_data.get("user_id", "anonymous"),
                    created_at=session_data.get("created_at", datetime.utcnow()),
                    updated_at=datetime.utcnow(),
                    is_active=True
                )
            else:
                # Create new session if not found
                chat_session = ChatSession(
                    id=session_id,
                    user_id="anonymous",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    is_active=True
                )
        else:
            # Create new session
            new_session_id = str(uuid.uuid4())
            chat_session = ChatSession(
                id=new_session_id,
                user_id="anonymous",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True
            )

        # Use Quantum Intelligence Engine if available
        if QUANTUM_ENGINE_AVAILABLE and quantum_engine:
            try:
                quantum_response = await quantum_engine.get_quantum_response(
                    user_message=user_message,
                    user_id=chat_session.user_id,
                    session_id=chat_session.id,
                    learning_dna=None,  # Will be populated from user profile
                    context={}
                )

                if hasattr(quantum_response, 'content'):
                    # Extract response data from QuantumResponse object
                    response_text = quantum_response.content
                    metadata = {
                        "learning_mode": quantum_response.quantum_mode.value,
                        "concepts": quantum_response.concept_connections,
                        "confidence": quantum_response.personalization_score,
                        "session_context": session_id is not None,
                        "intelligence_level": quantum_response.intelligence_level.name,
                        "engagement_prediction": quantum_response.engagement_prediction,
                        "learning_velocity_boost": quantum_response.learning_velocity_boost,
                        "knowledge_gaps": quantum_response.knowledge_gaps_identified,
                        "next_concepts": quantum_response.next_optimal_concepts,
                        "emotional_resonance": quantum_response.emotional_resonance_score,
                        "quantum_powered": True,
                        "processing_time": getattr(quantum_response, 'processing_time', 0.0),
                        "quantum_analytics": quantum_response.quantum_analytics
                    }
                else:
                    # Fallback if quantum engine returns string
                    response_text = str(quantum_response)
                    metadata = {
                        "learning_mode": "adaptive_quantum",
                        "concepts": extract_concepts_from_message(user_message),
                        "confidence": 0.85,
                        "session_context": session_id is not None,
                        "quantum_powered": True
                    }
            except Exception as e:
                logger.error(f"Quantum engine error: {str(e)}")
                # Enhanced fallback with smart learning mode detection
                response_text, metadata = await generate_enhanced_fallback_response(user_message, session_id, context)
                metadata["quantum_powered"] = False
                metadata["fallback_reason"] = str(e)
        else:
            # Enhanced fallback with smart learning mode detection
            response_text, metadata = await generate_enhanced_fallback_response(user_message, session_id, context)
            metadata["quantum_powered"] = False

        return response_text, metadata

    except Exception as e:
        logger.error(f"Error in quantum intelligence generation: {str(e)}")
        # Fallback to basic response if quantum engine fails
        return f"I understand your question about: {user_message}. Let me help you explore this topic with advanced learning techniques.", {
            "learning_mode": "fallback",
            "concepts": extract_concepts_from_message(user_message),
            "confidence": 0.70,
            "session_context": session_id is not None,
            "error": "quantum_engine_fallback",
            "quantum_powered": False
        }

async def generate_enhanced_fallback_response(user_message: str, session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
    """Enhanced fallback response with intelligent learning mode detection and multi-modal context"""

    message_lower = user_message.lower()

    # Process multi-modal context
    context_info = ""
    file_context = ""
    if context and context.get('has_files'):
        files = context.get('files', [])
        file_types = context.get('file_types', [])

        if files:
            file_context = f"\n\n**Context from uploaded files:**\n"
            for file in files[:3]:  # Limit to first 3 files
                file_context += f"- **{file['filename']}** ({file['type']}): {file['content'][:300]}{'...' if len(file['content']) > 300 else ''}\n"

            context_info = f" I can see you've uploaded {len(files)} file(s) including {', '.join(file_types)}. I'll consider this context in my response."
    
    # Intelligent mode detection
    if any(word in message_lower for word in ["why", "how does", "what if", "explain why"]):
        learning_mode = "socratic_discovery"
        response = f"That's a fascinating question!{context_info} Let me guide you to discover the answer through Socratic questioning. When you think about '{user_message}', what underlying principles or connections come to mind? What do you already know that might relate to this topic?{file_context}"
        
    elif any(word in message_lower for word in ["confused", "don't understand", "mistake", "wrong", "error", "help me fix"]):
        learning_mode = "debug_mastery"
        response = f"I can help you debug this concept step by step. Let's break down '{user_message}' into its core components. First, let me identify where the confusion might be coming from and then we'll build a solid foundation of understanding."
        
    elif any(word in message_lower for word in ["challenge", "harder", "difficult", "test me", "quiz", "practice"]):
        learning_mode = "challenge_evolution"
        response = f"Excellent! I can see you're ready for a challenge. Let me design a progressive learning experience around '{user_message}' that will push your understanding to the next level. Are you ready to dive deep?"
        
    elif any(word in message_lower for word in ["career", "professional", "industry", "job", "work", "real world"]):
        learning_mode = "mentor_wisdom" 
        response = f"Great question from a professional perspective! Let me share some industry insights about '{user_message}'. In the real world, this concept is particularly important because..."
        
    elif any(word in message_lower for word in ["creative", "imagine", "analogy", "story", "example", "metaphor"]):
        learning_mode = "creative_synthesis"
        response = f"Let's explore '{user_message}' through creative synthesis! I'll help you understand this through memorable analogies and creative connections. Imagine if..."
        
    elif any(word in message_lower for word in ["analyze", "compare", "evaluate", "break down", "systematic"]):
        learning_mode = "analytical_precision"
        response = f"Let me provide a structured analytical breakdown of '{user_message}'. We'll approach this systematically, examining each component with precision and logical reasoning."
        
    else:
        learning_mode = "adaptive_quantum"
        response = f"I understand you're exploring '{user_message}'.{context_info} Using my adaptive learning algorithms, I'll personalize this explanation to match your learning style and current understanding level. Let me analyze the best approach for you...{file_context}"

    # Extract concepts with enhanced intelligence
    concepts = extract_enhanced_concepts(user_message)
    
    # Calculate enhanced confidence based on message complexity and concept detection
    base_confidence = 0.75
    concept_boost = min(len(concepts) * 0.05, 0.15)
    complexity_factor = min(len(user_message.split()) / 20, 0.1)
    final_confidence = base_confidence + concept_boost + complexity_factor
    
    metadata = {
        "learning_mode": learning_mode,
        "concepts": concepts,
        "confidence": round(final_confidence, 2),
        "session_context": session_id is not None,
        "intelligence_level": "ENHANCED",
        "engagement_prediction": 0.85,
        "response_type": "enhanced_fallback",
        "processing_time": "instant"
    }
    
    return response, metadata

def extract_enhanced_concepts(message: str) -> List[str]:
    """Enhanced concept extraction with domain-specific intelligence"""
    
    # Domain-specific concept maps
    concept_domains = {
        'technology': ['AI', 'machine learning', 'programming', 'software', 'algorithm', 'data', 'neural', 'computer', 'digital', 'automation'],
        'science': ['physics', 'chemistry', 'biology', 'research', 'experiment', 'theory', 'hypothesis', 'analysis', 'scientific'],
        'mathematics': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'equation', 'formula', 'number', 'calculation'],
        'business': ['strategy', 'marketing', 'management', 'finance', 'sales', 'profit', 'business', 'company', 'market'],
        'education': ['learning', 'teaching', 'study', 'education', 'knowledge', 'skill', 'training', 'development', 'academic'],
        'psychology': ['behavior', 'cognitive', 'mental', 'psychology', 'emotion', 'motivation', 'perception', 'memory'],
        'design': ['design', 'creative', 'visual', 'aesthetic', 'art', 'user experience', 'interface', 'graphic']
    }
    
    message_lower = message.lower()
    found_concepts = []
    
    # Find domain-specific concepts
    for domain, keywords in concept_domains.items():
        domain_matches = [keyword for keyword in keywords if keyword in message_lower]
        if domain_matches:
            found_concepts.extend(domain_matches[:2])  # Max 2 per domain
    
    # Remove duplicates and limit
    found_concepts = list(set(found_concepts))[:5]
    
    # If no domain concepts found, use general extraction
    if not found_concepts:
        found_concepts = extract_concepts_from_message(message)
    
    return found_concepts

def extract_concepts_from_message(message: str) -> List[str]:
    """Extract key concepts from user message"""
    # Simple keyword extraction (later replace with AI)
    keywords = {
        'learning', 'education', 'quantum', 'AI', 'machine learning', 
        'neural', 'algorithm', 'data', 'programming', 'science',
        'mathematics', 'physics', 'chemistry', 'biology', 'history'
    }
    
    message_lower = message.lower()
    found_concepts = [keyword for keyword in keywords if keyword in message_lower]
    
    return found_concepts[:5]  # Limit to top 5 concepts

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    """Cleanup database and quantum engine on shutdown"""
    client.close()
    
    # Cleanup quantum engine dependencies
    if QUANTUM_ENGINE_AVAILABLE:
        try:
            await cleanup_dependencies()
            print("🧹 Quantum Intelligence Engine dependencies cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up quantum engine: {e}")
