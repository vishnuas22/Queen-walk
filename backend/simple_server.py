#!/usr/bin/env python3
"""
MasterX Simple Chat Server
A lightweight FastAPI server for testing chat functionality without database dependencies.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="MasterX Simple Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    task_type: Optional[str] = "general"
    provider: Optional[str] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    success: bool
    response: str
    message_id: str
    session_id: str
    provider_used: str
    timestamp: str

class SessionCreate(BaseModel):
    user_id: Optional[str] = "default_user"
    task_type: Optional[str] = "general"

class SessionResponse(BaseModel):
    session_id: str
    created_at: str

# In-memory storage for development
sessions: Dict[str, Dict] = {}
messages: List[Dict] = []

# Simple AI response generator
def generate_ai_response(message: str, task_type: str = "general") -> str:
    """Generate a simple AI response based on the message content."""
    
    message_lower = message.lower()
    
    # Programming-related responses
    if any(keyword in message_lower for keyword in ['python', 'code', 'programming', 'function', 'variable']):
        return f"""I'd be happy to help you with Python programming! 

Based on your question about "{message}", here are some key points:

🐍 **Python Best Practices:**
- Use descriptive variable names
- Follow PEP 8 style guidelines
- Write clear, readable code
- Add comments for complex logic

💡 **Quick Tips:**
- Use `print()` for debugging
- Try `help()` function for documentation
- Practice with small examples first

Would you like me to explain any specific Python concept or help you with a particular coding problem?"""

    # General learning responses
    elif any(keyword in message_lower for keyword in ['learn', 'study', 'understand', 'explain']):
        return f"""Great question! I'm here to help you learn and understand complex topics.

For your question: "{message}"

🧠 **Learning Approach:**
1. Break down complex concepts into smaller parts
2. Use examples and analogies
3. Practice with hands-on exercises
4. Ask follow-up questions

📚 **Study Tips:**
- Take notes while learning
- Teach concepts to others
- Apply knowledge to real projects
- Review regularly

What specific aspect would you like me to explain in more detail?"""

    # Greeting responses
    elif any(keyword in message_lower for keyword in ['hello', 'hi', 'hey', 'greetings']):
        return f"""Hello! Welcome to MasterX AI! 👋

I'm your quantum intelligence assistant, ready to help you with:

🚀 **Programming & Development**
- Python, JavaScript, and other languages
- Code debugging and optimization
- Best practices and design patterns

🧠 **Learning & Education**
- Concept explanations
- Study strategies
- Problem-solving techniques

💡 **Creative Problem Solving**
- Brainstorming ideas
- Project planning
- Technical solutions

How can I assist you today?"""

    # Default response
    else:
        return f"""Thank you for your message: "{message}"

I'm MasterX AI, your intelligent learning companion! I can help you with:

✨ **Programming & Development**
✨ **Learning & Education** 
✨ **Problem Solving**
✨ **Creative Projects**

Based on your message, I understand you're interested in {task_type} topics. Could you provide more specific details about what you'd like to learn or accomplish?

Feel free to ask me anything - I'm here to help you succeed! 🌟"""

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "MasterX Simple Chat API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat/send",
            "session": "/api/chat/session"
        }
    }

@app.post("/api/chat/session", response_model=SessionResponse)
async def create_chat_session(session_data: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "user_id": session_data.user_id,
        "task_type": session_data.task_type,
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    
    sessions[session_id] = session
    logger.info(f"Created new session: {session_id}")
    
    return SessionResponse(
        session_id=session_id,
        created_at=session["created_at"]
    )

@app.post("/api/chat/send", response_model=ChatResponse)
async def send_chat_message(chat_request: ChatRequest):
    """Send a message and get AI response"""
    try:
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Generate AI response
        ai_response = generate_ai_response(
            chat_request.message, 
            chat_request.task_type or "general"
        )
        
        # Create response
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Store message (in-memory for development)
        message_data = {
            "message_id": message_id,
            "session_id": session_id,
            "user_message": chat_request.message,
            "ai_response": ai_response,
            "task_type": chat_request.task_type,
            "provider_used": chat_request.provider or "simple_ai",
            "timestamp": timestamp
        }
        messages.append(message_data)
        
        logger.info(f"Processed message for session {session_id}: {chat_request.message[:50]}...")
        
        return ChatResponse(
            success=True,
            response=ai_response,
            message_id=message_id,
            session_id=session_id,
            provider_used=chat_request.provider or "simple_ai",
            timestamp=timestamp
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(sessions),
        "messages_count": len(messages)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
