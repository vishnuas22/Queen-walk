"""
WebSocket Router for MasterX Quantum Intelligence Platform

Real-time WebSocket API that provides bidirectional communication for
live learning interactions, real-time collaboration, and instant updates.

🔌 WEBSOCKET CAPABILITIES:
- Real-time bidirectional communication
- Live learning sessions
- Collaborative features
- Instant messaging and chat
- Real-time progress updates
- Live notifications and alerts

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException

from ..models import WebSocketMessage, UserProfile
# Authentication disabled for development
from ..auth import get_current_user
from ..utils import WebSocketManager

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# WEBSOCKET SERVICE
# ============================================================================

class WebSocketService:
    """WebSocket service for real-time communication"""
    
    def __init__(self):
        self.manager = WebSocketManager()
        self.active_sessions = {}
        logger.info("🔌 WebSocket Service initialized")
    
    async def authenticate_websocket(self, websocket: WebSocket, token: str) -> Optional[UserProfile]:
        """Authenticate WebSocket connection - Development mode: always allow"""

        try:
            # Development mode: return mock user
            from ..auth import get_current_user
            return get_current_user()
        except Exception as e:
            logger.error(f"WebSocket authentication failed: {e}")
            return None
    
    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """Handle new WebSocket connection"""
        
        connection_id = f"ws_{user_id}_{int(datetime.now().timestamp())}"
        
        try:
            await self.manager.connect(websocket, connection_id, user_id)
            
            # Send welcome message
            welcome_message = {
                "type": "connection",
                "message": "Connected to MasterX real-time service",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_personal_message(
                json.dumps(welcome_message),
                connection_id
            )
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Process message
                    await self.process_message(connection_id, user_id, message_data)
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self.send_error(connection_id, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"WebSocket message processing error: {e}")
                    await self.send_error(connection_id, str(e))
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.manager.disconnect(connection_id, user_id)
    
    async def process_message(self, connection_id: str, user_id: str, message_data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        
        message_type = message_data.get("type")
        
        if message_type == "ping":
            await self.handle_ping(connection_id)
        
        elif message_type == "chat":
            await self.handle_chat_message(connection_id, user_id, message_data)
        
        elif message_type == "learning_update":
            await self.handle_learning_update(connection_id, user_id, message_data)
        
        elif message_type == "join_session":
            await self.handle_join_session(connection_id, user_id, message_data)
        
        elif message_type == "leave_session":
            await self.handle_leave_session(connection_id, user_id, message_data)
        
        else:
            await self.send_error(connection_id, f"Unknown message type: {message_type}")
    
    async def handle_ping(self, connection_id: str):
        """Handle ping message"""
        
        pong_message = {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.manager.send_personal_message(
            json.dumps(pong_message),
            connection_id
        )
    
    async def handle_chat_message(self, connection_id: str, user_id: str, message_data: Dict[str, Any]):
        """Handle chat message"""
        
        try:
            # Process chat message (integrate with chat service)
            chat_response = {
                "type": "chat_response",
                "message_id": message_data.get("message_id"),
                "response": f"Echo: {message_data.get('message', '')}",
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_personal_message(
                json.dumps(chat_response),
                connection_id
            )
            
        except Exception as e:
            await self.send_error(connection_id, f"Chat processing error: {str(e)}")
    
    async def handle_learning_update(self, connection_id: str, user_id: str, message_data: Dict[str, Any]):
        """Handle learning progress update"""
        
        try:
            # Process learning update
            update_response = {
                "type": "learning_update_response",
                "update_id": message_data.get("update_id"),
                "status": "processed",
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_personal_message(
                json.dumps(update_response),
                connection_id
            )
            
            # Broadcast to other user connections if needed
            progress_notification = {
                "type": "progress_notification",
                "user_id": user_id,
                "progress_data": message_data.get("progress_data", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_user_message(
                json.dumps(progress_notification),
                user_id
            )
            
        except Exception as e:
            await self.send_error(connection_id, f"Learning update error: {str(e)}")
    
    async def handle_join_session(self, connection_id: str, user_id: str, message_data: Dict[str, Any]):
        """Handle join learning session"""
        
        try:
            session_id = message_data.get("session_id")
            
            if not session_id:
                await self.send_error(connection_id, "Session ID required")
                return
            
            # Add to session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "participants": [],
                    "created_at": datetime.now()
                }
            
            if user_id not in self.active_sessions[session_id]["participants"]:
                self.active_sessions[session_id]["participants"].append(user_id)
            
            # Notify user
            join_response = {
                "type": "session_joined",
                "session_id": session_id,
                "participants": self.active_sessions[session_id]["participants"],
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_personal_message(
                json.dumps(join_response),
                connection_id
            )
            
            # Notify other participants
            participant_notification = {
                "type": "participant_joined",
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
            for participant_id in self.active_sessions[session_id]["participants"]:
                if participant_id != user_id:
                    await self.manager.send_user_message(
                        json.dumps(participant_notification),
                        participant_id
                    )
            
        except Exception as e:
            await self.send_error(connection_id, f"Join session error: {str(e)}")
    
    async def handle_leave_session(self, connection_id: str, user_id: str, message_data: Dict[str, Any]):
        """Handle leave learning session"""
        
        try:
            session_id = message_data.get("session_id")
            
            if session_id in self.active_sessions:
                if user_id in self.active_sessions[session_id]["participants"]:
                    self.active_sessions[session_id]["participants"].remove(user_id)
                
                # Clean up empty sessions
                if not self.active_sessions[session_id]["participants"]:
                    del self.active_sessions[session_id]
                else:
                    # Notify remaining participants
                    participant_notification = {
                        "type": "participant_left",
                        "session_id": session_id,
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    for participant_id in self.active_sessions[session_id]["participants"]:
                        await self.manager.send_user_message(
                            json.dumps(participant_notification),
                            participant_id
                        )
            
            # Confirm to user
            leave_response = {
                "type": "session_left",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.manager.send_personal_message(
                json.dumps(leave_response),
                connection_id
            )
            
        except Exception as e:
            await self.send_error(connection_id, f"Leave session error: {str(e)}")
    
    async def send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.manager.send_personal_message(
            json.dumps(error_response),
            connection_id
        )
    
    async def broadcast_notification(self, notification: Dict[str, Any]):
        """Broadcast notification to all connections"""
        
        notification_message = {
            "type": "broadcast_notification",
            "notification": notification,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.manager.broadcast(json.dumps(notification_message))

# Initialize service
websocket_service = WebSocketService()

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """Main WebSocket endpoint for real-time communication"""
    
    # Authenticate user
    user = await websocket_service.authenticate_websocket(websocket, token)
    
    if not user:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    
    # Handle connection
    await websocket_service.handle_connection(websocket, user.user_id)

@router.websocket("/learning/{session_id}")
async def learning_session_websocket(websocket: WebSocket, session_id: str, token: str):
    """WebSocket endpoint for live learning sessions"""
    
    # Authenticate user
    user = await websocket_service.authenticate_websocket(websocket, token)
    
    if not user:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    
    connection_id = f"learning_{session_id}_{user.user_id}_{int(datetime.now().timestamp())}"
    
    try:
        await websocket_service.manager.connect(websocket, connection_id, user.user_id)
        
        # Auto-join the learning session
        await websocket_service.handle_join_session(connection_id, user.user_id, {"session_id": session_id})
        
        # Handle session-specific messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                message_data["session_id"] = session_id  # Ensure session context
                
                await websocket_service.process_message(connection_id, user.user_id, message_data)
            
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Learning session WebSocket error: {e}")
                await websocket_service.send_error(connection_id, str(e))
    
    finally:
        # Auto-leave the session
        await websocket_service.handle_leave_session(connection_id, user.user_id, {"session_id": session_id})
        websocket_service.manager.disconnect(connection_id, user.user_id)

@router.post("/broadcast")
async def broadcast_notification(
    notification: Dict[str, Any],
    current_user: UserProfile = Depends(get_current_user)
):
    """Broadcast notification to all WebSocket connections"""
    
    # Check admin permission
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        await websocket_service.broadcast_notification(notification)
        return {"message": "Notification broadcasted successfully"}
    except Exception as e:
        logger.error(f"Broadcast notification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast notification")
