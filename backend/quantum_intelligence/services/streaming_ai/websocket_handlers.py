"""
WebSocket Handlers for Streaming AI

Real-time WebSocket communication handlers for streaming AI services.
Provides bidirectional communication for live tutoring, instant feedback,
collaboration events, and adaptive content delivery.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import logging
from enum import Enum

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from .data_structures import (
    StreamingEventType, WebSocketMessage, StreamingEvent,
    create_websocket_message, create_streaming_event
)


class WebSocketMessageType(Enum):
    """WebSocket message types"""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    USER_ACTION = "user_action"
    FEEDBACK = "feedback"
    DIFFICULTY_ADJUSTMENT = "difficulty_adjustment"
    COLLABORATION = "collaboration"
    STREAM_QUALITY = "stream_quality"
    NETWORK_CONDITION = "network_condition"
    SESSION_STATE = "session_state"
    PERFORMANCE_ALERT = "performance_alert"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class StreamingWebSocketHandler:
    """
    🔄 STREAMING WEBSOCKET HANDLER
    
    Base WebSocket handler for real-time streaming AI communication.
    Provides low-latency bidirectional communication with automatic
    reconnection, message queuing, and delivery guarantees.
    """
    
    def __init__(self, orchestrator_ref=None):
        self.orchestrator = orchestrator_ref
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.session_participants: Dict[str, Set[str]] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Register default event handlers
        self._register_default_handlers()
        
        logger.info("Streaming WebSocket Handler initialized")
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_event_handler(WebSocketMessageType.CONNECT.value, self._handle_connect)
        self.register_event_handler(WebSocketMessageType.DISCONNECT.value, self._handle_disconnect)
        self.register_event_handler(WebSocketMessageType.USER_ACTION.value, self._handle_user_action)
        self.register_event_handler(WebSocketMessageType.HEARTBEAT.value, self._handle_heartbeat)
        self.register_event_handler(WebSocketMessageType.ERROR.value, self._handle_error)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific message type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def connect(self, user_id: str, session_id: str, connection_info: Dict[str, Any]) -> bool:
        """Handle new WebSocket connection"""
        try:
            # Create connection entry
            connection_id = f"{user_id}_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self.active_connections[connection_id] = {
                'user_id': user_id,
                'session_id': session_id,
                'connection_info': connection_info,
                'connected_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'messages_sent': 0,
                'messages_received': 0
            }
            
            # Create message queue for this connection
            self.message_queues[connection_id] = asyncio.Queue()
            
            # Add to session participants
            if session_id not in self.session_participants:
                self.session_participants[session_id] = set()
            self.session_participants[session_id].add(user_id)
            
            # Track connection stats
            self.connection_stats[connection_id] = {
                'latency_ms': [],
                'message_sizes': [],
                'errors': 0,
                'reconnects': 0
            }
            
            # Trigger connect handlers
            connect_message = create_websocket_message(
                WebSocketMessageType.CONNECT.value,
                user_id,
                session_id,
                {'connection_id': connection_id}
            )
            
            await self._trigger_event_handlers(WebSocketMessageType.CONNECT.value, connect_message)
            
            logger.info(f"WebSocket connection established: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error establishing WebSocket connection: {e}")
            return False
    
    async def disconnect(self, connection_id: str, reason: str = "client_disconnect") -> bool:
        """Handle WebSocket disconnection"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            connection = self.active_connections[connection_id]
            user_id = connection['user_id']
            session_id = connection['session_id']
            
            # Trigger disconnect handlers
            disconnect_message = create_websocket_message(
                WebSocketMessageType.DISCONNECT.value,
                user_id,
                session_id,
                {'reason': reason}
            )
            
            await self._trigger_event_handlers(WebSocketMessageType.DISCONNECT.value, disconnect_message)
            
            # Remove from session participants
            if session_id in self.session_participants and user_id in self.session_participants[session_id]:
                self.session_participants[session_id].remove(user_id)
                if not self.session_participants[session_id]:
                    del self.session_participants[session_id]
            
            # Clean up resources
            if connection_id in self.message_queues:
                del self.message_queues[connection_id]
            
            # Store connection stats before removing
            if connection_id in self.connection_stats:
                # Could persist these stats for analysis
                pass
            
            # Remove connection
            del self.active_connections[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling WebSocket disconnection: {e}")
            return False
    
    async def receive_message(self, connection_id: str, message_data: Dict[str, Any]) -> bool:
        """Process incoming WebSocket message"""
        try:
            if connection_id not in self.active_connections:
                logger.warning(f"Received message for unknown connection: {connection_id}")
                return False
            
            connection = self.active_connections[connection_id]
            connection['last_activity'] = datetime.utcnow().isoformat()
            connection['messages_received'] += 1
            
            # Parse message
            message_type = message_data.get('type', 'unknown')
            
            # Create WebSocket message
            message = WebSocketMessage(
                message_id=message_data.get('message_id', f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"),
                message_type=message_type,
                sender_id=connection['user_id'],
                recipient_id=message_data.get('recipient_id'),
                session_id=connection['session_id'],
                payload=message_data.get('payload', {}),
                priority=message_data.get('priority', 5),
                requires_acknowledgment=message_data.get('requires_acknowledgment', False),
                timestamp=datetime.utcnow()
            )
            
            # Trigger appropriate event handlers
            await self._trigger_event_handlers(message_type, message)
            
            # If message requires acknowledgment, send ack
            if message.requires_acknowledgment:
                await self.send_message(connection_id, {
                    'type': 'ack',
                    'message_id': message.message_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # If this is a user action, forward to orchestrator
            if message_type == WebSocketMessageType.USER_ACTION.value and self.orchestrator:
                event = create_streaming_event(
                    StreamingEventType.USER_ACTION,
                    connection['user_id'],
                    connection['session_id'],
                    message.payload
                )
                await self.orchestrator.add_streaming_event(event.__dict__)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            return False
    
    async def send_message(self, connection_id: str, message_data: Dict[str, Any]) -> bool:
        """Send message to WebSocket client"""
        try:
            if connection_id not in self.active_connections:
                logger.warning(f"Attempted to send message to unknown connection: {connection_id}")
                return False
            
            connection = self.active_connections[connection_id]
            connection['messages_sent'] += 1
            
            # Add to message queue
            if connection_id in self.message_queues:
                await self.message_queues[connection_id].put(message_data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    async def broadcast_to_session(self, session_id: str, message_data: Dict[str, Any]) -> int:
        """Broadcast message to all participants in a session"""
        if session_id not in self.session_participants:
            return 0
        
        sent_count = 0
        for user_id in self.session_participants[session_id]:
            # Find connection for this user in this session
            connection_id = self._find_connection_id(user_id, session_id)
            if connection_id:
                if await self.send_message(connection_id, message_data):
                    sent_count += 1
        
        return sent_count
    
    def _find_connection_id(self, user_id: str, session_id: str) -> Optional[str]:
        """Find connection ID for a user in a session"""
        for conn_id, conn in self.active_connections.items():
            if conn['user_id'] == user_id and conn['session_id'] == session_id:
                return conn_id
        return None
    
    async def _trigger_event_handlers(self, event_type: str, message: WebSocketMessage):
        """Trigger all registered handlers for an event type"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in WebSocket event handler for {event_type}: {e}")
    
    async def _handle_connect(self, message: WebSocketMessage):
        """Handle connection event"""
        logger.info(f"User {message.sender_id} connected to session {message.session_id}")
    
    async def _handle_disconnect(self, message: WebSocketMessage):
        """Handle disconnection event"""
        logger.info(f"User {message.sender_id} disconnected from session {message.session_id}")
    
    async def _handle_user_action(self, message: WebSocketMessage):
        """Handle user action event"""
        logger.debug(f"User action from {message.sender_id} in session {message.session_id}")
    
    async def _handle_heartbeat(self, message: WebSocketMessage):
        """Handle heartbeat message"""
        # Update last activity timestamp
        connection_id = self._find_connection_id(message.sender_id, message.session_id)
        if connection_id and connection_id in self.active_connections:
            self.active_connections[connection_id]['last_activity'] = datetime.utcnow().isoformat()
    
    async def _handle_error(self, message: WebSocketMessage):
        """Handle error message"""
        logger.error(f"WebSocket error from {message.sender_id}: {message.payload.get('error')}")
        
        # Track error in connection stats
        connection_id = self._find_connection_id(message.sender_id, message.session_id)
        if connection_id and connection_id in self.connection_stats:
            self.connection_stats[connection_id]['errors'] += 1


class TutoringSessionHandler(StreamingWebSocketHandler):
    """
    👨‍🏫 TUTORING SESSION HANDLER
    
    Specialized WebSocket handler for live tutoring sessions.
    Manages real-time communication for tutoring interactions,
    participant management, and session state synchronization.
    """
    
    def __init__(self, orchestrator_ref=None):
        super().__init__(orchestrator_ref)
        
        # Register tutoring-specific handlers
        self.register_event_handler('tutor_join', self._handle_tutor_join)
        self.register_event_handler('student_join', self._handle_student_join)
        self.register_event_handler('question_asked', self._handle_question_asked)
        self.register_event_handler('answer_provided', self._handle_answer_provided)
        self.register_event_handler('session_control', self._handle_session_control)
        
        logger.info("Tutoring Session Handler initialized")
    
    async def _handle_tutor_join(self, message: WebSocketMessage):
        """Handle tutor joining session"""
        logger.info(f"Tutor {message.sender_id} joined session {message.session_id}")
        
        # Broadcast tutor join to all participants
        await self.broadcast_to_session(message.session_id, {
            'type': 'participant_update',
            'action': 'tutor_joined',
            'tutor_id': message.sender_id,
            'tutor_info': message.payload.get('tutor_info', {})
        })
    
    async def _handle_student_join(self, message: WebSocketMessage):
        """Handle student joining session"""
        logger.info(f"Student {message.sender_id} joined session {message.session_id}")
        
        # Broadcast student join to all participants
        await self.broadcast_to_session(message.session_id, {
            'type': 'participant_update',
            'action': 'student_joined',
            'student_id': message.sender_id,
            'student_info': message.payload.get('student_info', {})
        })
    
    async def _handle_question_asked(self, message: WebSocketMessage):
        """Handle student question"""
        question = message.payload.get('question', '')
        logger.info(f"Question from {message.sender_id} in session {message.session_id}: {question[:50]}...")
        
        # Forward to orchestrator for instant feedback
        if self.orchestrator:
            event = create_streaming_event(
                StreamingEventType.USER_ACTION,
                message.sender_id,
                message.session_id,
                {
                    'action_type': 'question_asked',
                    'question': question,
                    'context': message.payload.get('context', {})
                },
                priority=8  # Higher priority for questions
            )
            await self.orchestrator.add_streaming_event(event.__dict__)
        
        # Broadcast question to session participants
        await self.broadcast_to_session(message.session_id, {
            'type': 'question',
            'student_id': message.sender_id,
            'question': question,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_answer_provided(self, message: WebSocketMessage):
        """Handle tutor answer"""
        answer = message.payload.get('answer', '')
        question_id = message.payload.get('question_id', '')
        logger.info(f"Answer from {message.sender_id} in session {message.session_id}: {answer[:50]}...")
        
        # Broadcast answer to session participants
        await self.broadcast_to_session(message.session_id, {
            'type': 'answer',
            'tutor_id': message.sender_id,
            'question_id': question_id,
            'answer': answer,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_session_control(self, message: WebSocketMessage):
        """Handle session control commands"""
        command = message.payload.get('command', '')
        logger.info(f"Session control from {message.sender_id} in session {message.session_id}: {command}")
        
        # Process session control commands
        if command == 'pause':
            # Pause session
            await self.broadcast_to_session(message.session_id, {
                'type': 'session_state',
                'state': 'paused',
                'initiated_by': message.sender_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        elif command == 'resume':
            # Resume session
            await self.broadcast_to_session(message.session_id, {
                'type': 'session_state',
                'state': 'active',
                'initiated_by': message.sender_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        elif command == 'end':
            # End session
            await self.broadcast_to_session(message.session_id, {
                'type': 'session_state',
                'state': 'ended',
                'initiated_by': message.sender_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Notify orchestrator to end session
            if self.orchestrator:
                await self.orchestrator.end_streaming_session(message.session_id)


class FeedbackHandler(StreamingWebSocketHandler):
    """
    💬 FEEDBACK HANDLER
    
    Specialized WebSocket handler for instant feedback delivery.
    Manages real-time feedback generation, delivery, and user responses
    with sub-100ms latency targets.
    """
    
    def __init__(self, orchestrator_ref=None):
        super().__init__(orchestrator_ref)
        
        # Register feedback-specific handlers
        self.register_event_handler('feedback_request', self._handle_feedback_request)
        self.register_event_handler('feedback_response', self._handle_feedback_response)
        self.register_event_handler('feedback_rating', self._handle_feedback_rating)
        
        # Feedback metrics
        self.feedback_metrics = {
            'requests': 0,
            'delivered': 0,
            'avg_latency_ms': 0,
            'ratings': []
        }
        
        logger.info("Feedback Handler initialized")
    
    async def _handle_feedback_request(self, message: WebSocketMessage):
        """Handle explicit feedback request"""
        start_time = time.time()
        self.feedback_metrics['requests'] += 1
        
        logger.info(f"Feedback request from {message.sender_id} in session {message.session_id}")
        
        # Forward to orchestrator for instant feedback
        if self.orchestrator:
            event = create_streaming_event(
                StreamingEventType.USER_ACTION,
                message.sender_id,
                message.session_id,
                {
                    'action_type': 'feedback_request',
                    'context': message.payload.get('context', {}),
                    'feedback_type': message.payload.get('feedback_type', 'suggestion')
                },
                priority=9  # High priority for explicit feedback requests
            )
            await self.orchestrator.add_streaming_event(event.__dict__)
        
        # Track latency
        processing_time = (time.time() - start_time) * 1000  # ms
        self._update_latency_metrics(processing_time)
    
    async def _handle_feedback_response(self, message: WebSocketMessage):
        """Handle user response to feedback"""
        logger.info(f"Feedback response from {message.sender_id} in session {message.session_id}")
        
        # Forward to orchestrator for analysis
        if self.orchestrator:
            event = create_streaming_event(
                StreamingEventType.USER_ACTION,
                message.sender_id,
                message.session_id,
                {
                    'action_type': 'feedback_response',
                    'feedback_id': message.payload.get('feedback_id', ''),
                    'response': message.payload.get('response', {}),
                    'response_time_ms': message.payload.get('response_time_ms', 0)
                }
            )
            await self.orchestrator.add_streaming_event(event.__dict__)
    
    async def _handle_feedback_rating(self, message: WebSocketMessage):
        """Handle user rating of feedback"""
        rating = message.payload.get('rating', 0)
        feedback_id = message.payload.get('feedback_id', '')
        
        logger.info(f"Feedback rating from {message.sender_id}: {rating} for feedback {feedback_id}")
        
        # Track rating
        self.feedback_metrics['ratings'].append(rating)
        
        # Forward to orchestrator for analysis
        if self.orchestrator:
            event = create_streaming_event(
                StreamingEventType.USER_ACTION,
                message.sender_id,
                message.session_id,
                {
                    'action_type': 'feedback_rating',
                    'feedback_id': feedback_id,
                    'rating': rating,
                    'comments': message.payload.get('comments', '')
                }
            )
            await self.orchestrator.add_streaming_event(event.__dict__)
    
    async def deliver_feedback(self, user_id: str, session_id: str, feedback_data: Dict[str, Any]) -> bool:
        """Deliver feedback to user"""
        connection_id = self._find_connection_id(user_id, session_id)
        if not connection_id:
            logger.warning(f"Cannot deliver feedback: no connection for user {user_id} in session {session_id}")
            return False
        
        # Send feedback message
        result = await self.send_message(connection_id, {
            'type': 'feedback',
            'feedback_id': feedback_data.get('feedback_id', ''),
            'feedback_type': feedback_data.get('feedback_type', ''),
            'content': feedback_data.get('content', ''),
            'suggested_actions': feedback_data.get('suggested_actions', []),
            'requires_response': feedback_data.get('requires_response', False),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if result:
            self.feedback_metrics['delivered'] += 1
        
        return result
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update feedback latency metrics"""
        current_avg = self.feedback_metrics['avg_latency_ms']
        current_count = self.feedback_metrics['requests']
        
        # Calculate new running average
        if current_count > 1:
            self.feedback_metrics['avg_latency_ms'] = (current_avg * (current_count - 1) + latency_ms) / current_count
        else:
            self.feedback_metrics['avg_latency_ms'] = latency_ms


class CollaborationHandler(StreamingWebSocketHandler):
    """
    👥 COLLABORATION HANDLER
    
    Specialized WebSocket handler for multi-participant collaboration.
    Manages real-time collaboration events, peer interactions, and
    group dynamics with low-latency synchronization.
    """
    
    def __init__(self, orchestrator_ref=None):
        super().__init__(orchestrator_ref)
        
        # Register collaboration-specific handlers
        self.register_event_handler('collaboration_action', self._handle_collaboration_action)
        self.register_event_handler('peer_interaction', self._handle_peer_interaction)
        self.register_event_handler('group_state_change', self._handle_group_state_change)
        
        # Track collaboration groups
        self.collaboration_groups = {}
        
        logger.info("Collaboration Handler initialized")
    
    async def _handle_collaboration_action(self, message: WebSocketMessage):
        """Handle collaboration action"""
        action_type = message.payload.get('action_type', '')
        logger.info(f"Collaboration action from {message.sender_id} in session {message.session_id}: {action_type}")
        
        # Forward to orchestrator
        if self.orchestrator:
            event = create_streaming_event(
                StreamingEventType.COLLABORATION_EVENT,
                message.sender_id,
                message.session_id,
                message.payload
            )
            await self.orchestrator.add_streaming_event(event.__dict__)
        
        # Broadcast to collaboration group
        group_id = message.payload.get('group_id')
        if group_id and group_id in self.collaboration_groups:
            for user_id in self.collaboration_groups[group_id]:
                if user_id != message.sender_id:  # Don't send back to sender
                    connection_id = self._find_connection_id(user_id, message.session_id)
                    if connection_id:
                        await self.send_message(connection_id, {
                            'type': 'collaboration_update',
                            'action_type': action_type,
                            'sender_id': message.sender_id,
                            'group_id': group_id,
                            'content': message.payload.get('content', {}),
                            'timestamp': datetime.utcnow().isoformat()
                        })
    
    async def _handle_peer_interaction(self, message: WebSocketMessage):
        """Handle peer-to-peer interaction"""
        target_user = message.payload.get('target_user_id')
        interaction_type = message.payload.get('interaction_type', '')
        
        logger.info(f"Peer interaction from {message.sender_id} to {target_user}: {interaction_type}")
        
        # Forward to target user
        if target_user:
            connection_id = self._find_connection_id(target_user, message.session_id)
            if connection_id:
                await self.send_message(connection_id, {
                    'type': 'peer_interaction',
                    'sender_id': message.sender_id,
                    'interaction_type': interaction_type,
                    'content': message.payload.get('content', {}),
                    'timestamp': datetime.utcnow().isoformat()
                })
    
    async def _handle_group_state_change(self, message: WebSocketMessage):
        """Handle collaboration group state change"""
        group_id = message.payload.get('group_id')
        action = message.payload.get('action', '')
        
        logger.info(f"Group state change for group {group_id}: {action}")
        
        if action == 'create_group':
            # Create new collaboration group
            members = message.payload.get('members', [])
            self.collaboration_groups[group_id] = set(members)
            
            # Notify all members
            for user_id in members:
                connection_id = self._find_connection_id(user_id, message.session_id)
                if connection_id:
                    await self.send_message(connection_id, {
                        'type': 'group_update',
                        'action': 'group_created',
                        'group_id': group_id,
                        'members': members,
                        'created_by': message.sender_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        elif action == 'add_member' and group_id in self.collaboration_groups:
            # Add member to group
            new_member = message.payload.get('user_id')
            if new_member:
                self.collaboration_groups[group_id].add(new_member)
                
                # Notify all members
                for user_id in self.collaboration_groups[group_id]:
                    connection_id = self._find_connection_id(user_id, message.session_id)
                    if connection_id:
                        await self.send_message(connection_id, {
                            'type': 'group_update',
                            'action': 'member_added',
                            'group_id': group_id,
                            'user_id': new_member,
                            'added_by': message.sender_id,
                            'timestamp': datetime.utcnow().isoformat()
                        })
        
        elif action == 'remove_member' and group_id in self.collaboration_groups:
            # Remove member from group
            member = message.payload.get('user_id')
            if member and member in self.collaboration_groups[group_id]:
                self.collaboration_groups[group_id].remove(member)
                
                # Notify all members
                for user_id in self.collaboration_groups[group_id]:
                    connection_id = self._find_connection_id(user_id, message.session_id)
                    if connection_id:
                        await self.send_message(connection_id, {
                            'type': 'group_update',
                            'action': 'member_removed',
                            'group_id': group_id,
                            'user_id': member,
                            'removed_by': message.sender_id,
                            'timestamp': datetime.utcnow().isoformat()
                        })
        
        elif action == 'dissolve_group' and group_id in self.collaboration_groups:
            # Dissolve collaboration group
            members = list(self.collaboration_groups[group_id])
            
            # Notify all members
            for user_id in members:
                connection_id = self._find_connection_id(user_id, message.session_id)
                if connection_id:
                    await self.send_message(connection_id, {
                        'type': 'group_update',
                        'action': 'group_dissolved',
                        'group_id': group_id,
                        'dissolved_by': message.sender_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            # Remove group
            del self.collaboration_groups[group_id]
