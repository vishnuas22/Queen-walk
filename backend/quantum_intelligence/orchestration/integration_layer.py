"""
Integration Layer for MasterX Quantum Intelligence Platform

Advanced integration layer that provides service-to-service communication
patterns, event-driven architecture, message queuing, async processing,
and data synchronization across all quantum intelligence services.

🔗 INTEGRATION LAYER CAPABILITIES:
- Service-to-service communication patterns and protocols
- Event-driven architecture with real-time event processing
- Advanced message queuing and async processing
- Data synchronization and consistency across services
- Cross-service transaction management and coordination
- Intelligent routing and service mesh functionality

Author: MasterX AI Team - Integration & Orchestration Division
Version: 1.0 - Phase 11 Integration & Orchestration
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import weakref

# Try to import advanced libraries with fallbacks
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    # Compatibility patch for Python 3.11 and aioredis
    import builtins
    original_timeout_error = builtins.TimeoutError

    # Temporarily patch TimeoutError to avoid duplicate base class issue
    import asyncio
    if hasattr(asyncio, 'TimeoutError') and asyncio.TimeoutError is original_timeout_error:
        # Create a unique TimeoutError for aioredis
        class AioredisTimeoutError(Exception):
            pass
        builtins.TimeoutError = AioredisTimeoutError

    import aioredis

    # Restore original TimeoutError
    builtins.TimeoutError = original_timeout_error

    import aiokafka
    ADVANCED_MESSAGING = True
except ImportError as e:
    ADVANCED_MESSAGING = False
    logger.warning(f"Advanced messaging libraries not available - using basic implementations: {e}")
except Exception as e:
    ADVANCED_MESSAGING = False
    logger.warning(f"Error loading advanced messaging libraries - using basic implementations: {e}")

# ============================================================================
# INTEGRATION LAYER ENUMS & DATA STRUCTURES
# ============================================================================

class CommunicationPattern(Enum):
    """Communication pattern types"""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAMING = "event_streaming"
    RPC_CALL = "rpc_call"
    BROADCAST = "broadcast"

class MessageType(Enum):
    """Message type enumeration"""
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"

class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels"""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

class SerializationFormat(Enum):
    """Serialization format options"""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"

@dataclass
class IntegrationMessage:
    """
    📨 INTEGRATION MESSAGE
    
    Standardized message format for inter-service communication
    """
    message_id: str
    message_type: MessageType
    source_service: str
    target_service: Optional[str]
    
    # Message content
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Routing and delivery
    topic: Optional[str] = None
    routing_key: Optional[str] = None
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    
    # Quality of service
    priority: int = 5  # 1-10 scale
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Tracing and correlation
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None

@dataclass
class ServiceEndpoint:
    """
    🔌 SERVICE ENDPOINT
    
    Represents a service endpoint for communication
    """
    service_name: str
    endpoint_name: str
    endpoint_url: str
    
    # Communication settings
    communication_pattern: CommunicationPattern
    supported_methods: List[str]
    serialization_format: SerializationFormat = SerializationFormat.JSON
    
    # Quality of service
    timeout_seconds: float = 30.0
    rate_limit_per_second: Optional[int] = None
    circuit_breaker_enabled: bool = True
    
    # Authentication and security
    requires_authentication: bool = True
    allowed_roles: List[str] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class IntegrationTransaction:
    """
    🔄 INTEGRATION TRANSACTION
    
    Represents a distributed transaction across services
    """
    transaction_id: str
    coordinator_service: str
    
    # Transaction participants
    participants: List[str] = field(default_factory=list)
    participant_states: Dict[str, str] = field(default_factory=dict)
    
    # Transaction state
    status: str = "ACTIVE"  # ACTIVE, PREPARING, COMMITTED, ABORTED
    isolation_level: str = "READ_COMMITTED"
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    timeout_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Compensation and rollback
    compensation_actions: List[Dict[str, Any]] = field(default_factory=list)
    rollback_completed: bool = False


class IntegrationLayer:
    """
    🔗 INTEGRATION LAYER
    
    Advanced integration layer that provides comprehensive service-to-service
    communication, event-driven architecture, message queuing, and data
    synchronization across all quantum intelligence services.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration layer"""
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Communication components
        self.message_broker = MessageBroker(self.config.get('message_broker', {}))
        self.event_processor = EventProcessor(self.config.get('event_processor', {}))
        self.service_mesh = ServiceMesh(self.config.get('service_mesh', {}))
        self.transaction_manager = TransactionManager(self.config.get('transaction_manager', {}))
        
        # Service registry
        self.service_endpoints: Dict[str, ServiceEndpoint] = {}
        self.service_connections: Dict[str, Any] = {}
        
        # Message routing
        self.message_router = MessageRouter()
        self.serialization_manager = SerializationManager()
        
        # Integration state
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Performance tracking
        self.integration_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'events_processed': 0,
            'transactions_completed': 0,
            'average_latency_ms': 0.0,
            'error_rate': 0.0
        }
        
        logger.info("🔗 Integration Layer initialized")
    
    async def start(self) -> bool:
        """
        Start the integration layer
        
        Returns:
            bool: True if startup successful, False otherwise
        """
        try:
            logger.info("🚀 Starting Integration Layer...")
            
            # Start core components
            await self.message_broker.start()
            await self.event_processor.start()
            await self.service_mesh.start()
            await self.transaction_manager.start()
            
            # Initialize service discovery
            await self._initialize_service_discovery()
            
            # Start message processing
            await self._start_message_processing()
            
            self.is_running = True
            
            logger.info("✅ Integration Layer started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Integration Layer: {e}")
            await self.shutdown()
            return False
    
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the integration layer
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            logger.info("🛑 Shutting down Integration Layer...")
            
            self.is_running = False
            
            # Stop message processing
            await self._stop_message_processing()
            
            # Shutdown components
            await self.transaction_manager.shutdown()
            await self.service_mesh.shutdown()
            await self.event_processor.shutdown()
            await self.message_broker.shutdown()
            
            logger.info("✅ Integration Layer shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during Integration Layer shutdown: {e}")
            return False
    
    async def register_service_endpoint(self, endpoint: ServiceEndpoint) -> bool:
        """
        Register a service endpoint
        
        Args:
            endpoint: Service endpoint to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Validate endpoint
            if not await self._validate_service_endpoint(endpoint):
                return False
            
            # Register endpoint
            endpoint_key = f"{endpoint.service_name}.{endpoint.endpoint_name}"
            self.service_endpoints[endpoint_key] = endpoint
            
            # Register with service mesh
            await self.service_mesh.register_endpoint(endpoint)
            
            logger.info(f"✅ Service endpoint registered: {endpoint_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register service endpoint: {e}")
            return False
    
    async def send_message(
        self,
        message: IntegrationMessage,
        communication_pattern: CommunicationPattern = CommunicationPattern.MESSAGE_QUEUE
    ) -> bool:
        """
        Send a message through the integration layer
        
        Args:
            message: Message to send
            communication_pattern: Communication pattern to use
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            start_time = time.time()
            
            # Validate message
            if not await self._validate_message(message):
                return False
            
            # Route message
            routing_info = await self.message_router.route_message(message, communication_pattern)
            
            # Serialize message
            serialized_message = await self.serialization_manager.serialize(message)
            
            # Send through appropriate channel
            if communication_pattern == CommunicationPattern.MESSAGE_QUEUE:
                success = await self.message_broker.send_to_queue(
                    routing_info['queue'], serialized_message
                )
            elif communication_pattern == CommunicationPattern.PUBLISH_SUBSCRIBE:
                success = await self.message_broker.publish_to_topic(
                    routing_info['topic'], serialized_message
                )
            elif communication_pattern == CommunicationPattern.EVENT_STREAMING:
                success = await self.event_processor.send_event(serialized_message)
            else:
                success = await self._send_direct_message(message, routing_info)
            
            # Update metrics
            if success:
                self.integration_metrics['messages_sent'] += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency_metric(latency)
            else:
                self._update_error_rate()
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            self._update_error_rate()
            return False
    
    async def receive_message(
        self,
        source: str,
        timeout_seconds: float = 30.0
    ) -> Optional[IntegrationMessage]:
        """
        Receive a message from the integration layer
        
        Args:
            source: Source to receive from (queue, topic, etc.)
            timeout_seconds: Timeout for receiving
            
        Returns:
            IntegrationMessage: Received message or None if timeout
        """
        try:
            # Receive serialized message
            serialized_message = await self.message_broker.receive_from_queue(
                source, timeout_seconds
            )
            
            if not serialized_message:
                return None
            
            # Deserialize message
            message = await self.serialization_manager.deserialize(serialized_message)
            
            # Update metrics
            self.integration_metrics['messages_received'] += 1
            
            return message
            
        except Exception as e:
            logger.error(f"❌ Failed to receive message: {e}")
            return None
    
    async def start_transaction(
        self,
        coordinator_service: str,
        participants: List[str],
        timeout_seconds: int = 300
    ) -> str:
        """
        Start a distributed transaction
        
        Args:
            coordinator_service: Service coordinating the transaction
            participants: List of participating services
            timeout_seconds: Transaction timeout
            
        Returns:
            str: Transaction ID
        """
        try:
            transaction = IntegrationTransaction(
                transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
                coordinator_service=coordinator_service,
                participants=participants,
                timeout_at=datetime.now() + timedelta(seconds=timeout_seconds)
            )
            
            # Start transaction through transaction manager
            success = await self.transaction_manager.start_transaction(transaction)
            
            if success:
                self.integration_metrics['transactions_completed'] += 1
                return transaction.transaction_id
            else:
                raise Exception("Failed to start transaction")
                
        except Exception as e:
            logger.error(f"❌ Failed to start transaction: {e}")
            raise
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a distributed transaction
        
        Args:
            transaction_id: Transaction ID to commit
            
        Returns:
            bool: True if commit successful, False otherwise
        """
        try:
            return await self.transaction_manager.commit_transaction(transaction_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to commit transaction: {e}")
            return False
    
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a distributed transaction
        
        Args:
            transaction_id: Transaction ID to rollback
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            return await self.transaction_manager.rollback_transaction(transaction_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback transaction: {e}")
            return False

    # ========================================================================
    # HELPER METHODS FOR INTEGRATION LAYER
    # ========================================================================

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default integration layer configuration"""

        return {
            'message_broker': {
                'type': 'memory',  # memory, redis, kafka
                'max_queue_size': 10000,
                'message_ttl_seconds': 3600,
                'retry_attempts': 3
            },
            'event_processor': {
                'buffer_size': 1000,
                'batch_size': 100,
                'processing_interval_ms': 100
            },
            'service_mesh': {
                'discovery_interval_seconds': 30,
                'health_check_interval_seconds': 15,
                'load_balancing_strategy': 'round_robin'
            },
            'transaction_manager': {
                'default_timeout_seconds': 300,
                'max_participants': 10,
                'compensation_enabled': True
            },
            'serialization': {
                'default_format': 'json',
                'compression_enabled': True,
                'encryption_enabled': False
            }
        }

    async def _initialize_service_discovery(self):
        """Initialize service discovery mechanisms"""
        try:
            await self.service_mesh.initialize_discovery()
            logger.info("✅ Service discovery initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize service discovery: {e}")
            raise

    async def _start_message_processing(self):
        """Start message processing loops"""
        try:
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._event_processing_loop())
            logger.info("✅ Message processing started")
        except Exception as e:
            logger.error(f"❌ Failed to start message processing: {e}")
            raise

    async def _stop_message_processing(self):
        """Stop message processing loops"""
        # Processing loops will stop when is_running becomes False
        pass

    async def _validate_service_endpoint(self, endpoint: ServiceEndpoint) -> bool:
        """Validate a service endpoint"""
        if not all([endpoint.service_name, endpoint.endpoint_name, endpoint.endpoint_url]):
            logger.error("Invalid service endpoint: missing required fields")
            return False
        return True

    async def _validate_message(self, message: IntegrationMessage) -> bool:
        """Validate an integration message"""
        if not all([message.message_id, message.source_service]):
            logger.error("Invalid message: missing required fields")
            return False
        return True

    async def _send_direct_message(self, message: IntegrationMessage, routing_info: Dict[str, Any]) -> bool:
        """Send a direct message to a service"""
        # Placeholder for direct service communication
        return True

    def _update_latency_metric(self, latency_ms: float):
        """Update average latency metric"""
        current_avg = self.integration_metrics['average_latency_ms']
        total_messages = self.integration_metrics['messages_sent']

        if total_messages == 1:
            self.integration_metrics['average_latency_ms'] = latency_ms
        else:
            new_avg = ((current_avg * (total_messages - 1)) + latency_ms) / total_messages
            self.integration_metrics['average_latency_ms'] = new_avg

    def _update_error_rate(self):
        """Update error rate metric"""
        total_messages = self.integration_metrics['messages_sent'] + 1  # Include failed message
        errors = total_messages - self.integration_metrics['messages_sent']
        self.integration_metrics['error_rate'] = errors / total_messages

    async def _message_processing_loop(self):
        """Background loop for message processing"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)
                # Process pending messages
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)

    async def _event_processing_loop(self):
        """Background loop for event processing"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)
                # Process pending events
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration layer status"""

        return {
            'is_running': self.is_running,
            'startup_time': self.startup_time,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'metrics': self.integration_metrics,
            'service_endpoints': {
                'total_registered': len(self.service_endpoints),
                'endpoints': list(self.service_endpoints.keys())
            },
            'message_broker': {
                'status': 'active' if self.message_broker else 'inactive',
                'queue_count': len(getattr(self.message_broker, 'queues', {}))
            },
            'event_processor': {
                'status': 'active' if self.event_processor else 'inactive',
                'events_processed': self.integration_metrics['events_processed']
            },
            'transaction_manager': {
                'status': 'active' if self.transaction_manager else 'inactive',
                'active_transactions': len(getattr(self.transaction_manager, 'active_transactions', {}))
            }
        }


# ============================================================================
# HELPER CLASSES FOR INTEGRATION LAYER
# ============================================================================

class MessageBroker:
    """Message broker for async communication"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queues = defaultdict(lambda: asyncio.Queue())
        self.topics = defaultdict(list)  # topic -> list of subscribers
        self.is_running = False

    async def start(self):
        """Start the message broker"""
        self.is_running = True
        logger.info("✅ Message broker started")

    async def shutdown(self):
        """Shutdown the message broker"""
        self.is_running = False
        logger.info("✅ Message broker shutdown")

    async def send_to_queue(self, queue_name: str, message: bytes) -> bool:
        """Send message to a queue"""
        try:
            await self.queues[queue_name].put(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send to queue {queue_name}: {e}")
            return False

    async def receive_from_queue(self, queue_name: str, timeout: float) -> Optional[bytes]:
        """Receive message from a queue"""
        try:
            return await asyncio.wait_for(self.queues[queue_name].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive from queue {queue_name}: {e}")
            return None

    async def publish_to_topic(self, topic_name: str, message: bytes) -> bool:
        """Publish message to a topic"""
        try:
            subscribers = self.topics[topic_name]
            for subscriber_queue in subscribers:
                await subscriber_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to topic {topic_name}: {e}")
            return False

    async def subscribe_to_topic(self, topic_name: str) -> asyncio.Queue:
        """Subscribe to a topic"""
        subscriber_queue = asyncio.Queue()
        self.topics[topic_name].append(subscriber_queue)
        return subscriber_queue


class EventProcessor:
    """Event processor for event-driven architecture"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_buffer = deque()
        self.event_handlers = defaultdict(list)
        self.is_running = False

    async def start(self):
        """Start the event processor"""
        self.is_running = True
        asyncio.create_task(self._event_processing_loop())
        logger.info("✅ Event processor started")

    async def shutdown(self):
        """Shutdown the event processor"""
        self.is_running = False
        logger.info("✅ Event processor shutdown")

    async def send_event(self, event_data: bytes) -> bool:
        """Send an event for processing"""
        try:
            self.event_buffer.append({
                'data': event_data,
                'timestamp': datetime.now()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)

    async def _event_processing_loop(self):
        """Process events from the buffer"""
        while self.is_running:
            try:
                if self.event_buffer:
                    event = self.event_buffer.popleft()
                    await self._process_event(event)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(1)

    async def _process_event(self, event: Dict[str, Any]):
        """Process a single event"""
        # Placeholder for event processing logic
        pass


class ServiceMesh:
    """Service mesh for service discovery and communication"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service_registry = {}
        self.is_running = False

    async def start(self):
        """Start the service mesh"""
        self.is_running = True
        logger.info("✅ Service mesh started")

    async def shutdown(self):
        """Shutdown the service mesh"""
        self.is_running = False
        logger.info("✅ Service mesh shutdown")

    async def initialize_discovery(self):
        """Initialize service discovery"""
        # Placeholder for service discovery initialization
        pass

    async def register_endpoint(self, endpoint: ServiceEndpoint):
        """Register a service endpoint"""
        endpoint_key = f"{endpoint.service_name}.{endpoint.endpoint_name}"
        self.service_registry[endpoint_key] = endpoint


class TransactionManager:
    """Transaction manager for distributed transactions"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_transactions = {}
        self.is_running = False

    async def start(self):
        """Start the transaction manager"""
        self.is_running = True
        logger.info("✅ Transaction manager started")

    async def shutdown(self):
        """Shutdown the transaction manager"""
        self.is_running = False
        logger.info("✅ Transaction manager shutdown")

    async def start_transaction(self, transaction: IntegrationTransaction) -> bool:
        """Start a distributed transaction"""
        try:
            self.active_transactions[transaction.transaction_id] = transaction
            return True
        except Exception as e:
            logger.error(f"Failed to start transaction: {e}")
            return False

    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        try:
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
                transaction.status = "COMMITTED"
                transaction.completed_at = datetime.now()
                del self.active_transactions[transaction_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            return False

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction"""
        try:
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
                transaction.status = "ABORTED"
                transaction.completed_at = datetime.now()
                del self.active_transactions[transaction_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
            return False


class MessageRouter:
    """Message router for intelligent routing"""

    def __init__(self):
        self.routing_rules = {}

    async def route_message(
        self,
        message: IntegrationMessage,
        pattern: CommunicationPattern
    ) -> Dict[str, Any]:
        """Route a message based on pattern and rules"""

        if pattern == CommunicationPattern.MESSAGE_QUEUE:
            return {'queue': f"queue_{message.target_service or 'default'}"}
        elif pattern == CommunicationPattern.PUBLISH_SUBSCRIBE:
            return {'topic': message.topic or 'default_topic'}
        else:
            return {'endpoint': f"{message.target_service}/api"}


class SerializationManager:
    """Serialization manager for message serialization"""

    def __init__(self):
        self.serializers = {
            SerializationFormat.JSON: self._json_serialize,
            SerializationFormat.MSGPACK: self._msgpack_serialize
        }
        self.deserializers = {
            SerializationFormat.JSON: self._json_deserialize,
            SerializationFormat.MSGPACK: self._msgpack_deserialize
        }

    async def serialize(self, message: IntegrationMessage) -> bytes:
        """Serialize a message"""
        try:
            # Convert message to dict
            message_dict = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'source_service': message.source_service,
                'target_service': message.target_service,
                'payload': message.payload,
                'headers': message.headers,
                'created_at': message.created_at.isoformat()
            }

            # Serialize using JSON by default
            return json.dumps(message_dict).encode('utf-8')

        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            raise

    async def deserialize(self, data: bytes) -> IntegrationMessage:
        """Deserialize a message"""
        try:
            # Deserialize using JSON by default
            message_dict = json.loads(data.decode('utf-8'))

            # Convert back to IntegrationMessage
            return IntegrationMessage(
                message_id=message_dict['message_id'],
                message_type=MessageType(message_dict['message_type']),
                source_service=message_dict['source_service'],
                target_service=message_dict.get('target_service'),
                payload=message_dict['payload'],
                headers=message_dict.get('headers', {}),
                created_at=datetime.fromisoformat(message_dict['created_at'])
            )

        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise

    def _json_serialize(self, data: Any) -> bytes:
        """JSON serialization"""
        return json.dumps(data).encode('utf-8')

    def _json_deserialize(self, data: bytes) -> Any:
        """JSON deserialization"""
        return json.loads(data.decode('utf-8'))

    def _msgpack_serialize(self, data: Any) -> bytes:
        """MessagePack serialization (placeholder)"""
        # Would use msgpack library in production
        return json.dumps(data).encode('utf-8')

    def _msgpack_deserialize(self, data: bytes) -> Any:
        """MessagePack deserialization (placeholder)"""
        # Would use msgpack library in production
        return json.loads(data.decode('utf-8'))
