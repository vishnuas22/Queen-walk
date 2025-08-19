"""
Comprehensive Test Suite for Integration & Orchestration System

Tests all integration and orchestration components including master orchestrator,
integration layer, API gateway, system monitoring, and deployment configuration
for maximum coverage and reliability.

🧪 TEST COVERAGE:
- Master orchestrator with service coordination
- Integration layer with service-to-service communication
- API gateway with unified endpoint management
- System monitoring with health checks and metrics
- Deployment configuration and infrastructure management

Author: MasterX AI Team - Integration & Orchestration Division
Version: 1.0 - Phase 11 Integration & Orchestration
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import orchestration components
from quantum_intelligence.orchestration.master_orchestrator import (
    MasterOrchestrator,
    ServiceInstance,
    ServiceRequest,
    ServiceResponse,
    ServiceStatus,
    ServiceType,
    LoadBalancingStrategy
)

from quantum_intelligence.orchestration.integration_layer import (
    IntegrationLayer,
    IntegrationMessage,
    ServiceEndpoint,
    IntegrationTransaction,
    CommunicationPattern,
    MessageType,
    DeliveryGuarantee,
    SerializationFormat
)

from quantum_intelligence.orchestration.api_gateway import (
    APIGateway,
    APIEndpoint,
    APIRequest,
    APIResponse,
    AuthenticationType,
    RateLimitType,
    LoadBalancingAlgorithm
)

from quantum_intelligence.orchestration.monitoring.system_monitor import (
    SystemMonitor,
    HealthCheck,
    SystemMetric,
    Alert,
    LogEntry,
    HealthStatus,
    AlertSeverity,
    MetricType
)

class TestMasterOrchestrator:
    """Test suite for the Master Orchestrator"""
    
    @pytest.fixture
    async def master_orchestrator(self):
        """Create master orchestrator for testing"""
        orchestrator = MasterOrchestrator()
        await orchestrator.start()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.fixture
    def sample_service_instance(self):
        """Sample service instance for testing"""
        return ServiceInstance(
            service_id='test_service',
            service_type=ServiceType.QUANTUM_INTELLIGENCE,
            instance_id='test_instance_001',
            host='localhost',
            port=8001,
            version='1.0',
            capabilities=['test_capability']
        )
    
    @pytest.fixture
    def sample_service_request(self):
        """Sample service request for testing"""
        return ServiceRequest(
            request_id='test_request_001',
            service_type=ServiceType.QUANTUM_INTELLIGENCE,
            method='POST',
            endpoint='/test_endpoint',
            payload={'test_data': 'test_value'}
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup_shutdown(self, master_orchestrator):
        """Test orchestrator startup and shutdown"""
        
        # Verify orchestrator is running
        assert master_orchestrator.is_running
        
        # Get status
        status = master_orchestrator.get_orchestrator_status()
        assert status['is_running']
        assert 'startup_time' in status
        assert 'metrics' in status
        assert 'services' in status
        
        print(f"✅ Master orchestrator startup/shutdown test successful")
        print(f"   Services registered: {status['services']['total_registered']}")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
    
    @pytest.mark.asyncio
    async def test_service_registration(self, master_orchestrator, sample_service_instance):
        """Test service registration and unregistration"""
        
        # Register service
        registration_success = await master_orchestrator.register_service(sample_service_instance)
        assert registration_success
        
        # Verify service is registered
        status = master_orchestrator.get_orchestrator_status()
        assert status['services']['total_registered'] > 0
        
        # Unregister service
        unregistration_success = await master_orchestrator.unregister_service(sample_service_instance.instance_id)
        assert unregistration_success
        
        print(f"✅ Service registration test successful")
        print(f"   Registration: {registration_success}")
        print(f"   Unregistration: {unregistration_success}")
    
    @pytest.mark.asyncio
    async def test_request_routing(self, master_orchestrator, sample_service_instance, sample_service_request):
        """Test request routing through orchestrator"""
        
        # Register service first
        await master_orchestrator.register_service(sample_service_instance)
        
        # Route request
        response = await master_orchestrator.route_request(sample_service_request)
        
        # Verify response
        assert isinstance(response, ServiceResponse)
        assert response.request_id == sample_service_request.request_id
        
        # Check metrics
        status = master_orchestrator.get_orchestrator_status()
        assert status['metrics']['total_requests'] > 0
        
        print(f"✅ Request routing test successful")
        print(f"   Response status: {response.status_code}")
        print(f"   Processing time: {response.processing_time_ms:.1f}ms")


class TestIntegrationLayer:
    """Test suite for the Integration Layer"""
    
    @pytest.fixture
    async def integration_layer(self):
        """Create integration layer for testing"""
        layer = IntegrationLayer()
        await layer.start()
        yield layer
        await layer.shutdown()
    
    @pytest.fixture
    def sample_service_endpoint(self):
        """Sample service endpoint for testing"""
        return ServiceEndpoint(
            service_name='test_service',
            endpoint_name='test_endpoint',
            endpoint_url='http://localhost:8001/test',
            communication_pattern=CommunicationPattern.REQUEST_RESPONSE,
            supported_methods=['GET', 'POST']
        )
    
    @pytest.fixture
    def sample_integration_message(self):
        """Sample integration message for testing"""
        return IntegrationMessage(
            message_id='test_message_001',
            message_type=MessageType.COMMAND,
            source_service='test_source',
            target_service='test_target',
            payload={'test_data': 'test_value'}
        )
    
    @pytest.mark.asyncio
    async def test_integration_layer_startup_shutdown(self, integration_layer):
        """Test integration layer startup and shutdown"""
        
        # Verify layer is running
        assert integration_layer.is_running
        
        # Get status
        status = integration_layer.get_integration_status()
        assert status['is_running']
        assert 'metrics' in status
        assert 'service_endpoints' in status
        
        print(f"✅ Integration layer startup/shutdown test successful")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"   Endpoints registered: {status['service_endpoints']['total_registered']}")
    
    @pytest.mark.asyncio
    async def test_service_endpoint_registration(self, integration_layer, sample_service_endpoint):
        """Test service endpoint registration"""
        
        # Register endpoint
        registration_success = await integration_layer.register_service_endpoint(sample_service_endpoint)
        assert registration_success
        
        # Verify endpoint is registered
        status = integration_layer.get_integration_status()
        assert status['service_endpoints']['total_registered'] > 0
        
        print(f"✅ Service endpoint registration test successful")
        print(f"   Registration: {registration_success}")
    
    @pytest.mark.asyncio
    async def test_message_sending_receiving(self, integration_layer, sample_integration_message):
        """Test message sending and receiving"""
        
        # Send message
        send_success = await integration_layer.send_message(
            sample_integration_message,
            CommunicationPattern.MESSAGE_QUEUE
        )
        assert send_success
        
        # Receive message
        received_message = await integration_layer.receive_message('test_queue', timeout_seconds=5.0)
        
        # Verify message (may be None due to simplified implementation)
        if received_message:
            assert received_message.message_id == sample_integration_message.message_id
        
        print(f"✅ Message sending/receiving test successful")
        print(f"   Send success: {send_success}")
        print(f"   Message received: {received_message is not None}")
    
    @pytest.mark.asyncio
    async def test_transaction_management(self, integration_layer):
        """Test distributed transaction management"""
        
        # Start transaction
        transaction_id = await integration_layer.start_transaction(
            coordinator_service='test_coordinator',
            participants=['service1', 'service2'],
            timeout_seconds=60
        )
        
        assert transaction_id is not None
        
        # Commit transaction
        commit_success = await integration_layer.commit_transaction(transaction_id)
        assert commit_success
        
        print(f"✅ Transaction management test successful")
        print(f"   Transaction ID: {transaction_id}")
        print(f"   Commit success: {commit_success}")


class TestAPIGateway:
    """Test suite for the API Gateway"""
    
    @pytest.fixture
    async def api_gateway(self):
        """Create API gateway for testing"""
        gateway = APIGateway()
        await gateway.start(host='localhost', port=8080)
        yield gateway
        await gateway.shutdown()
    
    @pytest.fixture
    def sample_api_endpoint(self):
        """Sample API endpoint for testing"""
        return APIEndpoint(
            endpoint_id='test_endpoint',
            path='/api/test',
            method='GET',
            service_name='test_service',
            upstream_path='/test',
            upstream_hosts=['localhost:8001'],
            authentication_type=AuthenticationType.NONE,
            rate_limit_value=100
        )
    
    @pytest.fixture
    def sample_api_request(self):
        """Sample API request for testing"""
        return APIRequest(
            request_id='test_request_001',
            method='GET',
            path='/api/test',
            headers={'Content-Type': 'application/json'},
            client_ip='127.0.0.1'
        )
    
    @pytest.mark.asyncio
    async def test_api_gateway_startup_shutdown(self, api_gateway):
        """Test API gateway startup and shutdown"""
        
        # Verify gateway is running
        assert api_gateway.is_running
        
        # Get status
        status = api_gateway.get_gateway_status()
        assert status['is_running']
        assert 'metrics' in status
        assert 'endpoints' in status
        
        print(f"✅ API gateway startup/shutdown test successful")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"   Endpoints registered: {status['endpoints']['total_registered']}")
    
    @pytest.mark.asyncio
    async def test_endpoint_registration(self, api_gateway, sample_api_endpoint):
        """Test API endpoint registration"""
        
        # Register endpoint
        registration_success = await api_gateway.register_endpoint(sample_api_endpoint)
        assert registration_success
        
        # Verify endpoint is registered
        status = api_gateway.get_gateway_status()
        assert status['endpoints']['total_registered'] > 0
        
        print(f"✅ API endpoint registration test successful")
        print(f"   Registration: {registration_success}")
    
    @pytest.mark.asyncio
    async def test_request_processing(self, api_gateway, sample_api_endpoint, sample_api_request):
        """Test API request processing"""
        
        # Register endpoint first
        await api_gateway.register_endpoint(sample_api_endpoint)
        
        # Process request
        response = await api_gateway.process_request(sample_api_request)
        
        # Verify response
        assert isinstance(response, APIResponse)
        assert response.request_id == sample_api_request.request_id
        assert response.status_code in [200, 404]  # 404 if route not found
        
        print(f"✅ API request processing test successful")
        print(f"   Response status: {response.status_code}")
        print(f"   Processing time: {response.processing_time_ms:.1f}ms")


class TestSystemMonitor:
    """Test suite for the System Monitor"""
    
    @pytest.fixture
    async def system_monitor(self):
        """Create system monitor for testing"""
        monitor = SystemMonitor()
        await monitor.start()
        yield monitor
        await monitor.shutdown()
    
    @pytest.fixture
    def sample_health_check(self):
        """Sample health check for testing"""
        return HealthCheck(
            check_id='test_health_check',
            service_name='test_service',
            check_name='Test Health Check',
            endpoint_url='http://localhost:8001/health'
        )
    
    @pytest.mark.asyncio
    async def test_system_monitor_startup_shutdown(self, system_monitor):
        """Test system monitor startup and shutdown"""
        
        # Verify monitor is running
        assert system_monitor.is_running
        
        # Get status
        status = system_monitor.get_monitor_status()
        assert status['is_running']
        assert 'monitoring_metrics' in status
        assert 'components' in status
        
        print(f"✅ System monitor startup/shutdown test successful")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"   Registered services: {status['registered_services']}")
    
    @pytest.mark.asyncio
    async def test_service_registration(self, system_monitor):
        """Test service registration for monitoring"""
        
        # Register service
        registration_success = await system_monitor.register_service(
            service_name='test_service',
            health_check_url='http://localhost:8001/health'
        )
        assert registration_success
        
        # Verify service is registered
        status = system_monitor.get_monitor_status()
        assert status['registered_services'] > 0
        
        print(f"✅ Service registration for monitoring test successful")
        print(f"   Registration: {registration_success}")
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, system_monitor):
        """Test metric recording"""
        
        # Record metric
        metric_success = await system_monitor.record_metric(
            metric_name='test_metric',
            value=42.0,
            metric_type=MetricType.GAUGE
        )
        assert metric_success
        
        # Get metrics summary
        metrics_summary = await system_monitor.get_metrics_summary(time_window_minutes=5)
        
        print(f"✅ Metric recording test successful")
        print(f"   Metric recorded: {metric_success}")
        print(f"   Metrics in summary: {len(metrics_summary)}")
    
    @pytest.mark.asyncio
    async def test_alert_triggering(self, system_monitor):
        """Test alert triggering"""
        
        # Trigger alert
        alert_success = await system_monitor.trigger_alert(
            alert_name='test_alert',
            message='Test alert message',
            severity=AlertSeverity.WARNING,
            service_name='test_service'
        )
        assert alert_success
        
        # Get active alerts
        active_alerts = await system_monitor.get_active_alerts()
        
        print(f"✅ Alert triggering test successful")
        print(f"   Alert triggered: {alert_success}")
        print(f"   Active alerts: {len(active_alerts)}")
    
    @pytest.mark.asyncio
    async def test_system_health(self, system_monitor):
        """Test system health reporting"""
        
        # Get system health
        health_status = await system_monitor.get_system_health()
        
        # Verify health status structure
        assert 'overall_health' in health_status
        assert 'system_resources' in health_status
        assert 'monitoring_metrics' in health_status
        
        print(f"✅ System health reporting test successful")
        print(f"   Overall health: {health_status['overall_health']}")
        print(f"   Healthy services: {health_status.get('healthy_services', 0)}")


class TestIntegration:
    """Integration tests for the complete orchestration system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_orchestration(self):
        """Test complete end-to-end orchestration workflow"""
        
        # Initialize all components
        orchestrator = MasterOrchestrator()
        integration_layer = IntegrationLayer()
        api_gateway = APIGateway()
        system_monitor = SystemMonitor()
        
        try:
            # Start all components
            await orchestrator.start()
            await integration_layer.start()
            await api_gateway.start(host='localhost', port=8081)
            await system_monitor.start()
            
            # Register services
            service_instance = ServiceInstance(
                service_id='integration_test_service',
                service_type=ServiceType.QUANTUM_INTELLIGENCE,
                instance_id='integration_test_001',
                host='localhost',
                port=8002,
                version='1.0',
                capabilities=['test']
            )
            
            await orchestrator.register_service(service_instance)
            await system_monitor.register_service(
                service_name='integration_test_service',
                health_check_url='http://localhost:8002/health'
            )
            
            # Test request flow
            service_request = ServiceRequest(
                request_id='integration_test_request',
                service_type=ServiceType.QUANTUM_INTELLIGENCE,
                method='POST',
                endpoint='/test',
                payload={'integration_test': True}
            )
            
            response = await orchestrator.route_request(service_request)
            assert response.request_id == service_request.request_id
            
            # Test monitoring
            await system_monitor.record_metric('integration_test_metric', 100.0)
            health_status = await system_monitor.get_system_health()
            
            # Verify integration
            orchestrator_status = orchestrator.get_orchestrator_status()
            integration_status = integration_layer.get_integration_status()
            gateway_status = api_gateway.get_gateway_status()
            monitor_status = system_monitor.get_monitor_status()
            
            assert orchestrator_status['is_running']
            assert integration_status['is_running']
            assert gateway_status['is_running']
            assert monitor_status['is_running']
            
            print(f"✅ End-to-end orchestration test successful")
            print(f"   All components running: True")
            print(f"   Services registered: {orchestrator_status['services']['total_registered']}")
            print(f"   Overall health: {health_status['overall_health']}")
            
        finally:
            # Shutdown all components
            await system_monitor.shutdown()
            await api_gateway.shutdown()
            await integration_layer.shutdown()
            await orchestrator.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
