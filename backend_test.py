#!/usr/bin/env python3
"""
Backend Testing Suite for MasterX Platform
Tests the actual implemented FastAPI backend functionality
"""

import requests
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
import uuid

# Load environment variables
sys.path.append('/app/frontend')
from dotenv import load_dotenv

# Load frontend .env to get the backend URL
load_dotenv('/app/frontend/.env')
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')

if not BACKEND_URL:
    print("❌ CRITICAL: REACT_APP_BACKEND_URL not found in frontend/.env")
    sys.exit(1)

print(f"🔗 Testing backend at: {BACKEND_URL}")

class BackendTester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.api_url = f"{self.base_url}/api"
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def test_server_health(self):
        """Test basic server connectivity and health"""
        print("\n🏥 Testing Server Health...")
        
        try:
            # Test root endpoint
            response = requests.get(f"{self.api_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('message') == 'Hello World':
                    self.log_test("Root endpoint (/api/)", True, f"Response: {data}")
                else:
                    self.log_test("Root endpoint (/api/)", False, f"Unexpected response: {data}")
            else:
                self.log_test("Root endpoint (/api/)", False, f"Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Root endpoint (/api/)", False, f"Connection error: {str(e)}")
    
    def test_cors_headers(self):
        """Test CORS configuration"""
        print("\n🌐 Testing CORS Configuration...")
        
        try:
            response = requests.options(f"{self.api_url}/", timeout=10)
            headers = response.headers
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            missing_headers = [h for h in cors_headers if h not in headers]
            
            if not missing_headers:
                self.log_test("CORS Headers", True, "All required CORS headers present")
            else:
                self.log_test("CORS Headers", False, f"Missing headers: {missing_headers}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("CORS Headers", False, f"Connection error: {str(e)}")
    
    def test_status_endpoints(self):
        """Test status check CRUD operations"""
        print("\n📊 Testing Status Check Endpoints...")
        
        # Test POST /api/status
        try:
            test_client_name = f"test_client_{uuid.uuid4().hex[:8]}"
            payload = {"client_name": test_client_name}
            
            response = requests.post(
                f"{self.api_url}/status",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if (data.get('client_name') == test_client_name and 
                    'id' in data and 'timestamp' in data):
                    self.log_test("POST /api/status", True, f"Created status check: {data['id']}")
                    created_id = data['id']
                else:
                    self.log_test("POST /api/status", False, f"Invalid response structure: {data}")
                    created_id = None
            else:
                self.log_test("POST /api/status", False, f"Status: {response.status_code}, Body: {response.text}")
                created_id = None
                
        except requests.exceptions.RequestException as e:
            self.log_test("POST /api/status", False, f"Connection error: {str(e)}")
            created_id = None
        
        # Test GET /api/status
        try:
            response = requests.get(f"{self.api_url}/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_test("GET /api/status", True, f"Retrieved {len(data)} status checks")
                    
                    # Verify our created item is in the list if we created one
                    if created_id:
                        found_item = any(item.get('id') == created_id for item in data)
                        if found_item:
                            self.log_test("Status Check Persistence", True, "Created item found in GET response")
                        else:
                            self.log_test("Status Check Persistence", False, "Created item not found in GET response")
                else:
                    self.log_test("GET /api/status", False, f"Expected list, got: {type(data)}")
            else:
                self.log_test("GET /api/status", False, f"Status: {response.status_code}, Body: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("GET /api/status", False, f"Connection error: {str(e)}")
    
    def test_database_connectivity(self):
        """Test database operations indirectly through API"""
        print("\n🗄️ Testing Database Connectivity...")
        
        # Create multiple status checks to test database operations
        test_items = []
        for i in range(3):
            try:
                payload = {"client_name": f"db_test_client_{i}_{uuid.uuid4().hex[:6]}"}
                response = requests.post(
                    f"{self.api_url}/status",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    test_items.append(response.json())
                    
            except requests.exceptions.RequestException:
                pass
        
        if len(test_items) >= 2:
            self.log_test("Database Write Operations", True, f"Successfully created {len(test_items)} items")
            
            # Test retrieval
            try:
                response = requests.get(f"{self.api_url}/status", timeout=10)
                if response.status_code == 200:
                    all_items = response.json()
                    found_count = sum(1 for item in all_items 
                                    if any(test_item['id'] == item.get('id') for test_item in test_items))
                    
                    if found_count == len(test_items):
                        self.log_test("Database Read Operations", True, f"All {len(test_items)} test items retrieved")
                    else:
                        self.log_test("Database Read Operations", False, f"Only {found_count}/{len(test_items)} items found")
                else:
                    self.log_test("Database Read Operations", False, f"GET failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test("Database Read Operations", False, f"Connection error: {str(e)}")
        else:
            self.log_test("Database Write Operations", False, f"Could only create {len(test_items)} items")
    
    def test_error_handling(self):
        """Test API error handling"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid JSON
        try:
            response = requests.post(
                f"{self.api_url}/status",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [400, 422]:  # Bad Request or Unprocessable Entity
                self.log_test("Invalid JSON Handling", True, f"Properly rejected with status {response.status_code}")
            else:
                self.log_test("Invalid JSON Handling", False, f"Unexpected status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Invalid JSON Handling", False, f"Connection error: {str(e)}")
        
        # Test missing required fields
        try:
            response = requests.post(
                f"{self.api_url}/status",
                json={},  # Missing client_name
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [400, 422]:
                self.log_test("Missing Field Validation", True, f"Properly rejected with status {response.status_code}")
            else:
                self.log_test("Missing Field Validation", False, f"Unexpected status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Missing Field Validation", False, f"Connection error: {str(e)}")
        
        # Test non-existent endpoint
        try:
            response = requests.get(f"{self.api_url}/nonexistent", timeout=10)
            
            if response.status_code == 404:
                self.log_test("404 Error Handling", True, "Non-existent endpoint properly returns 404")
            else:
                self.log_test("404 Error Handling", False, f"Unexpected status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("404 Error Handling", False, f"Connection error: {str(e)}")
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Backend Test Suite")
        print("=" * 50)
        
        self.test_server_health()
        self.test_cors_headers()
        self.test_status_endpoints()
        self.test_database_connectivity()
        self.test_error_handling()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n🚨 FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test']}: {result['details']}")
        
        return passed_tests, failed_tests, self.test_results

if __name__ == "__main__":
    tester = BackendTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    sys.exit(0 if failed == 0 else 1)