#!/usr/bin/env python3
"""
MasterX Load Testing Suite
Comprehensive performance testing for production readiness
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse

@dataclass
class TestResult:
    endpoint: str
    method: str
    response_time: float
    status_code: int
    success: bool
    error: str = None

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    async def make_request(self, session: aiohttp.ClientSession, method: str, endpoint: str, 
                          data: Dict[str, Any] = None, files: Dict[str, Any] = None) -> TestResult:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with session.get(url) as response:
                    await response.text()
                    return TestResult(
                        endpoint=endpoint,
                        method=method,
                        response_time=time.time() - start_time,
                        status_code=response.status,
                        success=response.status < 400
                    )
            
            elif method.upper() == "POST":
                if files:
                    # File upload test
                    form_data = aiohttp.FormData()
                    for key, value in files.items():
                        form_data.add_field(key, value)
                    
                    async with session.post(url, data=form_data) as response:
                        await response.text()
                        return TestResult(
                            endpoint=endpoint,
                            method=method,
                            response_time=time.time() - start_time,
                            status_code=response.status,
                            success=response.status < 400
                        )
                else:
                    # JSON POST test
                    async with session.post(url, json=data) as response:
                        await response.text()
                        return TestResult(
                            endpoint=endpoint,
                            method=method,
                            response_time=time.time() - start_time,
                            status_code=response.status,
                            success=response.status < 400
                        )
                        
        except Exception as e:
            return TestResult(
                endpoint=endpoint,
                method=method,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error=str(e)
            )

    async def test_health_endpoint(self, session: aiohttp.ClientSession, num_requests: int = 100):
        """Test health endpoint performance"""
        print(f"🏥 Testing health endpoint ({num_requests} requests)...")
        
        tasks = []
        for _ in range(num_requests):
            task = self.make_request(session, "GET", "/health")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        return results

    async def test_chat_endpoint(self, session: aiohttp.ClientSession, num_requests: int = 50):
        """Test chat endpoint performance"""
        print(f"💬 Testing chat endpoint ({num_requests} requests)...")
        
        test_messages = [
            "Hello, how are you?",
            "What are your capabilities?",
            "Can you help me with Python programming?",
            "Explain machine learning concepts",
            "What is quantum computing?"
        ]
        
        tasks = []
        for i in range(num_requests):
            message = test_messages[i % len(test_messages)]
            data = {
                "message": message,
                "session_id": f"load_test_session_{i % 10}"
            }
            task = self.make_request(session, "POST", "/api/chat/send", data=data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        return results

    async def test_file_upload(self, session: aiohttp.ClientSession, num_requests: int = 20):
        """Test file upload performance"""
        print(f"📁 Testing file upload ({num_requests} requests)...")
        
        # Create test file content
        test_content = "# Test Python File\nprint('Hello, World!')\n" * 100
        
        tasks = []
        for i in range(num_requests):
            files = {
                "file": ("test_file.py", test_content, "text/plain")
            }
            task = self.make_request(session, "POST", "/api/files/upload", files=files)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        return results

    async def test_websocket_connections(self, num_connections: int = 10):
        """Test WebSocket connection performance"""
        print(f"🔌 Testing WebSocket connections ({num_connections} connections)...")
        
        async def websocket_test():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(f"{self.base_url.replace('http', 'ws')}/ws/test_user") as ws:
                        # Send test message
                        await ws.send_str(json.dumps({
                            "type": "chat",
                            "message": "WebSocket test message",
                            "session_id": "ws_test_session"
                        }))
                        
                        # Wait for response
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                break
                        
                        return True
            except Exception as e:
                print(f"WebSocket error: {e}")
                return False
        
        tasks = [websocket_test() for _ in range(num_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        print(f"✅ WebSocket connections successful: {success_count}/{num_connections}")
        
        return results

    async def run_concurrent_load_test(self, concurrent_users: int = 50, duration_seconds: int = 60):
        """Run concurrent load test simulating multiple users"""
        print(f"🚀 Running concurrent load test ({concurrent_users} users, {duration_seconds}s)...")
        
        async def user_simulation(user_id: int):
            """Simulate a single user's behavior"""
            async with aiohttp.ClientSession() as session:
                end_time = time.time() + duration_seconds
                user_results = []
                
                while time.time() < end_time:
                    # Random user actions
                    actions = [
                        ("GET", "/health", None),
                        ("POST", "/api/chat/send", {"message": f"User {user_id} message", "session_id": f"user_{user_id}"}),
                        ("GET", "/api/performance", None),
                    ]
                    
                    for method, endpoint, data in actions:
                        if time.time() >= end_time:
                            break
                        
                        result = await self.make_request(session, method, endpoint, data)
                        user_results.append(result)
                        
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                
                return user_results
        
        # Run all user simulations concurrently
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for user_results in all_results:
            self.results.extend(user_results)
        
        return all_results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate performance report"""
        if not self.results:
            return {"error": "No test results to analyze"}
        
        # Group results by endpoint
        endpoint_stats = {}
        
        for result in self.results:
            key = f"{result.method} {result.endpoint}"
            if key not in endpoint_stats:
                endpoint_stats[key] = {
                    "response_times": [],
                    "success_count": 0,
                    "error_count": 0,
                    "status_codes": {}
                }
            
            endpoint_stats[key]["response_times"].append(result.response_time)
            
            if result.success:
                endpoint_stats[key]["success_count"] += 1
            else:
                endpoint_stats[key]["error_count"] += 1
            
            status = result.status_code
            endpoint_stats[key]["status_codes"][status] = endpoint_stats[key]["status_codes"].get(status, 0) + 1
        
        # Calculate statistics
        analysis = {
            "total_requests": len(self.results),
            "overall_success_rate": sum(1 for r in self.results if r.success) / len(self.results) * 100,
            "endpoints": {}
        }
        
        for endpoint, stats in endpoint_stats.items():
            response_times = stats["response_times"]
            total_requests = len(response_times)
            
            analysis["endpoints"][endpoint] = {
                "total_requests": total_requests,
                "success_rate": (stats["success_count"] / total_requests) * 100,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                "status_codes": stats["status_codes"]
            }
        
        return analysis

    async def run_comprehensive_test(self):
        """Run all performance tests"""
        print("🧪 Starting MasterX Load Testing Suite...")
        print("=" * 60)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Health endpoint
            await self.test_health_endpoint(session, 100)
            
            # Test 2: Chat endpoint
            await self.test_chat_endpoint(session, 50)
            
            # Test 3: File upload
            await self.test_file_upload(session, 20)
        
        # Test 4: WebSocket connections
        await self.test_websocket_connections(10)
        
        # Test 5: Concurrent load test
        await self.run_concurrent_load_test(25, 30)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST RESULTS")
        print("=" * 60)
        
        analysis = self.analyze_results()
        
        print(f"Total Test Duration: {total_time:.2f} seconds")
        print(f"Total Requests: {analysis['total_requests']}")
        print(f"Overall Success Rate: {analysis['overall_success_rate']:.2f}%")
        print()
        
        for endpoint, stats in analysis["endpoints"].items():
            print(f"📍 {endpoint}")
            print(f"   Requests: {stats['total_requests']}")
            print(f"   Success Rate: {stats['success_rate']:.2f}%")
            print(f"   Avg Response Time: {stats['avg_response_time']:.3f}s")
            print(f"   P95 Response Time: {stats['p95_response_time']:.3f}s")
            print(f"   Status Codes: {stats['status_codes']}")
            print()
        
        # Performance assessment
        avg_response_time = statistics.mean([r.response_time for r in self.results if r.success])
        success_rate = analysis['overall_success_rate']
        
        if success_rate >= 99 and avg_response_time <= 1.0:
            grade = "🏆 EXCELLENT"
        elif success_rate >= 95 and avg_response_time <= 2.0:
            grade = "✅ GOOD"
        elif success_rate >= 90 and avg_response_time <= 5.0:
            grade = "⚠️ ACCEPTABLE"
        else:
            grade = "❌ NEEDS IMPROVEMENT"
        
        print(f"🎯 Performance Grade: {grade}")
        print(f"📈 Production Readiness: {'✅ READY' if grade in ['🏆 EXCELLENT', '✅ GOOD'] else '⚠️ NEEDS OPTIMIZATION'}")
        
        return analysis

async def main():
    parser = argparse.ArgumentParser(description="MasterX Load Testing Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--users", type=int, default=25, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url)
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
