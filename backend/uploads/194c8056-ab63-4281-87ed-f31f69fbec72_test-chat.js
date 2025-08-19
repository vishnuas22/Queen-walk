// Simple test script to verify frontend-backend communication
// Run this in the browser console when on http://localhost:3000

console.log('🧪 Starting MasterX Chat Interface Test...');

// Test 1: Check if API configuration is correct
const API_BASE = 'http://localhost:8000/api';
console.log('📡 API Base URL:', API_BASE);

// Test 2: Test backend health endpoint
async function testBackendHealth() {
  try {
    const response = await fetch(`${API_BASE.replace('/api', '')}/health`);
    const data = await response.json();
    console.log('✅ Backend Health:', data);
    return true;
  } catch (error) {
    console.error('❌ Backend Health Check Failed:', error);
    return false;
  }
}

// Test 3: Test chat endpoint directly
async function testChatEndpoint() {
  try {
    const response = await fetch(`${API_BASE}/chat/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: 'Test message from frontend validation script',
        task_type: 'general'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Chat Endpoint Response:', data);
    return true;
  } catch (error) {
    console.error('❌ Chat Endpoint Test Failed:', error);
    return false;
  }
}

// Test 4: Check if chat interface elements are present
function testChatInterfaceElements() {
  const elements = {
    chatInput: document.querySelector('textarea[placeholder*="Ask me anything"]'),
    sendButton: document.querySelector('button[class*="quantum-button"]'),
    messagesContainer: document.querySelector('.quantum-scroll'),
    settingsButton: document.querySelector('button[class*="neural-network-button"]')
  };
  
  console.log('🔍 Chat Interface Elements:');
  Object.entries(elements).forEach(([name, element]) => {
    if (element) {
      console.log(`✅ ${name}: Found`);
    } else {
      console.log(`❌ ${name}: Not found`);
    }
  });
  
  return Object.values(elements).every(el => el !== null);
}

// Test 5: Simulate sending a message through the UI
function simulateMessageSend() {
  const textarea = document.querySelector('textarea[placeholder*="Ask me anything"]');
  const sendButton = document.querySelector('button[class*="quantum-button"]');
  
  if (textarea && sendButton) {
    console.log('🤖 Simulating message send...');
    
    // Set test message
    textarea.value = 'Hello from automated test!';
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    
    // Click send button
    sendButton.click();
    
    console.log('✅ Message send simulation completed');
    return true;
  } else {
    console.error('❌ Could not find chat input elements');
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('\n🚀 Running MasterX Chat Interface Tests...\n');
  
  const results = {
    backendHealth: await testBackendHealth(),
    chatEndpoint: await testChatEndpoint(),
    interfaceElements: testChatInterfaceElements(),
  };
  
  console.log('\n📊 Test Results Summary:');
  console.log('========================');
  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASSED' : 'FAILED'}`);
  });
  
  const allPassed = Object.values(results).every(result => result);
  console.log(`\n🎯 Overall Status: ${allPassed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}`);
  
  if (allPassed) {
    console.log('\n🎉 MasterX Chat Interface is fully operational!');
    console.log('💡 You can now test sending a message by running: simulateMessageSend()');
  }
  
  return results;
}

// Auto-run tests when script is loaded
runAllTests();

// Export functions for manual testing
window.masterxTest = {
  runAllTests,
  testBackendHealth,
  testChatEndpoint,
  testChatInterfaceElements,
  simulateMessageSend
};

console.log('\n💡 Available test functions:');
console.log('- masterxTest.runAllTests()');
console.log('- masterxTest.simulateMessageSend()');
console.log('- masterxTest.testBackendHealth()');
console.log('- masterxTest.testChatEndpoint()');
