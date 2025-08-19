// Test script to send a markdown message to the chat API
const testMessage = `# Welcome to MasterX Premium Markdown!

This is a **comprehensive test** of our *premium markdown rendering system*.

## Code Examples

Here's some JavaScript code:

\`\`\`javascript
function greetUser(name) {
    console.log(\`Hello, \${name}! Welcome to MasterX.\`);
    return {
        message: "Welcome!",
        timestamp: new Date(),
        user: name
    };
}

// Call the function
greetUser("Developer");
\`\`\`

And some Python:

\`\`\`python
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Generate first 10 Fibonacci numbers
for i in range(10):
    print(f"F({i}) = {calculate_fibonacci(i)}")
\`\`\`

## Lists and Features

### Unordered List:
- ✨ Premium glass morphism design
- 🎨 Quantum intelligence branding
- 💎 Syntax highlighting for 50+ languages
- 🚀 Real-time streaming compatibility
- 📱 Responsive design

### Ordered List:
1. **Headers** with gradient text effects
2. **Code blocks** with copy functionality
3. **Tables** with hover animations
4. **Links** with external indicators
5. **Blockquotes** with glass styling

## Blockquote Example

> "The future belongs to those who understand quantum intelligence and harness the power of AI to transform human potential."
> 
> — MasterX Philosophy

## Table Example

| Feature | Status | Quality |
|---------|--------|---------|
| Markdown Rendering | ✅ Active | Premium |
| Syntax Highlighting | ✅ Active | Enterprise |
| Glass Morphism | ✅ Active | Quantum |
| Streaming Support | ✅ Active | Real-time |

## Inline Elements

This text contains \`inline code\`, **bold text**, *italic text*, and [external links](https://example.com).

---

**Test completed!** If you can see this message with proper formatting, the premium markdown rendering system is working perfectly! 🎉`;

console.log('Sending test message with markdown content...');
console.log('Message length:', testMessage.length);
console.log('First 200 characters:', testMessage.substring(0, 200) + '...');

// Send the message to the API
fetch('http://localhost:8000/api/v1/chat/message', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        message: "Please respond with a comprehensive markdown example that includes headers, code blocks, lists, tables, and other formatting elements to test our premium markdown rendering system.",
        message_type: 'text'
    })
})
.then(response => response.json())
.then(data => {
    console.log('✅ Response received!');
    console.log('Session ID:', data.session_id);
    console.log('Response length:', data.response?.length || 0);
    console.log('First 300 characters of response:');
    console.log(data.response?.substring(0, 300) + '...');
})
.catch(error => {
    console.error('❌ Error:', error);
});
