# 🚀 MasterX Quantum Intelligence API - Quick Start Guide

## 📋 Prerequisites

- Python 3.11+
- Required packages (install with pip)

```bash
pip install fastapi uvicorn pydantic bcrypt aiohttp websockets sse-starlette python-jose passlib python-multipart PyJWT
```

## 🏃‍♂️ Quick Start

### 1. Start the API Server

```bash
cd /Users/Dataghost/MasterX
python3 -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔐 Authentication

### Login to Get Access Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "student@example.com",
    "password": "student123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user_info": {
    "user_id": "student_001",
    "email": "student@example.com",
    "name": "Test Student",
    "role": "student"
  }
}
```

### Use Token in Requests

```bash
curl -X GET "http://localhost:8000/api/v1/learning/goals" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 💬 Chat API Examples

### Send Chat Message

```bash
curl -X POST "http://localhost:8000/api/v1/chat/message" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, can you help me learn Python?",
    "message_type": "text"
  }'
```

### Stream Chat Response

```bash
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain machine learning concepts",
    "stream": true
  }'
```

## 📚 Learning API Examples

### Create Learning Goal

```bash
curl -X POST "http://localhost:8000/api/v1/learning/goals" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Learn Python Programming",
    "description": "Master Python fundamentals and advanced concepts",
    "subject": "Programming",
    "target_skills": ["variables", "functions", "classes", "modules"],
    "difficulty_level": 0.6,
    "estimated_duration_hours": 40
  }'
```

### Record Learning Session

```bash
curl -X POST "http://localhost:8000/api/v1/learning/sessions" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Python Programming",
    "duration_minutes": 45,
    "activities": [
      {"type": "reading", "content": "Python basics"},
      {"type": "coding", "content": "Hello World program"}
    ],
    "performance_metrics": {
      "comprehension": 0.8,
      "speed": 0.7
    },
    "engagement_score": 0.85
  }'
```

## 📊 Analytics API Examples

### Generate Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/analytics/predict" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_type": "learning_outcome",
    "time_horizon": "medium_term",
    "include_interventions": true
  }'
```

### Get Analytics Dashboard

```bash
curl -X GET "http://localhost:8000/api/v1/analytics/dashboard" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 📝 Content Generation Examples

### Generate Learning Content

```bash
curl -X POST "http://localhost:8000/api/v1/content/generate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Mathematics",
    "topic": "Linear Algebra",
    "content_type": "lesson",
    "difficulty_level": 0.7,
    "duration_minutes": 30,
    "learning_objectives": [
      "Understand vector operations",
      "Learn matrix multiplication"
    ]
  }'
```

## 📋 Assessment API Examples

### Create Assessment

```bash
curl -X POST "http://localhost:8000/api/v1/assessment/create" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Python Programming",
    "topics": ["variables", "functions", "loops"],
    "assessment_type": "quiz",
    "difficulty_level": 0.6,
    "duration_minutes": 30,
    "question_count": 10
  }'
```

## 🌊 Real-Time Features

### Server-Sent Events (SSE)

```bash
curl -X GET "http://localhost:8000/api/v1/streaming/events?event_types=notification,learning_update" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Accept: text/event-stream"
```

### WebSocket Connection (JavaScript)

```javascript
const token = "YOUR_ACCESS_TOKEN";
const ws = new WebSocket(`ws://localhost:8000/api/v1/websocket/connect?token=${token}`);

ws.onopen = function(event) {
    console.log("Connected to WebSocket");
    
    // Send a message
    ws.send(JSON.stringify({
        type: "chat",
        message: "Hello from WebSocket!"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log("Received:", data);
};
```

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# LLM API Keys
export GROQ_API_KEY="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"
export GEMINI_API_KEY="AIzaSyCmV-mlB7rag8GurIDj07ijRDhPuNwOiVA"
export OPENAI_API_KEY="your_openai_key_here"

# JWT Configuration
export JWT_SECRET="your-secret-key-change-in-production"
export JWT_EXPIRATION_HOURS="24"

# Database Configuration (if using external DB)
export DATABASE_URL="postgresql://user:password@localhost/masterx"
export REDIS_URL="redis://localhost:6379"
```

## 🧪 Testing

### Run API Integration Tests

```bash
cd /Users/Dataghost/MasterX
python3 backend/test_api_system.py
```

### Run Legacy File Analysis

```bash
cd /Users/Dataghost/MasterX
python3 backend/api/legacy_analysis.py
```

## 📚 Available Test Users

| Email | Password | Role |
|-------|----------|------|
| admin@masterx.ai | admin123 | admin |
| student@example.com | student123 | student |
| teacher@example.com | teacher123 | teacher |

## 🔍 API Endpoints Overview

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token

### Chat
- `POST /api/v1/chat/message` - Send chat message
- `POST /api/v1/chat/stream` - Stream chat response
- `GET /api/v1/chat/sessions/{session_id}` - Get chat session
- `GET /api/v1/chat/sessions` - List chat sessions

### Learning
- `POST /api/v1/learning/goals` - Create learning goal
- `GET /api/v1/learning/goals` - Get learning goals
- `PUT /api/v1/learning/goals/{goal_id}` - Update learning goal
- `POST /api/v1/learning/sessions` - Record learning session
- `POST /api/v1/learning/progress` - Get learning progress

### Personalization
- `GET /api/v1/personalization/profile` - Get learning DNA profile
- `POST /api/v1/personalization/update` - Update personalization

### Analytics
- `POST /api/v1/analytics/predict` - Generate predictions
- `GET /api/v1/analytics/dashboard` - Get analytics dashboard

### Content
- `POST /api/v1/content/generate` - Generate content
- `GET /api/v1/content/library` - Get content library

### Assessment
- `POST /api/v1/assessment/create` - Create assessment
- `POST /api/v1/assessment/submit/{assessment_id}` - Submit assessment
- `GET /api/v1/assessment/results/{assessment_id}` - Get assessment results

### Streaming
- `GET /api/v1/streaming/events` - Stream events (SSE)
- `GET /api/v1/streaming/progress` - Stream progress (SSE)
- `GET /api/v1/streaming/notifications` - Stream notifications (SSE)

### WebSocket
- `WS /api/v1/websocket/connect` - Main WebSocket connection
- `WS /api/v1/websocket/learning/{session_id}` - Learning session WebSocket

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: masterx-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: masterx-api
  template:
    metadata:
      labels:
        app: masterx-api
    spec:
      containers:
      - name: api
        image: masterx/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: groq-api-key
```

## 📞 Support

For questions or issues:
- Check the interactive documentation at `/docs`
- Review the implementation summary in `PHASE_12_IMPLEMENTATION_SUMMARY.md`
- Run the test suite to verify functionality
- Check logs for detailed error information

**The MasterX Quantum Intelligence API is now ready for production use!** 🎉
