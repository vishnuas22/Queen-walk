#!/bin/bash

# MasterX Production Startup Script
set -e

echo "🚀 Starting MasterX Quantum Intelligence Platform..."

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for MongoDB
if [ ! -z "$MONGODB_URL" ]; then
    echo "Waiting for MongoDB..."
    while ! nc -z mongo 27017; do
        sleep 1
    done
    echo "✅ MongoDB is ready"
fi

# Wait for Redis
if [ ! -z "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    while ! nc -z redis 6379; do
        sleep 1
    done
    echo "✅ Redis is ready"
fi

# Set up logging
mkdir -p /app/logs
touch /app/logs/masterx.log

# Start backend server
echo "🔧 Starting backend server..."
cd /app/backend

# Run database migrations if needed
if [ "$ENVIRONMENT" = "production" ]; then
    echo "🗄️ Running production setup..."
    python -c "
import asyncio
import sys
sys.path.append('/app/backend')
from server import setup_production_database
asyncio.run(setup_production_database())
" || echo "⚠️ Database setup completed with warnings"
fi

# Start the FastAPI server
uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info \
    --access-log \
    --log-config /app/docker/logging.conf &

BACKEND_PID=$!

# Start frontend server (if needed)
if [ -d "/app/frontend/.next" ]; then
    echo "🎨 Starting frontend server..."
    cd /app/frontend
    npm start --port 3000 &
    FRONTEND_PID=$!
fi

# Health check function
health_check() {
    echo "🏥 Performing health check..."
    
    # Check backend
    if ! curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "❌ Backend health check failed"
        return 1
    fi
    
    # Check frontend (if running)
    if [ ! -z "$FRONTEND_PID" ]; then
        if ! curl -f http://localhost:3000 >/dev/null 2>&1; then
            echo "❌ Frontend health check failed"
            return 1
        fi
    fi
    
    echo "✅ Health check passed"
    return 0
}

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Perform initial health check
if health_check; then
    echo "🎉 MasterX is running successfully!"
    echo "📊 Backend API: http://localhost:8000"
    echo "🎨 Frontend UI: http://localhost:3000"
    echo "📈 Health Check: http://localhost:8000/health"
    echo "📋 API Docs: http://localhost:8000/docs"
else
    echo "❌ Health check failed during startup"
    exit 1
fi

# Graceful shutdown handler
cleanup() {
    echo "🛑 Shutting down MasterX..."
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    kill $BACKEND_PID 2>/dev/null || true
    
    echo "✅ Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Keep the script running and monitor processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died, exiting..."
        exit 1
    fi
    
    # Check if frontend is still running (if it was started)
    if [ ! -z "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process died, exiting..."
        exit 1
    fi
    
    # Periodic health check
    if ! health_check; then
        echo "❌ Health check failed, exiting..."
        exit 1
    fi
    
    sleep 30
done
