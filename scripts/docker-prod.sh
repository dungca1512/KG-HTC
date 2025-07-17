#!/bin/bash

# KG-HTC Production Docker Setup Script
# This script helps you set up and run the KG-HTC project in production mode

set -e

echo "🚀 KG-HTC Production Docker Setup"
echo "================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run the development setup first or create .env manually."
    exit 1
fi

# Check if .env has been updated
if grep -q "your_openai_api_key_here" .env; then
    echo "❌ Please update your OpenAI API key in .env file first!"
    exit 1
fi

# Create SSL directory for nginx
echo "📁 Creating SSL directory..."
mkdir -p ssl

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p dataset database prompts/system/custom prompts/user/custom

# Build and run production containers
echo "🔨 Building production Docker containers..."
docker-compose -f docker-compose.prod.yml build

echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 45

# Check if services are running
echo "🔍 Checking service status..."
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "✅ Production services are running!"
    echo ""
    echo "🌐 Access your application:"
    echo "   - Main App: http://localhost (via nginx)"
    echo "   - Streamlit App: http://localhost:8501 (direct)"
    echo "   - Neo4j Browser: http://localhost:7474"
    echo ""
    echo "📊 Useful commands:"
    echo "   - View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "   - Stop services: docker-compose -f docker-compose.prod.yml down"
    echo "   - Restart services: docker-compose -f docker-compose.prod.yml restart"
    echo "   - Rebuild: docker-compose -f docker-compose.prod.yml up --build"
    echo ""
    echo "🔒 Security notes:"
    echo "   - Update nginx.conf for SSL certificates"
    echo "   - Use strong passwords in .env"
    echo "   - Consider using Docker secrets for sensitive data"
else
    echo "❌ Services failed to start. Check logs with: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

echo ""
echo "🎉 Production setup complete! Your KG-HTC application is now running in production mode." 