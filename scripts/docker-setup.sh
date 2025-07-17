#!/bin/bash

# KG-HTC Docker Setup Script
# This script helps you set up and run the KG-HTC project in Docker

set -e

echo "🚀 KG-HTC Docker Setup"
echo "======================"

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

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Optional: For production, use a stronger password
# NEO4J_PASSWORD=your_secure_password_here
EOF
    echo "✅ .env file created. Please edit it with your actual API keys."
    echo "⚠️  IMPORTANT: Update the .env file with your OpenAI API key before continuing!"
    read -p "Press Enter after updating .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p dataset database prompts/system/custom prompts/user/custom

# Function to check if .env has been updated
check_env() {
    if grep -q "your_openai_api_key_here" .env; then
        echo "❌ Please update your OpenAI API key in .env file first!"
        exit 1
    fi
}

# Build and run containers
echo "🔨 Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "🌐 Access your application:"
    echo "   - Streamlit App: http://localhost:8501"
    echo "   - Neo4j Browser: http://localhost:7474"
    echo ""
    echo "📊 Useful commands:"
    echo "   - View logs: docker-compose logs -f"
    echo "   - Stop services: docker-compose down"
    echo "   - Restart services: docker-compose restart"
    echo "   - Rebuild: docker-compose up --build"
else
    echo "❌ Services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your KG-HTC application is now running in Docker." 