# KG-HTC Docker Setup Guide

This guide will help you set up and run the KG-HTC (Knowledge Graph Hierarchical Text Classification) project using Docker.

## 🚀 Quick Start

### Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- OpenAI API key

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd KG-HTC
```

### 2. Run the Setup Script

```bash
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh
```

The script will:
- Check Docker installation
- Create a `.env` file template
- Build Docker containers
- Start all services

### 3. Update Environment Variables

Edit the `.env` file with your actual API keys:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_actual_openai_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 4. Access the Application

- **Streamlit App**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474

## 🐳 Docker Commands

### Development Mode

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

### Production Mode

```bash
# Start production services
docker-compose -f docker-compose.prod.yml up -d

# View production logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop production services
docker-compose -f docker-compose.prod.yml down
```

## 📁 Project Structure

```
KG-HTC/
├── Dockerfile                 # Main application container
├── docker-compose.yml         # Development services
├── docker-compose.prod.yml    # Production services
├── requirements.txt           # Python dependencies
├── .dockerignore             # Files to exclude from Docker build
├── nginx.conf                # Nginx configuration for production
├── scripts/
│   ├── docker-setup.sh       # Development setup script
│   └── docker-prod.sh        # Production setup script
├── src/                      # Source code
├── code_review/              # Review classification code
├── dataset/                  # Data files (mounted as volume)
├── database/                 # Vector database (mounted as volume)
└── prompts/                  # Prompt templates (mounted as volume)
```

## 🔧 Services

### 1. KG-HTC App (Streamlit)
- **Port**: 8501
- **Purpose**: Main web interface for review classification
- **Features**: Single review analysis, batch processing, results visualization

### 2. Neo4j Database
- **Ports**: 7474 (HTTP), 7687 (Bolt)
- **Purpose**: Graph database for hierarchical label relationships
- **Features**: Knowledge graph storage and querying

### 3. Nginx (Production only)
- **Ports**: 80 (HTTP), 443 (HTTPS)
- **Purpose**: Reverse proxy and load balancer
- **Features**: SSL termination, caching, security headers

## 🛠️ Customization

### Environment Variables

Key environment variables you can customize:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-3.5-turbo-0125

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password

# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Adding SSL Certificates

For production with HTTPS:

1. Place your SSL certificates in the `ssl/` directory:
   ```
   ssl/
   ├── cert.pem
   └── key.pem
   ```

2. Update `nginx.conf` to uncomment SSL configuration lines

3. Restart the production services

### Custom Prompts

You can customize the prompts by editing files in the `prompts/` directory:

- `prompts/system/custom/llm_graph.txt` - System prompts
- `prompts/user/custom/llm_graph.txt` - User prompts

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the port
   lsof -i :8501
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Neo4j connection issues**
   ```bash
   # Check Neo4j logs
   docker-compose logs neo4j
   
   # Restart Neo4j
   docker-compose restart neo4j
   ```

3. **OpenAI API errors**
   - Verify your API key in `.env`
   - Check API quota and billing
   - Ensure network connectivity

### Logs and Debugging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f kg-htc-app
docker-compose logs -f neo4j

# Access container shell
docker-compose exec kg-htc-app bash
docker-compose exec neo4j bash
```

### Performance Tuning

For production deployments:

1. **Resource Limits**: Adjust memory and CPU limits in `docker-compose.prod.yml`
2. **Neo4j Tuning**: Modify Neo4j memory settings in the environment variables
3. **Caching**: Configure nginx caching for static assets

## 🚀 Deployment

### Local Development
```bash
./scripts/docker-setup.sh
```

### Production Deployment
```bash
./scripts/docker-prod.sh
```

### Cloud Deployment

For cloud deployment (AWS, GCP, Azure):

1. Build and push the Docker image to a registry
2. Use the production compose file
3. Set up proper SSL certificates
4. Configure environment variables securely
5. Set up monitoring and logging

## 📊 Monitoring

### Health Checks
- Application: http://localhost:8501/_stcore/health
- Nginx: http://localhost/health

### Metrics
- Container resource usage: `docker stats`
- Application logs: `docker-compose logs -f`
- Neo4j metrics: http://localhost:7474

## 🔒 Security

### Best Practices

1. **Use strong passwords** for Neo4j
2. **Enable SSL** in production
3. **Keep dependencies updated**
4. **Use secrets management** for sensitive data
5. **Regular security audits**

### Network Security

- Services communicate over internal Docker network
- Only necessary ports are exposed
- Nginx provides additional security layer in production

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review logs and error messages
- Open an issue on GitHub 