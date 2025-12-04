---
title: "Chapter 2: Containerization Technology"
chapter_title: "Chapter 2: Containerization Technology"
subtitle: Portability and Reproducibility of Machine Learning Models with Docker
reading_time: 25-30 minutes
difficulty: Beginner-Intermediate
code_examples: 8
version: 1.0
created_at: "by:"
---

This chapter covers Containerization Technology. You will learn fundamental concepts of Docker containers, Create Dockerfiles for machine learning models, and Efficiently containerize ML models.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the fundamental concepts of Docker containers and their differences from virtual machines
  * ✅ Create Dockerfiles for machine learning models
  * ✅ Efficiently containerize ML models
  * ✅ Manage multiple services with Docker Compose
  * ✅ Build GPU-enabled ML containers

* * *

## 2.1 Docker Fundamentals

### What is a Container

**Container** is a technology that packages an application and its dependencies into an isolated environment.

> "Build once, Run anywhere" - Build it once, and it runs the same way everywhere

### Docker vs Virtual Machines

Feature | Docker Container | Virtual Machine (VM)  
---|---|---  
**Startup Time** | Seconds | Minutes  
**Resources** | Lightweight (MBs) | Heavy (GBs)  
**Isolation Level** | Process-level | Complete OS isolation  
**Performance** | Near-native | Has overhead  
**Portability** | High | Moderate  
      
    
    ```mermaid
    graph TD
        subgraph "Virtual Machine"
            A1[App1] --> B1[Guest OS1]
            A2[App2] --> B2[Guest OS2]
            B1 --> C[Hypervisor]
            B2 --> C
            C --> D[Host OS]
            D --> E[Physical Server]
        end
    
        subgraph "Docker Container"
            F1[App1] --> G[Docker Engine]
            F2[App2] --> G
            G --> H[Host OS]
            H --> I[Physical Server]
        end
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style F1 fill:#c8e6c9
        style F2 fill:#c8e6c9
    ```

### Basic Docker Commands
    
    
    # Check Docker version
    docker --version
    
    # List images
    docker images
    
    # List running containers
    docker ps
    
    # List all containers
    docker ps -a
    
    # Download image
    docker pull python:3.9-slim
    
    # Run container
    docker run -it python:3.9-slim bash
    
    # Stop container
    docker stop <container_id>
    
    # Remove container
    docker rm <container_id>
    
    # Remove image
    docker rmi <image_id>
    
    # Cleanup entire system
    docker system prune -a
    

### Relationship Between Images and Containers

**Image** : Blueprint of the application (read-only)

**Container** : Executable instance created from an image
    
    
    ```mermaid
    graph LR
        A[Dockerfile] -->|docker build| B[Docker Image]
        B -->|docker run| C[Container 1]
        B -->|docker run| D[Container 2]
        B -->|docker run| E[Container 3]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
    ```

> **Important** : Multiple containers can be started from a single image. Each container is an independent environment.

* * *

## 2.2 Creating a Dockerfile

### Selecting a Base Image

Representative base images for machine learning models:

Image | Size | Use Case  
---|---|---  
`python:3.9-slim` | ~120MB | Lightweight Python environment  
`python:3.9` | ~900MB | Full-featured Python environment  
`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` | ~2GB | GPU inference  
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | ~4GB | GPU development and training  
  
### Basic Dockerfile Structure
    
    
    # Specify base image
    FROM python:3.9-slim
    
    # Set working directory
    WORKDIR /app
    
    # Update and install system packages
    RUN apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python packages
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy application code
    COPY . .
    
    # Expose port
    EXPOSE 8000
    
    # Startup command
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### Multi-stage Build

A technique to reduce image size and improve security:
    
    
    # Stage 1: Build environment
    FROM python:3.9 as builder
    
    WORKDIR /build
    
    # Install dependencies
    COPY requirements.txt .
    RUN pip install --user --no-cache-dir -r requirements.txt
    
    # Stage 2: Runtime environment (lightweight)
    FROM python:3.9-slim
    
    WORKDIR /app
    
    # Copy only necessary files from build stage
    COPY --from=builder /root/.local /root/.local
    COPY . .
    
    # Set PATH
    ENV PATH=/root/.local/bin:$PATH
    
    EXPOSE 8000
    
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    

> **Effect** : Multi-stage builds can reduce image size by 50-70%.

### Optimization Techniques

#### Leveraging Layer Cache
    
    
    # ❌ Inefficient: Dependencies are reinstalled every time code changes
    FROM python:3.9-slim
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    
    # ✅ Efficient: Cache is used unless dependencies change
    FROM python:3.9-slim
    WORKDIR /app
    
    # Install dependencies first (changes infrequently)
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy code later (changes frequently)
    COPY . .
    

#### Excluding Unnecessary Files

Example .dockerignore file:
    
    
    # .dockerignore
    __pycache__
    *.pyc
    *.pyo
    *.pyd
    .Python
    *.so
    *.egg
    *.egg-info
    dist
    build
    .git
    .gitignore
    .env
    .venv
    venv/
    data/
    notebooks/
    tests/
    *.md
    Dockerfile
    docker-compose.yml
    

* * *

## 2.3 Containerizing ML Models

### Dockerfile for FastAPI + PyTorch
    
    
    # Multi-stage build
    FROM python:3.9 as builder
    
    WORKDIR /build
    
    # Copy dependency file
    COPY requirements.txt .
    
    # Install dependencies
    RUN pip install --user --no-cache-dir \
        torch==2.0.0 \
        torchvision==0.15.0 \
        fastapi==0.104.0 \
        uvicorn[standard]==0.24.0 \
        pydantic==2.5.0 \
        pillow==10.1.0
    
    # Runtime environment
    FROM python:3.9-slim
    
    WORKDIR /app
    
    # Copy dependencies from build stage
    COPY --from=builder /root/.local /root/.local
    
    # Copy application code and model
    COPY app/ ./app/
    COPY models/ ./models/
    
    # Set environment variables
    ENV PATH=/root/.local/bin:$PATH \
        PYTHONUNBUFFERED=1 \
        MODEL_PATH=/app/models/model.pth
    
    # Create non-root user (improved security)
    RUN useradd -m -u 1000 appuser && \
        chown -R appuser:appuser /app
    
    USER appuser
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD python -c "import requests; requests.get('http://localhost:8000/health')"
    
    EXPOSE 8000
    
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### Example requirements.txt
    
    
    # requirements.txt
    torch==2.0.0
    torchvision==0.15.0
    fastapi==0.104.0
    uvicorn[standard]==0.24.0
    pydantic==2.5.0
    pillow==10.1.0
    numpy==1.24.3
    python-multipart==0.0.6
    

### Building and Running Images
    
    
    # Build image
    docker build -t ml-api:v1.0 .
    
    # Display detailed build log
    docker build -t ml-api:v1.0 --progress=plain .
    
    # Build without cache
    docker build -t ml-api:v1.0 --no-cache .
    
    # Run container
    docker run -d \
        --name ml-api \
        -p 8000:8000 \
        -v $(pwd)/models:/app/models \
        ml-api:v1.0
    
    # Check logs
    docker logs ml-api
    
    # Display real-time logs
    docker logs -f ml-api
    
    # Execute command inside container
    docker exec -it ml-api bash
    
    # Stop and remove container
    docker stop ml-api
    docker rm ml-api
    

### Port Mapping

Option | Description | Example  
---|---|---  
`-p 8000:8000` | Host:Container | Host port 8000 to container port 8000  
`-p 8080:8000` | Different ports | Host port 8080 to container port 8000  
`-p 127.0.0.1:8000:8000` | Localhost only | Accessible only from localhost  
  
* * *

## 2.4 Orchestration with Docker Compose

### docker-compose.yml Configuration

Configuration file for integrated management of multiple services:
    
    
    # docker-compose.yml
    version: '3.8'
    
    services:
      # FastAPI application
      api:
        build:
          context: .
          dockerfile: Dockerfile
        container_name: ml-api
        ports:
          - "8000:8000"
        environment:
          - MODEL_PATH=/app/models/model.pth
          - REDIS_HOST=redis
          - REDIS_PORT=6379
        volumes:
          - ./models:/app/models:ro
          - ./logs:/app/logs
        depends_on:
          - redis
        restart: unless-stopped
        networks:
          - ml-network
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 40s
    
      # Redis cache
      redis:
        image: redis:7-alpine
        container_name: ml-redis
        ports:
          - "6379:6379"
        volumes:
          - redis-data:/data
        restart: unless-stopped
        networks:
          - ml-network
        command: redis-server --appendonly yes
    
    networks:
      ml-network:
        driver: bridge
    
    volumes:
      redis-data:
    

### Example of Multiple Service Integration
    
    
    # docker-compose.yml (extended version)
    version: '3.8'
    
    services:
      # ML model inference API
      ml-api:
        build: ./api
        ports:
          - "8000:8000"
        environment:
          - REDIS_HOST=redis
          - DB_HOST=postgres
        volumes:
          - ./models:/app/models:ro
        depends_on:
          - redis
          - postgres
        networks:
          - ml-network
    
      # Cache layer
      redis:
        image: redis:7-alpine
        volumes:
          - redis-data:/data
        networks:
          - ml-network
    
      # Database
      postgres:
        image: postgres:15-alpine
        environment:
          - POSTGRES_USER=mluser
          - POSTGRES_PASSWORD=mlpass
          - POSTGRES_DB=mldb
        volumes:
          - postgres-data:/var/lib/postgresql/data
        networks:
          - ml-network
    
      # Monitoring
      prometheus:
        image: prom/prometheus:latest
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
          - prometheus-data:/prometheus
        networks:
          - ml-network
    
    networks:
      ml-network:
        driver: bridge
    
    volumes:
      redis-data:
      postgres-data:
      prometheus-data:
    

### Volume Mounts

Type | Syntax | Use Case  
---|---|---  
**Bind Mount** | `./host/path:/container/path` | Code synchronization during development  
**Named Volume** | `volume-name:/container/path` | Persistent data storage  
**Read-only** | `./path:/path:ro` | Model files, etc.  
  
### Environment Variable Management

Example .env file:
    
    
    # .env
    MODEL_PATH=/app/models/resnet50.pth
    REDIS_HOST=redis
    REDIS_PORT=6379
    LOG_LEVEL=INFO
    MAX_WORKERS=4
    

Usage in docker-compose.yml:
    
    
    services:
      api:
        env_file:
          - .env
        # Or specify individually
        environment:
          - MODEL_PATH=${MODEL_PATH}
          - REDIS_HOST=${REDIS_HOST}
    

### Docker Compose Commands
    
    
    # Start services (background)
    docker-compose up -d
    
    # Start services (with logs)
    docker-compose up
    
    # Build and start services
    docker-compose up -d --build
    
    # Start only specific services
    docker-compose up -d api redis
    
    # Stop services
    docker-compose stop
    
    # Stop and remove services
    docker-compose down
    
    # Remove including volumes
    docker-compose down -v
    
    # Check logs
    docker-compose logs -f
    
    # Logs for specific service
    docker-compose logs -f api
    
    # Check service status
    docker-compose ps
    
    # Restart service
    docker-compose restart api
    

* * *

## 2.5 Hands-on: GPU-enabled ML Containers

### NVIDIA Docker Setup

Prerequisites:

  * Machine with NVIDIA GPU
  * NVIDIA driver installed
  * NVIDIA Container Toolkit installed

    
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Restart Docker
    sudo systemctl restart docker
    
    # Verify GPU
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    

### Dockerfile Using CUDA Image
    
    
    # Dockerfile for GPU inference
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
    
    # Install Python
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # Install PyTorch GPU version
    COPY requirements-gpu.txt .
    RUN pip3 install --no-cache-dir -r requirements-gpu.txt
    
    # Application code and model
    COPY app/ ./app/
    COPY models/ ./models/
    
    ENV PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES=0
    
    EXPOSE 8000
    
    CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### requirements-gpu.txt
    
    
    # requirements-gpu.txt
    torch==2.0.0+cu118
    torchvision==0.15.0+cu118
    --extra-index-url https://download.pytorch.org/whl/cu118
    fastapi==0.104.0
    uvicorn[standard]==0.24.0
    pydantic==2.5.0
    pillow==10.1.0
    numpy==1.24.3
    

### GPU Inference Implementation

Example app/main.py:
    
    
    # Requirements:
    # - Python 3.9+
    # - fastapi>=0.100.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Example app/main.py:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # app/main.py
    import torch
    from fastapi import FastAPI, File, UploadFile
    from PIL import Image
    import io
    
    app = FastAPI()
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = torch.load("/app/models/model.pth", map_location=device)
    model.eval()
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    
        # Preprocessing (omitted)
        # tensor = preprocess(image)
    
        # GPU inference
        with torch.no_grad():
            # tensor = tensor.to(device)
            # output = model(tensor)
            pass
    
        return {"prediction": "result"}
    

### Using GPU with Docker Compose
    
    
    # docker-compose-gpu.yml
    version: '3.8'
    
    services:
      ml-api-gpu:
        build:
          context: .
          dockerfile: Dockerfile.gpu
        container_name: ml-api-gpu
        ports:
          - "8000:8000"
        volumes:
          - ./models:/app/models:ro
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]
        environment:
          - CUDA_VISIBLE_DEVICES=0
        restart: unless-stopped
    

Startup commands:
    
    
    # Start GPU-enabled container
    docker-compose -f docker-compose-gpu.yml up -d
    
    # Check GPU usage
    docker exec ml-api-gpu nvidia-smi
    
    # Check logs
    docker-compose -f docker-compose-gpu.yml logs -f
    

### Performance Comparison

Environment | Inference Time (1 image) | Throughput (images/sec) | Notes  
---|---|---|---  
**CPU (8 cores)** | 150ms | 6.7 | python:3.9-slim  
**GPU (RTX 3090)** | 15ms | 66.7 | nvidia/cuda:11.8.0  
**Speedup** | 10x | 10x | Batch size 1  
  
> **Note** : GPU throughput can be further improved by increasing batch size.

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **Docker Fundamentals**

     * Differences between containers and virtual machines
     * Basic Docker commands
     * Relationship between images and containers
  2. **Creating Dockerfiles**

     * Selecting appropriate base images
     * Optimization with multi-stage builds
     * Leveraging layer cache
  3. **Containerizing ML Models**

     * Dockerizing FastAPI + PyTorch
     * Efficiency with .dockerignore
     * Security and health checks
  4. **Docker Compose Orchestration**

     * Integrated management of multiple services
     * Managing volumes and environment variables
     * Service dependencies
  5. **GPU-enabled ML Containers**

     * Setting up NVIDIA Docker
     * Using CUDA images
     * 10x performance compared to CPU

### Best Practices

Principle | Description  
---|---  
**Lightweight Images** | Prioritize slim or alpine-based images  
**Layer Optimization** | Place less frequently changed items first  
**Multi-stage Builds** | Separate build and runtime environments  
**Non-root User** | For improved security  
**.dockerignore** | Exclude unnecessary files  
**Health Checks** | Monitor service health  
**Environment Variables** | Externalize configuration  
  
### Next Chapter

In Chapter 3, we will learn about **Orchestration with Kubernetes** :

  * Basic Kubernetes concepts
  * Creating Pods, Services, and Deployments
  * Scaling and load balancing
  * Managing ConfigMaps and Secrets
  * Deployment to production environments
