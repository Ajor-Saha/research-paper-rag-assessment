# Installation Guide

This guide covers installation and setup for the Research Paper RAG Assessment system in both Docker and local environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Docker Installation](#docker-installation)
- [Local Installation](#local-installation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Common Requirements
- **Git**: For cloning the repository
- **Python 3.11+**: For running the application
- **PostgreSQL Database**: Neon cloud database 

### Docker-Specific Requirements
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher

### Local-Specific Requirements
- **Ollama**: For local LLM inference
- **Qdrant**: Vector database

---

## Docker Installation

Docker provides the easiest setup with all services containerized.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ajor-Saha/research-paper-rag-assessment.git
cd research-paper-rag-assessment
```

### Step 2: Set Up Environment Variables

> **⚠️ Important Note**: The `.env` file with actual credentials is included in this repository **for judgment purposes only**. These environment variables will be removed after the assessment is complete. In production, never commit sensitive credentials to version control.


### Step 3: Build and Start Services

```bash
# Start all services (Qdrant, Ollama, FastAPI)
docker-compose up -d
```


### Test uploading a paper:
```bash
curl -X POST "http://localhost:8000/api/papers/upload" \
  -F "file=@sample_papers/your_paper.pdf"
```



## Local Installation

For local development without Docker containers.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ajor-Saha/research-paper-rag-assessment.git
cd research-paper-rag-assessment
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Ollama

**macOS/Linux:**
```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Pull the model
ollama pull gemma2:2b

# Start Ollama (if not auto-started)
ollama serve

```

**Windows:**
- Download from: https://ollama.com/download/windows
- Install and run the installer
- Open terminal and run: `ollama pull gemma2:2b`

### Step 4: Install and Start Qdrant

**Option 1: Using Docker (Recommended)**
```bash
docker pull qdrant/qdrant

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option 2: Using Docker Compose (Partial)**
```bash
# Start only Qdrant service
docker-compose up -d qdrant
```


### Step 5: Set Up Environment Variables

> **⚠️ Important Note**: The `.env` file with actual credentials is included in this repository **for judgment purposes only**. These environment variables will be removed after the assessment is complete. In production, never commit sensitive credentials to version control.

```

### Step 6: Initialize Database

```bash
# Make sure virtual environment is activated
python -m src.init_db

# The database migration is already completed, and the .env file has been provided for this assessment. You don’t need to run this command.
```

### Step 7: Start FastAPI Server

```bash
# Development mode with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Step 8: Verify Installation

- **FastAPI Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Health Check**: http://localhost:8000/health

Test the system:
```bash
# Upload a paper
curl -X POST "http://localhost:8000/api/papers/upload" \
  -F "file=@sample_papers/your_paper.pdf"

# Query the system
curl -X POST "http://localhost:8000/api/papers/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main contribution of this paper?"}'
```
