# Tennis Knowledge Database - Local Setup Guide

## Quick Start (5 minutes)

This guide will help you run the Tennis Knowledge Database API locally on your machine for testing and development.

## Prerequisites

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Google API Key** (recommended) or Ollama installation (advanced)

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone [your-repository-url]
cd tennis-knowledge-database
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root directory with the following content:

```bash
# .env file

# LLM Provider (recommended: gemini for simplicity)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
FORCE_GEMINI=true

# Gemini Settings
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7

# Server Settings
LOG_LEVEL=INFO
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8000

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_CHUNKS=3

# Storage Paths
KNOWLEDGE_BASE_DIR=./knowledge_base_docs
VECTOR_STORE_DIR=./vector_store
```

### Step 4: Get Google API Key (Recommended)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key
5. Replace `your_google_api_key_here` in your `.env` file with the actual key

### Step 5: Run the Application

```bash
python main.py
```

The application will start and show you:
- **Desktop GUI**: For testing and document management
- **Backend API**: Automatically starts on `http://localhost:8000`

## Testing the Local Setup

### Option 1: Using the Desktop GUI
1. Click "Start Backend" in the GUI
2. Wait for "Backend: Running ✅" status
3. Click "Load Tennis Documents" to add knowledge
4. Select a folder with tennis documents (.txt, .md files)
5. Wait for ingestion to complete
6. Type a tennis question like "Who won Wimbledon 2023?" and click Send

### Option 2: Using the API Directly

**Check if API is running:**
```bash
curl http://localhost:8000/
```

**Test available models:**
```bash
curl http://localhost:8000/api/models
```

**Test chat functionality:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is tennis?",
    "model_name": "gemini-1.5-flash",
    "top_k_chunks": 3
  }'
```

**Test document ingestion:**
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/path/to/your/tennis/documents"}'
```

### Option 3: Interactive API Documentation

Open your browser and go to:
```
http://localhost:8000/docs
```

This provides a full interactive interface to test all API endpoints.

## Adding Tennis Knowledge

The system works without tennis documents but provides generic responses. To add tennis-specific knowledge:

### Using the GUI (Easiest)
1. Create a folder with tennis documents (`.txt` or `.md` files)
2. Click "Load Tennis Documents" in the GUI
3. Select your folder containing tennis documents
4. Wait for ingestion to complete (you'll see "Processed X documents, created Y chunks")

### Using the API
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/path/to/your/tennis/documents"}'
```

## Alternative Setup: Using Ollama (Advanced)

If you prefer to use local models instead of Google's API:

### Step 1: Install Ollama
- Download from [ollama.ai](https://ollama.ai)
- Follow installation instructions for your operating system

### Step 2: Start Ollama and Download Models
```bash
# Start Ollama service
ollama serve

# In another terminal, download a model
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Step 3: Update .env File
```bash
# Change these lines in your .env file
LLM_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Remove or comment out Gemini settings
# GOOGLE_API_KEY=...
# FORCE_GEMINI=true
```

## Troubleshooting

### Common Issues

**"Import Error" when starting:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**"Google API key not configured":**
- Check your `.env` file exists in the project root
- Verify the API key is correct and has no extra spaces/quotes

**"No models found" or connection errors:**
- For Gemini: Check your internet connection and API key
- For Ollama: Ensure `ollama serve` is running in another terminal

**GUI doesn't start or crashes:**
- Make sure you have PySide6 installed: `pip install PySide6`
- Check the console for error messages

**Ingestion shows "failed" but actually worked:**
- This is a known GUI bug - if you see "Processed X documents, created Y chunks" in the logs, it actually worked
- The system will have tennis knowledge even if the GUI shows "failed"

**Backend won't start:**
- Check if port 8000 is already in use
- Look at `backend_stdout.log` and `backend_stderr.log` for error details

### Getting Help

If you encounter issues:

1. **Check the console logs** - they show detailed error messages
2. **Verify your `.env` file** - ensure it's in the correct location and format
3. **Test the API directly** - use `curl` commands to isolate GUI vs backend issues
4. **Check log files** - `backend_stdout.log` and `backend_stderr.log` contain backend errors

## Project Structure

```
tennis-knowledge-database/
├── .env                    # Your configuration (create this)
├── main.py                 # Start the desktop application
├── requirements.txt        # Python dependencies
├── Procfile               # For Railway deployment
├── backend/
│   ├── api_server.py      # FastAPI backend server
│   └── api_handlers.py    # API endpoint definitions
├── gui/
│   ├── main_window.py     # Desktop GUI interface
│   └── api_client.py      # GUI's API client
├── llm_interface/         # LLM client implementations
│   ├── gemini_client.py   # Google Gemini client
│   └── ollama_client.py   # Ollama local client
├── rag/                   # RAG pipeline components
├── knowledge_base_docs/   # Place tennis documents here
└── vector_store/          # Vector database files (auto-created)
```

## Production Deployment

The same codebase is already deployed and running at:
```
https://web-production-668aa.up.railway.app/api
```

Your local setup and the production deployment use identical code and configuration, ensuring consistent behavior across environments.

## API Endpoints

### Available Endpoints

- **GET /api/models** - List available LLM models
- **POST /api/chat** - Send tennis queries and get responses
- **POST /api/ingest** - Load tennis documents into the knowledge base
- **GET /** - Health check and welcome message
- **GET /docs** - Interactive API documentation

### Example Mobile App Integration

```javascript
// Example API calls for mobile app integration
const baseURL = 'http://localhost:8000/api'; // or production URL

// Get available models
const models = await fetch(`${baseURL}/models`).then(r => r.json());

// Send a tennis query
const response = await fetch(`${baseURL}/chat`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query_text: "Who won Wimbledon 2023?",
    model_name: "gemini-1.5-flash",
    top_k_chunks: 3
  })
}).then(r => r.json());

console.log(response.answer); // Tennis answer with sources
```

## Cost Considerations

**Using Gemini (Recommended):**
- Very low cost: ~$0.0003 per query
- Example: 1,000 queries = ~$0.30
- No additional infrastructure needed

**Using Ollama:**
- Free to use
- Requires local compute resources
- Slower response times on average hardware

## Next Steps

1. **Test basic functionality** with the steps above
2. **Add tennis documents** to improve response quality
3. **Integrate with your mobile application** using the API endpoints
4. **Scale to production** using the Railway deployment when ready

For mobile app integration, use the same API endpoints that work locally (`/api/chat`, `/api/models`, `/api/ingest`) but point to either your local server (`http://localhost:8000/api`) or the production deployment (`https://web-production-668aa.up.railway.app/api`).