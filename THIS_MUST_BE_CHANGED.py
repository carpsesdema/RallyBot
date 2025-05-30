# .env - Tennis Knowledge Database Configuration

# Ollama Settings (Local LLM)
#OLLAMA_API_URL=http://localhost:11434
#OLLAMA_CHAT_MODEL=deepseek-llm:7b
#OLLAMA_EMBEDDING_MODEL=nomic-embed-text
#OLLAMA_EMBEDDING_DIMENSION=768

# Google Gemini Settings (Cloud LLM)
GOOGLE_API_KEY=AlzaSyDp51Oh4cRrz77mHOPSlvU2qn03D4j7qN0

#GEMINI_MODEL=models/gemini-2.5-flash-preview-05-20
#GEMINI_TEMPERATURE=0.7

# LLM Provider (ollama or gemini)
#LLM_PROVIDER=ollama

# Application Settings
#LOG_LEVEL=INFO
#API_SERVER_HOST=127.0.0.1
#API_SERVER_PORT=8000

# Knowledge Base Paths
#KNOWLEDGE_BASE_DIR=./knowledge_base_docs
#VECTOR_STORE_DIR=./vector_store
#VECTOR_STORE_FILE_NAME=tennis_faiss.index
#VECTOR_STORE_METADATA_FILE_NAME=tennis_faiss_metadata.pkl

# RAG Settings
#CHUNK_SIZE=1000
#CHUNK_OVERLAP=200
#TOP_K_CHUNKS=3