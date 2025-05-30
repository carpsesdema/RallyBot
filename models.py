# models.py
import datetime
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator  # Changed from validator to field_validator


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    sources: Optional[List[Dict[str, Any]]] = None  # e.g., [{'source_file': 'doc1.txt', 'chunk_text_preview': '...'}]

    @field_validator('role')  # Updated decorator
    @classmethod  # Keep classmethod if validator logic doesn't need instance (self)
    def role_must_be_user_or_assistant(cls, v: str) -> str:
        if v not in ['user', 'assistant']:
            raise ValueError('role must be "user" or "assistant"')
        return v


class QueryRequest(BaseModel):
    query_text: str
    session_id: Optional[str] = None
    top_k_chunks: int = Field(default=3, ge=1, le=10)  # Added sensible validation
    model_name: Optional[str] = None  # Added model_name field


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks_details: Optional[List[Dict[str, Any]]] = None


class IngestDirectoryRequest(BaseModel):
    directory_path: str  # This will be validated for existence in the handler/pipeline


class IngestDirectoryResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int


class AvailableModelsResponse(BaseModel):
    models: List[str]


class ApiErrorDetail(BaseModel):
    code: Optional[str] = None
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail


# --- Models for RAG System ---

class DocumentModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)  # e.g., {'source_filename': str}


class ChunkModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str  # ID of the parent DocumentModel
    text_content: str  # The actual text chunk
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Can inherit/extend document metadata
    # Example of potential metadata: {'source_filename': 'doc1.txt', 'chunk_char_start_idx': 0, 'chunk_char_end_idx': 1000}


if __name__ == '__main__':
    # This block is for testing and won't run if pydantic is not in the test environment
    # but the class definitions above are still valid.
    print("--- Pydantic Models Test (requires pydantic to be installed) ---")

    print("\nChatMessage:")
    try:
        # Example instantiation (assuming pydantic is available)
        # user_msg = ChatMessage(role="user", content="Hello!")
        # print(f"User: {user_msg.model_dump_json(indent=2)}")
        # assistant_msg = ChatMessage(
        #     role="assistant",
        #     content="Hi there! Found in doc.txt",
        #     sources=[{"source_file": "doc.txt", "preview": "The quick brown fox..."}]
        # )
        # print(f"Assistant: {assistant_msg.model_dump_json(indent=2)}")
        # invalid_msg = ChatMessage(role="system", content="Error") # Should fail validation
        print("Example ChatMessage tests would run here.")
    except Exception as e:  # Catch generic Exception if Pydantic isn't there
        print(f"Error (expected for 'system' role or if Pydantic not found): {e}")

    # Add more example tests here if needed, wrapped in try-except for Pydantic presence
    print("\n--- End Pydantic Models Test ---")