from models.models import (
    Document,
    DocumentMetadataFilter,
    QueryInput,
    QueryResult,
)
from pydantic import BaseModel
from typing import List, Optional

# Existing models

class UpsertRequest(BaseModel):
    documents: List[Document]

class UpsertResponse(BaseModel):
    ids: List[str]

class QueryRequest(BaseModel):
    queries: List[QueryInput]

class QueryResponse(BaseModel):
    results: List[QueryResult]

class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[DocumentMetadataFilter] = None
    delete_all: Optional[bool] = False

class DeleteResponse(BaseModel):
    success: bool

# New models for enhanced endpoints

class HealthResponse(BaseModel):
    status: str

class StatsResponse(BaseModel):
    # This assumes your datastore.stats() returns at least a document_count.
    document_count: int
    # Additional stats can be included as needed.
    additional_stats: Optional[dict] = None

class CollectionsResponse(BaseModel):
    # A list of collection names (if your datastore supports multiple collections)
    collections: List[str]

class DocumentResponse(BaseModel):
    # Wrapper for returning a single document
    document: Document