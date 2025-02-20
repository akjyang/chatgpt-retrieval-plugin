from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"
    programs = "programs"
    courses = "courses"
    attributes_flat = "attributes_flat"
    attributes_grouped = "attributes_grouped"
    courses_term = "courses_term"

class ChinguDocument(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None

class ChinguDocumentChunkMetadata(BaseModel):
    doc_type: Optional[str] = None  # e.g., "course", "attribute", "program"
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    course_unit: Optional[str] = None
    term: Optional[str] = None
    attribute: Optional[str] = None
    source: Optional[str] = None


class ChinguDocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: ChinguDocumentChunkMetadata
    embedding: Optional[List[float]] = None
    
class ChinguDocumentChunkWithScore(ChinguDocumentChunk):
    score: float

class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    # New optional fields for additional data types (e.g., courses, programs, attributes)
    doc_type: Optional[str] = None  # e.g., "course", "attribute", "program"
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    term: Optional[str] = None
    attribute: Optional[str] = None

    class Config:
        extra = "allow"  # allow arbitrary additional keys in metadata


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    source: Optional[Source] = None
    source_id: Optional[str] = None
    author: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None    # any date string format
    # Optionally add filtering fields for new metadata keys:
    doc_type: Optional[str] = None
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    term: Optional[str] = None
    attribute: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]