from pydantic import BaseModel, field_validator
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
    program_url: Optional[str] = None
    academic_level: Optional[str] = None
    school: Optional[str] = None
    format: Optional[str] = None
    major_minor: Optional[str] = None
    degree: Optional[str] = None
    requirements: Optional[str] = None
    subject_url: Optional[str] = None
    course_code_no: Optional[int] = None
    instructor: Optional[str] = None
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
    doc_type: Optional[str] = None  # e.g., "course", "attribute", "program"
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    term: Optional[str] = None
    attribute: Optional[str] = None

    class Config:
        extra = "allow"

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
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    doc_type: Optional[str] = None
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    course_unit: Optional[str] = None
    term: Optional[str] = None
    attribute: Optional[str] = None
    program_url: Optional[str] = None
    academic_level: Optional[str] = None
    school: Optional[str] = None
    format: Optional[str] = None
    major_minor: Optional[str] = None
    degree: Optional[str] = None
    requirements: Optional[str] = None
    subject_url: Optional[str] = None
    course_code_no: Optional[int] = None
    instructor: Optional[str] = None

class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 5

class QueryWithEmbedding(Query):
    embedding: List[float]
    top_k_programs: Optional[int] = 3
    top_k_courses: Optional[int] = 10
    top_k_attributes: Optional[int] = 7

class QueryResult(BaseModel):
    query: str
    results: List[ChinguDocumentChunkWithScore]

class QueryInput(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 5
    top_k_programs: Optional[int] = 3
    top_k_courses: Optional[int] = 10
    top_k_attributes: Optional[int] = 7

    @field_validator('top_k', 'top_k_programs', 'top_k_courses', 'top_k_attributes', mode='before')
    @classmethod
    def replace_zero(cls, v, info):
        defaults = {
            'top_k': 5,
            'top_k_programs': 3,
            'top_k_courses': 10,
            'top_k_attributes': 7,
        }
        if v is None or (isinstance(v, int) and v == 0):
            return defaults[info.field_name]
        return v