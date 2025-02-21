from models.models import (
    Document,
    DocumentMetadataFilter,
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List, Optional
from models.api import (
    UpsertRequest,
    UpsertResponse,
    QueryRequest,
    QueryResponse,
    DeleteRequest,
    DeleteResponse,
)
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import os
import uvicorn

# Security setup
bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

app = FastAPI(dependencies=[Depends(validate_token)])
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
    dependencies=[Depends(validate_token)],
)
app.mount("/sub", sub_app)

# New models for enhanced endpoints
class HealthResponse(BaseModel):
    status: str

class StatsResponse(BaseModel):
    document_count: int
    additional_stats: Optional[dict] = None

class CollectionsResponse(BaseModel):
    collections: List[str]

class DocumentResponse(BaseModel):
    document: Document

# --- Initialize datastore and MultiCollectionRetriever ---
from datastore.factory import get_datastore
from datastore.providers.chroma_datastore import MultiCollectionRetriever

datastore = None
multi_retriever = None

@app.on_event("startup")
async def startup():
    global datastore, multi_retriever
    datastore = await get_datastore()
    # Initialize the unified retriever to return 5 documents per collection.
    multi_retriever = MultiCollectionRetriever(datastore, k=5)

# --- Endpoints ---

@app.post("/upsert", response_model=UpsertResponse)
async def upsert(request: UpsertRequest = Body(...)):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest = Body(...)):
    try:
        if not request.queries or len(request.queries) == 0:
            raise HTTPException(status_code=400, detail="No query provided")
        # For simplicity, use the first query's text.
        query_text = request.queries[0].query
        results = await multi_retriever.get_relevant_documents(query_text)
        from models.models import QueryResult
        query_result = QueryResult(query=query_text, results=results)
        return QueryResponse(results=[query_result])
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@sub_app.post(
    "/query",
    response_model=QueryResponse,
    description="Queries across all collections and aggregates results (k=5 per collection).",
)
async def query_sub(request: QueryRequest = Body(...)):
    try:
        if not request.queries or len(request.queries) == 0:
            raise HTTPException(status_code=400, detail="No query provided")
        query_text = request.queries[0].query
        results = await multi_retriever.get_relevant_documents(query_text)
        from models.models import QueryResult
        query_result = QueryResult(query=query_text, results=results)
        return QueryResponse(results=[query_result])
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.delete("/delete", response_model=DeleteResponse)
async def delete(request: DeleteRequest = Body(...)):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        return HealthResponse(status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    try:
        document = await datastore.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentResponse(document=document)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update", response_model=UpsertResponse)
async def update_document(request: UpsertRequest = Body(...)):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def stats():
    try:
        s = await datastore.stats()
        return StatsResponse(document_count=s.get("total_document_count", 0), additional_stats=s)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    try:
        cols = await datastore.list_collections()
        return CollectionsResponse(collections=cols)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

def start():
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=True)