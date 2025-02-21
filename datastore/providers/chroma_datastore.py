"""
Chroma datastore support for the ChatGPT retrieval plugin.

Consult the Chroma docs and GitHub repo for more information:
- https://docs.trychroma.com/usage-guide?lang=py
- https://github.com/chroma-core/chroma
- https://www.trychroma.com/
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
from services.openai import get_embeddings

import chromadb
from chromadb.config import Settings

from datastore.datastore import DataStore
from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    Source,
)
from services.chunks import get_document_chunks

CHROMA_IN_MEMORY = os.environ.get("CHROMA_IN_MEMORY", "True")
CHROMA_PERSISTENCE_DIR = os.environ.get("CHROMA_PERSISTENCE_DIR", "openai")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://127.0.0.1")
CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")
# Default collection is now only used for backward compatibility.
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "attributes")

def sanitize_id(id_str: str) -> str:
    """
    Sanitize an ID string by replacing any non-word characters with underscores.
    """
    return re.sub(r'\W+', '_', id_str).strip('_')

class ChromaDataStore(DataStore):
    def __init__(
        self,
        in_memory: bool = CHROMA_IN_MEMORY,  # type: ignore
        persistence_dir: Optional[str] = CHROMA_PERSISTENCE_DIR,
        collection_name: str = CHROMA_COLLECTION,
        host: str = CHROMA_HOST,
        port: str = CHROMA_PORT,
        client: Optional[chromadb.Client] = None,
    ):
        print("CHROMA IN MEMORY:", in_memory)
        print(type(in_memory)) # "FALSE"
        in_memory = False if in_memory.lower() == "false" else True
        if client:
            self._client = client
        else:
            if in_memory:
                print("Using in-memory ChromaDB")
                settings = (
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persistence_dir,
                    )
                    if persistence_dir
                    else Settings()
                )
                self._client = chromadb.Client(settings=settings)
            else:
                print("NOT Using in-memory ChromaDB")
                self._client = chromadb.Client(
                    settings=Settings(
                        chroma_api_impl="rest",
                        chroma_server_host=host,
                        chroma_server_http_port=port,
                    )
                )
        # For legacy operations we still keep a default collection.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        """
        Groups documents by their 'doc_type' (from metadata) and upserts each group
        into its respective collection. Returns a flat list of document chunk ids.
        """
        groups: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            raw_doc_type = doc.metadata.doc_type if doc.metadata and getattr(doc.metadata, "doc_type", None) else self._collection.name
            doc_type = sanitize_id(raw_doc_type)
            groups[doc_type].append(doc)

        all_ids = []
        for group_name, docs in groups.items():
            collection = self._client.get_or_create_collection(
                name=group_name,
                embedding_function=None,
            )
            chunks = get_document_chunks(docs, chunk_token_size)
            collection.upsert(
                ids=[chunk.id for chunk_list in chunks.values() for chunk in chunk_list],
                embeddings=[
                    chunk.embedding for chunk_list in chunks.values() for chunk in chunk_list
                ],
                documents=[
                    chunk.text for chunk_list in chunks.values() for chunk in chunk_list
                ],
                metadatas=[
                    self._process_metadata_for_storage(chunk.metadata)
                    for chunk_list in chunks.values() for chunk in chunk_list
                ],
            )
            all_ids.extend([chunk.id for chunk_list in chunks.values() for chunk in chunk_list])
        return all_ids

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        self._collection.upsert(
            ids=[chunk.id for chunk_list in chunks.values() for chunk in chunk_list],
            embeddings=[chunk.embedding for chunk_list in chunks.values() for chunk in chunk_list],
            documents=[chunk.text for chunk_list in chunks.values() for chunk in chunk_list],
            metadatas=[
                self._process_metadata_for_storage(chunk.metadata)
                for chunk_list in chunks.values() for chunk in chunk_list
            ],
        )
        return [chunk.id for chunk_list in chunks.values() for chunk in chunk_list]
    
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        # For simplicity, delegate to multi_query (which aggregates over all collections)
        return await self.multi_query(queries)

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        if delete_all:
            self._collection.delete()
            return True
        if ids and len(ids) > 0:
            if len(ids) > 1:
                where_clause = {"$or": [{"document_id": id_} for id_ in ids]}
            else:
                (id_,) = ids
                where_clause = {"document_id": id_}
            if filter:
                where_clause = {
                    "$and": [self._where_from_query_filter(filter), where_clause]
                }
        elif filter:
            where_clause = self._where_from_query_filter(filter)
        self._collection.delete(where=where_clause)
        return True

    def _where_from_query_filter(self, query_filter: DocumentMetadataFilter) -> Dict:
        conditions = []
        # Add the flat keys that aren't start_date, end_date, or source.
        for k, v in query_filter.dict().items():
            if v is not None and k not in ("start_date", "end_date", "source"):
                conditions.append({k: v})
        # Add the source if provided.
        if query_filter.source:
            conditions.append({"source": query_filter.source.value})
        # Handle date filters.
        if query_filter.start_date and query_filter.end_date:
            conditions.extend([
                {"created_at": {"$gte": int(datetime.fromisoformat(query_filter.start_date).timestamp())}},
                {"created_at": {"$lte": int(datetime.fromisoformat(query_filter.end_date).timestamp())}}
            ])
        elif query_filter.start_date:
            conditions.append({
                "created_at": {"$gte": int(datetime.fromisoformat(query_filter.start_date).timestamp())}
            })
        elif query_filter.end_date:
            conditions.append({
                "created_at": {"$lte": int(datetime.fromisoformat(query_filter.end_date).timestamp())}
            })

        # If only one condition exists, return it directly; if multiple, wrap in $and.
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return {}


    def _process_metadata_for_storage(self, metadata: Any) -> Dict:
        stored_metadata = {}
        if metadata.source:
            stored_metadata["source"] = metadata.source.value
        if metadata.source_id:
            stored_metadata["source_id"] = metadata.source_id
        if metadata.url:
            stored_metadata["url"] = metadata.url
        if metadata.created_at:
            stored_metadata["created_at"] = int(datetime.fromisoformat(metadata.created_at).timestamp())
        if metadata.author:
            stored_metadata["author"] = metadata.author
        if metadata.document_id:
            stored_metadata["document_id"] = metadata.document_id

        # Get all the fields from the model (including extras)
        full_meta = metadata.dict()
        # Define the base keys we already handled
        base_keys = {"source", "source_id", "url", "created_at", "author", "document_id"}
        # Merge any extra keys (or even the ones defined in the model) that have a non-null value.
        for key, value in full_meta.items():
            if key not in base_keys and value is not None:
                stored_metadata[key] = value

        return stored_metadata


    def _process_metadata_from_storage(self, metadata: Dict) -> Any:
        # Reconstruct metadata for general document chunks.
        from models.models import DocumentChunkMetadata
        return DocumentChunkMetadata(
            source=Source(metadata["source"]) if "source" in metadata else None,
            source_id=metadata.get("source_id", None),
            url=metadata.get("url", None),
            created_at=datetime.fromtimestamp(metadata["created_at"]).isoformat()
            if "created_at" in metadata
            else None,
            author=metadata.get("author", None),
            doc_type=metadata.get("doc_type", None),
            course_code=metadata.get("course_code", None),
            course_title=metadata.get("course_title", None),
            term=metadata.get("term", None),
            attribute=metadata.get("attribute", None),
        )
    
    def _chingu_process_metadata_from_storage(self, metadata: Dict) -> Any:
        from models.models import ChinguDocumentChunkMetadata
        
        return ChinguDocumentChunkMetadata(
            doc_type=metadata.get("doc_type", None),
            course_code=metadata.get("course_code", None),
            course_title=metadata.get("course_title", None),
            course_unit=metadata.get("course_unit", None),
            term=metadata.get("term", None),    
            attribute=metadata.get("attribute", None),
            
            program_url=metadata.get("program_url", None),
            academic_level=metadata.get("academic_level", None),
            school=metadata.get("school", None),
            format=metadata.get("format", None),
            major_minor=metadata.get("major_minor", None),
            degree=metadata.get("degree", None),
            
            requirements=metadata.get("requirements", []),

            subject_url=metadata.get("subject_url", None),
            course_code_no=metadata.get("course_code_no", None),
            instructor=metadata.get("instructor", None),
            
            source=Source(metadata["source"]) if "source" in metadata else None,
        )
    
    async def ping(self) -> bool:
        try:
            _ = self._collection.count()
            return True
        except Exception:
            return False

    async def stats(self) -> dict:
        try:
            collections = self._client.list_collections()
            total_count = 0
            stats_per_collection = {}
            for coll in collections:
                count = coll.count()
                stats_per_collection[coll.name] = count
                total_count += count
            return {
                "total_document_count": total_count,
                "collections": stats_per_collection
            }
        except Exception as e:
            return {"error": str(e)}
        
    async def list_collections(self) -> List[str]:
        try:
            collections = self._client.list_collections()
            return [collection.name for collection in collections]
        except Exception:
            return [self._collection.name]

    async def get_document(self, document_id: str) -> Optional[Document]:
        result = self._collection.query(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        if not result or not result.get("ids"):
            return None
        doc_id = result["ids"][0]
        text = result["documents"][0]
        metadata_dict = result["metadatas"][0]
        metadata = self._process_metadata_from_storage(metadata_dict)
        return Document(id=doc_id, text=text, metadata=metadata)

    # --- Revised multi-query method for querying across all collections ---
    async def multi_query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        For each query in queries, iterate over all collections in the client,
        perform a similarity query on each using a collection-specific n_results,
        tag results with the collection name, and then aggregate and sort them.
        """
        aggregated_results = []
        # Get all collections from the client.
        all_collections = self._client.list_collections()
        for query in queries:
            combined_results = []
            for coll in all_collections:
                coll_name = coll.name.lower()
                # Determine the k value based on the collection name and the query parameters.
                if coll_name == "programs":
                    coll_k = query.top_k_programs if query.top_k_programs is not None else (query.top_k or 5)
                elif coll_name == "courses":
                    coll_k = query.top_k_courses if query.top_k_courses is not None else (query.top_k or 5)
                elif coll_name == "attributes":
                    coll_k = query.top_k_attributes if query.top_k_attributes is not None else (query.top_k or 5)
                else:
                    coll_k = query.top_k or 5

                # Ensure we do not request more than the collection's count.
                n_results = min(coll_k, coll.count() or coll_k)
                result = coll.query(
                    query_embeddings=[query.embedding],
                    include=["documents", "distances", "metadatas"],
                    n_results=n_results,
                    where=(self._where_from_query_filter(query.filter) if query.filter else {}),
                )
                if result and result.get("ids"):
                    ids = result["ids"][0]
                    documents = result["documents"][0]
                    metadatas = result["metadatas"][0]
                    distances = result["distances"][0]
                    for id_, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
                        # Tag each result with its source collection.
                        meta["source_collection"] = coll.name
                        from models.models import ChinguDocumentChunkWithScore
                        processed_meta = self._chingu_process_metadata_from_storage(meta)
                        combined_results.append(
                            ChinguDocumentChunkWithScore(
                                id=id_,
                                text=doc_text,
                                metadata=processed_meta,
                                score=distance,
                            )
                        )
            # Sort combined_results by score (assuming lower is better).
            combined_results.sort(key=lambda x: x.score)
            from models.models import QueryResult
            aggregated_results.append(QueryResult(query=query.query, results=combined_results))
        return aggregated_results

# --- Revised helper retriever class ---
class MultiCollectionRetriever:
    def __init__(self, datastore: "ChromaDataStore", k: int = 5):
        """
        :param datastore: An instance of ChromaDataStore.
        :param k: Default number of documents to return per collection if no granular value is provided.
        """
        self.datastore = datastore
        self.k = k

    async def get_relevant_documents(
        self, 
        query_text: str, 
        query_filter: Optional[DocumentMetadataFilter] = None
    ) -> List[Any]:
        """
        Computes the query embedding using OpenAI's embedding process (which returns a 1536-dimensional vector),
        constructs a QueryWithEmbedding (which now may include optional collection-specific top_k values and filters),
        and then performs a multi-collection query.
        """
        try:
            # Compute the embedding using your real embedding function.
            query_embedding = get_embeddings([query_text])[0]
        except Exception as e:
            raise Exception("Error obtaining embeddings: " + str(e))
        
        from models.models import QueryWithEmbedding
        query_obj = QueryWithEmbedding(
            query=query_text,
            embedding=query_embedding,
            top_k=self.k,
            # Optionally, you could also set:
            # top_k_programs=3,
            # top_k_courses=10,
            # top_k_attributes=7,
            filter=query_filter  # Pass the filter here.
        )
        results = await self.datastore.multi_query([query_obj])
        return results[0].results if results else []