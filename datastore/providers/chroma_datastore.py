"""
Chroma datastore support for the ChatGPT retrieval plugin.

Consult the Chroma docs and GitHub repo for more information:
- https://docs.trychroma.com/usage-guide?lang=py
- https://github.com/chroma-core/chroma
- https://www.trychroma.com/
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import re

import chromadb

from datastore.datastore import DataStore
from models.models import (
    Document,
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    ChinguDocumentChunkWithScore,
    ChinguDocumentChunkMetadata,
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
        if client:
            self._client = client
        else:
            if in_memory:
                settings = (
                    chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persistence_dir,
                    )
                    if persistence_dir
                    else chromadb.config.Settings()
                )
                self._client = chromadb.Client(settings=settings)
            else:
                self._client = chromadb.Client(
                    settings=chromadb.config.Settings(
                        chroma_api_impl="rest",
                        chroma_server_host=host,
                        chroma_server_http_port=port,
                    )
                )
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
            # Use sanitized doc_type if available; otherwise use the default collection's name.
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

    def _where_from_query_filter(self, query_filter: DocumentMetadataFilter) -> Dict:
        output = {
            k: v
            for (k, v) in query_filter.dict().items()
            if v is not None and k not in ("start_date", "end_date", "source")
        }
        if query_filter.source:
            output["source"] = query_filter.source.value
        if query_filter.start_date and query_filter.end_date:
            output["$and"] = [
                {
                    "created_at": {
                        "$gte": int(datetime.fromisoformat(query_filter.start_date).timestamp())
                    }
                },
                {
                    "created_at": {
                        "$lte": int(datetime.fromisoformat(query_filter.end_date).timestamp())
                    }
                },
            ]
        elif query_filter.start_date:
            output["created_at"] = {
                "$gte": int(datetime.fromisoformat(query_filter.start_date).timestamp())
            }
        elif query_filter.end_date:
            output["created_at"] = {
                "$lte": int(datetime.fromisoformat(query_filter.end_date).timestamp())
            }
        return output

    def _process_metadata_for_storage(self, metadata: DocumentChunkMetadata) -> Dict:
        stored_metadata = {}
        if metadata.source:
            stored_metadata["source"] = metadata.source.value
        if metadata.source_id:
            stored_metadata["source_id"] = metadata.source_id
        if metadata.url:
            stored_metadata["url"] = metadata.url
        if metadata.created_at:
            stored_metadata["created_at"] = int(
                datetime.fromisoformat(metadata.created_at).timestamp()
            )
        if metadata.author:
            stored_metadata["author"] = metadata.author
        if metadata.document_id:
            stored_metadata["document_id"] = metadata.document_id
        return stored_metadata

    def _process_metadata_from_storage(self, metadata: Dict) -> DocumentChunkMetadata:
        return DocumentChunkMetadata(
            source=Source(metadata["source"]) if "source" in metadata else None,
            source_id=metadata.get("source_id", None),
            url=metadata.get("url", None),
            created_at=datetime.fromtimestamp(metadata["created_at"]).isoformat()
            if "created_at" in metadata
            else None,
            author=metadata.get("author", None),
            document_id=metadata.get("document_id", None),
        )
    
    def _chingu_process_metadata_from_storage(self, metadata: Dict) -> DocumentChunkMetadata:
        return ChinguDocumentChunkMetadata(
            doc_type=metadata.get("doc_type", None),
            course_code=metadata.get("course_code", None),
            course_title=metadata.get("course_title", None),
            course_unit=metadata.get("course_unit", None),
            term=metadata.get("term", None),    
            attribute=metadata.get("attribute", None),
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

    async def list_collections(self) -> List[str]:
        try:
            collections = self._client.list_collections()
            return [collection.name for collection in collections]
        except Exception:
            return [self._collection.name]

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        results = [
            self._collection.query(
                query_embeddings=[query.embedding],
                include=["documents", "distances", "metadatas"],
                n_results=min(query.top_k, self._collection.count()),
                where=(self._where_from_query_filter(query.filter) if query.filter else {}),
            )
            for query in queries
        ]
        output = []
        for query, result in zip(queries, results):
            inner_results = []
            (ids,) = result["ids"]
            (documents,) = result["documents"]
            (metadatas,) = result["metadatas"]
            (distances,) = result["distances"]
            for id_, text, metadata, distance in zip(ids, documents, metadatas, distances):
                inner_results.append(
                    ChinguDocumentChunkWithScore(
                        id=id_,
                        text=text,
                        metadata=self._chingu_process_metadata_from_storage(metadata),
                        score=distance,
                    )
                )
            output.append(QueryResult(query=query.query, results=inner_results))
        return output

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