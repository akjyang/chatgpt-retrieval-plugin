from typing import Dict, List, Optional, Tuple
import uuid
from models.models import Document, DocumentChunk, DocumentChunkMetadata
import tiktoken
from services.openai import get_embeddings

# Global variables
tokenizer = tiktoken.get_encoding("cl100k_base")

# Constants
CHUNK_SIZE = 200                # Target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350      # Minimum size of each chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5   # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128     # Number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000          # Maximum number of chunks to generate from a text

def get_text_chunks(text: str, chunk_token_size: Optional[int]) -> List[str]:
    """
    Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.
    """
    if not text or text.isspace():
        return []
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    chunk_size = chunk_token_size or CHUNK_SIZE
    num_chunks = 0
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        chunk = tokens[:chunk_size]
        chunk_text = tokenizer.decode(chunk)
        if not chunk_text or chunk_text.isspace():
            tokens = tokens[len(chunk):]
            continue
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            chunk_text = chunk_text[: last_punctuation + 1]
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()
        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(chunk_text_to_append)
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())):]
        num_chunks += 1
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)
    return chunks

def create_document_chunks(doc: Document, chunk_token_size: Optional[int]) -> Tuple[List[DocumentChunk], str]:
    """
    Create document chunks from a Document. Generates a unique document id if missing,
    splits the text into chunks, and assigns each chunk a unique id by appending a suffix.
    """
    if not doc.text or doc.text.isspace():
        return [], doc.id or str(uuid.uuid4())
    doc_id = doc.id or str(uuid.uuid4())
    text_chunks = get_text_chunks(doc.text, chunk_token_size)
    # Create a metadata object for chunks and assign the document_id
    metadata = (
        DocumentChunkMetadata(**doc.metadata.__dict__)
        if doc.metadata is not None
        else DocumentChunkMetadata()
    )
    metadata.document_id = doc_id
    doc_chunks = []
    for i, text_chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{i}"
        doc_chunk = DocumentChunk(
            id=chunk_id,
            text=text_chunk,
            metadata=metadata,
        )
        doc_chunks.append(doc_chunk)
    return doc_chunks, doc_id

def get_document_chunks(documents: List[Document], chunk_token_size: Optional[int]) -> Dict[str, List[DocumentChunk]]:
    """
    Convert a list of Documents into a dictionary mapping each document id to a list of DocumentChunk objects.
    Also computes embeddings for each chunk.
    """
    chunks: Dict[str, List[DocumentChunk]] = {}
    all_chunks: List[DocumentChunk] = []
    for doc in documents:
        doc_chunks, doc_id = create_document_chunks(doc, chunk_token_size)
        all_chunks.extend(doc_chunks)
        chunks[doc_id] = doc_chunks
    if not all_chunks:
        return {}
    embeddings: List[List[float]] = []
    for i in range(0, len(all_chunks), EMBEDDINGS_BATCH_SIZE):
        batch_texts = [chunk.text for chunk in all_chunks[i : i + EMBEDDINGS_BATCH_SIZE]]
        batch_embeddings = get_embeddings(batch_texts)
        embeddings.extend(batch_embeddings)
    for i, chunk in enumerate(all_chunks):
        chunk.embedding = embeddings[i]
    return chunks
