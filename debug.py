#!/usr/bin/env python3
import os
import json
import logging
import argparse
import openai

# Configure logging to show INFO-level messages.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Step 1: Reset/Reinitialize the Embeddings File ---
def reset_chroma_embeddings(embeddings_file="/var/lib/chroma/chroma-embeddings.parquet"):
    """
    Delete the existing embeddings file if it exists.
    """
    if os.path.exists(embeddings_file):
        logging.info(f"Deleting existing embeddings file: {embeddings_file}")
        os.remove(embeddings_file)
    else:
        logging.info("No existing embeddings file found. Nothing to delete.")

# --- Step 2: Increase Logging in the Ingestion Process ---
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-3-small")
    data = response["data"]
    return data

def process_document_with_logging(document: dict):
    """
    Process a single document: generate an embedding and log details.
    """
    text = document.get("text")
    if not text:
        logging.error("Document missing 'text' field, skipping document: %s", document)
        return None

    # Generate embedding and log its dimension and first 5 values
    embedding = get_embedding(text)
    logging.info(
        f"Generated embedding for document id {document.get('id', 'N/A')}: "
        f"Dimension = {len(embedding)}, First 5 values = {embedding[:5]}"
    )
    
    # Add the embedding to the document (simulate further processing)
    document["embedding"] = embedding
    return document

# --- Step 3: Test with a Minimal Subset of Data ---
def process_minimal_subset(filepath: str, num_lines: int = 2):
    """
    Read and process only the first num_lines documents from the JSONL file.
    """
    minimal_data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            try:
                doc = json.loads(line)
                minimal_data.append(doc)
            except Exception as e:
                logging.error(f"Error parsing line {i}: {e}")
    
    processed_documents = []
    for doc in minimal_data:
        processed_doc = process_document_with_logging(doc)
        if processed_doc is not None:
            processed_documents.append(processed_doc)
    return processed_documents

# --- Main function to tie everything together ---
def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file with enhanced logging and optional reset.")
    parser.add_argument("--filepath", required=True, help="Path to the JSONL file to process")
    parser.add_argument("--reset", action="store_true", help="Delete existing embeddings file before processing")
    args = parser.parse_args()

    # Step 1: Reset the embeddings file if requested.
    if args.reset:
        reset_chroma_embeddings()

    logging.info(f"Processing minimal subset from file: {args.filepath}")
    # Step 3: Process a minimal subset (e.g., first 2 lines) for testing.
    processed_docs = process_minimal_subset(args.filepath, num_lines=2)
    logging.info("Processed minimal subset:")
    for doc in processed_docs:
        logging.info(doc)

if __name__ == "__main__":
    main()
