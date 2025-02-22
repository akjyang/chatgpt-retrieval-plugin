#!/usr/bin/env python3
import uuid
import json
import argparse
import asyncio

from models.models import Document, DocumentMetadata
from datastore.datastore import DataStore
from datastore.factory import get_datastore

from services.extract_metadata import extract_metadata_from_document
from services.pii_detection import screen_text_for_pii

DOCUMENT_UPSERT_BATCH_SIZE = 50

def serialize_metadata(meta):
    simple_meta = {}
    for key, value in meta.items():
        if not isinstance(value, (str, int, float)) and value is not None:
            simple_meta[key] = json.dumps(value)
        else:
            simple_meta[key] = value
    return simple_meta

async def process_jsonl_dump(
    filepath: str,
    datastore: DataStore,
    custom_metadata: dict,
    screen_for_pii: bool,
    extract_metadata: bool,
):
    # Print the parameters received for debugging
    print("Starting process_jsonl_dump with:")
    print("  filepath:", filepath)
    print("  custom_metadata:", custom_metadata)
    print("  screen_for_pii:", screen_for_pii)
    print("  extract_metadata:", extract_metadata)
    
    # Open the jsonl file as a generator of dictionaries
    with open(filepath, encoding="utf-8") as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    documents = []
    skipped_items = []
    # Iterate over the data and create document objects
    for item in data:
        if len(documents) % 20 == 0:
            print(f"Processed {len(documents)} documents so far.")

        try:
            # Extract basic fields from the JSONL item
            doc_id = item.get("id")
            # If no document ID exists, generate a new unique ID
            if not doc_id:
                doc_id = str(uuid.uuid4())
                print(f"No ID found in document; generated new ID: {doc_id}")
                
            text = item.get("text", None)

            if not text:
                print("No document text, skipping...")
                continue

            print("Item", item)
            # Build metadata dictionary from top-level keys
            metadata_dict = {
                "source": item.get("source", None),
                "source_id": item.get("source_id", None),
                "url": item.get("url", None),
                "created_at": item.get("created_at", None),
                "author": item.get("author", None),
            }

            # Check if there is a nested "metadata" field and merge it
            if "metadata" in item and isinstance(item["metadata"], dict):
                metadata_dict.update(item["metadata"])

            print("Metadata dict", metadata_dict)
            # Merge the parsed metadata and custom metadata into one dict:
            combined_metadata = {**metadata_dict, **custom_metadata}
            combined_metadata = serialize_metadata(combined_metadata)
            # Create the metadata instance once with the merged dictionary:
            metadata = DocumentMetadata(**combined_metadata)         
            screen_for_pii=False
            if screen_for_pii:
                pii_detected = screen_text_for_pii(text)
                if pii_detected:
                    print("PII detected in document, skipping")
                    skipped_items.append(item)
                    continue

            extract_metadata=False
            if extract_metadata:
                extracted_metadata = extract_metadata_from_document(
                    f"Text: {text}; Metadata: {str(metadata)}"
                )
                # Update the metadata with the extracted values (allowing extra fields)
                metadata = DocumentMetadata(**extracted_metadata)

            print("Metadata", metadata)
            # Create the document object with id, text, and updated metadata
            document = Document(
                id=doc_id,
                text=text,
                metadata=metadata,
            )
            documents.append(document)
        except Exception as e:
            print(f"Error processing {item}: {e}")
            skipped_items.append(item)

    # Upsert documents in batches
    for i in range(0, len(documents), DOCUMENT_UPSERT_BATCH_SIZE):
        batch_documents = documents[i : i + DOCUMENT_UPSERT_BATCH_SIZE]
        print(f"Upserting batch of {len(batch_documents)} documents (batch starting at index {i}).")
        ids = await datastore.upsert(batch_documents)
        print("Uploaded documents ids:", ids)

    print(f"Skipped {len(skipped_items)} items due to errors or PII detection:")
    for item in skipped_items:
        print(item)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True, help="The path to the jsonl dump")
    parser.add_argument(
        "--custom_metadata",
        default="{}",
        help="A JSON string of key-value pairs to update the metadata of the documents",
    )
    parser.add_argument(
        "--screen_for_pii",
        default=False,
        type=lambda x: x.lower() in ("true", "1", "yes", "y"),
        help="A boolean flag to indicate whether to try the PII detection function (True/False)",
    )
    parser.add_argument(
        "--extract_metadata",
        default=False,
        type=lambda x: x.lower() in ("true", "1", "yes", "y"),
        help="A boolean flag to indicate whether to try to extract metadata from the document (True/False)",
    )
    args = parser.parse_args()

    filepath = args.filepath
    try:
        custom_metadata = json.loads(args.custom_metadata)
    except Exception as e:
        print(f"Error parsing custom_metadata: {e}")
        custom_metadata = {}
    screen_for_pii = args.screen_for_pii
    extract_metadata = args.extract_metadata

    print("Arguments received in main():")
    print("  filepath:", filepath)
    print("  custom_metadata:", custom_metadata)
    print("  screen_for_pii:", screen_for_pii)
    print("  extract_metadata:", extract_metadata)

    datastore = await get_datastore()
    print("Datastore instance:", datastore)

    await process_jsonl_dump(
        filepath, datastore, custom_metadata, screen_for_pii, extract_metadata
    )

if __name__ == "__main__":
    asyncio.run(main())