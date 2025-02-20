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


async def process_jsonl_dump(
    filepath: str,
    datastore: DataStore,
    custom_metadata: dict,
    screen_for_pii: bool,
    extract_metadata: bool,
):
    # open the jsonl file as a generator of dictionaries
    with open(filepath) as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    documents = []
    skipped_items = []
    # iterate over the data and create document objects
    for item in data:
        if len(documents) % 20 == 0:
            print(f"Processed {len(documents)} documents")

        try:
            # extract basic fields from the JSONL item
            doc_id = item.get("id", None)
            text = item.get("text", None)
            source = item.get("source", None)
            source_id = item.get("source_id", None)
            url = item.get("url", None)
            created_at = item.get("created_at", None)
            author = item.get("author", None)

            if not text:
                print("No document text, skipping...")
                continue

            # create the initial metadata object
            metadata = DocumentMetadata(
                source=source,
                source_id=source_id,
                url=url,
                created_at=created_at,
                author=author,
            )
            # merge in the custom metadata (this works even if the key wasn't defined)
            metadata = metadata.copy(update=custom_metadata)

            # screen for PII if requested
            if screen_for_pii:
                pii_detected = screen_text_for_pii(text)
                if pii_detected:
                    print("PII detected in document, skipping")
                    skipped_items.append(item)
                    continue

            # extract metadata if requested
            if extract_metadata:
                extracted_metadata = extract_metadata_from_document(
                    f"Text: {text}; Metadata: {str(metadata)}"
                )
                # update the metadata with the extracted values (allowing extra fields)
                metadata = DocumentMetadata(**extracted_metadata)

            # create the document object with id, text, and updated metadata
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
        print(f"Upserting batch of {len(batch_documents)} documents, batch {i}")
        await datastore.upsert(batch_documents)

    print(f"Skipped {len(skipped_items)} items due to errors or PII detection")
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
        type=bool,
        help="A boolean flag to indicate whether to try the PII detection function",
    )
    parser.add_argument(
        "--extract_metadata",
        default=False,
        type=bool,
        help="A boolean flag to indicate whether to try to extract metadata from the document",
    )
    args = parser.parse_args()

    filepath = args.filepath
    custom_metadata = json.loads(args.custom_metadata)
    screen_for_pii = args.screen_for_pii
    extract_metadata = args.extract_metadata

    datastore = await get_datastore()
    await process_jsonl_dump(
        filepath, datastore, custom_metadata, screen_for_pii, extract_metadata
    )


if __name__ == "__main__":
    asyncio.run(main())
