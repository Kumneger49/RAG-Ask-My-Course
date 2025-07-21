import os
import logging
import pickle  # For saving Python objects to disk
from backend.utils.ingest import read_files_from_data_dir
from backend.core.chunk_and_embed import chunk_documents, embed_chunks

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    logging.info(f"Pipeline started. Reading from {data_dir}")

    # Step 1: Ingest documents
    documents = read_files_from_data_dir(data_dir)
    logging.info(f"Ingested {len(documents)} documents.")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
    logging.info(f"Created {len(chunks)} chunks.")

    # Save chunks to disk
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    logging.info("Chunks saved to chunks.pkl")

    # Step 3: Embed chunks
    embedded_chunks = embed_chunks(chunks)
    logging.info(f"Embedded {len(embedded_chunks)} chunks.")

    # Save embedded chunks to disk
    with open('embedded_chunks.pkl', 'wb') as f:
        pickle.dump(embedded_chunks, f)
    logging.info("Embedded chunks saved to embedded_chunks.pkl")

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main() 