import os
import pickle  # For loading/saving Python objects
import numpy as np  # For handling arrays
import faiss  # For the FAISS vector index
import logging  # For logging

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_faiss.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Path to the embedded chunks file
    embedded_chunks_path = os.path.join(os.path.dirname(__file__), '../../embedded_chunks.pkl')
    # Path to save the FAISS index
    faiss_index_path = os.path.join(os.path.dirname(__file__), 'faiss.index')
    # Path to save the metadata mapping
    metadata_path = os.path.join(os.path.dirname(__file__), 'faiss_metadata.pkl')

    logging.info(f"Loading embedded chunks from {embedded_chunks_path}")
    with open(embedded_chunks_path, 'rb') as f:
        embedded_chunks = pickle.load(f)
    logging.info(f"Loaded {len(embedded_chunks)} embedded chunks.")

    # Extract embeddings and build metadata mapping
    embeddings = []  # List to hold all embedding vectors
    metadata = {}    # Dict to map FAISS index to chunk metadata
    for i, chunk in enumerate(embedded_chunks):
        embeddings.append(chunk['embedding'])  # Add the embedding vector
        # Store metadata (everything except the embedding itself)
        meta = chunk.copy()
        del meta['embedding'] # delete the imbedding from the meta dict so that the metadata does not include the embedding but only the metadata about the embedidng
        metadata[i] = meta  # Map FAISS index to chunk metadata

    # Convert embeddings to a numpy array of type float32 (required by FAISS)
    embeddings_np = np.vstack(embeddings).astype('float32')
    logging.info(f"Embeddings shape: {embeddings_np.shape}")

    # Build a FAISS index (Flat L2 index)
    dim = embeddings_np.shape[1]  # Dimensionality of the embeddings
    index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance
    logging.info("Adding embeddings to FAISS index...")
    index.add(embeddings_np)  # Add all embeddings to the index
    logging.info(f"FAISS index contains {index.ntotal} vectors.")

    # Save the FAISS index to disk
    faiss.write_index(index, faiss_index_path)
    logging.info(f"FAISS index saved to {faiss_index_path}")

    # Save the metadata mapping to disk
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Metadata mapping saved to {metadata_path}")

    logging.info("FAISS index build complete.")

if __name__ == "__main__":
    main() 