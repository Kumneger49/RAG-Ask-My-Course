import os
import pickle  # For loading metadata
import numpy as np  # For handling arrays
import faiss  # For the FAISS vector index
from sentence_transformers import SentenceTransformer  # For embedding the query
import logging  # For logging

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieve.log'),
        logging.StreamHandler()
    ]
)

def load_faiss_and_metadata():
    """
    Loads the FAISS index and metadata mapping from disk.
    Returns:
        index (faiss.Index): The loaded FAISS index.
        metadata (dict): Mapping from FAISS index to chunk metadata.
    """
    # Path to the FAISS index file
    faiss_index_path = os.path.join(os.path.dirname(__file__), 'faiss.index')
    # Path to the metadata mapping file
    metadata_path = os.path.join(os.path.dirname(__file__), 'faiss_metadata.pkl')

    logging.info(f"Loading FAISS index from {faiss_index_path}")
    index = faiss.read_index(faiss_index_path)  # Load the FAISS index
    logging.info("FAISS index loaded.")

    logging.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)  # Load the metadata mapping
    logging.info("Metadata loaded.")

    return index, metadata

def retrieve_top_k(query: str, k: int = 5, model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Embeds the user query, searches the FAISS index, and returns the top-k most similar chunks.
    Args:
        query (str): The user's question.
        k (int): Number of top results to return.
        model_name (str): Embedding model to use.
    Returns:
        List[dict]: List of top-k chunk metadata dicts (including text, filename, etc.).
    """
    # Load the FAISS index and metadata
    index, metadata = load_faiss_and_metadata()

    # Load the embedding model
    logging.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logging.info("Model loaded.")

    # Embed the query (returns a numpy array)
    logging.info(f"Embedding query: {query}")
    query_vec = model.encode([query], convert_to_numpy=True)  # Shape: (1, dim)

    # Search the FAISS index for the top-k most similar vectors
    logging.info(f"Searching FAISS index for top {k} results...")
    distances, indices = index.search(query_vec, k)  # Returns (distances, indices)

    # Retrieve the metadata for each result
    results = []
    for idx, dist in zip(indices[0], distances[0]):  # indices[0] and distances[0] are arrays of length k
        chunk_meta = metadata[idx]  # Get the chunk's metadata
        chunk_meta = chunk_meta.copy()  # Copy to avoid mutating the original
        chunk_meta['faiss_index'] = int(idx)  # Add the FAISS index
        chunk_meta['distance'] = float(dist)  # Add the distance (lower = more similar)
        results.append(chunk_meta)  # Add to results list
    logging.info(f"Top {k} results retrieved.")
    return results  # Return the list of top-k chunk metadata dicts

# Example usage (uncomment to test):
if __name__ == "__main__":
    query = "What is a Linear combination?"
    top_chunks = retrieve_top_k(query, k=5)
    for i, chunk in enumerate(top_chunks):
        print(f"Result {i+1} (distance={chunk['distance']:.4f}):\n{chunk['text']}\n---") 