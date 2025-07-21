# Chunking and Embedding for RAG Pipeline
# This script provides functions to chunk documents using RecursiveCharacterTextSplitter
# and embed them using the BAAI/bge-small-en-v1.5 model from sentence-transformers.

from typing import List, Dict  # For type hints
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For smart chunking
from sentence_transformers import SentenceTransformer  # For embeddings
import logging  # For logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format with timestamp, level, and message
)

# Function to chunk a list of documents
# Each document should be a dict with 'filename' and 'text' keys

def chunk_documents(documents: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Splits each document's text into chunks using RecursiveCharacterTextSplitter.
    Args:
        documents (List[Dict]): List of documents with 'filename' and 'text'.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        List[Dict]: List of chunk dicts with metadata.
    """
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Max characters per chunk
        chunk_overlap=chunk_overlap,  # Overlap between chunks
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]  # Try to split at paragraphs, then sentences, etc.
    )
    all_chunks = []  # List to hold all chunks
    chunk_id = 0  # Unique ID for each chunk
    for doc in documents:  # Loop through each document
        filename = doc.get('filename', 'unknown')  # Get the filename
        text = doc.get('text', '')  # Get the text
        # Use the splitter to split the text into chunks
        chunks = splitter.split_text(text) ## MAIN CHUNKING LINE 
        logging.info(f"Chunked {filename} into {len(chunks)} chunks.")  # Log chunking
        for i, chunk_text in enumerate(chunks):  # Loop through each chunk
            all_chunks.append({
                'chunk_id': chunk_id,  # Unique chunk ID
                'filename': filename,  # Source file
                'chunk_index': i,  # Index within the file
                'text': chunk_text  # The chunk's text
            })
            chunk_id += 1  # Increment the global chunk ID
    logging.info(f"Total chunks created: {len(all_chunks)}")  # Log total chunks
    return all_chunks  # Return the list of chunks

# Function to embed a list of chunks using BAAI/bge-small-en-v1.5

def embed_chunks(chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5") -> List[Dict]:
    """
    Embeds each chunk's text using the specified sentence-transformers model.
    Args:
        chunks (List[Dict]): List of chunk dicts with 'text'.
        model_name (str): Name of the embedding model.
    Returns:
        List[Dict]: List of chunk dicts with added 'embedding' key.
    """
    # Load the embedding model (downloads if not cached)
    logging.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logging.info("Model loaded successfully.")
    texts = [chunk['text'] for chunk in chunks]  # Extract all chunk texts
    logging.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)  # Get embeddings
    # Add embeddings to the chunk dicts
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]  # Add the embedding vector
    logging.info("All chunks embedded.")
    return chunks  # Return the list with embeddings 