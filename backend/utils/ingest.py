import os  # For interacting with the file system
from typing import List, Dict  # For type hints
import logging  # For logging messages

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format with timestamp, level, and message
)

try:
    from PyPDF2 import PdfReader  # For reading PDF files
except ImportError:
    PdfReader = None  # Handle the case where PyPDF2 is not installed


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extracts all text from a PDF file.
    Args:
        filepath (str): Path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    if PdfReader is None:
        raise ImportError("PyPDF2 is not installed. Please install it to read PDF files.")
    text = ""  # Initialize an empty string to hold the extracted text
    try:
        reader = PdfReader(filepath)  # Create a PDF reader object
        for page in reader.pages:  # Iterate through each page in the PDF
            text += page.extract_text() or ""  # Extract text from the page and add to the string
        logging.info(f"Successfully extracted text from PDF: {filepath}")  # Log success
    except Exception as e:
        logging.error(f"Error reading PDF {filepath}: {e}")  # Log error if PDF can't be read
    return text  # Return the extracted text (may be empty if error)


def extract_text_from_txt(filepath: str) -> str:
    """
    Reads all text from a plain text or markdown file.
    Args:
        filepath (str): Path to the text or markdown file.
    Returns:
        str: The file's content as a string.
    """
    text = ""  # Initialize an empty string
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Open the file for reading
            text = f.read()  # Read the entire file content
        logging.info(f"Successfully read text file: {filepath}")  # Log success
    except Exception as e:
        logging.error(f"Error reading text file {filepath}: {e}")  # Log error if file can't be read
    return text  # Return the file content (may be empty if error)


def read_files_from_data_dir(data_dir: str) -> List[Dict]:
    """
    Reads all supported files from the data directory and extracts their text.
    Args:
        data_dir (str): Path to the data directory.
    Returns:
        List[Dict]: List of documents with filename, type, and text.
    """
    logging.info(f"Starting ingestion from directory: {data_dir}")  # Log start of ingestion
    documents = []  # Initialize an empty list to hold all documents
    for filename in os.listdir(data_dir):  # Loop through each file in the directory
        filepath = os.path.join(data_dir, filename)  # Get the full path to the file
        if not os.path.isfile(filepath):  # Skip if it's not a file (e.g., a subdirectory)
            logging.debug(f"Skipping non-file: {filepath}")  # Log skipped item
            continue  # Move to the next item
        if filename.lower().endswith('.pdf'):  # Check if the file is a PDF
            file_type = 'pdf'  # Set file type
            logging.info(f"Processing PDF file: {filename}")  # Log file type
            text = extract_text_from_pdf(filepath)  # Extract text from PDF
        elif filename.lower().endswith('.md'):  # Check if the file is a markdown file
            file_type = 'md'  # Set file type
            logging.info(f"Processing Markdown file: {filename}")  # Log file type
            text = extract_text_from_txt(filepath)  # Extract text from markdown
        elif filename.lower().endswith('.txt'):  # Check if the file is a text file
            file_type = 'txt'  # Set file type
            logging.info(f"Processing Text file: {filename}")  # Log file type
            text = extract_text_from_txt(filepath)  # Extract text from text file
        else:
            logging.warning(f"Skipping unsupported file type: {filename}")  # Log unsupported file
            continue  # Skip unsupported file types
        documents.append({  # Add a dictionary for each file to the list
            'filename': filename,  # Store the file's name
            'type': file_type,  # Store the file type
            'text': text  # Store the extracted text
        })
    logging.info(f"Ingestion complete. Total documents loaded: {len(documents)}")  # Log total count
    return documents  # Return the list of document dictionaries 