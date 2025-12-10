import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import os
import sys
import requests # Import requests for API calls
from typing import List # Import List for type hinting
import shutil # Import shutil for directory removal
from fastapi import HTTPException # Import HTTPException for error handling in remote_embed_call

# Add the parent directory to the Python path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='[%(asctime)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


def load_excel_data(file_path: str) -> pd.DataFrame:
    """Loads data from an Excel file into a Pandas DataFrame."""
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Excel file not found at {file_path}.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading Excel file {file_path}: {e}")
        sys.exit(1)


# --- Custom Embedding Wrapper Class (copied from main.py) ---
class CustomEmbeddingWrapper:
    """A wrapper class for SentenceTransformer models to provide a consistent encode interface."""
    def __init__(self, model_instance: SentenceTransformer):
        self.model = model_instance

    def encode(self, sentences: List[str], **kwargs) -> List[List[float]]:
        """Encodes a list of sentences into embeddings."""
        return self.model.encode(sentences, **kwargs).tolist()

def remote_embed_call(texts: List[str]) -> List[List[float]]:
    """Calls the remote embedding API to get embeddings for the given texts."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": config.REMOTE_EMBEDDING_MODEL_ID, # Use the explicitly defined remote model ID
        "input": texts
    }
    try:
        resp = requests.post(config.EMBEDDING_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        embeddings = [item['embedding'] for item in resp.json()['data']]
        return embeddings
    except requests.exceptions.RequestException as e:
        logging.error(f"Remote embedding API call failed: {e}")
        # In data_processor, we exit if remote call fails as it's critical for embedding
        sys.exit(1)


def initialize_embedding_model(model_path: str):
    """Initializes and returns the CustomEmbeddingWrapper (for local) or None if remote is to be used."""
    if os.path.isdir(model_path):
        try:
            model_instance = SentenceTransformer(model_path)
            logging.info(f"Local embedding model loaded successfully from {model_path}.")
            return CustomEmbeddingWrapper(model_instance) # Wrap the SentenceTransformer model
        except Exception as e:
            logging.warning(f"Failed to load local embedding model from {model_path}: {e}. Will attempt to use remote embedding API if configured.")
            return None
    else:
        logging.info(f"'{model_path}' is not a local directory. Will attempt to use remote embedding API.")
        return None


def initialize_chroma_client(persist_directory: str):
    """Initializes and returns the ChromaDB client."""
    try:
        # Ensure the persistence directory is clean if a fresh start is needed for dimension changes
        os.makedirs(persist_directory, exist_ok=True)

        client = chromadb.Client(chromadb.config.Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=persist_directory
        ))
        logging.info(f"Chroma client connected to {persist_directory} successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect Chroma client to {persist_directory}: {e}")
        sys.exit(1)


def process_and_store_data(
    df: pd.DataFrame,
    embedding_model,
    chroma_client,
    collection_name: str
):
    """Processes the DataFrame, generates embeddings, and stores data in ChromaDB."""
    required_columns = [
        config.EXCEL_COL_PRODUCT_MODEL,
        config.EXCEL_COL_PRODUCT_NUMBER,
        config.EXCEL_COL_FAULT_LOCATION,
        config.EXCEL_COL_FAULT_MODE,
        config.EXCEL_COL_FAULT_DESCRIPTION,
        config.EXCEL_COL_SOLUTION
    ]

    # Check if all required columns exist
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logging.error(f"Missing required Excel columns: {missing_cols}. Please check config.py and your Excel file.")
        sys.exit(1)

    documents = df[config.EXCEL_COL_FAULT_DESCRIPTION].astype(str).tolist()
    metadatas = df[required_columns].to_dict(orient='records')
    ids = [f"doc_{i}" for i in range(len(documents))]

    try:
        # Get or create the collection
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Determine the expected embedding dimension from the current model
        sample_text = ["test"]
        if embedding_model:
            expected_dimension = len(embedding_model.encode(sample_text)[0]) # No .tolist() needed for CustomEmbeddingWrapper
        elif config.EMBEDDING_API_URL and config.DEEPSEEK_API_KEY:
            expected_dimension = len(remote_embed_call(sample_text)[0])
        else:
            logging.error("No embedding model loaded and remote embedding API not configured. Cannot determine expected embedding dimension.")
            sys.exit(1)

        # Check if collection exists and has a different dimension. If so, delete and recreate.
        if collection.count() > 0:
            # Fetch a sample embedding to check its dimension
            # This is a bit indirect as ChromaDB doesn't expose collection dimension directly
            # A more robust way might be to store the dimension as part of collection metadata when created.
            # For now, we will rely on the error to trigger a full reset, or get an embedding to check.
            # Let's try to get a sample embedding if available and compare dimensions
            try:
                sample_embedding_from_db = collection.peek(1)['embeddings'][0]
                if len(sample_embedding_from_db) != expected_dimension:
                    logging.warning(f"ChromaDB collection '{collection_name}' has mismatched embedding dimension (expected {expected_dimension}, got {len(sample_embedding_from_db)}). Deleting and recreating collection.")
                    chroma_client.delete_collection(name=collection_name)
                    collection = chroma_client.get_or_create_collection(name=collection_name)
                else:
                    logging.info(f"ChromaDB collection '{collection_name}' dimensions match: {expected_dimension}.")
            except Exception as e:
                logging.warning(f"Could not verify ChromaDB collection dimension, possibly empty or error: {e}. Proceeding with potential recreation if needed.")
                # If peek fails (e.g., collection is truly empty), then it's fine, we proceed.

        # Clear existing data in collection before adding new data if it exists
        if collection.count() > 0:
            logging.warning(f"Collection '{collection_name}' already exists and contains {collection.count()} documents. Deleting existing documents.")
            existing_ids = collection.peek(collection.count())['ids']
            collection.delete(ids=existing_ids)

        logging.info("Generating embeddings...")
        if embedding_model:
            embeddings = embedding_model.encode(documents, show_progress_bar=True) # No .tolist() needed for CustomEmbeddingWrapper
            logging.info("Embeddings generated using local model.")
        else:
            # Use remote embedding API
            embeddings = remote_embed_call(documents)
            logging.info("Embeddings generated using remote API.")

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(documents)} documents to ChromaDB collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"Error processing and storing data in ChromaDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.info("Starting data preprocessing...")
    
    # Ensure data directory exists for the Excel file
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    df_data = load_excel_data(config.EXCEL_FILE_PATH)
    
    # embedding_model can be None if local model is not found/loaded
    embedding_model_instance = initialize_embedding_model(config.EMBEDDING_MODEL_PATH)
    chroma_client = initialize_chroma_client(config.CHROMA_DB_PATH)
    process_and_store_data(df_data, embedding_model_instance, chroma_client, config.COLLECTION_NAME)
    
    logging.info("Data preprocessing completed.")
