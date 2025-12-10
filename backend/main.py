from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import logging
import os
import sys

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

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(config.FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# --- Remote Embedding API Call Function ---
def remote_embed_call(texts: List[str]) -> List[List[float]]:
    """Calls the remote embedding API to get embeddings for the given texts."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": config.EMBEDDING_MODEL_PATH, # The model name from config is used as the remote model ID
        "input": texts
    }
    try:
        resp = requests.post(config.EMBEDDING_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        embeddings = [item['embedding'] for item in resp.json()['data']]
        return embeddings
    except requests.exceptions.RequestException as e:
        logging.error(f"Remote embedding API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Remote Embedding API error: {e}")

# --- Embedding Model Initialization ---
def initialize_embedding_model(model_path: str):
    """Initializes and returns the SentenceTransformer embedding model from a local path
    or returns None if a local model is not available, indicating that a remote API should be used.
    Avoids direct Hugging Face downloads if a local path is not explicitly provided and found.
    """
    if os.path.isdir(model_path):
        try:
            model = SentenceTransformer(model_path)
            logging.info(f"Local embedding model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logging.warning(f"Failed to load local embedding model from {model_path}: {e}. Attempting to use remote embedding API if configured.")
            return None
    else:
        logging.info(f"'{model_path}' is not a local directory. Will attempt to use remote embedding API.")
        return None

embedding_model = initialize_embedding_model(config.EMBEDDING_MODEL_PATH)

try:
    chroma_client = chromadb.Client(chromadb.config.Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=config.CHROMA_DB_PATH
    ))
    logging.info(f"Chroma client connected to {config.CHROMA_DB_PATH} successfully.")
except Exception as e:
    logging.error(f"Failed to connect Chroma client: {e}")
    sys.exit(1)

# --- ChromaDB Collection Initialization and Dimension Check ---
try:
    # Attempt to get the collection, it should have been created by data_processor.py
    rag_collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
    logging.info(f"ChromaDB collection '{config.COLLECTION_NAME}' loaded successfully.")

    # Determine the expected embedding dimension from the current model
    sample_text = ["test"]
    expected_dimension: int
    if embedding_model:
        expected_dimension = len(embedding_model.encode(sample_text).tolist()[0])
    elif config.EMBEDDING_API_URL and config.DEEPSEEK_API_KEY:
        expected_dimension = len(remote_embed_call(sample_text)[0])
    else:
        logging.error("No embedding model loaded and remote embedding API not configured. Cannot determine expected embedding dimension for backend. Exiting.")
        sys.exit(1)

    if rag_collection.count() > 0:
        try:
            sample_embedding_from_db = rag_collection.peek(1)['embeddings'][0]
            if len(sample_embedding_from_db) != expected_dimension:
                logging.error(f"ChromaDB collection '{config.COLLECTION_NAME}' has mismatched embedding dimension (expected {expected_dimension}, got {len(sample_embedding_from_db)}). Please rerun data_processor.py to re-initialize the database.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Could not verify ChromaDB collection dimension (error: {e}). Please check data_processor.py and rerun if needed. Exiting.")
            sys.exit(1)

except Exception as e:
    logging.error(f"Failed to load ChromaDB collection '{config.COLLECTION_NAME}': {e}. Please run data_processor.py first.")
    sys.exit(1)

# --- LLM API Call Function ---
def deepseek_call(prompt: str) -> str:
    """Calls the DeepSeek API with the given prompt."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": config.MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048 # Reduced max_tokens to a common value to avoid potential API limits
    }
    try:
        resp = requests.post(config.API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"DeepSeek API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM API error: {e}")

# --- FastAPI Models ---
class RAGQuery(BaseModel):
    query: str
    product_model: Optional[str] = None
    show_thought_process: bool = False # New field to request thought process

class RAGResponse(BaseModel):
    summary: str
    source_documents: List[Dict[str, Any]]

# --- FastAPI Endpoints ---
@app.post("/rag_query", response_model=RAGResponse)
async def rag_query(query_data: RAGQuery, request: Request):
    
    logging.info(f"Received RAG query: {query_data.query}, Product Model: {query_data.product_model}, User requests to show thought process: {query_data.show_thought_process}")

    # 1. Embed the user's query
    query_embedding: List[float]
    if embedding_model:
        query_embedding = embedding_model.encode(query_data.query).tolist()
        logging.info("Query embedded using local/downloaded model.")
    elif config.EMBEDDING_API_URL and config.DEEPSEEK_API_KEY:
        query_embedding = remote_embed_call([query_data.query])[0]
        logging.info("Query embedded using remote API.")
    else:
        logging.error("No embedding model loaded and remote embedding API not configured. Cannot process query.")
        raise HTTPException(status_code=500, detail="Embedding service not available.")

    # 2. Prepare the ChromaDB query filter
    where_clause = {}
    if query_data.product_model:
        where_clause[config.EXCEL_COL_PRODUCT_MODEL] = query_data.product_model
        logging.info(f"Filtering RAG query by product model: {query_data.product_model}")

    # 3. Query ChromaDB for relevant documents
    try:
        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=config.TOP_K_RESULTS,
            where=where_clause if where_clause else None,
            include=['documents', 'metadatas', 'distances']
        )
        logging.info(f"ChromaDB query returned {len(results['documents'][0]) if results['documents'] else 0} results.")
    except Exception as e:
        logging.error(f"ChromaDB query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector database query error: {e}")

    if not results or not results['documents'] or not results['documents'][0]:
        logging.warning("No relevant documents found in ChromaDB.")
        return RAGResponse(summary="No relevant information found.", source_documents=[])

    # 4. Construct the prompt for the LLM
    retrieved_info = []
    source_documents = []
    for i in range(len(results['documents'][0])):
        document = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        solution = metadata.get(config.EXCEL_COL_SOLUTION, "N/A")
        fault_desc = metadata.get(config.EXCEL_COL_FAULT_DESCRIPTION, "N/A")
        product_model = metadata.get(config.EXCEL_COL_PRODUCT_MODEL, "N/A")
        
        retrieved_info.append(
            f"Fault Description: {fault_desc}\n"\
            f"Suggested Solution: {solution}"
        )
        source_documents.append({
            "product_model": product_model,
            "fault_description": fault_desc,
            "solution": solution,
            "distance": results['distances'][0][i]
        })

    context_str = "\n\n".join(retrieved_info)
    prompt = (
        f"你是一位专业的故障诊断和解决方案助手。用户报告了以下故障：'{query_data.query}'。\n\n"
        f"你已获得以下历史故障记录及其处理方式：\n\n"
        f"{context_str}\n\n"
        f"请严格根据以上提供的历史记录，整理总结可能的故障处理方式和相关见解，并以中文回答。\n"
        f"如果提供的记录与用户的问题不相关，请明确指出并不要基于不相关的信息进行回答。不要编造记录中没有的信息。"
    )
    
    logging.info("Sending prompt to LLM...")
    llm_raw_response = deepseek_call(prompt)
    logging.info("Received raw response from LLM.")

    final_summary = llm_raw_response
    
    if not query_data.show_thought_process: # Corrected logic: if False, do NOT show thought process
        # If show_thought_process is False, extract content after </think>
        logging.info("Extracting final summary from LLM response (hiding thought process)...")
        think_tag = "</think>"
        if think_tag in llm_raw_response:
            final_summary = llm_raw_response.split(think_tag, 1)[1].strip()
        else:
            logging.warning("LLM response did not contain '</think>' tag, returning full response.")
    else:
        logging.info("Returning full LLM response including thought process.")
            
    logging.info(f"Final summary to be sent to frontend: {final_summary[:500]}...") # Log truncated final summary
    return RAGResponse(summary=final_summary, source_documents=source_documents)


@app.get("/product_models", response_model=List[str])
async def get_product_models():
    logging.info("Fetching unique product models.")
    try:
        # Get all metadatas (or a sample if the dataset is huge, but for now, all)
        # Note: ChromaDB doesn't have a direct 'get all unique metadata values' API efficiently.
        # We'll retrieve a large number of items and extract unique models.
        # For very large databases, this might need optimization (e.g., maintain a separate list).
        
        # A more robust way would be to query documents with only metadata included and a large n_results
        all_metadatas = []
        # Use peek to get some initial documents to check for size
        current_count = rag_collection.count()
        if current_count > 0:
            # Fetch all documents to extract metadata. For very large dbs, might need pagination.
            all_docs = rag_collection.get(ids=rag_collection.peek(current_count)['ids'], include=['metadatas'])
            all_metadatas = all_docs['metadatas']
        
        unique_models = sorted(list(set(md.get(config.EXCEL_COL_PRODUCT_MODEL) for md in all_metadatas if config.EXCEL_COL_PRODUCT_MODEL in md)))
        logging.info(f"Found {len(unique_models)} unique product models.")
        return unique_models
    except Exception as e:
        logging.error(f"Failed to retrieve unique product models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve product models: {e}")
