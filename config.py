'''Configuration file for the RAG system.'''

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# --- Data Files ---
EXCEL_FILE_PATH = os.path.join(DATA_DIR, "fault_data.xlsx")

# --- ChromaDB Configuration ---
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "fault_descriptions_local_model" # Changed collection name to force recreation
#COLLECTION_NAME = "fault_descriptions_remote_model" # Changed collection name to force recreation

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "model", "acge_text_embedding")
#EMBEDDING_MODEL_PATH = "force_remote_embedding" # Set to a non-existent path to force remote embedding
REMOTE_EMBEDDING_MODEL_ID = "BAAI/bge-large-zh-v1.5" # Explicitly define remote model ID
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"

# --- LLM API Configuration ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# --- Excel Column Mappings (adjust these to match your Excel file) ---
# These are the column names in your Excel sheet that the program will use.
# Make sure they exactly match the header names in your Excel file.
EXCEL_COL_PRODUCT_MODEL = "产品型号"
EXCEL_COL_PRODUCT_NUMBER = "产品编号"
EXCEL_COL_FAULT_LOCATION = "终判故障部位"
EXCEL_COL_FAULT_MODE = "终判故障模式"
EXCEL_COL_FAULT_DESCRIPTION = "故障描述"
EXCEL_COL_SOLUTION = "处理方式及未解决问题"

# --- RAG System Configuration ---
TOP_K_RESULTS = 2  # Number of similar records to retrieve from ChromaDB

# --- Logging Configuration ---
LOG_FILE_PATH = os.path.join(BASE_DIR, "rag_system.log")
LOG_LEVEL = "INFO"
