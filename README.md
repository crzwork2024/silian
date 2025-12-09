# RAG Fault Diagnosis System

This project implements a Retrieval-Augmented Generation (RAG) system for fault diagnosis, leveraging a local vector database (ChromaDB) and a Large Language Model (LLM) API. The system allows users to input a fault description, optionally filter by product model, and receive a summarized diagnosis along with relevant historical fault records and their solutions.

## Technology Stack

*   **Python:** Backend logic, data processing, and API.
*   **FastAPI:** Web framework for building the backend API.
*   **HTML/CSS/JavaScript:** Frontend user interface.
*   **ChromaDB:** Local vector database for storing embedded fault descriptions.
*   **Sentence Transformers:** For generating embeddings of text data.
*   **DeepSeek API (or similar LLM API):** For generating summarized diagnoses.
*   **Pandas:** For reading and processing Excel data.

## Project Structure

```
.
├── backend/
│   ├── main.py             # FastAPI application, RAG endpoint, product model endpoint
│   ├── data_processor.py   # Script to process Excel data and populate ChromaDB
│   └── dummy_excel.py      # Script to generate a dummy Excel file (for testing/demo)
├── data/
│   ├── fault_data.xlsx     # (Expected) Your Excel data file
│   └── chroma_db/          # Directory for ChromaDB persistence
├── frontend/
│   ├── index.html          # Main frontend HTML file
│   ├── style.css           # Frontend styling
│   └── script.js           # Frontend JavaScript for interactivity
├── model/                  # Directory for local embedding models (if used locally)
├── config.py               # Centralized configuration for the application
├── requirements.txt        # Python dependencies
└── README.md               # Project README (this file)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd silian
    ```

2.  **Create a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**

    *   Place your Excel data file (e.g., `fault_data.xlsx`) into the `data/` directory. Ensure it contains the columns specified in `config.py` (`EXCEL_COL_PRODUCT_MODEL`, `EXCEL_COL_FAULT_DESCRIPTION`, `EXCEL_COL_SOLUTION`, etc.).
    *   If you don't have an Excel file, you can generate a dummy one:
        ```bash
        python backend/dummy_excel.py
        ```

5.  **Configure Environment Variables (`.env` file):**
    *   Create a file named `.env` in the project root directory.
    *   Add your DeepSeek API key to this file in the format:
        ```
        DEEPSEEK_API_KEY="YOUR_ACTUAL_DEEPSEEK_API_KEY"
        ```
    *   **Important:** Do not commit your `.env` file to version control. It should be ignored by Git.
    *   Open `config.py` and adjust the paths and `EXCEL_COL_*` mappings to match your Excel file's column headers. The `API_URL` and `MODEL_ID` are already configured.

6.  **Process and embed your data into ChromaDB:**
    ```bash
    python backend/data_processor.py
    ```
    This script will read your Excel file, generate embeddings for the fault descriptions, and store them in the `data/chroma_db` directory.

## Running the Application

1.  **Start the FastAPI backend:**
    Navigate to the project root and run:
    ```bash
    uvicorn backend.main:app --reload
    ```
    The `--reload` flag is useful for development as it restarts the server on code changes.

2.  **Access the Frontend:**
    Open your web browser and navigate to `http://127.0.0.1:8000` (or the address where Uvicorn is running).

## Usage

1.  Enter a fault description into the text area.
2.  (Optional) Select a product model from the dropdown to filter search results.
3.  Click "提交查询" (Submit Query) to get a diagnosis from the LLM and see relevant historical records.

## Testing Considerations

*   **Unit Tests:** Implement unit tests for individual functions in `data_processor.py` and `backend/main.py` (e.g., `load_excel_data`, `deepseek_call`).
*   **Integration Tests:** Test the interaction between components, such as the `rag_query` endpoint with ChromaDB and the LLM API.
*   **Frontend Tests:** Use tools like Playwright or Selenium to test frontend interactivity and display.
*   **Data Validation:** Ensure robust validation for Excel data input and API request payloads.
*   **Performance Testing:** Evaluate query response times, especially for large datasets or high concurrency.
*   **Similarity/Distance:** The `distance` value returned from ChromaDB indicates the dissimilarity between the query and the retrieved document. A lower `distance` value signifies higher similarity (e.g., a distance of 0 means a perfect match).

## Future Enhancements

*   Implement user authentication and authorization.
*   Enhance frontend UI/UX with more advanced filtering and display options.
*   Add more sophisticated error handling and user feedback.
*   Explore different embedding models and LLMs.
*   Implement a caching mechanism for frequently asked queries.
*   Containerize the application using Docker for easier deployment.
