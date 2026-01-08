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

## 项目架构与配置说明（中文）

**项目概览：** 本项目实现了基于检索增强生成（RAG）的故障诊断系统。后端使用 FastAPI 提供查询接口，前端为一个简单的静态页面，向后端发送故障描述并展示 LLM 生成的诊断和检索到的历史记录。

**目录与关键文件说明：**
- **根目录**: 存放 `README.md`、`config.py`、`requirements.txt` 等项目级配置与说明。
- **`backend/`**: 后端代码。
    - `main.py`: FastAPI 应用，包含 RAG 查询接口和产品型号下拉接口。
    - `data_processor.py`: 读取 Excel、生成嵌入并将数据写入 ChromaDB 的脚本。
    - `dummy_excel.py`: 生成示例 Excel 文件以便测试。
- **`data/`**: 数据与 ChromaDB 持久化目录（默认 `data/chroma_db`）。
- **`frontend/`**: 前端静态页面（`index.html`、`script.js`、`style.css`）。
- **`model/`**: 本地嵌入模型（可选）。
- **`config.py`**: 中央配置文件，包含路径、API 地址、模型 ID 与 Excel 列名映射。

**`config.py` 主要配置项（要点）**
- `BASE_DIR`, `DATA_DIR`, `MODEL_DIR`, `BACKEND_DIR`, `FRONTEND_DIR`: 项目路径基准，通常不需修改。
- `EXCEL_FILE_PATH`: 默认指向 `data/mro例子数据.xlsx`，可替换为 `data/fault_data.xlsx`（或你的实际文件名）。确保文件存在且首行列名与下面映射一致。
- `CHROMA_DB_PATH`: ChromaDB 的持久化目录（默认 `data/chroma_db`）。
- `COLLECTION_NAME`: Chroma 中的集合名称，变更会导致程序在下次写入时创建新集合（可用于刷新索引）。
- `EMBEDDING_MODEL_PATH`: 优先使用的本地嵌入模型路径；如果想强制使用远端嵌入服务，可将其设置为一个无效路径（代码中有注释说明）。
- `REMOTE_EMBEDDING_MODEL_ID` 与 `EMBEDDING_API_URL`: 远端嵌入服务的模型 ID 与接口（当使用远端时生效）。
- `SILICONFLOW_API_KEY`: 从环境变量读取（通过 `.env` 文件加载）。请在项目根创建 `.env` 并设置该变量以调用 `siliconflow` 服务。
- `SILICONFLOW_API_URL`, `SILICONFLOW_MODEL_ID`: 使用 `siliconflow` 提供者时的 API 地址与模型 ID。
- `OLLAMA_API_URL`, `OLLAMA_MODEL_ID`: 本地 Ollama 服务的地址与模型 ID（如使用 Ollama 时配置）。
- `LLM_PROVIDER`: 指定使用的 LLM 提供者，选项示例：`"siliconflow"` 或 `"ollama"`。
- Excel 列映射（必须与 Excel 表头完全一致）：
    - `EXCEL_COL_PRODUCT_MODEL`：产品型号（示例值："产品型号"）
    - `EXCEL_COL_PRODUCT_NUMBER`：产品编号
    - `EXCEL_COL_FAULT_LOCATION`：终判故障部位
    - `EXCEL_COL_FAULT_MODE`：终判故障模式
    - `EXCEL_COL_FAULT_DESCRIPTION`：故障描述（用于生成嵌入与检索）
    - `EXCEL_COL_SOLUTION`：处理方式及未解决问题（展示给用户）
- `TOP_K_RESULTS`: 从向量库检索的相似记录数，默认 5，可根据需求调整。
- `LOG_FILE_PATH`, `LOG_LEVEL`: 日志文件与日志级别配置。

**快速上手（中文）**
1. 创建并激活虚拟环境：

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 在项目根创建 `.env`，至少包含：

```
SILICONFLOW_API_KEY="你的实际密钥"
```

4. （可选）生成示例 Excel：

```bash
python backend/dummy_excel.py
```

5. 处理并写入向量库（生成嵌入）：

```bash
python backend/data_processor.py
```

6. 启动后端：

```bash
uvicorn backend.main:app --reload
```

7. 打开浏览器访问 `http://127.0.0.1:8000` 查看前端页面。

**调试与注意事项**
- 如果你希望使用远端嵌入服务：将 `EMBEDDING_MODEL_PATH` 设置为不可用路径或直接修改使用远端的逻辑，并设置好 `REMOTE_EMBEDDING_MODEL_ID` 与 `EMBEDDING_API_URL`。
- 更改 `COLLECTION_NAME` 会创建新的集合（相当于刷新索引），如果只是想追加/更新记录请保持不变。
- Excel 列名必须精确匹配 `config.py` 中的映射，否则 `data_processor.py` 无法正确解析列数据。
- Chroma 返回的 `distance` 值表示相似度的反向度量，值越小表示越相似（0 表示完全匹配）。

如果你希望我继续：
- 我可以帮助检查 `backend/main.py` 的 API 文档化，或将 `README.md` 的中文部分翻译成更详细的使用示例并附带截图说明（如果需要）。

*   Containerize the application using Docker for easier deployment.
