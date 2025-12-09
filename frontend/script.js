document.addEventListener('DOMContentLoaded', () => {
    const faultDescriptionInput = document.getElementById('faultDescription');
    const productModelFilter = document.getElementById('productModelFilter');
    const submitQueryButton = document.getElementById('submitQuery');
    const diagnosisResultDiv = document.getElementById('diagnosisResult');
    const sourceDocumentsDiv = document.getElementById('sourceDocuments');
    const loadingSpinner = document.createElement('div');
    loadingSpinner.className = 'loading-spinner';
    document.querySelector('.container').appendChild(loadingSpinner);

    const API_BASE_URL = window.location.origin; // Assumes frontend and backend are on the same host:port

    // Function to fetch product models and populate the dropdown
    async function fetchProductModels() {
        try {
            const response = await fetch(`${API_BASE_URL}/product_models`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const models = await response.json();
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                productModelFilter.appendChild(option);
            });
        } catch (error) {
            console.error("Error fetching product models:", error);
        }
    }

    // Function to handle RAG query submission
    submitQueryButton.addEventListener('click', async () => {
        const query = faultDescriptionInput.value.trim();
        const productModel = productModelFilter.value;

        if (!query) {
            alert("请输入故障描述。");
            return;
        }

        diagnosisResultDiv.textContent = "";
        sourceDocumentsDiv.innerHTML = "";
        loadingSpinner.style.display = 'block'; // Show spinner

        try {
            const response = await fetch(`${API_BASE_URL}/rag_query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    product_model: productModel || null // Send null if "所有型号" is selected
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            diagnosisResultDiv.textContent = data.summary;

            if (data.source_documents && data.source_documents.length > 0) {
                data.source_documents.forEach(doc => {
                    const docDiv = document.createElement('div');
                    docDiv.className = 'source-document';
                    docDiv.innerHTML = `
                        <p><strong>产品型号:</strong> ${doc.product_model}</p>
                        <p><strong>故障描述:</strong> ${doc.fault_description}</p>
                        <p><strong>处理方式及未解决问题:</strong> ${doc.solution}</p>
                        <p><strong>相似度:</strong> ${doc.distance.toFixed(4)}</p>
                    `;
                    sourceDocumentsDiv.appendChild(docDiv);
                });
            } else {
                sourceDocumentsDiv.innerHTML = "<p>没有找到相关记录。</p>";
            }

        } catch (error) {
            console.error("Error submitting RAG query:", error);
            diagnosisResultDiv.textContent = `查询失败: ${error.message}`;
        } finally {
            loadingSpinner.style.display = 'none'; // Hide spinner
        }
    });

    // Initial fetch of product models when the page loads
    fetchProductModels();
});
