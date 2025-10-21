## **RAG Overview**

The script will automatically scan a directory for PDF files, process and index their content into a vector database, and then use a Large Language Model (LLM) to answer questions by retrieving the most relevant information from those documents.

## **Features**

* **Multi-PDF Support:** Automatically ingests and indexes all PDF files from a specified data directory.  
* **Local First:** Utilizes Ollama to run powerful LLMs like Llama 3 locally, ensuring data privacy.  
* **Persistent Vector Store:** Uses ChromaDB to create a persistent vector index, so you only need to process your documents once.  

## **The RAG Pipeline**

The system follows a standard RAG pipeline:

1. **Load:** All PDF documents in the data/ directory are loaded.  
2. **Split:** The documents are broken down into smaller, manageable chunks.  
3. **Embed:** Each chunk is converted into a numerical vector representation using an embedding model (BAAI/bge-small-en-v1.5).  
4. **Store:** These embeddings are stored in a ChromaDB vector database.  
5. **Retrieve:** When a user asks a question, the system embeds the query and retrieves the most semantically similar chunks from the database.  
6. **Generate:** The original query and the retrieved context chunks are passed to the LLM (llama3:8b), which generates a final, context-aware answer.

## **Prerequisites**

Before you begin, ensure you have the following installed:

1. **Python 3.8+**  
2. **Ollama:** You must have the Ollama application installed and running.  
   * Follow the official installation guide at [ollama.com](https://ollama.com/).  
   * After installation, ensure the Ollama service is running in the background.

## **Setup & Installation**

1. Clone the Repository:  
   git clone https://github.com/Jansen52x/rags2riches.git

2. Populate data:
   You can place your own PDF files inside the data/ directory. If you leave it empty, a mock PDF will be created on the first run. 
 
3. Install Python Dependencies:
   pip install \-r requirements.txt

4. Pull the LLM Model:  
   Use Ollama to download the Llama 3 model.  
   ollama pull llama3:8b

## **Usage**

1. Start the Ollama Service:  
   Ensure the Ollama application is running. You can typically start it from your applications folder or by running ollama serve in your terminal.  
2. Run the Python Script:  
   Execute the main script from your terminal:  
   python rag\_system.py

### **First Run**

On the very first run, the script will:

1. Notice that the vector\_db directory does not exist.  
2. Create a mock PDF file in the data/ directory (if no other PDFs are present).  
3. Load, chunk, and embed the content of all PDFs in the data/ directory.  
4. Create and save the ChromaDB index to the vector\_db/ directory. This may take some time depending on the number of documents and your computer's hardware.  
5. Proceed to answer the sample questions.

### **Subsequent Runs**

On all subsequent runs, the script will:

1. Detect the existing vector\_db directory.  
2. Skip the indexing process entirely.  
3. Immediately load the existing index and begin answering questions, which is much faster.

**Note:** If you add, remove, or change the PDF files in the data/ directory, you must first **delete the vector\_db directory** and re-run the script to create a fresh index.

## **Configuration**

You can modify the following constants at the top of the rag\_system.py script to change its behavior:

* VECTOR\_DB\_PATH: The directory to save and load the vector store.  
* DATA\_DIR: The directory where your source PDF files are located.  
* CHUNK\_SIZE: The number of characters in each text chunk.  
* CHUNK\_OVERLAP: The number of characters of overlap between adjacent chunks.  
* EMBEDDING\_MODEL\_NAME: The Hugging Face model to use for embeddings.  
* LLM\_MODEL: The Ollama model to use for generation.