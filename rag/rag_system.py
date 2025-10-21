import os
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


VECTOR_DB_PATH = "vector_db"
DATA_DIR = "data"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3:8b"


def create_mock_pdf():
    """
    Creates a mock PDF file if it doesn't exist. This ensures the script
    is runnable out-of-the-box.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    mock_pdf_path = os.path.join(DATA_DIR, "report.pdf")

    if not os.path.exists(mock_pdf_path):
        print(f"Creating mock PDF: {mock_pdf_path}")
        c = canvas.Canvas(mock_pdf_path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "Project Griffin: Q3 Financial & Operations Report")
        c.setFont("Helvetica", 11)
        text_intro = [
            "This document outlines the financial performance and operational highlights of Project Griffin",
            "for the third quarter. The project continues to show strong growth and innovation.",
            "Revenue saw a significant increase of 15% to $5.2 million.",
            "This growth was driven by the Phoenix-1 satellite constellation.",
        ]
        y_position = height - 108
        for line in text_intro:
            c.drawString(72, y_position, line)
            y_position -= 15
        c.save()

# TODO: Change this to handle multiple pdfs
def create_index_from_directory(directory_path: str, db_path: str = VECTOR_DB_PATH):
    """
    Creates a vector index from all PDF files found in a directory.
    """
    print(f"Starting to create index from all PDFs in '{directory_path}'...")
    all_docs = []

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the specified directory. Aborting.")
        return

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        print(f"Processing file: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_docs.extend(documents)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not all_docs:
        print("Could not load any documents from the PDF files. Aborting index creation.")
        return
    print(f"Loaded a total of {len(all_docs)} pages from {len(pdf_files)} PDF files.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(all_docs)
    if not chunks:
        print("Could not split the documents into chunks.")
        return
    print(f"Split documents into {len(chunks)} chunks.")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print(f"Creating and persisting vector store at '{db_path}'...")
    db = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=db_path
    )
    print("Index created successfully from all PDF files.")


def answer_query(query: str, db_path: str = VECTOR_DB_PATH):
    """
    Answers a query using the RAG system.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    llm = ChatOllama(model=LLM_MODEL)
    retriever = db.as_retriever()

    template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and professional.

Question: {question}

Context: {context}

Answer:
"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)


if __name__ == "__main__":
    create_mock_pdf()

    if not os.path.isdir(VECTOR_DB_PATH):
        print("Vector database not found. Creating a new one...")
        create_index_from_directory(directory_path=DATA_DIR)
    else:
        print("Existing vector database found.")

    queries = [
        "What was the revenue for Project Griffin in Q3 and what drove this change?",
        "What is the R&D department currently focused on?",
    ]

    for user_query in queries:
        print("-" * 50)
        print(f"❓ Query: {user_query}")
        answer = answer_query(query=user_query)
        print(f"🤖 Answer: {answer}")