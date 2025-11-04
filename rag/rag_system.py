import os
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sentence_transformers import CrossEncoder
from typing import List
from langchain_core.documents import Document

# Configuration
VECTOR_DB_PATH = "vector_db"
DATA_DIR = "data"

# Optimized for business reports (longer context, more overlap for coherence)
CHUNK_SIZE = 1500  # Larger chunks to capture complete ideas/sections
CHUNK_OVERLAP = 300  # Higher overlap to maintain context continuity

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Semantic re-ranker
LLM_MODEL = "llama3:8b"

# Retrieval parameters
INITIAL_K = 10  # Retrieve more documents initially
RERANK_TOP_K = 5  # Return top 5 after re-ranking


class SemanticReranker:
    """
    Semantic re-ranker using a cross-encoder model to improve retrieval quality.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        print(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = RERANK_TOP_K) -> List[Document]:
        """Re-ranks documents based on semantic similarity."""
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata['rerank_score'] = float(score)
            reranked_docs.append(doc)
        
        return reranked_docs


def create_mock_pdf():
    """
    Creates a mock PDF file for the SALESPERSON use case if it doesn't exist.
    This file simulates a client briefing document.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Use a different file name to avoid conflicts if old one exists
    mock_pdf_path = os.path.join(DATA_DIR, "client_brief_synthcore.pdf")
    
    if not os.path.exists(mock_pdf_path):
        print(f"Creating mock PDF: {mock_pdf_path}")
        c = canvas.Canvas(mock_pdf_path, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "Client Briefing: SynthCore Robotics")
        
        c.setFont("Helvetica", 11)
        
        # --- Page 1: Client Profile ---
        text_lines = [
            "1.0 CLIENT PROFILE: SYNTHCORE ROBOTICS",
            "Business Model: SynthCore operates on a B2B (Business-to-Business) model.",
            "They design, manufacture, and sell high-precision AI-driven robotic arms for",
            "logistics and semiconductor manufacturing.",
            "Strategy: Their core strategy is 'Deep Tech Differentiation'. They focus on",
            "patenting unique hardware (like their 'Cogni-Arm' product) and bundling it with",
            "a proprietary AI software platform, creating a strong ecosystem lock-in.",
            "They are aggressively targeting expansion into the European logistics market.",
            "",
            "2.0 SWOT ANALYSIS",
            "Strengths:",
            "- Patented 'Cogni-Arm' technology offers 15% higher precision than competitors.",
            "- Strong R&D division with key talent from top universities.",
            "- High-margin recurring revenue from their AI software subscription.",
            "Weaknesses:",
            "- High reliance on a single-source supplier for their custom-built microchips.",
            "- Low brand recognition in the European market compared to local players.",
            "- Sales cycle is long (9-12 months) due to high product cost.",
        ]
        
        y_position = height - 108
        for line in text_lines:
            c.drawString(72, y_position, line)
            y_position -= 15
        
        c.showPage() # Create a new page

        # --- Page 2: Industry & Competitor Analysis ---
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "Industry & Competitor Landscape")
        
        c.setFont("Helvetica", 11)
        text_lines_page_2 = [
            "3.0 INDUSTRY ANALYSIS: ROBOTIC AUTOMATION",
            "Challenges:",
            "- Navigating complex and fragmented regulatory hurdles for autonomous systems.",
            "- High cost of R&D and materials, putting pressure on margins.",
            "- Shortage of skilled technicians for maintenance and integration.",
            "Innovations:",
            "- The key innovation is the shift from 'dumb' automation to 'adaptive' automation.",
            "- Growing trend of 'swarm robotics' where multiple robots coordinate tasks.",
            "- Use of digital twins for simulating and optimizing robotic workflows.",
            "",
            "4.0 COMPETITOR ANALYSIS",
            "RoboCorp Global: The 800-pound gorilla. They compete on scale and price.",
            "Their weakness is their 'one-size-fits-all' software, which is less flexible.",
            "MechFuture Solutions: A nimble, VC-backed startup. They focus purely on",
            "the semiconductor space and are known for their cutting-edge vision systems.",
            "They do not have an equivalent to the 'Cogni-Arm'.",
        ]
        
        y_position = height - 108
        for line in text_lines_page_2:
            c.drawString(72, y_position, line)
            y_position -= 15

        c.save()


def create_index_from_directory(directory_path: str, db_path: str = VECTOR_DB_PATH):
    """
    Creates a vector index from all PDF files found in a directory.
    """
    # Clean out the old vector store if it exists, to force re-indexing
    if os.path.exists(db_path):
        print(f"Removing old vector database at '{db_path}'...")
        import shutil
        shutil.rmtree(db_path)
        
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
            
            for doc in documents:
                doc.metadata['source_file'] = pdf_file
            
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
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(all_docs)
    
    if not chunks:
        print("Could not split the documents into chunks.")
        return
    
    print(f"Split documents into {len(chunks)} chunks.")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
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


def answer_query(query: str, db_path: str = VECTOR_DB_PATH, use_reranker: bool = True):
    """
    Answers a query using the RAG system, evaluates the answer for faithfulness,
    and returns a structured dictionary.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    llm = ChatOllama(model=LLM_MODEL)
    
    reranker = SemanticReranker() if use_reranker else None
    
    retriever = db.as_retriever(
        search_kwargs={"k": INITIAL_K if use_reranker else RERANK_TOP_K}
    )
    
    # --- 1. RAG Generation Chain ---
    
    # ** NEW PROMPT ** - Tailored for a salesperson's needs
    template = """
You are an expert sales briefing assistant. You are helping a salesperson prepare for a client meeting.
Use the following pieces of retrieved context to answer the question.

Your goal is to be factual, concise, and directly useful.
- Extract key facts, strategies, names, and numbers.
- Structure your answer clearly. Use bullet points if helpful.
- If the information is not in the context, state that clearly. DO NOT make up information.

Question: {question}

Context: {context}

Answer:
"""
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        """Format documents with optional re-ranking scores"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            score_info = f" (Relevance: {doc.metadata.get('rerank_score', 'N/A'):.3f})" if 'rerank_score' in doc.metadata else ""
            source = doc.metadata.get('source_file', 'Unknown')
            formatted.append(f"[Document {i} from {source}{score_info}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    def retrieve_and_rerank(query: str) -> List[Document]:
        """Retrieve documents and optionally re-rank them"""
        docs = retriever.invoke(query)
        
        if use_reranker and reranker and docs:
            print(f"Retrieved {len(docs)} documents. Re-ranking...")
            docs = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
            print(f"Re-ranked to top {len(docs)} documents.")
        
        return docs
    
    generation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    # --- 2. Run Generation and Capture Intermediate Data ---
    
    print(f"Answering query: {query}")
    retrieved_docs = retrieve_and_rerank(query)
    formatted_context = format_docs(retrieved_docs)
    
    if not retrieved_docs:
        print("No documents retrieved. Cannot answer query.")
        return {
            "answer": "I could not find any relevant documents to answer this question.",
            "evaluation": {"faithfulness_score": "N/A", "reasoning": "No documents retrieved."},
            "context": ""
        }
        
    answer = generation_chain.invoke({
        "context": formatted_context, 
        "question": query
    })
    
    # --- 3. Faithfulness Evaluation Chain ---
    print("\n🔍 Evaluating answer faithfulness...")
    
    eval_llm = ChatOllama(model=LLM_MODEL, format="json")
    
    eval_template = """
You are an expert evaluator. Your task is to assess if the 'Answer' is faithfully supported
by the 'Context'. The answer is 'faithful' if all claims it makes can be verified 
from the context.

Respond *only* with a JSON object with two keys:
1. 'faithfulness_score': A float between 0.0 (not faithful) and 1.0 (fully faithful).
2. 'reasoning': A brief explanation for your score, citing evidence.

Context:
{context}

Answer:
{answer}

JSON:
"""
    
    eval_prompt = PromptTemplate.from_template(eval_template)
    
    evaluator_chain = eval_prompt | eval_llm | JsonOutputParser()
    
    try:
        evaluation = evaluator_chain.invoke({
            "context": formatted_context,
            "answer": answer
        })
    except Exception as e:
        print(f"Error during evaluation: {e}")
        evaluation = {"faithfulness_score": "Error", "reasoning": str(e)}

    return {
        "answer": answer,
        "evaluation": evaluation,
        "context": formatted_context
    }


if __name__ == "__main__":
    create_mock_pdf()
    
    # Force re-creation of the index if the script logic changes
    # For a real app, you'd check if the PDF content has changed
    force_reindex = True 
    
    if not os.path.isdir(VECTOR_DB_PATH) or force_reindex:
        print("Vector database not found or re-indexing is forced.")
        create_index_from_directory(directory_path=DATA_DIR)
    else:
        print("Existing vector database found.")
    
    # --- ** NEW QUERIES ** ---
    # These queries are tailored to the new "SynthCore Robotics" PDF
    queries = [
        # --- Good, direct questions based on the PDF ---
        "What is SynthCore Robotics' business model and core strategy?",
        "Summarize the key strengths and weaknesses for SynthCore.",
        "Who are SynthCore's main competitors and what are their weaknesses?",
        "What are the latest innovations in their industry?",
        
        # --- A question designed to fail (test faithfulness) ---
        "What was SynthCore's Q3 revenue?" 
    ]
    
    for user_query in queries:
        print("-" * 70)
        print(f"❓ Query: {user_query}")
        
        result = answer_query(query=user_query, use_reranker=True)
        
        print(f"🤖 Answer: {result['answer']}")
        print("\n" + "="*30 + " EVALUATION " + "="*30)
        
        if 'faithfulness_score' in result['evaluation']:
            print(f"✅ Faithfulness Score: {result['evaluation']['faithfulness_score']}")
            print(f"🤔 Reasoning: {result['evaluation']['reasoning']}")
        else:
            print(f"⚠️ Evaluation failed: {result['evaluation']}")
            
        print("="*72)
        print()