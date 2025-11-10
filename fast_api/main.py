import uvicorn
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from pathlib import Path
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env or secrets.env
load_dotenv()  # Loads .env by default
# Also try to load secrets.env if it exists
secrets_path = Path(__file__).resolve().parent.parent / "secrets.env"
if secrets_path.exists():
    load_dotenv(secrets_path)

from config import settings
from rag_services.embedding_service import EmbeddingService
from rag_services.llm_service import LLMService
from rag_services.rag_service import RAGService
from rag_services.query_builder import QueryBuilder

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# --- 1. Import agents and their specific types/data ---
from agents.fact_checker import (
    get_fact_check_graph, 
    FactCheckState, 
    AGENT_PROGRESS_STEPS
)
# from .agents.rag_agent import get_rag_graph, RagState, ...
# from .agents.marketing_agent import get_marketing_graph, ...
from agents.materials_decision_agent import (
    create_materials_decision_workflow,
    MaterialsDecisionState
)

# --- 2. Build agents ONCE at startup ---
fact_check_agent_app = get_fact_check_graph()
embedding_service = EmbeddingService()
llm_service = LLMService()
rag_service = RAGService(embedding_service, llm_service)
query_builder = QueryBuilder(rag_service)
# rag_agent_app = get_rag_graph()
# marketing_agent_app = get_marketing_graph()

# --- 3. Create the FastAPI app ---
app = FastAPI()

# Ensure generated content directory exists and mount it for static serving
GENERATED_CONTENT_DIR = Path(__file__).resolve().parent / "generated_content"
GENERATED_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated_content", StaticFiles(directory=GENERATED_CONTENT_DIR), name="generated_content")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Define Request Models (Pydantic) ---
class FactCheckRequest(BaseModel):
    claim: str
    salesperson_id: str
    client_context: str

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    k: Optional[int] = Field(None, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")

class BuilderQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    k: Optional[int] = Field(None, description="Number of results to return")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score (0.0-1.0)")
    include_sources: bool = Field(True, description="Include source documents in response")

class MaterialsRequest(BaseModel):
    verified_claims: List[Dict]
    salesperson_id: str
    client_context: str
    user_prompt: Optional[str] = None
# (Define other request models for RAG, Marketing, etc.)

# --- 5. Define the Streaming Generator ---
async def stream_fact_check(initial_state: FactCheckState):
    """
    This generator function runs the agent and yields progress 
    updates as JSON strings.
    """
    try:
        update_count = 0
        final_state = None
        # Use the imported agent app
        async for update in fact_check_agent_app.astream(initial_state):
            progress_data = AGENT_PROGRESS_STEPS[update_count].copy()
            progress_data["type"] = "progress"
            yield f"{json.dumps(progress_data)}\n"
            update_count += 1
            final_state = update
        
        claim_verdict = final_state["save"] 

        # Yield the final "complete" message and the result
        
        final_update = {
            "type" : "result",
            "status_code": 200,
            "final_verdict": claim_verdict
        }
        yield f"{json.dumps(final_update)}\n"

    except Exception as e:
        final_update = {
            "type": "error",
            "status_code": 500,
            "error": f"Agent failed: {str(e)}"
        }
        yield f"{json.dumps(final_update)}\n"

# --- 6. Define the API Endpoint ---
@app.post("/check-claim")
async def check_claim_endpoint(request: FactCheckRequest):
    """
    This is the main API endpoint. It takes the Streamlit request
    and returns a StreamingResponse.
    """
    initial_state = FactCheckState(
        claim_id=str(uuid.uuid4()),
        original_claim=request.claim,
        salesperson_id=request.salesperson_id,
        client_context=request.client_context,
        analyzed_claim="",
        claim_verdict={},
        evidence_log=[],
    )
    
    return StreamingResponse(
        stream_fact_check(initial_state), 
        media_type="application/x-ndjson"
    )

@app.post("/query_rag")
async def query(request: RAGQueryRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation)

    Retrieves relevant documents from the vector database and uses an LLM to generate an answer.
    """
    try:
        result = rag_service.query(
            query=request.query,
            k=request.k,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_rag/builder")
async def query_with_builder(request: BuilderQueryRequest):
    """
    Execute a structured query with filters and advanced options

    Allows filtering by metadata, setting score thresholds, and more control over results.
    """
    try:
        result = query_builder.build_query(
            query=request.query,
            filters=request.filters,
            k=request.k,
            score_threshold=request.score_threshold,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        logger.error(f"Error processing builder query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate-materials")
async def generate_materials_endpoint(request: MaterialsRequest):
    """
    Generate marketing materials based on verified claims.
    This orchestrates both materials decision + content generation.
    
    Flow:
    1. Analyze claims → Recommend materials
    2. Prioritize based on time constraints  
    3. Create generation queue
    4. Trigger content generation (charts, AI images, videos)
    5. Save decisions to database
    6. Return generated file paths
    """
    try:
        # Create initial state for materials decision workflow
        initial_state = MaterialsDecisionState(
            session_id=str(uuid.uuid4()),
            salesperson_id=request.salesperson_id,
            client_context=request.client_context,
            user_prompt=request.user_prompt,
            verified_claims=request.verified_claims,
            material_recommendations=[],
            selected_materials=[],
            generation_queue=[],
            generated_files=[],
            generation_status=None,
            decision_complete=False,
            user_feedback=None
        )
        
        # Create and run the workflow
        print(f"\n🚀 Starting materials generation for {len(request.verified_claims)} verified claims...")
        workflow = create_materials_decision_workflow()
        final_state = workflow.invoke(initial_state)
        
        print(f"✅ Materials generation complete!")
        print(f"   Generated {len(final_state.get('generated_files', []))} files")
        
        return {
            "status": "success",
            "session_id": final_state["session_id"],
            "generated_files": final_state.get("generated_files", []),
            "generation_count": len(final_state.get("generated_files", [])),
            "recommendations": final_state["material_recommendations"],
            "selected_materials": final_state["selected_materials"],
            "generation_status": final_state.get("generation_status", "completed")
        }
    except Exception as e:
        import traceback
        print(f"\n❌ Error in generate_materials_endpoint: {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/generated-files")
async def list_generated_files() -> Dict[str, Any]:
    """Return metadata for files in the generated content directory."""
    files: List[Dict[str, str]] = []
    for path in GENERATED_CONTENT_DIR.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(GENERATED_CONTENT_DIR)
            files.append({
                "name": rel_path.as_posix(),
                "url": f"/generated_content/{rel_path.as_posix()}"
            })
    return {"files": sorted(files, key=lambda x: x["name"])}

@app.post("/generate-materials-mock")
async def generate_materials_mock(request: MaterialsRequest):
    """
    Mock endpoint for UI testing that returns instant recommendations without
    invoking heavy generation or external LLMs. Useful for frontend testing.
    """
    recommendations = [
        {
            "title": "One-slide company summary",
            "material_type": "one_pager",
            "priority": "medium",
            "estimated_time_minutes": 15,
            "description": "A concise summary slide highlighting market share and expansion facts."
        },
        {
            "title": "Customer expansion infographic",
            "material_type": "infographic",
            "priority": "high",
            "estimated_time_minutes": 45,
            "description": "Visual showing expansion across Southeast Asia."
        }
    ]

    return {
        "status": "success",
        "session_id": str(uuid.uuid4()),
        "generated_files": [],
        "generation_count": 0,
        "recommendations": recommendations,
        "selected_materials": []
    }

# --- Add other endpoints for RAG ---
# @app.post("/chat-rag")
# async def ...
#
# @app.post("/generate-marketing")
# async def ...

# --- (Optional) Run for testing ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)