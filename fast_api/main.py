import uvicorn
import json
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional

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
# rag_agent_app = get_rag_graph()
# marketing_agent_app = get_marketing_graph()

# --- 3. Create the FastAPI app ---
app = FastAPI()

# --- 4. Define Request Models (Pydantic) ---
class FactCheckRequest(BaseModel):
    claim: str
    salesperson_id: str
    client_context: str

class MaterialsRequest(BaseModel):
    verified_claims: List[Dict]
    salesperson_id: str
    client_context: str
    user_prompt: Optional[str] = None


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
        print(final_update)
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

# --- 7. Materials Generation Endpoint ---
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

# --- Add other endpoints for RAG ---
# @app.post("/chat-rag")
# async def ...
#
# @app.post("/generate-marketing")
# async def ...

# --- (Optional) Run for testing ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)