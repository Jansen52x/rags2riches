from langchain_ollama import ChatOllama
from langchain.tools import tool
from typing_extensions import TypedDict, List, Optional, Dict, Literal
from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
import json
import uuid
import psycopg
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import traceback
import os

# Use proper package import - no sys.path hacking needed!
from .content_generation.content_generation_agent import create_content_generation_agent

# Paths for generated assets served via FastAPI static mount
GENERATED_CONTENT_DIR = Path(__file__).resolve().parent.parent / "generated_content"
FALLBACK_IMAGE_DIR = GENERATED_CONTENT_DIR / "images"
FALLBACK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_MODE_ENABLED = os.getenv("MATERIALS_AGENT_MODE", "").lower() in {"heuristic", "deterministic"}
SKIP_DB_SAVE = os.getenv("MATERIALS_AGENT_SKIP_DB", "").lower() in {"1", "true", "yes", "on"}
SKIP_GENERATION = os.getenv("MATERIALS_AGENT_SKIP_GENERATION", "").lower() in {"1", "true", "yes", "on"}


# State definitions
class MaterialType(str, Enum):
    SLIDE = "slide"
    INFOGRAPHIC = "infographic"
    CHART = "chart"
    VIDEO_EXPLAINER = "video_explainer"
    SOCIAL_MEDIA_POST = "social_media_post"
    PRESENTATION_DECK = "presentation_deck"

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class MaterialRequest:
    material_id: str
    material_type: MaterialType
    title: str
    description: str
    content_requirements: Dict
    priority: Priority
    claim_references: List[str]
    estimated_time_minutes: int
    
class MaterialsDecisionState(TypedDict):
    session_id: str
    salesperson_id: str
    client_context: str
    # Optional free-form creative guidance provided by user (e.g., "cafe theme, warm tones")
    user_prompt: Optional[str]
    verified_claims: List[Dict]  # Claims that passed fact-checking
    
    # Decision outputs
    material_recommendations: List[Dict]
    selected_materials: List[Dict]
    generation_queue: List[Dict]
    generated_files: List[str]
    generation_status: Optional[str]
    
    # Status tracking
    decision_complete: bool
    user_feedback: Optional[str]

# Core Decision Logic
def _attempt_parse_recommendations(raw: str) -> List[Dict]:
    """Try to parse LLM output into a recommendations list.

    Attempts standard JSON parse first, then a simple brace-extraction fallback.
    Returns an empty list if nothing valid is found.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return list(data.get("recommendations", []))
        if isinstance(data, list):
            return list(data)
    except Exception:
        pass

    # crude brace extraction fallback
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            data = json.loads(snippet)
            if isinstance(data, dict):
                return list(data.get("recommendations", []))
            if isinstance(data, list):
                return list(data)
    except Exception:
        pass
    return []


def _heuristic_recommendations(verified_claims: List[Dict], client_context: str) -> List[Dict]:
    """Generate simple, deterministic recommendations without an LLM.

    Maps common claim patterns to a small set of useful materials so the UI
    remains functional even when the LLM returns non-JSON.
    """
    recs: List[Dict] = []
    claim_ids = [c.get("claim_id", f"claim_{i+1:03d}") for i, c in enumerate(verified_claims)]

    def add(material_type, title, description, priority, minutes, refs=None, style="professional", dv="required", text="moderate", color="corporate", special=None, reasoning=""):
        recs.append({
            "material_id": str(uuid.uuid4()),
            "material_type": material_type,
            "title": title,
            "description": description,
            "claim_references": refs if refs is not None else claim_ids,
            "content_requirements": {
                "style": style,
                "data_visualization": dv,
                "text_amount": text,
                "color_scheme": color,
                "special_elements": special or ["charts", "statistics"]
            },
            "priority": priority,
            "estimated_time_minutes": minutes,
            "reasoning": reasoning or "Heuristic recommendation based on claim content and context."
        })

    has_market_size = any("market" in c.get("claim", "").lower() and ("sgd" in c["claim"].lower() or "billion" in c["claim"].lower()) for c in verified_claims)
    has_leader_share = any("lead" in c.get("claim", "").lower() or "market share" in c.get("claim", "").lower() for c in verified_claims)

    if has_market_size:
        add("chart", "Market Size Overview", "Chart showing market size and YoY growth.", "high", 35)
        add("slide", "Key Market Statistic", "One-pager slide with the headline figure and source.", "high", 20)

    if has_leader_share:
        add("infographic", "Market Share Snapshot", "Infographic comparing top players and share.", "medium", 40)

    # Always include a small focused deck tying claims to client context
    add("presentation_deck", "Client Opportunity Deck", "5-7 slides summarizing insights tailored to client.", "medium", 60,
        style="modern", dv="optional", text="moderate", color="corporate", special=["icons", "statistics"]) 

    # Cap to 3-6 items
    return recs[:6]


def analyze_claims_for_materials(state: MaterialsDecisionState) -> Command:
    """Analyze verified claims and recommend appropriate presentation materials"""
    
    verified_claims = state["verified_claims"]
    client_context = state["client_context"]
    recommendations: List[Dict]

    if EVAL_MODE_ENABLED:
        print("MATERIALS_AGENT_MODE=heuristic detected; using deterministic recommendations")
        recommendations = _heuristic_recommendations(verified_claims, client_context)
    else:
        prompt = f"""
    You are a presentation materials strategist helping a salesperson create compelling materials for a client meeting.
    
    Based on the verified claims below, recommend the most effective presentation materials to create.
    Consider the client context and what would be most persuasive and professional.
    
    VERIFIED CLAIMS:
    {json.dumps(verified_claims, indent=2)}
    
    CLIENT CONTEXT:
    {client_context}
    
    For each material recommendation, provide:
    1. Material type (slide, infographic, chart, video_explainer, social_media_post, presentation_deck)
    2. Title and description
    3. Which claims it should reference
    4. Content requirements (text, data visualization needs, style preferences)
    5. Priority level (high/medium/low)
    6. Estimated creation time in minutes
    
    Consider these factors:
    - What type of data/claims work best for each material type
    - Client preferences (if mentioned in context)
    - Professional presentation standards
    - Time efficiency for the salesperson
    
    Output as valid JSON in this format:
    {{
        "recommendations": [
            {{
                "material_type": "slide|infographic|chart|video_explainer|social_media_post|presentation_deck",
                "title": "Compelling title for the material",
                "description": "Brief description of what this material will show",
                "claim_references": ["claim_id_1", "claim_id_2"],
                "content_requirements": {{
                    "style": "professional|modern|minimalist|corporate",
                    "data_visualization": "required|optional|none",
                    "text_amount": "minimal|moderate|detailed",
                    "color_scheme": "corporate|vibrant|monochrome",
                    "special_elements": ["charts", "icons", "testimonials", "statistics"]
                }},
                "priority": "high|medium|low",
                "estimated_time_minutes": 30,
                "reasoning": "Why this material type is recommended"
            }}
        ],
        "overall_strategy": "Brief explanation of the recommended materials strategy"
    }}
    
    Recommend 3-6 materials maximum. Focus on quality and impact over quantity.
    """
        
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.3)
        response = llm.invoke(prompt).content
    
        # Try robust parsing first
        recommendations = _attempt_parse_recommendations(response)

        # Add IDs if any
        for rec in recommendations:
            rec.setdefault("material_id", str(uuid.uuid4()))

        # If still empty, fall back to deterministic heuristics
        if not recommendations:
            print("Error parsing LLM response, using fallback")
            recommendations = _heuristic_recommendations(verified_claims, client_context)

    print(f"Generated {len(recommendations)} material recommendations")
    
    return Command(
        update={"material_recommendations": recommendations, "decision_complete": True}
    )

def prioritize_materials(state: MaterialsDecisionState) -> Command:
    """Prioritize materials based on time constraints and impact"""
    
    recommendations = state["material_recommendations"]
    
    # Sort by priority (high -> medium -> low) and estimated time
    priority_order = {"high": 3, "medium": 2, "low": 1}
    
    sorted_materials = sorted(
        recommendations,
        key=lambda x: (priority_order.get(x.get("priority", "low"), 1), -x.get("estimated_time_minutes", 0)),
        reverse=True
    )
    
    # Select materials based on available time (assume 2 hours = 120 minutes)
    available_time = 120
    selected_materials = []
    total_time = 0
    
    for material in sorted_materials:
        material_time = material.get("estimated_time_minutes", 30)
        if total_time + material_time <= available_time:
            selected_materials.append(material)
            total_time += material_time
        else:
            # Add to queue for future consideration
            material["status"] = "queued"
    
    print(f"Selected {len(selected_materials)} materials for immediate creation (Total time: {total_time} minutes)")
    
    return Command(
        update={"selected_materials": selected_materials}
    )

def create_generation_queue(state: MaterialsDecisionState) -> Command:
    """Create a prioritized queue for material generation"""
    
    selected_materials = state["selected_materials"]
    user_prompt: Optional[str] = state.get("user_prompt")
    
    # Create generation tasks with specific instructions
    generation_queue = []
    
    print(f"\n🎯 Creating generation queue from {len(selected_materials)} selected materials:")
    
    for material in selected_materials:
        material_type = material["material_type"]
        print(f"\n   📝 Material: {material['title']}")
        print(f"      Type: {material_type}")
        print(f"      Priority: {material.get('priority', 'N/A')}")
        
        generation_task = {
            "task_id": str(uuid.uuid4()),
            "material_id": material["material_id"],
            "type": material["material_type"],  # This is what content agent will read
            "title": material["title"],
            "description": material["description"],
            "content_requirements": material["content_requirements"],
            "claim_references": material["claim_references"],
            "priority": material["priority"],
            "status": "pending",
            "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            "user_prompt": user_prompt  # User creative guidance
        }
        generation_queue.append(generation_task)
        print(f"      ✓ Added to queue with type='{material_type}'")
    
    print(f"\n✅ Generation queue created with {len(generation_queue)} tasks")
    
    return Command(
        update={"generation_queue": generation_queue}
    )

# Database operations
def save_materials_decision(state: MaterialsDecisionState) -> Command:
    """Save the materials decision to database"""
    
    session_id = state["session_id"]
    salesperson_id = state["salesperson_id"]
    recommendations = state["material_recommendations"]
    selected_materials = state["selected_materials"]

    if SKIP_DB_SAVE:
        print("Skipping materials decision database save (MATERIALS_AGENT_SKIP_DB enabled)")
        return Command(update={})
    
    try:
        with psycopg.connect("dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432") as conn:
            # Create materials_decisions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS materials_decisions (
                    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id TEXT NOT NULL,
                    salesperson_id TEXT NOT NULL,
                    recommendations JSONB NOT NULL,
                    selected_materials JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Insert the decision
            conn.execute(
                """INSERT INTO materials_decisions 
                   (session_id, salesperson_id, recommendations, selected_materials) 
                   VALUES (%s, %s, %s, %s)""",
                (session_id, salesperson_id, json.dumps(recommendations), json.dumps(selected_materials))
            )
            
        print("Materials decision saved to database")
        return Command(update={"status": "saved"})
        
    except Exception as e:
        print(f"Error saving materials decision: {e}")
        return Command(update={"status": "save_failed"})
    
def convert_queue_to_agent_input(queue: List[Dict], client_context: str = "") -> Dict:
    """Convert materials queue to content generation agent input format
    
    Args:
        queue: List of material generation tasks
        client_context: Client context string from the original request
    """
    
    # Extract different material types
    chart_specs = []
    ai_image_prompts = []
    video_specs = []
    
    
    print(f"\n🔍 Converting {len(queue)} items from queue to agent input:")
    
    for item in queue:
        # The queue uses "type" not "material_type"
        material_type = item.get("type")
        
        print(f"\n   📌 Processing item:")
        print(f"      Title: {item.get('title')}")
        print(f"      Type: {material_type}")
        print(f"      Description: {item.get('description', '')[:80]}...")
        
        if material_type in ["chart", "slide", "infographic"]:
            # Convert to chart specification
            # Infographics are data visualizations, not AI-generated images
            chart_spec = {
                "type": determine_chart_type(item),
                "title": item.get("title"),
                "data": extract_data_from_claims(item)
            }
            chart_specs.append(chart_spec)
            print(f"      ✓ Added to CHART_SPECS: {chart_spec['type']}")
            
        elif material_type in ["video_explainer", "presentation_deck"]:
            video_specs.append(item)
            print(f"      ✓ Added to VIDEO_SPECS")
        else:
            print(f"      ⚠️  Unknown material type: {material_type}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Chart Specifications: {len(chart_specs)}")
    print(f"   AI Image Prompts: {len(ai_image_prompts)}")
    print(f"   Video Specifications: {len(video_specs)}")
    
    # Build content agent input
    return {
        "context": {
            "meeting_type": "materials_generation",
            "client_name": queue[0].get("client_context", "Client") if queue else "Client",
            "sales_objectives": ["Create requested materials"]
        },
        "data_available": {
            "client_context": client_context,
            "chart_specifications": chart_specs,
            "ai_image_prompts": ai_image_prompts,
            "video_specifications": video_specs
        },
        "messages": [],
        "generated_files": [],
        "errors": []
    }

@tool
def trigger_content_generation(generation_queue: str) -> str:
    """Trigger content generation agent with material specifications
    
    Input: JSON string with generation queue containing material specs
    Output: Generated file paths and status
    """
    try:
        data = json.loads(generation_queue)
        
        # Handle both old format (just queue) and new format (queue + client_context)
        if isinstance(data, dict) and "generation_queue" in data:
            queue = data["generation_queue"]
            client_context = data.get("client_context", "")
        else:
            # Backward compatibility: data is the queue itself
            queue = data
            client_context = ""
        
        print(f"\n📋 Generation Queue received ({len(queue)} items):")
        for i, item in enumerate(queue, 1):
            print(f"   {i}. {item.get('title')} ({item.get('type')})")
        
        if client_context:
            print(f"\n📝 Client Context: {client_context[:100]}...")
        
        # Convert materials queue to content agent format
        input_state = convert_queue_to_agent_input(queue, client_context)
        
        print(f"\n🔄 Converted input state:")
        print(f"   AI Image Prompts: {len(input_state['data_available'].get('ai_image_prompts', []))}")
        print(f"   Chart Specs: {len(input_state['data_available'].get('chart_specifications', []))}")
        print(json.dumps(input_state, indent=2))
        
        # Create and invoke agent
        agent = create_content_generation_agent()
        result = agent.invoke(input_state)
        
        return json.dumps({
            "status": "success",
            "generated_files": result.get("generated_files", []),
            "generation_count": len(result.get("generated_files", []))
        })
        
    except Exception as e:
        print(f"\n❌ Error in trigger_content_generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

def determine_chart_type(material_spec: Dict) -> str:
    """Determine chart type from material specification"""
    title = material_spec.get("title", "").lower()
    description = material_spec.get("description", "").lower()
    material_type = material_spec.get("type", "").lower()
    
    # Check material type first
    if material_type == "infographic":
        # Infographics are visual data presentations - determine best chart type
        if "market share" in title or "market share" in description or "share" in title:
            return "market_share"
        elif "comparison" in title or "compare" in description:
            return "competitive_matrix"
        elif "swot" in title or "strengths" in description:
            return "swot_analysis"
        else:
            return "market_share"  # Default for infographics
    
    # Regular chart type detection
    if "market share" in title or "market share" in description:
        return "market_share"
    elif "swot" in title:
        return "swot_analysis"
    elif "growth" in title or "trend" in title:
        return "growth_trend"
    elif "competitive" in title or "positioning" in title:
        return "competitive_matrix"
    else:
        return "market_share"  # default

def extract_data_from_claims(material_spec: Dict) -> Dict:
    """Extract data from claims for chart generation"""
    claims = material_spec.get("claim_references", [])
    # Parse claims to extract structured data
    # This would need to be more sophisticated in practice
    return {
        "companies": ["Company A", "Company B", "Company C"],
        "market_share": [35, 30, 25]
    }


def _render_text_image(path: Path, title: str, description: str, priority: str) -> None:
    """Create a simple branded placeholder image for a material recommendation."""
    width, height = 1280, 720
    palette = {
        "high": (199, 56, 56),
        "medium": (240, 143, 35),
        "low": (40, 160, 92)
    }
    colour = palette.get(priority.lower(), (32, 74, 135))

    image = Image.new("RGB", (width, height), colour)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("Arial.ttf", 80)
        body_font = ImageFont.truetype("Arial.ttf", 44)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    wrapped_title = textwrap.fill(title, width=25)
    draw.multiline_text((70, 90), wrapped_title, fill="white", font=title_font, spacing=12)

    wrapped_desc = textwrap.fill(description, width=32)
    draw.multiline_text((70, 260), wrapped_desc, fill="white", font=body_font, spacing=8)

    footer = f"PRIORITY: {priority.upper()}"
    draw.text((70, height - 140), footer, fill="white", font=body_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _fallback_generate_assets(queue: List[Dict]) -> List[str]:
    """Generate simple visual placeholders when no assets are produced."""
    generated: List[str] = []
    for item in queue:
        material_id = item.get("material_id", str(uuid.uuid4()))
        material_type = item.get("type", "material")
        filename = f"{material_id}_{material_type}.png"
        output_path = FALLBACK_IMAGE_DIR / filename
        _render_text_image(
            output_path,
            title=item.get("title", "Marketing Asset"),
            description=item.get("description", ""),
            priority=item.get("priority", "medium")
        )
        generated.append(f"/generated_content/images/{filename}")
    return generated

def trigger_generation_node(state: MaterialsDecisionState) -> Command:
    """Trigger content generation for approved materials"""
    
    generation_queue = state.get("generation_queue", [])
    client_context = state.get("client_context", "")
    
    if not generation_queue:
        return Command(update={"status": "no_materials_to_generate"})
    
    if SKIP_GENERATION:
        print("Skipping content generation (MATERIALS_AGENT_SKIP_GENERATION enabled)")
        return Command(update={"generation_status": "skipped (evaluation)", "generated_files": []})
    
    generated_files: List[str] = []
    generation_status = "skipped"

    try:
        # Pass both queue and client_context
        payload = {
            "generation_queue": generation_queue,
            "client_context": client_context
        }
        result = trigger_content_generation.invoke(json.dumps(payload))
        result_data = json.loads(result)
        generation_status = result_data.get("status", "completed")
        generated_files = result_data.get("generated_files", [])
    except Exception as exc:
        generation_status = f"error: {exc}"
        print(f"⚠️ Generation agent failed, falling back to placeholders: {exc}")

    if not generated_files:
        fallback_files = _fallback_generate_assets(generation_queue)
        if fallback_files:
            generated_files = fallback_files
            generation_status = f"fallback_generated ({generation_status})"
    print(f"   → Final generated file list: {generated_files}")

    return Command(update={
        "generation_status": generation_status,
        "generated_files": generated_files
    })

# Setup function for testing
def create_materials_decision_workflow():
    """Create and return the materials decision workflow"""
    
    # Create the state graph
    graph = StateGraph(MaterialsDecisionState)
    
    # Add nodes
    graph.add_node("analyze_claims", analyze_claims_for_materials)
    graph.add_node("prioritize", prioritize_materials)
    graph.add_node("create_queue", create_generation_queue)
    graph.add_node("trigger_generation", trigger_generation_node)
    graph.add_node("save_decision", save_materials_decision)
    
    # Add edges
    graph.add_edge(START, "analyze_claims")
    graph.add_edge("analyze_claims", "prioritize")
    graph.add_edge("prioritize", "create_queue")
    graph.add_edge("create_queue", "trigger_generation")  # NEW
    graph.add_edge("trigger_generation", "save_decision")
    graph.add_edge("save_decision", END)
    
    return graph.compile()

# Example usage and testing
if __name__ == "__main__":
    # Example verified claims from fact-checker
    sample_verified_claims = [
        {
            "claim_id": "claim_001",
            "claim": "Singapore's e-commerce market reached SGD 9 billion in 2023",
            "verdict": "TRUE",
            "confidence": 0.9,
            "evidence": [
                {"source": "Singapore Government Trade Statistics", "summary": "Official data confirms SGD 9.1B e-commerce sales in 2023"}
            ]
        },
        {
            "claim_id": "claim_002", 
            "claim": "Shopee leads Singapore e-commerce market share",
            "verdict": "TRUE",
            "confidence": 0.85,
            "evidence": [
                {"source": "TechCrunch SE Asia Report", "summary": "Shopee holds 35% market share in Singapore"}
            ]
        }
    ]
    
    # Create initial state
    initial_state = MaterialsDecisionState(
        session_id=str(uuid.uuid4()),
        salesperson_id="SP12345",
        client_context="Small e-commerce startup in Singapore looking to understand market opportunities",
        verified_claims=sample_verified_claims,
        material_recommendations=[],
        selected_materials=[],
        generation_queue=[],
        generated_files=[],
        generation_status=None,
        decision_complete=False,
        user_feedback=None
    )
    
    # Create and run workflow
    workflow = create_materials_decision_workflow()
    final_state = workflow.invoke(initial_state)
    
    print("\n" + "="*50)
    print("MATERIALS DECISION COMPLETE")
    print("="*50)
    print(f"Recommendations: {len(final_state['material_recommendations'])}")
    print(f"Selected for creation: {len(final_state['selected_materials'])}")
    print(f"Generation queue: {len(final_state['generation_queue'])}")
    
    for i, material in enumerate(final_state['selected_materials'], 1):
        print(f"\n{i}. {material['title']} ({material['material_type']})")
        print(f"   Priority: {material['priority']} | Time: {material['estimated_time_minutes']}min")
        print(f"   Description: {material['description']}")