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
import traceback

# Use proper package import - no sys.path hacking needed!
from .content_generation.content_generation_agent import create_content_generation_agent


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
    
    for material in selected_materials:
        generation_task = {
            "task_id": str(uuid.uuid4()),
            "material_id": material["material_id"],
            "type": material["material_type"],
            "title": material["title"],
            "description": material["description"],
            "content_requirements": material["content_requirements"],
            "claim_references": material["claim_references"],
            "priority": material["priority"],
            "status": "pending",
            "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            # Include user-provided creative prompt so downstream generators can honor it
            "user_prompt": user_prompt,
            "instructions": generate_creation_instructions(material, user_prompt)
        }
        generation_queue.append(generation_task)
    
    return Command(
        update={"generation_queue": generation_queue}
    )

def generate_creation_instructions(material: Dict, user_prompt: Optional[str] = None) -> str:
    """Generate specific instructions for material creation agents"""
    
    material_type = material["material_type"]
    content_req = material["content_requirements"]
    
    base_instruction = f"""
    Create a {material_type} with the following specifications:
    
    Title: {material["title"]}
    Description: {material["description"]}
    
    Content Requirements:
    - Style: {content_req.get("style", "professional")}
    - Data visualization: {content_req.get("data_visualization", "optional")}
    - Text amount: {content_req.get("text_amount", "moderate")}
    - Color scheme: {content_req.get("color_scheme", "corporate")}
    """

    if user_prompt:
        base_instruction += f"\nAdditional creative direction: {user_prompt}\n"
    
    if material_type == "slide":
        return base_instruction + """
        
        SLIDE CREATION INSTRUCTIONS:
        - Create a single, impactful slide
        - Include headline, key statistics, and supporting visual
        - Use bullet points sparingly (max 3-4 points)
        - Ensure text is readable from a distance
        - Include source attribution for statistics
        """
    
    elif material_type == "infographic":
        return base_instruction + """
        
        INFOGRAPHIC CREATION INSTRUCTIONS:
        - Design a visually appealing infographic
        - Combine statistics, icons, and minimal text
        - Use a clear visual hierarchy
        - Include a compelling headline and conclusion
        - Optimize for social sharing if applicable
        """
    
    elif material_type == "chart":
        return base_instruction + """
        
        CHART CREATION INSTRUCTIONS:
        - Create clear, accurate data visualizations
        - Choose appropriate chart type for the data
        - Include clear labels and legends
        - Use color coding effectively
        - Add title and source information
        """
    
    elif material_type == "video_explainer":
        return base_instruction + """
        
        VIDEO CREATION INSTRUCTIONS:
        - Create a 30-60 second explainer video
        - Include voice-over script
        - Design simple, clean visuals
        - Use animations to highlight key points
        - Include call-to-action at the end
        """
    
    elif material_type == "presentation_deck":
        return base_instruction + """
        
        PRESENTATION DECK INSTRUCTIONS:
        - Create 5-8 slides maximum
        - Include title slide, key points, and conclusion
        - Maintain consistent design throughout
        - Use speaker notes for additional context
        - Include next steps/call-to-action slide
        """
    
    else:
        return base_instruction + "\n\nCreate according to best practices for this material type."

# Database operations
def save_materials_decision(state: MaterialsDecisionState) -> Command:
    """Save the materials decision to database"""
    
    session_id = state["session_id"]
    salesperson_id = state["salesperson_id"]
    recommendations = state["material_recommendations"]
    selected_materials = state["selected_materials"]
    
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

@tool
def trigger_content_generation(generation_queue: str) -> str:
    """Trigger content generation agent with material specifications
    
    Input: JSON string with generation queue containing material specs
    Output: Generated file paths and status
    """
    try:
        queue = json.loads(generation_queue)
        
        print(f"\n📋 Generation Queue received ({len(queue)} items):")
        for i, item in enumerate(queue, 1):
            print(f"   {i}. {item.get('title')} ({item.get('material_type')})")
        
        # Convert materials queue to content agent format
        input_state = convert_queue_to_agent_input(queue)
        
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

def convert_queue_to_agent_input(queue: List[Dict]) -> Dict:
    """Convert materials queue to content generation agent input format"""
    
    # Extract different material types
    chart_specs = []
    ai_image_prompts = []
    video_specs = []
    
    for item in queue:
        # The queue uses "type" not "material_type"
        material_type = item.get("type")
        
        print(f"   Processing: {item.get('title')} - Type: {material_type}")
        
        if material_type in ["chart", "slide"]:
            # Convert to chart specification
            chart_specs.append({
                "type": determine_chart_type(item),
                "title": item.get("title"),
                "data": extract_data_from_claims(item)
            })
            
        elif material_type in ["infographic"]:
            # Convert to AI image prompt
            ai_image_prompts.append({
                "prompt": generate_infographic_prompt(item),
                "aspect_ratio": "16:9",
                "filename": item.get("material_id")
            })
            
        elif material_type in ["video_explainer", "presentation_deck"]:
            video_specs.append(item)
    
    # Build content agent input
    return {
        "context": {
            "meeting_type": "materials_generation",
            "client_name": queue[0].get("client_context", "Client") if queue else "Client",
            "sales_objectives": ["Create requested materials"]
        },
        "data_available": {
            "chart_specifications": chart_specs,
            "ai_image_prompts": ai_image_prompts,
            "video_specifications": video_specs
        },
        "messages": [],
        "generated_files": [],
        "errors": []
    }
def determine_chart_type(material_spec: Dict) -> str:
    """Determine chart type from material specification"""
    title = material_spec.get("title", "").lower()
    description = material_spec.get("description", "").lower()
    
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

def generate_infographic_prompt(material_spec: Dict) -> str:
    """Generate AI image prompt from infographic specification"""
    title = material_spec.get("title", "")
    description = material_spec.get("description", "")
    requirements = material_spec.get("content_requirements", {})
    
    style = requirements.get("style", "professional")
    color_scheme = requirements.get("color_scheme", "corporate")
    
    return f"{title}: {description}, {style} style, {color_scheme} colors, high quality infographic design"

@tool
def trigger_video_generation(material_specs: str) -> str:
    """Trigger video generation agent with material specifications
    
    Input: JSON string with material specifications  
    Output: Generation task ID or status
    """
 
    return f"Video generation queued with task ID: {uuid.uuid4()}"

@tool
def get_client_preferences(client_id: str) -> str:
    """Retrieve client presentation preferences from database
    
    Input: Client ID
    Output: JSON string with client preferences
    """
    # Mock implementation - would query client preferences database
    return json.dumps({
        "preferred_style": "modern",
        "color_scheme": "corporate_blue",
        "presentation_length": "10-15_minutes",
        "visual_preference": "data_heavy"
    })

def trigger_generation_node(state: MaterialsDecisionState) -> Command:
    """Trigger content generation for approved materials"""
    
    generation_queue = state.get("generation_queue", [])
    
    if not generation_queue:
        return Command(update={"status": "no_materials_to_generate"})
    
    # Trigger content generation
    # @tool returns a StructuredTool which must be invoked explicitly
    result = trigger_content_generation.invoke(json.dumps(generation_queue))
    result_data = json.loads(result)
    
    return Command(update={
        "generation_status": result_data.get("status"),
        "generated_files": result_data.get("generated_files", [])
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