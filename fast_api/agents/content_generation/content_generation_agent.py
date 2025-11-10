from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from typing import Literal
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use relative imports within the package, with fallback for standalone execution

from content_state import ContentAgentState
from content_tools import content_generation_tools
from ai_image_tool import generate_ai_image

# LLM will be initialized in create_content_generation_agent
llm = None
llm_with_tools = None


def generate_ai_image_node(state: ContentAgentState) -> ContentAgentState:
    """
    Generates AI images if ai_image_prompts are present in data_available
    """
    print("\n🎨 AI Image Generation Node...")
    
    data_available = state.get("data_available", {})
    generated_files = state.get("generated_files", [])
    
    # Check ai_image_prompts
    ai_image_prompts = data_available.get("ai_image_prompts", [])
    if not ai_image_prompts or len(ai_image_prompts) == 0:
        print("   ✗ No ai_image_prompts provided - skipping AI image generation")
        if "ai_image_prompts" in data_available:
            del data_available["ai_image_prompts"]
    else:
        print(f"   ✓ Found {len(ai_image_prompts)} ai_image_prompts - generating now...")
        # Generate each AI image
        for idx, prompt_data in enumerate(ai_image_prompts):
            print(f"\n   📸 Generating AI image {idx + 1}/{len(ai_image_prompts)}...")
            
            # Convert prompt_data to JSON string (the tool expects a JSON string)
            prompt_json = json.dumps(prompt_data)
            
            # Call the tool directly
            result = generate_ai_image(prompt_json)
            
            # Extract file path from result
            if "✅" in result or "generated_content/" in result:
                # Parse the file path from the result message
                import re
                file_match = re.search(r'generated_content/[^\s]+\.png', result)
                if file_match:
                    file_path = file_match.group(0)
                    generated_files.append(file_path)
                    print(f"      ✓ Added to generated_files: {file_path}")
                else:
                    print(f"      ⚠️  Could not extract file path from result: {result[:100]}")
            else:
                print(f"      ❌ Generation failed: {result}")
        
        # Remove ai_image_prompts after processing so planning node doesn't try to generate them again
        print(f"\n   ✓ Completed AI image generation, removing from data_available")
        del data_available["ai_image_prompts"]
    
    return {
        **state,
        "data_available": data_available,
        "generated_files": generated_files
    }


def planning_node(state: ContentAgentState) -> ContentAgentState:
    """
    Agent analyzes context and plans what content to generate
    """
    print("\n🤔 Planning content generation...")
    
    context = state["context"]
    data_available = state["data_available"]

    # Check if we have pre-specified materials from materials agent
    chart_specs = data_available.get("chart_specifications", [])
    video_specs = data_available.get("video_specifications", [])
    
    system_prompt = f"""You are a sales content generation specialist. 

Your job is to analyze the meeting context and available data, then create visualizations for the sales meeting.

Available tools:
- generate_market_share_chart: For showing market share distribution (static PNG)
- generate_growth_trend_chart: For showing growth over time (static PNG)
- generate_competitive_matrix: For 2x2 strategic positioning (static PNG)
- generate_swot_analysis: For strengths/weaknesses/opportunities/threats (static PNG)
- generate_financial_comparison: For comparing financial metrics (static PNG)
- generate_video_presentation: For combining static images into a slideshow video
- generate_animated_video: For creating an animated presentation with dynamic charts


General Guidelines:
1. If chart_specifications are provided, create those specific charts
2. If video_specifications are provided, create those specific videos
3. If client_data exists, consider generating a SWOT analysis
4. If industry_data exists, consider generating a market share chart
5. Call tools one at a time with properly formatted JSON data
6. Only use the tools appropriate for the data you have been given
"""
    
    human_prompt = f"""
Meeting Context:
{json.dumps(context, indent=2)}

Available Data:
{json.dumps(data_available, indent=2)}

Generate the visualizations based on the specifications and data provided above.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    
    # Add previous messages to maintain context
    messages.extend(state.get("messages", []))
    
    result = llm_with_tools.invoke(messages)
    
    return {
        **state,
        "messages": [result]
    }


def route_after_planning(state: ContentAgentState) -> Literal["tools", "finalize"]:
    """
    Route to tools if agent wants to use them, otherwise finalize
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the agent wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "finalize"


def finalize_node(state: ContentAgentState) -> ContentAgentState:
    """
    Extract generated file paths and prepare final output
    """
    print("\n📦 Finalizing results...")
    
    # Extract file paths from tool responses
    generated_files = []
    
    for message in state.get("messages", []):
        content = str(message.content) if hasattr(message, "content") else str(message)
        
        # Debug: Print message content
        if "Generated" in content or "generated" in content:
            print(f"   DEBUG - Message: {content[:200]}")
        
        # Look for successful generation messages (both images and videos)
        if "✅ Generated" in content or "✅" in content:
            # Extract file paths - try multiple patterns
            import re
            
            # Pattern 1: generated_content/...
            png_match = re.search(r'generated_content/[^\s]+\.png', content)
            mp4_match = re.search(r'generated_content/[^\s]+\.mp4', content)
            
            # Pattern 2: ai_images/...
            ai_image_match = re.search(r'generated_content/ai_images/[^\s]+\.png', content)
            
            # Pattern 3: Any .png or .mp4 file path
            general_file_match = re.search(r'[^\s]+\.(png|mp4)', content)
            
            if png_match:
                generated_files.append(png_match.group(0))
            elif mp4_match:
                generated_files.append(mp4_match.group(0))
            elif ai_image_match:
                generated_files.append(ai_image_match.group(0))
            elif general_file_match:
                generated_files.append(general_file_match.group(0))
    
    print(f"   Found {len(generated_files)} generated files")
    if generated_files:
        for f in generated_files:
            print(f"      - {f}")
    
    return {
        **state,
        "generated_files": generated_files
    }


def create_content_generation_agent():
    """
    Create a LangGraph agent for content generation
    """
    global llm, llm_with_tools

    
    # Initialize LLM with tools (done here so environment variables are loaded)
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    llm_with_tools = llm.bind_tools(content_generation_tools)
    
    # Create graph
    workflow = StateGraph(ContentAgentState)
    
    # Add nodes
    workflow.add_node("generate_ai_images", generate_ai_image_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("tools", ToolNode(content_generation_tools))
    workflow.add_node("finalize", finalize_node)
    
    # Define flow
    workflow.set_entry_point("generate_ai_images")
    
    # After validation, always go to planning
    workflow.add_edge("generate_ai_images", "planning")

    # Conditional routing after planning
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "tools": "tools",
            "finalize": "finalize"
        }
    )
    
    # After using tools, go back to planning (agent can decide to use more tools or finish)
    workflow.add_edge("tools", "planning")
    
    # Finalize is the end
    workflow.add_edge("finalize", END)
    
    # Compile
    return workflow.compile()


# ============= STANDALONE USAGE =============
if __name__ == "__main__":
    # Create agent
    agent = create_content_generation_agent()

    
    # Example input (this would come from your RAG pipeline)
    input_state = {
        "context": {
            "meeting_type": "initial",
            "client_name": "TechCorp Inc",
            "sales_objectives": ["Establish credibility", "Show market understanding"]
        },
        "data_available": {
            "client_data": {
                "name": "TechCorp Inc",
                "sector": "Technology Services",
                "strengths": ["Strong brand", "Innovation", "Global presence"],
                "weaknesses": ["High costs", "Limited Asian presence"],
                "opportunities": ["Emerging markets", "Digital transformation"],
                "threats": ["Competition", "Regulations"]
            },
            "industry_data": {
                "top_companies": ["TechCorp Inc", "CompetitorA", "CompetitorB", "CompetitorC"],
                "market_shares": [30, 28, 25, 17]
            },
            "has_competitor_data": True,
            "has_financial_data": False,
            "has_historical_data": False,
            "ai_image_prompts": [
                {
                    "prompt": "modern technology office with diverse team collaborating around a conference table, professional corporate environment, natural lighting, high quality"
                },
                {
                    "prompt": "futuristic digital transformation concept with abstract technology network, glowing connections, blue and purple colors, professional business style, 3D illustration"
                },
                {
                    "prompt": "professional business handshake in modern office building lobby, corporate photography style, confident executives, bright natural lighting, glass windows"
                }
            ]
        },
        "messages": [],
        "generated_files": [],
        "errors": []
    }
    
    # Run agent
    print("🚀 Starting Content Generation Agent\n")
    print("=" * 60)
    
    result = agent.invoke(input_state)
    
    print("=" * 60)
    print("\n✅ Agent Complete!\n")
    
    # Show results
    print("📁 Generated Files:")
    for file in result["generated_files"]:
        print(f"   - {file}")