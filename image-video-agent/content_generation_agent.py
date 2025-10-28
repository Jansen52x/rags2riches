from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from typing import Literal
import json
import os

from content_state import ContentAgentState
from content_tools import content_generation_tools

# LLM will be initialized in create_content_generation_agent
llm = None
llm_with_tools = None


def planning_node(state: ContentAgentState) -> ContentAgentState:
    """
    Agent analyzes context and plans what content to generate
    """
    print("\n🤔 Planning content generation...")
    
    context = state["context"]
    data_available = state["data_available"]
    
    system_prompt = """You are a sales content generation specialist. 
    
Your job is to analyze the meeting context and available data, then decide 
what visualizations would be most effective for the sales meeting.

Available tools:
- generate_market_share_chart: For showing market share distribution (static PNG)
- generate_growth_trend_chart: For showing growth over time (static PNG)
- generate_competitive_matrix: For 2x2 strategic positioning (static PNG)
- generate_swot_analysis: For strengths/weaknesses/opportunities/threats (static PNG)
- generate_financial_comparison: For comparing financial metrics (static PNG)
- generate_video_presentation: For combining static images into a slideshow video
- generate_animated_video: For creating an animated presentation with dynamic charts
  (bars growing, lines drawing, smooth transitions) - RECOMMENDED for presentations

Based on the context, decide which visualizations to create and call the 
appropriate tools with the correct data.

Guidelines:
1. Prioritize high-impact visuals that tell a clear story
2. For static analysis: Create 2-3 individual charts
3. For presentations: Use generate_animated_video to create an engaging animated video
4. Use the data that's actually available
5. Match visualization type to the data and meeting goals
6. Call tools one at a time with properly formatted JSON data
7. When creating animated videos, pass the chart specifications in the "sections" array

Think step by step about what would be most valuable for this meeting.
"""
    
    human_prompt = f"""
Meeting Context:
{json.dumps(context, indent=2)}

Available Data:
{json.dumps(data_available, indent=2)}

What visualizations should we create? Call the appropriate tools with the data.
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
        
        # Look for successful generation messages (both images and videos)
        if "✅ Generated" in content:
            # Extract file paths for both PNG and MP4 files
            import re
            png_match = re.search(r'generated_content/[^\s]+\.png', content)
            mp4_match = re.search(r'generated_content/[^\s]+\.mp4', content)
            
            if png_match:
                generated_files.append(png_match.group(0))
            if mp4_match:
                generated_files.append(mp4_match.group(0))
    
    print(f"   Found {len(generated_files)} generated files")
    
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
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    llm_with_tools = llm.bind_tools(content_generation_tools)
    
    # Create graph
    workflow = StateGraph(ContentAgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("tools", ToolNode(content_generation_tools))
    workflow.add_node("finalize", finalize_node)
    
    # Define flow
    workflow.set_entry_point("planning")
    
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
            "has_historical_data": False
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