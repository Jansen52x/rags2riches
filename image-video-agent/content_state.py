from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import add_messages

class ContentAgentState(TypedDict):
    """State for the content generation agent only"""
    
    # Input from your pipeline
    context: Dict[str, Any]  # Meeting context, client data, industry data
    data_available: Dict[str, Any]  # What data is available
    
    # Agent's reasoning
    messages: Annotated[list, add_messages]
    
    # Agent's decisions
    content_plan: List[Dict[str, Any]]
    
    # Generated outputs
    generated_files: List[str]
    
    # Metadata
    total_generation_time: float
    errors: List[str]