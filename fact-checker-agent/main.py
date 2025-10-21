from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing import TypedDict, List, Optional, Dict
import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, END

class FactCheckState(TypedDict):
    original_claim: str
    salesperson_id: str
    client_context: Optional[str]
    timestamp: str

    # progressively added fields:
    verification_plan: Dict
    source_strategy: Dict
    queries: List
    raw_sources: List
    contexts: List
    numeric_report: Optional[Dict]
    source_stances: List
    credibility_scores: List
    conflict_report: Dict
    final_verdict: Dict

@tool
def claim_analyzer(state: FactCheckState) -> FactCheckState:
    """Analyse the claim and break it down"""

    claim = state["original_claim"]
    client_context = state["client_context"]

    prompt = f"""
    You must analyse the claim given for a salesperson who is intending to present this data to a client.
    Provide the following if necessary:
    1. A normalised version of the claim that is concise and unambiguous.
    2. A breakdown of the claim into sub-claims that are easier to verify by themselves.

    Provide the following always:
    3. For each sub-claim (if any) or the main claim if there aren't subclaims, 
    - Determine how verifiable the claim is in principle (e.g. the claim is objective and is verifiable in principle),
    - Determine if verifying the claim requires access to internal company data (e.g. the claim does not require internal data),
    - Estimate the complexity of verifying the claim in terms of effort required (e.g. the claim requires a lot of verification effort as it is complex/nuanced),
    - Establish how recent the data needs to be to verify the claim (e.g. recent data is needed).

    After this, output the results as a Python dictionary with the following structure:
    (note step 3's results are a concise string to pass to other LLMs)
    claims = [
        {"claim": str, "analysis": str} # The analysis is a string summarising step 3's results 
        ... other claims
    ]

    Claim: "{claim}"
    Client's Context (if any): "{client_context}"
    """


    response = llm.invoke(prompt).content

    # You can safely parse via json.loads() if you expect structured output
    state["verification_plan"] = response
    return state

@tool
def source_strategy_planner(state: FactCheckState) -> FactCheckState:
    """Plan a source strategy based on the verification plan - meaning the type of sources to utilise, how many sources etc."""

    verification_plan = state["verification_plan"]

    prompt = f"""
    You are given a verification plan that outlines the difficulty of verifying a claim, how recent the data used needs to be, and the claims you need to verify.
    Based on that verification plan, outline a source strategy to find relevant data sources to verify the claim.

    Verification Plan: {verification_plan}

    Output the source strategy as a Python dictionary with the following structure:
    {

        "preferred_sources": list[str],  # e.g. ["news", "industry reports", "social media", "review sites"]
        "min_num_sources": int,
        "diversity_required": bool,
        "reliability_criteria": list[str]
    }
    """

    response = llm.invoke(prompt).content

    state["source_strategy"] = response
    return state


llm = ChatOllama(model="llama3.2:3b", temperature=0)


# Example claim to fact-check
claim = "Singapore's digital payments market grew by 25% in 2023, reaching a total transaction value of SGD 150 billion."
client_context = "The client is a small fintech startup in Singapore"

sg_time = datetime.now(ZoneInfo("Asia/Singapore"))
timestamp = sg_time.replace(microsecond=0).isoformat()

initial_state = FactCheckState(
    original_claim=claim,
    salesperson_id="SP12345",
    client_context=client_context,
    timestamp=timestamp,
    verification_plan={},
    source_strategy={},
    queries=[],
    raw_sources=[],
    contexts=[],
    numeric_report=None,
    source_stances=[],
    credibility_scores=[],
    conflict_report={},
    final_verdict={}
)

graph =  StateGraph(FactCheckState)

# add tools/nodes
graph.add_node("claim_analyzer", claim_analyzer)
# add more nodes as needed...
graph.add_edge("claim_analyzer", END)  # for this example, only one node

# compile graph
app = graph.compile()

# invoke with initial state
final_state = app.invoke(initial_state)

#---------------------------------------------------------
# # Initialize the Tavily Web Search Tool
# search_tool = TavilySearch(max_results=5)

# # Initialize the language model
# llm = ChatOllama(model="llama3.2:3b", temperature=0)

# # Define the claim verification function as a tool
# @tool
# def verify_claim(claim: str, evidence: str) -> str:
#     """Decides if evidence supports or refutes a claim."""
#     prompt = f"""
#     Claim: "{claim}"
#     Evidence: "{evidence}"
    
#     Decide if the evidence SUPPORTS, REFUTES, or is INSUFFICIENT to judge the claim.
#     Respond with one word: SUPPORT / REFUTE / INSUFFICIENT.
#     """
#     response = llm.invoke(prompt)
#     return response.content
# tools = [search_tool, verify_claim]
# agent = create_agent(llm, tools)
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": claim}]}
# )