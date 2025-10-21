from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command

class FactCheckState(TypedDict):
    original_claim: str
    salesperson_id: str
    client_context: Optional[str]
    timestamp: str

    # progressively added fields:
    analyzed_claims: List[Dict]
    source_strategy: Dict
    queries: List
    raw_sources: List
    contexts: List
    numeric_report: Optional[Dict]
    source_stances: List
    credibility_scores: List
    conflict_report: Dict
    final_verdict: Dict

def claim_analyzer(state: FactCheckState) -> Command:
    """Analyse the claim and break it down"""

    claim = state["original_claim"]
    client_context = state["client_context"]

    prompt = f"""
    You must analyse the claim given for a salesperson who is intending to present this data to a client.

    If necessary, breakdown the claim into sub-claims that are easier to verify independently. Also, normalize the claim to be concise and unambiguous.


    Then for each claim (main claim or sub-claim), perform the following analysis: 
    - Determine how verifiable the claim is in principle (e.g. the claim is objective and is verifiable in principle),
    - Determine if verifying the claim requires access to internal company data (e.g. the claim does not require internal data),
    - Estimate the complexity of verifying the claim in terms of effort required (e.g. the claim requires a lot of verification effort as it is complex/nuanced),
    - Establish how recent the data needs to be to verify the claim (e.g. recent data is needed).

    After this, output the results as a Python list with the following structure:
    claims = [
        {{"claim": str, "analysis": str}} # The analysis is a string summarising step 3's results 
        ... other claims
    ]

    Claim: "{claim}"
    Client's Context (if any): "{client_context}"
    """

    response = llm.invoke(prompt).content

    return Command(
        update={"analyzed_claims": response}
    )

@tool
def source_strategy_planner(state: FactCheckState) -> FactCheckState:
    """Plan a source strategy based on the verification plan - meaning the type of sources to utilise, how many sources etc."""

    analyzed_claims = state["analyzed_claims"]

    prompt = f"""
    You are given a list of claims. For each claim, there's an analysis that outlining the effort needed to verify the claim.
    Based on this analysis, outline a strategy for sourcing evidence to verify the claims by determining the following:
    - Preferred source types (e.g. news, industry reports, social media, review sites)
    - How many sources are roughly needed to make a verdict on each claim
    - Are we searching for a variety of sources that both support and refute the claim, or is finding reliable sources that have a consensus good enough?
    - What constitutes a reliable source? (e.g. which type of news outlets, which platforms etc.)

    Claims: {analyzed_claims}

    Output the results as a Python list with the following structure:
    source_strategy = [
        {"claim": str, "strategy": str} # The strategy is a string summarising the source strategy for the claim
    ]
    """

    response = llm.invoke(prompt).content

    state["source_strategy"] = response
    return state

@tool
def query_formatter(state: FactCheckState) -> FactCheckState:
    """Format search queries based on the source strategy"""

    source_strategy = state["source_strategy"]

    prompt = f"""
    You are given a list of claims along with a source strategy for each claim.
    Based on this, generate specific search queries that can be used to find relevant sources to verify each claim.
    Search queries should be concise, unambiguous, and optimized for search engines.
    The content of the search query should reflect the claim and the source strategy, e.g. 2024, latest, data@gov, Singapore etc.

    Source Strategy: {source_strategy}

    Output the results as a Python list with the following structure:
    queries = [
        {"claim": str, "query": str}
    ]
    """

    response = llm.invoke(prompt).content

    state["queries"] = response
    return state

@tool
def search(state: FactCheckState) -> FactCheckState:
    """Search for sources based on the formatted queries"""

    system_prompt = f"""
    Your job is to find the best sources for this query.
    
    Budget remaining: ${state['budget_remaining']}
    Query: {state['query']}
    
    Strategy:
    1. Choose the most appropriate search tool(s) based on the query type and budget
    2. Analyze initial results
    3. Decide if you need to fetch full content from promising URLs
    4. You can call multiple tools in sequence
    
    Goal: Gather 5-10 high-quality sources efficiently."""

    queries = state["queries"]
    search_tool = TavilySearch(max_results=5)

    all_sources = []
    for item in queries:
        query = item["query"]
        search_results = search_tool.invoke(query)
        all_sources.append({"claim": item["claim"], "sources": search_results})

    state["raw_sources"] = all_sources
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
graph.add_edge(START, "claim_analyzer")
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