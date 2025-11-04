from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command, Send
import json
import wikipedia
import wikipediaapi
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
import sys
from ddgs import DDGS
import uuid
import psycopg
from operator import add

# Load variables from secrets.env
load_dotenv("secrets.env")

# Ensure project root is importable so we can access materials-agent modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# region LangGraph State
class FactCheckState(TypedDict):
    claim_id: str
    original_claim: str
    salesperson_id: str
    client_context: Optional[str]

    # progressively added fields:
    analyzed_claims: List[Dict]
    all_claim_verdicts: Annotated[list[dict], add]
    # optional handoff status
    materials_status: Optional[str]

# endregion

# region Pre-search
def analyze_node(state: FactCheckState) -> Command:
    """Analyse the claim and break it down/normalise it if needed + Identify sourcing strategy"""

    claim = state["original_claim"]
    client_context = state["client_context"]

    prompt = f"""
    You are a fact-checking assistant helping a salesperson prepare for a client presentation.

    Analyse the following claim carefully. 
    - If appropriate, break it down into sub-claims that can be verified independently.
    - Also ensure that the claims are specific and unambiguous.

    For each claim (main or sub-claim), determine:
    1. Preferred source types (e.g., news, industry reports, social media, review sites)
    2. What kind of information are most relevant (e.g., statistics, expert opinions, case studies)
    3. Approximate number of sources needed to make a verdict

    The goal of this analysis is to facilitate your searching for sources later on, so be as specific and actionable as possible.

    Output results in a similar format as the example below, and as **valid JSON**: 
    e.g. {{
        "claims": [
            {{
                "claim": "<string>",
                "analysis": {{
                    "num_sources_needed": 5,
                    "source_types": ["academic", "news", "government"],
                    "focus_areas": ["expert opinions", "statistics"]
                }}
            }},
            ...
        ]
    }}

    Important:
    - Do NOT include any Python code or extra text.
    - Do NOT treat the client's context as a claim. It is background information only.
    - Use the client's context only to make your analyses more relevant or specific.

    Claim: "{claim}"
    Client Context (for background only): "{client_context}"
    """
    print("Processing claim analysis...")
    response = llm.invoke(prompt).content

    try:
        data = json.loads(response)
        claims_list = data.get("claims", [])
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        claims_list = response

    print(f"Claim analysis complete for claim")

    return Command(
        update={"analyzed_claims": claims_list}
    )

#endregion

# region Search + Evaluate for result
def fan_out_searches(state: FactCheckState) -> Command:
    """Create a search task for each subclaim, allowing for parallel execution"""
    
    claims = state["analyzed_claims"]
    print(f"Fanning out searches for {len(claims)} claims...")

    # Provide 1 search instance per claim and add 1 "fanned out search" to the search count
    return Command(
        goto=[
            Send("search_single_claim", {
                "claim": sc['claim'], 
                "analysis": sc['analysis']
            })
            for sc in claims
        ]
    )

def search_single_claim(state: FactCheckState) -> Command:
    """Search for ONE claim with its strategy"""
    claim = state['claim']
    strategy = state['analysis']

    print(f"Starting search for claim: {claim}")
    
    prompt = f"""
    You are fact-checking this claim: {claim}
    You need:
    - {strategy['num_sources_needed']} credible sources at minimum
    - to prioritise these source types: {', '.join(strategy['source_types'])}
    - to focus on these types of information: {', '.join(strategy['focus_areas'])}
    
    You are given a set of tools that allow you to search the web and find the best sources to fact-check this claim. 
    You MUST use these tools and their output as a base to formulate your claim
    You may also query the RAG system that provided this claim in the first place if the web search does not yield sufficient information
    You may use the tools as many times as needed to make a verdict, or to determine that you cannot make a verdict.
    Make your queries simple so that it is easy to get relevant results, then be more specific if there are too many. 
    Do not call the same tool with the same query as it will definitely give you the same results.
    
    Be sure to evaluate the sources for credibility and relevance.
    
    Provide your verdict with the following information:
    1. Overall Verdict: TRUE, FALSE, or CANNOT BE DETERMINED
    2. Explanation: A concise explanation of how you arrived at the verdict

    If you cannot make a claim based on the sources then just say "CANNOT BE DETERMINED". An absence of evidence does not necessarily mean it's false, so think critically.
    """
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    print(response)

    tools_evidence = []

    messages = response.get('messages')
    
    # --- Part 1: Get the Verdict and Explanation ---
    # Find the *last* AIMessage that has non-empty content
    for msg in reversed(messages):
        # Check type by class name (to avoid import issues)
        if msg.__class__.__name__ == 'AIMessage' and msg.content:
            final_ai_message_content = msg.content
            break

    # --- Part 2: Get the Tools Used (Evidence) ---
    tool_calls = []
    # Use a dict to map responses back to calls by their unique ID
    tool_responses = {} 
    
    for msg in messages:
        # Find the message with the tool *calls*
        if msg.__class__.__name__ == 'AIMessage' and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # msg.tool_calls is a list of dicts
            tool_calls.extend(msg.tool_calls)
        
        # Find the messages with tool *responses*
        if msg.__class__.__name__ == 'ToolMessage':
            # msg.tool_call_id links it to the original call
            tool_responses[msg.tool_call_id] = msg.content
    
    # Now, combine them into a single evidence log
    for call in tool_calls:
        call_id = call.get('id')
        tools_evidence.append({
            "tool_called": call.get('name'),
            "tool_input": call.get('args'),
            "tool_output": tool_responses.get(call_id, "No response found for this call")
        })

    print(f"Search complete for {claim}.")
    return Command(
        goto=Send("process_search_result", {
                "raw_verdict": final_ai_message_content,
                "evidence_log": tools_evidence
            })
    )

def process_search_result(state: FactCheckState) -> Command:
    """Process the result from search_single_claim and update the claim_verdicts list"""
    raw_verdict = state.get("raw_verdict", {})
    evidence_log = state.get("evidence_log", [])
    print("Processing search result")
    prompt = f"""
    Verdict: {raw_verdict}
    Evidence Log: {json.dumps(evidence_log, indent=2)}

    Given this verdict from the agent, determine if the claim should be passed onto a materials generation agent that creates sales presentation materials.
    Typically, false claims should not be passed on, while true claims can be, as you won't want to create materials based on false information.
    However, if you believe certain caveats can be used to present the claim accurately, you may choose to pass it on with appropriate notes.
    At the same time, extract the info in the following JSON format:
    {{
        "overall_verdict": "<TRUE/FALSE/CANNOT BE DETERMINED>",
        "explanation": "<concise explanation>",
        "main_evidence": [
            {{
                "source": "<actual source name or URL>",
                "summary": "<one line summary of the evidence>"
            }},
            ...
        ],
        "pass_to_materials_agent": <true/false>
    }}
    
    Do not provide any other text outside the JSON block. Do not write code.
    """

    response = llm.invoke(prompt).content
    try:
        current_claim_result = json.loads(response)
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        current_claim_result = response

    print("Processed search result for claim.")
    print(current_claim_result)
    return {"all_claim_verdicts": [current_claim_result]}

#endregion

# region terminus
def save_to_db(state: FactCheckState) -> Command:
    """
    Save the final verdicts to the database
    Verdict_data should contain:
        - salesperson_id: str
        - claim_id: str
        - overall_verdict: bool
        - explanation: str
        - main_evidence: list of dicts with 'source' and 'summary'

    """

    print("Saving verdicts to database...")
    verdicts = state["claim_verdicts"]
    salesperson_id = state.get("salesperson_id")
    claim_id = state.get("claim_id")

    # Connect to an existing database
    with psycopg.connect("dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432") as conn:

        # Inserting data
        for verdict in verdicts:
            try: 
                verdict_str = str(verdict.get('overall_verdict', '')).upper()
                overall_bool = True if verdict_str == 'TRUE' else False
                evidence_json = json.dumps(verdict.get('main_evidence', []))
                conn.execute(
                    "INSERT INTO claim_verifications (salesperson_id, claim_id, overall_verdict, explanation, main_evidence) VALUES (%s, %s, %s, %s, %s)",
                    (
                        salesperson_id,
                        claim_id,
                        overall_bool,
                        verdict.get('explanation'),
                        evidence_json
                    )
                )
                print("Verdict saved to database.")

            except Exception as e:
                print(f"Error saving verdict to database: {e}")
                return Command(update={"status": "verdict_failed"}, goto=END)
            
    return Command(update={"status": "verdict_saved"})


def pass_to_materials(state: FactCheckState) -> Command:
    """Trigger the Materials Decision Agent with approved claims and persist decisions.

    Uses a dynamic import to load the integration bridge from the sibling
    'materials-agent' folder (hyphenated, not importable as a package by default).
    """
    import importlib.util
    import os
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    materials_dir = os.path.join(base_dir, "materials-agent")
    bridge_path = os.path.join(materials_dir, "integration_bridge.py")

    approved_claims = [v for v in state.get("claim_verdicts", []) if v.get("pass_to_materials_agent") is True]
    if not approved_claims:
        print("No claims approved for materials generation")
        return Command(goto=END)

    if not os.path.isfile(bridge_path):
        print(f"Materials integration bridge not found at {bridge_path}")
        return Command(goto=END)

    try:
        # Ensure 'materials-agent' directory is on sys.path so that
        # integration_bridge can import sibling modules like materials_decision_agent
        if materials_dir not in sys.path:
            sys.path.append(materials_dir)

        spec = importlib.util.spec_from_file_location("materials_integration_bridge", bridge_path)
        bridge = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(bridge)

        print(f"Passing {len(approved_claims)} verified claims to materials agent...")
        result = bridge.run_complete_pipeline(state)

        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
        print(f"Materials agent result: {status}")

        # Optionally persist summary in state
        update = {
            "materials_agent_status": status,
        }
        if isinstance(result, dict):
            update.update({
                "materials_session_id": result.get("session_id"),
                "materials_recommendations_count": result.get("recommendations_count"),
                "materials_selected_count": result.get("selected_materials_count"),
            })

        return Command(update=update, goto=END)

    except Exception as e:
        print(f"Error invoking materials agent: {e}")
        return Command(update={"materials_agent_status": "error", "materials_error": str(e)}, goto=END)

#endregion

# region TOOLS
@tool 
def get_wikipedia_page_name(query: str) -> list:
    """ Get a list of Wikipedia page titles for a given query. Use this to find the name of relevant wikipedia pages, then use the 'search_wikipedia' tool to get summaries.
        Input: A search query (e.g. Python)
        Output: A list of page titles (e.g. Python (programming language), Pythonidae, Monty Python)
    """
    print("Getting Wikipedia page titles...")
    search_results = wikipedia.search(query, results=10)
    return search_results

@tool
def search_wikipedia(page_title: str) -> str: # Tested independently
    """Parse the specified wikipedia page for a summary of the page 
    Input: A valid (exact) wikipedia page title
    Output: Summary of the requested Wikipedia page
    """
    print("Searching Wikipedia")
    wiki = wikipediaapi.Wikipedia(user_agent='Rags2Riches-Bot/0.0 (locally-run; yongray.teo.2022@scis.smu.edu.sg)', language='en')
    page = wiki.page(page_title)
    if page.exists():
        return page.summary
    else:
        return "Page does not exist."
    
@tool
def get_news_articles(query: str) -> list: # Tested independently
    """ Fetch news articles related to the query via NewsAPI.
    Each call to this tool will return you up to 10 news articles. Do not call it multiple times unless you intend to change the query.
    This tool will only return news articles - it may not be able to find other sources like statistical reports or sentiment.
    Input: A search query
    Output: A list of news articles with title, source, description, and URL
    """
    print("Performing News API search...")
    news_api_key = os.getenv("NEWS_API_KEY")

    # Init
    newsapi = NewsApiClient(api_key=news_api_key)

    # /v2/everything
    all_articles = newsapi.get_everything(q=query,
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=10,
                                        page=1)

    truncated_articles = []

    for article in all_articles['articles']:
        truncated_articles.append({
            'source': article['source']['name'],
            'title': article['title'],
            'description': article['description'],
            'url': article['url']
        })

    return truncated_articles

@tool 
def duckduckgo_search_text(query:str) -> str: # Tested independently
    """ Perform a search on the DuckDuckGo search engine on the web for textual results.
    This might provide you more sources aside from just news articles.
    Input: Search query
    Output: Search results from DDG, with title, href link, and brief body. 
    """
    print("Performing DuckDuckGo search for text...")
    results = DDGS().text(query, max_results=10)
    return results

@tool
def tavily_search(query:str) -> str:
    """Use Tavily to search the web, it will retrieve sources and output an answer as well as the sources used.
    Use sparingly as there are rate limits, and take answer given by Tavily as a guide, not absolute truth. You may
    utilise the sources used to make your own judgement. 

    Input: Search query string
    Output: Search results from Tavily including Tavily's LLM answer and sources
    """
    print("Performing Tavily search...")
    search = TavilySearch(max_results=5)
    results = search.run(query)
    return results

@tool
def query_rag_system(refined_query: str) -> str: # Not yet implemented
    """
    Queries the RAG system if the information on the web is insufficient to make an informed verdict.

    Input: A refined or new query
        
    Output: Response from RAG system.
    """
    print("Querying RAG system...")
    # Mock response that looks real but provides no new info
    return "No additional info available from source documents."

# endregion

# region Setup
# Creating search agent
llm = ChatOllama(model="llama3.2:3b", temperature=0)
# bigLM = ChatOllama(model="qwen3:4b", temperature=0)
bigLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
tools = [duckduckgo_search_text, tavily_search, search_wikipedia, get_news_articles, query_rag_system] # Agent needs the search tools, the scraper and the RAG query tool
agent = create_agent(bigLM, tools)

# Example Claim
claim = "Shopee was leading in market share in 2023"
# claim = "By 2023, Singapore’s e-commerce market reached SGD 9 billion in sales, with Shopee leading in market share, while over 80% of consumers reported shopping online at least once a month."
# client_context = "The client is a small e-commerce startup in Singapore"
client_context = ""

sg_time = datetime.now(ZoneInfo("Asia/Singapore"))
timestamp = sg_time.replace(microsecond=0).isoformat()

initial_state = FactCheckState(
    claim_id= str(uuid.uuid4()),  # randomly generated unique ID for the claim
    original_claim=claim,
    salesperson_id="SP12345",
    client_context=client_context,
    analyzed_claims=[],
    claim_verdicts=[]
)

graph =  StateGraph(FactCheckState)

# add nodes to graph
graph.add_node("analyze", analyze_node)
graph.add_node("search", fan_out_searches)
graph.add_node("search_single_claim", search_single_claim)
graph.add_node("process_search_result", process_search_result)
graph.add_node("save", save_to_db)
graph.add_node("pass_to_materials", pass_to_materials)

# add edges to graph
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "search")
graph.add_edge("search_single_claim", "save")
graph.add_edge("save", "pass_to_materials")
graph.add_edge("pass_to_materials", END)

# compile graph
app = graph.compile()

# invoke with initial state
final_state = app.invoke(initial_state)

#endregion
