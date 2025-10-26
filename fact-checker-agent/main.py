from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict
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
from duckduckgo_search import DDGS
from newspaper import Article
import uuid
import psycopg

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
    search_count: int
    analyzed_claims: List[Dict]
    claim_verdicts: List[Dict]
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

    print("Claim analysis complete.")

    return Command(
        update={"analyzed_claims": claims_list}
    )

#endregion

# region Search + Evaluate for result
def fan_out_searches(state: FactCheckState) -> Command:
    """Run searches sequentially within this node and accumulate verdicts.

    This avoids concurrency/recursion complexity and returns a single update
    with all claim_verdicts populated.
    """
    claims = state.get("analyzed_claims", [])

    gathered: list[dict] = []

    for sc in claims:
        try:
            claim = sc["claim"]
            strategy = sc["analysis"]
        except Exception:
            # Skip malformed entries
            continue

        prompt = f"""
    You are fact-checking this claim: {claim}

    An LLM before you has analyzed this claim and determined that you need:
    - {strategy['num_sources_needed']} credible sources at minimum
    - to prioritise these source types: {', '.join(strategy['source_types'])}
    - to focus on these types of information: {', '.join(strategy['focus_areas'])}
    
    Your job is to formulate an appropriate search query, find the best sources to fact-check this claim, 
    and provide a verdict of the claim's credibility based on the sources found.

    You may continue searching until you feel confident of a verdict, but you may only search up to a maximum of 3 searches. This is a HARD LIMIT.
    
    Make sure you critically evaluate the sources for credibility and relevance.

    You must also decide on the next step to take, by determining if the claim is suitable to be passed to a materials agent that generates presentation materials based on this claim.

    Typically, claims that are FALSE or CANNOT BE DETERMINED should not be passed to the materials agent.
    However, if you believe that the claim can be used with caveats, you may continue to pass it to the materials agent.

    Thereafter, provide all your findings and decision as JSON output in this format:
    {{
        overall_verdict: [TRUE / FALSE / CANNOT BE DETERMINED], 
        explanation: str, 
        main_evidence: list of dicts with 'source' and 'summary', # describing the key sources used to make your verdict, with a 1 line short summary
        pass_to_materials_agent: [TRUE / FALSE]
    }}

    Do not provide any extra text or Python code outside of the JSON output.
    """

        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        try:
            content = response.get("results") or response
            if isinstance(content, str):
                current_result = json.loads(content)
            elif isinstance(content, dict) and "output" in content:
                current_result = json.loads(content["output"])  # heuristic
            else:
                current_result = json.loads(str(content))
        except Exception as e:
            current_result = {
                "overall_verdict": "CANNOT BE DETERMINED",
                "explanation": f"Parse error: {e}",
                "main_evidence": [],
                "pass_to_materials_agent": False,
            }

        print(current_result)
        gathered.append(current_result)
        print(f"Search complete for {claim}.")

    return Command(update={"claim_verdicts": gathered})

def search_single_claim(state: FactCheckState) -> Command:
    """Search for ONE claim with its strategy"""
    claim = state['claim']
    strategy = state['analysis']
    
    prompt = f"""
    You are fact-checking this claim: {claim}

    An LLM before you has analyzed this claim and determined that you need:
    - {strategy['num_sources_needed']} credible sources at minimum
    - to prioritise these source types: {', '.join(strategy['source_types'])}
    - to focus on these types of information: {', '.join(strategy['focus_areas'])}
    
    Your job is to formulate an appropriate search query, find the best sources to fact-check this claim, 
    and provide a verdict of the claim's credibility based on the sources found.

    You may continue searching until you feel confident of a verdict, but you may only search up to a maximum of 3 searches. This is a HARD LIMIT.
    
    Make sure you critically evaluate the sources for credibility and relevance.

    You must also decide on the next step to take, by determining if the claim is suitable to be passed to a materials agent that generates presentation materials based on this claim.

    Typically, claims that are FALSE or CANNOT BE DETERMINED should not be passed to the materials agent.
    However, if you believe that the claim can be used with caveats, you may continue to pass it to the materials agent.

    Thereafter, provide all your findings and decision as JSON output in this format:
    {{
        overall_verdict: [TRUE / FALSE / CANNOT BE DETERMINED], 
        explanation: str, 
        main_evidence: list of dicts with 'source' and 'summary', # describing the key sources used to make your verdict, with a 1 line short summary
        pass_to_materials_agent: [TRUE / FALSE]
    }}

    Do not provide any extra text or Python code outside of the JSON output.
    """
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    print(response)
    # Expect response to include an LLM-generated JSON string
    try:
        content = response.get("results") or response
        if isinstance(content, str):
            current_claim_result = json.loads(content)
        elif isinstance(content, dict) and "output" in content:
            current_claim_result = json.loads(content["output"])  # heuristic
        else:
            # last resort: try to stringify and parse
            current_claim_result = json.loads(str(content))
    except Exception as e:
        current_claim_result = {"overall_verdict": "CANNOT BE DETERMINED", "explanation": f"Parse error: {e}", "main_evidence": [], "pass_to_materials_agent": False}
    print(current_claim_result)
    all_claim_verdicts = list(state.get('claim_verdicts', []))
    all_claim_verdicts.append(current_claim_result)

    print(f"Search complete for {claim}.")

    # If there are still remaining claims, loop back to 'search' to dispatch the next
    has_more = bool(state.get("remaining_claims"))
    if has_more:
        return Command(update={"claim_verdicts": all_claim_verdicts}, goto="search")
    else:
        return Command(update={"claim_verdicts": all_claim_verdicts})

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
def tavily_search_tool(query: str) -> str: # Tested independently
    """Use Tavily to search the web for information. This outputs an answer as well as the sources used.
    Use sparingly as there are rate limits, and take answer given by Tavily as a guide, not absolute truth. You may
    utilise the sources used to make your own judgement. 

    Input: Search query string
    Output: Search results from Tavily including Tavily's LLM answer and sources
    """
    search = TavilySearch(max_results=5)
    results = search.run(query)
    return results

@tool
def search_wikipedia_for_page_title(query: str) -> str: # Tested independently
    """Search Wikipedia for page titles related to the query
    Input: Search query string
    Output: List of relevant Wikipedia page titles
    """
    search_results = wikipedia.search(query)
    if search_results:
        return search_results  # Return the title of the first search result
    else:
        return "No relevant Wikipedia page found."

@tool
def parse_wikipedia_page(page_title: str) -> str: # Tested independently
    """Parse a wikipedia page for the full text
    Input: Valid wikipedia page title
    Output: Full text of the wikipedia page
    """
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(page_title)
    if page.exists():
        return page.text  # Return first 1000 characters for brevity
    else:
        return "Page does not exist."
    

@tool
def get_news_articles(query: str) -> list: # Tested independently
    """ Fetch news articles related to the query using NewsAPI
    Input: A search query
    Output: A list of news articles with title, source, description, and URL
    """
    
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
    """ Perform a DuckDuckGo search for textual results
    Input: Search query
    Output: Search results from DDG, with title, href link, and brief body. 
    """
    results = DDGS().text(query, max_results=10)
    return results

@tool 
def duckduckgo_search_news(query:str) -> str: # Tested independently
    """ Perform a DuckDuckGo search for news
    Input: Search query
    Output: News search results from DDG, with source, date, title, url link, and brief body.
    """
    results = DDGS().news(keywords=query, max_results=10)
    return results

@tool
def scrape_webpage(url: str) -> str: # Tested independently
    """ Scrape the content of a webpage given its URL
    Input: URL of the webpage
    Output: Text content of the webpage
    """

    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error scraping {url}: {e}"

@tool
def query_rag_system(refined_query: str) -> str: # Not yet implemented
    """
    Query the RAG system for claim clarification or additional context.

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
tools = [tavily_search_tool, duckduckgo_search_text, duckduckgo_search_news, search_wikipedia_for_page_title, parse_wikipedia_page, get_news_articles, #Search
         
         ] # Agent needs the search tools, the scraper, the redirect tools etc.
agent = create_agent(llm, tools)

# Example Claim
claim = "By 2023, Singapore’s e-commerce market reached SGD 9 billion in sales, with Shopee leading in market share, while over 80% of consumers reported shopping online at least once a month."
client_context = "The client is a small e-commerce startup in Singapore"

sg_time = datetime.now(ZoneInfo("Asia/Singapore"))
timestamp = sg_time.replace(microsecond=0).isoformat()

initial_state = FactCheckState(
    claim_id= str(uuid.uuid4()),  # randomly generated unique ID for the claim
    original_claim=claim,
    salesperson_id="SP12345",
    client_context=client_context,
    search_count=0,
    analyzed_claims=[],
    claim_verdicts=[]
)

graph =  StateGraph(FactCheckState)

# add nodes to graph
graph.add_node("analyze", analyze_node)
graph.add_node("search", fan_out_searches)
graph.add_node("search_single_claim", search_single_claim)
graph.add_node("save", save_to_db)
graph.add_node("pass_to_materials", pass_to_materials)

# add edges to graph
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "search")
graph.add_edge("search", "save")
graph.add_edge("save", "pass_to_materials")
graph.add_edge("pass_to_materials", END)

# compile graph
app = graph.compile()

# invoke with initial state
final_state = app.invoke(initial_state)

#endregion
