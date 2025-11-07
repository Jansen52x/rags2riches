import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import asyncio
import psycopg
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
import json
import wikipedia
import wikipediaapi
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
import sys
from ddgs import DDGS

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
    analyzed_claim: Dict
    claim_verdict: Dict
    evidence_log: List[Dict]


# --- 2. Node Functions (The actual work) ---
# (Replace these with your real, imported agent logic)
async def analyze_node(state: FactCheckState) -> Command:
    print("Step 1/4: Analyzing claim...")
    """Analyse the claim and normalise it if needed + Identify sourcing strategy"""

    claim = state["original_claim"]
    client_context = state["client_context"]

    prompt = f"""
    You are a fact-checking assistant helping a salesperson prepare for a client presentation.

    Analyse the following claim carefully. Also ensure that the claims are specific and unambiguous.

    Determine:
    1. Preferred source types (e.g., news, industry reports, social media, review sites)
    2. What kind of information are most relevant (e.g., statistics, expert opinions, case studies)
    3. Approximate number of sources needed to make a verdict

    The goal of this analysis is to facilitate your searching for sources later on, so be as specific and actionable as possible.

    Output results in a similar format as the example below, and as **valid JSON**: 
    e.g. {{
            "claim": "<string>",
            "analysis": {{
                "num_sources_needed": 5,
                "source_types": ["academic", "news", "government"],
                "focus_areas": ["expert opinions", "statistics"]
            }}
    }}

    Important:
    - Do NOT include any Python code or extra text.
    - Do NOT treat the client's context as a claim. It is background information only.
    - Use the client's context only to make your analyses more relevant or specific.

    Claim: "{claim}"
    Client Context (for background only): "{client_context}"
    """
    response = await llm.ainvoke(prompt).content

    try:
        claim = json.loads(response)
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        claim = response

    print(f"Claim analysis complete")

    return Command(
        update={"analyzed_claim": claim}
    )

async def search_claim(state: FactCheckState) -> FactCheckState:
    print("Step 2/4: Starting search for claim: {claim}...")
    """Search for claim with its strategy"""
    analyzed_claim = state['analyzed_claim']
    claim = analyzed_claim['claim']
    strategy = analyzed_claim['analysis']
    
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
    
    response = agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

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
        update={"raw_verdict": final_ai_message_content, "evidence_log": tools_evidence}
    )

async def process_search_result(state: FactCheckState) -> FactCheckState:
    print("Step 3/4: Processing results...")
    """Process the result from search_single_claim and update the claim_verdicts list"""
    raw_verdict = state.get("raw_verdict", {})
    evidence_log = state.get("evidence_log", [])
    original_claim = state.get("original_claim", "Unknown Claim")
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

    response = llm.ainvoke(prompt).content
    try:
        claim_result = json.loads(response)
        
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        claim_result = response

    print("Processed search result for claim.")
    
    return {"claim_verdict": claim_result}

async def save_to_db(state: FactCheckState) -> Command:
    """
    Save the verdict to the database (asynchronously)
    """
    print("Step 4/4: Saving verdict to database (async)...")
    verdict = state.get("claim_verdict")
    salesperson_id = state.get("salesperson_id")
    claim_id = state.get("claim_id")
    
    try: 
        async with await psycopg.AsyncConnection.connect("dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432") as aconn:
            
            original_claim = verdict.get('claim', 'Unknown Claim')
            verdict_str = str(verdict.get('overall_verdict', '')).upper()
            overall_bool = True if verdict_str == 'TRUE' else False
            evidence_json = json.dumps(verdict.get('main_evidence', []))
            pass_to_materials = verdict.get('pass_to_materials_agent', False)
            
            await aconn.execute(
                "INSERT INTO claim_verifications (original_claim, original_claim_id, salesperson_id, overall_verdict, explanation, main_evidence, pass_to_materials_agent) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    original_claim,
                    salesperson_id,
                    claim_id,
                    overall_bool,
                    verdict.get('explanation'),
                    evidence_json,
                    pass_to_materials
                )
            )
            print(f"Verdict saved to database (async) (pass_to_materials_agent: {pass_to_materials})")

    except Exception as e:
        print(f"Error saving verdict to database: {e}")
        return Command(update={"claim_verdict": verdict}, goto=END)
            
    return Command(update={"claim_verdict": verdict}, goto=END)

# --- 3. Progress Steps (Specific to this agent) ---
AGENT_PROGRESS_STEPS = [
    {"value": 50, "text": "Step 2/4: Searching the web to gain evidence and make a verdict..."},
    {"value": 75, "text": "Step 3/4: Processing results..."},
    {"value": 90, "text": "Step 4/4: Saving verdict..."},
    {"value": 100, "text": "Claim verification complete!"}
]

# --- 4. Graph Builder Function ---
def get_fact_check_graph():
    """
    Builds and returns the compiled LangGraph agent.
    """
    graph = StateGraph(FactCheckState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("search", search_claim)
    graph.add_node("process", process_search_result)
    graph.add_node("save", save_to_db)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "search")
    graph.add_edge("search", "process")
    graph.add_edge("process", "save")
    graph.add_edge("save", END)
    
    # Return the compiled, runnable agent
    return graph.compile()


# region TOOLS

# --- Wikipedia Page Name ---
@tool
async def get_wikipedia_page_name(query: str) -> list:
    """
    Get a list of Wikipedia page titles for a given query asynchronously.
    Input: A search query (e.g. Python)
    Output: A list of page titles (e.g. Python (programming language), Pythonidae, Monty Python)
    """
    print("Getting Wikipedia page titles...")
    search_results = await asyncio.to_thread(
        wikipedia.search, query, results=10
    )
    return search_results

# --- Wikipedia Search Summary ---

def _blocking_search_wikipedia(page_title: str) -> str:
    """Internal blocking function for wikipediaapi search."""
    print("Searching Wikipedia (blocking thread)...")
    wiki = wikipediaapi.Wikipedia(
        user_agent='Rags2Riches-Bot/0.0 (locally-run; yongray.teo.2022@scis.smu.edu.sg)', 
        language='en'
    )
    page = wiki.page(page_title)
    if page.exists():
        return page.summary
    else:
        return "Page does not exist."

@tool
async def search_wikipedia(page_title: str) -> str:
    """
    Parse the specified wikipedia page for a summary of the page asynchronously.
    Input: A valid (exact) wikipedia page title
    Output: Summary of the requested Wikipedia page
    """
    print("Searching Wikipedia")
    summary = await asyncio.to_thread(
        _blocking_search_wikipedia, page_title
    )
    return summary

# --- News Articles ---

def _blocking_get_news_articles(query: str) -> list:
    """Internal blocking function for NewsAPI fetch."""
    print("Performing News API search (blocking thread)...")
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        print("Error: NEWS_API_KEY not set.")
        return []

    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        all_articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=10,
            page=1
        )

        truncated_articles = []
        for article in all_articles.get('articles', []):
            truncated_articles.append({
                'source': article.get('source', {}).get('name'),
                'title': article.get('title'),
                'description': article.get('description'),
                'url': article.get('url')
            })
        return truncated_articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

@tool
async def get_news_articles(query: str) -> list:
    """
    Fetch news articles related to the query via NewsAPI asynchronously.
    Input: A search query
    Output: A list of news articles with title, source, description, and URL
    """
    print("Performing News API search...")
    articles = await asyncio.to_thread(
        _blocking_get_news_articles, query
    )
    return articles

# --- DuckDuckGo Search ---

def _blocking_duckduckgo_search(query: str) -> str:
    """Internal blocking function for DuckDuckGo search."""
    print("Performing DuckDuckGo search (blocking thread)...")
    try:
        results = DDGS().text(query, max_results=10)
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return "Error performing DuckDuckGo search."

@tool 
async def duckduckgo_search_text(query: str) -> str:
    """
    Perform an async search on the DuckDuckGo search engine for textual results.
    Input: Search query
    Output: Search results from DDG, with title, href link, and brief body. 
    """
    print("Performing DuckDuckGo search...")
    results = await asyncio.to_thread(
        _blocking_duckduckgo_search, query
    )
    return results

# --- Tavily Search ---

def _blocking_tavily_search(query: str) -> str:
    """Internal blocking function for Tavily search."""
    print("Performing Tavily search (blocking thread)...")
    try:
        search = TavilySearch(max_results=5)
        results = search.run(query)
        return results
    except Exception as e:
        print(f"Tavily search error: {e}")
        return "Error performing Tavily search."

@tool
async def tavily_search(query: str) -> str:
    """
    Use Tavily to search the web asynchronously.
    Input: Search query string
    Output: Search results from Tavily including Tavily's LLM answer and sources
    """
    print("Performing Tavily search...")
    results = await asyncio.to_thread(
        _blocking_tavily_search, query
    )
    return results

# --- RAG System ---

def _blocking_query_rag_system(refined_query: str) -> str:
    """Internal blocking function for RAG query."""
    print(f"Querying RAG system (blocking thread) with: {refined_query}")
    # Mock response
    return "No additional info available from source documents."

@tool
async def query_rag_system(refined_query: str) -> str:
    """
    Queries the RAG system asynchronously.
    Input: A refined or new query
    Output: Response from RAG system.
    """
    print("Querying RAG system...")
    response = await asyncio.to_thread(
        _blocking_query_rag_system, refined_query
    )
    return response

# endregion

llm = ChatOllama(model="llama3.2:3b", temperature=0)
bigLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
tools = [duckduckgo_search_text, tavily_search, search_wikipedia, get_news_articles, query_rag_system] # Agent needs the search tools, the scraper and the RAG query tool
agent = create_agent(bigLM, tools)

