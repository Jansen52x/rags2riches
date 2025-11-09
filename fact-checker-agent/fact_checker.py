from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
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
import psycopg
import uuid

# Load variables from secrets.env
load_dotenv("../secrets.env")

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

# endregion

# region Graph nodes
def analyze_node(state: FactCheckState) -> Command:
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
    print("Processing claim analysis...")
    response = llm.invoke(prompt).content

    try:
        claim = json.loads(response)
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        claim = response

    print(f"Claim analysis complete for claim")

    return Command(
        update={"analyzed_claim": claim}
    )

def search_claim(state: FactCheckState) -> Command:
    """Search for claim with its strategy"""
    analyzed_claim = state['analyzed_claim']
    claim = analyzed_claim['claim']
    strategy = analyzed_claim['analysis']

    print(f"Starting search for claim: {claim}")
    
    prompt = f"""
    You are fact-checking this claim: {claim}

    REQUIREMENTS:
    - Find at least {strategy.get('num_sources_needed', 3)} credible sources
    - Prioritize these source types: {', '.join(strategy.get('source_types', ['news', 'academic', 'government']))}
    - Focus on: {', '.join(strategy.get('focus_areas', ['accuracy', 'context']))}

    TOOLS:
    - Use the web search tools to find sources
    - Start with simple, broad queries, then refine if needed
    - Do NOT repeat identical queries for the same tool (same input = same output)
    - Stop after 5-7 unique searches if you haven't found sufficient reliable information

    EVALUATION CRITERIA:
    - Assess source credibility (authoritative, recent, primary when possible)
    - Look for corroboration across multiple independent sources
    - If sources conflict, note this and weigh by credibility
    - Absence of evidence ≠ evidence of falseness (think critically about what would be documented)


    VERDICT RULES:
    - TRUE: Multiple credible sources confirm the claim
    - FALSE: Credible sources clearly contradict the claim
    - CANNOT BE DETERMINED: Insufficient evidence, conflicting reliable sources, or absence of information

    OUTPUT FORMAT:
    1. Overall Verdict: TRUE, FALSE, or CANNOT BE DETERMINED
    2. Explanation: A concise explanation of how you arrived at the verdict    
    """
    
    response = agent.invoke(
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

def process_search_result(state: FactCheckState) -> Command:
    """Process the result from search_single_claim and update the claim_verdicts list"""
    raw_verdict = state.get("raw_verdict", {})
    evidence_log = state.get("evidence_log", [])
    original_claim = state.get("original_claim", "Unknown Claim")
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
        claim_result = json.loads(response)
        
    except json.JSONDecodeError:
        # fallback in case the LLM outputs plain text instead of valid JSON
        claim_result = response

    print("Processed search result for claim.")
    
    return {"claim_verdict": claim_result}

# def save_to_db(state: FactCheckState) -> Command:
#     """
#     Save the verdict to the database
#     Verdict_data should contain:
#         - salesperson_id: str
#         - claim_id: str
#         - overall_verdict: bool
#         - explanation: str
#         - main_evidence: list of dicts with 'source' and 'summary'
#         - pass_to_materials_agent: bool

#     """

#     print("Saving verdict to database...")
#     verdict = state.get("claim_verdict")
#     salesperson_id = state.get("salesperson_id")
#     claim_id = state.get("claim_id")

#     # Connect to an existing database
#     with psycopg.connect("dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432") as conn:

#         # Inserting data
#         try: 
#             original_claim = verdict.get('claim', 'Unknown Claim')
#             verdict_str = str(verdict.get('overall_verdict', '')).upper()
#             overall_bool = True if verdict_str == 'TRUE' else False
#             evidence_json = json.dumps(verdict.get('main_evidence', []))
#             pass_to_materials = verdict.get('pass_to_materials_agent', False)
            
#             conn.execute(
#                 "INSERT INTO claim_verifications (original_claim, original_claim_id, salesperson_id, overall_verdict, explanation, main_evidence, pass_to_materials_agent) VALUES (%s, %s, %s, %s, %s, %s, %s)",
#                 (
#                     original_claim,
#                     salesperson_id,
#                     claim_id,
#                     overall_bool,
#                     verdict.get('explanation'),
#                     evidence_json,
#                     pass_to_materials
#                 )
#             )
#             print(f"Verdict saved to database (pass_to_materials_agent: {pass_to_materials})")

#         except Exception as e:
#             print(f"Error saving verdict to database: {e}")
#             return Command(update=state, goto=END)
            
#     return Command(update=state, goto=END)


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
bigLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
# tools = [duckduckgo_search_text, tavily_search, search_wikipedia, get_news_articles, query_rag_system] # Agent needs the search tools, the scraper and the RAG query tool
tools = [query_rag_system] # Agent needs the search tools, the scraper and the RAG query tool
agent = create_agent(bigLM, tools)

graph = StateGraph(FactCheckState)
graph.add_node("analyze", analyze_node)
graph.add_node("search", search_claim)
graph.add_node("process", process_search_result)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "search")
graph.add_edge("search", "process")
graph.add_edge("process", END)

app = graph.compile()