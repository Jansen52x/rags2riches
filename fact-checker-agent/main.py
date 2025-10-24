from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict, List, Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command, Parallel
from langgraph.constants import Send
import json
import wikipedia
import wikipediaapi
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
from ddgs import DDGS
import newspaper
import uuid
import psycopg

# Load variables from secrets.env
load_dotenv("secrets.env")

# region LangGraph State
class FactCheckState(TypedDict):
    claim_id: str
    original_claim: str
    salesperson_id: str
    client_context: Optional[str]

    # progressively added fields:
    analyzed_claims: List[Dict]
    claim_verdicts: List[Dict]

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
    """Create a search task for each subclaim, allowing for parallel execution"""
    
    claims = state["analyzed_claims"]

    # Provide 1 search instance per claim and add 1 "fanned out search" to the search count
    return Parallel([
        Command(Send("search_single_claim", {"claim": sc['claim'], "analysis": sc['analysis']}))
        for sc in claims
    ])

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
    current_claim_result = json.load(response["results"])
    print(current_claim_result)
    all_claim_verdicts = state['claim_verdicts']
    all_claim_verdicts.append(current_claim_result)

    print(f"Search complete for {claim}.")

    return Command(
        update={"claim_verdicts": all_claim_verdicts}
    )

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
     # Connect to an existing database
    with psycopg.connect("dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432") as conn:

        # Inserting data
        for verdict in verdicts:
            try: 
                conn.execute(
                    "INSERT INTO claim_verifications (salesperson_id, claim_id, overall_verdict, explanation, main_evidence) VALUES (%s, %s, %s, %s, %s)",
                    (
                        verdict['salesperson_id'],
                        verdict['claim_id'],
                        verdict['overall_verdict'],
                        verdict['explanation'],
                        verdict['main_evidence']
                    )
                )
                print("Verdict saved to database.")

            except Exception as e:
                print(f"Error saving verdict to database: {e}")
                return Command(update={"status": "verdict_failed"}, goto=END)
            
    return Command(update={"status": "verdict_saved"})

def pass_to_materials(state: FactCheckState) -> Command:
    verdicts = state["claim_verdicts"] # List of verdict dictionaries
    for verdict in verdicts:
        if verdict["pass_to_materials_agent"] == True:
            #pass to materials
            pass

    return Command(goto=END)

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

    article = newspaper.article(url)
    return article.text

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
