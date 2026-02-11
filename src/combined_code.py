import asyncio
import uuid
import networkx as nx
import matplotlib.pyplot as plt
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel
import argparse
import csv
import sys


from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field


SYSTEM_PROMPT = """
You are a web-browsing agent that maps company relationship data.

Your job, given a single company name, is to:
1. Find that company's official website (or main site).
2. Use the browser tools to navigate to pages that likely list partners, clients, vendors, customers, or integrations.
3. Extract relationship statements between the original company and other companies.
4. Return ONLY JSON with this exact structure:

{
  "relationships": [
    {
      "source": "<original_company_name>",
      "relationship": "<one of: partner_of, client_of, vendor_of, reseller_of, integrator_of, uses>",
      "target": "<other_company_name>",
      "reason": "<summary of the evidence of their relationship>",
      "references": [
        {
          "url" : "<current url>",
          "text": "<one or a few quotes/sentences from the url page that explicitly/implicitly show the relationship>"
      }
      ]
    }
  ],
  "new_companies": ["<other_company_name_1>", "<other_company_name_2>", "..."]
}

Rules:
- Always treat the company passed in the user message as the "source" for relationships (unless it is clearly acting as the client of another vendor).
- Infer relationships from context when needed, e.g. "system integrator", "case study with", "trusted partner", "client list".
- Ignore individuals and non-company entities.
- If you find nothing, return:
  {"relationships": [], "new_companies": []}
- DO NOT include any explanation or text outside the JSON.
- When interacting with the page (e.g., clicking or filling forms),
  if targeting text on a button, you **MUST use Playwright's text selector syntax**,
  like this: `text=Button Text`. DO NOT use `:contains()`.
  DO NOT GO TO .pdf webpages or other download required pages, this will crash the browser
""".strip()

link_selector_prompt = """
You are helping select links that are likely to contain information
about company relationships with other companies:
- partners
- clients/customers
- vendors/suppliers
- integrations
- resellers
- system integrators

Run a search for a relevant query using the search tool. Then narrow it down
to a list of only relevant links.

Return a JSON list of links, e.g.:
["apple.com/partenrs", "google.com/customers/case-studies/acme"]

Page context (short summary of current page):
{page_context}

Links (JSON):
{links_json}
""".strip()

BACKGROUND_PROMPT = """
You are a Lead Market Research Strategist. 
Your goal is to understand a company's business model to determine how to best find their B2B relationships (partners, clients, suppliers).

1. Analyze the search results provided about the target company.
2. Summarize: What do they do? (e.g., "SaaS CRM platform", "Industrial Manufacturer", "Logistics Provider").
3. Generate 3-5 HIGHLY SPECIFIC Google search queries to find their business relationships.

Strategy:
- If they are SaaS, look for: "integrations", "marketplace", "technology partners".
- If they are an Agency, look for: "client list", "our work", "case studies".
- If they are Manufacturing, look for: "distributors", "suppliers", "vendor portal".

Return a JSON with the summary and the list of queries.
"""

class linkGetterResponse(BaseModel):
  links: List[str]
  page_context: str

class Reference(BaseModel):
    url: str
    text: str

class Relationship(BaseModel):
    source: str
    relationship: str
    target: str
    reason: str
    references: list[Reference]

class websiteCrawlerResponse(BaseModel):
  relationships: List[Relationship]
  new_companies: List[str]

class AgentState(TypedDict):
    target: str
    messages: List[BaseMessage]
    search_result: Optional[Dict[str, Any]]
    selected_links: List[str]
    extracted_relationships: List[Relationship]
    discovered_companies: List[str]
    companiestoexplore: List[str]
    background_summary: Optional[str]
    generated_queries: List[str]

class BackgroundStrategy(BaseModel):
    summary: str = Field(description="A brief 1-sentence summary of what the company does.")
    search_queries: List[str] = Field(description="A list of 3-5 Google search queries optimized to find partners/clients for this specific company type.")

def load_companies_from_file(filepath: str) -> List[str]:
    companies = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.csv'):
                reader = csv.reader(f)
                # Assumes the company name is in the first column
                for row in reader:
                    if row: 
                        companies.append(row[0].strip())
            else:
                # Assumes one company per line for .txt
                companies = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
    return companies

def save_relationships_to_csv(relationships: List[Relationship], filename: str):
    """Saves the extracted relationships to a CSV file."""
    if not relationships:
        print("No relationships to save.")
        return

    headers = ["Source", "Relationship", "Target", "Reason", "Evidence URL", "Evidence Text"]
    
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for rel in relationships:
                # Handle potentially missing or list-based references
                ref_url = "N/A"
                ref_text = "N/A"
                
                # Check if references exist and extract the first one
                if rel.references and isinstance(rel.references, list) and len(rel.references) > 0:
                    first_ref = rel.references[0]
                    # Handle both object (Pydantic) and dictionary access just in case
                    if isinstance(first_ref, dict):
                        ref_url = first_ref.get("url", "N/A")
                        ref_text = first_ref.get("text", "N/A")
                    else:
                        ref_url = getattr(first_ref, "url", "N/A")
                        ref_text = getattr(first_ref, "text", "N/A")

                writer.writerow([
                    rel.source, 
                    rel.relationship, 
                    rel.target, 
                    rel.reason, 
                    ref_url, 
                    ref_text
                ])
        print(f"\nSuccessfully saved {len(relationships)} relationships to '{filename}'")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

master_relationships = []
master_discovered_companies = set()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()

        serper = GoogleSerperAPIWrapper(k=100)

        @tool("Search_results")
        def search(query: str, page: int) -> str:
            """useful for when you need to look up links with search"""
            return serper.results(query, page=page)
        
        searchtool = [search]


        llm = init_chat_model("gpt-5.2", model_provider="openai", max_retries=50).bind_tools(tools)

        linkGetterllm = create_agent(model=llm, tools=searchtool, system_prompt=link_selector_prompt, response_format=linkGetterResponse).with_config({"recursion_limit": 50})
        websitecrawleragent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, response_format=websiteCrawlerResponse).with_config({"recursion_limit": 50})
        background_agent = create_agent(
            model=llm, 
            tools=searchtool, 
            system_prompt=BACKGROUND_PROMPT, 
            response_format=BackgroundStrategy
        )

        async def BackgroundInfoandqueries(state: AgentState, config: RunnableConfig):
            company_name = state["target"]

            homepage_text = "Not available (Scraping failed or skipped)"
            top_link = None
            snippets = []
            combined_snippets = ""
            search_query = f"{company_name} official site / wikipedia" 
            try:
                search_results_json = serper.results(search_query)
                if "organic" in search_results_json:
                    for result in search_results_json["organic"]:
                        snippets.append(f"{result.get('title')}: {result.get('snippet')}")
                        if not top_link: 
                            top_link = result.get("link")
                    
                    combined_snippets = "\n".join(snippets[:5])
            except Exception as e:
                print(f"   > Search failed: {e}")

            if top_link:
                print(f"   > Identified homepage: {top_link}")
                page = None
                try:
                    page = await browser.new_page()
                    await page.goto(top_link, timeout=15000, wait_until="domcontentloaded")
                    
                    body_text = await page.evaluate("document.body.innerText")
                    homepage_text = " ".join(body_text.split())[:5000]
                    print(f"   > Scraped {len(homepage_text)} chars from homepage.")
                    
                except Exception as e:
                    print(f"   > Manual scrape failed: {e}")
                finally:
                    if page: await page.close()

            print(f"   > Generating search strategy...")
            try:
                context_prompt = f"""
                TARGET: {company_name}
                
                SEARCH SNIPPETS:
                {combined_snippets}
                
                HOMEPAGE CONTENT (URL: {top_link}):
                {homepage_text[:2000]} 
                """
                
                result = await background_agent.ainvoke(
                    {"messages": [HumanMessage(content=context_prompt)]},
                    config=config
                )
                strategy = result["structured_response"]
                
                print(f"   > Queries: {strategy.search_queries}")
                
                return {
                    "background_summary": strategy.summary,
                    "generated_queries": strategy.search_queries
                }

            except Exception as e:
                print(f"   > Error in strategy generation: {e}")
                return {
                    "generated_queries": [
                        f"{company_name} partners", 
                        f"{company_name} client list",
                        f"{company_name} integrations",
                        f"{company_name} vendors / suppliers"
                    ]
                }

            

        async def linkGetterNode(state: AgentState, config: RunnableConfig):
            company_name = state["target"]

            queries = state.get("generated_queries", [])
            all_search_results = []

            try:
                for q in queries: 
                    search_results = await searchtool[0].ainvoke({"query": q, "page": 0})
                    all_search_results.append(str(search_results))
            except Exception as e:
                search_results = f"Search failed: {e}"
            
            print(f"Selecting links for {company_name}...")
            combined_search_context = "\n\n".join(all_search_results)
            
            try:
                prompt_content = f"""
                Here are search results for company: {company_name}:
                {combined_search_context}
                Based on the text above, extract a list of relevant links.
                """

                # Pass config to maintain thread_id
                selected_links_response = await linkGetterllm.ainvoke({
                    "messages": [HumanMessage(content=prompt_content)]
                }, config=config)
                
                data = selected_links_response["structured_response"]
                selected_links = data.links
            except Exception as e:
                print(f"LLM Error in linkGetter: {e}")
                data = []
                selected_links = []

            print(f"Selected Links: {selected_links}")

            return {
                "target": company_name,
                "selected_links": selected_links,
                "extracted_relationships": state.get("extracted_relationships", []),
                "discovered_companies": state.get("discovered_companies", []),
                "companiestoexplore": state.get("companiestoexplore", [])
            }

        async def websiteCrawlerNode(state: AgentState, config: RunnableConfig):
            company_name = state["target"]
            selected_links = state["selected_links"]
            discovered_companies = state["discovered_companies"]
            companiestoexplore = state["companiestoexplore"]
            extracted_relationships = state["extracted_relationships"]

            if not selected_links:
                print(f"No links left to crawl for {company_name}")
                return {
                    "target": company_name,
                    "selected_links": [],
                    "extracted_relationships": extracted_relationships,
                    "discovered_companies": discovered_companies,
                    "companiestoexplore": companiestoexplore,
                }

            nextlink = selected_links.pop(0)
            print(f"Crawling: {nextlink}")

            existing_rels_str = str(extracted_relationships) if extracted_relationships else "None"
            existing_companies_str = str(discovered_companies) if discovered_companies else "None"

            prompt_content = f"""
            You are an expert market researcher analyzing a webpage for business intelligence.
            ### TARGET CONTEXT
            Target Company: {company_name}
            URL to Investigate: {nextlink}

            ### CURRENT KNOWLEDGE BASE
            We have ALREADY found the following information:
            - Known Relationships: {existing_rels_str}
            - Discovered Companies: {existing_companies_str}

            ### YOUR MISSION
            1. Access the URL and scan for **NEW** business relationships.
            2. Ignore generic footer links.
            
            ### OUTPUT
            Return a structured list of NEW relationships found on this page.
            """

            try:
                result = await websitecrawleragent.ainvoke({"messages" : [HumanMessage(prompt_content)]}, config=config)
                data = result["structured_response"]
                new_relationships = data.relationships
                new_companies = data.new_companies
                
                extracted_relationships.extend(new_relationships)
                for company in new_companies:
                    if company not in discovered_companies:
                        discovered_companies.append(company)
                        companiestoexplore.append(company)

                print(f"Extracted {len(new_relationships)} new relationships.")

            except Exception as e:
                print(f"LLM Error in Crawler: {e}")

            return {
                "target": company_name,
                "selected_links": selected_links,
                "extracted_relationships": extracted_relationships,
                "discovered_companies": discovered_companies,
                "companiestoexplore": companiestoexplore,
            }

        def should_continue(state: AgentState):
            if state["selected_links"] and len(state["selected_links"]) > 0:
                return "continue"
            return "end"


        workflow = StateGraph(AgentState)
        workflow.add_node("background", BackgroundInfoandqueries)
        workflow.add_node("link_getter", linkGetterNode)
        workflow.add_node("website_crawler", websiteCrawlerNode)
        workflow.set_entry_point("background")
        workflow.add_edge("background", "link_getter")
        workflow.add_edge("link_getter", "website_crawler")
        workflow.add_conditional_edges(
            "website_crawler",
            should_continue,
            {
                "continue": "website_crawler",
                "end": END
            }
        )
        
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)

        parser = argparse.ArgumentParser(description="Company Relationship Crawler")
        
        parser.add_argument(
            "companies", 
            nargs="*", 
            help="List of companies to crawl (e.g., 'Apple' 'Microsoft')"
        )
        
        parser.add_argument(
            "-f", "--file", 
            help="Path to a .txt or .csv file containing a list of companies"
        )
        parser.add_argument("-o", "--output", default="output.csv", help="Filename for the output CSV (default: output.csv)")

        args = parser.parse_args()
        
        target_companies = []

        if args.file:
            print(f"Loading companies from file: {args.file}")
            target_companies.extend(load_companies_from_file(args.file))

        if args.companies:
            target_companies.extend(args.companies)

        if not target_companies:
            print("No companies provided via arguments or file.")
            print("Usage examples:")
            print("  python script.py \"Home Depot\" \"Lowe's\"")
            print("  python script.py -f companies.txt")
            return
        
        target_companies = list(set(target_companies))


        print(f"Starting batch process for: {target_companies}\n" + "="*50)

        for company in target_companies:
            print(f"\n>>> PROCESSING TARGET: {company}")
            initial_state = {
                "target": company,
                "messages": [],
                "background_summary": None,   
                "generated_queries": [],      
                "selected_links": [],
                "extracted_relationships": [],
                "discovered_companies": [],
                "companiestoexplore": []
            }

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

            try:
                final_state = await app.ainvoke(initial_state, config)
            except GraphRecursionError:
                print(f"Graph recursion limit reached for {company}")
                snapshot = await app.aget_state(config)
                if snapshot and snapshot.values:
                    final_state = snapshot.values
                else:
                    final_state = initial_state
            except Exception as e:
                print(f"An error occurred for {company}: {e}")
                final_state = initial_state

            current_relationships = final_state.get('extracted_relationships', [])
            master_relationships.extend(current_relationships)

            current_discovered = final_state.get('discovered_companies', [])
            for c in current_discovered:
                master_discovered_companies.add(c)

        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETE")
        print("="*50)
        save_relationships_to_csv(master_relationships, args.output)

if __name__ == "__main__":
    asyncio.run(main())
