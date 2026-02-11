"""Define the crawling agent graph.

This module contains the workflow nodes and graph compilation for the
company relationship crawling agent, compatible with LangGraph Studio.
"""

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from playwright.async_api import async_playwright

from react_agent.state import (
    AgentState,
    BackgroundStrategy,
    linkGetterResponse,
    websiteCrawlerResponse
)
from react_agent.prompts import SYSTEM_PROMPT, LINK_SELECTOR_PROMPT, BACKGROUND_PROMPT
from react_agent.tools import get_search_tool, get_serper_wrapper
from react_agent.context import CrawlingContext


# Module-level browser context manager
class BrowserManager:
    """Manages Playwright browser lifecycle for the crawling agent."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None

    async def start(self, headless: bool = True):
        """Start the browser if not already started."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"]
            )
        return self._browser

    async def stop(self):
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    def get_browser(self):
        """Get the current browser instance."""
        return self._browser


# Global browser manager instance
_browser_manager = BrowserManager()


async def BackgroundInfoandqueries(
    state: AgentState,
    config: RunnableConfig,
    runtime: Runtime[CrawlingContext]
):
    """Node: Research company background and generate search queries.

    Args:
        state: Current agent state
        config: Runnable configuration
        runtime: Runtime with context configuration

    Returns:
        Updated state with background summary and generated queries
    """
    company_name = state["target"]
    context = runtime.context

    # Ensure browser is started
    browser = await _browser_manager.start(headless=context.browser_headless)

    homepage_text = "Not available (Scraping failed or skipped)"
    top_link = None
    snippets = []
    combined_snippets = ""
    search_query = f"{company_name} official site / wikipedia"

    # Get serper wrapper for search
    serper = get_serper_wrapper(context.max_search_results)

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
            if page:
                await page.close()

    print(f"   > Generating search strategy...")

    # Get tools and create agents
    searchtool = get_search_tool(context.max_search_results)
    llm = init_chat_model(
        context.model_name,
        model_provider=context.model_provider,
        max_retries=50
    )

    background_agent = create_agent(
        model=llm,
        tools=searchtool,
        system_prompt=BACKGROUND_PROMPT,
        response_format=BackgroundStrategy
    )

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


async def linkGetterNode(
    state: AgentState,
    config: RunnableConfig,
    runtime: Runtime[CrawlingContext]
):
    """Node: Get and select relevant links from search results.

    Args:
        state: Current agent state
        config: Runnable configuration
        runtime: Runtime with context configuration

    Returns:
        Updated state with selected links
    """
    company_name = state["target"]
    context = runtime.context
    queries = state.get("generated_queries", [])
    all_search_results = []

    # Get tools
    searchtool = get_search_tool(context.max_search_results)

    try:
        for q in queries:
            search_results = await searchtool[0].ainvoke({"query": q, "page": 0})
            all_search_results.append(str(search_results))
    except Exception as e:
        search_results = f"Search failed: {e}"

    print(f"Selecting links for {company_name}...")
    combined_search_context = "\n\n".join(all_search_results)

    # Create link getter agent
    llm = init_chat_model(
        context.model_name,
        model_provider=context.model_provider,
        max_retries=50
    )

    linkGetterllm = create_agent(
        model=llm,
        tools=searchtool,
        system_prompt=LINK_SELECTOR_PROMPT,
        response_format=linkGetterResponse
    ).with_config({"recursion_limit": context.recursion_limit})

    try:
        prompt_content = f"""
        Here are search results for company: {company_name}:
        {combined_search_context}
        Based on the text above, extract a list of relevant links.
        """

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


async def websiteCrawlerNode(
    state: AgentState,
    config: RunnableConfig,
    runtime: Runtime[CrawlingContext]
):
    """Node: Crawl website and extract relationships.

    Args:
        state: Current agent state
        config: Runnable configuration
        runtime: Runtime with context configuration

    Returns:
        Updated state with extracted relationships
    """
    company_name = state["target"]
    context = runtime.context
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

    # Get browser and tools
    browser = await _browser_manager.start(headless=context.browser_headless)

    # Import here to avoid circular dependency
    from react_agent.tools import get_browser_tools
    tools = get_browser_tools(browser)

    # Create crawler agent
    llm = init_chat_model(
        context.model_name,
        model_provider=context.model_provider,
        max_retries=50
    ).bind_tools(tools)

    websitecrawleragent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=websiteCrawlerResponse
    ).with_config({"recursion_limit": context.recursion_limit})

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
        result = await websitecrawleragent.ainvoke(
            {"messages": [HumanMessage(prompt_content)]},
            config=config
        )
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
    """Conditional edge: Determine whether to continue crawling.

    Args:
        state: Current agent state

    Returns:
        "continue" if there are more links to crawl, "end" otherwise
    """
    if state["selected_links"] and len(state["selected_links"]) > 0:
        return "continue"
    return "end"


# Build the graph
builder = StateGraph(AgentState, context_schema=CrawlingContext)

# Add nodes
builder.add_node("background", BackgroundInfoandqueries)
builder.add_node("link_getter", linkGetterNode)
builder.add_node("website_crawler", websiteCrawlerNode)

# Set entry point and edges
builder.set_entry_point("background")
builder.add_edge("background", "link_getter")
builder.add_edge("link_getter", "website_crawler")

# Add conditional edges
builder.add_conditional_edges(
    "website_crawler",
    should_continue,
    {
        "continue": "website_crawler",
        "end": END
    }
)

# Compile the graph (LangGraph Studio provides persistence automatically)
graph = builder.compile()

# Export for LangGraph Studio compatibility
__all__ = ["graph", "BrowserManager", "_browser_manager"]
