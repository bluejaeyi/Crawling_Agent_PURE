"""This module provides tools for web scraping and search functionality.

It includes Playwright browser tools and Google Serper search functionality
for crawling company websites and extracting relationship data.
"""

from typing import Any, Callable, List
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from playwright.async_api import Browser


def get_browser_tools(browser: Browser) -> List[Callable[..., Any]]:
    """Get Playwright browser tools from an async browser instance.

    Args:
        browser: An async Playwright browser instance

    Returns:
        List of browser tools for web navigation and interaction
    """
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools()


def get_search_tool(k: int = 100) -> List[Callable[..., Any]]:
    """Get Google Serper search tool.

    Args:
        k: Maximum number of search results to return

    Returns:
        List containing the search tool
    """
    serper = GoogleSerperAPIWrapper(k=k)

    @tool("Search_results")
    def search(query: str, page: int) -> str:
        """useful for when you need to look up links with search"""
        return serper.results(query, page=page)

    return [search]


def get_serper_wrapper(k: int = 100) -> GoogleSerperAPIWrapper:
    """Get a GoogleSerperAPIWrapper instance for direct use.

    Args:
        k: Maximum number of search results to return

    Returns:
        GoogleSerperAPIWrapper instance
    """
    return GoogleSerperAPIWrapper(k=k)
