"""Define the state structures for the agent."""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class Reference(BaseModel):
    """Reference to evidence for a relationship."""
    url: str
    text: str


class Relationship(BaseModel):
    """Represents a business relationship between companies."""
    source: str
    relationship: str
    target: str
    reason: str
    references: List[Reference]


class websiteCrawlerResponse(BaseModel):
    """Response from the website crawler agent."""
    relationships: List[Relationship]
    new_companies: List[str]


class linkGetterResponse(BaseModel):
    """Response from the link getter agent."""
    links: List[str]
    page_context: str


class BackgroundStrategy(BaseModel):
    """Strategy for background research and query generation."""
    summary: str = Field(description="A brief 1-sentence summary of what the company does.")
    search_queries: List[str] = Field(description="A list of 3-5 Google search queries optimized to find partners/clients for this specific company type.")


class AgentState(TypedDict):
    """State for the crawling agent workflow."""
    target: str
    messages: List[BaseMessage]
    search_result: Optional[Dict[str, Any]]
    selected_links: List[str]
    extracted_relationships: List[Relationship]
    discovered_companies: List[str]
    companiestoexplore: List[str]
    background_summary: Optional[str]
    generated_queries: List[str]
