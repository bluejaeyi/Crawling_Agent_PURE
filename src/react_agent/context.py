"""Define the configurable parameters for the crawling agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass(kw_only=True)
class CrawlingContext:
    """Configuration context for the crawling agent."""

    model_name: str = field(
        default="gpt-5.2",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Default is gpt-5.2 from OpenAI."
        },
    )

    model_provider: str = field(
        default="openai",
        metadata={
            "description": "The model provider (e.g., 'openai', 'anthropic')."
        },
    )

    max_search_results: int = field(
        default=100,
        metadata={
            "description": "The maximum number of search results to return from Google Serper."
        },
    )

    recursion_limit: int = field(
        default=50,
        metadata={
            "description": "Maximum recursion depth for the agent graph."
        },
    )

    browser_headless: bool = field(
        default=True,
        metadata={
            "description": "Whether to run the browser in headless mode."
        },
    )

    output_csv: str = field(
        default="output.csv",
        metadata={
            "description": "Default filename for the output CSV file."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_value = os.environ.get(f.name.upper())
                if env_value is not None:
                    # Handle type conversion for non-string fields
                    if isinstance(f.default, bool):
                        setattr(self, f.name, env_value.lower() in ('true', '1', 'yes'))
                    elif isinstance(f.default, int):
                        setattr(self, f.name, int(env_value))
                    else:
                        setattr(self, f.name, env_value)
