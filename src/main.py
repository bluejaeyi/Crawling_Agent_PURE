"""Main entry point for the company relationship crawling agent.

This script orchestrates the crawling workflow, processing multiple companies,
and saving results to CSV.
"""

import asyncio
import argparse
import uuid
from typing import List, Set
from langgraph.errors import GraphRecursionError

from react_agent.graph import graph, _browser_manager
from react_agent.context import CrawlingContext
from react_agent.utils import load_companies_from_file, save_relationships_to_csv
from react_agent.state import Relationship


# Global storage for aggregated results
master_relationships: List[Relationship] = []
master_discovered_companies: Set[str] = set()


async def main():
    """Main function to run the crawling agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Company Relationship Crawler - Extract B2B relationships from company websites"
    )

    parser.add_argument(
        "companies",
        nargs="*",
        help="List of companies to crawl (e.g., 'Apple' 'Microsoft')"
    )

    parser.add_argument(
        "-f", "--file",
        help="Path to a .txt or .csv file containing a list of companies"
    )

    parser.add_argument(
        "-o", "--output",
        default="output.csv",
        help="Filename for the output CSV (default: output.csv)"
    )

    parser.add_argument(
        "-m", "--model",
        default="gpt-5.2",
        help="Model name to use (default: gpt-5.2)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)"
    )

    args = parser.parse_args()

    # Initialize context
    context = CrawlingContext(
        model_name=args.model,
        output_csv=args.output,
        browser_headless=args.headless
    )

    # Collect target companies
    target_companies = []

    if args.file:
        print(f"Loading companies from file: {args.file}")
        target_companies.extend(load_companies_from_file(args.file))

    if args.companies:
        target_companies.extend(args.companies)

    if not target_companies:
        print("No companies provided via arguments or file.")
        print("\nUsage examples:")
        print("  python main.py \"Home Depot\" \"Lowe's\"")
        print("  python main.py -f companies.txt")
        print("  python main.py -f companies.csv -o results.csv")
        return

    # Remove duplicates
    target_companies = list(set(target_companies))

    print(f"Starting batch process for: {target_companies}")
    print("=" * 50)

    try:
        # Process each company
        for company in target_companies:
            print(f"\n>>> PROCESSING TARGET: {company}")

            # Initialize state for this company
            initial_state = {
                "target": company,
                "messages": [],
                "search_result": None,
                "background_summary": None,
                "generated_queries": [],
                "selected_links": [],
                "extracted_relationships": [],
                "discovered_companies": [],
                "companiestoexplore": []
            }

            # Create unique thread ID for this run
            thread_id = str(uuid.uuid4())
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    # Pass context parameters to Runtime
                    "model_name": context.model_name,
                    "model_provider": context.model_provider,
                    "max_search_results": context.max_search_results,
                    "browser_headless": context.browser_headless,
                    "recursion_limit": context.recursion_limit,
                    "output_csv": context.output_csv
                },
                "recursion_limit": context.recursion_limit
            }

            try:
                # Run the crawling workflow
                final_state = await graph.ainvoke(initial_state, config)

            except GraphRecursionError:
                print(f"Graph recursion limit reached for {company}")
                # Try to get the current state from the snapshot
                snapshot = await graph.aget_state(config)
                if snapshot and snapshot.values:
                    final_state = snapshot.values
                else:
                    final_state = initial_state

            except Exception as e:
                print(f"An error occurred for {company}: {e}")
                final_state = initial_state

            # Aggregate results
            current_relationships = final_state.get('extracted_relationships', [])
            master_relationships.extend(current_relationships)

            current_discovered = final_state.get('discovered_companies', [])
            for c in current_discovered:
                master_discovered_companies.add(c)

            print(f"✓ Completed {company}: Found {len(current_relationships)} relationships")

    finally:
        # Clean up browser resources
        await _browser_manager.stop()

    # Save results
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total relationships found: {len(master_relationships)}")
    print(f"Total companies discovered: {len(master_discovered_companies)}")
    save_relationships_to_csv(master_relationships, args.output)


if __name__ == "__main__":
    asyncio.run(main())
