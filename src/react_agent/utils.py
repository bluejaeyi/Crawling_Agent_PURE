"""Utility & helper functions."""

import csv
import sys
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from react_agent.state import Relationship


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def load_companies_from_file(filepath: str) -> List[str]:
    """Load companies from a text or CSV file.

    Args:
        filepath: Path to the file containing company names

    Returns:
        List of company names

    Raises:
        SystemExit: If there's an error reading the file
    """
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
    """Save extracted relationships to a CSV file.

    Args:
        relationships: List of Relationship objects to save
        filename: Output CSV filename
    """
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
