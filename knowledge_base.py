"""
knowledge_base.py - A very small RAG (Retrieval-Augmented Generation) helper.

RAG means: before asking the LLM to write a reply, we first *retrieve*
the most relevant company policies and paste them into the prompt.
That way the LLM answers with real facts instead of guessing.

To keep it easy to understand we use simple keyword matching instead of
embeddings / a vector database. The idea is exactly the same.
"""

import json
import re

import config

# Common words that carry no meaning for matching.
STOP_WORDS = {
    "the", "and", "for", "you", "your", "with", "this", "that", "have", "from",
    "are", "was", "but", "not", "can", "will", "please", "thanks", "thank",
    "hello", "regards", "would", "could", "what", "there", "they", "been",
}


def get_keywords(text: str) -> set[str]:
    """
    Split text into a set of useful lowercase words.

    Args:
        text: Any text, e.g. an email body.

    Returns:
        Set of words with 3+ letters that are not stop words.
    """
    words = re.findall(r"[a-z]+", text.lower())
    return {word for word in words if len(word) >= 3 and word not in STOP_WORDS}


def load_knowledge_base() -> list[dict]:
    """
    Read the knowledge base markdown file and split it into sections.

    Every "## Heading" in the file starts a new section.

    Returns:
        List of dicts like {"title": "Refund policy", "text": "..."}.
    """
    content = config.KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8")
    sections = []
    for chunk in content.split("\n## ")[1:]:  # [0] is the file title, skip it
        title, _, text = chunk.partition("\n")
        sections.append({"title": title.strip(), "text": text.strip()})
    return sections


def search_knowledge_base(query: str, top_k: int = config.KB_TOP_K) -> str:
    """
    Find the knowledge base sections that best match the query.

    Score = number of keywords the query and the section have in common.

    Args:
        query: Text to search with (we pass the email subject + body).
        top_k: How many sections to return.

    Returns:
        The matching sections joined into one string, ready for a prompt.
    """
    query_words = get_keywords(query)
    scored_sections = []
    for section in load_knowledge_base():
        section_words = get_keywords(section["title"] + " " + section["text"])
        score = len(query_words & section_words)
        scored_sections.append((score, section))

    scored_sections.sort(key=lambda item: item[0], reverse=True)
    best = [section for score, section in scored_sections[:top_k] if score > 0]

    if not best:
        return "No relevant policy found."
    return "\n\n".join(f"[{s['title']}]\n{s['text']}" for s in best)


def get_past_response(category: str) -> str:
    """
    Get a historical reply we sent before for the same kind of email.

    The Drafting Agent uses it as a style example.

    Args:
        category: Email category, e.g. "billing".

    Returns:
        The example reply text (falls back to the "other" example).
    """
    past_responses = json.loads(config.PAST_RESPONSES_FILE.read_text(encoding="utf-8"))
    return past_responses.get(category, past_responses["other"])
