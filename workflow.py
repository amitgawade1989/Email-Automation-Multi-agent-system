"""
workflow.py - Connects the agents together using LangGraph.

The graph for ONE email looks like this:

    START -> classifier -> drafting -> review --(approved)--------> scheduler -> END
                              ^           |
                              |           +--(not approved, can retry)
                              +-----------+

`process_inbox` runs this graph for every email in the batch and then
sorts the results so the most urgent emails come first.
"""

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agents import classifier_agent, drafting_agent, review_agent, scheduler_agent


class EmailState(TypedDict, total=False):
    """
    The shared "notebook" that every agent reads from and writes to.
    (TypedDict is just a dict with named keys - LangGraph needs it to
    know which fields exist.)
    """
    email: dict                    # the incoming email (from email_loader)
    other_emails: str              # other subjects from the same sender (inbox context)
    category: str                  # set by classifier_agent
    sentiment: str
    urgency: str
    summary: str
    classifier_confidence: float
    priority_score: int
    priority: str
    knowledge: str                 # set by drafting_agent (RAG result)
    draft: str
    revision_count: int
    confidence: float              # set by review_agent
    approved: bool
    review_feedback: str
    scheduled_time: str            # set by scheduler_agent
    status: str


def route_after_review(state: EmailState) -> str:
    """
    Decide where to go after the Review Agent (a "conditional edge").

    Args:
        state: The current state after review.

    Returns:
        "drafting"  if the draft was rejected and we may still rewrite it,
        "scheduler" otherwise.
    """
    if not state["approved"] and state["revision_count"] <= config.MAX_REVISIONS:
        return "drafting"
    return "scheduler"


def build_graph():
    """
    Create and compile the LangGraph workflow.

    Returns:
        A compiled graph. Call graph.invoke(initial_state) to run it.
    """
    graph = StateGraph(EmailState)

    # 1. Register each agent as a node.
    graph.add_node("classifier", classifier_agent)
    graph.add_node("drafting", drafting_agent)
    graph.add_node("review", review_agent)
    graph.add_node("scheduler", scheduler_agent)

    # 2. Connect them with edges.
    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "drafting")
    graph.add_edge("drafting", "review")
    graph.add_conditional_edges("review", route_after_review, ["drafting", "scheduler"])
    graph.add_edge("scheduler", END)

    return graph.compile()


def find_other_emails(email: dict, all_emails: list[dict]) -> str:
    """
    Give the classifier context about the rest of the inbox: has this
    customer already written to us about something else today?

    Args:
        email:      The email being processed.
        all_emails: Every email in the batch.

    Returns:
        Subjects of other emails from the same sender, or "None".
    """
    subjects = [
        other["subject"]
        for other in all_emails
        if other["sender_email"] == email["sender_email"] and other["id"] != email["id"]
    ]
    return "; ".join(subjects) if subjects else "None"


def process_inbox(emails: list[dict], on_progress=None) -> list[dict]:
    """
    Run the agent workflow on every email and sort the results by priority.

    Args:
        emails:      List of email dicts from email_loader.load_emails().
        on_progress: Optional function(done, total, email) called after each
                     email - the Streamlit UI uses it for the progress bar.

    Returns:
        List of final states (dicts), highest priority_score first.
    """
    graph = build_graph()
    results = []

    for index, email in enumerate(emails, start=1):
        initial_state = {
            "email": email,
            "other_emails": find_other_emails(email, emails),
            "revision_count": 0,
        }
        try:
            results.append(graph.invoke(initial_state))
        except Exception as error:  # one bad email should not stop the batch
            results.append({
                "email": email,
                "status": "error",
                "review_feedback": f"Agent failed: {error}",
                "priority_score": 0,
                "priority": "low",
                "confidence": 0.0,
            })
        if on_progress:
            on_progress(index, len(emails), email)

    results.sort(key=lambda result: result.get("priority_score", 0), reverse=True)
    return results


def build_outbox(results: list[dict]) -> list[dict]:
    """
    Turn the workflow results into simple outbox entries (one per reply).

    Args:
        results: Output of process_inbox (drafts may have been edited in the UI).

    Returns:
        List of dicts: to, subject, body, priority, confidence, scheduled_time, status.
    """
    return [
        {
            "to": result["email"]["sender_email"],
            "subject": "Re: " + result["email"]["subject"],
            "body": result.get("draft", ""),
            "priority": result.get("priority"),
            "confidence": result.get("confidence"),
            "scheduled_time": result.get("scheduled_time"),
            "status": result.get("status"),
        }
        for result in results
        if result.get("status") != "error"
    ]


def save_outbox(results: list[dict]) -> list[dict]:
    """
    Save the scheduled replies to data/outbox.json (instead of really sending).

    Args:
        results: Output of process_inbox.

    Returns:
        The list of outbox entries that was written to the file.
    """
    outbox = build_outbox(results)
    config.OUTBOX_FILE.write_text(json.dumps(outbox, indent=2), encoding="utf-8")
    return outbox
