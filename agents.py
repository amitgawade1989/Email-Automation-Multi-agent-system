"""
agents.py - The four agents (plain functions) used by the LangGraph workflow.

Every agent follows the same simple contract:
    - it receives the current `state` (a dict with everything known so far)
    - it returns a dict with ONLY the fields it wants to add or update
LangGraph merges that returned dict back into the state for the next agent.

    1. classifier_agent  -> category, sentiment, urgency, priority     (LLM)
    2. drafting_agent    -> reply draft using RAG + past replies       (LLM)
    3. review_agent      -> confidence score + feedback                (LLM)
    4. scheduler_agent   -> send time + status                         (plain Python)
"""

from datetime import datetime, timedelta

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

import config
from knowledge_base import get_past_response, search_knowledge_base

# ===========================================================================
# Prompts. Words inside {curly_braces} are filled in when the chain runs.
# ===========================================================================

CLASSIFIER_SYSTEM_PROMPT = """You are the Email Classifier Agent for {company_name} customer support.
Read the customer email and return ONLY a JSON object with these keys:
- "category": one of {categories}
- "sentiment": one of {sentiments}
- "urgency": one of {urgencies}
- "summary": one short sentence saying what the customer wants
- "confidence": a number between 0 and 1 showing how sure you are

Urgency guide:
- critical: money lost, chargeback or legal threats, or a repeated complaint
- high: customer is blocked or has a deadline within a day or two
- medium: needs help but is not time sensitive
- low: thanks, feedback, suggestions or general questions"""

CLASSIFIER_HUMAN_PROMPT = """From: {sender_name} <{sender_email}>
Subject: {subject}

{body}

Other emails from this same sender in today's inbox: {other_emails}"""

DRAFT_SYSTEM_PROMPT = """You are the Drafting Agent for {company_name} customer support.
Write a reply to the customer email.

Rules:
- Greet the customer by first name.
- Match the tone to their mood. If they are angry or upset, apologise sincerely first.
- Use ONLY facts from the knowledge base below. Never invent policies, prices, dates or reference numbers.
- If the reply needs a value you do not have (e.g. a refund reference), write a placeholder like [REFUND REFERENCE].
- Write plain text only (no markdown such as ** or #).
- Answer every question they asked and give clear next steps.
- Keep it under 180 words.
- Sign off as: {signature}
- Return only the email body, no subject line.

Knowledge base:
{knowledge}

A past reply we sent for a similar email (copy the style, not the details):
{past_response}"""

DRAFT_HUMAN_PROMPT = """Customer email:
From: {sender_name}
Subject: {subject}

{body}

Classifier notes: category={category}, sentiment={sentiment}, urgency={urgency}
Summary: {summary}

Reviewer feedback on your previous draft (fix these points, if any): {review_feedback}"""

REVIEW_SYSTEM_PROMPT = """You are the Review Agent (quality checker) for {company_name} customer support.
Check the draft reply against the customer email and the knowledge base:
1. Does it answer every question the customer asked?
2. Is the tone right for the customer's mood?
3. Is every fact supported by the knowledge base? Made-up promises, numbers or reference IDs
   are serious errors (placeholders in [BRACKETS] are fine).
4. Is it clear, polite and professional?

Return ONLY a JSON object with these keys:
- "confidence": a number between 0 and 1 - how safe it is to send this reply without a human
- "feedback": one string listing the problems to fix (empty string if there are none)"""

REVIEW_HUMAN_PROMPT = """Customer email:
Subject: {subject}

{body}

Knowledge base:
{knowledge}

Draft reply:
{draft}"""


# ===========================================================================
# LLM helpers
# ===========================================================================

def get_llm() -> ChatMistralAI:
    """
    Create the Mistral chat model using settings from config.py.

    Returns:
        A LangChain chat model that talks to the Mistral API.
    """
    return ChatMistralAI(
        api_key=config.MISTRAL_API_KEY,
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
    )


def run_chain(system_prompt: str, human_prompt: str, inputs: dict, want_json: bool):
    """
    Build and run a LangChain chain:  prompt  ->  LLM  ->  output parser.

    The "|" operator connects the steps (LangChain Expression Language).

    Args:
        system_prompt: Instructions telling the LLM its role.
        human_prompt:  The actual task / data for the LLM.
        inputs:        Values for the {placeholders} in both prompts.
        want_json:     True -> return a dict parsed from JSON, False -> plain text.

    Returns:
        A dict (when want_json=True) or a string.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    parser = JsonOutputParser() if want_json else StrOutputParser()
    chain = prompt | get_llm() | parser
    return chain.invoke(inputs)


# ===========================================================================
# Small pure-Python helpers (no LLM) - easy to test and explain
# ===========================================================================

def pick_allowed(value, allowed: list[str], default: str) -> str:
    """
    Make sure an LLM answer is one of the allowed labels.

    Args:
        value:   What the LLM returned, e.g. "Angry".
        allowed: The list of valid labels.
        default: Label to use if the value is not valid.

    Returns:
        A valid lowercase label.
    """
    value = str(value).strip().lower()
    return value if value in allowed else default


def to_score(value, default: float = 0.5) -> float:
    """
    Convert an LLM number (e.g. "0.8" or 80) into a float between 0 and 1.

    Args:
        value:   The raw value from the LLM.
        default: Value to use if it cannot be converted.

    Returns:
        A float in the range 0.0 - 1.0.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1:  # the LLM answered in percent
        number = number / 100
    return max(0.0, min(1.0, number))


def calculate_priority(urgency: str, sentiment: str) -> tuple[int, str]:
    """
    Turn urgency + sentiment into a priority score and a label.

    score = URGENCY_POINTS[urgency] + SENTIMENT_POINTS[sentiment]   (0 - 100)
    Example: critical (60) + angry (40) = 100 -> "urgent"

    Args:
        urgency:   One of config.URGENCIES.
        sentiment: One of config.SENTIMENTS.

    Returns:
        (score, label), e.g. (100, "urgent").
    """
    score = config.URGENCY_POINTS[urgency] + config.SENTIMENT_POINTS[sentiment]
    for minimum_score, label in config.PRIORITY_LEVELS:
        if score >= minimum_score:
            return score, label
    return score, "low"


def calculate_send_time(priority: str, now: datetime) -> datetime:
    """
    Decide when a reply should be dispatched, based on its priority.

    Args:
        priority: "urgent", "high", "medium" or "low".
        now:      The current time.

    Returns:
        The datetime at which the reply should be sent.
    """
    return now + timedelta(minutes=config.SEND_DELAY_MINUTES[priority])


# ===========================================================================
# The agents (LangGraph nodes)
# ===========================================================================

def classifier_agent(state: dict) -> dict:
    """
    Agent 1 - Email Classifier.

    Asks the LLM for category, sentiment and urgency, then calculates the
    priority score with a simple, transparent formula.

    Args:
        state: Must contain "email" and "other_emails".

    Returns:
        category, sentiment, urgency, summary, classifier_confidence,
        priority_score and priority.
    """
    email = state["email"]
    result = run_chain(
        CLASSIFIER_SYSTEM_PROMPT,
        CLASSIFIER_HUMAN_PROMPT,
        {
            "company_name": config.COMPANY_NAME,
            "categories": config.CATEGORIES,
            "sentiments": config.SENTIMENTS,
            "urgencies": config.URGENCIES,
            "sender_name": email["sender_name"],
            "sender_email": email["sender_email"],
            "subject": email["subject"],
            "body": email["body"],
            "other_emails": state["other_emails"],
        },
        want_json=True,
    )

    category = pick_allowed(result.get("category"), config.CATEGORIES, "other")
    sentiment = pick_allowed(result.get("sentiment"), config.SENTIMENTS, "neutral")
    urgency = pick_allowed(result.get("urgency"), config.URGENCIES, "medium")
    priority_score, priority = calculate_priority(urgency, sentiment)

    return {
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "summary": result.get("summary", ""),
        "classifier_confidence": to_score(result.get("confidence")),
        "priority_score": priority_score,
        "priority": priority,
    }


def drafting_agent(state: dict) -> dict:
    """
    Agent 2 - Drafting.

    1. Retrieves relevant policies from the knowledge base (RAG).
    2. Loads a past reply of the same category as a style example.
    3. Asks the LLM to write a personalised reply.
    If the Review Agent sent feedback, the LLM sees it and fixes the draft.

    Args:
        state: Needs "email" and the classifier fields.

    Returns:
        knowledge, draft and the updated revision_count.
    """
    email = state["email"]
    knowledge = search_knowledge_base(email["subject"] + " " + email["body"])

    draft = run_chain(
        DRAFT_SYSTEM_PROMPT,
        DRAFT_HUMAN_PROMPT,
        {
            "company_name": config.COMPANY_NAME,
            "signature": config.AGENT_SIGNATURE,
            "knowledge": knowledge,
            "past_response": get_past_response(state["category"]),
            "sender_name": email["sender_name"],
            "subject": email["subject"],
            "body": email["body"],
            "category": state["category"],
            "sentiment": state["sentiment"],
            "urgency": state["urgency"],
            "summary": state["summary"],
            "review_feedback": state.get("review_feedback") or "None - this is the first draft.",
        },
        want_json=False,
    )

    return {
        "knowledge": knowledge,
        "draft": draft.strip(),
        "revision_count": state.get("revision_count", 0) + 1,
    }


def review_agent(state: dict) -> dict:
    """
    Agent 3 - Review.

    Asks the LLM to grade the draft and give a confidence score (0 - 1).
    The draft is "approved" when confidence >= config.CONFIDENCE_THRESHOLD.

    Args:
        state: Needs "email", "knowledge" and "draft".

    Returns:
        confidence, approved and review_feedback.
    """
    email = state["email"]
    result = run_chain(
        REVIEW_SYSTEM_PROMPT,
        REVIEW_HUMAN_PROMPT,
        {
            "company_name": config.COMPANY_NAME,
            "subject": email["subject"],
            "body": email["body"],
            "knowledge": state["knowledge"],
            "draft": state["draft"],
        },
        want_json=True,
    )

    confidence = to_score(result.get("confidence"))
    return {
        "confidence": confidence,
        "approved": confidence >= config.CONFIDENCE_THRESHOLD,
        "review_feedback": str(result.get("feedback", "")),
    }


def scheduler_agent(state: dict) -> dict:
    """
    Agent 4 - Scheduler (no LLM needed, just rules).

    Approved drafts are scheduled for dispatch based on priority.
    Drafts that failed review are held for a human to check first.

    Args:
        state: Needs "priority" and "approved".

    Returns:
        scheduled_time (ISO text) and status.
    """
    send_time = calculate_send_time(state["priority"], datetime.now())
    status = "scheduled" if state["approved"] else "needs_human_review"
    return {
        "scheduled_time": send_time.isoformat(timespec="minutes"),
        "status": status,
    }
