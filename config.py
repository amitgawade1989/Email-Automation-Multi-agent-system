"""
config.py - All settings for the project live here.

Secrets (API keys) are read from the .env file.
Everything else (model name, folders, scoring weights) is plain Python
so you can change behaviour without touching the agent code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file and put its values into environment variables.
load_dotenv()

# ---------------------------------------------------------------------------
# Secrets (from .env)
# ---------------------------------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# Any chat model listed at https://docs.mistral.ai/getting-started/models works here,
# e.g. "ministral-14b-latest", "ministral-8b-latest" or "mistral-small-latest".
# (On the free tier some models, like mistral-small/medium, may hit rate limits.)
MODEL_NAME = "ministral-14b-latest"
TEMPERATURE = 0.3  # low = more consistent answers

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMAILS_FOLDER = DATA_DIR / "emails"                     # incoming .eml / .txt files
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.md"    # company policies (for RAG)
PAST_RESPONSES_FILE = DATA_DIR / "past_responses.json"  # historical replies
OUTBOX_FILE = DATA_DIR / "outbox.json"                  # scheduled replies are saved here

# ---------------------------------------------------------------------------
# Company details used inside the prompts
# ---------------------------------------------------------------------------
COMPANY_NAME = "QuickCart"
AGENT_SIGNATURE = "Priya, QuickCart Customer Support"

# ---------------------------------------------------------------------------
# Classification labels the Classifier Agent is allowed to use
# ---------------------------------------------------------------------------
CATEGORIES = ["billing", "shipping", "technical", "sales", "cancellation", "feedback", "other"]
SENTIMENTS = ["angry", "negative", "neutral", "positive"]
URGENCIES = ["critical", "high", "medium", "low"]

# ---------------------------------------------------------------------------
# Prioritisation: priority_score = URGENCY_POINTS + SENTIMENT_POINTS (0 - 100)
# ---------------------------------------------------------------------------
URGENCY_POINTS = {"critical": 60, "high": 45, "medium": 25, "low": 5}
SENTIMENT_POINTS = {"angry": 40, "negative": 25, "neutral": 10, "positive": 0}

# Score thresholds for each priority label (checked from top to bottom).
PRIORITY_LEVELS = [
    (75, "urgent"),
    (50, "high"),
    (25, "medium"),
    (0, "low"),
]

# How long to wait before sending a reply, per priority (in minutes).
SEND_DELAY_MINUTES = {"urgent": 15, "high": 60, "medium": 240, "low": 1440}

# ---------------------------------------------------------------------------
# Review settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75  # below this, a human must check the reply
MAX_REVISIONS = 1            # how many times the Drafting Agent may rewrite

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
KB_TOP_K = 3  # how many knowledge base sections to give the Drafting Agent
