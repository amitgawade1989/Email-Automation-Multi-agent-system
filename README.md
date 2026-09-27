# Agentic AI for Email Automation

A simple multi-agent system that reads customer emails, prioritises them by
sentiment and urgency, drafts personalised replies with company knowledge (RAG),
reviews its own drafts, and schedules them for dispatch. You view everything in a
Streamlit UI.

**Tech:** Python, LangGraph (agent orchestration), LangChain (prompts and LLM calls),
Mistral AI API, Streamlit

---

## 1. How to run

Full step-by-step instructions are in [HOW_TO_RUN.md](HOW_TO_RUN.md). Short version:

```bash
pip install -r requirements.txt
# put MISTRAL_API_KEY=your_key_here in the .env file
streamlit run app.py
```

---

## 2. Project structure

```
.env                    secrets (MISTRAL_API_KEY) - never commit this
config.py               all settings: model, folders, scoring weights, thresholds
email_loader.py         reads .eml / .txt files into simple dicts
knowledge_base.py       tiny RAG: finds relevant policies and past replies
agents.py               the 4 agents (plain functions) and their prompts
workflow.py             LangGraph: connects the agents, runs the whole inbox
app.py                  Streamlit UI
data/
    emails/             sample incoming emails (8 files, .txt and .eml)
    knowledge_base.md   company policies used for RAG
    past_responses.json historical replies, one per category (style examples)
    outbox.json         created when you save - the "scheduled" replies
```

A good reading order is `config.py`, `email_loader.py`, `knowledge_base.py`,
`agents.py`, `workflow.py`, then `app.py`.

---

## 3. Agent orchestration strategy

There are four agents, and each one is just a Python function. LangGraph connects
them into a graph.

```
 email --> [Classifier] --> [Drafting] --> [Review] --approved--> [Scheduler] --> outbox
             (LLM)          (LLM + RAG)      (LLM)                  (rules)
                                ^               |
                                |               | not approved and revisions left
                                +---------------+
                                 feedback sent back
```

| Agent | What it does | Output added to the state |
|---|---|---|
| Classifier Agent | Reads the email plus the inbox context (other emails from the same sender today) and labels it | category, sentiment, urgency, summary, priority score |
| Drafting Agent | Retrieves matching policies from the knowledge base (RAG), loads a past reply of the same category as a style example, and writes the reply | draft, knowledge used |
| Review Agent | Checks the draft for completeness, tone and made-up facts, and gives a confidence score (0 to 1) plus feedback | confidence, approved, feedback |
| Scheduler Agent | Pure rules, with no LLM. It sets the send time from the priority. Low-confidence drafts are held for a human | scheduled_time, status |

Key ideas:

- **Shared state.** All agents read and write one dictionary (`EmailState` in
  `workflow.py`). Each agent returns only the fields it changes, and LangGraph merges them in.
- **Conditional edge (self-correction loop).** After review, `route_after_review()`
  decides the next step. If the draft is not good enough, it goes back to the Drafting
  Agent with the reviewer's feedback, up to `MAX_REVISIONS` times.
- **Human in the loop.** If confidence is still below `CONFIDENCE_THRESHOLD`, the reply
  is not auto-scheduled. Its status becomes `needs_human_review`.
- **Batch processing.** `process_inbox()` runs the graph once per email and then sorts
  the results by priority score, so the most urgent replies come first.
- **LLM for judgement, code for rules.** The LLM does the language work (classify,
  write, critique). Plain Python does the maths (priority score, send time), so those
  parts are predictable and easy to explain.

---

## 4. Email prioritisation logic

1. The Classifier Agent (LLM) returns:
   - `sentiment`: angry / negative / neutral / positive
   - `urgency`: critical / high / medium / low. The prompt defines each level. For
     example, money lost, a chargeback threat or a repeated complaint counts as
     critical. The agent sees whether the same sender already emailed today.
2. Plain Python converts these into a priority score (0 to 100) using the weights in
   `config.py`:

   | Urgency | Points | | Sentiment | Points |
   |---|---|---|---|---|
   | critical | 60 | | angry | 40 |
   | high | 45 | | negative | 25 |
   | medium | 25 | | neutral | 10 |
   | low | 5 | | positive | 0 |

   `priority_score = urgency points + sentiment points`

3. The score maps to a label, and the label sets the send delay:

   | Score | Priority | Reply is scheduled |
   |---|---|---|
   | 75 or more | urgent | in 15 minutes |
   | 50 to 74 | high | in 1 hour |
   | 25 to 49 | medium | in 4 hours |
   | below 25 | low | in 24 hours |

   Example: "Charged twice ... I will raise a chargeback" gives critical (60) +
   angry (40) = 100, so the email is urgent.

---

## 5. Response generation logic

1. **Retrieve (RAG).** `search_knowledge_base()` splits `knowledge_base.md` into
   sections (one per `## heading`), scores each section by how many keywords it shares
   with the email, and keeps the top 3. It uses simple keyword matching instead of a
   vector DB so the idea is easy to see. You could swap in embeddings later without
   changing the agents.
2. **Historical responses.** `get_past_response()` picks a past reply from the same
   category so the new reply matches the company's tone and format.
3. **Generate.** The Drafting Agent prompt tells the LLM to greet the customer by name,
   apologise first if they are upset, use only knowledge base facts, write a placeholder
   such as `[REFUND REFERENCE]` for anything it does not know, answer every question,
   and stay under 180 words.
4. **Review and score.** The Review Agent grades the draft and returns a confidence
   score. A score of 0.75 or more is auto-scheduled. Anything lower is sent back for one
   rewrite with feedback. If it is still low, a human must review it.

Every LLM call uses the same LangChain pattern in `run_chain()`:

```python
chain = prompt | llm | parser      # ChatPromptTemplate | ChatMistralAI | JSON/String parser
result = chain.invoke(inputs)
```

---

## 6. Customising

| Want to | Change |
|---|---|
| Use another Mistral model | `MODEL_NAME` in `config.py` |
| Make auto-sending stricter or looser | `CONFIDENCE_THRESHOLD` in `config.py` |
| Change how priority is scored | `URGENCY_POINTS`, `SENTIMENT_POINTS`, `PRIORITY_LEVELS` in `config.py` |
| Change send timing | `SEND_DELAY_MINUTES` in `config.py` |
| Add company knowledge | Add a `## Section` to `data/knowledge_base.md` |
| Use a different LLM provider (OpenAI, Claude, etc.) | Only `get_llm()` in `agents.py` |
