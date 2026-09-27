"""
app.py - Streamlit user interface.

Run with:   streamlit run app.py

What you can do in the UI:
    1. See the incoming emails (sample inbox and/or your uploaded files).
    2. Click "Run AI Agents" to classify, draft, review and schedule replies.
    3. Read and edit each draft, see its confidence score and send time.
    4. Save everything to the outbox (data/outbox.json).
"""

import json

import streamlit as st

import config
from email_loader import load_emails, parse_email_text
from workflow import build_outbox, process_inbox, save_outbox

STATUS_LABELS = {
    "scheduled": "Scheduled",
    "needs_human_review": "Needs human review",
    "error": "Error",
}


def get_inbox(use_samples: bool, uploaded_files) -> list[dict]:
    """
    Collect the emails to process from the sample folder and/or uploads.

    Args:
        use_samples:    True to include the emails in data/emails.
        uploaded_files: Files from st.file_uploader (may be empty).

    Returns:
        List of email dicts.
    """
    emails = load_emails(config.EMAILS_FOLDER) if use_samples else []
    for file in uploaded_files or []:
        emails.append(parse_email_text(file.getvalue().decode("utf-8"), file.name))
    return emails


def show_inbox(emails: list[dict]) -> None:
    """
    Show the incoming emails (before the agents run).

    Args:
        emails: List of email dicts.
    """
    st.subheader(f"Incoming emails ({len(emails)})")
    for email in emails:
        with st.expander(f"{email['subject']} - {email['sender_name']}"):
            st.caption(f"From: {email['sender_email']}  |  Date: {email['date']}  |  File: {email['id']}")
            st.text(email["body"])


def show_summary(results: list[dict]) -> None:
    """
    Show headline numbers and a table of all results, most urgent first.

    Args:
        results: Output of workflow.process_inbox.
    """
    scheduled = sum(1 for r in results if r.get("status") == "scheduled")
    review = sum(1 for r in results if r.get("status") == "needs_human_review")
    average_confidence = sum(r.get("confidence", 0) for r in results) / len(results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Emails processed", len(results))
    col2.metric("Auto-scheduled", scheduled)
    col3.metric("Need human review", review)
    col4.metric("Average confidence", f"{average_confidence:.0%}")

    table = [
        {
            "Priority": r.get("priority", ""),
            "Score": r.get("priority_score", 0),
            "From": r["email"]["sender_name"],
            "Subject": r["email"]["subject"],
            "Category": r.get("category", "-"),
            "Sentiment": r.get("sentiment", "-"),
            "Urgency": r.get("urgency", "-"),
            "Confidence": f"{r.get('confidence', 0):.0%}",
            "Send at": r.get("scheduled_time", "-").replace("T", " "),
            "Status": STATUS_LABELS.get(r.get("status"), r.get("status")),
        }
        for r in results
    ]
    st.dataframe(table, width="stretch", hide_index=True)


def show_result(result: dict) -> None:
    """
    Show one processed email: original on the left, AI reply on the right.
    The draft is editable; edits are kept in st.session_state.

    Args:
        result: One final state from the workflow.
    """
    email = result["email"]
    title = f"[{result.get('priority', '?').upper()}] {email['subject']} - {email['sender_name']}"

    with st.expander(title):
        if result.get("status") == "error":
            st.error(result["review_feedback"])
            return

        left, right = st.columns(2)

        with left:
            st.markdown("**Original email**")
            st.caption(f"From: {email['sender_email']}  |  {email['date']}")
            st.text(email["body"])
            st.markdown("**Classifier Agent**")
            st.write(
                f"Category: `{result['category']}` | Sentiment: `{result['sentiment']}` | "
                f"Urgency: `{result['urgency']}` | Priority score: `{result['priority_score']}/100`"
            )
            st.write(f"Summary: {result['summary']}")
            st.caption(f"Classifier confidence: {result['classifier_confidence']:.0%}")

        with right:
            st.markdown(f"**Drafted reply** - Re: {email['subject']}")
            st.text_area("Edit before sending", value=result["draft"], height=300,
                         key=f"draft_{email['id']}")

            st.markdown("**Review Agent**")
            st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")
            if result.get("review_feedback"):
                st.caption(f"Feedback: {result['review_feedback']}")
            st.caption(f"Drafts written: {result['revision_count']}")

            st.markdown("**Scheduler Agent**")
            st.write(f"{STATUS_LABELS[result['status']]} - send at "
                     f"**{result['scheduled_time'].replace('T', ' ')}**")

        with st.popover("Knowledge used (RAG)"):
            st.text(result["knowledge"])


def apply_edits(results: list[dict]) -> list[dict]:
    """
    Copy any drafts the user edited in the UI back into the results.

    Args:
        results: Output of workflow.process_inbox.

    Returns:
        The same list with edited drafts.
    """
    for result in results:
        key = f"draft_{result['email']['id']}"
        if key in st.session_state:
            result["draft"] = st.session_state[key]
    return results


def main() -> None:
    """Build the whole Streamlit page."""
    st.set_page_config(page_title="AI Email Agent", layout="wide")
    st.title("Agentic AI Email Assistant")
    st.caption("Classifier Agent -> Drafting Agent (RAG) -> Review Agent -> Scheduler Agent | "
               "built with LangGraph + Mistral")

    # ---- Sidebar: choose emails and run -----------------------------------
    with st.sidebar:
        st.header("Inbox")
        use_samples = st.checkbox("Use sample emails (data/emails)", value=True)
        uploaded_files = st.file_uploader("Or upload emails", type=["eml", "txt"],
                                          accept_multiple_files=True)
        run_clicked = st.button("Run AI Agents", type="primary", width="stretch")
        st.divider()
        st.caption(f"Model: {config.MODEL_NAME}")
        st.caption(f"Auto-send threshold: {config.CONFIDENCE_THRESHOLD:.0%} confidence")

    if not config.MISTRAL_API_KEY or config.MISTRAL_API_KEY.startswith("your_"):
        st.warning("Add your MISTRAL_API_KEY to the .env file, then restart the app.")

    emails = get_inbox(use_samples, uploaded_files)
    if not emails:
        st.info("No emails yet. Tick 'Use sample emails' or upload .eml / .txt files.")
        return

    # ---- Run the agents ----------------------------------------------------
    if run_clicked:
        progress = st.progress(0.0, text="Starting agents...")

        def update_progress(done, total, email):
            progress.progress(done / total, text=f"Processed {done}/{total}: {email['subject']}")

        st.session_state["results"] = process_inbox(emails, on_progress=update_progress)
        progress.empty()

    # ---- Show everything ---------------------------------------------------
    inbox_tab, results_tab = st.tabs(["Incoming emails", "Responses & schedule"])

    with inbox_tab:
        show_inbox(emails)

    with results_tab:
        results = st.session_state.get("results")
        if not results:
            st.info("Click **Run AI Agents** in the sidebar to generate replies.")
            return

        show_summary(results)
        st.subheader("Replies (most urgent first)")
        for result in results:
            show_result(result)

        st.divider()
        if st.button("Save to outbox", type="primary"):
            outbox = save_outbox(apply_edits(results))
            st.success(f"Saved {len(outbox)} replies to {config.OUTBOX_FILE.name}")
        st.download_button(
            "Download outbox JSON",
            data=json.dumps(build_outbox(apply_edits(results)), indent=2),
            file_name="outbox.json",
            mime="application/json",
        )


main()
