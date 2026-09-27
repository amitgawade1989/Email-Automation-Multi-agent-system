"""
email_loader.py - Reads customer emails from .eml and .txt files.

Both formats use the same layout (headers, a blank line, then the body),
so Python's built-in `email` package can parse both of them.
"""

from email import message_from_string, policy
from email.utils import parseaddr
from pathlib import Path

SUPPORTED_EXTENSIONS = (".eml", ".txt")


def parse_email_text(raw_text: str, file_name: str) -> dict:
    """
    Turn the raw text of one email into a simple dictionary.

    Args:
        raw_text:  Full email text, e.g. "From: ...\\nSubject: ...\\n\\nHello..."
        file_name: Name of the file it came from (used as a unique id).

    Returns:
        A dict with keys: id, sender_name, sender_email, subject, date, body.
    """
    message = message_from_string(raw_text, policy=policy.default)

    # "Rahul Mehta <rahul@example.com>"  ->  ("Rahul Mehta", "rahul@example.com")
    sender_name, sender_email = parseaddr(message.get("From", ""))

    # get_body() finds the plain-text part, even inside multipart .eml files.
    body_part = message.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part else ""

    return {
        "id": file_name,
        "sender_name": sender_name or sender_email or "Customer",
        "sender_email": sender_email,
        "subject": message.get("Subject", "(no subject)"),
        "date": message.get("Date", ""),
        "body": body.strip(),
    }


def load_emails(folder: Path) -> list[dict]:
    """
    Load every .eml and .txt email from a folder.

    Args:
        folder: Path to the folder that holds the email files.

    Returns:
        A list of email dicts (see parse_email_text), sorted by file name.
    """
    emails = []
    for file_path in sorted(Path(folder).iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            raw_text = file_path.read_text(encoding="utf-8")
            emails.append(parse_email_text(raw_text, file_path.name))
    return emails
