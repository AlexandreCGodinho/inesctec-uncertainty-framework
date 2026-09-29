import re
from typing import Tuple

VERDICT_REGEX = re.compile(
    r"^(?:\*\**)?VERDICT\s*:\s*(approved|needs_revision)",
    re.IGNORECASE | re.MULTILINE,
)


def extract_answer_span(text: str) -> Tuple[str, str]:
    """
    Extract answer span from telemetry final_answer based on VERDICT: line position.

    Returns (extracted_text, category) where category is one of:
      - "no_verdict": no VERDICT: line present (passthrough)
      - "approved": VERDICT: approved found, text after verdict extracted
      - "needs_revision": VERDICT: needs_revision found, empty span
      - "empty_approved": VERDICT: approved but no text after, fallback to before
      - "approved_fallback": VERDICT: approved, text after empty, used text before

    Logic:
      1. Find all VERDICT: lines anchored at start of line.
      2. If last VERDICT is "needs_revision" → category "needs_revision", span "".
      3. If last VERDICT is "approved":
           a. Extract text after the last VERDICT: line.
           b. If text after exists (non-empty after strip) → category "approved", return it.
           c. If text after is empty → fallback to text before the first VERDICT: line.
              - If text before exists → category "empty_approved", return it.
              - If no text before → category "approved_fallback", return "".
      4. If no VERDICT: line present → category "no_verdict", return text byte-identical.
    """
    if not text or not text.strip():
        return ("", "no_verdict")

    # Find all VERDICT: lines anchored at start of line
    verdict_positions = list(VERDICT_REGEX.finditer(text))

    if not verdict_positions:
        # No VERDICT: line present — passthrough byte-identical
        return (text, "no_verdict")

    # Process based on the last VERDICT: line (most recent)
    last_verdict = verdict_positions[-1]
    verdict_text = last_verdict.group(1).lower()  # "approved" or "needs_revision"
    verdict_start = last_verdict.end()  # position after "VERDICT: <value>"

    # Get text after the last VERDICT: line
    after_text = text[verdict_start:]

    if verdict_text == "needs_revision":
        # Case A: needs_revision → empty span, no fallback, abstention flag
        return ("", "needs_revision")

    # verdict_text == "approved"
    # Try text after the last VERDICT:
    after_stripped = after_text.strip()

    if after_stripped:
        # Case Ba: text after exists → extract it, category "approved"
        return (after_stripped, "approved")

    # Case Bb: text after is empty → fallback to text before the first VERDICT: line
    first_verdict = verdict_positions[0]
    before_text = text[:first_verdict.start()]

    if before_text.strip():
        # Fallback: extract text before first VERDICT:
        return (before_text.strip(), "empty_approved")

    # No text before either → approved but nothing to extract
    return ("", "approved_fallback")