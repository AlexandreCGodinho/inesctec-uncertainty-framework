import pytest

from src.scoring.answer_span import extract_answer_span


class TestExtractAnswerSpan:
    def test_no_verdict_passthrough(self):
        """Case C: no VERDICT line -> byte-identical passthrough."""
        text = "The answer is 42."
        span, cat = extract_answer_span(text)
        assert cat == "no_verdict"
        assert span == text

    def test_needs_revision_no_fallback(self):
        """Case A: VERDICT: needs_revision -> empty span, no fallback."""
        text = "Some analysis.\nVERDICT: needs_revision\nMore text."
        span, cat = extract_answer_span(text)
        assert cat == "needs_revision"
        assert span == ""

    def test_approved_with_text_after(self):
        """Case Ba: VERDICT: approved with text after -> extract after."""
        text = "Reasoning...\nVERDICT: approved\nThe answer is 42."
        span, cat = extract_answer_span(text)
        assert cat == "approved"
        assert span == "The answer is 42."

    def test_approved_fallback_before(self):
        """Case Bb: VERDICT: approved alone -> fallback to text before."""
        text = "The answer is 42.\nVERDICT: approved"
        span, cat = extract_answer_span(text)
        assert cat == "empty_approved"
        assert span == "The answer is 42."

    def test_approved_fallback_no_before(self):
        """Case B: VERDICT: approved alone, nothing before -> empty fallback."""
        text = "VERDICT: approved"
        span, cat = extract_answer_span(text)
        assert cat == "approved_fallback"
        assert span == ""

    def test_anti_collision_final_verdit_title(self):
        """Anti-collision: 'Final Verdict:' section title should be ignored."""
        text = "Analysis...\nFinal Verdict: Some title\nVERDICT: approved\nThe answer."
        span, cat = extract_answer_span(text)
        assert cat == "approved"
        assert span == "The answer."

    def test_anti_collision_prompt_citation(self):
        """Anti-collision: prompt citation with 'VERDICT:' should be ignored."""
        text = 'The prompt said "VERDICT: correct".\nVERDICT: approved\nReal answer.'
        span, cat = extract_answer_span(text)
        assert cat == "approved"
        assert span == "Real answer."

    def test_empty_string(self):
        """Empty input -> no_verdict, empty span."""
        span, cat = extract_answer_span("")
        assert cat == "no_verdict"
        assert span == ""

    def test_only_verdict_line(self):
        """Only VERDICT line, no other text -> approved_fallback (no before/after text)."""
        text = "VERDICT: approved"
        span, cat = extract_answer_span(text)
        assert cat == "approved_fallback"
        assert span == ""