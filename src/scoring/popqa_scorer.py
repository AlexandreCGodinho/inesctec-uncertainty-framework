import re
import string

from src.scoring.answer_span import extract_answer_span
from src.scoring.base import Scorer


class PopqaScorer(Scorer):
    def score(self, record: dict, reference: dict) -> dict:
        raw_prediction = record.get("final_answer", "")
        ref = reference.get("answer", reference.get("obj", reference.get("reference", "")))
        possible = reference.get("possible_answers", None)
        if possible is not None:
            if isinstance(possible, list):
                aliases = [str(a) for a in possible if a]
            else:
                import ast
                try:
                    parsed = ast.literal_eval(possible)
                    aliases = [str(a) for a in parsed if a] if isinstance(parsed, list) else [str(parsed)]
                except (ValueError, SyntaxError):
                    aliases = [ref]
        else:
            aliases = [ref]

        # Extract answer span per Doc 08 protocol
        span, category = extract_answer_span(raw_prediction)

        # Operational Abstention: Doc 08 Case A (needs_revision)
        if category == "needs_revision":
            return {
                "match": 0.0,
                "match_word_boundary": 0.0,
                "match_naive_substring": 0.0,
                "match_extracted": 0.0,
                "match_jaccard_word": 0.0,
                "match_jaccard_bigram": 0.0,
                "match_acronym": 0.0,
                "score_popqa_raw": 0.0,
                "extracted_answer": "",
                "abstain": True,
                "span_category": "needs_revision",
            }

        # For approved deliberation (Pattern 4) or unmoderated passthrough (Patterns 1-3)
        if category in ("approved", "empty_approved", "approved_fallback"):
            extracted = span
        else:
            extracted = _extract_answer(raw_prediction)

        raw_scores = _compute_scores(aliases, raw_prediction)
        extracted_scores = _compute_scores(aliases, extracted)
        
        # Naive substring containment (Mallen et al. baseline)
        valid_aliases = [a.lower() for a in aliases if a and a.strip()]
        naive_match = 1.0 if any(a in raw_prediction.lower() for a in valid_aliases) else 0.0

        # Primary metric: Word-boundary containment on full final_answer (Doc 09)
        wb_match = raw_scores["match"]
        em_match = extracted_scores["match"]

        return {
            "match": wb_match,
            "match_word_boundary": wb_match,
            "match_naive_substring": naive_match,
            "match_extracted": em_match,
            "match_jaccard_word": extracted_scores["match_jaccard_word"],
            "match_jaccard_bigram": extracted_scores["match_jaccard_bigram"],
            "match_acronym": extracted_scores["match_acronym"],
            "score_popqa_raw": wb_match,
            "extracted_answer": extracted,
            "abstain": bool(not extracted or not extracted.strip()),
            "span_category": category,
        }


def _compute_scores(aliases: list[str], prediction: str) -> dict:
    ref = aliases[0] if aliases else ""
    return {
        "match": 1.0 if _is_alias_match(aliases, prediction) else 0.0,
        "match_jaccard_word": _jaccard_word_score(ref, prediction),
        "match_jaccard_bigram": _jaccard_bigram_score(ref, prediction),
        "match_acronym": _acronym_score(ref, prediction),
    }


_NOISE_PATTERNS = [
    r"^(verdict|critique|##)",
    r"^[=\-*]{2,}$",
    r"^(i hope|let me know|feel free|if you have|please|thank|you.re welcome)",
]


def _extract_answer(text: str) -> str:
    if not text or not text.strip():
        return text or ""
    lines = text.strip().split("\n")

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_noise = any(re.search(p, stripped, re.IGNORECASE) for p in _NOISE_PATTERNS)
        if not is_noise:
            cleaned.append(stripped)

    if not cleaned:
        return text

    final_answer_lines = [l for l in cleaned if "final answer" in l.lower()]
    if final_answer_lines:
        last = final_answer_lines[-1]
        _, _, after = last.partition(":")
        if after.strip():
            return after.strip()

    for line in reversed(cleaned):
        if re.search(r"[a-zA-Z0-9]", line):
            return line

    return text


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return [w for w in words if len(w) > 1]


def _is_alias_match(aliases: list[str], prediction: str) -> bool:
    """Check if any alias matches in prediction using word-boundary lookarounds.

    Uses r"(?<!\\w)alias(?!\\w)" to avoid inside-word matches (e.g., "US" in "USA").
    """
    if not aliases:
        return True
    if not prediction:
        return False
    # Filter out empty aliases
    valid_aliases = [a for a in aliases if a and a.strip()]
    if not valid_aliases:
        return True  # vacuous truth: no valid aliases to match against
    pred_lower = prediction.lower()
    for alias in valid_aliases:
        alias_lower = alias.lower()
        # Use lookarounds to match only at word boundaries, not inside words
        pattern = r"(?<!\w)" + re.escape(alias_lower) + r"(?!\w)"
        if re.search(pattern, pred_lower):
            return True
    return False


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _jaccard_word_score(reference: str, prediction: str) -> float:
    ref_words = _normalize(reference)
    pred_words = _normalize(prediction)
    if not ref_words and not pred_words:
        return 1.0
    if not ref_words or not pred_words:
        return 0.0
    ref_set = set(ref_words)
    pred_set = set(pred_words)
    if ref_set == pred_set or ref_set.issubset(pred_set) or pred_set.issubset(ref_set):
        return 1.0
    j = _jaccard(ref_set, pred_set)
    return 1.0 if j >= 0.5 else 0.0


def _jaccard_bigram_score(reference: str, prediction: str) -> float:
    def bigrams(t: str) -> set[str]:
        t = re.sub(r"\s+", "", t.lower())
        return {t[i : i + 2] for i in range(len(t) - 1)}

    ba = bigrams(reference)
    bb = bigrams(prediction)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    j = _jaccard(ba, bb)
    return 1.0 if j >= 0.5 else 0.0


def _acronym_score(reference: str, prediction: str) -> float:
    ref_words = _normalize(reference)
    pred_words = _normalize(prediction)
    if len(ref_words) == 0 or len(pred_words) == 0:
        return 0.0
    if len(ref_words) == 1 and len(pred_words) == 1:
        return 0.0
    single, multi = (ref_words, pred_words) if len(ref_words) < len(pred_words) else (pred_words, ref_words)
    short_str = "".join(single)
    if len(short_str) > 5:
        return 0.0
    sig_words = [w for w in multi if len(w) > 2]
    if len(sig_words) < len(short_str):
        return 0.0
    return 1.0 if all(sig_words[i].startswith(c) for i, c in enumerate(short_str)) else 0.0
