import re
import string

from src.scoring.base import Scorer


class PopqaScorer(Scorer):
    def score(self, record: dict, reference: dict) -> dict:
        prediction_raw = record.get("final_answer", "")
        prediction_extracted = extract_answer_for_popqa(prediction_raw)
        
        ref = reference.get("answer", reference.get("reference", ""))
        result = {}
        
        # New extraction-based headline metric
        result["match"] = 1.0 if _is_alias_match(ref, prediction_extracted) else 0.0
        
        # Keep old metric as diagnostic column
        result["score_popqa_raw"] = 1.0 if _is_alias_match(ref, prediction_raw) else 0.0
        
        # Diagnostic metrics (using extracted text for accuracy)
        result["match_jaccard_word"] = _jaccard_word_score(ref, prediction_extracted)
        result["match_jaccard_bigram"] = _jaccard_bigram_score(ref, prediction_extracted)
        result["match_acronym"] = _acronym_score(ref, prediction_extracted)
        return result

def extract_answer_for_popqa(raw: str) -> str:
    if not raw:
        return ""
    
    lines = raw.split("\n")
    filtered_lines = []
    
    # 1. Remove noise lines
    for line in lines:
        s_line = line.strip()
        # skip VERDICT:, CRITIQUE:, ## or punctuation-only
        if s_line.startswith("VERDICT:") or s_line.startswith("CRITIQUE:") or s_line.startswith("##"):
            continue
        # Check if contains at least one letter or digit
        if not any(char.isalnum() for char in s_line):
            continue
        filtered_lines.append(s_line)
        
    # 2. On FINAL ANSWER:, keep line content after colon (last one wins)
    for i in range(len(filtered_lines) - 1, -1, -1):
        if filtered_lines[i].startswith("FINAL ANSWER:"):
            return filtered_lines[i].split(":", 1)[1].strip()
            
    # 3. Return the last remaining line that contains a letter or digit
    if filtered_lines:
        return filtered_lines[-1]
        
    # 4. If none, return raw unchanged
    return raw


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return [w for w in words if len(w) > 1]


def _is_alias_match(reference: str, prediction: str) -> bool:
    ref_words = _normalize(reference)
    pred_words = _normalize(prediction)
    if not ref_words and not pred_words:
        return True
    if not ref_words:
        return True
    if not pred_words:
        return False
    return ref_words == pred_words


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
