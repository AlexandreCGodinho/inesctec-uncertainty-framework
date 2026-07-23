from src.scoring.popqa_scorer import PopqaScorer, _extract_answer


class TestPopqaScorer:
    def _score(self, prediction: str, reference: str) -> dict:
        return PopqaScorer().score(
            record={"final_answer": prediction},
            reference={"answer": reference},
        )

    def test_headline_exact_match(self):
        result = self._score("Paris", "Paris")
        assert result["match"] == 1.0

    def test_headline_case_insensitive(self):
        result = self._score("paris", "Paris")
        assert result["match"] == 1.0

    def test_headline_punctuation_stripped(self):
        result = self._score("Paris!", "Paris")
        assert result["match"] == 1.0

    def test_headline_fails_on_word_subset(self):
        result = self._score("It is Paris", "Paris")
        assert result["match"] == 0.0

    def test_headline_fails_on_alias(self):
        result = self._score("USA", "United States of America")
        assert result["match"] == 0.0

    def test_headline_empty_prediction(self):
        result = self._score("", "Paris")
        assert result["match"] == 0.0

    def test_headline_empty_reference(self):
        result = self._score("any answer", "")
        assert result["match"] == 1.0

    def test_headline_known_correct(self):
        result = self._score("Albert Einstein", "Albert Einstein")
        assert result["match"] == 1.0

    def test_secondary_jaccard_word_contains(self):
        result = self._score("It is Paris", "Paris")
        assert result["match_jaccard_word"] == 1.0

    def test_secondary_jaccard_bigram_match(self):
        result = self._score("New York City", "New York")
        assert result["match_jaccard_bigram"] == 1.0

    def test_secondary_acronym_match(self):
        result = self._score("USA", "United States of America")
        assert result["match_acronym"] == 1.0

    def test_secondary_jaccard_word_low(self):
        result = self._score("Albert Einstein", "Albert Camus")
        assert result["match_jaccard_word"] == 0.0

    def test_all_keys_returned(self):
        result = self._score("Paris", "Paris")
        assert set(result.keys()) == {
            "match", "match_jaccard_word", "match_jaccard_bigram",
            "match_acronym", "score_popqa_raw",
        }

    def test_deterministic(self):
        r1 = self._score("Paris", "paris")
        r2 = self._score("Paris", "paris")
        assert r1 == r2


class TestExtractAnswer:
    def test_final_answer_simple(self):
        assert _extract_answer("FINAL ANSWER: Paris") == "Paris"

    def test_final_answer_last_wins(self):
        text = "Analysis...\nFINAL ANSWER: Paris\nFINAL ANSWER: London"
        assert _extract_answer(text) == "London"

    def test_noise_removed_before_final_answer(self):
        text = "I hope this helps.\nFINAL ANSWER: Einstein"
        assert _extract_answer(text) == "Einstein"

    def test_verdict_line_removed(self):
        text = "VERDICT: correct\nFINAL ANSWER: Einstein"
        assert _extract_answer(text) == "Einstein"

    def test_noise_only_returns_raw(self):
        text = "I hope this helps.\nLet me know."
        assert _extract_answer(text) == text

    def test_last_substantive_fallback(self):
        assert _extract_answer("A\nB\nC") == "C"

    def test_numeric_answer(self):
        assert _extract_answer("FINAL ANSWER: 42") == "42"

    def test_empty_string(self):
        assert _extract_answer("") == ""

    def test_no_final_answer_no_noise(self):
        assert _extract_answer("Just a random sentence") == "Just a random sentence"

    def test_hash_line_removed(self):
        text = "## Analysis\nThe capital is Paris."
        assert _extract_answer(text) == "The capital is Paris."

    def test_heading_line_removed(self):
        text = "## Step 1\nParis"
        assert _extract_answer(text) == "Paris"

    def test_dash_line_removed(self):
        text = "---\nParis"
        assert _extract_answer(text) == "Paris"

    def test_decorative_line_removed(self):
        text = "***\nParis"
        assert _extract_answer(text) == "Paris"

    def test_whitespace_line_skipped(self):
        text = "A\n  \nB"
        assert _extract_answer(text) == "B"

    def test_diagnostic_raw_match_preserved(self):
        result = PopqaScorer().score(
            record={"final_answer": "Paris"},
            reference={"answer": "Paris"},
        )
        assert result["score_popqa_raw"] == 1.0

    def test_diagnostic_raw_no_match(self):
        result = PopqaScorer().score(
            record={"final_answer": "London"},
            reference={"answer": "Paris"},
        )
        assert result["score_popqa_raw"] == 0.0

    def test_extraction_improves_raw(self):
        result = PopqaScorer().score(
            record={"final_answer": "FINAL ANSWER: Paris"},
            reference={"answer": "Paris"},
        )
        assert result["match"] == 1.0
        assert result["score_popqa_raw"] == 0.0

    def test_extraction_answer_on_next_line(self):
        result = PopqaScorer().score(
            record={"final_answer": "FINAL ANSWER: \nParis"},
            reference={"answer": "Paris"},
        )
        assert result["match"] == 1.0
        assert result["score_popqa_raw"] == 0.0

    def test_extraction_answer_on_next_line_no_space(self):
        result = PopqaScorer().score(
            record={"final_answer": "FINAL ANSWER:\nParis"},
            reference={"answer": "Paris"},
        )
        assert result["match"] == 1.0
