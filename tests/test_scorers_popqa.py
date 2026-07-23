from src.scoring.popqa_scorer import PopqaScorer


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
            "match", "match_jaccard_word", "match_jaccard_bigram", "match_acronym", "score_popqa_raw"
        }

    def test_deterministic(self):
        r1 = self._score("Paris", "paris")
        r2 = self._score("Paris", "paris")
        assert r1 == r2
