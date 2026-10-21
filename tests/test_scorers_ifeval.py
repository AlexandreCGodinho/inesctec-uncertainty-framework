from src.scoring.ifeval_scorer import IfevalScorer


class TestIfevalScorer:
    def _score(self, response: str, prompt: str) -> dict:
        return IfevalScorer().score(
            record={"final_answer": response},
            reference={"prompt": prompt},
        )

    def test_min_length_strict_pass(self):
        result = self._score("Hello World!!", "Write at least 10 characters")
        assert result["strict_prompt"] == 1.0
        assert result["loose_prompt"] == 1.0
        assert result["strict_instruction"] == 1.0
        assert result["loose_instruction"] == 1.0

    def test_min_length_strict_fail_loose_pass(self):
        result = self._score("Bon jour", "Write at least 10 characters")
        assert result["strict_prompt"] == 0.0
        assert result["strict_instruction"] == 0.0
        assert result["loose_prompt"] == 1.0
        assert result["loose_instruction"] == 1.0

    def test_all_constraints_pass(self):
        result = self._score(
            "First, I think therefore I am.",
            "Include the word 'therefore' and start with 'First'",
        )
        assert result["strict_prompt"] == 1.0

    def test_one_constraint_fails(self):
        result = self._score(
            "hello",
            "Include the word 'hello' and write at least 10 characters",
        )
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 1.0
        assert result["strict_instruction"] == 0.5
        assert result["loose_instruction"] == 0.5

    def test_empty_response(self):
        result = self._score("", "Include the word 'hello'")
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 0.0

    def test_no_parsable_constraints(self):
        result = self._score("Nature is beautiful", "Write a poem about nature")
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 0.0
        assert result["strict_instruction"] == 0.0
        assert result["loose_instruction"] == 0.0

    def test_exact_bullets_loose_pass(self):
        result = self._score("- A\n- B\n- C\n- D", "Use exactly 3 bullet points")
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 1.0
        assert result["strict_instruction"] == 0.0
        assert result["loose_instruction"] == 1.0

    def test_exact_bullets_strict_pass(self):
        result = self._score("- A\n- B\n- C", "Use exactly 3 bullet points")
        assert result["strict_prompt"] == 1.0
        assert result["loose_prompt"] == 1.0

    def test_min_words(self):
        result = self._score("The cat sat on mat", "Write at least 5 words")
        assert result["strict_prompt"] == 1.0

    def test_max_words_strict_fail_loose_pass(self):
        result = self._score("The cat sat on the mat", "Write at most 5 words")
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 1.0

    def test_contains_word(self):
        result = self._score("I think therefore I am", "Include the word therefore")
        assert result["strict_prompt"] == 1.0

    def test_contains_word_case_insensitive(self):
        result = self._score("therefore is used", "Include the word Therefore")
        assert result["strict_prompt"] == 1.0

    def test_starts_with(self):
        result = self._score("Once upon a time", "Start with Once upon")
        assert result["strict_prompt"] == 1.0

    def test_ends_with(self):
        result = self._score("This is the end", "End with the end")
        assert result["strict_prompt"] == 1.0

    def test_min_sentences(self):
        result = self._score("First. Second. Third.", "Write at least 3 sentences")
        assert result["strict_prompt"] == 1.0

    def test_min_sentences_strict_fail_loose_pass(self):
        result = self._score("First. Second.", "Write at least 3 sentences")
        assert result["strict_prompt"] == 0.0
        assert result["loose_prompt"] == 1.0
        assert result["strict_instruction"] == 0.0
        assert result["loose_instruction"] == 1.0

    def test_all_four_values_returned(self):
        result = self._score("hello", "Include the word 'hello'")
        assert {
            "strict_prompt", "loose_prompt",
            "strict_instruction", "loose_instruction",
        }.issubset(set(result.keys()))

    def test_deterministic(self):
        r1 = self._score("hello", "Include the word 'hello'")
        r2 = self._score("hello", "Include the word 'hello'")
        assert r1 == r2
