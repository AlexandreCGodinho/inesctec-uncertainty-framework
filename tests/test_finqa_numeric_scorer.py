import math
import pytest

from src.scoring.finqa_numeric_scorer import finqa_numeric_scorer, _extract_numeric_answer, _candidates_from_base


class TestExtractNumericAnswer:
    def test_basic_number(self):
        result = _extract_numeric_answer("The answer is 42.")
        assert result == 42.0

    def test_percentage(self):
        result = _extract_numeric_answer("57.5%")
        assert result == 0.575

    def test_million(self):
        result = _extract_numeric_answer("3.5 million")
        assert abs(result - 3_500_000.0) < 1.0

    def test_billion(self):
        result = _extract_numeric_answer("2.1 billion")
        assert abs(result - 2_100_000_000.0) < 1.0

    def test_thousand(self):
        result = _extract_numeric_answer("1.5 thousand")
        assert abs(result - 1_500.0) < 1.0

    def test_no_number(self):
        result = _extract_numeric_answer("No numeric answer here.")
        assert result is None

    def test_empty(self):
        result = _extract_numeric_answer("")
        assert result is None


class TestFinqaNumericScorer:
    """Tests for finqa_numeric_scorer matching the 10 validation cases from docs/07."""

    def test_nke_2009_percent_rounding(self):
        """NKE/2009: 0.81372 vs 81.37%"""
        record = {"final_answer": "81.37%"}
        reference = {"exe_ans": "0.81372"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_abc_2005_percent(self):
        """ABC/2005: 0.9 vs 90.0%"""
        record = {"final_answer": "90.0%"}
        reference = {"exe_ans": "0.9"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_mro_2013_percent_trailing_year(self):
        """MRO/2013: 0.24744 vs 24.7%"""
        record = {"final_answer": "24.7%"}
        reference = {"exe_ans": "0.24744"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_aal_2016_percent_rounding(self):
        """AAL/2016: 0.13202 vs 13.2%"""
        record = {"final_answer": "13.2%"}
        reference = {"exe_ans": "0.13202"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_tmus_2017_plain_number(self):
        """TMUS/2017: 82352.94118 vs 82352.94"""
        record = {"final_answer": "82352.94"}
        reference = {"exe_ans": "82352.94118"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_mro_2003_thousand_unit(self):
        """MRO/2003: 1056.0 vs 1,056 (thousand unit)"""
        record = {"final_answer": "1,056"}
        reference = {"exe_ans": "1056.0"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_amt_2014_million_table_thousands(self):
        """AMT/2014: 33458.0 vs 63458 (human quirk: million in table-thousands)"""
        record = {"final_answer": "63458"}
        reference = {"exe_ans": "33458"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_etfc_2011_ratio_tail(self):
        """ETFC/2011: 4.6 vs 38.6 (to 1) — model wrong, expects 0.0"""
        record = {"final_answer": "38.6 to 1"}
        reference = {"exe_ans": "4.6"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 0.0

    def test_amt_2003_omits_percent(self):
        """AMT/2003: 0.33333 vs 33.3% — model omits the %"""
        record = {"final_answer": "33.3%"}
        reference = {"exe_ans": "0.33333"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 0.0, f"Expected match 0.0, got {result}"

    def test_jkhy_2019_yes_no(self):
        """JKHY/2019: "no" vs "no" — yes/no exact match"""
        record = {"final_answer": "no"}
        reference = {"exe_ans": "no"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_yes_no_affirmative(self):
        """Yes/no: affirmative answer matches 'yes'"""
        record = {"final_answer": "yes"}
        reference = {"exe_ans": "yes"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 1.0

    def test_yes_no_case_insensitive(self):
        """Yes/no: case insensitive match"""
        record = {"final_answer": "YES"}
        reference = {"exe_ans": "no"}
        result = finqa_numeric_scorer(record, reference)
        assert result["match"] == 0.0


class TestCandidatesFromBase:
    def test_no_unit_no_scaling(self):
        """Plain number with no unit word is NOT scaled."""
        cands = _candidates_from_base(1056.0, "")
        assert 1056.0 in cands
        assert 1_056_000.0 not in cands  # no automatic scaling

    def test_million_unit_candidates(self):
        """Million unit: candidates = {base, base/1e3, base/1e6}"""
        cands = _candidates_from_base(1_000_000.0, "million")
        assert 1_000_000.0 in cands  # raw
        assert 1_000.0 in cands  # thousands
        assert 1.0 in cands  # millions

    def test_thousand_unit_candidates(self):
        """Thousand unit: candidates = {base, base/1e3, base/1e6}"""
        cands = _candidates_from_base(1_000_000.0, "thousand")
        assert 1_000_000.0 in cands
        assert 1_000.0 in cands
        assert 1.0 in cands