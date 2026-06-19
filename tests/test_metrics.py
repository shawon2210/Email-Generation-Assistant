"""
Unit Tests — Custom Metrics
============================
Tests Metric 1 (FRS) with synthetic inputs to validate correctness.
Metrics 2 and 3 (TAS, FPS) involve live LLM calls so they are tested
via integration-style smoke tests gated behind a --integration flag.

Run:
  pytest tests/test_metrics.py -v
  pytest tests/test_metrics.py -v --integration   (requires GOOGLE_API_KEY)
"""

import pytest
from src.metrics import (
    _tokenize,
    _extract_numbers,
    compute_fact_recall_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_removes_stopwords(self):
        tokens = _tokenize("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_lowercases(self):
        tokens = _tokenize("Meeting SCHEDULED For June")
        assert "meeting" in tokens
        assert "scheduled" in tokens
        assert "june" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("a an is to in of")
        # All are stopwords or short — should be empty or very small
        assert len(tokens) == 0

    def test_empty_string(self):
        assert _tokenize("") == set()


class TestExtractNumbers:
    def test_extracts_integers(self):
        nums = _extract_numbers("We have 200 employees and 3 teams")
        assert "200" in nums
        assert "3" in nums

    def test_extracts_percentages(self):
        nums = _extract_numbers("Reduce costs by 30% this quarter")
        assert "30%" in nums

    def test_extracts_dollar_amounts(self):
        nums = _extract_numbers("Budget is $50,000")
        assert "50" in nums  # comma splits it

    def test_empty_string(self):
        assert _extract_numbers("") == set()


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Fact Recall Score (FRS)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactRecallScore:

    # ── Perfect recall ──────────────────────────────────────────────────────
    def test_all_facts_recalled(self):
        facts = [
            "The meeting was scheduled for June 15th",
            "Three action items were assigned to the team",
            "The project budget is $50,000",
        ]
        email = (
            "Subject: Follow-Up from June 15th Meeting\n\n"
            "Dear Team,\n\n"
            "Following our meeting on June 15th, three action items were assigned "
            "to the team. The total project budget remains $50,000.\n\n"
            "Best regards,\nJohn"
        )
        result = compute_fact_recall_score(facts, email)
        assert result["score"] == 1.0
        assert result["recalled_count"] == 3
        assert result["total_facts"] == 3

    # ── Partial recall ──────────────────────────────────────────────────────
    def test_partial_recall(self):
        facts = [
            "The delivery was delayed by three days",
            "We are offering a 15% discount on the next order",
            "The order number is 4521",
        ]
        email = (
            "Dear Customer,\n\n"
            "We sincerely apologise for the inconvenience caused by the delay "
            "in processing your order. We value your business and hope to serve "
            "you better in the future.\n\n"
            "Best regards"
        )
        result = compute_fact_recall_score(facts, email)
        # "delayed" / "delay" should partially match; 15%, 4521 absent
        assert result["score"] < 1.0
        assert result["total_facts"] == 3

    # ── Zero recall ─────────────────────────────────────────────────────────
    def test_zero_recall(self):
        facts = [
            "Revenue grew by 42% in Q3",
            "The partnership involves 1 million active users",
        ]
        email = "Hi there! Hope you are doing well. Let me know if you need anything."
        result = compute_fact_recall_score(facts, email)
        assert result["score"] == 0.0
        assert result["recalled_count"] == 0

    # ── Number matching aids recall ─────────────────────────────────────────
    def test_number_match_boosts_recall(self):
        facts = ["The project budget is $50,000"]
        email = (
            "As discussed, we have allocated a total of 50000 dollars "
            "for the upcoming initiative."
        )
        result = compute_fact_recall_score(facts, email)
        # Should recall via number match even if token overlap is low
        assert result["recalled_count"] == 1

    # ── Empty facts list ────────────────────────────────────────────────────
    def test_empty_facts_list(self):
        result = compute_fact_recall_score([], "Any email content here.")
        assert result["score"] == 0.0
        assert result["total_facts"] == 0

    # ── Single fact ─────────────────────────────────────────────────────────
    def test_single_fact_recalled(self):
        facts = ["The next meeting is scheduled for July 1st"]
        email = "Please note our follow-up meeting is on July 1st."
        result = compute_fact_recall_score(facts, email)
        assert result["score"] == 1.0

    def test_single_fact_not_recalled(self):
        facts = ["The partnership spans five years"]
        email = "We look forward to working with you in the future."
        result = compute_fact_recall_score(facts, email)
        assert result["score"] == 0.0

    # ── Score is between 0 and 1 ────────────────────────────────────────────
    def test_score_bounded(self):
        facts = ["Fact one about a project", "Fact two about a budget of $20,000"]
        email = "We are excited about the project and have allocated $20,000."
        result = compute_fact_recall_score(facts, email)
        assert 0.0 <= result["score"] <= 1.0

    # ── Fact details structure ──────────────────────────────────────────────
    def test_fact_details_structure(self):
        facts = ["The deadline is June 20th", "The team has 5 members"]
        email = "The project deadline is June 20th and the team consists of 5 members."
        result = compute_fact_recall_score(facts, email)
        assert "fact_details" in result
        assert len(result["fact_details"]) == 2
        for detail in result["fact_details"]:
            assert "fact" in detail
            assert "overlap_ratio" in detail
            assert "recalled" in detail

    # ── Case insensitivity ──────────────────────────────────────────────────
    def test_case_insensitive_matching(self):
        facts = ["BUDGET IS FIFTY THOUSAND DOLLARS"]
        email = "The budget allocated is fifty thousand dollars for this initiative."
        result = compute_fact_recall_score(facts, email)
        assert result["recalled_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (skipped unless --integration flag is passed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def integration(request):
    return request.config.getoption("--integration")


class TestToneAccuracyScoreIntegration:
    def test_formal_tone_detected(self, integration):
        if not integration:
            pytest.skip("Pass --integration to run LLM-based tests")

        import os
        from dotenv import load_dotenv
        from src.metrics import compute_tone_accuracy_score, make_judge_client

        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not found in .env")
        client = make_judge_client(api_key)

        formal_email = (
            "Dear Mr. Johnson,\n\nI am writing to formally notify you of the "
            "scheduled performance review on June 25th. Please bring your "
            "completed self-assessment form.\n\nYours sincerely,\nHR Department"
        )
        result = compute_tone_accuracy_score("Formal and professional", formal_email, client)
        assert 0.0 <= result["score"] <= 1.0
        assert result["score"] >= 0.6  # should score well for formal tone
        assert "raw_score" in result
        assert "reasoning" in result


class TestFluencyProfessionalismScoreIntegration:
    def test_professional_email_scores_well(self, integration):
        if not integration:
            pytest.skip("Pass --integration to run LLM-based tests")

        import os
        from dotenv import load_dotenv
        from src.metrics import compute_fluency_professionalism_score, make_judge_client

        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not found in .env")
        client = make_judge_client(api_key)

        email = (
            "Dear Team,\n\nI hope this message finds you well. I am pleased to "
            "confirm that our project is progressing on schedule. Please find the "
            "weekly status update below.\n\nBest regards,\n[Name]"
        )
        result = compute_fluency_professionalism_score(email, client)
        assert 0.0 <= result["score"] <= 1.0
        assert "readability_score" in result
        assert "flesch_reading_ease" in result
        assert "professionalism_score" in result

