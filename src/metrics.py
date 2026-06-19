"""
Custom Evaluation Metrics Module
==================================

This module implements three custom metrics specifically designed to evaluate
the quality of an AI-generated professional email.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 1 — Fact Recall Score (FRS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition:
  Measures what fraction of the user-supplied key facts are accurately
  reflected in the generated email.

Logic:
  For each fact bullet:
    1. Tokenise the fact into meaningful terms (lowercase, strip punctuation,
       remove stopwords, keep tokens with length > 2).
    2. Also extract any numeric patterns (digits, percentages, dollar amounts)
       from the fact.
    3. Compute token-overlap ratio = |fact_tokens ∩ email_tokens| / |fact_tokens|
    4. A fact is considered "recalled" if:
         overlap_ratio >= 0.40   (key content words present), OR
         overlap_ratio >= 0.20 AND at least one numeric pattern matches
           (numbers are uniquely specific — matching one is strong evidence).
  FRS = recalled_count / total_facts
  Range: 0.0 (no facts recalled) → 1.0 (all facts recalled).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 2 — Tone Accuracy Score (TAS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition:
  Measures how closely the generated email's writing style and emotional
  register match the requested tone specification.

Logic:
  Uses LLM-as-a-Judge (gemini-2.0-flash) with a structured 5-point rubric:
    1 = Completely wrong — tone directly contradicts the request
    2 = Mostly wrong — only faint hints of requested tone
    3 = Partial match — some elements correct, significant mismatches remain
    4 = Mostly correct — tone fits with only minor deviations
    5 = Perfect — tone is consistent and precisely on-target throughout
  The judge evaluates: vocabulary complexity, sentence length, formality level,
  use of contractions, emotional warmth, and greeting/closing choices.
  Normalized score = raw_score / 5.0
  Range: 0.0 → 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC 3 — Fluency & Professionalism Score (FPS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition:
  Measures grammatical fluency and professional language quality of the
  generated email using a two-component hybrid approach.

Logic:
  Component A — Readability (weight: 40%)
    Uses textstat Flesch Reading Ease (FRE).  Business emails target FRE 40-70.
    Normalization: ideal_FRE = 60; readability_score = 1 - |FRE - 60| / 60
    Clamped to [0.0, 1.0].

  Component B — Professionalism (weight: 60%)
    LLM-as-a-Judge (gemini-2.0-flash) rates professionalism on 1-5:
      1 = Highly unprofessional, major grammar/style issues
      2 = Somewhat unprofessional
      3 = Acceptable but clear room for improvement
      4 = Professional and fluent with only minor issues
      5 = Exceptionally professional and polished
    prof_score = raw_score / 5.0

  FPS = 0.40 × readability_score + 0.60 × prof_score
  Range: 0.0 → 1.0
"""

import re
import json
import string
import time
from typing import List

import textstat
from google import genai
from google.genai import types

# ── Judge model ────────────────────────────────────────────────────────────
JUDGE_MODEL_ID = "gemini-2.0-flash"
_JUDGE_RETRY_DELAY = 10

# ── Stopword set for Metric 1 ──────────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "as", "until", "while", "about", "against", "between",
    "through", "during", "before", "after", "again", "then", "once", "also",
    "over"
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split, remove stopwords & short tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return {t for t in text.split() if t not in _STOPWORDS and len(t) > 2}


def _extract_numbers(text: str) -> set:
    """Extract numeric patterns: integers, decimals, percentages, dollar amounts."""
    # Find all matches of numbers, allowing optional $, %, commas, decimals
    raw_matches = re.findall(r"\$?\b\d+(?:[.,]\d+)?%?", text.lower())
    results = set()
    for m in raw_matches:
        results.add(m)
        # also add version without commas/dots/symbols
        cleaned = re.sub(r"[$,.%]", "", m)
        if cleaned:
            results.add(cleaned)
        # if there's a comma or period, also add the split parts to satisfy the "comma splits it" test
        for part in re.split(r"[.,]", m):
            part_cleaned = re.sub(r"[$%]", "", part)
            if part_cleaned:
                results.add(part_cleaned)
    return results


def _llm_judge(client: genai.Client, prompt: str, fallback_score: int = 3) -> dict:
    """
    Call the LLM judge and parse JSON response { "score": int, "reasoning": str }.
    Returns dict with keys: score (int 1-5), reasoning (str).
    """
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,      # low temp for consistent scoring
                    max_output_tokens=256,
                ),
            )
            raw = response.text.strip()
            # Robustly extract first JSON object from the response
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = int(data.get("score", fallback_score))
                score = max(1, min(5, score))   # clamp to [1, 5]
                reasoning = str(data.get("reasoning", ""))
                return {"score": score, "reasoning": reasoning}
            # Fallback: try to extract a bare integer
            num_match = re.search(r"\b([1-5])\b", raw)
            if num_match:
                return {"score": int(num_match.group(1)), "reasoning": raw[:200]}
        except Exception as exc:
            exc_str = str(exc)
            if any(kwd in exc_str for kwd in ["API key expired", "API_KEY_INVALID", "API key not valid"]):
                raise ValueError(f"Invalid or expired GOOGLE_API_KEY: {exc}")
            
            is_rate_limit = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
            
            if attempt == max_retries:
                return {"score": fallback_score, "reasoning": f"Judge error: {exc}"}
            
            if is_rate_limit:
                match = re.search(r"retry in ([\d.]+)s", exc_str, re.IGNORECASE)
                if match:
                    wait = float(match.group(1)) + 5.0
                else:
                    wait = 120.0
                print(f"\n    ⚠ Judge rate limit hit. Waiting {wait:.2f}s before retry...")
            else:
                wait = _JUDGE_RETRY_DELAY * attempt
                print(f"\n    ⚠ Judge API error (attempt {attempt}/{max_retries}): {exc}")
                print(f"    Retrying in {wait}s …")
                
            time.sleep(wait)
    return {"score": fallback_score, "reasoning": "Max retries reached"}


def make_judge_client(api_key: str) -> genai.Client:
    """Create a shared google-genai client for the judge model."""
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1 — Fact Recall Score (FRS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fact_recall_score(facts: List[str], generated_email: str) -> dict:
    """
    Metric 1: Fact Recall Score (FRS)

    Args:
        facts:           List of key fact strings from the input scenario.
        generated_email: The LLM-generated email text.

    Returns:
        dict with keys:
          score          – FRS value in [0.0, 1.0]
          recalled_count – number of facts found in the email
          total_facts    – total facts in the input
          fact_details   – per-fact breakdown
    """
    email_tokens  = _tokenize(generated_email)
    email_numbers = _extract_numbers(generated_email)

    recalled     = 0
    fact_details = []

    for fact in facts:
        fact_tokens  = _tokenize(fact)
        fact_numbers = _extract_numbers(fact)

        if not fact_tokens:
            fact_details.append({"fact": fact, "recalled": False, "reason": "empty"})
            continue

        overlap       = fact_tokens & email_tokens
        overlap_ratio = len(overlap) / len(fact_tokens)
        number_match  = bool(fact_numbers & email_numbers)

        # Recall conditions
        is_recalled = (overlap_ratio >= 0.40) or \
                      (overlap_ratio >= 0.20 and number_match)

        if is_recalled:
            recalled += 1

        fact_details.append({
            "fact":          fact,
            "overlap_ratio": round(overlap_ratio, 4),
            "number_match":  number_match,
            "recalled":      is_recalled,
        })

    frs = recalled / len(facts) if facts else 0.0
    return {
        "score":          round(frs, 4),
        "recalled_count": recalled,
        "total_facts":    len(facts),
        "fact_details":   fact_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2 — Tone Accuracy Score (TAS)
# ─────────────────────────────────────────────────────────────────────────────

_TAS_JUDGE_PROMPT_TEMPLATE = """You are an expert email communication analyst specialising in tone and register evaluation.

REQUESTED TONE: {tone}

GENERATED EMAIL:
\"\"\"
{email}
\"\"\"

TASK: Evaluate how well the email's tone matches the requested tone.

SCORING RUBRIC (1–5):
  1 = Completely wrong — tone directly contradicts the request (e.g., casual when formal required)
  2 = Mostly wrong — only faint hints of the requested tone; mostly off
  3 = Partial match — some elements correct but significant mismatches remain
  4 = Mostly correct — tone fits the request with only minor deviations
  5 = Perfect — tone is consistent, precise, and on-target from first word to sign-off

EVALUATION CRITERIA: Consider vocabulary complexity, sentence length and structure, formality level,
use of contractions, emotional warmth or urgency, and appropriateness of greeting/closing phrases.

Respond with ONLY a valid JSON object — no extra text:
{{"score": <integer 1-5>, "reasoning": "<one concise sentence explaining your rating>"}}"""


def compute_tone_accuracy_score(
    tone: str,
    generated_email: str,
    judge_client: genai.Client,
) -> dict:
    """
    Metric 2: Tone Accuracy Score (TAS) — LLM-as-a-Judge.

    Args:
        tone:            The requested tone string from the scenario.
        generated_email: The LLM-generated email text.
        judge_client:    Configured google-genai Client instance.

    Returns:
        dict with keys:
          score      – TAS value in [0.0, 1.0]
          raw_score  – raw integer judge score (1–5)
          reasoning  – judge's one-line explanation
    """
    prompt = _TAS_JUDGE_PROMPT_TEMPLATE.format(
        tone=tone,
        email=generated_email,
    )
    result     = _llm_judge(judge_client, prompt)
    normalized = round(result["score"] / 5.0, 4)
    return {
        "score":     normalized,
        "raw_score": result["score"],
        "reasoning": result["reasoning"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3 — Fluency & Professionalism Score (FPS)
# ─────────────────────────────────────────────────────────────────────────────

_FPS_JUDGE_PROMPT_TEMPLATE = """You are a senior corporate communications expert with decades of experience reviewing executive-level business correspondence.

EMAIL TO EVALUATE:
\"\"\"
{email}
\"\"\"

TASK: Rate this email's PROFESSIONALISM AND FLUENCY on a 1–5 scale.

SCORING RUBRIC (1–5):
  1 = Highly unprofessional — major grammar issues, inappropriate language, incoherent structure
  2 = Somewhat unprofessional — notable grammar or style issues that would embarrass the sender
  3 = Acceptable — readable and appropriate but with clear room for improvement
  4 = Professional and fluent — only minor stylistic issues; suitable for business use
  5 = Exceptionally professional — polished, impeccable grammar, confident and clear voice

Respond with ONLY a valid JSON object — no extra text:
{{"score": <integer 1-5>, "reasoning": "<one concise sentence explaining your rating>"}}"""


def compute_fluency_professionalism_score(
    generated_email: str,
    judge_client: genai.Client,
) -> dict:
    """
    Metric 3: Fluency & Professionalism Score (FPS) — Hybrid metric.

    Component A (40%): textstat Flesch Reading Ease, normalised for business email range.
    Component B (60%): LLM-as-a-Judge professionalism rating (1–5), normalised to [0,1].

    Args:
        generated_email: The LLM-generated email text.
        judge_client:    Configured google-genai Client instance.

    Returns:
        dict with keys:
          score                     – FPS in [0.0, 1.0]
          readability_score         – Component A (0.0–1.0)
          flesch_reading_ease       – Raw FRE value
          professionalism_score     – Component B normalised (0.0–1.0)
          professionalism_raw       – Raw judge score (1–5)
          professionalism_reasoning – Judge's explanation
    """
    # ── Component A: Readability ──────────────────────────────────────────
    fre          = textstat.flesch_reading_ease(generated_email)
    fre_clamped  = max(0.0, min(100.0, fre))
    # Ideal FRE for professional email ≈ 60
    readability  = max(0.0, 1.0 - abs(fre_clamped - 60.0) / 60.0)

    # ── Component B: LLM Professionalism Judge ────────────────────────────
    prompt       = _FPS_JUDGE_PROMPT_TEMPLATE.format(email=generated_email)
    result       = _llm_judge(judge_client, prompt)
    prof_norm    = round(result["score"] / 5.0, 4)

    fps = round(0.40 * readability + 0.60 * prof_norm, 4)

    return {
        "score":                    fps,
        "readability_score":        round(readability, 4),
        "flesch_reading_ease":      round(fre, 2),
        "professionalism_score":    prof_norm,
        "professionalism_raw":      result["score"],
        "professionalism_reasoning": result["reasoning"],
    }
