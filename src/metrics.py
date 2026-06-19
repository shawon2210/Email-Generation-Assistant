"""
Custom Evaluation Metrics Module
==================================
Implements three custom metrics for evaluating AI-generated professional emails.

METRIC 1 — Fact Recall Score (FRS)
  Token overlap + numeric matching. Range: 0.0 → 1.0

METRIC 2 — Tone Accuracy Score (TAS)
  LLM-as-a-Judge with 5-point rubric. Range: 0.0 → 1.0

METRIC 3 — Fluency & Professionalism Score (FPS)
  Hybrid: 40% textstat readability + 60% LLM judge. Range: 0.0 → 1.0
"""

import re
import json
import string
import time
from typing import List

import textstat
from openai import OpenAI

# ── Judge model ─────────────────────────────────────────────────────────────
JUDGE_MODEL_ID = "openrouter/auto"
_JUDGE_RETRY_DELAY = 5

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
    "over",
}


# ── Helper utilities ────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split, remove stopwords & short tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return {t for t in text.split() if t not in _STOPWORDS and len(t) > 2}


def _extract_numbers(text: str) -> set:
    """Extract numeric patterns: integers, decimals, percentages, dollar amounts."""
    raw_matches = re.findall(r"\$?\b\d+(?:[.,]\d+)?%?", text.lower())
    results = set()
    for m in raw_matches:
        results.add(m)
        cleaned = re.sub(r"[$,.%]", "", m)
        if cleaned:
            results.add(cleaned)
        for part in re.split(r"[.,]", m):
            part_cleaned = re.sub(r"[$%]", "", part)
            if part_cleaned:
                results.add(part_cleaned)
    return results


def make_judge_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenAI:
    """Create a shared OpenAI-compatible client for the judge model."""
    return OpenAI(api_key=api_key, base_url=base_url)


def _llm_judge(client: OpenAI, prompt: str, fallback_score: int = 3) -> dict:
    """
    Call the LLM judge and parse JSON response { "score": int, "reasoning": str }.
    Returns dict with keys: score (int 1-5), reasoning (str).
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            raw = (response.choices[0].message.content or "").strip()
            # Robustly extract first JSON object
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = int(data.get("score", fallback_score))
                score = max(1, min(5, score))
                reasoning = str(data.get("reasoning", ""))
                return {"score": score, "reasoning": reasoning}
            # Fallback: try to extract a bare integer
            num_match = re.search(r"\b([1-5])\b", raw)
            if num_match:
                return {"score": int(num_match.group(1)), "reasoning": raw[:200]}
        except Exception as exc:
            if attempt == max_retries:
                return {"score": fallback_score, "reasoning": f"Judge error: {exc}"}
            time.sleep(_JUDGE_RETRY_DELAY * attempt)
    return {"score": fallback_score, "reasoning": "Max retries reached"}


# ── METRIC 1 — Fact Recall Score (FRS) ─────────────────────────────────────

def compute_fact_recall_score(facts: List[str], generated_email: str) -> dict:
    """
    Metric 1: Fact Recall Score (FRS)
    Range: 0.0 → 1.0
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


# ── METRIC 2 — Tone Accuracy Score (TAS) ───────────────────────────────────

_TAS_JUDGE_PROMPT_TEMPLATE = """You are an expert email communication analyst specialising in tone and register evaluation.

REQUESTED TONE: {tone}

GENERATED EMAIL:
\"\"\"
{email}
\"\"\"

TASK: Evaluate how well the email's tone matches the requested tone.

SCORING RUBRIC (1–5):
  1 = Completely wrong — tone directly contradicts the request
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
    judge_client: OpenAI,
) -> dict:
    """
    Metric 2: Tone Accuracy Score (TAS) — LLM-as-a-Judge.
    Range: 0.0 → 1.0
    """
    prompt = _TAS_JUDGE_PROMPT_TEMPLATE.format(tone=tone, email=generated_email)
    result = _llm_judge(judge_client, prompt)
    normalized = round(result["score"] / 5.0, 4)
    return {
        "score":     normalized,
        "raw_score": result["score"],
        "reasoning": result["reasoning"],
    }


# ── METRIC 3 — Fluency & Professionalism Score (FPS) ───────────────────────

_FPS_JUDGE_PROMPT_TEMPLATE = """You are a senior corporate communications expert with decades of experience reviewing executive-level business correspondence.

EMAIL TO EVALUATE:
\"\"\"
{email}
\"\"\"

TASK: Rate this email's PROFESSIONALISM AND FLUENCY on a 1–5 scale.

SCORING RUBRIC (1–5):
  1 = Highly unprofessional — major grammar issues, inappropriate language, incoherent structure
  2 = Somewhat unprofessional — notable grammar or style issues
  3 = Acceptable — readable and appropriate but with clear room for improvement
  4 = Professional and fluent — only minor stylistic issues; suitable for business use
  5 = Exceptionally professional — polished, impeccable grammar, confident and clear voice

Respond with ONLY a valid JSON object — no extra text:
{{"score": <integer 1-5>, "reasoning": "<one concise sentence explaining your rating>"}}"""


def compute_fluency_professionalism_score(
    generated_email: str,
    judge_client: OpenAI,
) -> dict:
    """
    Metric 3: Fluency & Professionalism Score (FPS) — Hybrid metric.
    Component A (40%): textstat Flesch Reading Ease.
    Component B (60%): LLM-as-a-Judge professionalism rating.
    Range: 0.0 → 1.0
    """
    # Component A: Readability
    fre         = textstat.flesch_reading_ease(generated_email)
    fre_clamped = max(0.0, min(100.0, fre))
    readability = max(0.0, 1.0 - abs(fre_clamped - 60.0) / 60.0)

    # Component B: LLM Professionalism Judge
    prompt    = _FPS_JUDGE_PROMPT_TEMPLATE.format(email=generated_email)
    result    = _llm_judge(judge_client, prompt)
    prof_norm = round(result["score"] / 5.0, 4)

    fps = round(0.40 * readability + 0.60 * prof_norm, 4)

    return {
        "score":                    fps,
        "readability_score":        round(readability, 4),
        "flesch_reading_ease":      round(fre, 2),
        "professionalism_score":    prof_norm,
        "professionalism_raw":      result["score"],
        "professionalism_reasoning": result["reasoning"],
    }
