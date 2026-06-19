"""
Email Generator Module
======================
Supports two generation strategies:

  Model A  –  gemini-2.0-flash  +  Advanced Prompt
              (Role-Playing system persona + Few-Shot examples + Chain-of-Thought)

  Model B  –  gemini-2.0-flash-lite  +  Zero-Shot Baseline
              (No role, no examples, no CoT – plain single instruction)

Uses the current google-genai SDK (google.genai).
"""

import re
import time
from google import genai
from google.genai import types

from src.prompts import (
    SYSTEM_PROMPT,
    build_advanced_prompt,
    build_simple_prompt,
)

# ── Model identifiers ──────────────────────────────────────────────────────
MODEL_A_ID = "gemini-2.0-flash"
MODEL_B_ID = "gemini-2.0-flash"  # Use same model for both to reduce rate limit pressure

_MAX_RETRIES = 5
_RETRY_DELAY = 15  # seconds


class EmailGenerator:
    """
    Generates professional emails using two distinct model/prompt strategies.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialise the generator with a google-genai client.

        Args:
            api_key: Google Generative AI API key.
        """
        self._client = genai.Client(api_key=api_key)

        # Generation configs
        self._config_a = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
        )
        self._config_b = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        intent: str,
        facts: list,
        tone: str,
        use_model_a: bool = True,
    ) -> dict:
        """
        Generate a professional email.

        Args:
            intent:      Core purpose of the email.
            facts:       List of key fact strings to incorporate.
            tone:        Desired tone / writing style.
            use_model_a: True → Model A (advanced); False → Model B (baseline).

        Returns:
            dict with keys: model, strategy, generated_email, prompt_used
        """
        if use_model_a:
            prompt   = build_advanced_prompt(intent, facts, tone)
            model_id = MODEL_A_ID
            config   = self._config_a
            strategy = "Advanced (Role-Playing + Few-Shot + Chain-of-Thought)"
        else:
            prompt   = build_simple_prompt(intent, facts, tone)
            model_id = MODEL_B_ID
            config   = self._config_b
            strategy = "Baseline (Zero-Shot, No System Role)"

        generated_text = self._call_with_retry(model_id, prompt, config)

        return {
            "model":           model_id,
            "strategy":        strategy,
            "generated_email": generated_text,
            "prompt_used":     prompt,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _call_with_retry(
        self,
        model_id: str,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """Call the model with rate-limit-aware retry on transient and 429 errors."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config,
                )
                return response.text.strip()
            except Exception as exc:
                exc_str = str(exc)
                if any(kwd in exc_str for kwd in ["API key expired", "API_KEY_INVALID", "API key not valid"]):
                    raise ValueError(f"Invalid or expired GOOGLE_API_KEY: {exc}")
                
                is_rate_limit = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
                
                if attempt == max_retries:
                    return (
                        f"[GENERATION ERROR after {max_retries} attempts: {exc}]"
                    )
                
                if is_rate_limit:
                    match = re.search(r"retry in ([\d.]+)s", exc_str, re.IGNORECASE)
                    if match:
                        wait = float(match.group(1)) + 5.0
                    else:
                        wait = 120.0
                    print(f"\n    ⚠ Rate limit hit. Waiting {wait:.2f}s before retry...")
                else:
                    wait = 15.0 * (2 ** (attempt - 1))
                    print(f"\n    ⚠ API error (attempt {attempt}/{max_retries}): {exc}")
                    print(f"    Retrying in {wait}s …")
                
                time.sleep(wait)
        return "[GENERATION ERROR]"
