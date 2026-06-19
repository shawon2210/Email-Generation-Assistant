"""
Email Generator Module
======================
Supports two generation strategies:

  Model A  —  openrouter/auto  +  Advanced Prompt
              (Role-Playing system persona + Few-Shot examples + Chain-of-Thought)

  Model B  —  openrouter/auto  +  Zero-Shot Baseline
              (No role, no examples, no CoT – plain single instruction)

Uses the OpenAI-compatible API via OpenRouter.
"""

import re
import time
from openai import OpenAI

from src.prompts import (
    SYSTEM_PROMPT,
    build_advanced_prompt,
    build_simple_prompt,
)

# ── Model configuration ─────────────────────────────────────────────────────
MODEL_A_ID = "openrouter/auto"
MODEL_B_ID = "openrouter/auto"

_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds


class EmailGenerator:
    """
    Generates professional emails using two distinct model/prompt strategies.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: float = 30.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        self._config_a = {"temperature": 0.3, "max_tokens": 1024}
        self._config_b = {"temperature": 0.3, "max_tokens": 1024}

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
        config: dict,
    ) -> str:
        """Call the model with retry on transient errors."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Build messages: system prompt + user prompt
                messages = []
                if "system_instruction" in config:
                    messages.append({"role": "system", "content": config["system_instruction"]})
                messages.append({"role": "user", "content": prompt})

                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=config.get("temperature", 0.3),
                    max_tokens=config.get("max_tokens", 1024),
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                exc_str = str(exc)
                if any(kwd in exc_str for kwd in ["API key expired", "API_KEY_INVALID", "Unauthorized"]):
                    raise ValueError(f"Invalid API key: {exc}")

                is_rate_limit = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str

                if attempt == _MAX_RETRIES:
                    return f"[GENERATION ERROR after {_MAX_RETRIES} attempts: {exc}]"

                if is_rate_limit:
                    wait = 30.0
                    print(f"\n    ⚠ Rate limit hit. Waiting {wait:.0f}s before retry...")
                else:
                    wait = _RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"\n    ⚠ API error (attempt {attempt}/{_MAX_RETRIES}): {exc}")
                    print(f"    Retrying in {wait:.0f}s ...")

                time.sleep(wait)
        return "[GENERATION ERROR]"
