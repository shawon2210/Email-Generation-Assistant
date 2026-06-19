#!/usr/bin/env python3
"""
run_evaluation.py — Main Entry Point
======================================
Runs the full Email Generation Assistant evaluation pipeline:

  1. Generates emails for all 10 scenarios using Model A and Model B
  2. Computes 3 custom metrics (FRS, TAS, FPS) for every generated email
  3. Saves structured output: CSV, JSON, and Markdown comparative analysis

Usage
-----

  # Full evaluation (requires OPENROUTER_API_KEY in .env)
  python run_evaluation.py

  # Dry-run: validate imports and scenario loading without API calls
  python run_evaluation.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Email Generation Assistant — Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenarios",
        default="data/scenarios.json",
        help="Path to scenarios JSON file (default: data/scenarios.json)",
    )
    parser.add_argument(
        "--output",
        default="data/results",
        help="Output directory for reports (default: data/results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls",
    )
    return parser.parse_args()


def dry_run(scenarios_path: str) -> None:
    """Validate project setup without spending API quota."""
    print("\n" + "═" * 60)
    print("  DRY-RUN MODE — No API calls will be made")
    print("═" * 60)

    path = Path(scenarios_path)
    if not path.exists():
        print(f"  ✗ Scenarios file not found: {scenarios_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data.get("scenarios", [])
    print(f"  ✓ Scenarios file loaded: {len(scenarios)} scenarios found")

    for s in scenarios:
        assert "intent" in s and "key_facts" in s and "tone" in s, \
            f"Scenario {s.get('id')} missing required fields"
        assert len(s["key_facts"]) >= 3, \
            f"Scenario {s.get('id')} should have at least 3 key facts"
    print(f"  ✓ All {len(scenarios)} scenarios validated")

    try:
        from src.prompts import build_advanced_prompt, build_simple_prompt, SYSTEM_PROMPT
        print("  ✓ src.prompts imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        sys.exit(1)

    try:
        from src.metrics import (
            compute_fact_recall_score,
            compute_tone_accuracy_score,
            compute_fluency_professionalism_score,
        )
        print("  ✓ src.metrics imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        sys.exit(1)

    try:
        import textstat
        print(f"  ✓ textstat available (version: {textstat.__version__})")
    except ImportError:
        print("  ✗ textstat not installed — run: pip install textstat")
        sys.exit(1)

    try:
        from openai import OpenAI
        print("  ✓ openai (OpenRouter client) available")
    except ImportError:
        print("  ✗ openai not installed — run: pip install openai")
        sys.exit(1)

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        masked = api_key[:10] + "…" + api_key[-4:]
        print(f"  ✓ API key found: {masked}")
    else:
        print("  ✗ No API key found — add OPENROUTER_API_KEY to .env")
        sys.exit(1)

    # Prompt preview
    sample = scenarios[0]
    prompt_a = build_advanced_prompt(sample["intent"], sample["key_facts"], sample["tone"])
    prompt_b = build_simple_prompt(sample["intent"], sample["key_facts"], sample["tone"])
    print(f"\n  ── Model A Prompt Preview (first 200 chars) ──")
    print(f"  {prompt_a[:200].replace(chr(10), chr(10)+'  ')}…")
    print(f"\n  ── Model B Prompt Preview ──")
    print(f"  {prompt_b}")

    print("\n  ✅ Dry-run complete — everything looks good!")
    print("  Run without --dry-run to execute the full evaluation.\n")


def _run_demo_evaluation(scenarios_path: str) -> list:
    """Run the full evaluation using built-in demo emails (no API calls)."""
    from app import _demo_email
    from src.metrics import (
        compute_fact_recall_score,
        compute_tone_accuracy_score,
        compute_fluency_professionalism_score,
    )

    with open(scenarios_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data.get("scenarios", [])
    total = len(scenarios)
    results = []

    DEMO_MODEL_ID = "demo-mode"
    DEMO_ADVANCED_STRATEGY = "Demo — Advanced (Role + Few-Shot + CoT)"
    DEMO_BASELINE_STRATEGY = "Demo — Baseline (Zero-Shot)"

    for idx, scenario in enumerate(scenarios, start=1):
        intent = scenario["intent"]
        facts = scenario["key_facts"]
        tone = scenario["tone"]

        print(f"\n[{idx:02d}/{total}] {scenario['name']}")

        print("  ► Generating Model A email (demo) …", end=" ", flush=True)
        email_a = _demo_email(intent, facts, tone, advanced=True)
        print("✓")

        print("  ► Generating Model B email (demo) …", end=" ", flush=True)
        email_b = _demo_email(intent, facts, tone, advanced=False)
        print("✓")

        # Model A metrics
        print("  ► Computing metrics for Model A …", end=" ", flush=True)
        frs_a = compute_fact_recall_score(facts, email_a)
        tas_score_a = _heuristic_tone_score(tone, email_a)
        tas_a = {"score": tas_score_a, "reasoning": "Heuristic scoring (no API)"}
        fps_score_a = _heuristic_fps_score(email_a)
        fps_a = {"score": fps_score_a, "readability_score": 0.6, "flesch_reading_ease": 55.0, "professionalism_score": 0.6, "professionalism_raw": 3, "professionalism_reasoning": "Heuristic scoring"}
        composite_a = round((frs_a["score"] + tas_score_a + fps_score_a) / 3, 4)
        print("✓")
        print(f"     FRS={frs_a['score']:.3f}  TAS={tas_score_a:.3f}  FPS={fps_score_a:.3f}  → Composite={composite_a:.3f}")

        # Model B metrics
        print("  ► Computing metrics for Model B …", end=" ", flush=True)
        frs_b = compute_fact_recall_score(facts, email_b)
        tas_score_b = _heuristic_tone_score(tone, email_b)
        tas_b = {"score": tas_score_b, "reasoning": "Heuristic scoring (no API)"}
        fps_score_b = _heuristic_fps_score(email_b)
        fps_b = {"score": fps_score_b, "readability_score": 0.5, "flesch_reading_ease": 62.0, "professionalism_score": 0.4, "professionalism_raw": 2, "professionalism_reasoning": "Heuristic scoring"}
        composite_b = round((frs_b["score"] + tas_score_b + fps_score_b) / 3, 4)
        print("✓")
        print(f"     FRS={frs_b['score']:.3f}  TAS={tas_score_b:.3f}  FPS={fps_score_b:.3f}  → Composite={composite_b:.3f}")

        results.append({
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "intent": intent,
            "tone": tone,
            "key_facts": facts,
            "human_reference_email": scenario.get("human_reference_email", ""),
            "model_a": {
                "model": DEMO_MODEL_ID,
                "strategy": DEMO_ADVANCED_STRATEGY,
                "generated_email": email_a,
                "prompt_used": "[Demo mode — no API call]",
                "frs": frs_a,
                "tas": tas_a,
                "fps": fps_a,
                "composite_score": composite_a,
            },
            "model_b": {
                "model": DEMO_MODEL_ID,
                "strategy": DEMO_BASELINE_STRATEGY,
                "generated_email": email_b,
                "prompt_used": "[Demo mode — no API call]",
                "frs": frs_b,
                "tas": tas_b,
                "fps": fps_b,
                "composite_score": composite_b,
            },
        })

    return results


def _heuristic_tone_score(tone: str, email: str) -> float:
    """Fallback tone scoring without API. Same logic as app.py."""
    tone_lower = tone.lower()
    email_lower = email.lower()
    score = 0.5
    formal_indicators = ["dear", "sincerely", "regards", "respectfully", "kindly", "best regards"]
    casual_indicators = ["hey", "hi,", "cheers", "thanks,", "awesome", "cool", "great week"]
    urgent_indicators = ["urgent", "immediately", "asap", "deadline", "time-sensitive"]
    empathetic_indicators = ["understand", "apologize", "sorry", "appreciate", "feel", "apolog"]
    if "formal" in tone_lower:
        score += 0.1 * sum(1 for w in formal_indicators if w in email_lower)
        score -= 0.05 * sum(1 for w in casual_indicators if w in email_lower)
    elif "casual" in tone_lower or "friendly" in tone_lower:
        score += 0.1 * sum(1 for w in casual_indicators if w in email_lower)
    elif "urgent" in tone_lower:
        score += 0.15 * sum(1 for w in urgent_indicators if w in email_lower)
    elif "empathetic" in tone_lower or "apolog" in tone_lower:
        score += 0.15 * sum(1 for w in empathetic_indicators if w in email_lower)
    else:
        score += 0.1
    return min(1.0, max(0.1, round(score, 4)))


def _heuristic_fps_score(email: str) -> float:
    """Fallback FPS scoring without API. Same logic as app.py."""
    has_subject = email.lower().startswith("subject:")
    has_greeting = any(email.lower().startswith(g) for g in ["dear", "hi,", "hello", "hey"])
    has_closing = any(c in email.lower() for c in ["regards", "sincerely", "best,", "cheers"])
    wc = len(email.split())
    structure_score = (
        (0.2 if has_subject else 0) + (0.2 if has_greeting else 0) +
        (0.2 if has_closing else 0) + (0.2 if 50 <= wc <= 300 else 0.1) + 0.2
    )
    return min(1.0, round(structure_score, 4))


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run(args.scenarios)
        return

    print("\n" + "═" * 60)
    print("  EMAIL GENERATION ASSISTANT — FULL EVALUATION")
    print("═" * 60)
    print("  This will:")
    print("   • Generate 10 × 2 = 20 emails via OpenRouter API")
    print("   • Run 3 metrics × 20 emails = 60 metric computations")
    print("   • Save CSV, JSON, and Markdown reports")
    print("═" * 60)

    from src.evaluator import run_evaluation
    try:
        results = run_evaluation(scenarios_path=args.scenarios)
    except KeyboardInterrupt:
        print("\n  ⚠ Evaluation interrupted by user")
        results = []
    except EnvironmentError as e:
        # API key not set — run in demo mode instead of failing
        print(f"\n  ⚠ {e}")
        print("  → Switching to DEMO MODE (no API calls needed)\n")
        results = _run_demo_evaluation(scenarios_path=args.scenarios)

    if not results:
        print("No results returned. Exiting.")
        sys.exit(1)

    from src.report_generator import save_all
    paths = save_all(results, output_dir=args.output)

    import statistics
    comp_a = round(statistics.mean(r["model_a"]["composite_score"] for r in results), 4)
    comp_b = round(statistics.mean(r["model_b"]["composite_score"] for r in results), 4)
    winner_name = results[0]["model_a"]["model"] if comp_a >= comp_b else results[0]["model_b"]["model"]

    print("\n" + "═" * 60)
    print("  EVALUATION COMPLETE")
    print("═" * 60)
    print(f"  Model A composite avg : {comp_a:.4f}")
    print(f"  Model B composite avg : {comp_b:.4f}")
    print(f"  Recommended model     : {winner_name}")
    print(f"\n  Output files:")
    for label, p in paths.items():
        print(f"    [{label:25s}] {p}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
