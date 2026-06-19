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
  # Full evaluation (requires GOOGLE_API_KEY in .env)
  python run_evaluation.py

  # Dry-run: validate imports and scenario loading without API calls
  python run_evaluation.py --dry-run

  # Custom scenarios file or output directory
  python run_evaluation.py --scenarios data/scenarios.json --output data/results
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

    # Check scenarios file
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
    print(f"  ✓ All {len(scenarios)} scenarios validated (intent, key_facts, tone present)")

    # Check imports
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
        from google import genai
        print("  ✓ google-genai available")
    except ImportError:
        print("  ✗ google-genai not installed — run: pip install google-genai")
        sys.exit(1)

    # Check API key
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        masked = api_key[:6] + "…" + api_key[-4:]
        print(f"  ✓ GOOGLE_API_KEY found: {masked}")
    else:
        print("  ✗ GOOGLE_API_KEY not set — copy .env.example to .env and add your key")
        sys.exit(1)

    # Prompt preview
    sample = scenarios[0]
    prompt_a = build_advanced_prompt(sample["intent"], sample["key_facts"], sample["tone"])
    prompt_b = build_simple_prompt(sample["intent"], sample["key_facts"], sample["tone"])
    print(f"\n  ── Model A Prompt Preview (first 300 chars) ──")
    print(f"  {prompt_a[:300].replace(chr(10), chr(10)+'  ')}…")
    print(f"\n  ── Model B Prompt Preview ──")
    print(f"  {prompt_b}")

    print("\n  ✅ Dry-run complete — everything looks good!")
    print("  Run without --dry-run to execute the full evaluation.\n")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run(args.scenarios)
        return

    print("\n" + "═" * 60)
    print("  EMAIL GENERATION ASSISTANT — FULL EVALUATION")
    print("═" * 60)
    print("  This will:")
    print("   • Generate 10 × 2 = 20 emails via Gemini API")
    print("   • Run 3 metrics × 20 emails = 60 metric computations")
    print("   • Save CSV, JSON, and Markdown reports")
    print("  Estimated time: ~5–8 minutes (API rate-limit delays included)")
    print("═" * 60)

    # ── Run evaluation ─────────────────────────────────────────────────────
    from src.evaluator import run_evaluation
    results = run_evaluation(scenarios_path=args.scenarios)

    if not results:
        print("No results returned. Exiting.")
        sys.exit(1)

    # ── Save reports ───────────────────────────────────────────────────────
    from src.report_generator import save_all
    paths = save_all(results, output_dir=args.output)

    # ── Final summary ──────────────────────────────────────────────────────
    import statistics
    comp_a = round(statistics.mean(r["model_a"]["composite_score"] for r in results), 4)
    comp_b = round(statistics.mean(r["model_b"]["composite_score"] for r in results), 4)
    winner = results[0]["model_a"]["model"] if comp_a >= comp_b else results[0]["model_b"]["model"]

    print("\n" + "═" * 60)
    print("  EVALUATION COMPLETE")
    print("═" * 60)
    print(f"  Model A composite avg : {comp_a:.4f}")
    print(f"  Model B composite avg : {comp_b:.4f}")
    print(f"  Recommended model     : {winner}")
    print()
    print("  Output files:")
    for label, p in paths.items():
        print(f"    [{label:25s}] {p}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
