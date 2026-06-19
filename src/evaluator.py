"""
Evaluation Runner
==================
Orchestrates the full evaluation pipeline:
  1. Load 10 test scenarios from data/scenarios.json
  2. Generate emails with Model A (advanced prompt) and Model B (zero-shot)
  3. Compute all 3 custom metrics for each generated email
  4. Collect and return structured results for reporting
"""

import json
import time
import os
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from src.generator import EmailGenerator
from src.metrics import (
    make_judge_client,
    compute_fact_recall_score,
    compute_tone_accuracy_score,
    compute_fluency_professionalism_score,
)

_INTER_CALL_DELAY     = 5    # seconds between API calls (rate-limit buffer)
_INTER_SCENARIO_DELAY = 10   # seconds between scenarios


def run_evaluation(scenarios_path: str = "data/scenarios.json") -> list:
    """
    Run the full dual-model evaluation over all 10 scenarios.

    Args:
        scenarios_path: Path to the scenarios JSON file.

    Returns:
        List of result dicts, one per scenario.
    """
    # ── Load environment & configure API ──────────────────────────────────
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    judge_client = make_judge_client(api_key)
    generator    = EmailGenerator(api_key)

    # ── Load scenarios ─────────────────────────────────────────────────────
    with open(scenarios_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data["scenarios"]
    total     = len(scenarios)

    results = []
    print(f"\n{'═'*65}")
    print(f"  EMAIL GENERATION ASSISTANT — EVALUATION PIPELINE")
    print(f"  {total} scenarios × 2 models × 3 metrics")
    print(f"{'═'*65}")

    for idx, scenario in enumerate(scenarios, start=1):
        intent = scenario["intent"]
        facts  = scenario["key_facts"]
        tone   = scenario["tone"]

        print(f"\n[{idx:02d}/{total}] {scenario['name']}")
        print(f"  Tone   : {tone}")
        short_intent = intent[:70] + "…" if len(intent) > 70 else intent
        print(f"  Intent : {short_intent}")

        # ── Step 1: Generate emails ────────────────────────────────────────
        print("  ► Generating Model A email …", end=" ", flush=True)
        gen_a = generator.generate(intent, facts, tone, use_model_a=True)
        print("✓")
        time.sleep(_INTER_CALL_DELAY)

        print("  ► Generating Model B email …", end=" ", flush=True)
        gen_b = generator.generate(intent, facts, tone, use_model_a=False)
        print("✓")
        time.sleep(_INTER_CALL_DELAY)

        # ── Step 2: Compute metrics — Model A ─────────────────────────────
        print("  ► Computing metrics for Model A …", end=" ", flush=True)

        frs_a = compute_fact_recall_score(facts, gen_a["generated_email"])
        time.sleep(_INTER_CALL_DELAY)

        tas_a = compute_tone_accuracy_score(
            tone, gen_a["generated_email"], judge_client
        )
        time.sleep(_INTER_CALL_DELAY)

        fps_a = compute_fluency_professionalism_score(
            gen_a["generated_email"], judge_client
        )
        time.sleep(_INTER_CALL_DELAY)

        composite_a = round(
            (frs_a["score"] + tas_a["score"] + fps_a["score"]) / 3, 4
        )
        print("✓")
        print(
            f"     FRS={frs_a['score']:.3f}  "
            f"TAS={tas_a['score']:.3f}  "
            f"FPS={fps_a['score']:.3f}  "
            f"→ Composite={composite_a:.3f}"
        )

        # ── Step 3: Compute metrics — Model B ─────────────────────────────
        print("  ► Computing metrics for Model B …", end=" ", flush=True)

        frs_b = compute_fact_recall_score(facts, gen_b["generated_email"])
        time.sleep(_INTER_CALL_DELAY)

        tas_b = compute_tone_accuracy_score(
            tone, gen_b["generated_email"], judge_client
        )
        time.sleep(_INTER_CALL_DELAY)

        fps_b = compute_fluency_professionalism_score(
            gen_b["generated_email"], judge_client
        )
        time.sleep(_INTER_CALL_DELAY)

        composite_b = round(
            (frs_b["score"] + tas_b["score"] + fps_b["score"]) / 3, 4
        )
        print("✓")
        print(
            f"     FRS={frs_b['score']:.3f}  "
            f"TAS={tas_b['score']:.3f}  "
            f"FPS={fps_b['score']:.3f}  "
            f"→ Composite={composite_b:.3f}"
        )

        # ── Assemble result record ─────────────────────────────────────────
        results.append({
            "scenario_id":           scenario["id"],
            "scenario_name":         scenario["name"],
            "intent":                intent,
            "tone":                  tone,
            "key_facts":             facts,
            "human_reference_email": scenario["human_reference_email"],

            "model_a": {
                "model":           gen_a["model"],
                "strategy":        gen_a["strategy"],
                "generated_email": gen_a["generated_email"],
                "prompt_used":     gen_a["prompt_used"],
                "frs":             frs_a,
                "tas":             tas_a,
                "fps":             fps_a,
                "composite_score": composite_a,
            },
            "model_b": {
                "model":           gen_b["model"],
                "strategy":        gen_b["strategy"],
                "generated_email": gen_b["generated_email"],
                "prompt_used":     gen_b["prompt_used"],
                "frs":             frs_b,
                "tas":             tas_b,
                "fps":             fps_b,
                "composite_score": composite_b,
            },
        })

        time.sleep(_INTER_SCENARIO_DELAY)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  EVALUATION COMPLETE — SUMMARY")
    print(f"{'═'*65}")
    print(
        f"  {'Scenario':<35} {'A-Comp':>7} {'B-Comp':>7} {'Winner':>8}"
    )
    print(f"  {'-'*60}")
    for r in results:
        winner = (
            "Model A"
            if r["model_a"]["composite_score"] >= r["model_b"]["composite_score"]
            else "Model B"
        )
        print(
            f"  {r['scenario_name'][:35]:<35} "
            f"{r['model_a']['composite_score']:>7.3f} "
            f"{r['model_b']['composite_score']:>7.3f} "
            f"{winner:>8}"
        )

    avg_a = round(
        sum(r["model_a"]["composite_score"] for r in results) / len(results), 4
    )
    avg_b = round(
        sum(r["model_b"]["composite_score"] for r in results) / len(results), 4
    )
    print(f"  {'-'*60}")
    print(f"  {'OVERALL AVERAGE':<35} {avg_a:>7.3f} {avg_b:>7.3f}")
    print(f"{'═'*65}\n")

    return results
