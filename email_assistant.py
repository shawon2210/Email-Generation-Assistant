#!/usr/bin/env python3
"""
email_assistant.py — Interactive CLI Email Generation Assistant
================================================================
Run locally with: python3 email_assistant.py

Supports two modes:
  1. LIVE mode — Generates real emails via OpenRouter API (requires API key with credits)
  2. DEMO mode  — Shows pre-built examples + runs full evaluation (no API needed)

The application demonstrates:
  - Role-Based + Few-Shot + Chain-of-Thought prompt engineering
  - Side-by-side comparison of Advanced vs Baseline strategies
  - 3 custom evaluation metrics (FRS, TAS, FPS)
  - 10 test scenarios across 5 business categories
"""

import os
import sys
import json
import textwrap
from datetime import datetime
from pathlib import Path

# ── Load environment ────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Imports ─────────────────────────────────────────────────────────────────
from src.prompts import build_advanced_prompt, build_simple_prompt, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, CHAIN_OF_THOUGHT_SCAFFOLD
from src.metrics import compute_fact_recall_score

# ── ANSI colors for terminal ────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║              📧  EMAIL GENERATION ASSISTANT              ║
║                                                          ║
║  Advanced Prompt Engineering • Custom Metrics • CLI      ║
╚══════════════════════════════════════════════════════════╝{C.RESET}""")


def divider(char="─", length=60):
    print(f"{C.DIM}{char * length}{C.RESET}")


def section(title):
    print(f"\n{C.BOLD}{C.CYAN}━━━ {title} ━━━{C.RESET}\n")


def show_prompt_engineering_docs():
    section("ADVANCED PROMPT ENGINEERING TECHNIQUES")

    print(f"{C.BOLD}1. Role-Playing (System Persona){C.RESET}")
    print(f"{C.DIM}Assigns the LLM a specific expert identity to anchor vocabulary and judgment.{C.RESET}")
    print(f"  {C.GREEN}Example:{C.RESET} \"You are a world-class professional business writer with 20+ years")
    print(f"  of experience crafting impactful corporate communications...\"\n")

    print(f"{C.BOLD}2. Few-Shot Examples{C.RESET}")
    print(f"{C.DIM}Prepends curated in-context demonstrations showing expected output format.{C.RESET}")
    print(f"  {C.GREEN}Included:{C.RESET} 2 complete examples (Formal + Casual tone)")
    print(f"  Each shows: Intent → Key Facts → Tone → Full polished email\n")

    print(f"{C.BOLD}3. Chain-of-Thought (CoT) Scaffold{C.RESET}")
    print(f"{C.DIM}Forces systematic reasoning before generation. Kept internal (never in output).{C.RESET}")
    print(f"  {C.GREEN}Steps:{C.RESET}")
    print(f"    STEP 1 — ANALYZE THE GOAL (audience, action needed)")
    print(f"    STEP 2 — FACT INTEGRATION PLAN (where each fact fits)")
    print(f"    STEP 3 — TONE CALIBRATION (vocabulary, contractions, warmth)")
    print(f"    STEP 4 — STRUCTURE PLAN (subject → body → CTA → closing)")
    print(f"    STEP 5 — WRITE THE EMAIL (execute the plan)\n")

    divider()


def show_metrics_docs():
    section("CUSTOM EVALUATION METRICS")

    print(f"{C.BOLD}Metric 1: Fact Recall Score (FRS){C.RESET} — Range: 0.0 → 1.0")
    print(f"  Token overlap + regex numeric matching. Deterministic, no LLM needed.")
    print(f"  overlap_ratio ≥ 0.40 → fact recalled")
    print(f"  overlap_ratio ≥ 0.20 + number match → fact recalled\n")

    print(f"{C.BOLD}Metric 2: Tone Accuracy Score (TAS){C.RESET} — Range: 0.0 → 1.0")
    print(f"  LLM-as-a-Judge with 5-point rubric.")
    print(f"  Evaluates: vocabulary, sentence structure, formality, contractions, warmth\n")

    print(f"{C.BOLD}Metric 3: Fluency & Professionalism Score (FPS){C.RESET} — Range: 0.0 → 1.0")
    print(f"  Hybrid: 40% textstat Flesch Reading Ease + 60% LLM-as-Judge\n")

    divider()


def interactive_generate():
    section("INTERACTIVE EMAIL GENERATION")

    print(f"{C.YELLOW}Enter email details (or 'demo' for demo mode, 'quit' to exit):{C.RESET}\n")

    while True:
        print(f"{C.BOLD}Intent:{C.RESET} ", end="")
        intent = input().strip()
        if intent.lower() in ("quit", "q", "exit"):
            return
        if intent.lower() == "demo":
            run_demo_mode()
            return

        print(f"{C.BOLD}Key Facts{C.RESET} (one per line, empty line to finish):")
        facts = []
        while True:
            fact = input(f"  {C.DIM}•{C.RESET} ").strip()
            if not fact:
                break
            facts.append(fact)

        if not facts:
            print(f"{C.RED}Please enter at least one fact.{C.RESET}")
            continue

        print(f"{C.BOLD}Tone{C.RESET} [Formal/Professional/Casual/Friendly/Urgent/Empathetic]: ", end="")
        tone = input().strip() or "Professional"

        print(f"\n{C.GREEN}Generating emails...\n{C.RESET}")

        # Build prompts
        adv_prompt = build_advanced_prompt(intent, facts, tone)
        base_prompt = build_simple_prompt(intent, facts, tone)

        # Show prompts
        section("MODEL A — ADVANCED PROMPT")
        print(f"{C.DIM}System Prompt:{C.RESET}")
        print(textwrap.fill(SYSTEM_PROMPT, width=70, initial_indent="  ", subsequent_indent="  "))
        print(f"\n{C.DIM}User Prompt (first 400 chars):{C.RESET}")
        user_msg = adv_prompt.split("YOUR TASK")[1][:400] if "YOUR TASK" in adv_prompt else adv_prompt[:400]
        print(f"  {user_msg}...")

        section("MODEL B — BASELINE PROMPT")
        print(f"  {base_prompt[:300]}...")

        # Try live generation
        api_key = OPENROUTER_API_KEY or GOOGLE_API_KEY
        if api_key and OPENROUTER_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

                print(f"\n{C.GENERATING} Model A email via API...{C.RESET}")
                r1 = client.chat.completions.create(
                    model="openrouter/auto",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": adv_prompt}
                    ],
                    temperature=0.3, max_tokens=1024
                )
                adv_email = r1.choices[0].message.content.strip()

                print(f"{C.GREEN} Generating Model B email via API...{C.RESET}")
                r2 = client.chat.completions.create(
                    model="openrouter/auto",
                    messages=[{"role": "user", "content": base_prompt}],
                    temperature=0.3, max_tokens=1024
                )
                base_email = r2.choices[0].message.content.strip()

                # Display results
                section("RESULTS — MODEL A (ADVANCED)")
                print(f"{C.GREEN}{adv_email}{C.RESET}")

                section("RESULTS — MODEL B (BASELINE)")
                print(f"{C.BLUE}{base_email}{C.RESET}")

                # Compute FRS
                frs_a = compute_fact_recall_score(facts, adv_email)
                frs_b = compute_fact_recall_score(facts, base_email)

                section("EVALUATION (FRS — Fact Recall Score)")
                print(f"  Model A: {frs_a['score']:.2f} ({frs_a['recalled_count']}/{frs_a['total_facts']} facts recalled)")
                print(f"  Model B: {frs_b['score']:.2f} ({frs_b['recalled_count']}/{frs_b['total_facts']} facts recalled)")
                for d in frs_a['fact_details']:
                    icon = "✅" if d['recalled'] else "❌"
                    print(f"    {icon} {d['fact']}")

            except Exception as e:
                print(f"\n{C.RED}API Error: {e}{C.RESET}")
                print(f"{C.YELLOW}Falling back to demo mode...{C.RESET}")
                run_demo_mode()
                return
        else:
            print(f"\n{C.YELLOW}No API key configured. Showing demo...{C.RESET}")
            run_demo_mode()
            return

        print(f"\n{C.BOLD}Generate another? (y/n):{C.RESET} ", end="")
        if input().strip().lower() != 'y':
            break


def run_demo_mode():
    section("DEMO MODE — Full Evaluation Results")

    # Load results
    results_path = Path("data/results/evaluation_results.json")
    if not results_path.exists():
        print(f"{C.YELLOW}No results found. Running simulation first...{C.RESET}")
        os.system(f"{sys.executable} run_demo.py")

    with open(results_path) as f:
        data = json.load(f)
    meta = data["evaluation_metadata"]
    results = data["scenario_results"]
    agg = meta["aggregate_scores"]

    print(f"{C.BOLD}Evaluation Date:{C.RESET} {meta['timestamp'][:19]}")
    print(f"{C.BOLD}Scenarios:{C.RESET} {meta['total_scenarios']}")
    print(f"{C.BOLD}Model A:{C.RESET} {meta['model_a']['id']} — {meta['model_a']['strategy']}")
    print(f"{C.BOLD}Model B:{C.RESET} {meta['model_b']['id']} — {meta['model_b']['strategy']}")

    # Score table
    section("AGGREGATE SCORES")
    print(f"  {'Metric':<35} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
    divider("─", 70)
    print(f"  {'Fact Recall Score (FRS)':<35} {agg['model_a']['avg_frs']:>10.4f} {agg['model_b']['avg_frs']:>10.4f} {agg['model_a']['avg_frs'] - agg['model_b']['avg_frs']:>+10.4f}")
    print(f"  {'Tone Accuracy Score (TAS)':<35} {agg['model_a']['avg_tas']:>10.4f} {agg['model_b']['avg_tas']:>10.4f} {agg['model_a']['avg_tas'] - agg['model_b']['avg_tas']:>+10.4f}")
    print(f"  {'Fluency & Professionalism (FPS)':<35} {agg['model_a']['avg_fps']:>10.4f} {agg['model_b']['avg_fps']:>10.4f} {agg['model_a']['avg_fps'] - agg['model_b']['avg_fps']:>+10.4f}")
    divider("─", 70)
    print(f"  {'COMPOSITE AVERAGE':<35} {C.GREEN}{agg['model_a']['avg_composite']:>10.4f}{C.RESET} {agg['model_b']['avg_composite']:>10.4f} {C.GREEN}{agg['model_a']['avg_composite'] - agg['model_b']['avg_composite']:>+10.4f}{C.RESET}")

    # Per-scenario table
    section("PER-SCENARIO RESULTS")
    print(f"  {'#':<4} {'Scenario':<32} {'A-Comp':>8} {'B-Comp':>8} {'Winner':>8}")
    divider("─", 60)
    for r in results:
        a_comp = r['model_a']['composite_score']
        b_comp = r['model_b']['composite_score']
        winner = f"{C.GREEN}A{C.RESET}" if a_comp >= b_comp else f"{C.RED}B{C.RESET}"
        print(f"  {r['scenario_id']:<4} {r['scenario_name'][:30]:<32} {a_comp:>8.4f} {b_comp:>8.4f} {winner:>8}")

    # Show sample emails
    section("SAMPLE EMAILS — Scenario 1")
    print(f"{C.BOLD}{C.GREEN}Model A (Advanced):{C.RESET}")
    print(textwrap.fill(results[0]['model_a']['generated_email'][:500], width=70, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{C.BOLD}{C.BLUE}Model B (Baseline):{C.RESET}")
    print(textwrap.fill(results[0]['model_b']['generated_email'][:500], width=70, initial_indent="  ", subsequent_indent="  "))

    # Show all 10 generated email pairs
    section("ALL GENERATED EMAILS")
    for r in results:
        print(f"\n{C.BOLD}Scenario {r['scenario_id']}: {r['scenario_name']} ({r['tone']}){C.RESET}")
        print(f"  Facts: {', '.join(r['key_facts'][:2])}...")
        print(f"  {C.GREEN}A:{C.RESET} {r['model_a']['generated_email'][:120].strip()}...")
        print(f"  {C.BLUE}B:{C.RESET} {r['model_b']['generated_email'][:120].strip()}...")

    # Recommendation
    section("PRODUCTION RECOMMENDATION")
    print(f"  {C.GREEN}{C.BOLD}✅ Deploy Model A (Advanced Prompt Engineering){C.RESET}")
    print(f"  • +{agg['model_a']['avg_tas'] - agg['model_b']['avg_tas']:.4f} TAS advantage (tone controllability)")
    print(f"  • +{agg['model_a']['avg_frs'] - agg['model_b']['avg_frs']:.4f} FRS advantage (fact coverage)")
    print(f"  • +{agg['model_a']['avg_fps'] - agg['model_b']['avg_fps']:.4f} FPS advantage (professionalism)")
    wins = sum(1 for r in results if r['model_a']['composite_score'] >= r['model_b']['composite_score'])
    print(f"  • Wins {wins}/{len(results)} scenarios")


def main_menu():
    banner()

    while True:
        section("MAIN MENU")
        print(f"  {C.GREEN}1.{C.RESET} Interactive Email Generation")
        print(f"  {C.GREEN}2.{C.RESET} Run Full Demo Evaluation")
        print(f"  {C.GREEN}3.{C.RESET} View Prompt Engineering Docs")
        print(f"  {C.GREEN}4.{C.RESET} View Metrics Documentation")
        print(f"  {C.GREEN}5.{C.RESET} Run Unit Tests")
        print(f"  {C.GREEN}6.{C.RESET} View Full Evaluation Report")
        print(f"  {C.RED}0.{C.RESET} Exit")
        print()

        choice = input(f"  {C.BOLD}Select option:{C.RESET} ").strip()

        if choice == "1":
            interactive_generate()
        elif choice == "2":
            run_demo_mode()
        elif choice == "3":
            show_prompt_engineering_docs()
        elif choice == "4":
            show_metrics_docs()
        elif choice == "5":
            section("RUNNING UNIT TESTS")
            os.system(f"{sys.executable} -m pytest tests/test_metrics.py -v")
        elif choice == "6":
            report_path = Path("data/results/comparative_analysis.md")
            if report_path.exists():
                section("FULL EVALUATION REPORT")
                with open(report_path) as f:
                    print(f.read())
            else:
                print(f"{C.YELLOW}No report found. Run demo first.{C.RESET}")
        elif choice == "0":
            print(f"\n{C.CYAN}Goodbye! 👋{C.RESET}\n")
            break
        else:
            print(f"{C.RED}Invalid option.{C.RESET}")


if __name__ == "__main__":
    main_menu()
