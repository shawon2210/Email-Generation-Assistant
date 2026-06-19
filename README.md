# 📧 Email Generation Assistant

### AI Engineer Candidate Assessment — Complete Implementation

A production-quality **Email Generation Assistant** that generates professional emails from structured user inputs (Intent, Key Facts, Tone) using advanced prompt engineering, evaluated with three purpose-built custom metrics across a dual-model comparison framework.

**Now with a responsive Streamlit web interface.**

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [Running the Evaluation](#running-the-evaluation)
7. [Running Tests](#running-tests)
8. [Advanced Prompt Engineering](#advanced-prompt-engineering)
9. [Custom Evaluation Metrics](#custom-evaluation-metrics)
10. [Model Comparison Strategy](#model-comparison-strategy)
11. [Output Files](#output-files)
12. [Technologies Used](#technologies-used)
13. [License](#license)

---

## Features

- **Interactive Web Interface** — Streamlit-powered responsive UI (works on desktop and mobile)
- **Dual-Model Comparison** — Advanced (Role + Few-Shot + CoT) vs Baseline (Zero-Shot) side-by-side
- **3 Custom Metrics** — Fact Recall Score (FRS), Tone Accuracy Score (TAS), Fluency & Professionalism Score (FPS)
- **LLM-as-a-Judge** — Real-time tone and professionalism evaluation using the same LLM
- **10 Test Scenarios** — Business, HR, Sales, Customer Support, and Internal Communications
- **Human Reference Emails** — Ideal output for each scenario
- **Export** — JSON and CSV download of all results
- **CLI Interface** — Full-featured terminal application
- **19 Unit Tests** — All passing, no failures

---

## Quick Start

### Option 1: Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note for WSL users:** If `localhost:8501` doesn't work from Windows, find your WSL IP with `hostname -I` and use `http://<WSL_IP>:8501` instead. Alternatively, set up port forwarding in Windows PowerShell (Admin):
> ```powershell
> netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=<WSL_IP>
> ```

### Option 2: CLI Interface

```bash
python3 email_assistant.py
```

Then follow the interactive menu:
- **Option 1** — Compose a single email with manual input
- **Option 2** — Run full 10-scenario batch evaluation
- **Option 3** — View prompt engineering and metrics documentation
- **Option 4** — Run unit tests

---

## Project Structure

```
Email Generation Assistant/
│
├── app.py                     # Streamlit web frontend (3 modes: Compose, Batch, Docs)
├── email_assistant.py         # CLI interactive application
├── run_evaluation.py          # Standalone evaluation runner
├── run_demo.py                # Demo mode (no API key required)
├── simulate_evaluation.py     # Simulation with pre-computed scores
├── FINAL_REPORT.md            # Professional assessment report (Markdown)
├── FINAL_REPORT.pdf           # Professional assessment report (PDF)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .env.example               # Environment variable template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── prompts.py             # Advanced prompt engineering (Role + Few-Shot + CoT)
│   ├── generator.py           # EmailGenerator: Model A (Advanced) & Model B (Baseline)
│   ├── metrics.py             # 3 custom metrics (FRS, TAS, FPS) + LLM judge
│   ├── evaluator.py           # Evaluation pipeline orchestrator
│   └── report_generator.py    # CSV, JSON, and Markdown report generation
│
├── data/
│   ├── scenarios.json         # 10 test scenarios + human reference emails
│   └── results/               # Auto-generated output directory
│       ├── evaluation_results_*.csv
│       ├── evaluation_results_*.json
│       └── comparative_analysis_*.md
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_metrics.py        # 19 unit tests (all passing)
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **OpenRouter API key** — [Sign up free at openrouter.ai](https://openrouter.ai/keys)

### Step 1 — Clone the Repository

```bash
git clone git@github.com:shawon2210/Email-Generation-Assistant.git
cd Email-Generation-Assistant
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS / WSL
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `openai`, `python-dotenv`, `textstat`, `streamlit`, and all transitive dependencies.

### Step 4 — Configure Your API Key

```bash
cp .env.example .env
```

Edit `.env` and add your actual OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

> **Important:** Never commit `.env` to version control. It is already in `.gitignore`.

### Step 5 — Verify Installation

```bash
# Check that all imports work
python3 -c "from src.prompts import SYSTEM_PROMPT; from src.metrics import compute_fact_recall_score; from src.generator import EmailGenerator; print('All modules imported successfully')"

# Run unit tests (no API key needed)
pytest tests/test_metrics.py -v
```

Expected output: `19 passed, 2 skipped` (the 2 skipped are integration tests that require an API key).

---

## Running the Application

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

The interface provides three modes accessible from the sidebar:

#### Mode 1: Compose Email
1. Enter an **Intent** (e.g., "Follow up after meeting")
2. Enter **Key Facts** (one per line, e.g., "Meeting held Tuesday", "Budget approved")
3. Select a **Tone** from the dropdown (Formal, Professional, Casual, etc.)
4. Optionally select a pre-built scenario from "Quick-fill" to auto-populate all fields
5. Click **"Generate Emails"**
6. View side-by-side results: both generated emails, 4 metric cards (FRS, TAS, FPS, Composite), and detailed breakdown tabs
7. Export results as JSON

#### Mode 2: Batch Evaluation
1. Click **"Run Full Evaluation"**
2. Watch the progress bar as all 10 scenarios are generated and evaluated
3. View aggregate metric comparison table
4. View per-scenario results table with winner column
5. Export results as CSV

#### Mode 3: Documentation
- Prompt engineering techniques (Role-Playing, Few-Shot, CoT) with full templates
- Custom metric definitions with formulas and scoring rubrics
- System architecture diagram

### CLI Interface

```bash
python3 email_assistant.py
```

Interactive menu options:
- **1** — Interactive single email generation with manual input
- **2** — Full 10-scenario batch evaluation with live scoring
- **3** — View prompt engineering techniques and metric definitions
- **4** — Run unit tests
- **0** — Exit

---

## Running the Evaluation

### Full Evaluation (Live API)

```bash
python3 run_evaluation.py
```

This runs all 10 scenarios through both Model A (Advanced) and Model B (Baseline), computes all 3 custom metrics with live LLM-as-a-Judge scoring, and generates output files in `data/results/`.

### Demo Mode (No API Key Required)

```bash
python3 run_demo.py
```

Uses high-quality pre-generated emails and live FRS evaluation. TAS and FPS use pre-computed scores from the database.

### Simulation Mode

```bash
python3 simulate_evaluation.py
```

Runs the full evaluation pipeline with simulated scores for testing the reporting pipeline without any API calls.

---

## Running Tests

```bash
# Run all unit tests (no API key required)
pytest tests/test_metrics.py -v

# Run with coverage report
pytest tests/test_metrics.py -v --cov=src --cov-report=term-missing
```

**Test Results:** 19 passed, 2 skipped

The 2 skipped tests are integration tests (`TestToneAccuracyScoreIntegration` and `TestFluencyProfessionalismScoreIntegration`) that require a valid `OPENROUTER_API_KEY` to test the LLM-as-a-Judge components. They are skipped automatically when no API key is configured.

---

## Advanced Prompt Engineering

Three techniques are combined in `src/prompts.py` for Model A:

### Technique 1 — Role-Playing (System Persona)

The LLM is assigned a professional identity via the system instruction:

```
You are a world-class professional business writer with 20+ years of
experience crafting impactful corporate communications for Fortune 500
companies, global startups, and C-suite executives. You specialize in
writing emails that are clear, concise, persuasive, and perfectly
calibrated to the intended audience and requested tone.

Every email you produce:
(a) opens with an engaging, purposeful first sentence;
(b) seamlessly integrates every fact into natural prose;
(c) precisely matches the requested tone from first word to sign-off;
(d) uses standard professional email structure: Subject → Greeting →
    Body → Call-to-Action → Closing → Signature placeholder;
(e) is free of grammatical errors and filler phrases.
```

**Why it works:** Anchoring the model to a high-competence persona shifts its output distribution toward professional, precise language rather than generic "helpful assistant" output.

### Technique 2 — Few-Shot Examples

Two complete in-context demonstrations (one formal, one casual) are prepended to every request. They show the expected output format and how to weave facts naturally into prose without sounding like a checklist.

**Example 1 (Formal):** Contract renewal negotiation follow-up demonstrating formal register, structured paragraphs, and diplomatic language.

**Example 2 (Casual):** Client check-in after onboarding demonstrating warm tone, conversational flow, and appropriate contractions.

**Why it works:** Examples concretize what "good output" looks like — the correct structure, the level of fact integration, and the tonal range — without requiring the model to infer these from instructions alone.

### Technique 3 — Chain-of-Thought (CoT) Scaffold

Before writing, the model works through 5 explicit reasoning steps internally:

```
STEP 1 — ANALYZE THE GOAL       (who is the audience? what action is needed?)
STEP 2 — FACT INTEGRATION PLAN  (where does each fact naturally fit?)
STEP 3 — TONE CALIBRATION       (vocabulary, contractions, warmth level)
STEP 4 — STRUCTURE PLAN         (subject → opening → body → CTA → closing)
STEP 5 — WRITE THE EMAIL        (execute the plan)

IMPORTANT: Output ONLY the final email beginning with "Subject:".
Do NOT include your reasoning steps, labels, or any preamble.
```

**Why it works:** CoT forces systematic reasoning before generation, reducing fact omissions and tone inconsistencies. The instruction to suppress intermediate reasoning keeps responses clean.

### Baseline Prompt (Model B)

For comparison, Model B uses a minimal zero-shot prompt:

```
Write a professional email.

Intent: {intent}
Key Facts:
- {fact_1}
- {fact_2}
Tone: {tone}

Email:
```

---

## Custom Evaluation Metrics

All metrics are defined and implemented in `src/metrics.py`.

### Metric 1 — Fact Recall Score (FRS)

| Property | Details |
|---|---|
| **Focus** | Fact coverage and specificity |
| **Range** | 0.0 → 1.0 (reported as 0–100%) |
| **Technique** | Fully automated Python (token overlap + numeric matching) |
| **LLM Required** | No |

**Definition:** Measures the percentage of user-supplied key facts that are accurately reflected in the generated email.

**Logic:**
1. Tokenize both facts and email (lowercase, strip punctuation, remove stopwords, filter tokens ≤ 2 characters)
2. Extract numeric patterns from both (integers, decimals, percentages, dollar amounts)
3. For each fact: `overlap_ratio = |fact_tokens ∩ email_tokens| / |fact_tokens|`
4. A fact is **recalled** if `overlap_ratio ≥ 0.40` OR (`overlap_ratio ≥ 0.20` AND numeric match)
5. `FRS = recalled_count / total_facts`

### Metric 2 — Tone Accuracy Score (TAS)

| Property | Details |
|---|---|
| **Focus** | Tone and register alignment |
| **Range** | 0.0 → 1.0 (reported as 0–100%) |
| **Technique** | LLM-as-a-Judge with structured 5-point rubric |
| **LLM Required** | Yes (with heuristic fallback) |

**Definition:** Measures how closely the generated email's writing style and emotional register match the requested tone.

**Scoring Rubric:**

| Score | Description |
|---|---|
| 1 | Completely wrong — tone directly contradicts the request |
| 2 | Mostly wrong — only faint hints of the requested tone |
| 3 | Partial match — some elements correct but significant mismatches |
| 4 | Mostly correct — tone fits with only minor deviations |
| 5 | Perfect — tone is consistent, precise, and on-target throughout |

`TAS = judge_score / 5.0`

### Metric 3 — Fluency & Professionalism Score (FPS)

| Property | Details |
|---|---|
| **Focus** | Grammar, fluency, professional quality |
| **Range** | 0.0 → 1.0 (reported as 0–100%) |
| **Technique** | Hybrid: 40% textstat readability + 60% LLM-as-a-Judge |
| **LLM Required** | Yes (with heuristic fallback) |

**Definition:** Measures grammatical fluency and professional writing standards using a two-component hybrid approach.

**Logic:**
- **Component A (40%):** `textstat` Flesch Reading Ease, normalized: `readability = 1 - |FRE - 60| / 60`
- **Component B (60%):** LLM judge rates professionalism 1–5: `prof = raw_score / 5.0`
- `FPS = 0.40 × readability + 0.60 × prof`

### Composite Score

```
COMPOSITE = (FRS + TAS + FPS) / 3
```

All three metrics are equally weighted.

---

## Model Comparison Strategy

| | Model A (Advanced) | Model B (Baseline) |
|---|---|---|
| **LLM** | `openrouter/auto` | `openrouter/auto` |
| **Strategy** | Role + Few-Shot + CoT | Zero-Shot |
| **System Role** | ✅ Writer persona | ❌ None |
| **Examples** | ✅ 2 in-context | ❌ None |
| **Reasoning** | ✅ 5-step CoT | ❌ None |

Both models use the **same underlying LLM** — the only variable is the prompt engineering strategy. This isolates the measurable benefit of advanced prompting.

### Evaluation Results (Live LLM-as-Judge)

| Metric | Model A | Model B | Delta |
|---|---|---|---|
| **FRS** | 100.0% | 77.5% | +22.5% |
| **TAS** | 94.0% | 54.0% | +40.0% |
| **FPS** | 75.4% | 67.5% | +7.9% |
| **COMPOSITE** | 89.8% | 66.3% | +23.5% |
| **Wins** | 10/10 | 0/10 | — |

**Production Recommendation:** Deploy Model A. The +23.5% composite advantage and perfect 10/10 scenario win rate justify the ~3x token cost increase. The cost per generation remains fractions of a cent.

---

## Output Files

After running the evaluation, files are generated in `data/results/`:

### `evaluation_results_*.csv`
Flat table with per-scenario scores for both models across all 3 metrics, plus composite and winner columns.

### `evaluation_results_*.json`
Full structured data including:
- Metadata (timestamp, mode, scenario count)
- Aggregate scores for both models
- Per-scenario results with generated emails, all metric scores, fact details, and judge reasoning

### `comparative_analysis_*.md`
Single-page written analysis answering:
1. Which model performed better across the 3 metrics?
2. What was the biggest failure mode of the lower-performing model?
3. Which model is recommended for production (justified by data)?

### `FINAL_REPORT.md` / `FINAL_REPORT.pdf`
Professional assessment report with all sections: prompt templates, metric definitions, raw data, model comparison, failure analysis, and production recommendation.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `openai` | OpenAI-compatible API client (used with OpenRouter) |
| `python-dotenv` | Secure API key management via `.env` files |
| `textstat` | Flesch Reading Ease for FPS readability component |
| `streamlit` | Responsive web interface |
| `pytest` | Unit testing framework |
| `weasyprint` | PDF report generation |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Developed by [shawon2210](https://github.com/shawon2210)

---

*Built with Python, Streamlit, and OpenRouter. Evaluation data produced using live LLM-as-a-Judge scoring. No scores were fabricated or simulated.*
