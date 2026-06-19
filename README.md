# 📧 Email Generation Assistant
### AI Engineer Candidate Assessment — Full Implementation

A production-quality Email Generation Assistant that generates professional
emails using advanced prompt engineering, evaluated with three custom metrics
across a dual-model comparison framework.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Running the Evaluation](#running-the-evaluation)
4. [Advanced Prompt Engineering](#advanced-prompt-engineering)
5. [Custom Evaluation Metrics](#custom-evaluation-metrics)
6. [Model Comparison Strategy](#model-comparison-strategy)
7. [Output Files](#output-files)
8. [Running Tests](#running-tests)

---

## Project Structure

```
Email Generation Assistant/
├── run_evaluation.py          # ← Main entry point
├── requirements.txt
├── .env.example               # Copy to .env and add GOOGLE_API_KEY
│
├── src/
│   ├── prompts.py             # Advanced prompt engineering (Role + Few-Shot + CoT)
│   ├── generator.py           # EmailGenerator: Model A & Model B
│   ├── metrics.py             # 3 custom metrics (FRS, TAS, FPS)
│   ├── evaluator.py           # Evaluation pipeline orchestrator
│   └── report_generator.py   # CSV, JSON, and Markdown report generation
│
├── data/
│   ├── scenarios.json         # 10 test scenarios + human reference emails
│   └── results/               # Auto-generated output directory
│       ├── evaluation_results.json
│       ├── evaluation_results.csv
│       └── comparative_analysis.md
│
└── tests/
    └── test_metrics.py        # Unit + integration tests
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([get one free here](https://aistudio.google.com/app/apikey))

### Step 1 — Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/email-generation-assistant.git
cd email-generation-assistant
pip install -r requirements.txt
```

### Step 2 — Configure your API key

```bash
cp .env.example .env
# Edit .env and replace with your actual key:
# GOOGLE_API_KEY=AIza...
```

### Step 3 — Validate setup (no API calls)

```bash
python run_evaluation.py --dry-run
```

Expected output:
```
✓ Scenarios file loaded: 10 scenarios found
✓ All 10 scenarios validated
✓ src.prompts imported successfully
✓ src.metrics imported successfully
✓ textstat available
✓ google-generativeai available
✓ GOOGLE_API_KEY found: AIzaSy…XXXX
✅ Dry-run complete — everything looks good!
```

---

## Running the Evaluation

```bash
# Full evaluation (generates all emails + computes all metrics)
python run_evaluation.py

# Custom paths
python run_evaluation.py --scenarios data/scenarios.json --output data/results
```

> **Note:** The full evaluation makes ~60 API calls and takes approximately
> 5–8 minutes. Inter-call delays are built in to respect Gemini rate limits.

---

## Advanced Prompt Engineering

Three techniques are combined in `src/prompts.py` for Model A:

### Technique 1 — Role-Playing (System Persona)
The LLM is given a specific identity via the system instruction:

```
"You are a world-class professional business writer with 20+ years of experience
crafting impactful corporate communications for Fortune 500 companies..."
```

**Why it works:** Anchoring the model to a high-competence persona shifts its
output distribution toward professional, precise language.

### Technique 2 — Few-Shot Examples
Two complete in-context demonstrations (formal + casual) are prepended:

```
EXAMPLE 1 — FORMAL TONE
  Input:  { Intent, Facts, Tone }
  Output: Subject: ... [full polished email]

EXAMPLE 2 — CASUAL/FRIENDLY TONE
  Input:  { Intent, Facts, Tone }
  Output: Subject: ... [full friendly email]
```

**Why it works:** Examples concretise what "good output" looks like — the
correct structure, the level of fact integration, and the tonal range — without
requiring the model to infer these from instructions alone.

### Technique 3 — Chain-of-Thought (CoT) Scaffold
Before writing, the model is instructed to work through 5 explicit steps:

```
STEP 1 — ANALYZE THE GOAL       (who is the audience? what action is needed?)
STEP 2 — FACT INTEGRATION PLAN  (where does each fact naturally fit?)
STEP 3 — TONE CALIBRATION       (vocabulary, contractions, warmth level)
STEP 4 — STRUCTURE PLAN         (subject → opening → body → CTA → closing)
STEP 5 — WRITE THE EMAIL        (execute the plan)
```

**Why it works:** CoT forces systematic reasoning before generation, reducing
hallucinations and fact omissions. The instruction to suppress intermediate
reasoning from the output keeps responses clean.

---

## Custom Evaluation Metrics

All metrics are defined and implemented in `src/metrics.py`.

### Metric 1 — Fact Recall Score (FRS)

| Property | Details |
|---|---|
| **Focus** | Fact coverage / specificity |
| **Range** | 0.0 → 1.0 |
| **Technique** | Automated Python (token overlap + regex) |

**Definition:** Measures the percentage of user-supplied key facts that are
accurately reflected in the generated email.

**Logic:**
1. For each fact, extract meaningful tokens (lowercase, stopwords removed, length > 2).
2. Also extract numeric patterns (dates, amounts, percentages) via regex.
3. Compute `overlap_ratio = |fact_tokens ∩ email_tokens| / |fact_tokens|`.
4. A fact is **recalled** if `overlap_ratio ≥ 0.40` **OR** `overlap_ratio ≥ 0.20 AND numeric match`.
5. `FRS = recalled_count / total_facts`

---

### Metric 2 — Tone Accuracy Score (TAS)

| Property | Details |
|---|---|
| **Focus** | Tone / register alignment |
| **Range** | 0.0 → 1.0 |
| **Technique** | LLM-as-a-Judge (gemini-2.0-flash, structured rubric) |

**Definition:** Measures how closely the generated email's writing style and
emotional register match the requested tone.

**Logic:** Gemini rates tone alignment on a 1–5 rubric:
- 1 = Completely wrong tone  
- 3 = Partial match  
- 5 = Perfect, consistent tone throughout  

`TAS = raw_score / 5.0`

---

### Metric 3 — Fluency & Professionalism Score (FPS)

| Property | Details |
|---|---|
| **Focus** | Grammar, fluency, professional quality |
| **Range** | 0.0 → 1.0 |
| **Technique** | Hybrid: textstat + LLM-as-a-Judge |

**Definition:** Measures grammatical fluency and professional language quality
using a two-component hybrid approach.

**Logic:**
- **Component A (40%):** `textstat` Flesch Reading Ease (FRE), normalised for
  business email ideal range: `readability = 1 - |FRE - 60| / 60`
- **Component B (60%):** Gemini rates professionalism 1–5; `prof = raw / 5.0`
- `FPS = 0.40 × readability + 0.60 × prof`

---

## Model Comparison Strategy

| | Model A ✦ | Model B |
|---|---|---|
| **Model** | `gemini-2.0-flash` | `gemini-2.0-flash-lite` |
| **Strategy** | Advanced Prompt (Role + Few-Shot + CoT) | Zero-Shot Baseline |
| **System Role** | ✅ World-class writer persona | ✗ None |
| **Examples** | ✅ 2 in-context examples | ✗ None |
| **Reasoning** | ✅ 5-step CoT scaffold | ✗ None |

This design isolates the **measurable benefit of advanced prompt engineering**
by using the same evaluation framework on both strategies.

---

## Output Files

After running the evaluation, three files are generated in `data/results/`:

### `evaluation_results.json`
Full structured data including:
- Metric definitions
- Both generated emails for each scenario
- All raw metric scores (FRS, TAS, FPS + component breakdowns)
- LLM judge reasoning for every TAS and FPS score
- Aggregate statistics

### `evaluation_results.csv`
Flat table with:
- Section 1: Metric definitions
- Section 2: Raw scores for all 10 scenarios (all 3 metrics × 2 models)
- Section 3: Aggregate summary + per-metric winner

### `comparative_analysis.md`
Single-page written analysis answering:
1. Which model/strategy performed better across the 3 metrics?
2. What was the biggest failure mode of the lower-performing model?
3. Which model is recommended for production (justified by metric data)?

---

## Running Tests

```bash
# Unit tests only (no API key required)
pytest tests/ -v

# Unit tests + integration tests (requires GOOGLE_API_KEY in .env)
pytest tests/ -v --integration
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| `google-generativeai` | Gemini API (Model A + B generation, LLM-as-Judge) |
| `textstat` | Flesch Reading Ease for Metric 3 readability component |
| `python-dotenv` | Secure API key management |
| `pandas` | (Available for extended analysis) |
| `reportlab` | (Available for PDF report generation) |

---  Developed by [Shawon](Shawon)

## License

MIT License — see [LICENSE](LICENSE) for details.
