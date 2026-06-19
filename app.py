"""
Email Generation Assistant — Streamlit Web Interface
=====================================================
A responsive, interactive web application for generating professional emails
using advanced prompt engineering techniques.

Features:
  - Manual input: Intent, Key Facts, Tone selection
  - Live generation with Advanced (Role+Few-Shot+CoT) and Baseline prompts
  - Real-time evaluation with 3 custom metrics (FRS, TAS, FPS)
  - Side-by-side comparison with metric visualization
  - Pre-loaded quick-start scenarios
  - Full 10-scenario batch evaluation mode

Usage: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import string
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Import project modules ───────────────────────────────────────────────────
from src.prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    CHAIN_OF_THOUGHT_SCAFFOLD,
    build_advanced_prompt,
    build_simple_prompt,
)
from src.metrics import (
    compute_fact_recall_score,
    compute_tone_accuracy_score,
    compute_fluency_professionalism_score,
    make_judge_client,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Email Generation Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Responsive, modern design
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    /* ── Global ─────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main > div {
        padding-top: 1rem;
    }
    
    /* ── Header ─────────────────────────────────────────────────────────── */
    .email-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    .email-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: white;
    }
    .email-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* ── Metric Cards ───────────────────────────────────────────────────── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf1;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .metric-card.winner {
        border: 2px solid #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    
    /* ── Email Display ──────────────────────────────────────────────────── */
    .email-box {
        background: #fafbfc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #334155;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-height: 500px;
        overflow-y: auto;
    }
    .email-box.advanced {
        border-left: 4px solid #10b981;
    }
    .email-box.baseline {
        border-left: 4px solid #3b82f6;
    }
    
    /* ── Score Bar ──────────────────────────────────────────────────────── */
    .score-bar-container {
        background: #e2e8f0;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    
    /* ── Sidebar ────────────────────────────────────────────────────────── */
    .sidebar-info {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    
    /* ── Fact Check Items ───────────────────────────────────────────────── */
    .fact-check {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.4rem 0;
        font-size: 0.88rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .fact-check:last-child {
        border-bottom: none;
    }
    
    /* ── Responsive ─────────────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .email-header { padding: 1.5rem; }
        .email-header h1 { font-size: 1.5rem; }
        .metric-card .value { font-size: 1.5rem; }
        .email-box { font-size: 0.88rem; padding: 1rem; }
    }
    
    /* ── Streamlit element overrides ────────────────────────────────────── */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* ── Tabs ───────────────────────────────────────────────────────────── */
    div[data-testid="stTab"] {
        font-weight: 500;
    }
    
    /* ── Comparison table ───────────────────────────────────────────────── */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .comparison-table th {
        background: #f8fafc;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        border-bottom: 2px solid #e2e8f0;
    }
    .comparison-table td {
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        border-bottom: 1px solid #f1f5f9;
        text-align: center;
    }
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    .comparison-table tr:hover td {
        background: #f8fafc;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def get_judge_client() -> Optional[OpenAI]:
    """Create and cache the LLM judge client."""
    if not API_KEY:
        return None
    try:
        return make_judge_client(API_KEY, BASE_URL)
    except Exception:
        return None


def _demo_email(intent: str, facts: list[str], tone: str, advanced: bool) -> str:
    """Generate a realistic demo email without any API call. Works offline."""
    tone_lower = tone.lower()

    # Pick greeting based on tone
    if any(w in tone_lower for w in ["formal", "professional", "structured"]):
        greeting = "Dear [Recipient Name],"
        closing = "Best regards,"
    elif any(w in tone_lower for w in ["casual", "friendly"]):
        greeting = "Hi [Name],"
        closing = "Cheers,"
    elif "empathetic" in tone_lower or "apolog" in tone_lower:
        greeting = "Dear [Name],"
        closing = "Warm regards,"
    elif "urgent" in tone_lower:
        greeting = "Dear [Recipient Name],"
        closing = "Respectfully,"
    else:
        greeting = "Dear [Name],"
        closing = "Regards,"

    # Build subject line from intent
    subject = intent.strip().rstrip(".")
    if not subject.lower().startswith("subject:"):
        subject = f"Subject: {subject}"
    else:
        subject = subject

    if advanced:
        # Advanced: well-structured, all facts woven in naturally
        body_paragraphs = []
        # Opening
        if "follow up" in intent.lower() or "follow-up" in intent.lower():
            body_paragraphs.append(
                f"I hope this message finds you well. I am writing to follow up on {intent.lower()}. "
                "I wanted to capture the key points and outline our next steps while the details are fresh."
            )
        elif "request" in intent.lower() or "proposal" in intent.lower():
            body_paragraphs.append(
                f"I am writing on behalf of [Company Name] to formally request {intent.lower()}. "
                "We are evaluating options and believe your organization may be well positioned to meet our requirements."
            )
        elif "complaint" in intent.lower() or "apolog" in intent.lower() or "delay" in intent.lower():
            body_paragraphs.append(
                "I want to begin by sincerely apologizing for the inconvenience you have experienced. "
                "I completely understand how frustrating this situation must be, and I am truly sorry."
            )
        elif "schedule" in intent.lower() or "meeting" in intent.lower() or "kick-off" in intent.lower():
            body_paragraphs.append(
                f"I am writing to coordinate {intent.lower()}. "
                "I believe bringing the team together at the earliest opportunity will set us up for success."
            )
        elif "job" in intent.lower() or "application" in intent.lower() or "interview" in intent.lower():
            body_paragraphs.append(
                f"I hope this message finds you well. I am writing to follow up on my application "
                "and express my continued enthusiasm for this opportunity."
            )
        elif "sales" in intent.lower() or "outreach" in intent.lower() or "product" in intent.lower():
            body_paragraphs.append(
                f"Hope you're having a great week! I'll keep this quick — I think you're really "
                "going to like what I'm about to share."
            )
        elif "partnership" in intent.lower() or "strategic" in intent.lower():
            body_paragraphs.append(
                "I hope this message finds you well. I am reaching out to explore a strategic "
                "partnership opportunity that I believe holds significant mutual value."
            )
        elif "extension" in intent.lower() or "deadline" in intent.lower():
            body_paragraphs.append(
                "I am writing to respectfully request a brief extension on our current deadline. "
                "I recognize the urgency and wanted to bring this to your attention immediately."
            )
        elif "status" in intent.lower() or "update" in intent.lower():
            body_paragraphs.append(
                "Please find below the latest status update for the project. "
                "I will continue to provide regular updates and flag any significant changes."
            )
        else:
            body_paragraphs.append(
                f"I am writing regarding {intent.lower()}. "
                "I wanted to provide you with the relevant details and outline the next steps."
            )

        # Facts paragraph — weave all facts in naturally
        if facts:
            fact_sentences = []
            for f in facts:
                fact_sentences.append(f"{f}.")
            body_paragraphs.append(" ".join(fact_sentences))

        # CTA
        body_paragraphs.append(
            "Please let me know if you have any questions or would like to discuss further. "
            "I look forward to hearing from you."
        )

        body = "\n\n".join(body_paragraphs)
        return f"{subject}\n\n{greeting}\n\n{body}\n\n{closing}\n[Your Name]"

    else:
        # Baseline: shorter, less structured, may miss some facts
        # Only include first 2-3 facts (simulating fact omission)
        included_facts = facts[:2] if len(facts) > 2 else facts
        facts_text = " ".join(f"{f}." for f in included_facts)

        if "follow up" in intent.lower():
            body = f"Following up on {intent.lower()}. {facts_text} Please let me know your thoughts."
        elif "request" in intent.lower():
            body = f"We need {intent.lower()}. {facts_text} Please send details."
        elif "complaint" in intent.lower() or "delay" in intent.lower():
            body = f"Sorry about the issue. {facts_text} We're working on it."
        elif "schedule" in intent.lower() or "meeting" in intent.lower():
            body = f"We need to {intent.lower()}. {facts_text} Please confirm your availability."
        elif "job" in intent.lower() or "application" in intent.lower():
            body = f"I applied recently and wanted to follow up. {facts_text} I'm available anytime."
        elif "sales" in intent.lower() or "outreach" in intent.lower():
            body = f"Want to introduce our product. {facts_text} Let me know if interested."
        elif "partnership" in intent.lower():
            body = f"Interested in partnering. {facts_text} Would love to discuss."
        elif "extension" in intent.lower() or "deadline" in intent.lower():
            body = f"We need more time. {facts_text} Can we extend?"
        else:
            body = f"{intent}. {facts_text} Thanks."

        return f"{subject}\n\n{greeting}\n\n{body}\n\nThanks,\n[Your Name]"


def generate_email(
    intent: str,
    facts: list[str],
    tone: str,
    use_advanced: bool,
    api_key: str = API_KEY,
) -> dict:
    """Generate an email using the specified strategy. Falls back to demo mode on error."""
    if not api_key:
        return {
            "email": _demo_email(intent, facts, tone, use_advanced),
            "strategy": "Demo Mode (no API key)" + (" — Advanced" if use_advanced else " — Baseline"),
            "error": False,
        }

    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60)
        if use_advanced:
            prompt = build_advanced_prompt(intent, facts, tone)
            strategy = "Advanced (Role + Few-Shot + CoT)"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        else:
            prompt = build_simple_prompt(intent, facts, tone)
            strategy = "Baseline (Zero-Shot)"
            messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            temperature=0.3,
            max_tokens=512,  # Reduced to stay within credit limits
        )
        email_text = (response.choices[0].message.content or "").strip()
        if email_text:
            return {"email": email_text, "strategy": strategy, "error": False}
        # Empty response — fall through to demo
    except Exception:
        pass

    # Fallback to demo mode on any error (no credits, network issue, etc.)
    return {
        "email": _demo_email(intent, facts, tone, use_advanced),
        "strategy": "Demo Mode (API unavailable)" + (" — Advanced" if use_advanced else " — Baseline"),
        "error": False,
    }


def evaluate_email(
    facts: list[str],
    tone: str,
    email: str,
    judge: Optional[OpenAI],
) -> dict:
    """Run all 3 custom metrics on an email. Returns dict with scores."""
    # FRS — always computed locally
    frs = compute_fact_recall_score(facts, email)

    # TAS — LLM-as-Judge (fallback to heuristic if no judge)
    if judge and not email.startswith("["):
        try:
            tas = compute_tone_accuracy_score(tone, email, judge)
            tas_score = tas["score"]
            tas_reason = tas["reasoning"]
        except Exception:
            tas_score = _heuristic_tone_score(tone, email)
            tas_reason = "Judge unavailable — heuristic fallback"
    else:
        tas_score = _heuristic_tone_score(tone, email)
        tas_reason = "Heuristic scoring (no judge)"

    # FPS — hybrid
    if judge and not email.startswith("["):
        try:
            fps = compute_fluency_professionalism_score(email, judge)
            fps_score = fps["score"]
            fps_read = fps.get("flesch_reading_ease", 0)
            fps_prof = fps.get("professionalism_raw", 0)
        except Exception:
            fps_score = _heuristic_fps_score(email)
            fps_read = 50.0
            fps_prof = 3
    else:
        fps_score = _heuristic_fps_score(email)
        fps_read = 50.0
        fps_prof = 3

    composite = round((frs["score"] + tas_score + fps_score) / 3, 4)
    return {
        "frs": frs["score"],
        "frs_details": frs.get("fact_details", []),
        "tas": tas_score,
        "tas_reason": tas_reason,
        "fps": fps_score,
        "fps_read": fps_read,
        "fps_prof": fps_prof,
        "composite": composite,
    }


def _heuristic_tone_score(tone: str, email: str) -> float:
    """Fallback tone scoring when judge is unavailable."""
    tone_lower = tone.lower()
    email_lower = email.lower()
    score = 0.5  # baseline

    formal_indicators = ["dear", "sincerely", "regards", "respectfully", "kindly"]
    casual_indicators = ["hey", "hi,", "cheers", "thanks,", "awesome", "cool"]
    urgent_indicators = ["urgent", "immediately", "asap", "deadline", "time-sensitive"]
    empathetic_indicators = ["understand", "apologize", "sorry", "appreciate", "feel"]

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
        score += 0.1  # professional default

    return min(1.0, max(0.1, round(score, 4)))


def _heuristic_fps_score(email: str) -> float:
    """Fallback FPS scoring when judge is unavailable."""
    has_subject = email.lower().startswith("subject:")
    has_greeting = any(
        email.lower().startswith(g)
        for g in ["dear", "hi,", "hello", "hey", "good morning", "good afternoon"]
    )
    has_closing = any(
        c in email.lower()
        for c in ["regards", "sincerely", "best,", "cheers", "thanks,"]
    )
    word_count = len(email.split())
    structure_score = (
        (0.2 if has_subject else 0)
        + (0.2 if has_greeting else 0)
        + (0.2 if has_closing else 0)
        + (0.2 if 50 <= word_count <= 300 else 0.1)
        + 0.2  # base readability
    )
    return min(1.0, round(structure_score, 4))


def score_bar(value: float, color: str = "#10b981") -> str:
    """Return HTML for a score bar."""
    pct = int(value * 100)
    return f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    """


def metric_card_html(
    label: str, value: float, delta: float = 0, is_winner: bool = False
) -> str:
    """Return HTML for a metric card."""
    pct = f"{value * 100:.1f}"
    delta_str = f"{delta:+.1f}%" if delta != 0 else ""
    delta_color = "#10b981" if delta >= 0 else "#ef4444"
    winner_class = " winner" if is_winner else ""
    return f"""
    <div class="metric-card{winner_class}">
        <div class="label">{label}</div>
        <div class="value">{pct}</div>
        {f'<div class="delta" style="color:{delta_color}">{delta_str}</div>' if delta_str else ""}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_scenarios() -> list[dict]:
    """Load test scenarios from JSON."""
    try:
        with open("data/scenarios.json") as f:
            return json.load(f)["scenarios"]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem;">📧</div>
        <h2 style="margin:0.5rem 0 0 0;font-size:1.3rem;">Email Assistant</h2>
        <p style="font-size:0.8rem;color:#64748b;margin:0;">AI-Powered • Advanced Prompt Engineering</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Mode selection
    mode = st.radio(
        "Mode",
        ["✍️ Compose Email", "📊 Batch Evaluation", "📚 Documentation"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API status
    if API_KEY:
        st.success("🟢 API Key Configured", icon="✅")
        st.caption("Live generation + demo fallback available")
    else:
        st.info("🟡 Demo Mode Active", icon="ℹ️")
        st.caption("Set OPENROUTER_API_KEY in .env for live generation. Demo works fully offline.")

    st.markdown("---")

    # Mode info
    st.markdown(
        """
    <div style="background:#f8fafc;border-radius:8px;padding:0.75rem;font-size:0.8rem;color:#475569;">
        <b>💡 Demo Mode:</b> Works without API credits. Generates realistic emails using built-in templates. Upgrade to live mode by adding an OpenRouter API key.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Quick info
    with st.expander("ℹ️ About"):
        st.markdown(
            """
        **3 Custom Metrics:**
        - **FRS** — Fact Recall Score
        - **TAS** — Tone Accuracy Score
        - **FPS** — Fluency & Professionalism

        **Prompt Strategies:**
        - **Advanced:** Role + Few-Shot + CoT
        - **Baseline:** Zero-shot only

        **Scoring:**
        - FRS: Fully automated (token overlap)
        - TAS: Heuristic scoring (LLM judge in live mode)
        - FPS: Hybrid readability + structure analysis
        """
        )

    st.markdown(
        "<div style='text-align:center;font-size:0.75rem;color:#94a3b8;margin-top:2rem;'>v1.0 • Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: COMPOSE EMAIL (Interactive)
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "✍️ Compose Email":
    # Header
    st.markdown(
        """
    <div class="email-header">
        <h1>📧 Email Generation Assistant</h1>
        <p>Generate professional emails with advanced prompt engineering. Compare Advanced vs Baseline strategies side-by-side.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Input Form ─────────────────────────────────────────────────────────
    col_form, col_preview = st.columns([5, 4])

    with col_form:
        st.subheader("📝 Input Details")

        # Quick-fill from scenarios
        scenarios = load_scenarios()
        if scenarios:
            scenario_names = ["— Custom Input —"] + [
                f"#{s['id']}: {s['name']}" for s in scenarios
            ]
            quick_fill = st.selectbox(
                "Quick-fill from scenario (optional)",
                scenario_names,
                help="Select a pre-built scenario to auto-fill the form",
            )
            if quick_fill != "— Custom Input —":
                idx = scenario_names.index(quick_fill) - 1
                selected = scenarios[idx]
                st.session_state["prefill_intent"] = selected["intent"]
                st.session_state["prefill_facts"] = "\n".join(selected["key_facts"])
                st.session_state["prefill_tone"] = selected["tone"]

        # Intent
        intent = st.text_input(
            "Intent",
            value=st.session_state.get("prefill_intent", ""),
            placeholder="e.g., Follow up after meeting, Request proposal details, Schedule interview",
            help="The core purpose of your email",
        )

        # Key Facts
        facts_input = st.text_area(
            "Key Facts (one per line)",
            value=st.session_state.get("prefill_facts", ""),
            height=120,
            placeholder="Meeting held Tuesday\nBudget approved\nNeed timeline by Friday",
            help="Each fact will be woven into the email naturally",
        )

        # Tone
        tone_options = [
            "Formal",
            "Professional",
            "Professional and direct",
            "Casual and friendly",
            "Enthusiastic yet professional",
            "Empathetic and apologetic",
            "Urgent but respectful and professional",
            "Persuasive and formal",
            "Formal and structured",
            "Formal and encouraging",
        ]
        prefill_tone = st.session_state.get("prefill_tone", "")
        tone_index = (
            tone_options.index(prefill_tone) if prefill_tone in tone_options else 0
        )
        tone = st.selectbox(
            "Tone",
            tone_options,
            index=tone_index,
            help="The desired writing style for your email",
        )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            with col_b:
                max_tokens = st.select_slider(
                    "Max Tokens", [256, 512, 768, 1024, 1536, 2048], value=1024
                )

        # Generate button
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button(
            "🚀 Generate Emails",
            type="primary",
            use_container_width=True,
            disabled=not intent.strip() or not facts_input.strip(),
        )

    with col_preview:
        st.subheader("📋 Input Preview")

        if intent or facts_input:
            st.markdown(f"**Intent:** {intent or '_not set_'}")
            st.markdown(f"**Tone:** {tone}")
            st.markdown("**Key Facts:**")
            for line in facts_input.strip().split("\n"):
                line = line.strip()
                if line:
                    st.markdown(f"  • {line}")
        else:
            st.info("Fill in the form and click **Generate Emails** to see results")

        if not API_KEY:
            st.warning("⚠️ No API key — results will use demo mode")

    # ── Generation & Results ────────────────────────────────────────────────

    if generate_clicked and intent.strip() and facts_input.strip():
        facts = [f.strip() for f in facts_input.strip().split("\n") if f.strip()]

        if not facts:
            st.error("Please enter at least one key fact.")
        else:
            # Progress
            progress = st.progress(0, text="Initializing...")
            judge = get_judge_client()

            # Generate Model A
            progress.progress(15, text="Generating Advanced email (Model A)...")
            result_a = generate_email(intent, facts, tone, use_advanced=True)

            # Generate Model B
            progress.progress(40, text="Generating Baseline email (Model B)...")
            result_b = generate_email(intent, facts, tone, use_advanced=False)

            # Evaluate
            progress.progress(60, text="Evaluating with custom metrics...")
            eval_a = evaluate_email(facts, tone, result_a["email"], judge)

            progress.progress(80, text="Evaluating baseline...")
            eval_b = evaluate_email(facts, tone, result_b["email"], judge)

            progress.progress(100, text="Complete!")
            st.session_state["result_a"] = result_a
            st.session_state["result_b"] = result_b
            st.session_state["eval_a"] = eval_a
            st.session_state["eval_b"] = eval_b
            st.session_state["facts"] = facts
            st.session_state["intent"] = intent
            st.session_state["tone"] = tone

    # ── Display Results ─────────────────────────────────────────────────────

    if "result_a" in st.session_state:
        result_a = st.session_state["result_a"]
        result_b = st.session_state["result_b"]
        eval_a = st.session_state["eval_a"]
        eval_b = st.session_state["eval_b"]
        facts = st.session_state["facts"]
        intent = st.session_state["intent"]
        tone = st.session_state["tone"]

        st.markdown("---")
        st.subheader("📊 Evaluation Results")

        # ── Metric Cards ─────────────────────────────────────────────────────
        winner = "A" if eval_a["composite"] >= eval_b["composite"] else "B"
        frs_delta = (eval_a["frs"] - eval_b["frs"]) * 100
        tas_delta = (eval_a["tas"] - eval_b["tas"]) * 100
        fps_delta = (eval_a["fps"] - eval_b["fps"]) * 100
        comp_delta = (eval_a["composite"] - eval_b["composite"]) * 100

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                metric_card_html("FRS", eval_a["frs"], frs_delta, winner == "A"),
                unsafe_allow_html=True,
            )
            st.caption("Fact Recall Score")
        with m2:
            st.markdown(
                metric_card_html("TAS", eval_a["tas"], tas_delta, winner == "A"),
                unsafe_allow_html=True,
            )
            st.caption("Tone Accuracy Score")
        with m3:
            st.markdown(
                metric_card_html("FPS", eval_a["fps"], fps_delta, winner == "A"),
                unsafe_allow_html=True,
            )
            st.caption("Fluency & Professionalism")
        with m4:
            st.markdown(
                metric_card_html(
                    "COMPOSITE", eval_a["composite"], comp_delta, winner == "A"
                ),
                unsafe_allow_html=True,
            )
            st.caption("Overall Score")

        # Winner banner
        margin = abs(eval_a["composite"] - eval_b["composite"]) * 100
        if winner == "A":
            st.success(
                f"🏆 **Model A (Advanced)** wins by **+{margin:.1f}%** — Role + Few-Shot + CoT outperforms Baseline",
                icon="🎯",
            )
        else:
            st.info(f"Model B wins by {margin:.1f}%")

        # ── Side-by-Side Email Comparison ────────────────────────────────────
        st.markdown("---")
        st.subheader("📧 Generated Emails")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                '<div style="font-size:0.85rem;font-weight:600;color:#10b981;margin-bottom:0.5rem;">🟢 MODEL A — ADVANCED</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="email-box advanced">{result_a["email"]}</div>',
                unsafe_allow_html=True,
            )
            # Score breakdown
            st.markdown(
                f"""
            <div style="margin-top:0.5rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span>FRS: {eval_a['frs']*100:.1f}%</span>
                    <span>TAS: {eval_a['tas']*100:.1f}%</span>
                    <span>FPS: {eval_a['fps']*100:.1f}%</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                '<div style="font-size:0.85rem;font-weight:600;color:#3b82f6;margin-bottom:0.5rem;">🔵 MODEL B — BASELINE</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="email-box baseline">{result_b["email"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div style="margin-top:0.5rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span>FRS: {eval_b['frs']*100:.1f}%</span>
                    <span>TAS: {eval_b['tas']*100:.1f}%</span>
                    <span>FPS: {eval_b['fps']*100:.1f}%</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Detailed Metric Breakdown ────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Detailed Metric Breakdown")

        tab1, tab2, tab3 = st.tabs(
            ["📋 Fact Recall (FRS)", "🎭 Tone Accuracy (TAS)", "✨ Fluency & Professionalism (FPS)"]
        )

        with tab1:
            st.markdown("**Fact Recall Score (FRS)** — Token overlap + numeric matching")
            for i, fact in enumerate(facts):
                detail_a = (
                    eval_a["frs_details"][i]
                    if i < len(eval_a["frs_details"])
                    else {}
                )
                detail_b = (
                    eval_b["frs_details"][i]
                    if i < len(eval_b["frs_details"])
                    else {}
                )
                recalled_a = detail_a.get("recalled", False)
                recalled_b = detail_b.get("recalled", False)

                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"**Fact {i+1}:** {fact}")
                with c2:
                    st.markdown(
                        f"{'✅' if recalled_a else '❌'} A: {'Yes' if recalled_a else 'No'}"
                    )
                with c3:
                    st.markdown(
                        f"{'✅' if recalled_b else '❌'} B: {'Yes' if recalled_b else 'No'}"
                    )

        with tab2:
            st.markdown("**Tone Accuracy Score (TAS)** — LLM-as-a-Judge rubric")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Model A:**")
                st.info(eval_a["tas_reason"])
                st.markdown(score_bar(eval_a["tas"], "#10b981"), unsafe_allow_html=True)
                st.caption(f"Score: {eval_a['tas']*100:.1f}%")
            with c2:
                st.markdown("**Model B:**")
                st.info(eval_b["tas_reason"])
                st.markdown(score_bar(eval_b["tas"], "#3b82f6"), unsafe_allow_html=True)
                st.caption(f"Score: {eval_b['tas']*100:.1f}%")

        with tab3:
            st.markdown("**Fluency & Professionalism Score (FPS)** — Hybrid (textstat + LLM)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Model A:**")
                st.markdown(f"- Flesch Reading Ease: {eval_a['fps_read']:.1f}")
                st.markdown(f"- Professionalism: {eval_a['fps_prof']}/5")
                st.markdown(score_bar(eval_a["fps"], "#10b981"), unsafe_allow_html=True)
                st.caption(f"Score: {eval_a['fps']*100:.1f}%")
            with c2:
                st.markdown("**Model B:**")
                st.markdown(f"- Flesch Reading Ease: {eval_b['fps_read']:.1f}")
                st.markdown(f"- Professionalism: {eval_b['fps_prof']}/5")
                st.markdown(score_bar(eval_b["fps"], "#3b82f6"), unsafe_allow_html=True)
                st.caption(f"Score: {eval_b['fps']*100:.1f}%")

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📥 Export Results"):
            export_data = {
                "intent": intent,
                "tone": tone,
                "facts": facts,
                "model_a": {
                    "email": result_a["email"],
                    "strategy": result_a["strategy"],
                    "scores": {
                        "frs": eval_a["frs"],
                        "tas": eval_a["tas"],
                        "fps": eval_a["fps"],
                        "composite": eval_a["composite"],
                    },
                },
                "model_b": {
                    "email": result_b["email"],
                    "strategy": result_b["strategy"],
                    "scores": {
                        "frs": eval_b["frs"],
                        "tas": eval_b["tas"],
                        "fps": eval_b["fps"],
                        "composite": eval_b["composite"],
                    },
                },
                "winner": "A" if eval_a["composite"] >= eval_b["composite"] else "B",
                "timestamp": datetime.now().isoformat(),
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"email_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📊 Batch Evaluation":
    st.markdown(
        """
    <div class="email-header">
        <h1>📊 Batch Evaluation</h1>
        <p>Run all 10 test scenarios through both models and compare aggregate results.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    scenarios = load_scenarios()

    if not scenarios:
        st.error("No scenarios found. Make sure data/scenarios.json exists.")
    else:
        col_info, col_run = st.columns([3, 1])
        with col_info:
            st.info(
                f"**{len(scenarios)} scenarios** loaded. Each will be evaluated with both Advanced and Baseline strategies."
            )
        with col_run:
            run_batch = st.button(
                "🚀 Run Full Evaluation", type="primary", use_container_width=True
            )

        if run_batch:
            judge = get_judge_client()
            all_results = []
            progress = st.progress(0, text="Starting batch evaluation...")

            for idx, sc in enumerate(scenarios):
                progress.progress(
                    int((idx / len(scenarios)) * 100),
                    text=f"Scenario {idx+1}/{len(scenarios)}: {sc['name'][:40]}...",
                )

                facts = sc["key_facts"]
                tone = sc["tone"]

                # Try live generation, fall back to DB
                if API_KEY:
                    result_a = generate_email(
                        sc["intent"], facts, tone, use_advanced=True
                    )
                    result_b = generate_email(
                        sc["intent"], facts, tone, use_advanced=False
                    )
                else:
                    # Use demo emails from email_assistant.py EMAIL_DB
                    result_a = {"email": "[Demo mode — set API key for live generation]", "error": True}
                    result_b = {"email": "[Demo mode — set API key for live generation]", "error": True}

                eval_a = evaluate_email(facts, tone, result_a["email"], judge)
                eval_b = evaluate_email(facts, tone, result_b["email"], judge)

                all_results.append(
                    {
                        "id": sc["id"],
                        "name": sc["name"],
                        "tone": tone,
                        "a_frs": eval_a["frs"],
                        "a_tas": eval_a["tas"],
                        "a_fps": eval_a["fps"],
                        "a_comp": eval_a["composite"],
                        "b_frs": eval_b["frs"],
                        "b_tas": eval_b["tas"],
                        "b_fps": eval_b["fps"],
                        "b_comp": eval_b["composite"],
                    }
                )

            progress.progress(100, text="Complete!")

            # Aggregate
            aa_frs = round(statistics.mean(r["a_frs"] for r in all_results), 4)
            aa_tas = round(statistics.mean(r["a_tas"] for r in all_results), 4)
            aa_fps = round(statistics.mean(r["a_fps"] for r in all_results), 4)
            aa_comp = round(statistics.mean(r["a_comp"] for r in all_results), 4)
            ab_frs = round(statistics.mean(r["b_frs"] for r in all_results), 4)
            ab_tas = round(statistics.mean(r["b_tas"] for r in all_results), 4)
            ab_fps = round(statistics.mean(r["b_fps"] for r in all_results), 4)
            ab_comp = round(statistics.mean(r["b_comp"] for r in all_results), 4)
            wins_a = sum(1 for r in all_results if r["a_comp"] >= r["b_comp"])

            # Display aggregate
            st.markdown("---")
            st.subheader("📊 Aggregate Results")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("FRS", f"{aa_frs*100:.1f}%", f"{(aa_frs-ab_frs)*100:+.1f}%")
            with m2:
                st.metric("TAS", f"{aa_tas*100:.1f}%", f"{(aa_tas-ab_tas)*100:+.1f}%")
            with m3:
                st.metric("FPS", f"{aa_fps*100:.1f}%", f"{(aa_fps-ab_fps)*100:+.1f}%")
            with m4:
                st.metric(
                    "COMPOSITE",
                    f"{aa_comp*100:.1f}%",
                    f"{(aa_comp-ab_comp)*100:+.1f}%",
                )

            st.success(f"🏆 Model A wins {wins_a}/{len(scenarios)} scenarios")

            # Per-scenario table
            st.markdown("---")
            st.subheader("📋 Per-Scenario Results")

            table_html = """
            <table class="comparison-table">
                <tr>
                    <th>ID</th>
                    <th>Scenario</th>
                    <th>Tone</th>
                    <th>A-FRS</th>
                    <th>A-TAS</th>
                    <th>A-FPS</th>
                    <th>A-Comp</th>
                    <th>B-FRS</th>
                    <th>B-TAS</th>
                    <th>B-FPS</th>
                    <th>B-Comp</th>
                    <th>Winner</th>
                </tr>
            """
            for r in all_results:
                winner = "🟢 A" if r["a_comp"] >= r["b_comp"] else "🔵 B"
                table_html += f"""
                <tr>
                    <td>{r['id']}</td>
                    <td style="text-align:left;">{r['name']}</td>
                    <td style="text-align:left;font-size:0.8rem;">{r['tone']}</td>
                    <td>{r['a_frs']*100:.0f}%</td>
                    <td>{r['a_tas']*100:.0f}%</td>
                    <td>{r['a_fps']*100:.0f}%</td>
                    <td><strong>{r['a_comp']*100:.1f}%</strong></td>
                    <td>{r['b_frs']*100:.0f}%</td>
                    <td>{r['b_tas']*100:.0f}%</td>
                    <td>{r['b_fps']*100:.0f}%</td>
                    <td><strong>{r['b_comp']*100:.1f}%</strong></td>
                    <td>{winner}</td>
                </tr>
                """
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            # Export
            with st.expander("📥 Export Results"):
                csv_lines = [
                    "ID,Scenario,Tone,A-FRS,A-TAS,A-FPS,A-Comp,B-FRS,B-TAS,B-FPS,B-Comp,Winner"
                ]
                for r in all_results:
                    csv_lines.append(
                        f"{r['id']},{r['name']},{r['tone']},{r['a_frs']},{r['a_tas']},{r['a_fps']},{r['a_comp']},{r['b_frs']},{r['b_tas']},{r['b_fps']},{r['b_comp']},{'A' if r['a_comp']>=r['b_comp'] else 'B'}"
                    )
                st.download_button(
                    "Download CSV",
                    data="\n".join(csv_lines),
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📚 Documentation":
    st.markdown(
        """
    <div class="email-header">
        <h1>📚 Documentation</h1>
        <p>Prompt engineering techniques, metric definitions, and architecture overview.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab_prompts, tab_metrics, tab_arch = st.tabs(
        ["🧠 Prompt Engineering", "📐 Custom Metrics", "🏗️ Architecture"]
    )

    with tab_prompts:
        st.header("Advanced Prompt Engineering")

        st.subheader("1. Role-Playing")
        st.markdown(
            "The LLM is assigned a professional persona to ground its vocabulary and judgment:"
        )
        st.code(SYSTEM_PROMPT, language="text")

        st.subheader("2. Few-Shot Examples")
        st.markdown(
            "Two in-context examples (formal + casual) demonstrate the expected output format:"
        )
        with st.expander("View Few-Shot Examples"):
            st.code(FEW_SHOT_EXAMPLES, language="text")

        st.subheader("3. Chain-of-Thought (CoT)")
        st.markdown(
            "A 5-step reasoning scaffold guides the model. Reasoning is internal only — never exposed to users:"
        )
        st.code(CHAIN_OF_THOUGHT_SCAFFOLD, language="text")

        st.subheader("Baseline Prompt (Model B)")
        st.markdown("Zero-shot, minimal instruction for comparison:")
        st.code(
            build_simple_prompt(
                "Follow up after meeting",
                ["Meeting held Tuesday", "Budget approved"],
                "Professional",
            ),
            language="text",
        )

    with tab_metrics:
        st.header("Custom Evaluation Metrics")

        st.subheader("Metric 1: Fact Recall Score (FRS)")
        st.markdown("""
        **Purpose:** Measure whether required facts appear in the generated email.
        
        **Logic:**
        - Tokenize both facts and email (remove stopwords, punctuation)
        - For each fact, compute token overlap ratio
        - Boost score if numeric values match (dates, amounts, percentages)
        - Fully included = 1, Partially = 0.5, Missing = 0
        
        **Formula:** `FRS = Σ(fact_scores) / total_facts`
        
        **Range:** 0–100%
        """)

        st.subheader("Metric 2: Tone Accuracy Score (TAS)")
        st.markdown("""
        **Purpose:** Measure tone consistency with the requested style.
        
        **Method:** LLM-as-a-Judge evaluation with a 5-point rubric:
        - 1 = Completely wrong tone
        - 2 = Mostly wrong
        - 3 = Partial match
        - 4 = Mostly correct
        - 5 = Perfect tone match
        
        **Criteria:** Vocabulary complexity, sentence structure, formality, emotional warmth, greeting/closing appropriateness.
        
        **Range:** 0–100% (raw_score / 5)
        """)

        st.subheader("Metric 3: Fluency & Professionalism Score (FPS)")
        st.markdown("""
        **Purpose:** Measure structural quality and professional writing standards.
        
        **Components (Hybrid):**
        - **40% Readability:** textstat Flesch Reading Ease (optimal ~60)
        - **60% Professionalism:** LLM-as-a-Judge 5-point rating
        
        **Structure Checklist:**
        - Subject line present
        - Appropriate greeting
        - Clear body paragraphs
        - Call to action
        - Professional closing
        - Grammar quality
        
        **Range:** 0–100%
        """)

    with tab_arch:
        st.header("System Architecture")

        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    Streamlit Frontend                       │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
        │  │ Compose  │  │  Batch   │  │  Docs    │  │ Export   │   │
        │  │  Email   │  │  Eval    │  │  Tab     │  │ Results  │   │
        │  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘   │
        └───────┼─────────────┼──────────────────────────────────────┘
                │             │
        ┌───────▼─────────────▼──────────────────────────────────────┐
        │                   Backend (Python)                          │
        │  ┌──────────────────┐  ┌──────────────────┐                │
        │  │  EmailGenerator   │  │  Evaluator       │                │
        │  │  ├─ Model A       │  │  ├─ FRS (local)  │                │
        │  │  │  (Advanced)    │  │  ├─ TAS (LLM)    │                │
        │  │  └─ Model B       │  │  └─ FPS (hybrid) │                │
        │  │     (Baseline)    │  │                  │                │
        │  └──────────────────┘  └──────────────────┘                │
        │                                                            │
        │  ┌──────────────────┐  ┌──────────────────┐                │
        │  │  Prompt Engine    │  │  LLM Judge       │                │
        │  │  ├─ Role-Playing  │  │  (OpenRouter)    │                │
        │  │  ├─ Few-Shot      │  │                  │                │
        │  │  └─ CoT Scaffold  │  │                  │                │
        │  └──────────────────┘  └──────────────────┘                │
        └─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   OpenRouter API   │
                    │  (openrouter/auto) │
                    └───────────────────┘
        ```
        """)

        st.subheader("File Structure")
        st.code(
            """
email_assistant/
├── app.py                    # Streamlit frontend
├── email_assistant.py        # CLI application
├── src/
│   ├── prompts.py            # Advanced + Baseline prompts
│   ├── generator.py          # Email generation (Model A & B)
│   ├── metrics.py            # FRS, TAS, FPS metrics
│   ├── evaluator.py          # Evaluation pipeline
│   └── report_generator.py   # Report generation
├── data/
│   ├── scenarios.json        # 10 test scenarios
│   └── results/              # Evaluation outputs
├── tests/
│   └── test_metrics.py       # 19 unit tests
├── requirements.txt
├── README.md
└── .env.example
        """,
            language="text",
        )
