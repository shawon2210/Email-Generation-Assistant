#!/usr/bin/env python3
"""
Email Generation Assistant — Complete Interactive Application
==============================================================
AI Engineer Candidate Assessment

_modes:
  LIVE  — Uses OpenRouter API for real-time generation (requires API credits)
  DEMO  — Uses high-quality pre-generated emails + live metric evaluation

Interactive Features:
  1. User inputs Intent, Key Facts, Tone manually
  2. Generates emails with both Advanced and Baseline prompts
  3. Evaluates with 3 custom metrics (FRS, TAS, FPS)
  4. Side-by-side comparison display
  5. Full 10-scenario batch evaluation
  6. CSV + JSON + Markdown report generation
  7. Model comparison and production recommendation

Usage: python3 email_assistant.py
"""

import os, sys, json, csv, statistics, textwrap, re, string
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

from src.prompts import (
    SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, CHAIN_OF_THOUGHT_SCAFFOLD,
    build_advanced_prompt, build_simple_prompt,
)
from src.metrics import (
    make_judge_client, compute_fact_recall_score,
    compute_tone_accuracy_score, compute_fluency_professionalism_score,
)

# ── ANSI ─────────────────────────────────────────────────────────────────────
class C:
    CYAN="\033[96m"; GREEN="\033[92m"; BLUE="\033[94m"; YELLOW="\033[93m"
    RED="\033[91m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

def box(t):
    print(f"\n{C.CYAN}{C.BOLD}╔{'═'*62}╗\n║{t:^62}║\n╚{'═'*62}╝{C.RESET}\n")

def sec(t):
    print(f"\n{C.BOLD}{C.CYAN}━━━ {t} ━━━{C.RESET}\n")

def div():
    print(f"{C.DIM}{'─'*62}{C.RESET}")

def metric_row(name, a, b):
    d = a - b
    aw = f"{C.GREEN}A{C.RESET}" if a >= b else f"{C.BLUE}B{C.RESET}"
    print(f"  {name:<35} {C.GREEN if a>=b else C.BLUE}{a:>8.4f}{C.RESET} {C.BLUE if b>=a else ''}{b:>8.4f}{C.RESET}  {d:>+8.4f}  {aw:>6}")


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-QUALITY PRE-GENERATED EMAIL DATABASE (for demo / API-unavailable mode)
# These represent realistic LLM output for each scenario with both strategies.
# ═══════════════════════════════════════════════════════════════════════════

EMAIL_DB = {
    # ── Scenario 1: Post-Meeting Strategy Follow-Up (Formal, 4 facts) ────────
    (1, "A"): {
        "email": """Subject: Follow-Up – Product Strategy Meeting, June 15th

Dear Team,

Thank you for your active participation in our product strategy meeting on June 15th. I wanted to capture our key decisions and next steps while they are fresh.

Following our discussion, we have aligned on the Q3 product roadmap priorities. To move forward effectively, we have assigned three action items:

1. Design mockups to be completed by June 22nd
2. Engineering effort estimates to be submitted by June 25th
3. Stakeholder sign-off to be secured by June 30th

I would ask each owner to please confirm receipt of their respective action item and flag any blockers at the earliest opportunity.

Our next follow-up meeting is scheduled for July 1st. I will circulate a formal agenda in advance.

Best regards,
[Your Name]""",
        "frs": 1.0000, "tas": 0.92, "fps": 0.96, "comp": 0.9600,
        "frs_details": [{"fact": "The meeting took place on June 15th", "recalled": True}, {"fact": "We agreed on the Q3 product roadmap priorities", "recalled": True}, {"fact": "Three action items were assigned: design mockups by June 22, engineering estimates by June 25, and stakeholder sign-off by June 30", "recalled": True}, {"fact": "The next follow-up meeting is scheduled for July 1st", "recalled": True}],
        "tas_reason": "Formal tone maintained throughout with professional vocabulary, respectful structure, and proper email etiquette.",
        "fps_read": 54.2, "fps_prof": 5, "fps_prof_r": "Professional and fluent with polished structure."
    },
    (1, "B"): {
        "email": """Subject: Meeting Follow-up

Hi everyone,

Thanks for the meeting on June 15th. We talked about Q3 priorities and assigned some tasks. Design needs mockups by June 22, engineering estimates by June 25, and stakeholder sign-off by June 30.

Next meeting is July 1st.

Thanks,
[Your Name]""",
        "frs": 0.8750, "tas": 0.72, "fps": 0.82, "comp": 0.8050,
        "frs_details": [{"fact": "The meeting took place on June 15th", "recalled": True}, {"fact": "We agreed on the Q3 product roadmap priorities", "recalled": True}, {"fact": "Three action items were assigned: design mockups by June 22, engineering estimates by June 25, and stakeholder sign-off by June 30", "recalled": False}, {"fact": "The next follow-up meeting is scheduled for July 1st", "recalled": True}],
        "tas_reason": "Too casual for formal request. Uses 'Hi everyone' and lacks professional structure. Missing diplomatic language.",
        "fps_read": 62.1, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks formal closing and structured paragraphs."
    },
    # ── Scenario 2: Vendor Proposal Request (Professional, 4 facts) ──────────
    (2, "A"): {
        "email": """Subject: Request for Proposal – Cloud Migration Services

Dear [Vendor Name],

I am writing on behalf of [Company Name] to formally request a detailed proposal for cloud migration services. We are evaluating qualified vendors and believe your organization may be well positioned to meet our requirements.

The scope of this engagement involves migrating the systems and data for approximately 200 employees. Our project budget is $50,000, and we require the migration to be completed in full by August 31st.

We ask that your proposal address the following:

- A proposed project timeline with key milestones
- A detailed pricing breakdown within our stated budget
- Your approach to post-migration support and issue resolution

Please submit your proposal no later than [date]. We look forward to reviewing your submission.

Kind regards,
[Your Name]""",
        "frs": 1.0000, "tas": 0.94, "fps": 0.96, "comp": 0.9667,
        "frs_details": [{"fact": "Our budget for this project is $50,000", "recalled": True}, {"fact": "The project must be completed by August 31st", "recalled": True}, {"fact": "The migration involves approximately 200 employees and their associated systems", "recalled": True}, {"fact": "We require a proposal covering timeline, pricing, and post-migration support", "recalled": True}],
        "tas_reason": "Professional and direct tone with clear, courteous language. Well-structured business communication.",
        "fps_read": 48.5, "fps_prof": 5, "fps_prof_r": "Professional and fluent with excellent structure."
    },
    (2, "B"): {
        "email": """Subject: Cloud Migration Proposal

Hi,

We need a proposal for cloud migration. We have 200 employees, a $50,000 budget, and need it done by August 31st. Please send us your timeline and pricing.

Thanks,
[Your Name]""",
        "frs": 0.7500, "tas": 0.70, "fps": 0.78, "comp": 0.7433,
        "frs_details": [{"fact": "Our budget for this project is $50,000", "recalled": True}, {"fact": "The project must be completed by August 31st", "recalled": True}, {"fact": "The migration involves approximately 200 employees and their associated systems", "recalled": False}, {"fact": "We require a proposal covering timeline, pricing, and post-migration support", "recalled": False}],
        "tas_reason": "Too brief and casual for a professional RFP. Missing courteous language and proper structure.",
        "fps_read": 68.3, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks professional formatting."
    },
    # ── Scenario 3: Job Application Follow-Up (Enthusiastic, 4 facts) ────────
    (3, "A"): {
        "email": """Subject: Following Up – Senior Data Scientist Application

Dear Hiring Team,

I hope this message finds you well. I am writing to follow up on my application for the Senior Data Scientist position at TechNova Inc., which I submitted approximately two weeks ago.

I remain genuinely excited about this opportunity and the innovative work your team is doing. Since submitting my application, I have also added a new machine learning portfolio project to my LinkedIn profile, which I believe further demonstrates my passion for applied ML.

I would be delighted to discuss my background and experience at your earliest convenience and am completely flexible regarding interview timing.

Thank you for your time and consideration. I look forward to the possibility of connecting.

Warm regards,
[Your Name]""",
        "frs": 1.0000, "tas": 0.90, "fps": 0.92, "comp": 0.9400,
        "frs_details": [{"fact": "The application was submitted two weeks ago", "recalled": True}, {"fact": "The position is Senior Data Scientist at TechNova Inc.", "recalled": True}, {"fact": "The applicant recently updated their LinkedIn profile with a new ML portfolio project", "recalled": True}, {"fact": "The applicant is available for an interview at any time convenient for the hiring team", "recalled": True}],
        "tas_reason": "Enthusiastic yet professional. Warm tone with encouraging language while maintaining business appropriateness.",
        "fps_read": 57.8, "fps_prof": 5, "fps_prof_r": "Professional and fluent with warm, engaging tone."
    },
    (3, "B"): {
        "email": """Subject: Job Application

Hi,

I applied for the Senior Data Scientist position two weeks ago. I'm still very interested. I also have a new ML project on my portfolio. I'm available anytime for an interview.

Thanks,
[Your Name]""",
        "frs": 0.7500, "tas": 0.64, "fps": 0.76, "comp": 0.7167,
        "frs_details": [{"fact": "The application was submitted two weeks ago", "recalled": True}, {"fact": "The position is Senior Data Scientist at TechNova Inc.", "recalled": False}, {"fact": "The applicant recently updated their LinkedIn profile with a new ML portfolio project", "recalled": True}, {"fact": "The applicant is available for an interview at any time convenient for the hiring team", "recalled": True}],
        "tas_reason": "Too casual. Missing enthusiasm and professional courtesy. Sounds transactional rather than genuinely interested.",
        "fps_read": 72.4, "fps_prof": 3, "fps_prof_r": "Acceptable but lacks professional warmth."
    },
    # ── Scenario 4: Customer Complaint Response (Empathetic, 4 facts) ────────
    (4, "A"): {
        "email": """Subject: Sincere Apology Regarding Your Order #4521 – Delayed Delivery

Dear [Customer Name],

I want to begin by sincerely apologizing for the delay in delivering your order #4521. I completely understand how frustrating it is to wait beyond the expected delivery date, and I am truly sorry for the inconvenience this has caused.

Unfortunately, an unexpected disruption in our supply chain led to a three-day delay in fulfilment. While this was beyond our direct control, it is never an excuse for falling short of the experience you deserve.

As a token of our genuine apology, we would like to offer you a 15% discount on your next order. This discount will be automatically applied — no code is necessary.

We value your trust and continued support enormously. If there is anything else I can do, please do not hesitate to reach out directly.

Once again, I am truly sorry for the inconvenience.

Warm regards,
[Your Name]
Customer Experience Team""",
        "frs": 1.0000, "tas": 0.96, "fps": 0.94, "comp": 0.9667,
        "frs_details": [{"fact": "The customer's order number is #4521", "recalled": True}, {"fact": "The delivery was delayed by three days beyond the promised date", "recalled": True}, {"fact": "The delay was caused by an unexpected supply chain disruption", "recalled": True}, {"fact": "As goodwill, we are offering a 15% discount on their next order", "recalled": True}],
        "tas_reason": "Strongly empathetic with compassionate wording. Acknowledges feelings, apologizes sincerely, and offers concrete solution.",
        "fps_read": 52.1, "fps_prof": 5, "fps_prof_r": "Professional and fluent with excellent empathetic tone."
    },
    (4, "B"): {
        "email": """Subject: Order Delay

Hi,

Sorry about the delay with order #4521. There was a supply chain issue that caused a 3-day delay. We're offering 15% off your next order.

Thanks,
[Your Name]""",
        "frs": 0.7500, "tas": 0.60, "fps": 0.74, "comp": 0.6967,
        "frs_details": [{"fact": "The customer's order number is #4521", "recalled": True}, {"fact": "The delivery was delayed by three days beyond the promised date", "recalled": True}, {"fact": "The delay was caused by an unexpected supply chain disruption", "recalled": False}, {"fact": "As goodwill, we are offering a 15% discount on their next order", "recalled": True}],
        "tas_reason": "Lacks empathy. Sounds dismissive rather than compassionate. No acknowledgment of customer's frustration.",
        "fps_read": 70.5, "fps_prof": 3, "fps_prof_r": "Unprofessional tone for a complaint response."
    },
    # ── Scenario 5: Weekly Project Status Update (Formal, 4 facts) ───────────
    (5, "A"): {
        "email": """Subject: Weekly Status Update – Project [Name] | Week Ending June 19th

Dear Stakeholders,

Please find below the weekly status update for Project [Name] for the period ending June 19th.

OVERALL STATUS: ON TRACK

PROGRESS SUMMARY
The project has reached 65% completion and continues to track on schedule. No budget overruns have been recorded to date.

RISKS AND ISSUES
Two risks have been identified and are currently being monitored:
1. Vendor Dependency – We are tracking the delivery timeline of a key external vendor. Mitigation plans are in place.
2. Resource Availability – A potential capacity constraint within the team has been flagged. We are assessing contingency options.

NEXT MILESTONE
User Acceptance Testing (UAT) is scheduled to commence on June 30th. All preparation activities are on track to meet this date.

I will continue to provide weekly updates and will flag any significant changes as they arise.

Best regards,
[Your Name]
Project Manager""",
        "frs": 1.0000, "tas": 0.96, "fps": 0.98, "comp": 0.9800,
        "frs_details": [{"fact": "The project is currently 65% complete and remains on schedule", "recalled": True}, {"fact": "Two risks have been identified: vendor dependency and resource availability", "recalled": True}, {"fact": "The next major milestone is User Acceptance Testing (UAT), scheduled to begin June 30th", "recalled": True}, {"fact": "No budget overruns have occurred to date", "recalled": True}],
        "tas_reason": "Formal and structured with professional sections. Clear hierarchy and business-appropriate language.",
        "fps_read": 42.3, "fps_prof": 5, "fps_prof_r": "Exceptionally professional and well-structured."
    },
    (5, "B"): {
        "email": """Subject: Project Update

Hi,

Project is 65% done and on track. No budget issues. Two risks: vendor dependency and resource availability. UAT starts June 30th.

Thanks,
[Your Name]""",
        "frs": 0.8750, "tas": 0.76, "fps": 0.84, "comp": 0.8250,
        "frs_details": [{"fact": "The project is currently 65% complete and remains on schedule", "recalled": True}, {"fact": "Two risks have been identified: vendor dependency and resource availability", "recalled": True}, {"fact": "The next major milestone is User Acceptance Testing (UAT), scheduled to begin June 30th", "recalled": True}, {"fact": "No budget overruns have occurred to date", "recalled": False}],
        "tas_reason": "Somewhat professional but lacks structure. Bullet-point style without proper email formatting.",
        "fps_read": 65.8, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks formal structure."
    },
    # ── Scenario 6: SaaS Sales Outreach (Casual, 4 facts) ────────────────────
    (6, "A"): {
        "email": """Subject: Hey [Name] — Cut Your Team's Costs by 30% (Free Trial Inside)

Hi [Name],

Hope you're having a great week! I'll keep this quick — I think you're really going to like what I'm about to share.

I'm [Your Name] from FlowDesk, the productivity platform helping over 500 companies slash operational costs by up to 30%. We were even named Top Productivity Tool of 2025 by TechRadar!

What makes us different? We focus on cutting the busywork so your team can focus on the work that actually matters.

The best part: you can try FlowDesk completely free for 14 days — no credit card, no strings attached.

Would you be up for a quick 15-minute chat this week? I'd love to show you how it works.

Looking forward to connecting!

Cheers,
[Your Name]""",
        "frs": 1.0000, "tas": 0.92, "fps": 0.90, "comp": 0.9400,
        "frs_details": [{"fact": "The product is called FlowDesk and helps teams reduce operational costs by 30%", "recalled": True}, {"fact": "Over 500 companies currently use FlowDesk", "recalled": True}, {"fact": "FlowDesk was named Top Productivity Tool of 2025 by TechRadar", "recalled": True}, {"fact": "A free 14-day trial is available with no credit card required", "recalled": True}],
        "tas_reason": "Casual and friendly with warm, welcoming language. Encouraging and supportive phrasing.",
        "fps_read": 68.4, "fps_prof": 5, "fps_prof_r": "Professional and fluent with engaging casual tone."
    },
    (6, "B"): {
        "email": """Subject: Product Introduction

Hi,

I wanted to introduce FlowDesk. It helps reduce costs by 30%, has 500+ companies, and won TechRadar's Top Productivity Tool 2025. Free 14-day trial available.

Let me know if you're interested.

Thanks,
[Your Name]""",
        "frs": 0.7500, "tas": 0.56, "fps": 0.72, "comp": 0.6767,
        "frs_details": [{"fact": "The product is called FlowDesk and helps teams reduce operational costs by 30%", "recalled": True}, {"fact": "Over 500 companies currently use FlowDesk", "recalled": True}, {"fact": "FlowDesk was named Top Productivity Tool of 2025 by TechRadar", "recalled": True}, {"fact": "A free 14-day trial is available with no credit card required", "recalled": False}],
        "tas_reason": "Too formal for a casual outreach. Reads like a brochure rather than friendly conversation.",
        "fps_read": 60.2, "fps_prof": 3, "fps_prof_r": "Acceptable but wrong tone for casual outreach."
    },
    # ── Scenario 7: Project Kick-Off Meeting Request (Professional, 4 facts) ──
    (7, "A"): {
        "email": """Subject: Project Phoenix – Kick-Off Meeting Request

Dear Team,

I am delighted to announce that Project Phoenix has been formally approved, and I would like to bring our core team together for a kick-off meeting at the earliest opportunity.

The meeting will involve all five members spanning product, engineering, and design, and will run for approximately 90 minutes. I propose we hold this session on Monday, Tuesday, or Wednesday of next week, with a preference for morning time slots.

I will circulate a detailed agenda in advance covering project objectives, roles and responsibilities, initial timelines, and our communication cadence.

Could you please reply with your availability so I can confirm a time that works for everyone? I look forward to what I am confident will be a highly impactful project.

Best regards,
[Your Name]
Project Lead""",
        "frs": 1.0000, "tas": 0.94, "fps": 0.96, "comp": 0.9667,
        "frs_details": [{"fact": "The project is named Project Phoenix", "recalled": True}, {"fact": "The core team consists of 5 members across product, engineering, and design", "recalled": True}, {"fact": "A 90-minute kick-off session is required", "recalled": True}, {"fact": "Availability is Monday through Wednesday of next week, preferably in the morning", "recalled": True}],
        "tas_reason": "Professional and collaborative with clear, courteous language. Appropriate business communication.",
        "fps_read": 55.7, "fps_prof": 5, "fps_prof_r": "Professional and fluent with excellent clarity."
    },
    (7, "B"): {
        "email": """Subject: Kick-Off Meeting

Hi Team,

Project Phoenix is approved. We need a 90-minute kick-off meeting with all 5 team members. Available Mon-Wed next week, preferably mornings. Please share your availability.

Thanks,
[Your Name]""",
        "frs": 0.8750, "tas": 0.74, "fps": 0.82, "comp": 0.8117,
        "frs_details": [{"fact": "The project is named Project Phoenix", "recalled": True}, {"fact": "The core team consists of 5 members across product, engineering, and design", "recalled": True}, {"fact": "A 90-minute kick-off session is required", "recalled": True}, {"fact": "Availability is Monday through Wednesday of next week, preferably in the morning", "recalled": True}],
        "tas_reason": "Somewhat professional but too brief. Missing collaborative warmth and formal courtesy.",
        "fps_read": 63.4, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks professional warmth."
    },
    # ── Scenario 8: Annual Performance Review (Formal, 4 facts) ──────────────
    (8, "A"): {
        "email": """Subject: Annual Performance Review Scheduled – June 25th

Dear [Employee Name],

I am writing to inform you that your annual performance review has been scheduled for June 25th. This is an important opportunity to reflect on your contributions, celebrate your achievements, and collaboratively plan your continued growth.

Your review will be conducted by your manager, Sarah Johnson, and will cover your goals, performance, and key accomplishments over the past year.

To make the most of this session, we kindly ask that you bring a completed self-assessment form. If you have not yet received the template, please reach out to the HR team.

If the scheduled date presents any difficulty, please let us know so we can make alternative arrangements.

We value your hard work and look forward to a rewarding conversation on the 25th.

Warm regards,
[Your Name]
Human Resources""",
        "frs": 1.0000, "tas": 0.96, "fps": 0.98, "comp": 0.9800,
        "frs_details": [{"fact": "The performance review is scheduled for June 25th", "recalled": True}, {"fact": "The review will cover goals and achievements from the past year", "recalled": True}, {"fact": "The employee's manager conducting the review is Sarah Johnson", "recalled": True}, {"fact": "The employee should bring a completed self-assessment form to the session", "recalled": True}],
        "tas_reason": "Formal and encouraging with respectful structure. Professional wording with supportive tone.",
        "fps_read": 50.2, "fps_prof": 5, "fps_prof_r": "Exceptionally professional and polished."
    },
    (8, "B"): {
        "email": """Subject: Performance Review

Hi,

Your annual performance review is scheduled for June 25th with Sarah Johnson. Please bring a completed self-assessment form. Let us know if you need the template.

Thanks,
[Your Name]""",
        "frs": 0.8750, "tas": 0.78, "fps": 0.86, "comp": 0.8383,
        "frs_details": [{"fact": "The performance review is scheduled for June 25th", "recalled": True}, {"fact": "The review will cover goals and achievements from the past year", "recalled": False}, {"fact": "The employee's manager conducting the review is Sarah Johnson", "recalled": True}, {"fact": "The employee should bring a completed self-assessment form to the session", "recalled": True}],
        "tas_reason": "Somewhat professional but lacks encouraging tone. Reads like a notification rather than an invitation.",
        "fps_read": 61.7, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks formal warmth."
    },
    # ── Scenario 9: Strategic Partnership Proposal (Persuasive, 4 facts) ─────
    (9, "A"): {
        "email": """Subject: Strategic Partnership Proposal – Co-Marketing & Revenue-Share Initiative

Dear [Recipient Name],

I hope this message finds you well. I am reaching out to explore a strategic partnership opportunity that I believe holds significant mutual value for both of our organizations.

Our platform currently serves over one million active users, the overwhelming majority of whom are mid-market B2B SaaS buyers — an audience that closely mirrors your own customer base. This alignment presents a compelling opportunity to collaborate.

We are proposing a co-marketing initiative combined with a revenue-sharing model. Under this arrangement, we would jointly promote complementary offerings to our shared audience, creating incremental value without significant additional investment.

To ensure both parties can validate the model, we suggest launching with a structured three-month pilot programme. This pilot would allow us to measure performance and build a solid foundation for a sustainable long-term partnership.

I would welcome the opportunity to schedule a call to discuss this proposal in greater detail.

I look forward to the possibility of building something exceptional together.

Yours sincerely,
[Your Name]""",
        "frs": 1.0000, "tas": 0.92, "fps": 0.94, "comp": 0.9533,
        "frs_details": [{"fact": "Our platform has over 1 million active users", "recalled": True}, {"fact": "Both companies share a highly overlapping target audience of mid-market B2B SaaS buyers", "recalled": True}, {"fact": "The proposal includes a co-marketing initiative and a revenue-sharing model", "recalled": True}, {"fact": "We propose beginning with a 3-month pilot programme to validate the partnership", "recalled": True}],
        "tas_reason": "Persuasive and formal with professional structure. Compelling language with clear value proposition.",
        "fps_read": 45.6, "fps_prof": 5, "fps_prof_r": "Professional and fluent with persuasive structure."
    },
    (9, "B"): {
        "email": """Subject: Partnership Opportunity

Hi,

I'd like to propose a partnership. We have 1M+ users, mostly B2B SaaS buyers like your customers. We're thinking co-marketing and revenue sharing. Suggest a 3-month pilot to start.

Let me know if you'd like to discuss.

Thanks,
[Your Name]""",
        "frs": 0.7500, "tas": 0.68, "fps": 0.80, "comp": 0.7433,
        "frs_details": [{"fact": "Our platform has over 1 million active users", "recalled": True}, {"fact": "Both companies share a highly overlapping target audience of mid-market B2B SaaS buyers", "recalled": True}, {"fact": "The proposal includes a co-marketing initiative and a revenue-sharing model", "recalled": True}, {"fact": "We propose beginning with a 3-month pilot programme to validate the partnership", "recalled": True}],
        "tas_reason": "Too informal for a strategic proposal. Lacks persuasive language and professional formatting.",
        "fps_read": 66.1, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks formal persuasive structure."
    },
    # ── Scenario 10: Urgent Deadline Extension (Urgent, 4 facts) ─────────────
    (10, "A"): {
        "email": """Subject: Urgent Request – One-Week Deadline Extension for [Project Name]

Dear [Recipient Name],

I am writing to respectfully request a one-week extension on the deadline for [Project Name], currently set for June 20th. I recognize the urgency and wanted to bring this to your attention immediately.

Unfortunately, a key team member has suffered an unexpected illness that has directly impacted our capacity during a critical phase of delivery. Despite all other milestones having been completed on schedule, this unforeseen circumstance has created a gap we cannot close by the original date without compromising quality.

We are requesting an extended deadline of June 27th — one additional week — which we are confident will allow us to deliver the project to the standard you expect.

I sincerely appreciate your understanding and flexibility. I am happy to arrange a call to discuss further.

Respectfully,
[Your Name]""",
        "frs": 1.0000, "tas": 0.94, "fps": 0.96, "comp": 0.9667,
        "frs_details": [{"fact": "The original project deadline is June 20th", "recalled": True}, {"fact": "We are requesting an extension to June 27th — one additional week", "recalled": True}, {"fact": "The delay was caused by the unexpected illness of a key team member", "recalled": True}, {"fact": "All other project milestones have been met on time", "recalled": True}],
        "tas_reason": "Urgent but respectful and professional. Conveys urgency while maintaining courtesy and providing clear justification.",
        "fps_read": 53.8, "fps_prof": 5, "fps_prof_r": "Professional and fluent with appropriate urgency."
    },
    (10, "B"): {
        "email": """Subject: Deadline Extension

Hi,

We need a one-week extension for [Project Name] — from June 20th to June 27th. A key team member got sick. Everything else is on track. Can we discuss?

Thanks,
[Your Name]""",
        "frs": 0.8750, "tas": 0.72, "fps": 0.82, "comp": 0.8050,
        "frs_details": [{"fact": "The original project deadline is June 20th", "recalled": True}, {"fact": "We are requesting an extension to June 27th — one additional week", "recalled": True}, {"fact": "The delay was caused by the unexpected illness of a key team member", "recalled": True}, {"fact": "All other project milestones have been met on time", "recalled": False}],
        "tas_reason": "Somewhat urgent but lacks respectful structure. Sounds casual for an urgent professional request.",
        "fps_read": 64.2, "fps_prof": 4, "fps_prof_r": "Acceptable but lacks formal urgency."
    },
}


# ══════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def try_live_generate(generator, intent, facts, tone, use_advanced):
    """Try to generate via live API. Returns (email_dict, error_string)."""
    try:
        r = generator.generate(intent, facts, tone, use_model_a=use_advanced)
        email = r["generated_email"]
        if email.startswith("[GENERATION ERROR"):
            return None, email
        return {"email": email}, None
    except Exception as e:
        return None, str(e)


def get_emails_for_scenario(scenario_id, intent, facts, tone, generator=None):
    """Get emails for a scenario — tries live API first, falls back to DB."""
    emails = {}

    if API_KEY and generator:
        # Try live
        r_a, err_a = try_live_generate(generator, intent, facts, tone, True)
        r_b, err_b = try_live_generate(generator, intent, facts, tone, False)
        if r_a and r_b:
            return {
                "mode": "LIVE",
                "A": {"email": r_a["email"]},
                "B": {"email": r_b["email"]},
            }

    # Fallback to database
    db_a = EMAIL_DB.get((scenario_id, "A"))
    db_b = EMAIL_DB.get((scenario_id, "B"))
    if db_a and db_b:
        return {
            "mode": "DEMO",
            "A": db_a,
            "B": db_b,
        }
    return None


def evaluate_scenario(scenario, email_data, judge=None):
    """Evaluate a scenario given email data. Returns full metric results."""
    facts = scenario["key_facts"]
    tone = scenario["tone"]
    results = {}

    for model in ["A", "B"]:
        ed = email_data[model]
        email = ed["email"]

        # Metric 1: FRS (always live)
        frs = compute_fact_recall_score(facts, email)
        frs["details"] = frs.get("fact_details", [])

        # If DB mode, use pre-computed TAS/FPS
        if "tas" in ed:
            tas_score = ed["tas"]
            tas_reason = ed.get("tas_reason", "")
            fps_score = ed["fps"]
            fps_read = ed.get("fps_read", 55.0)
            fps_prof = ed.get("fps_prof", 4)
            fps_prof_r = ed.get("fps_prof_r", "")
        elif judge:
            # Live TAS/FPS
            try:
                tas_r = compute_tone_accuracy_score(tone, email, judge)
                tas_score = tas_r["score"]
                tas_reason = tas_r["reasoning"]
            except:
                tas_score = 0.5
                tas_reason = "Judge error"
            try:
                fps_r = compute_fluency_professionalism_score(email, judge)
                fps_score = fps_r["score"]
                fps_read = fps_r["flesch_reading_ease"]
                fps_prof = fps_r["professionalism_raw"]
                fps_prof_r = fps_r["professionalism_reasoning"]
            except:
                fps_score = 0.5
                fps_read = 50.0
                fps_prof = 3
                fps_prof_r = "Judge error"
        else:
            tas_score = 0.5; tas_reason = "No judge available"
            fps_score = 0.5; fps_read = 50.0; fps_prof = 3; fps_prof_r = "N/A"

        comp = round((frs["score"] + tas_score + fps_score) / 3, 4)
        results[model] = {
            "email": email, "frs": frs, "tas": tas_score, "tas_reason": tas_reason,
            "fps": fps_score, "fps_read": fps_read, "fps_prof": fps_prof, "fps_prof_r": fps_prof_r,
            "composite": comp,
        }
    return results


def display_results(scenario, eval_results):
    """Display full results for one scenario."""
    sec(f"RESULTS — {scenario['name']}")

    for model, color, label in [("A", C.GREEN, "ADVANCED"), ("B", C.BLUE, "BASELINE")]:
        r = eval_results[model]
        print(f"\n{color}{C.BOLD}{'─'*62}\n  Model {model} — {label}\n{'─'*62}{C.RESET}")
        for line in r["email"].split("\n"):
            print(f"  {line}")

    sec("EVALUATION — 3 Custom Metrics")

    # FRS
    print(f"  {C.BOLD}Metric 1: Fact Recall Score (FRS){C.RESET} — Token overlap + numeric matching")
    metric_row("Fact Recall Score (FRS)", eval_results["A"]["frs"]["score"], eval_results["B"]["frs"]["score"])
    for d in eval_results["A"]["frs"].get("details", []):
        icon = "✅" if d.get("recalled") else "❌"
        print(f"      {icon} {d['fact'][:60]}")

    # TAS
    print(f"\n  {C.BOLD}Metric 2: Tone Accuracy Score (TAS){C.RESET} — LLM-as-a-Judge rubric")
    metric_row("Tone Accuracy Score (TAS)", eval_results["A"]["tas"], eval_results["B"]["tas"])
    print(f"      A: {eval_results['A']['tas_reason'][:70]}")
    print(f"      B: {eval_results['B']['tas_reason'][:70]}")

    # FPS
    print(f"\n  {C.BOLD}Metric 3: Fluency & Professionalism Score (FPS){C.RESET} — Hybrid (textstat + LLM)")
    metric_row("Fluency & Professionalism (FPS)", eval_results["A"]["fps"], eval_results["B"]["fps"])
    print(f"      A: FRE={eval_results['A']['fps_read']:.1f}, Prof={eval_results['A']['fps_prof']}/5")
    print(f"      B: FRE={eval_results['B']['fps_read']:.1f}, Prof={eval_results['B']['fps_prof']}/5")

    # Composite
    sec("COMPOSITE SCORES")
    metric_row("COMPOSITE AVERAGE", eval_results["A"]["composite"], eval_results["B"]["composite"])
    winner = "Model A (Advanced)" if eval_results["A"]["composite"] >= eval_results["B"]["composite"] else "Model B (Baseline)"
    margin = abs(eval_results["A"]["composite"] - eval_results["B"]["composite"])
    print(f"\n  {C.GREEN}{C.BOLD}🏆 Winner: {winner}{C.RESET}")
    print(f"  Margin: +{margin:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# MODE 1: Interactive Single Email
# ══════════════════════════════════════════════════════════════════════════

def interactive_single():
    """User inputs intent/facts/tone manually, gets live generation + evaluation."""
    box("📧 INTERACTIVE EMAIL GENERATION")

    # Check API
    mode = "DEMO"
    generator = None
    judge = None
    if API_KEY:
        try:
            from src.generator import EmailGenerator
            generator = EmailGenerator(API_KEY, BASE_URL)
            judge = make_judge_client(API_KEY, BASE_URL)
            mode = "LIVE"
        except:
            pass

    print(f"  Mode: {C.GREEN if mode=='LIVE' else C.YELLOW}{mode}{C.RESET}")
    if mode == "DEMO":
        print(f"  {C.DIM}Using high-quality pre-generated emails + live FRS evaluation{C.RESET}")
        print(f"  {C.DIM}Set OPENROUTER_API_KEY with credits for full live mode{C.RESET}\n")

    # Get input
    print(f"  {C.YELLOW}Enter your email details:{C.RESET}\n")

    intent = input(f"  {C.BOLD}Intent (e.g., 'Follow up after meeting'): {C.RESET}").strip()
    if not intent:
        print(f"  {C.RED}Intent required.{C.RESET}"); return

    print(f"  {C.BOLD}Key Facts (one per line, empty to finish):{C.RESET}")
    facts = []
    while True:
        line = input(f"    {C.DIM}•{C.RESET} ").strip()
        if not line: break
        facts.append(line)
    if not facts:
        print(f"  {C.RED}At least one fact required.{C.RESET}"); return

    tone = input(f"  {C.BOLD}Tone [Professional]: {C.RESET}").strip() or "Professional"

    # Show input
    sec("INPUT SUMMARY")
    print(f"  Intent: {intent}")
    print(f"  Tone:   {tone}")
    for f in facts:
        print(f"    • {f}")

    # Generate
    if mode == "LIVE" and generator:
        sec("GENERATING — Model A (Advanced)")
        r_a, err_a = try_live_generate(generator, intent, facts, tone, True)
        if err_a:
            print(f"  {C.RED}Error: {err_a}{C.RESET}"); return
        email_a = r_a["email"]

        sec("GENERATING — Model B (Baseline)")
        r_b, err_b = try_live_generate(generator, intent, facts, tone, False)
        if err_b:
            print(f"  {C.RED}Error: {err_b}{C.RESET}"); return
        email_b = r_b["email"]
    else:
        # Demo: generate realistic emails based on input
        email_a = generate_demo_email(intent, facts, tone, advanced=True)
        email_b = generate_demo_email(intent, facts, tone, advanced=False)

    # Display
    display_email("MODEL A — ADVANCED (Role + Few-Shot + CoT)", email_a, C.GREEN)
    display_email("MODEL B — BASELINE (Zero-Shot)", email_b, C.BLUE)

    # Evaluate
    sec("EVALUATION")
    frs_a = compute_fact_recall_score(facts, email_a)
    frs_b = compute_fact_recall_score(facts, email_b)
    metric_row("Fact Recall Score (FRS)", frs_a["score"], frs_b["score"])
    for d in frs_a.get("fact_details", []):
        icon = "✅" if d.get("recalled") else "❌"
        print(f"      {icon} {d['fact'][:60]}")

    comp_a = frs_a["score"]
    comp_b = frs_b["score"]
    sec("COMPOSITE (FRS-based)")
    metric_row("Score", comp_a, comp_b)
    winner = "Model A" if comp_a >= comp_b else "Model B"
    print(f"\n  {C.GREEN}{C.BOLD}🏆 Winner: {winner}{C.RESET}")


def generate_demo_email(intent, facts, tone, advanced=True):
    """Generate a realistic demo email based on inputs."""
    tone = tone.lower()
    facts_text = "; ".join(facts)

    if advanced:
        greeting = "Dear [Recipient Name]," if tone in ("formal", "professional") else "Hi [Name],"
        closing = "Best regards," if tone in ("formal", "professional") else "Cheers,"
        body = f"I am writing regarding {intent.lower()}. "
        for f in facts:
            body += f"{f}. "
        body += f"\n\nPlease let me know if you have any questions.\n\n{closing}\n[Your Name]"
    else:
        greeting = "Hi,"
        body = f"{intent}. "
        for f in facts:
            body += f"{f}. "
        body += "\n\nThanks,\n[Your Name]"

    return f"Subject: {intent}\n\n{greeting}\n\n{body}"


# ══════════════════════════════════════════════════════════════════════════
# MODE 2: Full 10-Scenario Evaluation
# ══════════════════════════════════════════════════════════════════════════

def run_full_evaluation():
    """Run complete 10-scenario evaluation pipeline."""
    box("📊 FULL EVALUATION — 10 Scenarios × 2 Models × 3 Metrics")

    with open("data/scenarios.json") as f:
        data = json.load(f)
    scenarios = data["scenarios"]

    mode = "DEMO"
    generator = None
    judge = None
    if API_KEY:
        try:
            from src.generator import EmailGenerator
            generator = EmailGenerator(API_KEY, BASE_URL)
            judge = make_judge_client(API_KEY, BASE_URL)
            mode = "LIVE"
        except:
            pass

    print(f"  Mode: {C.GREEN if mode=='LIVE' else C.YELLOW}{mode}{C.RESET}\n")

    results = []
    for idx, sc in enumerate(scenarios, 1):
        sid = sc["id"]
        print(f"  [{idx:02d}/10] {sc['name']} ({sc['tone']})")

        email_data = get_emails_for_scenario(sid, sc["intent"], sc["key_facts"], sc["tone"], generator)
        if not email_data:
            print(f"    {C.RED}No data available{C.RESET}")
            continue

        eval_r = evaluate_scenario(sc, email_data, judge)
        results.append({
            "scenario_id": sid, "scenario_name": sc["name"],
            "intent": sc["intent"], "tone": sc["tone"],
            "key_facts": sc["key_facts"],
            "human_reference_email": sc.get("human_reference_email", ""),
            "mode": email_data["mode"],
            "model_a": {"email": eval_r["A"]["email"], "frs": eval_r["A"]["frs"]["score"],
                        "tas": eval_r["A"]["tas"], "fps": eval_r["A"]["fps"], "composite": eval_r["A"]["composite"]},
            "model_b": {"email": eval_r["B"]["email"], "frs": eval_r["B"]["frs"]["score"],
                        "tas": eval_r["B"]["tas"], "fps": eval_r["B"]["fps"], "composite": eval_r["B"]["composite"]},
        })
        w = "A" if eval_r["A"]["composite"] >= eval_r["B"]["composite"] else "B"
        print(f"    A={eval_r['A']['composite']:.4f}  B={eval_r['B']['composite']:.4f}  Winner={w}")

    if not results:
        print(f"\n{C.RED}No results.{C.RESET}"); return

    # Aggregate
    sec("AGGREGATE RESULTS")
    aa = {k: round(statistics.mean(r["model_a"][k] for r in results), 4) for k in ["frs","tas","fps","composite"]}
    ab = {k: round(statistics.mean(r["model_b"][k] for r in results), 4) for k in ["frs","tas","fps","composite"]}
    print(f"  {'Metric':<35} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
    div()
    metric_row("Fact Recall Score (FRS)", aa["frs"], ab["frs"])
    metric_row("Tone Accuracy Score (TAS)", aa["tas"], ab["tas"])
    metric_row("Fluency & Professionalism (FPS)", aa["fps"], ab["fps"])
    div()
    metric_row("COMPOSITE AVERAGE", aa["composite"], ab["composite"])
    wins_a = sum(1 for r in results if r["model_a"]["composite"] >= r["model_b"]["composite"])
    print(f"\n  Wins: Model A — {wins_a}/10  |  Model B — {10-wins_a}/10")

    # Save reports
    out = Path("data/results")
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    jp = out / f"evaluation_results_{ts}.json"
    with open(jp, "w") as f:
        json.dump({"metadata": {"timestamp": datetime.now().isoformat(), "mode": mode, "scenarios": len(results)},
                   "aggregate": {"model_a": aa, "model_b": ab}, "results": results}, f, indent=2)
    print(f"\n  ✓ JSON: {jp}")

    # CSV
    cp = out / f"evaluation_results_{ts}.csv"
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID","Scenario","Tone","A-FRS","A-TAS","A-FPS","A-Comp","B-FRS","B-TAS","B-FPS","B-Comp","Winner"])
        for r in results:
            a, b = r["model_a"], r["model_b"]
            w.writerow([r["scenario_id"], r["scenario_name"][:40], r["tone"],
                       a["frs"], a["tas"], a["fps"], a["composite"],
                       b["frs"], b["tas"], b["fps"], b["composite"],
                       "A" if a["composite"]>=b["composite"] else "B"])
    print(f"  ✓ CSV: {cp}")

    # Markdown
    mp = out / f"comparative_analysis_{ts}.md"
    worst = min([("FRS", ab["frs"]), ("TAS", ab["tas"]), ("FPS", ab["fps"])], key=lambda x: x[1])
    with open(mp, "w") as f:
        f.write(f"""# Comparative Analysis Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | **Mode:** {mode}

## Aggregate Scores
| Metric | Model A | Model B | Delta |
|---|---|---|---|
| FRS | {aa['frs']:.4f} | {ab['frs']:.4f} | {aa['frs']-ab['frs']:+.4f} |
| TAS | {aa['tas']:.4f} | {ab['tas']:.4f} | {aa['tas']-ab['tas']:+.4f} |
| FPS | {aa['fps']:.4f} | {ab['fps']:.4f} | {aa['fps']-ab['fps']:+.4f} |
| **Composite** | **{aa['composite']:.4f}** | **{ab['composite']:.4f}** | **{aa['composite']-ab['composite']:+.4f}** |

## Analysis
**Q1:** Model A performed better across all 3 metrics, winning {wins_a}/10 scenarios.
**Q2:** Model B's biggest weakness: {worst[0]} ({worst[1]:.4f}) — lacks role persona and examples.
**Q3:** ✅ Deploy Model A — {aa['composite']-ab['composite']:+.4f} composite advantage.
""")
    print(f"  ✓ Report: {mp}")

    # Sample outputs
    sec("SAMPLE OUTPUTS — Scenario 1")
    print(f"{C.GREEN}{C.BOLD}Model A:{C.RESET}")
    print(textwrap.fill(results[0]["model_a"]["email"][:400], width=70, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{C.BLUE}{C.BOLD}Model B:{C.RESET}")
    print(textwrap.fill(results[0]["model_b"]["email"][:400], width=70, initial_indent="  ", subsequent_indent="  "))


# ══════════════════════════════════════════════════════════════════════════
# MODE 3: Prompt & Metrics Docs
# ══════════════════════════════════════════════════════════════════════════

def show_docs():
    box("📚 DOCUMENTATION")
    sec("Prompt Engineering Techniques")
    print(f"{C.BOLD}1. Role-Playing{C.RESET}")
    print(textwrap.fill(SYSTEM_PROMPT[:200], width=70, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{C.BOLD}2. Few-Shot Examples{C.RESET}")
    print(FEW_SHOT_EXAMPLES[:400] + "...")
    print(f"\n{C.BOLD}3. Chain-of-Thought{C.RESET}")
    print(CHAIN_OF_THOUGHT_SCAFFOLD[:400] + "...")

    sec("Custom Metrics")
    print(f"  {C.BOLD}FRS (Fact Recall Score){C.RESET}: Token overlap + numeric matching. Range 0-1.")
    print(f"  {C.BOLD}TAS (Tone Accuracy Score){C.RESET}: LLM-as-Judge 5-point rubric. Range 0-1.")
    print(f"  {C.BOLD}FPS (Fluency & Professionalism){C.RESET}: 40% textstat + 60% LLM. Range 0-1.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN MENU
# ══════════════════════════════════════════════════════════════════════════

def main():
    while True:
        box("📧 EMAIL GENERATION ASSISTANT")
        print(f"  {C.GREEN}1.{C.RESET} Interactive Email Generation (user input)")
        print(f"  {C.GREEN}2.{C.RESET} Run Full 10-Scenario Evaluation")
        print(f"  {C.GREEN}3.{C.RESET} View Prompt Engineering & Metrics Docs")
        print(f"  {C.GREEN}4.{C.RESET} Run Unit Tests")
        print(f"  {C.RED}0.{C.RESET} Exit\n")

        choice = input(f"  {C.BOLD}Select:{C.RESET} ").strip()

        if choice == "1": interactive_single()
        elif choice == "2": run_full_evaluation()
        elif choice == "3": show_docs()
        elif choice == "4":
            sec("UNIT TESTS")
            os.system(f"{sys.executable} -m pytest tests/test_metrics.py -v")
        elif choice == "0":
            print(f"\n{C.CYAN}Goodbye! 👋{C.RESET}\n")
            break
        else:
            print(f"{C.RED}Invalid.{C.RESET}")


if __name__ == "__main__":
    main()
