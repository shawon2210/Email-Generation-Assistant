"""
Advanced Prompt Engineering Module
=====================================

This module implements three advanced prompting techniques combined to
maximize email generation quality and reliability:

1. ROLE-PLAYING
   The LLM is assigned the persona of a "world-class professional business
   writer with 20+ years of Fortune-500 experience."  This persona grounds
   the model's vocabulary, register, and judgment toward high-quality
   business writing rather than generic text generation.

2. FEW-SHOT EXAMPLES
   Two carefully crafted in-context examples—one formal, one casual—are
   prepended to every request.  They demonstrate the expected output format
   (Subject line + full email) and show how to weave facts naturally into
   prose without sounding like a checklist.

3. CHAIN-OF-THOUGHT (CoT)
   An explicit five-step reasoning scaffold ("Analyze → Plan facts →
   Calibrate tone → Structure → Write") guides the model through the
   generation process.  The instruction explicitly tells the model NOT to
   include its reasoning in the output, so only the polished email is
   returned.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE 1 – ROLE-PLAYING: System-level persona
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a world-class professional business writer with 20+ years of "
    "experience crafting impactful corporate communications for Fortune 500 "
    "companies, global startups, and C-suite executives. You specialize in "
    "writing emails that are clear, concise, persuasive, and perfectly "
    "calibrated to the intended audience and requested tone. "
    "Every email you produce: "
    "(a) opens with an engaging, purposeful first sentence; "
    "(b) seamlessly integrates every required fact into natural prose; "
    "(c) precisely matches the requested tone from first word to sign-off; "
    "(d) uses standard professional email structure: Subject → Greeting → "
    "Body → Call-to-Action → Closing → Signature placeholder; "
    "(e) is free of grammatical errors and filler phrases."
)

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE 2 – FEW-SHOT EXAMPLES: In-context demonstrations
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
════════════════════════════════════════════════════════
EXAMPLE 1 — FORMAL TONE
════════════════════════════════════════════════════════
INPUT:
  Intent:    Follow up on contract renewal negotiation
  Key Facts:
    - Current contract expires on December 31st
    - We are requesting a 10% price reduction
    - We have maintained a 5-year partnership
    - We are open to a 2-year renewal term
  Tone: Formal

OUTPUT:
Subject: Contract Renewal Discussion – Agreement Expiring December 31st

Dear [Recipient Name],

I hope this message finds you well. I am writing to formally initiate our
contract renewal discussion, as our current agreement is scheduled to expire
on December 31st.

Over the course of our five-year partnership, we have built a strong and
mutually beneficial relationship, and we remain committed to continuing that
collaboration. As we enter renewal negotiations, we respectfully propose a
10% reduction in pricing, reflecting both our long-standing commitment and
prevailing market conditions.

We are open to a two-year renewal term, which we believe would provide
operational stability and long-term value for both organizations. I would
welcome the opportunity to schedule a call at your earliest convenience to
discuss the proposed terms in detail.

Thank you for your continued partnership. I look forward to hearing from you.

Sincerely,
[Your Name]
[Your Title] | [Your Company]
[Phone] | [Email]

════════════════════════════════════════════════════════
EXAMPLE 2 — CASUAL / FRIENDLY TONE
════════════════════════════════════════════════════════
INPUT:
  Intent:    Check in with a client after product onboarding
  Key Facts:
    - Onboarding was completed 2 weeks ago
    - Checking in to see if they need any help
    - A new product feature just launched
    - Inviting them to an upcoming user webinar
  Tone: Casual and friendly

OUTPUT:
Subject: How's Everything Going? We've Got Exciting Updates! 🎉

Hi [Name],

Hope you're having a great week! It's been about two weeks since you wrapped
up onboarding, and I just wanted to swing by and see how things are going.

Are you finding everything you need? If anything has been tricky or you have
questions, don't hesitate to reach out — I'm always happy to help.

Also, exciting news: we just launched a brand-new feature that I think you're
really going to love. I'll drop a link below so you can check it out at your
own pace.

Oh, and we're hosting a live user webinar next week packed with power tips
and Q&A time. It'd be great to see you there!

Let me know how it's going — looking forward to catching up!

Cheers,
[Your Name]
════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE 3 – CHAIN-OF-THOUGHT: Structured reasoning scaffold
# ─────────────────────────────────────────────────────────────────────────────
CHAIN_OF_THOUGHT_SCAFFOLD = """
Before writing the email, work through these five reasoning steps internally:

  STEP 1 — ANALYZE THE GOAL
    What is the single primary action or response this email must prompt?
    Who is the audience and what do they care about most?

  STEP 2 — FACT INTEGRATION PLAN
    Review every key fact. Decide exactly where each fact fits naturally
    (opening hook, body paragraph 1, body paragraph 2, CTA, or closing).
    No fact should appear as a raw bullet — each must flow as natural prose.

  STEP 3 — TONE CALIBRATION
    Map the requested tone to specific language choices:
    vocabulary complexity, sentence length, use of contractions, emotional
    warmth, level of directness, and appropriate greeting/closing phrases.

  STEP 4 — STRUCTURE PLAN
    Subject line → Opening sentence → Body (facts woven in) →
    Clear call-to-action → Professional closing → Signature block.

  STEP 5 — WRITE THE EMAIL
    Execute your plan. Produce a polished, complete email.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: Output ONLY the final email beginning with "Subject:".
Do NOT include your reasoning steps, labels, or any preamble.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def build_advanced_prompt(intent: str, facts: list, tone: str) -> str:
    """
    Builds the full Model A prompt combining:
      - Few-shot examples (Technique 2)
      - Chain-of-thought scaffold (Technique 3)
    The system prompt (Technique 1) is injected separately via system_instruction.

    Args:
        intent: The core purpose of the email.
        facts:  List of key fact strings to include.
        tone:   Desired writing tone/style.

    Returns:
        The complete prompt string for Model A.
    """
    facts_block = "\n".join(f"    - {f}" for f in facts)
    return (
        f"{FEW_SHOT_EXAMPLES}\n"
        f"{CHAIN_OF_THOUGHT_SCAFFOLD}\n"
        f"════════════════════════════════════════════════════════\n"
        f"YOUR TASK\n"
        f"════════════════════════════════════════════════════════\n"
        f"INPUT:\n"
        f"  Intent:    {intent}\n"
        f"  Key Facts:\n"
        f"{facts_block}\n"
        f"  Tone: {tone}\n\n"
        f"OUTPUT:"
    )


def build_simple_prompt(intent: str, facts: list, tone: str) -> str:
    """
    Zero-shot baseline prompt for Model B.
    No role, no examples, no CoT – plain instruction only.

    Args:
        intent: The core purpose of the email.
        facts:  List of key fact strings to include.
        tone:   Desired writing tone/style.

    Returns:
        The simple baseline prompt string for Model B.
    """
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        f"Write a professional email.\n\n"
        f"Intent: {intent}\n"
        f"Key Facts:\n{facts_block}\n"
        f"Tone: {tone}\n\n"
        f"Email:"
    )
