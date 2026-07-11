#!/usr/bin/env python3
"""
Patent Scout — Patentability & Prior Art Screening Agent
==========================================================

Given a plain-language description of a mechanical/engineering invention,
this agent:

  1. (technical-parser subagent) Breaks the invention down into its
     distinct novel elements — the specific mechanical/functional
     features that could be the basis of a patent claim.

  2. (prior-art-search subagent) Searches for existing patents and
     public prior art relevant to each element.

  3. (main agent) Synthesizes both into a written patentability screening
     memo: which elements look novel, which look anticipated by prior art,
     and an overall risk read — saved as a Markdown report.

This is a SCREENING tool, not legal advice. It is meant to speed up the
first pass of "is this worth taking to a patent attorney," the same way
an engineer might triage before an IP consult.

Usage:
    python agent.py "path/to/invention_description.txt"
    python agent.py --text "A ratcheting hinge that locks at 15-degree..."
    python agent.py --interview
        (an interviewer agent asks you questions in the terminal and builds
        the invention description for you before running the screen)

Requires ONE of:
    - A Claude Pro/Max subscription login: run `claude login` once (see README)
      and this script will use that session automatically. No separate API key
      or extra charge beyond your existing subscription.
    - ANTHROPIC_API_KEY set in the environment (or a .env file, see .env.example)
      for pay-as-you-go API billing instead.
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AgentDefinition,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
)

load_dotenv()

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

TECHNICAL_PARSER = AgentDefinition(
    description=(
        "Extracts the distinct novel technical/mechanical elements from an "
        "invention description. Use this first, before any prior art search, "
        "whenever a new invention needs to be broken down into claim-sized pieces."
    ),
    prompt="""You are a mechanical engineer supporting a patentability screen.

Given an invention description, break it into a short numbered list of its
DISTINCT technical elements — the specific mechanisms, structures, or
functional relationships that could plausibly anchor an independent or
dependent patent claim. Do not evaluate novelty yet; that happens later.

For each element:
- Name it in a few words (e.g. "spring-loaded ratchet pawl geometry")
- Describe, in one or two sentences, what makes it functionally specific
  (not just "a hinge" but what is distinct about THIS hinge)
- Note the general technical field it falls under (e.g. mechanisms,
  materials, fluid systems, electromechanical, controls)

Ignore generic, well-known components mentioned only for context (e.g. "a
standard bearing") unless the invention description claims something novel
about how they're used.

Output ONLY the numbered list. Be precise and concise — this list is
consumed by a downstream search step, so it needs to be specific enough to
search on, not vague marketing language.""",
    tools=["Read"],
    model="sonnet",
)

PRIOR_ART_SEARCH = AgentDefinition(
    description=(
        "Searches the web and patent databases (Google Patents, USPTO) for "
        "prior art and existing patents related to a given technical element. "
        "Use this after the technical elements of an invention have been "
        "identified, once per element or small group of related elements."
    ),
    prompt="""You are a prior art researcher supporting a patentability screen.

You will be given one or more specific technical elements from an invention.
For each element:

1. Search for existing patents (prioritize Google Patents and USPTO results)
   and any other public prior art (products, papers, standards) that relate
   to it.
2. Identify the 2-4 most relevant results. For each, note: title/patent
   number if found, publication/filing date if found, and in ONE sentence
   how it relates to the element (same mechanism, adjacent mechanism,
   same problem different solution, etc).
3. Give a brief read on how close the element sits to what you found —
   "closely anticipated," "partially anticipated," or "no close match found"
   — and why, in a sentence or two.

Be honest when search results are thin or ambiguous — say so rather than
overstating confidence. You are not making a legal determination, only
reporting what prior art you found and how close it looks.

Output your findings as a short section per element, clearly labeled.""",
    tools=["WebSearch", "WebFetch"],
    model="sonnet",
)


# ---------------------------------------------------------------------------
# Interviewer — conversational front-end that builds an invention description
# ---------------------------------------------------------------------------

SUMMARY_START = "===INVENTION_SUMMARY==="
SUMMARY_END = "===END_SUMMARY==="

INTERVIEWER_SYSTEM_PROMPT = f"""You are interviewing an inventor to gather enough detail about
their mechanical/engineering invention to run a patentability screen.

IMPORTANT: Do not assume the inventor has an engineering background. Most
inventors know what they want their invention to DO, not the mechanical
details of HOW it should work internally. Adapt to whichever kind of person
you're talking to.

Ask ONE clear, specific question at a time — never a list of questions.

Start with plain-language questions, in this order:
1. What problem does it solve, and in what situation/context is it used?
2. What should it actually DO from the user's point of view? (e.g. "it
   should tighten itself automatically" — not how, just what)
3. What's out there already that's closest to this, that the inventor
   is aware of?

Only after that, try to get into mechanism detail: how it actually works,
materials, dimensions.

Handling non-technical answers — this is critical:
- If the inventor gives a vague or "I don't know" answer to a mechanism
  question, do NOT keep pushing for detail they don't have. Instead, use
  your own engineering knowledge to propose 2-3 plausible, concrete
  mechanical approaches that could achieve the outcome they described, in
  plain language, and ask which sounds closest to what they're picturing
  (or if they'd like you to just consider all of them).
  Example: "A few ways this could work mechanically: (a) a spring that
  tightens automatically as tension drops, like a seatbelt, (b) a ratchet
  you click by hand, (c) a motor and sensor. Does one of those sound like
  what you're picturing, or should I just explore a couple of options?"
- If the inventor answers a couple of questions vaguely in a row, shift
  your remaining questions to be simpler and more about function/goals
  rather than technical mechanism specifics — meet them at their level
  instead of assuming they'll suddenly get more technical.
- Never make the inventor feel like they gave a wrong or bad answer.
  Not knowing the mechanism is completely normal and expected.

Cover, across the interview (adapting depth as above):
- The problem and context of use
- What it should do / the core mechanism (inferred by you if needed)
- What's different about it vs. existing solutions
- Materials/components/dimensions IF the inventor knows or cares to guess —
  otherwise skip this and let the technical-parser and prior-art-search
  agents work with functional descriptions instead
- Anything the inventor already suspects might not be novel

Ask as few or as many questions as you actually need — usually 4 to 8. Stop
asking as soon as you could write a specific paragraph, not before. Don't
pad the interview with redundant questions.

The inventor may type "done" at any point to end early. If that happens,
write the best summary you can with whatever you have gathered so far,
using your own engineering judgment to fill in plausible mechanism detail
where the inventor couldn't provide it — but clearly mark any such inferred
detail as an assumption, e.g. "(assumed mechanism, not confirmed by
inventor: ...)" inside the summary, so downstream steps and the inventor
both know what was actually confirmed versus inferred.

When you are ready to end the interview (either because you have enough
detail, or the inventor said done), respond with ONLY the following, and
nothing else before or after it:

{SUMMARY_START}
<a clear, technical paragraph description of the invention, written the way
an engineer would write it for a patent search — specific mechanisms,
specific differences from existing approaches, no marketing language>
{SUMMARY_END}

Until you are ready to end the interview, respond with ONLY your next
question — no preamble, no summary, no markdown formatting."""


async def run_interview() -> str:
    """Conduct an interactive Q&A with the user in the terminal and return
    the resulting invention description as plain text."""
    options = ClaudeAgentOptions(
        system_prompt=INTERVIEWER_SYSTEM_PROMPT,
        allowed_tools=[],
        cwd=str(Path(__file__).parent),
    )

    print("=== Invention Interview ===")
    print("Answer each question in your own words. Type 'done' at any point")
    print("to end early and let the interviewer work with what it has.\n")

    async def get_response_text(client: ClaudeSDKClient) -> str:
        text = ""
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text += block.text
        return text

    def extract_summary(text: str) -> str | None:
        if SUMMARY_START not in text:
            return None
        start = text.index(SUMMARY_START) + len(SUMMARY_START)
        end = text.find(SUMMARY_END, start)
        return text[start : end if end != -1 else None].strip()

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Begin the interview. Ask your first question now.")
        response_text = await get_response_text(client)

        while True:
            summary = extract_summary(response_text)
            if summary:
                print("\n--- Interview complete. Invention summary ---\n")
                print(summary)
                print()
                return summary

            print(f"\nInterviewer: {response_text.strip()}\n")
            answer = input("You: ").strip()

            if answer.lower() == "done":
                await client.query(
                    "The inventor wants to stop here. Write the best invention "
                    "summary you can with the information gathered so far, in "
                    "the required format."
                )
            else:
                await client.query(answer)

            response_text = await get_response_text(client)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

MAIN_SYSTEM_PROMPT = """You are orchestrating a patentability screening pipeline for a mechanical
invention. You have two subagents available:

- technical-parser: breaks the invention description into distinct novel
  technical elements.
- prior-art-search: searches for prior art relevant to those elements.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the prior-art-search agent on the resulting elements (you may group
   related elements into one call, or call it multiple times — use your
   judgment for what produces good search coverage).
3. Write a single Markdown patentability screening memo that synthesizes
   both steps. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each technical element: what it is, what prior art was found, and a
   novelty read (Likely Novel / Possibly Anticipated / Likely Anticipated /
   Insufficient Information).

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole invention,
   and a plain recommendation on whether this looks worth a real attorney
   consult, and if so which elements to lead with.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability or freedom-to-operate opinion, and does not
   replace a search and opinion from a licensed patent attorney or agent.

4. Save this memo using the Write tool to the exact path given to you in the
   user prompt. Preserve the subagents' findings substantively — don't
   compress away the specific prior art citations they found.
"""


async def run_pipeline(invention_text: str, output_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Invention description:
---
{invention_text}
---

Run the full pipeline described in your instructions on this invention.
Today's date is {today}. Save the final memo to exactly this path:
{output_path}
"""

    options = ClaudeAgentOptions(
        system_prompt=MAIN_SYSTEM_PROMPT,
        allowed_tools=["Read", "Write", "WebSearch", "WebFetch", "Agent"],
        agents={
            "technical-parser": TECHNICAL_PARSER,
            "prior-art-search": PRIOR_ART_SEARCH,
        },
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )

    print("Running patentability screening pipeline...\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name == "Agent":
                    subagent = block.input.get("subagent_type", "unknown")
                    print(f"  -> delegating to subagent: {subagent}")
                elif isinstance(block, TextBlock) and block.text.strip():
                    # Main agent's own reasoning/narration, printed as progress
                    print(f"  {block.text.strip()[:200]}")

        if hasattr(message, "result") and message.result:
            print("\n--- Pipeline finished ---")

    if output_path.exists():
        print(f"\nMemo saved to: {output_path}")
    else:
        print(
            "\nWarning: expected output file was not found. "
            "Check the run log above for errors."
        )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "invention"


async def async_main(args: argparse.Namespace) -> None:
    if args.interview:
        invention_text = await run_interview()
        title_hint = invention_text[:40]
    elif args.file:
        invention_text = Path(args.file).read_text()
        title_hint = Path(args.file).stem
    else:
        invention_text = args.text
        title_hint = invention_text[:40]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = REPORTS_DIR / f"{timestamp}-{slugify(title_hint)}.md"

    await run_pipeline(invention_text, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a patentability & prior art screen on an invention description."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Path to a text file describing the invention")
    group.add_argument("--text", help="Invention description passed directly as text")
    group.add_argument(
        "--interview",
        action="store_true",
        help="Answer questions from an interviewer agent to build the description interactively",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "No ANTHROPIC_API_KEY found in the environment — that's fine if you're "
            "authenticated via a Claude Pro/Max subscription login (`claude login`). "
            "This run will use that login instead of a separate API key.\n"
            "If you intended to use an API key instead, copy .env.example to .env "
            "and add it there.\n"
        )

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
