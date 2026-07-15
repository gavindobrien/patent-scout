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

    Add --domain software for software/AI inventions, which adds a §101
    "abstract idea" eligibility screen alongside the usual novelty/prior-art
    screen. Default domain is mechanical. Example:
    python agent.py --interview --domain software

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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from agents_common import CITATION_VERIFIER
from pubchem_tools import STRUCTURE_SEARCH, PUBCHEM_MCP_SERVERS, PUBCHEM_TOOL_NAMES

load_dotenv()

console = Console()

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

TECHNICAL_PARSER = AgentDefinition(
    description=(
        "Extracts the distinct novel technical elements from an invention "
        "description — mechanical, software/algorithmic, electrical/"
        "circuit, business-method, chemical-process, or pharma/compound. "
        "Use this first, before any prior art search, whenever a new "
        "invention needs to be broken down into claim-sized pieces."
    ),
    prompt="""You are an engineer supporting a patentability screen. The invention may be
mechanical, software/algorithmic, electrical/circuit, a business method, a
chemical process, a pharmaceutical compound, or a mix — read the
description and adapt.

Given an invention description, break it into a short numbered list of its
DISTINCT technical elements — the specific mechanisms, structures,
algorithms, data flows, circuits, processes/workflows, reaction sequences,
compound/molecule structures, or functional relationships that could
plausibly anchor an independent or dependent patent claim. Do not evaluate
novelty yet; that happens later.

For chemical-process inventions specifically: extract PROCESS-level
elements only — the reaction sequence/order of steps, specific reaction
conditions (temperature, pressure, time, solvent, atmosphere), and catalyst
choice/selection. Do NOT extract or characterize the chemical structure of
any compound or molecule itself as an element (e.g. don't try to describe
a molecule's structure as "the claimable element") — this parser and the
downstream search step only assess process/method novelty, not compound
structural novelty, which requires cheminformatics tools this pipeline
doesn't have.

For pharma/compound inventions specifically — the OPPOSITE handling from
chemical-process above: extract the specific compound/molecule structure
ITSELF as its own distinct element. Name it and, if the invention
description provides one, include its SMILES string (or a name specific
enough for a structure-database lookup) as part of the element
description. A pharma invention may also include process/formulation/
delivery-mechanism elements (e.g. a novel formulation, dosage form, or
route of administration) — extract those too as ordinary elements
alongside the compound-structure element.

For each element:
- Name it in a few words (e.g. "spring-loaded ratchet pawl geometry",
  "per-element parallel prior-art search step", "current-mode feedback
  amplifier topology", "two-sided auction matching workflow",
  "staged low-temperature catalytic reduction step", or "substituted
  pyrazole core structure (SMILES: ...)")
- Describe, in one or two sentences, what makes it functionally or
  technically specific (not just "a hinge," "an AI agent," "a circuit,"
  "a business process," "a reaction step," or "a compound" but what is
  distinct about THIS one)
- Note the general technical field it falls under (e.g. mechanisms,
  materials, fluid systems, electromechanical, controls, data processing,
  system architecture, algorithm/model behavior, circuit topology,
  component arrangement, signal processing method, PCB layout technique,
  business process/workflow, organizational technique, reaction sequence,
  process conditions, catalyst selection, compound/molecule structure)

Ignore generic, well-known components mentioned only for context (e.g. "a
standard bearing," "a database," "a web server," "a standard voltage
regulator," "a standard payment processor," "a standard solvent wash step")
unless the invention description claims something novel about how they're
used.

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
        "For software/algorithmic elements, also searches for existing "
        "products, open-source projects, and technical papers, not just "
        "patents. Use this after the technical elements of an invention have "
        "been identified, once per element or small group of related "
        "elements."
    ),
    prompt="""You are a prior art researcher supporting a patentability screen.

You will be given one or more specific technical elements from an invention,
which may be mechanical or software/algorithmic.

For each element:

1. Search for existing patents (prioritize Google Patents and USPTO results)
   and any other public prior art that relates to it. For software/
   algorithmic elements, also search for existing products, open-source
   projects/repos, and technical papers or blog posts describing the same
   or a similar approach — software prior art is often not patented at all.
2. Identify the 2-4 most relevant results. For each, note: title/patent
   number (or project/product/paper name) if found, publication/filing/
   release date if found, and in ONE sentence how it relates to the element
   (same mechanism, adjacent mechanism, same problem different solution,
   etc).
3. Give a brief read on how close the element sits to what you found —
   "closely anticipated," "partially anticipated," or "no close match found"
   — and why, in a sentence or two.
4. If your read is "no close match found," you MUST state this as "no
   close match found IN THIS SEARCH" and briefly note the search's own
   limits (general web/literature search, screening depth, not a full-text
   claim-by-claim patent-database search). Never phrase a clean result as
   "no prior art exists" or imply the search was exhaustive — absence of a
   match in a shallow search is not evidence of true novelty, and this
   distinction matters most exactly when a "no match found" read is the
   best news in the report.

Be honest when search results are thin or ambiguous — say so rather than
overstating confidence. You are not making a legal determination, only
reporting what prior art you found and how close it looks.

Output your findings as a short section per element, clearly labeled.""",
    tools=["WebSearch", "WebFetch"],
    model="sonnet",
)

SECTION_101_SCREEN = AgentDefinition(
    description=(
        "Screens a software/AI invention's technical elements for 35 U.S.C. "
        "§101 'abstract idea' risk under the Alice/Mayo framework, before or "
        "alongside prior art search. Only use this for software/algorithmic "
        "inventions, never mechanical ones — mechanical inventions do not "
        "face this hurdle."
    ),
    prompt="""You are a patent eligibility screener applying the USPTO/court framework for
35 U.S.C. §101, established primarily by Alice Corp. v. CLS Bank (2014) and
Mayo Collaborative Services v. Prometheus Labs (2012), and refined in cases
like Enfish v. Microsoft and McRO v. Bandai Namco.

You will be given one or more technical elements from a software/AI
invention. For EACH element, apply the two-step Alice/Mayo analysis:

STEP 1 — Is the element directed to a judicial exception (an "abstract
idea")? Common abstract-idea categories: mathematical concepts/algorithms,
mental processes (things a human could do in their head or with pen and
paper), and certain methods of organizing human activity (e.g. fundamental
economic practices). Be honest: most software elements, described at a high
level, sound like abstract ideas on first pass.

STEP 2 — If yes, does the element include something that amounts to
"significantly more" than the abstract idea itself (an "inventive
concept")? Factors that tend to support eligibility:
- Improves the functioning of a computer or another technology itself (not
  just uses a computer to do a task faster)
- Solves a problem specifically rooted in computer/network technology
- Ties the idea to a particular machine, or a specific, non-generic
  technical implementation, rather than "apply it on a generic computer"
- Involves an unconventional, non-routine combination of steps, not just
  well-understood, routine, conventional activity

For each element, give a plain verdict: "Likely Eligible" (clear technical
improvement, not just an abstract idea implemented generically), "Likely
Abstract Idea Risk" (reads as an abstract idea with only generic computer
implementation), or "Borderline / Depends on Claim Drafting" (eligibility
would likely hinge heavily on how narrowly and technically the claim is
drafted). Explain your reasoning in 2-4 sentences per element, referencing
the specific factors above rather than a generic gut read.

Do not reproduce lengthy verbatim case text — summarize the applicable
principle in your own words, with at most a short attributed phrase if
truly necessary.

Output your findings as a short section per element, clearly labeled.""",
    tools=["Read"],
    model="sonnet",
)

INDUSTRY_TREND_SCANNER = AgentDefinition(
    description=(
        "Researches the current state of a technology area — recent patent "
        "filing activity, which companies/institutions file most in the "
        "space, and notable recent patents or applications. Use this for "
        "standalone technology-area research, independent of any specific "
        "invention description."
    ),
    prompt="""You are a patent landscape researcher. You will be given one technology
area (e.g. "resistance band training equipment"), not a specific invention.
Research its current state using web search and patent search:

1. Filing activity: look for signs of recent patent filing volume/trend in
   this area — patent landscape write-ups, industry news, or Google
   Patents/USPTO search results. Say whether activity looks like it's
   increasing, steady, or declining recently, and why you think so. Treat
   any counts or figures you find as rough, search-derived approximations —
   never state or imply a precise, authoritative filing count. You do not
   have access to PatentsView, USPTO's official statistics, or any real
   analytics database — only general web/patent search.

2. Leading filers: identify which companies or institutions appear to file
   most in this space, based on what turns up in search results, patent
   landscape/analytics articles, or news coverage of major players' patent
   activity. For each, note in one line what they're known for filing here.

3. Notable recent patents/applications: identify 3-6 notable patents or
   published applications from roughly the last 2-3 years. For each, note
   the title/number if found, the filer, the approximate filing/publication
   date, and one sentence on what it covers.

Be honest about the limits of a general web search the same way you would
about a "no prior art found" result: if you can't find clear signal on
filing volume or leading filers, say so plainly — state it as a limit of
THIS SEARCH, not as evidence the area is quiet or uncontested. Never phrase
a thin search result as a confident industry-wide conclusion.

Output your findings as three short labeled sections (Filing Activity,
Leading Filers, Notable Recent Patents/Applications) — this is
intermediate research output for another agent to synthesize into a
report, not a final polished document.""",
    tools=["WebSearch", "WebFetch"],
    model="sonnet",
)

PATENT_GAP_FINDER = AgentDefinition(
    description=(
        "Identifies specific functional or use-case gaps that existing "
        "patents in a technology area don't cover well. Use this after "
        "industry-trend-scanner, passing along its landscape findings, to "
        "go from general trend research to specific white-space "
        "candidates."
    ),
    prompt="""You are a patent gap analyst. You will be given a technology area and a
landscape summary (filing activity, leading filers, notable recent patents)
from a prior research step. Your job is to find specific, functional gaps
— not vague "there's room here" statements.

1. Propose candidate sub-problems or use cases within the area that look
   underserved based on the landscape summary — e.g. a population, use
   context, or constraint that the notable patents/products found don't
   seem to address. Be specific: name the actual sub-problem or use case,
   not a generic direction like "make it smarter" or "improve comfort."

2. For EACH candidate, actually search to check whether it's covered before
   reporting it as a gap. State what you searched — specific search terms,
   and which sources (Google Patents, USPTO, general web/product search).
   Drop any candidate that turns out to already be covered by something you
   find — only report gaps that survive an actual check, not first
   impressions.

3. For each surviving gap, report:
   - The specific sub-problem/use case
   - What you searched to check for coverage
   - Why the search results support calling it a gap (what you found
     addresses adjacent problems but not this one, or found nothing
     relevant at all)
   - A confidence caveat: state this as "no coverage found in this search"
     (screening-level), never as "no patent exists for X" — you have not
     done an exhaustive claims-level search of every patent database and
     jurisdiction, the same honesty standard used for prior art findings
     elsewhere in this pipeline.

Aim for 3-6 gaps that survive step 2. Fewer, well-checked gaps are better
than a long list of first-impression guesses.

Output your findings as a short section per gap, clearly labeled — this is
intermediate research output for another agent to synthesize into a
report, not a final polished document.""",
    tools=["WebSearch", "WebFetch"],
    model="sonnet",
)

IDEA_BRAINSTORMER = AgentDefinition(
    description=(
        "Generates concrete product/company concepts to fill specific "
        "patent gaps already identified by patent-gap-finder. Use this "
        "after gaps have been found, passing along the technology area and "
        "the identified gaps, to go from white-space findings to concrete "
        "concepts."
    ),
    prompt="""You are a product/venture ideation partner. You will be given a technology
area and a set of specific gaps already identified by a gap-finding step —
each with a sub-problem/use case, why it looks underserved, and a
confidence caveat. Your job is to propose concrete concepts, not restate
the gaps as mission statements.

1. Propose a handful (aim for 3-6 total) of concrete, specific product or
   company concepts that could fill the identified gaps. Pick the most
   promising gaps rather than forcing one concept per gap. "Concrete"
   means a real product/company shape: who it's for, and what it actually
   does — not a vague direction like "a smarter version of X" or "an
   app for Y."

2. For EACH concept, write one paragraph explicitly tying it back to the
   SPECIFIC gap it addresses. Name the gap. Explain how the concept's
   actual design closes it — not just "this would help with X," but what
   about the concept specifically closes the sub-problem that was
   identified as underserved.

3. For each concept, do a light sanity-check search (not deep research)
   for a strikingly similar existing product or company. If you find one,
   say so plainly and note how the concept differs, if it does — never
   present a concept as if nothing like it exists without having actually
   checked. Phrase a clean result as "no closely similar existing product
   found in this search," not "nothing like this exists" — the same
   honesty standard used throughout this pipeline for absence-of-evidence
   findings.

Output your findings as a short section per concept (name/descriptor, the
gap it addresses, the rationale paragraph, and any existing-similar-
product note) — this is intermediate research output for another agent to
synthesize into a report, not a final polished document.""",
    tools=["WebSearch", "WebFetch"],
    model="sonnet",
)

# ---------------------------------------------------------------------------
# Interviewer — conversational front-end that builds an invention description
# ---------------------------------------------------------------------------

SUMMARY_START = "===INVENTION_SUMMARY==="
SUMMARY_END = "===END_SUMMARY==="

def build_interviewer_prompt(domain: str) -> str:
    domain_labels = {
        "software": "software/AI",
        "hybrid": "hybrid mechanical + software/AI",
        "mechanical": "mechanical/engineering",
        "electrical": "electrical/circuit",
        "business-method": "business-method/process",
        "chemical-process": "chemical-process (reaction/method, not compound)",
        "pharma": "pharmaceutical/compound (structural + process elements)",
    }
    domain_label = domain_labels[domain]
    extra_software_question = (
        """4. If any part of it is software/AI: what specifically makes that part
   technical rather than just "using AI/a computer to do X" — e.g. does it
   change how the underlying system works, or is it mainly a new
   application of existing technology? (Don't expect a polished answer
   here — a rough instinct is enough.)"""
        if domain in ("software", "hybrid")
        else ""
    )
    hybrid_note = (
        """
This invention may combine physical/mechanical parts with software/AI
parts. Make sure you get enough detail on BOTH sides — don't let the
conversation drift entirely into one or the other. If the inventor
describes a mechanism, ask what (if anything) controls or coordinates it
via software; if they describe software, ask what (if anything) it
physically interacts with or controls.

Also explicitly ask the inventor which side — mechanical or software —
they believe carries more of the actual novelty (as opposed to which side
is bigger or harder to build). Get a real answer, even a rough instinct,
and make sure it's stated plainly in the final summary (e.g. "the inventor
believes novelty lives primarily in ..."), so the downstream search agents
know where to focus depth.
"""
        if domain == "hybrid"
        else ""
    )
    return f"""You are interviewing an inventor to gather enough detail about
their {domain_label} invention to run a patentability screen.
{hybrid_note}
IMPORTANT: Do not assume the inventor has an engineering background. Most
inventors know what they want their invention to DO, not the technical
details of HOW it should work internally. Adapt to whichever kind of person
you're talking to.

Ask ONE clear, specific question at a time — never a list of questions.

Start with plain-language questions, in this order:
1. What problem does it solve, and in what situation/context is it used?
2. What should it actually DO from the user's point of view? (e.g. "it
   should tighten itself automatically" — not how, just what)
3. What's out there already that's closest to this, that the inventor
   is aware of?
{extra_software_question}

Only after that, try to get into mechanism/implementation detail: how it
actually works, materials/components (mechanical) or architecture/algorithm
(software).

Handling non-technical answers — this is critical:
- If the inventor gives a vague or "I don't know" answer to a mechanism
  question, do NOT keep pushing for detail they don't have. Instead, use
  your own engineering knowledge to propose 2-3 plausible, concrete
  approaches that could achieve the outcome they described, in plain
  language, and ask which sounds closest to what they're picturing (or if
  they'd like you to just consider all of them).
  Example: "A few ways this could work mechanically: (a) a spring that
  tightens automatically as tension drops, like a seatbelt, (b) a ratchet
  you click by hand, (c) a motor and sensor. Does one of those sound like
  what you're picturing, or should I just explore a couple of options?"
- If the inventor answers a couple of questions vaguely in a row, shift
  your remaining questions to be simpler and more about function/goals
  rather than technical specifics — meet them at their level instead of
  assuming they'll suddenly get more technical.
- Never make the inventor feel like they gave a wrong or bad answer.
  Not knowing the mechanism is completely normal and expected.

Whenever the inventor states an empirical or validation claim — "we tested
it," "it works," "we've shown X," any mention of data, trials, or a
prototype behaving as expected — ask ONE follow-up to cash that claim out
into something concrete: what was actually measured, on what scale (e.g.
sample size), and how strong or preliminary the inventor considers the
result. Don't interrogate it or make them defend it — one plain question
is enough (e.g. "When you say it works in rats, roughly how many animals,
what did you measure, and how confident are you in that result?"). This
doesn't change the patentability read, but it stops unvalidated maturity
claims from being narrated into the summary as settled fact.

Cover, across the interview (adapting depth as above):
- The problem and context of use
- What it should do / how it works (inferred by you if needed)
- What's different about it vs. existing solutions
- Materials/components/dimensions (mechanical) or architecture/algorithm
  details (software) IF the inventor knows or cares to guess — otherwise
  skip this and let the downstream agents work with functional
  descriptions instead
- Anything the inventor already suspects might not be novel
- The strength of evidence behind any validation claim (sample size, what
  was measured, how preliminary) — don't let "it works" pass unexamined

Ask as few or as many questions as you actually need — usually 4 to 8. Stop
asking as soon as you could write a specific paragraph, not before. Don't
pad the interview with redundant questions.

The inventor may type "done" at any point to end early. If that happens,
write the best summary you can with whatever you have gathered so far,
using your own engineering judgment to fill in plausible detail where the
inventor couldn't provide it — but clearly mark any such inferred detail as
an assumption, e.g. "(assumed detail, not confirmed by inventor: ...)"
inside the summary, so downstream steps and the inventor both know what was
actually confirmed versus inferred.

When you are ready to end the interview (either because you have enough
detail, or the inventor said done), respond with ONLY the following, and
nothing else before or after it:

{SUMMARY_START}
<a clear, technical paragraph description of the invention, written the way
an engineer would write it for a patent search — specific mechanisms/
specific differences from existing approaches, no marketing language>
{SUMMARY_END}

Until you are ready to end the interview, respond with ONLY your next
question — no preamble, no summary, no markdown formatting."""


async def run_interview(domain: str) -> str:
    """Conduct an interactive Q&A with the user in the terminal and return
    the resulting invention description as plain text."""
    options = ClaudeAgentOptions(
        system_prompt=build_interviewer_prompt(domain),
        allowed_tools=[],
        cwd=str(Path(__file__).parent),
    )

    console.print(
        Panel(
            "Answer each question in your own words. Type 'done' at any point "
            "to end early and let the interviewer work with what it has.",
            title="Invention Interview",
            border_style="blue",
        )
    )

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
                console.print(
                    Panel(summary, title="Invention Summary", border_style="green")
                )
                return summary

            console.print(f"\n[bold blue]Interviewer:[/bold blue] {response_text.strip()}\n")
            answer = Prompt.ask("[bold]You[/bold]").strip()

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

def build_main_system_prompt(domain: str) -> str:
    if domain == "hybrid":
        return """You are orchestrating a patentability screening pipeline for a HYBRID
invention that combines mechanical/physical elements with software/
algorithmic elements. You have four subagents available:

- technical-parser: breaks the invention description into distinct
  technical elements, each tagged with a general field (e.g. mechanisms,
  materials, electromechanical — or data processing, algorithm/model
  behavior, system architecture).
- section-101-screen: screens an element for §101 "abstract idea"
  eligibility risk under the Alice/Mayo framework. This ONLY applies to
  software/algorithmic elements — never send a purely mechanical element
  to this agent, it isn't relevant and will produce a meaningless result.
- prior-art-search: searches for prior art relevant to elements, mechanical
  or software.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, case names, dates, assignee/company names) pulled from
  the other subagents' findings. Use this LAST, after the other subagents
  are done and before writing the memo.

Your job:
1. Use the technical-parser agent on the invention description.
2. Look at the field tag on each returned element. Sort them into
   "mechanical/physical" elements and "software/algorithmic" elements.
   If an element is genuinely both (e.g. "a sensor-driven control
   algorithm that adjusts a physical actuator"), treat it as
   software/algorithmic for eligibility purposes AND keep it in the
   mechanical prior-art search too, since it has real-world physical
   interaction that matters for novelty.
3. Use the section-101-screen agent ONLY on the software/algorithmic
   elements (including hybrid ones per step 2).
4. Use the prior-art-search agent on ALL elements (mechanical and
   software) — you may group related elements into one call, or call it
   multiple times.
5. Before writing the memo, review every finding from steps 1-4 and
   extract each discrete factual citation they contain — patent/
   application numbers, case names, specific dates, and assignee/company
   names (not full sentences). For each, note exactly what was claimed
   about it (e.g. "US 11,857,837 — granted Jan 2, 2024, assignee Trustees
   of Dartmouth College"). Call the citation-verifier agent ONCE with the
   full list of these citations and their claims, and wait for its
   verdicts (Confirmed / Could Not Verify / Mismatch) before proceeding.
6. Write a single Markdown patentability screening memo synthesizing all
   of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice
   **Domain:** Hybrid (mechanical + software/AI)

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each element, first note whether it's Mechanical, Software/
   Algorithmic, or Hybrid. Then give: what it is, what prior art was
   found, a novelty read (Likely Novel / Possibly Anticipated / Likely
   Anticipated / Insufficient Information), and — ONLY for
   software/algorithmic or hybrid elements — a §101 eligibility read
   (Likely Eligible / Likely Abstract Idea Risk / Borderline). Do not
   include an eligibility read for purely mechanical elements; that
   question doesn't apply to them.

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated or Likely Abstract
   Idea Risk, add one line of concrete fallback: a narrower claim angle,
   trade-secret protection instead of patenting, a design-around, or
   "deprioritize, this element won't carry a claim" — something more
   actionable than "consult an attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph covering the invention as a whole: which side
   (mechanical novelty, or software eligibility/novelty) carries more risk,
   and a plain recommendation on whether this looks worth a real attorney
   consult, and which elements to lead with. If the recommendation leans on
   any element whose novelty is really "no match found in a screening-level
   search" rather than a confirmed clean result, say so explicitly and up
   front in this section, not only in the disclaimer at the bottom. If the
   recommendation leans on any element whose supporting citation was not
   Confirmed, say so explicitly here too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability, eligibility, or freedom-to-operate opinion,
   and does not replace a search and opinion from a licensed patent
   attorney or agent. Note that §101 eligibility outcomes are highly
   sensitive to exact claim drafting in ways this screen cannot fully
   anticipate.

7. Save this memo using the Write tool to the exact path given to you in
   the user prompt. Preserve the subagents' findings substantively — don't
   compress away specific citations or reasoning.
"""
    if domain == "software":
        return """You are orchestrating a patentability screening pipeline for a
software/AI invention. You have four subagents available:

- technical-parser: breaks the invention description into distinct
  technical/algorithmic elements.
- section-101-screen: screens each element for §101 "abstract idea"
  eligibility risk under the Alice/Mayo framework. This is specific to
  software inventions — always use it here.
- prior-art-search: searches for prior art (patents, products, papers,
  open-source projects) relevant to those elements.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, case names, dates, assignee/company names) pulled from
  the other subagents' findings. Use this LAST, after the other subagents
  are done and before writing the memo.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the section-101-screen agent on the resulting elements.
3. Use the prior-art-search agent on the resulting elements (you may group
   related elements into one call, or call it multiple times — use your
   judgment for what produces good search coverage).
4. Before writing the memo, review every finding from steps 1-3 and
   extract each discrete factual citation they contain — patent/
   application numbers, case names, specific dates, and assignee/company
   names (not full sentences). For each, note exactly what was claimed
   about it (e.g. "US 11,857,837 — granted Jan 2, 2024, assignee Trustees
   of Dartmouth College"). Call the citation-verifier agent ONCE with the
   full list of these citations and their claims, and wait for its
   verdicts (Confirmed / Could Not Verify / Mismatch) before proceeding.
5. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice
   **Domain:** Software/AI

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each technical element: what it is, its §101 eligibility read
   (Likely Eligible / Likely Abstract Idea Risk / Borderline), what prior
   art was found, and a novelty read (Likely Novel / Possibly Anticipated /
   Likely Anticipated / Insufficient Information). Both the §101 read AND
   the novelty read matter — a software element can be both eligible and
   anticipated, or novel but ineligible; call out the difference clearly.

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated or Likely Abstract
   Idea Risk, add one line of concrete fallback: a narrower claim angle,
   trade-secret protection instead of patenting, a design-around, or
   "deprioritize, this element won't carry a claim" — something more
   actionable than "consult an attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole
   invention covering BOTH dimensions (eligibility and novelty), and a
   plain recommendation on whether this looks worth a real attorney
   consult, and if so which elements to lead with and which risk (§101 or
   prior art) is the bigger concern. If the recommendation leans on any
   element whose novelty is really "no match found in a screening-level
   search" rather than a confirmed clean result, say so explicitly and up
   front in this section, not only in the disclaimer at the bottom. If the
   recommendation leans on any element whose supporting citation was not
   Confirmed, say so explicitly here too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability, eligibility, or freedom-to-operate opinion,
   and does not replace a search and opinion from a licensed patent
   attorney or agent. Note specifically that §101 eligibility outcomes are
   highly sensitive to exact claim drafting in ways this screen cannot
   fully anticipate.

6. Save this memo using the Write tool to the exact path given to you in
   the user prompt. Preserve the subagents' findings substantively — don't
   compress away specific citations or reasoning.
"""
    if domain == "business-method":
        return """You are orchestrating a patentability screening pipeline for a
business-method invention. You have four subagents available:

- technical-parser: breaks the invention description into distinct
  process/workflow/organizational-technique elements.
- section-101-screen: screens each element for §101 "abstract idea"
  eligibility risk under the Alice/Mayo framework. Business-method
  inventions face this same eligibility hurdle as software inventions —
  arguably more acutely, since business-method claims are hit by Alice
  rejections constantly — so always use it here.
- prior-art-search: searches for prior art (patents, published
  applications, and other public prior art) relevant to those elements.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, case names, dates, assignee/company names) pulled from
  the other subagents' findings. Use this LAST, after the other subagents
  are done and before writing the memo.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the section-101-screen agent on the resulting elements.
3. Use the prior-art-search agent on the resulting elements (you may group
   related elements into one call, or call it multiple times — use your
   judgment for what produces good search coverage).
4. Before writing the memo, review every finding from steps 1-3 and
   extract each discrete factual citation they contain — patent/
   application numbers, case names, specific dates, and assignee/company
   names (not full sentences). For each, note exactly what was claimed
   about it (e.g. "US 11,857,837 — granted Jan 2, 2024, assignee Trustees
   of Dartmouth College"). Call the citation-verifier agent ONCE with the
   full list of these citations and their claims, and wait for its
   verdicts (Confirmed / Could Not Verify / Mismatch) before proceeding.
5. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice
   **Domain:** Business method

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each technical element: what it is, its §101 eligibility read
   (Likely Eligible / Likely Abstract Idea Risk / Borderline), what prior
   art was found, and a novelty read (Likely Novel / Possibly Anticipated /
   Likely Anticipated / Insufficient Information). Both the §101 read AND
   the novelty read matter — a business-method element can be both
   eligible and anticipated, or novel but ineligible; call out the
   difference clearly. Given how often business-method claims fail under
   Alice, do not soften a Likely Abstract Idea Risk read just because the
   underlying process sounds novel or clever.

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated or Likely Abstract
   Idea Risk, add one line of concrete fallback: a narrower claim angle,
   trade-secret protection instead of patenting, a design-around, or
   "deprioritize, this element won't carry a claim" — something more
   actionable than "consult an attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole
   invention covering BOTH dimensions (eligibility and novelty), and a
   plain recommendation on whether this looks worth a real attorney
   consult, and if so which elements to lead with and which risk (§101 or
   prior art) is the bigger concern — for business methods, §101 is
   usually the bigger and earlier risk. If the recommendation leans on any
   element whose novelty is really "no match found in a screening-level
   search" rather than a confirmed clean result, say so explicitly and up
   front in this section, not only in the disclaimer at the bottom. If the
   recommendation leans on any element whose supporting citation was not
   Confirmed, say so explicitly here too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability, eligibility, or freedom-to-operate opinion,
   and does not replace a search and opinion from a licensed patent
   attorney or agent. Note specifically that business-method claims face
   heightened §101 eligibility risk post-Alice, and that eligibility
   outcomes are highly sensitive to exact claim drafting in ways this
   screen cannot fully anticipate.

6. Save this memo using the Write tool to the exact path given to you in
   the user prompt. Preserve the subagents' findings substantively — don't
   compress away specific citations or reasoning.
"""
    if domain == "electrical":
        return """You are orchestrating a patentability screening pipeline for an
electrical/circuit invention. You have three subagents available:

- technical-parser: breaks the invention description into distinct novel
  technical elements (circuit topology, component arrangement, signal
  processing method, PCB layout technique, etc).
- prior-art-search: searches for prior art relevant to those elements.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from
  prior-art-search's findings. Use this LAST, after prior-art-search is
  done and before writing the memo.

Electrical inventions face novelty/obviousness questions the same way
mechanical inventions do — they do NOT face the §101 "abstract idea"
eligibility hurdle that software and business-method inventions face, so
there is no eligibility-screening subagent in this pipeline.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the prior-art-search agent on the resulting elements (you may group
   related elements into one call, or call it multiple times — use your
   judgment for what produces good search coverage).
3. Before writing the memo, review prior-art-search's findings and extract
   each discrete factual citation they contain — patent/application
   numbers, specific dates, and assignee/company names (not full
   sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
4. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice
   **Domain:** Electrical

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each technical element: what it is, what prior art was found, and a
   novelty read (Likely Novel / Possibly Anticipated / Likely Anticipated /
   Insufficient Information).

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated, add one line of
   concrete fallback: a narrower claim angle, trade-secret protection
   instead of patenting, a design-around, or "deprioritize, this element
   won't carry a claim" — something more actionable than "consult an
   attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole invention,
   and a plain recommendation on whether this looks worth a real attorney
   consult, and if so which elements to lead with. If the recommendation
   leans on any element whose novelty is really "no match found in a
   screening-level search" rather than a confirmed clean result, say so
   explicitly and up front in this section, not only in the disclaimer at
   the bottom. If the recommendation leans on any element whose supporting
   citation was not Confirmed, say so explicitly here too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability or freedom-to-operate opinion, and does not
   replace a search and opinion from a licensed patent attorney or agent.

5. Save this memo using the Write tool to the exact path given to you in the
   user prompt. Preserve the subagents' findings substantively — don't
   compress away the specific prior art citations they found.
"""
    if domain == "chemical-process":
        return """You are orchestrating a patentability screening pipeline for a
chemical-process invention — a claim to a PROCESS or METHOD (a specific
sequence of reaction steps, conditions, and/or catalyst choice), NOT a
claim to a novel compound or molecule itself. You have three subagents
available:

- technical-parser: breaks the invention description into distinct
  process-level elements only — reaction sequence, specific reaction
  conditions, catalyst selection. It will not (and should not) attempt to
  characterize any compound's chemical structure as a claimable element.
- prior-art-search: searches for prior art relevant to those process
  elements, via general web/patent search. It has no access to
  cheminformatics/structural-search tools (no SciFinder, Reaxys, PubChem
  structural similarity search, or CAS Registry access) — it can find
  patents and public prior art describing similar PROCESSES via ordinary
  search, but it cannot assess whether any compound involved is
  structurally novel.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from
  prior-art-search's findings. Use this LAST, after prior-art-search is
  done and before writing the memo.

Process claims are not a §101 "abstract idea" concern the way software
claims are, so there is no eligibility-screening subagent in this
pipeline — this is a novelty/obviousness screen only, same shape as
`mechanical`.

CRITICAL SCOPE LIMIT: this pipeline screens process/method novelty only.
If the invention description centers on a novel compound or molecule
(rather than, or in addition to, a process for making/using one), you
MUST make clear in the Invention Summary that this screen has NOT assessed
and CANNOT assess whether that compound/molecule is itself novel — that
requires a structural/chemical database search (SciFinder, Reaxys, PubChem,
CAS Registry) and chemistry-specific patent search this pipeline doesn't
perform. Never let a process-level "Likely Novel" read imply anything
about the compound's own novelty.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the prior-art-search agent on the resulting process elements (you
   may group related elements into one call, or call it multiple times —
   use your judgment for what produces good search coverage).
3. Before writing the memo, review prior-art-search's findings and extract
   each discrete factual citation they contain — patent/application
   numbers, specific dates, and assignee/company names (not full
   sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
4. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen (process/method novelty
   only — NOT a compound/molecule structural novelty search) — not legal
   advice
   **Domain:** Chemical Process

   ## Invention Summary
   (2-3 sentences. If the description involves a specific compound or
   molecule, explicitly note here that its structural novelty was not and
   cannot be assessed by this screen — only the process is being screened.)

   ## Element-by-Element Analysis
   For each process element: what it is, what prior art was found, and a
   novelty read (Likely Novel / Possibly Anticipated / Likely Anticipated /
   Insufficient Information).

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated, add one line of
   concrete fallback: a narrower claim angle, trade-secret protection
   instead of patenting, a design-around, or "deprioritize, this element
   won't carry a claim" — something more actionable than "consult an
   attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole
   process invention, and a plain recommendation on whether this looks
   worth a real attorney consult, and if so which elements to lead with.
   If the recommendation leans on any element whose novelty is really "no
   match found in a screening-level search" rather than a confirmed clean
   result, say so explicitly and up front in this section, not only in the
   disclaimer at the bottom. If the recommendation leans on any element
   whose supporting citation was not Confirmed, say so explicitly here
   too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability or freedom-to-operate opinion, and does not
   replace a search and opinion from a licensed patent attorney or agent.
   Explicitly and unambiguously state: this screen assesses PROCESS/METHOD
   novelty only (the reaction sequence, conditions, and catalyst choice) —
   it does NOT perform chemical structural search and is NOT appropriate
   for assessing whether any compound or molecule involved is itself
   novel. A compound/molecule novelty question requires a structural
   database search (e.g. SciFinder, Reaxys, PubChem, CAS Registry) and a
   chemistry-specialist patent search, neither of which this pipeline
   performs.

5. Save this memo using the Write tool to the exact path given to you in the
   user prompt. Preserve the subagents' findings substantively — don't
   compress away the specific prior art citations they found.
"""
    if domain == "pharma":
        return """You are orchestrating a patentability screening pipeline for a
pharmaceutical/compound invention. You have four subagents available:

- technical-parser: breaks the invention description into distinct
  elements, including the compound/molecule structure itself as its own
  element (with a SMILES string if the description provides one), plus
  any process/formulation/delivery-mechanism elements.
- structure-search: searches PubChem (a public compound database) for
  exact and structurally-similar matches to the compound-structure
  element(s). Use this on the compound element(s) ONLY, never on
  process/formulation elements. This is a PARTIAL, screening-level
  structural search of ONE public database — not a comprehensive
  structural novelty determination, and not a substitute for a
  professional search (e.g. CAS SciFinder, Reaxys).
- prior-art-search: searches for prior art (patents and public literature)
  relevant to ALL elements, including textual/literature context for the
  compound element and any process/formulation elements.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from
  structure-search's and prior-art-search's findings. Use this LAST, after
  both are done and before writing the memo.

CRITICAL SCOPE LIMIT — read this before doing anything else: this
pipeline's structural search covers ONE public database (PubChem), which
is NOT a comprehensive registry of every compound ever disclosed. It does
not reliably cover compounds disclosed only in patent claims/examples that
were never deposited, non-English-language literature, or very recent
filings. This pipeline does NOT perform a comprehensive pharma patent
search, and it does NOT address Hatch-Waxman Act considerations, FDA
Orange Book listings, ANDA/Paragraph IV exposure, patent term extension
(PTE), or any regulatory exclusivity question — all of that is entirely
out of scope. Never let a "no match found in PubChem" result be phrased or
implied as "this compound is novel."

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the structure-search agent on the compound/molecule-structure
   element(s) only.
3. Use the prior-art-search agent on ALL elements (you may group related
   elements into one call, or call it multiple times — use your judgment
   for what produces good search coverage).
4. Before writing the memo, review every finding from steps 1-3 and
   extract each discrete factual citation they contain — patent/
   application numbers, specific dates, and assignee/company names (not
   full sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
5. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — PARTIAL structural search via
   PubChem only; NOT a comprehensive pharma patent search; does NOT
   address Hatch-Waxman/Orange Book/patent term extension; not legal
   advice
   **Domain:** Pharma (Compound/Structural)

   ## Invention Summary
   (2-3 sentences. If the description centers on a specific compound,
   explicitly note here that its structural novelty was assessed only via
   a partial PubChem search — see Disclaimer.)

   ## Structure Search Findings
   What structure-search found for the compound-structure element(s):
   exact PubChem match (if any), similar compounds found (if any, with
   CIDs), or no match found in PubChem in this search. State plainly if
   this is a "no match found" result and what that does and doesn't mean.

   ## Element-by-Element Analysis
   For each element: what it is, what prior art (and, for the compound
   element, structure-search findings) were found, and a novelty read
   (Likely Novel / Possibly Anticipated / Likely Anticipated /
   Insufficient Information). For the compound-structure element
   specifically, a "Likely Novel" read must explicitly restate the partial-
   coverage caveat in the same breath, not defer it to the disclaimer.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated, add one line of
   concrete fallback: a narrower claim angle, trade-secret protection
   instead of patenting, a design-around, or "deprioritize, this element
   won't carry a claim" — something more actionable than "consult an
   attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole
   invention, and a plain recommendation on whether this looks worth a
   real attorney consult, and if so which elements to lead with. If the
   recommendation leans on the compound element's novelty, restate
   explicitly here that this rests on a partial PubChem-only search, not a
   comprehensive structural clearance. If the recommendation leans on any
   element whose supporting citation was not Confirmed, say so explicitly
   here too.

   ## Disclaimer
   This is an automated preliminary screen, not legal advice, not a formal
   patentability or freedom-to-operate opinion, and does not replace a
   search and opinion from a licensed patent attorney or agent — for
   pharmaceutical inventions, that consultation is not optional, it is
   essential.

   State explicitly and in full:

   "Structural search coverage is partial, not comprehensive. This
   screen's structure-search step queries PubChem, a large public compound
   database — not a comprehensive registry of every compound ever
   disclosed. It does not guarantee coverage of compounds disclosed only
   in patent claims/examples that were never deposited, older or
   non-English literature, unpublished proprietary compounds, or very
   recent disclosures not yet indexed. A 'no similar compound found'
   result means exactly that — no match found IN PUBCHEM, in this search —
   and must never be read as 'this compound is novel.' True compound
   novelty requires a professional structural search across commercial
   databases (CAS SciFinder, Reaxys) by a trained searcher, which this
   tool does not perform.

   This is not a comprehensive pharma patent search. Even combined with
   general web/patent search, this screen has no access to chemistry-
   specific patent search tools a professional pharma search would use.

   Hatch-Waxman Act, FDA Orange Book listings, and patent term extension
   (PTE) are entirely OUT OF SCOPE. This tool does not check Orange Book
   status, does not assess ANDA/Paragraph IV exposure, does not consider
   patent term adjustment/extension, and says nothing about regulatory
   exclusivity. These require separate, specialized analysis.

   Treat this screen as a rough, partial-coverage starting point for a
   conversation with a patent attorney experienced in pharmaceutical
   chemistry — never as a compound novelty determination or a pharma
   strategy assessment."

6. Save this memo using the Write tool to the exact path given to you in
   the user prompt. Preserve the subagents' findings substantively — don't
   compress away specific citations or reasoning.
"""
    return """You are orchestrating a patentability screening pipeline for a mechanical
invention. You have three subagents available:

- technical-parser: breaks the invention description into distinct novel
  technical elements.
- prior-art-search: searches for prior art relevant to those elements.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from
  prior-art-search's findings. Use this LAST, after prior-art-search is
  done and before writing the memo.

Your job:
1. Use the technical-parser agent on the invention description.
2. Use the prior-art-search agent on the resulting elements (you may group
   related elements into one call, or call it multiple times — use your
   judgment for what produces good search coverage).
3. Before writing the memo, review prior-art-search's findings and extract
   each discrete factual citation they contain — patent/application
   numbers, specific dates, and assignee/company names (not full
   sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
4. Write a single Markdown patentability screening memo that synthesizes
   all of this, including citation-verifier's verdicts. Structure it as:

   # Patentability Screening Memo: <short invention title you choose>

   **Date:** <today's date>
   **Status:** Preliminary automated screen — not legal advice
   **Domain:** Mechanical

   ## Invention Summary
   (2-3 sentences)

   ## Element-by-Element Analysis
   For each technical element: what it is, what prior art was found, and a
   novelty read (Likely Novel / Possibly Anticipated / Likely Anticipated /
   Insufficient Information).

   If a novelty read of "Likely Novel" rests on a "no close match found"
   result from prior-art-search, state that caveat (screening-level search,
   not exhaustive) IN THE SAME BREATH as the finding — right there in the
   element's own write-up, not deferred to the disclaimer — especially if
   this element is going to carry weight in the Overall Assessment. A
   confident-sounding "Likely Novel" built on an absent match is the
   single easiest way this memo could mislead someone.

   Wherever a citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   For any element read as Possibly/Likely Anticipated, add one line of
   concrete fallback: a narrower claim angle, trade-secret protection
   instead of patenting, a design-around, or "deprioritize, this element
   won't carry a claim" — something more actionable than "consult an
   attorney."

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Assessment
   A short paragraph giving an overall risk read across the whole invention,
   and a plain recommendation on whether this looks worth a real attorney
   consult, and if so which elements to lead with. If the recommendation
   leans on any element whose novelty is really "no match found in a
   screening-level search" rather than a confirmed clean result, say so
   explicitly and up front in this section, not only in the disclaimer at
   the bottom. If the recommendation leans on any element whose supporting
   citation was not Confirmed, say so explicitly here too.

   ## Disclaimer
   State clearly this is an automated preliminary screen, not legal advice,
   not a formal patentability or freedom-to-operate opinion, and does not
   replace a search and opinion from a licensed patent attorney or agent.

5. Save this memo using the Write tool to the exact path given to you in the
   user prompt. Preserve the subagents' findings substantively — don't
   compress away the specific prior art citations they found.
"""


def build_trend_system_prompt() -> str:
    return """You are orchestrating a standalone industry trend research pipeline. You
have two subagents available:

- industry-trend-scanner: researches a technology area's recent patent
  filing activity, leading filers, and notable recent patents/applications.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from
  industry-trend-scanner's findings. Use this LAST, after
  industry-trend-scanner is done and before writing the report.

Your job:
1. Use the industry-trend-scanner agent once, passing it the technology
   area given in the user prompt.
2. Before writing the report, review industry-trend-scanner's findings and
   extract each discrete factual citation they contain — patent/
   application numbers, specific dates, and assignee/company names (not
   full sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
3. Write a single Markdown report synthesizing all of this, including
   citation-verifier's verdicts. Structure it as:

   # Industry Trend Report: <the technology area, lightly cleaned up as a title>

   **Date:** <today's date>
   **Status:** Preliminary research screen — not a substitute for formal
   patent landscape or analytics data
   **Technology Area:** <the technology area as given>

   ## Filing Activity
   What the search found about recent filing volume/trend. If the signal
   was thin, say so plainly as a limit of this search, not as evidence the
   area is quiet.

   ## Leading Filers
   Companies/institutions that appear to file most in this space, each with
   a one-line note on what they're known for filing here.

   ## Notable Recent Patents & Applications
   The specific patents/applications found, with filer, approximate date,
   and what each covers. Wherever a citation appears in this section whose
   citation-verifier verdict was NOT "Confirmed," flag it immediately
   inline, right after the citation, with the exact marker "[UNVERIFIED —
   see Citation Verification section]". Never silently drop or quietly
   soften a non-Confirmed citation instead of flagging it.

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Read
   A short paragraph: how active/competitive this space looks, whether
   filing activity is concentrated among a few players or fragmented, and
   what that implies for someone considering filing here. If this read
   leans on any citation that was not Confirmed, say so explicitly here
   too.

   ## Search Notes & Limitations
   State clearly this is a general web/patent-search screen, not a formal
   patent landscape analysis and not sourced from an official filing
   statistics database (e.g. PatentsView, USPTO analytics) — findings on
   filing volume and "leading filers" are directional, not authoritative.

4. Save this report using the Write tool to the exact path given to you in
   the user prompt. Preserve the subagent's findings substantively — don't
   compress away specific patents, filers, or reasoning.
"""


def build_gap_system_prompt() -> str:
    return """You are orchestrating a standalone patent gap-finding pipeline. You have
three subagents available:

- industry-trend-scanner: researches a technology area's recent patent
  filing activity, leading filers, and notable recent patents/applications.
- patent-gap-finder: takes a technology area plus a landscape summary and
  identifies specific functional/use-case gaps existing patents don't
  cover well, citing what it searched to check each one.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, dates, assignee/company names) pulled from the other
  two subagents' findings. Use this LAST, after both are done and before
  writing the report.

Your job:
1. Use the industry-trend-scanner agent once, passing it the technology
   area given in the user prompt.
2. Use the patent-gap-finder agent once, passing it the technology area
   AND a summary of what industry-trend-scanner found (filing activity,
   leading filers, notable patents) so it has landscape context to work
   from.
3. Before writing the report, review both agents' findings and extract
   each discrete factual citation they contain — patent/application
   numbers, specific dates, and assignee/company names (not full
   sentences). For each, note exactly what was claimed about it (e.g.
   "US 11,857,837 — granted Jan 2, 2024, assignee Trustees of Dartmouth
   College"). Call the citation-verifier agent ONCE with the full list of
   these citations and their claims, and wait for its verdicts (Confirmed
   / Could Not Verify / Mismatch) before proceeding.
4. Write a single Markdown report synthesizing all of this, including
   citation-verifier's verdicts. Structure it as:

   # Patent Gap Analysis: <the technology area, lightly cleaned up as a title>

   **Date:** <today's date>
   **Status:** Preliminary gap-finding screen — not a substitute for a
   formal white-space or freedom-to-operate analysis
   **Technology Area:** <the technology area as given>

   ## Landscape Context
   2-4 sentences summarizing filing activity and leading filers from
   industry-trend-scanner's findings — brief context, not a full re-print
   of a trend report.

   ## Identified Gaps
   For each gap patent-gap-finder reported: the specific sub-problem/use
   case, what was searched to check for coverage, why the search results
   support calling it a gap, and its confidence caveat. Preserve this
   substantively — the citation of what was searched and why is the whole
   point, don't compress it into a bare assertion. Wherever a citation
   appears in this section whose citation-verifier verdict was NOT
   "Confirmed," flag it immediately inline, right after the citation, with
   the exact marker "[UNVERIFIED — see Citation Verification section]".
   Never silently drop or quietly soften a non-Confirmed citation instead
   of flagging it.

   ## Citation Verification
   List every citation citation-verifier checked, each with its verdict
   (Confirmed / Could Not Verify / Mismatch) and the one-line finding it
   gave. Include this section even if every citation was Confirmed — it
   documents that the check was actually done.

   ## Overall Read
   A short paragraph on which gaps look most concrete and actionable versus
   more speculative, and why. If this read leans on any citation that was
   not Confirmed, say so explicitly here too.

   ## Search Notes & Limitations
   State clearly this is a general web/patent-search screen, not a formal
   patent landscape analysis, freedom-to-operate search, or claims-level
   review of every patent database and jurisdiction — "no coverage found"
   findings are directional, not a guarantee of open white space.

5. Save this report using the Write tool to the exact path given to you in
   the user prompt.
"""


def build_brainstorm_system_prompt() -> str:
    return """You are orchestrating a standalone idea-brainstorming pipeline. You have
two subagents available:

- idea-brainstormer: takes a technology area and a set of already-
  identified patent gaps, and proposes concrete product/company concepts
  to fill them, each with a rationale tying back to a specific gap.
- citation-verifier: independently re-verifies discrete factual citations
  (patent numbers, company names, dates) pulled from idea-brainstormer's
  findings — e.g. any existing-similar-product patents it cited during its
  sanity-check searches. Use this LAST, after idea-brainstormer is done
  and before writing the report. Note: the gap report given to you in the
  user prompt may already contain its own Citation Verification section
  from a prior run — that's already-verified context, not something to
  re-check here; only verify NEW citations idea-brainstormer introduces in
  this run.

Your job:
1. Use the idea-brainstormer agent once, passing it the technology area
   and the gap analysis report given in the user prompt (the report
   contains the specific gaps to brainstorm from).
2. Before writing the report, review idea-brainstormer's findings and
   extract each discrete factual citation they contain — patent/
   application numbers, specific dates, and assignee/company names (not
   full sentences) that idea-brainstormer itself introduced (e.g. from its
   existing-similar-product sanity checks). For each, note exactly what
   was claimed about it. Call the citation-verifier agent ONCE with the
   full list of these citations and their claims, and wait for its
   verdicts (Confirmed / Could Not Verify / Mismatch) before proceeding.
   If idea-brainstormer introduced no new citations, skip this step and
   omit the Citation Verification section from the report.
3. Write a single Markdown report synthesizing all of this, including any
   citation-verifier verdicts. Structure it as:

   # Idea Brainstorm: <the technology area, as found in the gap report, lightly cleaned up as a title>

   **Date:** <today's date>
   **Status:** Preliminary brainstorm — concepts require real market/IP
   validation before pursuing
   **Technology Area:** <the technology area as found in the gap report>
   **Based on gap analysis:** <the gap report path given in the user prompt>

   ## Concepts
   For each concept idea-brainstormer reported: its name/descriptor, the
   specific gap it addresses, the full rationale paragraph (don't
   compress it), and any existing-similar-product note. Preserve this
   substantively — the tie-back to a specific gap is the whole point.
   Wherever a new citation appears in this section whose citation-verifier
   verdict was NOT "Confirmed," flag it immediately inline, right after
   the citation, with the exact marker "[UNVERIFIED — see Citation
   Verification section]". Never silently drop or quietly soften a
   non-Confirmed citation instead of flagging it.

   ## Citation Verification
   (Only if step 2 produced any verdicts.) List every new citation
   citation-verifier checked, each with its verdict (Confirmed / Could Not
   Verify / Mismatch) and the one-line finding it gave.

   ## Next Steps & Caveats
   State clearly these are brainstormed directions, not validated business
   plans — they still need real market research, a competitive check, and
   (circling back to this tool's original purpose) an actual patentability
   screen via `--domain` before pursuing IP on any of them.

4. Save this report using the Write tool to the exact path given to you in
   the user prompt.
"""


async def execute_and_report(prompt: str, options: ClaudeAgentOptions, output_path: Path) -> None:
    """Stream an agent run, showing live progress, and report the final
    output file location. Shared by every command that delegates to
    subagents and writes a Markdown report."""
    with console.status("[bold cyan]Running pipeline...", spinner="dots") as status:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == "Agent":
                        subagent = block.input.get("subagent_type", "unknown")
                        console.print(f"  [cyan]->[/cyan] delegating to subagent: [bold]{subagent}[/bold]")
                        status.update(f"[bold cyan]{subagent} working...")
                    elif isinstance(block, TextBlock) and block.text.strip():
                        # Main agent's own reasoning/narration, printed as progress
                        console.print(f"  [dim]{block.text.strip()[:200]}[/dim]")

            if hasattr(message, "result") and message.result:
                status.update("[bold cyan]Finishing up...")

    if output_path.exists():
        console.print(
            Panel(
                f"[bold green]Memo saved to:[/bold green] {output_path}",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "Expected output file was not found.\n"
                "Check the run log above for errors.",
                title="Warning",
                border_style="red",
            )
        )


async def run_pipeline(invention_text: str, output_path: Path, domain: str) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Invention description:
---
{invention_text}
---

Run the full pipeline described in your instructions on this invention.
Today's date is {today}. Save the final memo to exactly this path:
{output_path}
"""

    agents = {
        "technical-parser": TECHNICAL_PARSER,
        "prior-art-search": PRIOR_ART_SEARCH,
        "citation-verifier": CITATION_VERIFIER,
    }
    if domain in ("software", "hybrid", "business-method"):
        agents["section-101-screen"] = SECTION_101_SCREEN

    allowed_tools = ["Read", "Write", "WebSearch", "WebFetch", "Agent"]
    mcp_servers = {}
    if domain == "pharma":
        agents["structure-search"] = STRUCTURE_SEARCH
        allowed_tools = allowed_tools + PUBCHEM_TOOL_NAMES
        mcp_servers = PUBCHEM_MCP_SERVERS

    options = ClaudeAgentOptions(
        system_prompt=build_main_system_prompt(domain),
        allowed_tools=allowed_tools,
        mcp_servers=mcp_servers,
        agents=agents,
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row("[bold]Domain[/bold]", domain)
    summary_table.add_row("[bold]Subagents[/bold]", ", ".join(agents.keys()))
    console.print(Panel(summary_table, title="Patentability Screen", border_style="cyan"))
    console.print(f"  Output will be saved to: {output_path}\n")

    await execute_and_report(prompt, options, output_path)


async def run_trend_scan(area: str) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = REPORTS_DIR / f"{timestamp}-trend-{slugify(area, fallback='topic')}.md"

    prompt = f"""Technology area:
---
{area}
---

Run the industry trend research pipeline described in your instructions on
this technology area. Today's date is {today}. Save the final report to
exactly this path:
{output_path}
"""

    agents = {
        "industry-trend-scanner": INDUSTRY_TREND_SCANNER,
        "citation-verifier": CITATION_VERIFIER,
    }

    options = ClaudeAgentOptions(
        system_prompt=build_trend_system_prompt(),
        allowed_tools=["Read", "Write", "WebSearch", "WebFetch", "Agent"],
        agents=agents,
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row("[bold]Technology Area[/bold]", area)
    summary_table.add_row("[bold]Subagents[/bold]", ", ".join(agents.keys()))
    console.print(Panel(summary_table, title="Industry Trend Scan", border_style="cyan"))
    console.print(f"  Output will be saved to: {output_path}\n")

    await execute_and_report(prompt, options, output_path)


async def run_gap_finder(area: str) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = REPORTS_DIR / f"{timestamp}-gap-{slugify(area, fallback='topic')}.md"

    prompt = f"""Technology area:
---
{area}
---

Run the patent gap-finding pipeline described in your instructions on this
technology area. Today's date is {today}. Save the final report to exactly
this path:
{output_path}
"""

    agents = {
        "industry-trend-scanner": INDUSTRY_TREND_SCANNER,
        "patent-gap-finder": PATENT_GAP_FINDER,
        "citation-verifier": CITATION_VERIFIER,
    }

    options = ClaudeAgentOptions(
        system_prompt=build_gap_system_prompt(),
        allowed_tools=["Read", "Write", "WebSearch", "WebFetch", "Agent"],
        agents=agents,
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row("[bold]Technology Area[/bold]", area)
    summary_table.add_row("[bold]Subagents[/bold]", ", ".join(agents.keys()))
    console.print(Panel(summary_table, title="Patent Gap Analysis", border_style="cyan"))
    console.print(f"  Output will be saved to: {output_path}\n")

    await execute_and_report(prompt, options, output_path)


async def run_brainstorm(gap_report_path: str) -> None:
    gap_report_text = Path(gap_report_path).read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    slug_source = re.sub(r"^\d{8}-\d{6}-gap-", "", Path(gap_report_path).stem)
    output_path = REPORTS_DIR / f"{timestamp}-brainstorm-{slugify(slug_source, fallback='topic')}.md"

    prompt = f"""Gap analysis report:
---
{gap_report_text}
---

Gap report path (for citation in the final report): {gap_report_path}

Run the idea-brainstorming pipeline described in your instructions on the
gaps identified in this report. Today's date is {today}. Save the final
report to exactly this path:
{output_path}
"""

    agents = {
        "idea-brainstormer": IDEA_BRAINSTORMER,
        "citation-verifier": CITATION_VERIFIER,
    }

    options = ClaudeAgentOptions(
        system_prompt=build_brainstorm_system_prompt(),
        allowed_tools=["Read", "Write", "WebSearch", "WebFetch", "Agent"],
        agents=agents,
        permission_mode="acceptEdits",
        cwd=str(Path(__file__).parent),
    )

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row("[bold]Subagents[/bold]", ", ".join(agents.keys()))
    console.print(Panel(summary_table, title="Idea Brainstorm", border_style="cyan"))
    console.print(f"  Gap report: {gap_report_path}")
    console.print(f"  Output will be saved to: {output_path}\n")

    await execute_and_report(prompt, options, output_path)


def slugify(text: str, fallback: str = "invention") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or fallback


async def async_main(args: argparse.Namespace) -> None:
    if args.trends:
        await run_trend_scan(args.trends)
        return

    if args.gaps:
        await run_gap_finder(args.gaps)
        return

    if args.brainstorm:
        await run_brainstorm(args.brainstorm)
        return

    if args.interview:
        invention_text = await run_interview(args.domain)
        title_hint = invention_text[:40]
    elif args.file:
        invention_text = Path(args.file).read_text()
        title_hint = Path(args.file).stem
    else:
        invention_text = args.text
        title_hint = invention_text[:40]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = REPORTS_DIR / f"{timestamp}-{slugify(title_hint)}.md"

    await run_pipeline(invention_text, output_path, args.domain)


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
    group.add_argument(
        "--trends",
        metavar="TECHNOLOGY_AREA",
        help=(
            "Research the current state of a technology area (filing "
            "activity, leading filers, notable recent patents) independent "
            "of any specific invention, e.g. --trends "
            "\"resistance band training equipment\""
        ),
    )
    group.add_argument(
        "--gaps",
        metavar="TECHNOLOGY_AREA",
        help=(
            "Identify specific functional/use-case gaps existing patents "
            "don't cover well in a technology area (runs a fresh trend "
            "scan internally first for landscape context), e.g. --gaps "
            "\"resistance band training equipment\""
        ),
    )
    group.add_argument(
        "--brainstorm",
        metavar="GAP_REPORT_PATH",
        help=(
            "Generate concrete product/company concepts for the gaps in a "
            "previously generated --gaps report, e.g. --brainstorm "
            "reports/20260713-144400-gap-resistance-band-training-equipment.md"
        ),
    )
    parser.add_argument(
        "--domain",
        choices=[
            "mechanical",
            "software",
            "hybrid",
            "electrical",
            "business-method",
            "chemical-process",
            "pharma",
        ],
        default="mechanical",
        help=(
            "Type of invention. 'software' adds a §101 abstract-idea "
            "eligibility screen alongside the novelty/prior-art screen. "
            "'hybrid' is for inventions with both mechanical and software/"
            "algorithmic elements — it applies the §101 screen only to the "
            "software-flavored elements. 'electrical' is for circuit/"
            "electrical inventions — novelty/obviousness screen only, no "
            "§101 screen, same as 'mechanical'. 'business-method' is for "
            "process/workflow/organizational-technique inventions — adds "
            "the §101 screen since business methods face the same Alice/"
            "Mayo eligibility risk as software, arguably more acutely. "
            "'chemical-process' is for process/method claims (reaction "
            "sequence, conditions, catalyst choice) — novelty/obviousness "
            "screen only, same as 'mechanical'; it does NOT search "
            "chemical structures and is NOT for compound/molecule novelty "
            "questions. 'pharma' adds a structure-search agent that queries "
            "PubChem for exact/similar compound matches — a PARTIAL, "
            "screening-level structural search of one public database, NOT "
            "a comprehensive pharma patent search, and does NOT address "
            "Hatch-Waxman/Orange Book/patent term extension. "
            "Default: mechanical. Ignored when --trends, --gaps, or "
            "--brainstorm is used."
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[yellow]No ANTHROPIC_API_KEY found in the environment[/yellow] - that's "
            "fine if you're authenticated via a Claude Pro/Max subscription login "
            "(`claude login`). This run will use that login instead of a separate "
            "API key.\nIf you intended to use an API key instead, copy .env.example "
            "to .env and add it there.\n"
        )

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
