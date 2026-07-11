# Patent Scout

A multi-agent patentability screening tool for mechanical inventions. Give it
a plain-language invention description; it breaks the invention into its
distinct technical elements, searches for relevant prior art on each one,
and writes a preliminary patentability screening memo.

Built with the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
using a pipeline of specialized agents:

```
 (optional)         ┌─────────────────────┐
 interactive  ─────► │  invention-interviewer│  → structured invention
 Q&A in terminal    │  (conversational)     │    description
                    └─────────────────────┘
                              │
        ── or a description you provide directly ──
                              │
                              ▼
                    ┌─────────────────────┐
                    │  technical-parser     │  → distinct claimable elements
                    │  (subagent)           │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  prior-art-search     │  → patents / prior art per element
                    │  (subagent, web search)│
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  main agent           │  → synthesized Markdown memo
                    │  (synthesis + write)  │
                    └─────────────────────┘
```

## Why this exists

This started as a way to combine two things: a mechanical engineering
background and an interest in patent law. Instead of doing either
separately, this tool encodes actual domain judgment from both sides —
what counts as a distinct claimable mechanical element, and what counts as
relevant prior art — into an agent pipeline, rather than just wrapping a
single API call in a script.

It is a **screening tool**, not a substitute for a patent attorney. See the
disclaimer section below and in every generated memo.

## Setup

Requires Python 3.10+.

```bash
git clone <this-repo>
cd patent-scout
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Authentication — pick one

**Option A: Claude Pro/Max subscription login (recommended)**

If you have a Claude Pro or Max subscription, this runs inside your existing
plan's usage limits — no separate charge.

```bash
npm install -g @anthropic-ai/claude-code   # installs the Claude Code CLI
claude login                                # opens a browser to log into your Claude account
```

Do this once. After that, don't create a `.env` file or set `ANTHROPIC_API_KEY`
— leave it unset, and the script will use your logged-in session automatically.

Note: Anthropic has changed how subscription-based Agent SDK usage is billed
more than once in 2026 (they announced moving it to a separate metered credit,
then paused that change). Check
[the current policy](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
before assuming this is permanently free within your plan — if you use this
heavily, verify current terms.

**Option B: API key (pay-as-you-go)**

Use this if you don't have a Pro/Max subscription, or prefer usage-based
billing separate from it.

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY from platform.claude.com
```

## Usage

```bash
# From a file
python agent.py examples/sample_invention.txt

# From inline text
python agent.py --text "A ratcheting hinge that locks at 15-degree increments..."

# Interactive interview (no description needed up front)
python agent.py --interview
```

`--interview` launches an **invention-interviewer agent** that asks you
questions in the terminal — one at a time, adapting based on your answers
— covering what problem it solves, how the mechanism actually works, what's
different about it versus existing solutions, and key materials/dimensions.
Type `done` at any point to end early. Once it has enough detail, it writes
a structured invention description itself and automatically hands it off
to the same technical-parser → prior-art-search → memo pipeline described
above — no need to write the description yourself first.

Output is saved as a timestamped Markdown file in `reports/`, e.g.
`reports/20260710-143022-sample-invention.md`.

## Architecture notes (why it's built this way)

- **The interviewer uses a different interface than the other two agents.**
  `technical-parser` and `prior-art-search` are one-shot subagents invoked
  through `query()` — they get a task, do it, return a result. The
  interviewer needs an actual back-and-forth conversation with a human, so
  it's built with `ClaudeSDKClient` instead, which keeps a live session open
  across multiple turns. It's not a subagent in the same pipeline — it runs
  first, standalone, and its output (the invention description) becomes the
  input to the rest of the pipeline exactly as if you'd typed it yourself
  with `--text`.
- **Two subagents instead of one big prompt.** The technical-parsing task
  and the prior-art-search task require different tools (`Read` vs.
  `WebSearch`/`WebFetch`) and different failure modes — parsing errors look
  very different from bad search queries. Isolating them keeps each
  subagent's context focused and makes the pipeline easier to debug and
  extend (e.g. swapping in a real USPTO/Google Patents API later without
  touching the parsing logic).
- **The main agent only orchestrates and synthesizes.** It doesn't do the
  technical parsing or searching itself — it delegates and then writes the
  final memo, which keeps the main conversation's context small even as
  the invention description or search results grow.
- **Read-only, narrowly-scoped tool access per subagent.** `technical-parser`
  only gets `Read`; `prior-art-search` only gets `WebSearch`/`WebFetch`. This
  is the same principle used in production agent design — restrict each
  component to exactly what it needs.

## Scope and known limitations

This tool is scoped to **mechanical/physical inventions** — the
technical-parser and prior-art-search agents were built and tuned around
novelty and obviousness analysis (does prior art already disclose this
mechanism, or make it obvious), which is the right framework for that
category.

**It does not currently handle software/AI inventions correctly.** Software
and AI-based inventions face an additional, earlier legal hurdle that
mechanical inventions don't: **35 U.S.C. §101 and the "abstract idea"
doctrine** (from *Alice Corp. v. CLS Bank*, 2014). A software invention can
be entirely novel and non-obvious and still be unpatentable if it's judged
to be an abstract idea merely implemented on a computer, rather than a
claim tied to a specific technical improvement. This tool's agents don't
reason about that distinction at all — they'll screen a software/AI
invention purely on novelty/obviousness grounds and can produce a
misleadingly optimistic read as a result. This was discovered by running
the tool on a description of itself; see the memo in `reports/` from that
run for a concrete example of the gap.

If extending this tool to cover software inventions, a `section-101`
subagent (or a revised technical-parser) would need to be added
specifically to flag claims that read as applying generic computing/AI
steps to an abstract process, before any novelty search is even run.

## Possible extensions

- Swap general web search for the actual Google Patents / USPTO PatFT API
  for more structured, citable results.
- Add a `claim-drafter` subagent that turns "Likely Novel" elements into
  draft claim language.
- Batch mode: run against a folder of invention disclosures and produce a
  triage ranking across all of them.

## Disclaimer

This tool produces a preliminary, automated screening memo. It is **not
legal advice**, not a formal patentability or freedom-to-operate opinion,
and does not replace a proper prior art search and opinion from a licensed
patent attorney or patent agent. Search coverage depends on what a general
web search surfaces and is not exhaustive of all patent databases or
jurisdictions.
