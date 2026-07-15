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
           (--domain software or hybrid — hybrid only screens
              the software/algorithmic elements)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  section-101-screen   │  → eligibility risk per element
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
                    │  citation-verifier    │  → independently re-checks every
                    │  (subagent, web search)│    patent #/case/date/assignee cited
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
# From a file (mechanical invention, the default domain)
python agent.py examples/sample_invention.txt

# From inline text
python agent.py --text "A ratcheting hinge that locks at 15-degree increments..."

# Interactive interview (no description needed up front)
python agent.py --interview

# Software/AI invention — adds a §101 abstract-idea eligibility screen
python agent.py --interview --domain software
python agent.py --domain software --text "A system that ..."

# Hybrid invention (mechanical + software, e.g. IoT/smart hardware) —
# runs the §101 screen only on the software-flavored elements
python agent.py --interview --domain hybrid

# Electrical/circuit invention — novelty/obviousness screen only, same as mechanical
python agent.py --domain electrical --text "A current-mode feedback amplifier topology that..."

# Business-method invention — adds the same §101 screen software gets
python agent.py --domain business-method --text "A two-sided auction matching process that..."

# Chemical-process invention (reaction sequence/conditions/catalyst) — NOT for
# compound/molecule novelty questions, process/method novelty only
python agent.py --domain chemical-process --text "A three-step catalytic reduction process run at..."

# Pharma/compound invention — adds a PubChem structural search, but this is
# PARTIAL coverage only, not a comprehensive pharma patent search
python agent.py --domain pharma --text "A novel substituted pyrazole compound (SMILES: ...) for..."

# Industry trend research — independent of any specific invention
python agent.py --trends "resistance band training equipment"

# Patent gap-finding — specific functional gaps in a technology area
python agent.py --gaps "resistance band training equipment"

# Idea brainstorm — concrete concepts for gaps a --gaps report already found
python agent.py --brainstorm reports/20260713-160000-gap-resistance-band-training-equipment.md
```

`--domain` accepts `mechanical` (default), `software`, `hybrid`,
`electrical`, `business-method`, `chemical-process`, or `pharma`:

- **`mechanical`** — novelty/obviousness screen only.
- **`software`** — adds `section-101-screen`, screening every element for
  35 U.S.C. §101 "abstract idea" risk under the Alice/Mayo framework — the
  eligibility hurdle software/AI inventions face that mechanical
  inventions don't.
- **`hybrid`** — for inventions combining physical and software elements
  (e.g. a smart device with an embedded control algorithm). The
  technical-parser tags each element by field, and only the
  software/algorithmic (or mixed) elements go through the §101 screen —
  purely mechanical elements are screened for novelty only, since
  eligibility questions don't apply to them.
- **`electrical`** — for circuit/electrical inventions (circuit topology,
  component arrangement, signal processing method, PCB layout technique,
  etc). Novelty/obviousness screen only, wired exactly like `mechanical`
  (`technical-parser` + `prior-art-search` + `citation-verifier`, no
  `section-101-screen`) — electrical inventions face the same kind of
  novelty/obviousness questions mechanical inventions do, not the §101
  abstract-idea eligibility hurdle.
- **`business-method`** — for process/workflow/organizational-technique
  inventions. Wired exactly like `software`, reusing `section-101-screen`
  as-is: business-method claims face the same Alice/Mayo abstract-idea
  eligibility risk software claims do — arguably more acutely, since
  business-method patents get hit by Alice rejections constantly. This is
  the same subagent, not a separate business-method-specific eligibility
  screener.
- **`chemical-process`** — for **process/method claims only**: a specific
  sequence of reaction steps, conditions (temperature, pressure, time,
  solvent, atmosphere), and/or catalyst choice. Wired exactly like
  `mechanical` (`technical-parser` + `prior-art-search` +
  `citation-verifier`, no `section-101-screen` — process claims aren't a
  §101 concern the way software claims are).
  **This is explicitly NOT for compound/molecule novelty questions.** It
  does not perform any chemical structural search (no SciFinder, Reaxys,
  PubChem structural-similarity search, or CAS Registry access) and cannot
  tell you whether a specific molecule is itself novel — only whether the
  *process* (the reaction sequence/conditions/catalyst) looks novel. Every
  memo this domain produces states this limitation explicitly in its
  Disclaimer section, and flags it in the Invention Summary too if the
  description centers on a specific compound. If your actual question is
  "is this molecule new," use `pharma` instead (below) — though read its
  caveats carefully, since even `pharma`'s structural search is partial.
- **`pharma`** — for pharmaceutical/compound inventions where the actual
  question is whether a specific COMPOUND is novel, not just a process.
  Adds a **`structure-search`** subagent that queries
  [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (a free, public compound
  database — no API key required) via a custom tool integration for exact
  and 2D-similarity structural matches, alongside `technical-parser` +
  `prior-art-search` + `citation-verifier`. No `section-101-screen` (a
  compound/process claim isn't a §101 concern the way software is).
  **This is explicitly a PARTIAL structural search, not a comprehensive
  pharma patent search.** PubChem is a large public compound database, not
  a registry of every compound ever disclosed — it doesn't reliably cover
  compounds disclosed only in patent claims/examples that were never
  deposited, non-English literature, or very recent filings. A "no
  similar compound found" result means exactly that — not found in
  PubChem, in this search — never "novel." This domain also does **NOT**
  address the Hatch-Waxman Act, FDA Orange Book listings, ANDA/Paragraph
  IV exposure, or patent term extension (PTE) — all entirely out of
  scope. Every `pharma` memo states this in a more prominent disclaimer
  than any other domain (both in the `**Status:**` line at the top and in
  full in the Disclaimer section), because the cost of a chemist or
  attorney over-trusting a "no match" result here is real.

The resulting memo reports both an eligibility read (Likely Eligible /
Likely Abstract Idea Risk / Borderline) and the usual novelty read per
element for `software`/`hybrid`/`business-method`, since an element can be
novel but ineligible, or eligible but already anticipated — they're
independent questions. `mechanical`/`electrical` memos report novelty only.

`--interview` launches an **invention-interviewer agent** that asks you
questions in the terminal — one at a time, adapting based on your answers
— covering what problem it solves, how the mechanism actually works, what's
different about it versus existing solutions, and key materials/dimensions
(or architecture/algorithm details, for software). Type `done` at any point
to end early. Once it has enough detail, it writes a structured invention
description itself and automatically hands it off to the pipeline above —
no need to write the description yourself first.

Output is saved as a timestamped Markdown file in `reports/`, e.g.
`reports/20260710-143022-sample-invention.md`.

`--trends "<technology area>"` runs a separate, standalone pipeline that
doesn't take an invention description at all — it researches the current
state of a technology area using an **industry-trend-scanner** subagent:
recent patent filing activity, which companies/institutions file most in
the space, and notable recent patents/applications. Useful before
describing an invention, to get a sense of how crowded or active a space
is. `--domain` is ignored in this mode. Output is saved alongside invention
memos in `reports/`, distinguished by a `-trend-` filename segment, e.g.
`reports/20260713-153000-trend-resistance-band-training-equipment.md`.

`--gaps "<technology area>"` runs a **patent-gap-finder** subagent, a step
beyond trend research: it identifies specific functional or use-case gaps
that existing patents in the area don't cover well. It first runs a fresh
industry-trend-scanner pass internally for landscape context (no need to
run `--trends` separately first), then checks candidate gaps against
actual searches before reporting them — each gap in the report states what
was searched and why it looks uncovered, not just an assertion that "there's
room here." `--domain` is ignored in this mode. Output is saved alongside
the other reports in `reports/`, distinguished by a `-gap-` filename
segment, e.g.
`reports/20260713-160000-gap-resistance-band-training-equipment.md`.

`--brainstorm <path to a --gaps report>` runs an **idea-brainstormer**
subagent that takes the gaps a `--gaps` run already identified and
proposes a handful of concrete product/company concepts to fill them,
each with a paragraph tying it back to the specific gap it addresses. This
is the one command that takes a file path rather than free text — it
doesn't re-run trend research or gap-finding, it builds directly on a
report you already have, since brainstorming concepts is synthesis over
gaps already found rather than new research. `--domain` is ignored in
this mode. Output is saved alongside the other reports in `reports/`,
distinguished by a `-brainstorm-` filename segment, e.g.
`reports/20260713-170000-brainstorm-resistance-band-training-equipment.md`.

### Citation verification (every command)

Every command above ends with a **citation-verifier** pass before its
memo/report is written. It's given only the discrete factual citations
pulled out of the other subagents' findings for that run — patent/
application numbers, case names, dates, assignee/company names — and it
must independently search for and find each one itself; it's explicitly
instructed not to just judge whether a citation looks plausible or
re-trust the claim it was handed. Each citation gets one of three
verdicts: **Confirmed** (found, matches what was claimed), **Could Not
Verify** (no matching result found), or **Mismatch** (something real
exists under that identifier, but it describes something different than
claimed). Every memo/report includes a "Citation Verification" section
listing all of them, and any citation that isn't Confirmed is flagged
inline wherever it's used, e.g. `[UNVERIFIED — see Citation Verification
section]` — never silently dropped. This exists because a hallucinated or
mismatched patent number is exactly the kind of error that's easy to miss
in an otherwise well-written memo and expensive to act on.

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
- **`citation-verifier` never sees the other subagents' search results,
  only their conclusions.** It's handed the extracted citations and claims,
  not the calling agent's search transcript, and is explicitly told not to
  treat a well-formatted claim as verification of itself. It has to run its
  own fresh `WebSearch`/`WebFetch` calls to independently find each
  citation — the whole point is a check that doesn't inherit the first
  search's blind spots.
- **`structure-search` (pharma) uses a custom tool, not `WebFetch`, because
  PubChem's own docs site is a JS-rendered SPA that page-fetching can't
  read, and because structural search needs a real API call (exact/
  similarity lookup), not a web page.** `pubchem_tools.py` wraps PubChem's
  PUG REST endpoints as SDK MCP tools (`@tool` + `create_sdk_mcp_server`)
  that call the API directly with `httpx` and hand back clean JSON — the
  agent never has to construct a PUG REST URL itself. The MCP server is
  registered at the top-level `ClaudeAgentOptions.mcp_servers`, not inside
  the subagent's own definition — a live in-process MCP server holds a
  `Server` object that isn't JSON-serializable, and subagent definitions
  get serialized over the control channel to the CLI subprocess.

## Scope and known limitations

This tool supports seven domains, selected with `--domain`:

- **`mechanical`** (default) — screens for novelty/obviousness only, which
  is the right framework for physical/mechanical inventions.
- **`software`** — adds a `section-101-screen` agent that screens for 35
  U.S.C. §101 "abstract idea" eligibility risk (from *Alice Corp. v. CLS
  Bank*, 2014) in addition to novelty/obviousness. Software and AI
  inventions face this additional, earlier legal hurdle that mechanical
  inventions don't: a software invention can be entirely novel and still be
  unpatentable if it's judged to be an abstract idea merely implemented on
  a computer, rather than a claim tied to a specific technical improvement.
- **`hybrid`** — for inventions combining physical and software elements
  (e.g. a smart device with an embedded control algorithm). The
  technical-parser tags each element by field, and only the
  software/algorithmic (or mixed) elements go through the §101 screen —
  purely mechanical elements are screened for novelty only, since
  eligibility questions don't apply to them.
- **`electrical`** — for circuit/electrical inventions. Wired exactly like
  `mechanical` (novelty/obviousness only, no `section-101-screen`):
  electrical inventions face the same kind of novelty/obviousness
  questions mechanical inventions do, not the §101 abstract-idea hurdle.
- **`business-method`** — for process/workflow/organizational-technique
  inventions. Wired exactly like `software`, reusing the same
  `section-101-screen` agent unmodified: business methods face the same
  Alice/Mayo abstract-idea eligibility risk software does — arguably more
  acutely, since business-method claims are hit by Alice rejections
  constantly. There is no separate business-method-specific eligibility
  screener; it's deliberately the identical §101 check software gets.
- **`chemical-process`** — for process/method claims only (reaction
  sequence, conditions, catalyst choice). Wired exactly like `mechanical`
  (novelty/obviousness only, no `section-101-screen` — process claims
  aren't a §101 concern the way software claims are). **Explicitly not for
  compound/molecule novelty questions.** This domain has no access to
  chemical structural search (no SciFinder, Reaxys, PubChem
  structural-similarity search, or CAS Registry) and cannot tell you
  whether a specific molecule is novel — only whether the process is.
  Every `chemical-process` memo states this limitation explicitly in its
  Disclaimer, and flags it in the Invention Summary if the description
  centers on a specific compound rather than (or in addition to) a
  process.
- **`pharma`** — for pharmaceutical/compound inventions where compound
  novelty is the actual question. Adds a `structure-search` agent
  (`technical-parser` → `structure-search` → `prior-art-search` →
  `citation-verifier`, no `section-101-screen`) that queries PubChem via a
  custom tool integration (`pubchem_tools.py`) for exact and 2D-similarity
  structural matches. **This is a PARTIAL structural search of one public
  database, not a comprehensive pharma patent search**, and it does
  **NOT** address the Hatch-Waxman Act, FDA Orange Book listings,
  ANDA/Paragraph IV exposure, or patent term extension — all entirely out
  of scope. Every `pharma` memo carries the strongest, most prominent
  disclaimer of any domain in this tool, both up top in the `**Status:**`
  line and in full in the Disclaimer section.

The software/§101 path (used by `software`, `hybrid`, and `business-method`)
is newer and less battle-tested than the mechanical path — it was added
after running the tool on a description of itself and finding it gave a
misleadingly optimistic read on a software invention by only checking
novelty/obviousness. See `reports/` for that original test memo as a
concrete before/after reference. §101 eligibility outcomes are also
unusually sensitive to exact claim drafting in ways an automated screen can
only approximate — treat the eligibility read as a starting point for an
attorney conversation, not a verdict. The `hybrid` domain in particular
relies on the main agent correctly sorting elements by field tag before
routing them to §101 screening; misclassification there would silently
skip or over-apply the eligibility check. The `electrical`,
`business-method`, `chemical-process`, and `pharma` domains are the newest
and least battle-tested paths in the tool; `business-method` inherits all
of `section-101-screen`'s existing caveats without any business-method-
specific tuning, and `chemical-process`'s scope limit (process novelty
only, never compound novelty) depends on the main agent and
technical-parser consistently honoring that boundary rather than any
hard enforcement — misclassifying a compound-novelty question as a
process question would silently produce a memo that looks like a
novelty screen but doesn't answer the question actually being asked.
`pharma`'s `structure-search` step was validated with a standalone smoke
test (a real PubChem lookup through the exact subagent+custom-tool
wiring used in the pipeline) but has not yet been run through the full
`--domain pharma` pipeline end-to-end — see the note in `pubchem_tools.py`
for the PUG REST integration details and known gaps (exact rate-limit
numbers not independently re-verified this session; only `fastsimilarity_2d`
similarity search is wired up, not substructure search).

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
