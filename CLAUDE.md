# patent-scout

Patentability screening agent pipeline that takes a plain-language invention
description and runs it through a multi-agent screen (novelty elements,
prior-art angles, mechanical alternatives, etc.).

## Rules

- **Always verify `agent.py` compiles after any edit.** Run
  `python -m py_compile agent.py` and fix any errors before considering an
  edit done.
- **Never commit real invention reports.** Only `reports/20260710-204904-sample-invention.md`
  (the sample invention) is allowed in git — everything else under `reports/`
  contains real user inventions and must stay untracked. This is enforced by
  `.gitignore` (`reports/*.md` with an exception for the sample report); don't
  bypass it with `git add -f`.
- **Domain options** are selected via the `--domain` flag (default `mechanical`).
  The valid values are `mechanical`, `software`, `hybrid`, `electrical`,
  `business-method`, `chemical-process`, and `pharma`. The authoritative list is
  the `choices=[...]` list on the `--domain` argument in `agent.py` — if this
  bullet and that list ever disagree, `agent.py` is correct and this file is
  stale. Fix this file; do not "fix" agent.py to match it.
  - `software`, `hybrid`, and `business-method` add a `section-101-screen`
    subagent for Alice/Mayo §101 abstract-idea eligibility risk. `hybrid`
    applies it only to elements the technical-parser tags as
    software/algorithmic.
  - `mechanical`, `electrical`, and `chemical-process` run a novelty/obviousness
    screen only, with no §101 screen. `chemical-process` is for process/method
    claims and does NOT search chemical structures.
  - `pharma` adds a `structure-search` subagent that queries PubChem for
    exact/similar compound matches. This is a screening-level search of one
    public database, not a comprehensive compound novelty determination.
