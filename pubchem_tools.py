"""
PubChem PUG REST integration — structural/compound search for the
--domain pharma pipeline.

PUG REST is a free, public, unauthenticated API (no API key). Verified
capabilities: exact lookup by name/SMILES/InChIKey/CID, and synchronous
2D-similarity search via the "fast*" endpoints (no async ListKey polling
needed, unlike the classic similarity/substructure endpoints). Every
response carries an X-Throttling-Control header (Green/Yellow/Red);
PubChem's long-documented policy is roughly 5 requests/sec, 400
requests/min, 300s running time/min — not independently re-verified as
text (the docs site is a JS-rendered SPA that doesn't expose body text to
fetch tools), but the throttling header mechanism itself was confirmed
live.

IMPORTANT SCOPE LIMIT: PubChem is a large public compound database, not a
comprehensive registry of every compound ever disclosed — it does not
reliably cover compounds disclosed only in patent claims/examples that
were never deposited, non-English-language literature, or very recent
filings. A "no match" result from these tools means "not found in
PubChem, in this search," never "novel." See the --domain pharma
disclaimer in agent.py for the full caveat surfaced in every memo this
domain produces.

Integration notes for the AgentDefinition that uses these tools:
- Register PUBCHEM_MCP_SERVERS at the top-level ClaudeAgentOptions.mcp_servers,
  NOT inside an AgentDefinition's own mcpServers field — a live SDK MCP
  server config contains a Server instance that isn't JSON-serializable,
  and AgentDefinition gets serialized over the control channel.
- The namespaced tool names in PUBCHEM_TOOL_NAMES must appear in BOTH the
  subagent's own `tools=[...]` AND the top-level ClaudeAgentOptions
  `allowed_tools` — the subagent's tools list alone is not sufficient to
  grant permission.
"""

import urllib.parse
from typing import Annotated

import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server, AgentDefinition

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "patent-scout-structure-search/1.0 (screening research tool)"
LOOKUP_PROPERTIES = "MolecularFormula,CanonicalSMILES,InChIKey,IUPACName,MolecularWeight"


def _throttling_note(resp: httpx.Response) -> str:
    status = resp.headers.get("X-Throttling-Control", "")
    if status and "Green" not in status:
        return f"\n\n[PubChem throttling status: {status} — further searches may be rate-limited soon.]"
    return ""


@tool(
    "pubchem_lookup",
    "Exact compound lookup in PubChem by name, SMILES, InChIKey, or CID. "
    "Returns CID, molecular formula, canonical SMILES, InChIKey, IUPAC "
    "name, and molecular weight if an exact match is found. Does NOT tell "
    "you whether a compound is novel — only whether PubChem already has "
    "an entry for it.",
    {
        "query": str,
        "namespace": Annotated[str, "One of: name, smiles, inchikey, cid"],
    },
)
async def pubchem_lookup(args: dict) -> dict:
    query = args["query"]
    namespace = args.get("namespace", "name")
    if namespace not in ("name", "smiles", "inchikey", "cid"):
        return {
            "content": [{
                "type": "text",
                "text": f"Invalid namespace '{namespace}'. Must be one of: name, smiles, inchikey, cid.",
            }],
            "is_error": True,
        }

    encoded_query = urllib.parse.quote(query, safe="")
    url = f"{PUG_BASE}/compound/{namespace}/{encoded_query}/property/{LOOKUP_PROPERTIES}/JSON"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as exc:
        return {
            "content": [{"type": "text", "text": f"Network error calling PubChem: {exc}"}],
            "is_error": True,
        }

    if resp.status_code == 404:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"No exact match found in PubChem for {namespace}='{query}'. "
                    "This means not found IN PUBCHEM, in this search — not that "
                    "the compound is novel. PubChem does not have comprehensive "
                    "coverage of every compound ever disclosed."
                ),
            }],
        }

    if resp.status_code != 200:
        return {
            "content": [{"type": "text", "text": f"PubChem error {resp.status_code}: {resp.text}"}],
            "is_error": True,
        }

    return {"content": [{"type": "text", "text": resp.text + _throttling_note(resp)}]}


@tool(
    "pubchem_similarity_search",
    "2D structural similarity search in PubChem (Tanimoto, synchronous "
    "'fast' endpoint) given a SMILES string. Returns CIDs and basic "
    "properties of structurally similar compounds already in PubChem. "
    "This is a SCREENING-LEVEL check against one public database, not a "
    "comprehensive novelty determination.",
    {
        "smiles": str,
        "threshold": Annotated[int, "2D Tanimoto similarity threshold, 0-100. Default 90."],
        "max_records": Annotated[int, "Maximum number of matches to return. Default 10."],
    },
)
async def pubchem_similarity_search(args: dict) -> dict:
    smiles = args["smiles"]
    threshold = args.get("threshold", 90)
    max_records = args.get("max_records", 10)
    encoded_smiles = urllib.parse.quote(smiles, safe="")
    cids_url = (
        f"{PUG_BASE}/compound/fastsimilarity_2d/smiles/{encoded_smiles}/cids/JSON"
        f"?Threshold={threshold}&MaxRecords={max_records}"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            cids_resp = await client.get(cids_url, headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as exc:
        return {
            "content": [{"type": "text", "text": f"Network error calling PubChem: {exc}"}],
            "is_error": True,
        }

    if cids_resp.status_code == 404:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"No compounds found in PubChem within a {threshold}% 2D "
                    "Tanimoto similarity threshold of this structure. This is a "
                    "screening-level result against one public database, not a "
                    "comprehensive novelty determination — absence of a similar "
                    "match here does not mean the compound is novel."
                ),
            }],
        }

    if cids_resp.status_code != 200:
        return {
            "content": [{"type": "text", "text": f"PubChem error {cids_resp.status_code}: {cids_resp.text}"}],
            "is_error": True,
        }

    try:
        cids = cids_resp.json()["IdentifierList"]["CID"]
    except (KeyError, ValueError):
        return {
            "content": [{"type": "text", "text": f"Unexpected PubChem response shape: {cids_resp.text}"}],
            "is_error": True,
        }

    if not cids:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"No compounds found in PubChem within a {threshold}% 2D "
                    "Tanimoto similarity threshold of this structure "
                    "(screening-level result, not a novelty determination)."
                ),
            }],
        }

    cid_list = ",".join(str(c) for c in cids)
    props_url = f"{PUG_BASE}/compound/cid/{cid_list}/property/MolecularFormula,CanonicalSMILES,IUPACName/JSON"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            props_resp = await client.get(props_url, headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as exc:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Found {len(cids)} similar CID(s) in PubChem ({cid_list}) but "
                    f"could not fetch their properties due to a network error: {exc}"
                ),
            }],
            "is_error": True,
        }

    if props_resp.status_code != 200:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Found {len(cids)} similar CID(s) in PubChem ({cid_list}) but "
                    f"property lookup failed: {props_resp.status_code} {props_resp.text}"
                ),
            }],
        }

    return {"content": [{"type": "text", "text": props_resp.text + _throttling_note(props_resp)}]}


PUBCHEM_MCP_SERVER = create_sdk_mcp_server(
    name="pubchem",
    tools=[pubchem_lookup, pubchem_similarity_search],
)

PUBCHEM_MCP_SERVERS = {"pubchem": PUBCHEM_MCP_SERVER}

PUBCHEM_TOOL_NAMES = [
    "mcp__pubchem__pubchem_lookup",
    "mcp__pubchem__pubchem_similarity_search",
]

STRUCTURE_SEARCH = AgentDefinition(
    description=(
        "Searches PubChem (a public compound database) for exact and "
        "structurally-similar matches to a specific compound/molecule "
        "described in a pharma invention. Use this for compound-structure "
        "elements only, never for process/method elements — this is a "
        "PARTIAL, screening-level structural search of ONE public "
        "database, not a comprehensive compound novelty determination."
    ),
    prompt="""You are a chemical structure search assistant supporting a patentability
screen. You will be given a specific compound or molecule (a name, and/or
a SMILES string if the invention description provides one).

1. Exact match: use pubchem_lookup to check whether PubChem already has an
   entry exactly matching this compound (by name if that's what you have,
   otherwise by SMILES/InChIKey if given). Report the CID and identifying
   data if found.

2. Similarity: if you have or can derive a SMILES string for the compound,
   use pubchem_similarity_search to check for structurally similar
   compounds already in PubChem (threshold 90% 2D Tanimoto unless told
   otherwise). Report what you find, including CIDs and basic identifying
   data for each match.

3. Be precise about what a result does and does not mean:
   - An exact match in PubChem means this compound (or something with an
     identical registered structure) is already a known, catalogued
     compound.
   - A similarity-search hit means something structurally CLOSE already
     exists — not identical, but potentially relevant prior art for an
     obviousness argument.
   - NO match (exact or similar) means "not found in PubChem, in this
     search" — NEVER state or imply this means the compound is novel.
     PubChem is a large public compound database, not a comprehensive
     registry of every compound ever disclosed (it does not reliably
     cover compounds disclosed only in patent claims/examples that were
     never deposited, non-English literature, or very recent filings).

4. If a tool call reports an error (network failure, invalid input, or a
   PubChem-side error), report that plainly as a limitation of this
   search — do not silently drop it or imply the search succeeded.

Output your findings as a short labeled section — this is intermediate
research output for another agent to synthesize into a report, not a
final polished document. Do not draw a novelty conclusion yourself; state
findings only.""",
    tools=PUBCHEM_TOOL_NAMES,
    model="sonnet",
)
