"""
recommender.py

Loops over hirer conversation files and produces one recommendations
JSON per conversation. Pipeline, matching what we designed:

  1. EXTRACTION (LLM, fixed JSON schema, one call): read the raw
     conversation text (WhatsApp export / phone notes / email thread --
     confirmed all three formats appear in the real dataset) and produce
     a structured "expanded request" object. Category and other
     enumerable fields are constrained to fixed vocab; free-text fields
     stay free text. Every field is tagged stated/assumed/unknown so
     later stages (relaxation, refinement questions) can reason about
     WHY a field has the value it has.

  2. DETERMINISTIC SQL SEARCH (no LLM): query artists.db on the
     structured fields only -- category as a hard filter (never
     relaxed), location as a soft filter relaxed via the small
     NEARBY_CITY_GROUPS lookup already built into make-db.py. Every
     relaxation step is logged into the output, not just applied
     silently.

  3. PER-CANDIDATE, PER-DIMENSION SCORING (LLM, one call per candidate,
     candidates scored BLIND to each other -- never compared in the same
     call). The LLM outputs a 1-5 score PLUS a short reasoning string
     for each fixed rubric dimension; it does not decide the ranking.

  4. RANKING (code, not LLM): a declared, fixed weighted sum of the
     per-dimension scores is computed here, and candidates are sorted by
     that number. This is what makes the ranking reproducible and
     defensible -- "why is A above B" always reduces to arithmetic on
     visible per-dimension scores, never a free-floating LLM opinion.

  5. OUTPUT: recommendations.json with top 2 (or fewer / a stated "no
     plausible match" if the pool is empty even after relaxation),
     each with per-dimension scores + reasoning + weighted total, plus
     up to 2 refinement questions tied to specific assumed/unknown
     fields that could change the ranking.

Re-ranking (the follow-up file) is intentionally a SEPARATE script
(reranker.py) that reuses these same building blocks against a
persisted expanded-request object, rather than being crammed into this
one -- see decision_note.md for why re-ranking is not "treat it as a
new request".
"""

import os
import json
import sqlite3
import warnings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please ensure GEMINI_API_KEY is in your .env file.")
genai.configure(api_key=api_key)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
HIRER_CONVERSATIONS_DIR = REPO_ROOT / "Data-set" / "hirer_conversations"
DB_PATH = REPO_ROOT / "artists.db"
OUTPUT_DIR = REPO_ROOT / "recommendations"

VALID_CATEGORIES = {"musicians", "photographers", "video_editors"}

# Same small, stated location grouping used in make-db.py -- imported
# by re-declaring here rather than importing across the two sibling
# folders, to keep this script runnable standalone. Keep these two
# lists in sync manually; see decision_note.md limitation note.
NEARBY_CITY_GROUPS = {
    "delhi ncr": {"delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
}


# ---------------------------------------------------------
# 1. EXTRACTION: raw conversation text -> structured expanded request
# ---------------------------------------------------------
EXTRACTION_PROMPT_TEMPLATE = """You are extracting a structured hiring request from a raw conversation
between a hirer and support staff. The conversation may be a WhatsApp export, phone-call notes, or an
email thread -- read it for CONTENT regardless of format.

Output ONLY a JSON object (no markdown fences, no preamble) with EXACTLY these keys:

{{
  "category": {{"value": one of ["musicians", "photographers", "video_editors", "unclear"], "source": "stated" or "inferred", "evidence": "short quote or paraphrase"}},
  "location_city": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "budget_max_inr": {{"value": number or null, "source": "stated"/"inferred"/"unknown"}},
  "deadline_or_date": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "key_requirements": {{"value": [list of short strings, each a specific stated constraint or need], "source": "stated"}},
  "skill_signals_needed": {{"value": [list of short strings describing what capability/skill matters for this job], "source": "stated"/"inferred"}},
  "unknowns": [list of short strings -- things NOT resolved by the conversation that would matter for finding the right artist]
}}

Rules:
- "category" must be exactly one of the four listed values. Infer from context if not stated outright
  (e.g. "live music" -> musicians, "reel editor" -> video_editors, "photographer" -> photographers).
  Use "unclear" only if genuinely ambiguous.
- Every value must be traceable to something actually in the conversation. Do not invent specifics
  (exact prices, dates, or requirements) that aren't stated or reasonably implied.
- "source": "stated" means explicitly said; "inferred" means a reasonable read of context, not a direct
  quote; "unknown" means the conversation doesn't resolve it.
- Keep "key_requirements" and "skill_signals_needed" short (a few words each), not full sentences.

Conversation:
{conversation_text}
"""


def extract_request(conversation_text: str) -> dict:
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(conversation_text=conversation_text)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


# ---------------------------------------------------------
# 2. DETERMINISTIC SQL SEARCH WITH LOGGED RELAXATION
# ---------------------------------------------------------
def normalize_city(location_text):
    if not location_text:
        return None
    return location_text.split(",")[0].split("/")[0].strip().lower()


def city_region(city):
    if not city:
        return None
    if city in NEARBY_CITY_GROUPS:
        return city
    for region, members in NEARBY_CITY_GROUPS.items():
        if city in members:
            return region
    return None


def search_candidates(conn, expanded_request: dict) -> tuple[list[dict], list[str]]:
    """
    Returns (candidate_rows, relaxation_log). Category is a hard filter,
    never relaxed -- a musician request should never surface a
    photographer. Location is relaxed in stages: exact city -> region
    match -> no location filter at all. Each relaxation actually applied
    is recorded in relaxation_log so the output stays honest about how
    the candidate pool was found, not just what it ended up being.
    """
    cursor = conn.cursor()
    relaxation_log = []

    category = expanded_request.get("category", {}).get("value")
    if category not in VALID_CATEGORIES:
        return [], [f"category '{category}' is not a recognized category -- no search performed"]

    hirer_city = normalize_city(expanded_request.get("location_city", {}).get("value"))
    hirer_region = city_region(hirer_city)

    # Stage 1: exact city match (if a city was even given)
    if hirer_city:
        cursor.execute(
            "SELECT artist_id, name, data FROM artist_documents WHERE category = ? AND location_city = ?",
            (category, hirer_city),
        )
        rows = cursor.fetchall()
        if rows:
            return [_row_to_dict(r) for r in rows], relaxation_log

        relaxation_log.append(f"no exact city match for '{hirer_city}' -- relaxing to region")

    # Stage 2: region match (e.g. Gurgaon hirer, Delhi-listed artist)
    if hirer_region:
        cursor.execute(
            "SELECT artist_id, name, data FROM artist_documents WHERE category = ? AND location_region = ?",
            (category, hirer_region),
        )
        rows = cursor.fetchall()
        if rows:
            return [_row_to_dict(r) for r in rows], relaxation_log

        relaxation_log.append(f"no match within region '{hirer_region}' -- relaxing location entirely")
    elif hirer_city:
        relaxation_log.append(f"'{hirer_city}' is not in any known region grouping -- relaxing location entirely")

    # Stage 3: category only, location unconstrained
    cursor.execute("SELECT artist_id, name, data FROM artist_documents WHERE category = ?", (category,))
    rows = cursor.fetchall()
    if rows:
        relaxation_log.append("location filter fully relaxed -- results are category-only matches")
    else:
        relaxation_log.append(f"no artists at all found for category '{category}'")

    return [_row_to_dict(r) for r in rows], relaxation_log


def _row_to_dict(row) -> dict:
    artist_id, name, data_json = row
    profile = json.loads(data_json)
    return {"artist_id": artist_id, "name": name, "profile": profile}


# ---------------------------------------------------------
# 3. PER-CANDIDATE, PER-DIMENSION SCORING (LLM scores, never ranks)
# ---------------------------------------------------------
# Fixed rubric. Declared here, not left implicit -- see decision_note.md
# for the reasoning: an LLM asked to freely "rank candidates" is prone
# to order effects and unfalsifiable judgments. Scoring one candidate at
# a time, against a fixed set of named dimensions, keeps every score
# traceable to a specific criterion and comparable across candidates.
SCORING_DIMENSIONS = [
    "skill_match",       # does evidence support the specific skills this job needs?
    "requirement_fit",   # do stated key_requirements/constraints fit this candidate?
    "evidence_strength",  # how strong/verified is the evidence backing their claims overall?
]

# Declared weights -- arbitrary but FIXED and visible, not tuned per
# candidate. Recompute this if the rubric changes; don't hand-adjust
# per output.
SCORING_WEIGHTS = {
    "skill_match": 0.5,
    "requirement_fit": 0.3,
    "evidence_strength": 0.2,
}

SCORING_PROMPT_TEMPLATE = """You are scoring ONE candidate artist against ONE hiring request. You are not
comparing this candidate to any other candidate -- score this candidate alone, on their own merits, against
the request below.

Output ONLY a JSON object (no markdown fences, no preamble) with EXACTLY these keys, one entry per
dimension:

{{
  "skill_match": {{"score": integer 1-5, "reasoning": "one or two sentences citing specific evidence from the candidate profile"}},
  "requirement_fit": {{"score": integer 1-5, "reasoning": "..."}},
  "evidence_strength": {{"score": integer 1-5, "reasoning": "..."}}
}}

Scoring guide (apply consistently): 1 = no supporting evidence / clear mismatch, 3 = partial or
unverified fit, 5 = strong, directly-evidenced fit. Base every score and reasoning ONLY on what's
actually in the candidate profile below (their bio, evaluated multimodal Q&A, signal metrics, vibe
summary) -- do not invent capabilities not evidenced there.

Hiring Request (structured):
{expanded_request_json}

Candidate Profile:
{candidate_profile_json}
"""


def score_candidate(expanded_request: dict, candidate_profile: dict) -> dict:
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
    prompt = SCORING_PROMPT_TEMPLATE.format(
        expanded_request_json=json.dumps(expanded_request, indent=2),
        candidate_profile_json=json.dumps(candidate_profile, indent=2),
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


def compute_weighted_total(dimension_scores: dict) -> float:
    """
    Pure arithmetic, not an LLM call -- this is what makes ranking
    reproducible. Missing a dimension (e.g. a malformed LLM response)
    scores that dimension as 0 rather than silently excluding it from
    the weighted sum, so a scoring failure shows up as a low total
    (investigatable) rather than an inflated one from a smaller
    denominator.
    """
    total = 0.0
    for dim, weight in SCORING_WEIGHTS.items():
        score = dimension_scores.get(dim, {}).get("score", 0)
        total += weight * score
    return round(total, 3)


# ---------------------------------------------------------
# 4. REFINEMENT QUESTIONS: tied to specific assumed/unknown fields
# ---------------------------------------------------------
def build_refinement_questions(expanded_request: dict) -> list[dict]:
    """
    At most 2 questions, each tied to a field that is genuinely
    unresolved (source == "unknown", or "inferred" rather than
    "stated"), with a short note on why it could change the ranking.
    This is deliberately NOT an LLM call -- it's a direct read of the
    "source" tags already produced during extraction, which is exactly
    what those tags are for.
    """
    candidates = []
    for field_name in ["location_city", "budget_max_inr", "deadline_or_date"]:
        field = expanded_request.get(field_name, {})
        if field.get("source") in ("unknown", "inferred"):
            candidates.append({
                "field": field_name,
                "question": f"Can you confirm {field_name.replace('_', ' ')}?",
                "why_it_matters": f"This was '{field.get('source')}', not explicitly stated -- "
                                  f"confirming it could change which candidates are eligible or how they rank.",
            })
    for unknown in expanded_request.get("unknowns", []):
        candidates.append({
            "field": "unknowns",
            "question": f"Can you clarify: {unknown}?",
            "why_it_matters": "Listed as an open unknown during request extraction.",
        })
    return candidates[:2]


# ---------------------------------------------------------
# 5. MAIN PER-HIRER PIPELINE
# ---------------------------------------------------------
def process_hirer_conversation(conversation_path: Path, conn) -> dict:
    conversation_text = conversation_path.read_text(encoding="utf-8")

    print(f"\n{'='*60}\nProcessing: {conversation_path.name}\n{'='*60}")

    print(" -> Extracting structured request from conversation...")
    expanded_request = extract_request(conversation_text)

    print(" -> Running deterministic SQL search...")
    candidates, relaxation_log = search_candidates(conn, expanded_request)
    for entry in relaxation_log:
        print(f"    [relaxation] {entry}")
    print(f"    {len(candidates)} candidate(s) found.")

    scored_candidates = []
    for candidate in candidates:
        print(f" -> Scoring candidate {candidate['artist_id']} ({candidate['name']}) [blind to other candidates]...")
        try:
            dimension_scores = score_candidate(expanded_request, candidate["profile"])
            weighted_total = compute_weighted_total(dimension_scores)
        except Exception as e:
            print(f"    [ERROR] Scoring failed for {candidate['artist_id']}: {e}")
            dimension_scores = {"error": str(e)}
            weighted_total = 0.0

        scored_candidates.append({
            "artist_id": candidate["artist_id"],
            "name": candidate["name"],
            "dimension_scores": dimension_scores,
            "weighted_total_score": weighted_total,
        })

    # Ranking is computed HERE, in code, by sorting on the already-
    # computed weighted total -- the LLM never sees or decides the
    # ranking, only the per-candidate, per-dimension scores above.
    scored_candidates.sort(key=lambda c: c["weighted_total_score"], reverse=True)
    top_candidates = scored_candidates[:2]

    result = {
        "hirer_conversation_file": conversation_path.name,
        "expanded_request": expanded_request,
        "search_relaxation_log": relaxation_log,
        "candidate_pool_size": len(candidates),
        "scoring_weights_used": SCORING_WEIGHTS,
        "recommendations": top_candidates,
        "refinement_questions": build_refinement_questions(expanded_request),
    }

    if not candidates:
        result["no_match_reason"] = (
            "No plausible candidates found for this category even after full location "
            "relaxation. See search_relaxation_log for what was tried."
        )

    return result


if __name__ == "__main__":
    if not HIRER_CONVERSATIONS_DIR.exists():
        print(f"[ERROR] Hirer conversations directory not found: {HIRER_CONVERSATIONS_DIR}")
        exit(1)
    if not DB_PATH.exists():
        print(f"[ERROR] artists.db not found at: {DB_PATH}. Run make-db.py first.")
        exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conversation_files = sorted(HIRER_CONVERSATIONS_DIR.glob("*.txt"))
    if not conversation_files:
        print("No hirer conversation files found.")
        exit(0)

    for conversation_path in conversation_files:
        result = process_hirer_conversation(conversation_path, conn)

        output_filename = conversation_path.stem + "_recommendations.json"
        output_path = OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f" -> Saved {output_path.name}")

    conn.close()
    print(f"\nDone. {len(conversation_files)} recommendation file(s) written to {OUTPUT_DIR}")