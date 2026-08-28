"""
recommender.py

Loops over hirer conversation files and produces one recommendations
JSON per conversation. Pipeline, matching what we designed:

  1. EXTRACTION (LLM, fixed JSON schema, one call): read the raw
     conversation text (WhatsApp export / phone notes / email thread --
     confirmed all three formats appear in the real dataset) and produce
     a structured "expanded request" object. Category and other
     enumerable fields are constrained to fixed vocab; free-text fields
     stay free text. Every field is tagged stated/inferred/unknown so
     later stages (relaxation) can reason about WHY a field has the
     value it has.

     Category vocabulary here ("musician" / "video-editor" /
     "photographer") is deliberately kept in lockstep with what
     make-db.py now writes into artist_documents.category (derived from
     the artist_profiles filename prefix, M/V/P) -- these two lists have
     to match exactly or the hard category filter below returns nothing.
     This was the root cause of empty candidate pools before: make-db.py
     was writing the docx's own free-text category ("Vocalist &
     Guitarist", "Video Editor", ...) into the `category` column instead
     of a value from this fixed vocabulary, so `WHERE category = ?`
     never matched anything the extraction step could produce.

  2. DETERMINISTIC SQL SEARCH (no LLM): query artists.db.
     - category is the FIRST, hard filter -- always required, never
       relaxed. A musician request should never surface a photographer.
     - location and niche are the SECOND filter, applied together as a
       soft constraint and relaxed in stages (see search_candidates):
       exact city + niche -> exact city -> region + niche -> region ->
       niche only -> category only. The category-only stage is a
       guaranteed floor: as long as any artist of that category exists
       in the DB, the search returns *something* rather than an empty
       list, so scoring/recommendation isn't starved by an over-narrow
       filter. Every relaxation step actually applied is logged into the
       output, not just applied silently.

  3. PER-CANDIDATE, PER-DIMENSION SCORING (LLM, one call per candidate,
     candidates scored BLIND to each other -- never compared in the same
     call). The LLM outputs a 1-5 score PLUS a short reasoning string
     for each fixed rubric dimension; it does not decide the ranking.

  4. RANKING (code, not LLM): a declared, fixed weighted sum of the
     per-dimension scores is computed here, and candidates are sorted by
     that number. This is what makes the ranking reproducible and
     defensible -- "why is A above B" always reduces to arithmetic on
     visible per-dimension scores, never a free-floating LLM opinion. An
     `overall_reasoning` string (code, not LLM) is also assembled per
     candidate by stitching together the per-dimension reasoning, so the
     "why this score" answer is visible at a glance without digging into
     nested dimension_scores.

  5. IMPROVE YOUR MATCHES (LLM, one call, AFTER ranking -- "show results
     before questions"): at most 2 follow-up questions for the hirer,
     each with reasoning tied to specific candidates/scores explaining
     how an answer could materially change the ranking. Grounded in the
     actual expanded_request source tags (stated/inferred/unknown) and
     the actual per-candidate reasoning already produced -- not a
     templated "field is unknown, therefore ask about it" rule. Can
     return zero questions if none genuinely qualify; never invents
     questions to pad the list to 2.

  6. OUTPUT: recommendations.json with top 2 (or fewer / a stated "no
     plausible match" if the pool is empty even after relaxation), each
     with per-dimension scores + reasoning + weighted total, plus the
     improve_your_matches section from step 5.

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

# Must match FILENAME_PREFIX_TO_CATEGORY.values() in make-db.py exactly.
VALID_CATEGORIES = {"musician", "video-editor", "photographer"}

NEARBY_CITY_GROUPS = {
    "delhi ncr": {"delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
}

# Words too generic to usefully narrow a niche LIKE match.
NICHE_STOPWORDS = {
    "and", "or", "the", "a", "an", "for", "with", "of", "in", "on", "to",
    "artist", "creator",
}


# ---------------------------------------------------------
# 1. EXTRACTION: raw conversation text -> structured expanded request
# ---------------------------------------------------------
EXTRACTION_PROMPT_TEMPLATE = """You are extracting a structured hiring request from a raw conversation
between a hirer and support staff. The conversation may be a WhatsApp export, phone-call notes, or an
email thread -- read it for CONTENT regardless of format.

Output ONLY a JSON object (no markdown fences, no preamble) with EXACTLY these keys:

{{
  "category": {{"value": one of ["musician", "video-editor", "photographer", "unclear"], "source": "stated" or "inferred", "evidence": "short quote or paraphrase"}},
  "niche": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "location_city": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "budget_max_inr": {{"value": number or null, "source": "stated"/"inferred"/"unknown"}},
  "deadline_or_date": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "key_requirements": {{"value": [list of short strings, each a specific stated constraint or need], "source": "stated"}},
  "skill_signals_needed": {{"value": [list of short strings describing what capability/skill matters for this job], "source": "stated"/"inferred"}},
  "unknowns": [list of short strings -- things NOT resolved by the conversation that would matter for finding the right artist]
}}

Rules:
- "category" must be exactly one of the four listed values. Infer from context if not stated outright
  (e.g. "live music" -> musician, "reel editor" -> video-editor, "photographer" -> photographer).
  Use "unclear" only if genuinely ambiguous.
- "niche" is a short, specific description of the kind of artist within that category the hirer wants
  (e.g. "wedding photographer", "acoustic vocalist and guitarist", "short-form reel editor",
  "event videographer"). This is matched against each artist's own stated specialty, so keep it
  concrete and a few words long, not a full sentence. Use null if the conversation gives no signal
  beyond the bare category.
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


def niche_keywords(niche_text):
    """
    Break a free-text niche request ("acoustic vocalist and guitarist")
    into a small set of lowercase keywords to LIKE-match against the
    DB's `niche` column (itself free text pulled straight from each
    artist's docx-stated category, e.g. "Vocalist & Guitarist"). This is
    deliberately loose -- it's a candidate-pool filter to cut down how
    many profiles the judging LLM has to score, not a precise search.
    """
    if not niche_text:
        return []
    cleaned = niche_text.replace("&", " ").replace("/", " ").replace(",", " ")
    words = [w.strip().lower() for w in cleaned.split()]
    return [w for w in words if len(w) >= 3 and w not in NICHE_STOPWORDS]


def _row_to_dict(row) -> dict:
    artist_id, name, data_json = row
    profile = json.loads(data_json)
    return {"artist_id": artist_id, "name": name, "profile": profile}


def search_candidates(conn, expanded_request: dict) -> tuple[list[dict], list[str]]:
    """
    Returns (candidate_rows, relaxation_log).

    Filter 1 (hard, always applied): category. A musician request never
    surfaces a photographer or video editor -- this is never relaxed.

    Filter 2 (soft, staged): location AND niche together, relaxed in
    order of specificity until something is found:
      city + niche  ->  city only  ->  region + niche  ->  region only
      ->  niche only (no location)  ->  category only (guaranteed floor)

    The last stage never fails as long as at least one artist of that
    category exists in the DB, so a request that gives no usable
    location/niche signal (or one that doesn't match anything specific)
    still surfaces the right category of artist instead of an empty
    list. Every stage that was tried and came up empty is recorded in
    relaxation_log.
    """
    cursor = conn.cursor()
    relaxation_log = []

    category = expanded_request.get("category", {}).get("value")
    if category not in VALID_CATEGORIES:
        return [], [f"category '{category}' is not a recognized category -- no search performed"]

    hirer_city = normalize_city(expanded_request.get("location_city", {}).get("value"))
    hirer_region = city_region(hirer_city)
    keywords = niche_keywords(expanded_request.get("niche", {}).get("value"))

    def run(extra_where: str, extra_params: list):
        query = "SELECT artist_id, name, data FROM artist_documents WHERE category = ?" + extra_where
        cursor.execute(query, (category, *extra_params))
        return cursor.fetchall()

    def niche_clause():
        if not keywords:
            return "", []
        clause = " AND (" + " OR ".join("niche LIKE ?" for _ in keywords) + ")"
        params = [f"%{kw}%" for kw in keywords]
        return clause, params

    niche_where, niche_params = niche_clause()

    # Stage 1/2: exact city match, with then without the niche filter.
    if hirer_city:
        if keywords:
            rows = run(" AND location_city = ?" + niche_where, [hirer_city] + niche_params)
            if rows:
                return [_row_to_dict(r) for r in rows], relaxation_log
            relaxation_log.append(
                f"no match for category+city '{hirer_city}'+niche {keywords} -- dropping niche filter"
            )

        rows = run(" AND location_city = ?", [hirer_city])
        if rows:
            return [_row_to_dict(r) for r in rows], relaxation_log
        relaxation_log.append(f"no exact city match for '{hirer_city}' -- relaxing to region")

    # Stage 3/4: region match, with then without the niche filter.
    if hirer_region:
        if keywords:
            rows = run(" AND location_region = ?" + niche_where, [hirer_region] + niche_params)
            if rows:
                return [_row_to_dict(r) for r in rows], relaxation_log
            relaxation_log.append(
                f"no match within region '{hirer_region}'+niche {keywords} -- dropping niche filter"
            )

        rows = run(" AND location_region = ?", [hirer_region])
        if rows:
            return [_row_to_dict(r) for r in rows], relaxation_log
        relaxation_log.append(f"no match within region '{hirer_region}' -- relaxing location entirely")
    elif hirer_city:
        relaxation_log.append(f"'{hirer_city}' is not in any known region grouping -- relaxing location entirely")

    # Stage 5: niche only, location unconstrained.
    if keywords:
        rows = run(niche_where, niche_params)
        if rows:
            relaxation_log.append("location filter fully relaxed -- results are category+niche matches")
            return [_row_to_dict(r) for r in rows], relaxation_log
        relaxation_log.append(f"no niche match for {keywords} either -- dropping niche filter")

    # Stage 6: category only -- guaranteed floor, never relaxed further.
    rows = run("", [])
    if rows:
        relaxation_log.append("all location/niche filters relaxed -- results are category-only matches")
    else:
        relaxation_log.append(f"no artists at all found for category '{category}'")
    return [_row_to_dict(r) for r in rows], relaxation_log


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
summary) -- do not invent capabilities not evidenced there. The "reasoning" field is required for every
dimension and must cite something specific from the profile, not a generic statement.

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


def build_overall_reasoning(dimension_scores: dict) -> str:
    """
    Stitches the per-dimension LLM reasoning strings into one summary
    line, in fixed rubric order, each labelled with its score and
    weight, so "why did this candidate get this total" is answerable by
    reading one field instead of three. Pure string formatting -- not an
    LLM call -- so it can never drift from the dimension_scores it's
    summarizing.
    """
    if "error" in dimension_scores:
        return f"Scoring failed: {dimension_scores['error']}"

    parts = []
    for dim in SCORING_DIMENSIONS:
        entry = dimension_scores.get(dim, {})
        score = entry.get("score", "?")
        weight = SCORING_WEIGHTS.get(dim, 0)
        reasoning = entry.get("reasoning", "no reasoning returned")
        parts.append(f"{dim} ({score}/5, weight {weight}): {reasoning}")
    return " | ".join(parts)


# ---------------------------------------------------------
# 4. IMPROVE YOUR MATCHES: up to 2 questions, each with reasoning about
#    material impact on the ranking (LLM call, shown AFTER the shortlist
#    per the brief -- "show results before questions")
# ---------------------------------------------------------
IMPROVE_MATCHES_PROMPT_TEMPLATE = """You already produced a ranked shortlist of candidates for a hiring
request. Your job now is NOT to re-score anyone -- it's to propose at most 2 follow-up questions to ask
the hirer that could plausibly CHANGE this ranking if answered.

You are given:
1. The structured hiring request, where each field is tagged "stated" (the hirer said this directly),
   "inferred" (a reasonable guess from context, not confirmed), or "unknown" (the conversation never
   resolved this).
2. The current top candidates with their per-dimension scores and reasoning.

Propose a question ONLY if you can point to a SPECIFIC way the answer could realistically change who
ranks #1 or #2 -- for example: a field that is "inferred" or "unknown" and where the top candidates'
scores plausibly hinge on what the true answer is, or a gap in the request that caused a candidate's
score to be capped (visible in their reasoning) because information was missing.

Do NOT propose generic clarifying questions just because a field happens to be unconfirmed -- only ask
if getting an answer could concretely move the ranking. If you cannot identify any question that clears
that bar, return an empty list. Never propose more than 2 questions.

Output ONLY a JSON object (no markdown fences, no preamble) with EXACTLY this shape:

{{
  "questions": [
    {{
      "question": "a specific, answerable question to put to the hirer",
      "relates_to_field": "the expanded_request field name this question would resolve, or null if it doesn't map to one of the listed fields",
      "reasoning": "1-2 sentences: which candidate(s)/scores this could change, and how -- be concrete, reference the actual candidates and dimension scores below, not a generic statement"
    }}
  ]
}}

("questions" may be an empty list -- do not pad it to 2 if fewer than 2 genuinely qualify.)

Hiring Request (structured, with stated/inferred/unknown tags):
{expanded_request_json}

Current Top Candidates (with per-dimension scores and reasoning):
{top_candidates_json}
"""


def generate_improve_your_matches(expanded_request: dict, top_candidates: list[dict]) -> list[dict]:
    """
    LLM call, made AFTER ranking (never influences the ranking itself).
    Grounded in the actual expanded_request source tags and the actual
    per-candidate reasoning already produced, so questions are tied to
    something real rather than templated off "any field marked unknown."
    Returns [] (not a crash) if the model call fails or returns
    something unparseable -- an empty "improve your matches" section is
    fine; a broken pipeline run is not.
    """
    # Strip fields the model doesn't need and that would just add noise/
    # cost to the prompt (full candidate profile JSON already informed
    # the dimension scores/reasoning below; no need to resend it here).
    slim_candidates = [
        {
            "artist_id": c["artist_id"],
            "name": c["name"],
            "dimension_scores": c["dimension_scores"],
            "weighted_total_score": c["weighted_total_score"],
        }
        for c in top_candidates
    ]

    model = genai.GenerativeModel("gemini-3.5-flash-lite")
    prompt = IMPROVE_MATCHES_PROMPT_TEMPLATE.format(
        expanded_request_json=json.dumps(expanded_request, indent=2),
        top_candidates_json=json.dumps(slim_candidates, indent=2),
    )
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(response.text)
        questions = parsed.get("questions", [])
        return questions[:2]
    except Exception as e:
        print(f"    [ERROR] Improve-your-matches generation failed: {e}")
        return []


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
            "overall_reasoning": build_overall_reasoning(dimension_scores),
            "weighted_total_score": weighted_total,
        })

    # Ranking is computed HERE, in code, by sorting on the already-
    # computed weighted total -- the LLM never sees or decides the
    # ranking, only the per-candidate, per-dimension scores above.
    scored_candidates.sort(key=lambda c: c["weighted_total_score"], reverse=True)
    top_candidates = scored_candidates[:2]

    # "Improve your matches" is generated AFTER the shortlist and never
    # feeds back into it -- results before questions, per the brief.
    improve_your_matches = []
    if top_candidates:
        print(" -> Generating improve-your-matches questions...")
        improve_your_matches = generate_improve_your_matches(expanded_request, top_candidates)
        print(f"    {len(improve_your_matches)} question(s) proposed.")

    result = {
        "hirer_conversation_file": conversation_path.name,
        "expanded_request": expanded_request,
        "search_relaxation_log": relaxation_log,
        "candidate_pool_size": len(candidates),
        "scoring_weights_used": SCORING_WEIGHTS,
        "recommendations": top_candidates,
        "improve_your_matches": improve_your_matches,
    }

    if not candidates:
        result["no_match_reason"] = (
            "No plausible candidates found for this category even after full location "
            "and niche relaxation. See search_relaxation_log for what was tried."
        )

    return result


def write_recommendations_jsonl(results: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


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

    results = []
    for conversation_path in conversation_files:
        result = process_hirer_conversation(conversation_path, conn)
        results.append(result)

        output_filename = conversation_path.stem + "_recommendations.json"
        output_path = OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f" -> Saved {output_path.name}")

    jsonl_path = REPO_ROOT / "submission-files" / "recommendations.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    write_recommendations_jsonl(results, jsonl_path)
    print(f" -> Saved {jsonl_path.name}")

    conn.close()
    print(f"\nDone. {len(conversation_files)} recommendation file(s) written to {OUTPUT_DIR}")