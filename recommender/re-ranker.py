"""
reranker.py

Handles the follow-up request for a hirer conversation uses previous expanded request and new followup conversation to 
retrieve and rerank candidates.

Pipeline:
  1. Load the EXISTING expanded_request from the prior
     recommendations.json
  2. Read the follow-up text file. Send BOTH the existing expanded
     request and the new text to the LLM, asking it to produce an
     UPDATED expanded request.
  3. Re-run the exact same deterministic search (search_candidates) and
     per-candidate scoring (score_candidate) from recommender.py against
     the updated expanded request. These are imported, not
     reimplemented, so search/scoring logic can never drift between the
     first pass and the re-rank pass.
  4. Diff the OLD ranking (from the input recommendations.json) against
     the NEW ranking computed here -- which candidates moved, which
     appeared/disappeared -- and record that diff explicitly in the
     output.
    5. Save update_recommendation.json in the same shape as
     recommendations.json, plus the added "what_changed" section.
"""

import json
import sqlite3
import warnings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
import recommender

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RECOMMENDATIONS_DIR = REPO_ROOT / "recommendations"
FOLLOW_UP_DIR = REPO_ROOT / "Data-set" / "follow_up_update"
DB_PATH = REPO_ROOT / "artists.db"
OUTPUT_DIR = REPO_ROOT / "recommendations"


# 1. MERGE: existing expanded_request + follow-up text -> updated expanded_request
MERGE_PROMPT_TEMPLATE = """You previously extracted a structured hiring request from a conversation. The
hirer has now sent FOLLOW-UP information. Update the structured request to reflect what's now known --
do not regenerate it from scratch, and do not discard anything from the original that the follow-up
doesn't contradict.

Output ONLY a JSON object (no markdown fences, no preamble), in EXACTLY this same shape as the original:

{{
  "category": {{"value": one of ["musician", "video-editor", "photographer", "unclear"], "source": "stated"/"inferred", "evidence": "..."}},
  "niche": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "location_city": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "budget_max_inr": {{"value": number or null, "source": "stated"/"inferred"/"unknown"}},
  "deadline_or_date": {{"value": string or null, "source": "stated"/"inferred"/"unknown"}},
  "key_requirements": {{"value": [list of short strings], "source": "stated"}},
  "skill_signals_needed": {{"value": [list of short strings], "source": "stated"/"inferred"}},
  "unknowns": [list of short strings]
}}

Rules:
- Any field the follow-up newly confirms should have "source": "stated" now, even if it was
  "inferred" or "unknown" before.
- Any field the follow-up contradicts should be updated to the NEW value, with "source": "stated".
- Any field the follow-up doesn't mention should be carried over UNCHANGED from the original.
- Remove an item from "unknowns" if the follow-up resolves it; you may add new unknowns if the
  follow-up raises new open questions.
- Do not invent values not supported by either the original conversation or the follow-up text.

Original structured request:
{original_request_json}

Follow-up message:
{follow_up_text}
"""


def merge_expanded_request(original_request: dict, follow_up_text: str) -> dict:
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
    prompt = MERGE_PROMPT_TEMPLATE.format(
        original_request_json=json.dumps(original_request, indent=2),
        follow_up_text=follow_up_text,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)

# 2. DIFF: old ranking vs. new ranking, tied to what actually changed
def diff_expanded_requests(old_request: dict, new_request: dict) -> list[dict]:
    """
    Field-by-field comparison. Only reports fields
    whose VALUE changed. Fields moved from inferred to states are 
    noted explicitely as a reason of ranking change.
    """
    changes = []
    simple_fields = ["category", "niche", "location_city", "budget_max_inr", "deadline_or_date"]
    for field_name in simple_fields:
        old_field = old_request.get(field_name, {})
        new_field = new_request.get(field_name, {})
        old_value = old_field.get("value")
        new_value = new_field.get("value")
        if old_value != new_value:
            changes.append({
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "old_source": old_field.get("source"),
                "new_source": new_field.get("source"),
            })
        elif old_field.get("source") != new_field.get("source"):
            changes.append({
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "old_source": old_field.get("source"),
                "new_source": new_field.get("source"),
                "note": "value unchanged, but confidence/source changed",
            })
    return changes


def diff_rankings(old_recommendations: list[dict], new_recommendations: list[dict]) -> dict:
    """
    Compares two ranked top-2 lists by artist_id .
    Reports rank movement, new entrants, and dropped candidates, plus
    the score delta for anyone present in both, so "what changed" is
    always backed by an actual number, not a restated opinion.
    """
    old_ranked_ids = [c["artist_id"] for c in old_recommendations]
    new_ranked_ids = [c["artist_id"] for c in new_recommendations]
    old_scores = {c["artist_id"]: c["weighted_total_score"] for c in old_recommendations}
    new_scores = {c["artist_id"]: c["weighted_total_score"] for c in new_recommendations}

    movements = []
    for artist_id in set(old_ranked_ids) | set(new_ranked_ids):
        old_rank = old_ranked_ids.index(artist_id) + 1 if artist_id in old_ranked_ids else None
        new_rank = new_ranked_ids.index(artist_id) + 1 if artist_id in new_ranked_ids else None
        old_score = old_scores.get(artist_id)
        new_score = new_scores.get(artist_id)

        if old_rank is None:
            status = "new_entrant"
        elif new_rank is None:
            status = "dropped_out"
        elif old_rank != new_rank:
            status = "moved_up" if new_rank < old_rank else "moved_down"
        else:
            status = "unchanged_rank"

        movements.append({
            "artist_id": artist_id,
            "old_rank": old_rank,
            "new_rank": new_rank,
            "old_score": old_score,
            "new_score": new_score,
            "score_delta": (round(new_score - old_score, 3) if (old_score is not None and new_score is not None) else None),
            "status": status,
        })

    # Sort by new_rank (None/dropped last) so the reader sees the new
    # top candidates first, consistent with how recommendations reads.
    movements.sort(key=lambda m: (m["new_rank"] is None, m["new_rank"] or 999))
    return {"candidate_movements": movements}

def process_follow_up(prior_recommendation_path: Path, follow_up_path: Path, conn) -> dict:
    with open(prior_recommendation_path, "r", encoding="utf-8") as f:
        prior_result = json.load(f)

    original_expanded_request = prior_result["expanded_request"]
    follow_up_text = follow_up_path.read_text(encoding="utf-8")

    print(f"\n{'='*60}\nRe-ranking: {prior_recommendation_path.name} + {follow_up_path.name}\n{'='*60}")

    print(" -> Merging follow-up into existing expanded request...")
    updated_expanded_request = merge_expanded_request(original_expanded_request, follow_up_text)

    field_changes = diff_expanded_requests(original_expanded_request, updated_expanded_request)
    for change in field_changes:
        print(f"    [changed] {change['field']}: {change['old_value']!r} -> {change['new_value']!r} "
              f"({change['old_source']} -> {change['new_source']})")

    print(" -> Re-running deterministic SQL search on updated request...")
    candidates, relaxation_log = recommender.search_candidates(conn, updated_expanded_request)
    for entry in relaxation_log:
        print(f"    [relaxation] {entry}")
    print(f"    {len(candidates)} candidate(s) found.")

    scored_candidates = []
    for candidate in candidates:
        print(f" -> Re-scoring candidate {candidate['artist_id']} ({candidate['name']}) [blind to other candidates]...")
        try:
            dimension_scores = recommender.score_candidate(updated_expanded_request, candidate["profile"])
            weighted_total = recommender.compute_weighted_total(dimension_scores)
        except Exception as e:
            print(f"    [ERROR] Scoring failed for {candidate['artist_id']}: {e}")
            dimension_scores = {"error": str(e)}
            weighted_total = 0.0

        scored_candidates.append({
            "artist_id": candidate["artist_id"],
            "name": candidate["name"],
            "dimension_scores": dimension_scores,
            "overall_reasoning": recommender.build_overall_reasoning(dimension_scores),
            "weighted_total_score": weighted_total,
        })

    scored_candidates.sort(key=lambda c: c["weighted_total_score"], reverse=True)
    new_top_candidates = scored_candidates[:2]

    ranking_diff = diff_rankings(prior_result.get("recommendations", []), new_top_candidates)

    result = {
        "hirer_conversation_file": prior_result.get("hirer_conversation_file"),
        "follow_up_file": follow_up_path.name,
        "expanded_request": updated_expanded_request,
        "search_relaxation_log": relaxation_log,
        "candidate_pool_size": len(candidates),
        "scoring_weights_used": recommender.SCORING_WEIGHTS,
        "recommendations": new_top_candidates,
        "what_changed": {
            "expanded_request_field_changes": field_changes,
            "ranking_changes": ranking_diff["candidate_movements"],
        },
    }

    if not candidates:
        result["no_match_reason"] = (
            "No plausible candidates found for this category even after full location "
            "and niche relaxation. See search_relaxation_log for what was tried."
        )

    return result


if __name__ == "__main__":
    if not RECOMMENDATIONS_DIR.exists():
        print(f"[ERROR] Recommendations directory not found: {RECOMMENDATIONS_DIR}. Run recommender.py first.")
        exit(1)
    if not FOLLOW_UP_DIR.exists():
        print(f"[ERROR] Follow-up directory not found: {FOLLOW_UP_DIR}")
        exit(1)
    if not DB_PATH.exists():
        print(f"[ERROR] artists.db not found at: {DB_PATH}. Run make-db.py first.")
        exit(1)

    conn = sqlite3.connect(DB_PATH)

    # "01_cafe_music_update.txt" is matched to a prior recommendation
    # with 01_cafe_music_whatsapp.txt's output.
    follow_up_files = sorted(FOLLOW_UP_DIR.glob("*.txt"))
    if not follow_up_files:
        print("No follow-up files found.")
        exit(0)

    for follow_up_path in follow_up_files:
        # Strip the trailing "_update" to get the shared prefix, e.g.
        # "01_cafe_music_update" -> "01_cafe_music"
        follow_up_prefix = follow_up_path.stem.rsplit("_update", 1)[0]

        matches = [
            p for p in RECOMMENDATIONS_DIR.glob("*_recommendations.json")
            if p.stem.startswith(follow_up_prefix)
        ]
        if not matches:
            print(f"[WARNING] No matching prior recommendation found for follow-up "
                  f"'{follow_up_path.name}' (looked for a file starting with '{follow_up_prefix}'). Skipping.")
            continue
        if len(matches) > 1:
            print(f"[WARNING] Multiple prior recommendations match '{follow_up_path.name}': "
                  f"{[m.name for m in matches]}. Using the first: {matches[0].name}")

        prior_recommendation_path = matches[0]
        result = process_follow_up(prior_recommendation_path, follow_up_path, conn)

        output_filename = "update_recommendation.json"
        output_path = OUTPUT_DIR / "submission-files" / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f" -> Saved {output_path.name}")

    conn.close()
    print(f"\nDone. {len(follow_up_files)} follow-up(s) processed.")