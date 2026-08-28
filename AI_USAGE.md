# AI Usage

Two separate uses of AI in this submission: AI as a **runtime component of the system**
(Gemini, called by the code itself), and AI as a **coding assistant** during development
(Claude, used interactively while building/debugging). Disclosed separately below.

## AI as a runtime component (part of the system's design)

Two tiers, used for different jobs — this split was deliberate, not incidental:

**Local, open-weight multimodal models (via `transformers`, run locally — no API cost)**
handle the actual evidence-grounded understanding of audio/image/video, where the model
needs to look at real media:
- **Qwen2-Audio-7B-Instruct** (`Multimodal/music-processor.py`) — answers the
  standardized-benchmark and bio-verification questions against pooled audio evidence
  (all of an artist's audio files sent together for one question) for musicians.
- **Qwen3-VL-4B-Instruct** (`Multimodal/music-processor.py`,`Multimodal/image-processor.py`, `Multimodal/video-processor.py`) — answers video-evidence
  questions for musicians who also have video media (e.g. confirming a claimed solo
  live performance visually).answers image-evidence questions for photographers, and video-evidence questions
  (transitions, pacing, color continuity, narrative flow) for video editors, against
  scene-cut-sampled frames.

Running these locally rather than through a hosted API kept the actual heavy
lifting — dozens of media-grounded Q&A calls per artist across 15 artists — off the
₹300 paid-API cap entirely, while still using models genuinely capable of audio/video
understanding rather than approximating it with text-only prompting.

**Gemini (`gemini-3.5-flash-lite`, via `google-generativeai`, hosted API)** handles
everything that's read/judgement over text rather than raw media:
- `Question-generator.py` — generating bio-verification questions per artist.
- `artist-summary.py` — synthesizing each artist's 2–3 sentence vibe summary from their
  full evaluated profile.
- `recommender.py` — extracting the structured hiring request from raw conversation
  text, scoring each candidate per-dimension (one call per candidate, blind to other
  candidates), and generating up to 2 follow-up questions.
- `re-ranker.py` — merging a follow-up message into the existing structured request.

This split is also why the reproducible parts of the pipeline (SQL filtering, the
location-relaxation staging, weighted score arithmetic, ranking, diffing old vs. new
rankings) are plain code with no model call at all, local or hosted — anything that
doesn't need judgement doesn't get one, so it can't silently vary between runs.

Approximate spend: Used Gemini within free API limits. Local Qwen inference had no
marginal API cost — run on Nvidia A6000 48 gb VRAM using 4 bit Quantization.

## AI as a coding assistant (Claude, used interactively during development)

Claude (Anthropic) was used conversationally throughout development to:
- Diagnose why `make-db.py` was producing empty search results: traced to the DB's
  `category` column being populated from the profile JSON's own free-text `category`
  field rather than a value from the fixed taxonomy the search filters on.
- Diagnose a silent data-loss bug: the DB's `artist_id` primary key was trusted from the
  profile JSON, but two profiles' JSON-internal `artist_id` values collided
  (`V03_Rahul_Gupta.json` and `VO5_Roshan.json` both internally claim `"V03"`), so one
  silently overwrote the other on insert (confirmed empirically: 15 profile files in, 14
  DB rows out).
- Diagnose a generation-quality bug affecting photographer and video-editor evidence:
  `image-processor.py` / `video-processor.py` load `Qwen3-VL-4B-Thinking` but cap output
  at `max_new_tokens=256`, so the model's reasoning trace consumes the whole budget
  before reaching a conclusion, and the raw, truncated trace was being stored as the
  answer instead of a real one. Found by reading actual generated profile JSON for one
  artist per category, not by code review alone.
- Implement the staged location/niche relaxation logic in
  `search_candidates()` (exact city+niche → exact city → region+niche → region →
  niche-only → category-only guaranteed floor), so a search never returns zero
  candidates for a valid category.
- Implement the "improve your matches" follow-up-question feature: an LLM
  call made *after* ranking , instructed to only propose a
  question when it can point to a concrete way the answer could change who ranks #1/#2,
  rather than generically asking about any unconfirmed field.
