# Decision Note 

## Decision Supported 

My system tries to understand and expand on unstructured hirer requests.
It tries to extract implicit and explicit requirement signals from the hirer conversation 
into a structured format , that supports the retrieval of recommneded artists and provides 
a basis for tracebacks to the decision behind the recommendation. It provides the user with 
two more questions along with the artist suggestion clarification of which could lead 
to reranking of artists. 

## First version scope 

- Extracting the unstructured artists data from docx provided to a strutured JSON object using standard rule based parsing.
- Re structuring the JSON data into a SQL db for deterministic searchiblity using the hirer conversation 
- Expanding artist ability data using artist media provided (uses LLM and Multimodal models suitable to the media present)
- Building evidence for the expanded artist ability data for tracebility.
- Per artist scoring based on expanded artist ability data the expanded hirer query on a strictly structured 3 dimensional rubric giving out a scalar score for ranking and reranking.
- Explicit declaration of scope change in follow up conversation and reasoning for change in artist ranking.

## Out of Scope 

- Tracing of follow up hirer question accoring to hirer scession or ID , the code for follow up and re ranking right now is hardcoded to point at the expanded query/recommnedation/ evidence document of the first query and follow up conversation .txt file provided.
- Only visually and auditarily verifiable evidence is used to generate artist abilit or artist_intelligence.json file, does not include any meta signals such as professionalism, punctuality etc according to the assignment doc.
- A general geo location system for location or proximity matching for artists based on location is restricted strictly to the names provided in the doc.
- frontend, web scraping, model training, or deployment — per the assignment's scope boundary.
- Comparing candidates to each other inside a single LLM call. Every candidate is scored blind, one call each, specifically to avoid order effects and to keep each score attributable to the candidate's own evidence rather than a relative judgement.

## Category specific capability dimension

Rather than one generic rubric, `Question-generator.py` asks category-specific
"Standardized Benchmark" questions per media type:

- **Musicians** (audio): audio clarity, mixing balance, rhythmic consistency emotional tone, genre/instrumentation.
- **Photographers** (image): focus & sharpness, lighting quality, composition, color palette, stylistic consistency across the portfolio.
- **Video editors** (video): transition quality, audio-visual sync, color continuity, narrative flow, motion graphics integration.
- On top of these, each artist also gets a small set of **bio-verification** questions generated from their own stated bio these are what actually let the system distinguish a *claim* from *demonstrated evidence*
- To keep the LLM calls and costs at a minimum metrics such as `audio_signal_metrics` (BPM, rhythmic intensity, duration) are computed locally with librosa, at zero LLM cost, and used both as evidence in their
own right and as context fed into the VLM/audio-LLM questions.
-At recommendation time, scoring further narrows this to 3 dimensions specific to *this job*, not the artist's category in general:
- `skill_match` = does the evidence support the skills this job needs 
-  `requirement_fit` = do the hirer's stated constraints fit this candidate
-  `evidence_strength` = how strong/verified is the evidence overall 
-  Weighted 0.5 / 0.3 / 0.2 — skill match dominates because it's the most direct predictor of "will this booking go well," fit and evidence strength are real but secondary signals.

## Main Assumptions / risk 

- The artist categories are derived from the folder names 'M' for Musician , 'V'for Video editors and 'P' for Photographers and uses the professions mentioned inside the docx as niche . This is a brittle method and breaks if the folder structure and the data inside do not match. Same follows for artist id's
- The media pipeline search for the word media inside each folder to retrieve evidence for multimodal Q/A which i realized is brittle and breaks at one of the video editors too late could have accounted for that before hand.
- The current extraction of information from the docx to a structured JSON is rule based and is based on observation of the limited examples present in dataset , it is brittle and can break if a new doc does not follow the same structures.
- Some documents have unrelated info in related fields such as in work preference field one doc could have written remote other could have written professional brand work, The system does not take this into the account.
- The extraction of info from hirer converstion for SQL query generation is done by an LLM and can lead to hallucination but regex faces the it's own pitfalls.
- The weighing on judgement calls (0.5/0.3/0.2) is not derived from any evidence backed data and is written basis my own intution.




