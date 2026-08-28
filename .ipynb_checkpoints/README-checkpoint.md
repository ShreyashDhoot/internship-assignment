# Artist Intelligence and recommendation - README 

- This file covers approach, setup, media selection, implemented choices.
- for limitations check `decision_note.md`.
- Find the required submission file in `submission_files\`

## Approach + Media Selection + implementation choices 
"All three are interlinced and thus are explained simultaneously"

**1. Artist Intelligence :**
- Run `data_pipeline.py` 
- The script runs `processing_dataset/docx-to-json.py` ->`processing_dataset/Question-generator.py`-> `Multimodal/music-processor.py` -> `Multimodal/video-processor.py` -> `Multimodal/image-processor.py` -> `processing_dataset/artist-summary.py` -> `processing_dataset/make-db.py` -> `processing_dataset/artist-profiles-to-jsonl.py` in order to complete the full data processing.
- `processing_dataset/docx-to-json.py` - Parses all the profile.docx provided and forms a structured JSON for each artist using rule based parsing , folder name and folder structure and the info provided inside the doc.
- `processing_dataset/Question-generator.py` - Uses 'Gemini-3.5-flash-lite' LLM to generate 5 common artistic endavour based Questions (like 5 common questions to judge music for each musician etc) and 5 profile based questions to verify + extract artistic ability plus overall vibe of the artist using the media evidence provided.
-  `Multimodal/music-processor.py` - looks at each file in the media folder for musicians. If the file is .mp4 extracts audio concatnates audio to other .mp3 and .wav files present in the folder. Extracts information such as bpm , rythemic or dense , rhythemic intensity using basic signal processing. Sends the Audio file to 'Qwen2-Audio-7B-Instruct' model to answer the Questions from the artist-intelligence documents produced by 'Gemini-3.5-flash-lite'. The .mp4 video files are sent to 'Qwen3-VL-4B-Instruct' for answering video based question in the artist intelligence document as music performance has both audio and visual aspects.
- `Multimodal/image-processor.py` - Uses the images inside the media folder for photographers and 'Qwen3-VL-4B-Instruct' to answer the question for the artist-intelligence JSON file. 
- `Multimodal/video-processor.py` - Uses signal processing to identify the key frames around the cuts and extracts frames around these to feed to 'Qwen3-VL-4B-Instruct' instead of feeding whole videos at once we give an upper bound for the amount of frames the model and our GPU can process and extract those many clips proportionately to the length of the video concanate them and send to the model. This helps us answer Questions across the artist portfolio rather than just a few video , extract editing style and save money on LLM calls.
- All model perform a free pass trying to extract artist abilities from media files that do not reflect in both the Questions asked by LLM and the artist written profile.docx 
- `processing_dataset/artist-summary.py` - looks at all the Questions and Answers and the model free pass text and write a artist-vibe-summary about the artist into each artist intelligence folder.
- `processing_dataset/make-db.py` -> `processing_dataset/artist-profiles-to-jsonl.py` - write the whole artists data into a single artists.db file and combines all the individual artist intellience json file into a single artist-intelligence.jsonl as asked for the submission.

**2. Recommendation :**
- Run `recommender/recommender.py`
- Loops over per hirer converation , analyzes hirer conversation and extracts info for a determinsitic SQL call for first level of filtering, because sending whole data to LLM is impractical even over a dataset of 15 candidates.
- The LLM builds a expanded Query analyzing the hirer intent and breaking the conversation down into explicit requirement , implicit or implied requirement and uncertainty in the conversation.
- The LLM then takes in the artists retrieved over the SQL calls and individually scores each artist on a 3 dimensional rubric :
- `skill_match` = does the evidence support the skills this job needs 
-  `requirement_fit` = do the hirer's stated constraints fit this candidate
-  `evidence_strength` = how strong/verified is the evidence overall 
-  The LLM also provides valid reasoning behind each score.
-  The weighted composit of the score is used to rank the candidates.
-  The LLM then uses the uncertainty in the conversation to frame follow up questions that could change artist ranking.
-  The artists are all send to the LLM in individual calls without batching as LLM are autoregressive models and the order of candidates in batching can change the results. 
-  The models are never cross compared into the same LLM call as the autoregressive nature and the context window both are a concern.
  
**3. re rank :**
- Run `recommender/re-ranker.py`
- Hardcoded to take in the previous recommendation doc and the new extended conversation as there is no hirer id or scession id to rely on.
- follows the same mechanism as the recommender. 
- takes in the new requirement sees if any SQL field needs to be changes, makes an SQL call.
- re expands the request and maintains a detail accound of the requirement switch and returns the reranked candidates as well as the reranking reasoning.

## Setup and use 

```bash
pip install -r requirements.txt
```
- Make a .env file and setup your gemini api key as : GEMINI_API_KEY=''

## Run order

```bash
# 1. Artist intelligence pipeline (run once per new/changed profile)
python data-pipeline.py # -> submission-files/artist_intelligence.jsonl
# 2. Recommendation pipeline (run once per hirer conversation batch)
python recommender/recommender.py # -> submission-files/recommendations.json
python recommender/re-ranker.py # -> submission-files/updated_recommendation.json
```
