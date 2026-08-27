import os
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. ENVIRONMENT & API SETUP
# ---------------------------------------------------------
load_dotenv(find_dotenv())
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please ensure GEMINI_API_KEY is in your .env file.")
genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 2. DYNAMIC RELATIVE PATH RESOLUTION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# JSON profiles now live in the separate flat folder written by
# docx-to-json.py (one file per artist, named after the artist's
# folder -- e.g. M01_Meera_Arjun.json), not nested under
# Data-set/artist_profiles/<category>/<artist>/profile.json anymore.
PROFILES_PATH = REPO_ROOT / "artist_profiles"
if not PROFILES_PATH.exists():
    PROFILES_PATH = SCRIPT_DIR / "artist_profiles"


# ---------------------------------------------------------
# 3. SYNTHESIZE VIBE SUMMARY USING GEMINI
# ---------------------------------------------------------
def synthesize_vibe_summary(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)

    # Skip if already summarized -- avoids re-spending an API call on
    # every run for artists already done, same pattern as
    # Question-generator.py's "already updated" guard. Checked here
    # (rather than only in the __main__ loop) so this function stays
    # safe to call directly too.
    if profile_data.get("artist_vibe_summary"):
        print(f"Skipping {json_file_path.stem} - vibe summary already present.")
        return

    # Initialize fast, cheap LLM
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    prompt = f"""
    You are a professional creative director and art curator. Review the complete profile data below, 
    including the artist's bio, portfolio metadata, pre-computed signal metrics, and the multimodal evaluation Q&A results.
    Write a compelling, concise 2-to-3 sentence "artist vibe summary" that captures their true aesthetic signature, technical strengths, and professional capability based on concrete evidence from their evaluated work.
    Complete Artist Profile:
    {json.dumps(profile_data, indent=2)}
    """
    print(f"Synthesizing vibe summary for: {profile_data.get('name', 'Unknown')} ({profile_data.get('artist_id')})...")

    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()

        # Inject into profile dictionary
        profile_data["artist_vibe_summary"] = summary

        # Overwrite JSON file
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=4)

        # NOTE: previously this logged json_file_path.parent.name, which
        # was the artist's own folder under the old nested layout. Now
        # that all profile JSONs live flat in one folder, .parent.name
        # would print "artist_profiles" for every artist -- using
        # json_file_path.stem (the filename itself, e.g.
        # "M01_Meera_Arjun") instead, which is what actually identifies
        # the artist now.
        print(f"  -> Successfully updated vibe summary for {json_file_path.stem}\n")

    except Exception as e:
        print(f"  -> [ERROR] Failed for {json_file_path.name}: {e}\n")


# ---------------------------------------------------------
# 4. MAIN EXECUTION LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    if not PROFILES_PATH.exists():
        print(f"[ERROR] Could not locate profiles directory at: {PROFILES_PATH}")
        exit(1)

    json_files = sorted(PROFILES_PATH.glob("*.json"))

    if not json_files:
        print("No profile JSON files found.")
    else:
        print(f"Found {len(json_files)} profiles. Generating vibe summaries...\n")
        for json_file in json_files:
            synthesize_vibe_summary(json_file)

        print("All artist vibe summaries generated and saved successfully.")