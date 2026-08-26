import os
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

# Suppress the deprecation warning
warnings.filterwarnings("ignore", module="google.generativeai")
load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------
# 1. HARDCODED GENERIC BENCHMARK QUESTIONS (5 per category)
# ---------------------------------------------------------
STANDARD_BENCHMARKS = {
    "musicians": [
        {"question": "Is the audio recording free from background hiss, clipping, or distortion?", "media_type": "audio", "vibe_metric": "Audio Clarity", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Are the primary vocals or lead instruments well-balanced in the mix?", "media_type": "audio", "vibe_metric": "Mixing Balance", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Does the performance maintain a consistent tempo and rhythm?", "media_type": "audio", "vibe_metric": "Rhythmic Consistency", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Is the emotional tone of the performance energetic, melancholic, or neutral?", "media_type": "audio", "vibe_metric": "Emotional Tone", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "What is the primary genre and instrumentation of this track?", "media_type": "audio", "vibe_metric": "Genre Classification", "question_type": "Standardized Benchmark", "answer": ""}
    ],
    "photographers": [
        {"question": "Are the primary subjects in the photographs sharply in focus?", "media_type": "image", "vibe_metric": "Focus & Sharpness", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Does the lighting feel natural, artificially staged, or poorly exposed?", "media_type": "image", "vibe_metric": "Lighting Quality", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Do the images follow standard compositional rules like the rule of thirds?", "media_type": "image", "vibe_metric": "Composition", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Is the color grading heavily stylized, natural, or desaturated?", "media_type": "image", "vibe_metric": "Color Palette", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Does the portfolio show consistency in style across different shots?", "media_type": "image", "vibe_metric": "Stylistic Consistency", "question_type": "Standardized Benchmark", "answer": ""}
    ],
    "video_editors": [
        {"question": "Are the transitions between clips smooth and purposeful?", "media_type": "video", "vibe_metric": "Transition Quality", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Is the visual pacing matched effectively to the background audio/music?", "media_type": "video", "vibe_metric": "Audio-Visual Sync", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Is the color grading consistent across different shots in the sequence?", "media_type": "video", "vibe_metric": "Color Continuity", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Does the edit tell a clear narrative or maintain a cohesive theme?", "media_type": "video", "vibe_metric": "Narrative Flow", "question_type": "Standardized Benchmark", "answer": ""},
        {"question": "Are graphical overlays or text (if any) professionally integrated?", "media_type": "video", "vibe_metric": "Motion Graphics", "question_type": "Standardized Benchmark", "answer": ""}
    ]
}

# ---------------------------------------------------------
# 2. LLM SCHEMA FOR BIO-SPECIFIC QUESTIONS
# ---------------------------------------------------------
# Notice we only ask the LLM for 5 questions now.
class BioSpecificQuestion(BaseModel):
    question: str = Field(description="The exact question to ask the Multimodal AI")
    media_type: str = Field(description="Must be 'audio', 'video', or 'image'")
    vibe_metric: str = Field(description="What this question evaluates based on the bio")

class ProfileAnalysis(BaseModel):
    bio_questions: list[BioSpecificQuestion] = Field(
        description="Exactly 5 questions to verify specific claims in the artist's bio."
    )

# ---------------------------------------------------------
# 3. GENERATION & JSON INJECTION
# ---------------------------------------------------------
def process_artist_profile(json_file_path, category_folder_name):
    # Load the existing profile.json
    with open(json_file_path, 'r', encoding='utf-8') as f:
        artist_data = json.load(f)

    # 1. Fetch the 5 hardcoded generic questions based on the genre
    generic_questions = STANDARD_BENCHMARKS.get(category_folder_name, [])

    # 2. Use LLM to generate 5 bio-specific questions
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    
    prompt = f"""
    Review this artist profile. Generate EXACTLY 5 highly specific questions 
    to ask a Multimodal AI to verify the unique claims made in their Bio and 
    exctract nore information about their artistic abilities to recommend 
    them to employers.
    
    For example: If they claim to do 'cinematic travel weddings', ask if the video 
    shows travel locations and wedding subjects. If they claim 'acoustic duo', 
    ask if two distinct instruments/voices are heard.
    
    Profile Data:
    {json.dumps(artist_data, indent=2)}
    """

    print(f"Generating 5 bio questions for {artist_data.get('name', 'Unknown')}...")
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ProfileAnalysis,
            ),
        )

        llm_output = json.loads(response.text)
        
        # 3. Format the LLM output to match our schema and add the empty answer field
        bio_questions = []
        for q in llm_output.get("bio_questions", []):
            bio_questions.append({
                "question": q["question"],
                "media_type": q["media_type"],
                "vibe_metric": q["vibe_metric"],
                "question_type": "Bio Verification",
                "answer": "" # Intentionally left blank for the multimodal phase
            })

        # 4. Inject the empty summary and the combined 10 questions into the dictionary
        artist_data["artist_vibe_summary"] = ""
        artist_data["multimodal_questions"] = generic_questions + bio_questions

        # 5. Overwrite the original profile.json file with the new data
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(artist_data, f, indent=4)
            
        print(f"  -> Successfully updated {json_file_path.name}")
        
    except Exception as e:
        print(f"  -> [ERROR] Failed to update {json_file_path.name}: {e}")

if __name__ == "__main__":
    dataset_path = Path(r"D:\INTERNSHIP-assigment\Data-set\artist_profiles")
    json_files = list(dataset_path.rglob("profile.json"))
    
    if not json_files:
        print("No profile.json files found.")
    else:
        for json_file in json_files:
            # We get the category dynamically from the folder structure
            # e.g., artist_profiles / video_editors / V01_Nisha / profile.json
            category_folder = json_file.parent.parent.name 
            
            # Skip if we already added the questions (prevents duplicates on re-runs)
            with open(json_file, 'r', encoding='utf-8') as f:
                if "multimodal_questions" in json.load(f):
                    print(f"Skipping {json_file.parent.name} - already updated.")
                    continue
                    
            process_artist_profile(json_file, category_folder)
            
        print("\nSuccessfully injected questions into all profiles.")