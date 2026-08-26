import os
import json
import warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
import cv2
from scenedetect import detect, ContentDetector
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen3VLForConditionalGeneration

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. DYNAMIC RELATIVE PATH RESOLUTION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_PATH = REPO_ROOT / "Data-set" / "artist_profiles"

if not DATASET_PATH.exists():
    DATASET_PATH = SCRIPT_DIR / "Data-set" / "artist_profiles"

# ---------------------------------------------------------
# 2. LOCAL SIGNAL PROCESSING FOR AUDIO (Zero Model Calls)
# ---------------------------------------------------------
def analyze_audio_structure(audio_path):
    """Trims silence, calculates tempo (BPM), and measures onset/energy density."""
    print(f"   [Signal Processing] Analyzing audio track: {audio_path.name}")
    try:
        y, sr = librosa.load(str(audio_path), sr=16000) # Qwen2-Audio prefers 16kHz
        
        # Trim silence
        y_trimmed, index = librosa.effects.trim(y, top_db=30)
        
        # Tempo & Beats
        tempo, beat_frames = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        
        # Onset density (measures rhythm complexity / percussiveness)
        onset_env = librosa.onset.onset_strength(y=y_trimmed, sr=sr)
        avg_onset_strength = float(np.mean(onset_env))
        
        duration = float(len(y_trimmed) / sr)

        metrics = {
            "duration_seconds": round(duration, 2),
            "estimated_bpm": round(float(tempo), 1) if not isinstance(tempo, np.ndarray) else round(float(tempo[0]), 1),
            "rhythmic_intensity": round(avg_onset_strength, 2),
            "is_instrumental_or_dense": bool(avg_onset_strength > 1.5)
        }
        return y_trimmed, sr, metrics
    except Exception as e:
        print(f"    [Error processing audio]: {e}")
        return None, 16000, {}

# ---------------------------------------------------------
# 3. MODEL INFERENCE FUNCTIONS
# ---------------------------------------------------------
def evaluate_audio_with_model(audio_model, audio_processor, audio_array, sr, question_text, metrics_context):
    """Sends audio waveform + metrics to Qwen2-Audio."""
    prompt = f"""
    You are evaluating a musician's audio track.
    Signal Analysis: Estimated BPM: {metrics_context.get('estimated_bpm', 'Unknown')}, Duration: {metrics_context.get('duration_seconds', 0)}s.
    Answer this question directly and concisely in 1-3 sentences without reasoning steps:
    {question_text}
    """
    
    inputs = audio_processor(
        text=prompt, 
        audios=audio_array, 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    ).to(audio_model.device)

    with torch.no_grad():
        generated_ids = audio_model.generate(**inputs, max_new_tokens=150)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = audio_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text

def evaluate_video_with_vlm(vlm_model, vlm_processor, frames, question_text):
    """Sends performance frames to Qwen3-VL for visual evaluation."""
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({
        "type": "text", 
        "text": f"Evaluate this musician's performance video directly and concisely in 1-3 sentences: {question_text}"
    })
    
    messages = [{"role": "user", "content": content}]
    inputs = vlm_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=150)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

# ---------------------------------------------------------
# 4. MAIN PIPELINE LOOP FOR MUSICIANS
# ---------------------------------------------------------
def process_musicians():
    musicians_dir = DATASET_PATH / "musicians"
    if not musicians_dir.exists():
        print(f"[ERROR] Directory not found: {musicians_dir}")
        return

    print("Loading Models (Qwen2-Audio & Qwen3-VL)...")
    audio_model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(audio_model_id, torch_dtype=torch.float16, device_map="auto")
    audio_processor = AutoProcessor.from_pretrained(audio_model_id)

    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(vlm_model_id, dtype="auto", device_map="auto")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)

    for artist_dir in musicians_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        json_file = artist_dir / "profile.json"
        media_dir = artist_dir / "media"

        if not json_file.exists():
            continue

        valid_extensions = {".mp3", ".wav", ".mp4", ".mov", ".m4a"}
        media_files = []
        if media_dir.exists():
            media_files = [f for f in media_dir.iterdir() if f.suffix.lower() in valid_extensions and not f.name.startswith("~$")]

        if not media_files:
            print(f"Skipping {artist_dir.name}: No valid media found.")
            continue

        print(f"\n==================================================")
        print(f"Processing Musician: {artist_dir.name} ({len(media_files)} files found)")
        print(f"==================================================")

        with open(json_file, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # Process primary media file
        primary_media = media_files[0]
        ext = primary_media.suffix.lower()

        audio_array, sr, audio_metrics = None, 16000, {}
        video_frames = []

        if ext in {".mp3", ".wav", ".m4a"}:
            audio_array, sr, audio_metrics = analyze_audio_structure(primary_media)
        elif ext in {".mp4", ".mov"}:
            # Extract audio for Qwen2-Audio and frames for VLM
            audio_array, sr, audio_metrics = analyze_audio_structure(primary_media)
            
            cap = cv2.VideoCapture(str(primary_media))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample 8 uniform frames across the video duration
            frame_indices = np.linspace(0, max(0, total_frames - 1), 8, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        profile_data["audio_signal_metrics"] = audio_metrics

        # Answer Questions
        multimodal_questions = profile_data.get("multimodal_questions", [])
        for idx, q_obj in enumerate(multimodal_questions, start=1):
            if not q_obj.get("answer"):
                media_type = q_obj.get("media_type")
                question = q_obj["question"]
                print(f" -> Q{idx}/10 [{media_type}]: Asking model...")

                try:
                    if media_type == "audio" and audio_array is not None:
                        q_obj["answer"] = evaluate_audio_with_model(
                            audio_model, audio_processor, audio_array, sr, question, audio_metrics
                        )
                    elif media_type == "video" and video_frames:
                        q_obj["answer"] = evaluate_video_with_vlm(
                            vlm_model, vlm_processor, video_frames, question
                        )
                    else:
                        q_obj["answer"] = "Skipped: Missing compatible media format for this question type."
                except Exception as e:
                    q_obj["answer"] = f"Evaluation failed: {str(e)}"

        # Save back to JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)

        print(f"Saved evaluation updates to {json_file.name}")

if __name__ == "__main__":
    process_musicians()