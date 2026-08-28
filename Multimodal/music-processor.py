import os
import json
import subprocess
import tempfile
import warnings
from pathlib import Path
import numpy as np
import librosa
import torch
import cv2
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

AUDIO_ONLY_EXTENSIONS = {".mp3", ".wav", ".m4a"}
VIDEO_CONTAINER_EXTENSIONS = {".mp4", ".mov"}


# ---------------------------------------------------------
# 2. AUDIO EXTRACTION FROM VIDEO CONTAINERS
# ---------------------------------------------------------
def extract_audio_to_wav(video_path: Path):

    tmp_wav = Path(tempfile.gettempdir()) / f"{video_path.stem}_extracted_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # decode to plain PCM wav
        "-ar", "16000",          # Qwen2-Audio prefers 16kHz
        "-ac", "1",              # mono
        str(tmp_wav),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not tmp_wav.exists():
            print(f"    [ffmpeg] Could not extract audio from {video_path.name}: {result.stderr[-300:]}")
            return None
        return tmp_wav
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    [ffmpeg] Extraction failed for {video_path.name}: {e}")
        return None


# ---------------------------------------------------------
# 3. LOCAL SIGNAL PROCESSING FOR AUDIO (Zero Model Calls)
# ---------------------------------------------------------
def analyze_audio_structure(audio_path: Path):
    """Trims silence, calculates tempo (BPM), and measures onset/energy density."""
    print(f"   [Signal Processing] Analyzing audio track: {audio_path.name}")
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y_trimmed, sr=sr)
        avg_onset_strength = float(np.mean(onset_env))
        duration = float(len(y_trimmed) / sr)

        metrics = {
            "duration_seconds": round(duration, 2),
            "estimated_bpm": round(float(tempo), 1) if not isinstance(tempo, np.ndarray) else round(float(tempo[0]), 1),
            "rhythmic_intensity": round(avg_onset_strength, 2),
            "is_instrumental_or_dense": bool(avg_onset_strength > 1.5),
        }
        return y_trimmed, sr, metrics
    except Exception as e:
        print(f"    [Error processing audio]: {e}")
        return None, 16000, {}


def load_audio_evidence_for_file(media_file: Path):
    """
    Returns (audio_array, sr, metrics, source_filename) for ONE media
    file, handling the two real cases in this dataset:
      - pure audio files (.mp3/.wav/.m4a): loaded directly.
      - video containers (.mp4/.mov): audio track extracted via ffmpeg
        first, since librosa can't reliably demux every mp4 codec directly.
    Returns (None, 16000, {}, filename) if extraction/loading genuinely fails
    -- caller filters out the None case, this function is called exactly
    once per file (no redundant re-processing).
    """
    ext = media_file.suffix.lower()
    if ext in AUDIO_ONLY_EXTENSIONS:
        arr, sr, metrics = analyze_audio_structure(media_file)
        return arr, sr, metrics, media_file.name
    elif ext in VIDEO_CONTAINER_EXTENSIONS:
        extracted_wav = extract_audio_to_wav(media_file)
        if extracted_wav is None:
            return None, 16000, {}, media_file.name
        arr, sr, metrics = analyze_audio_structure(extracted_wav)
        extracted_wav.unlink(missing_ok=True)  # clean up temp file
        return arr, sr, metrics, media_file.name
    return None, 16000, {}, media_file.name


def extract_video_frames(video_path: Path, num_frames: int = 8):
    """Uniform frame sampling across one video file's duration."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        frame_indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---------------------------------------------------------
# 4. MODEL INFERENCE FUNCTIONS
# ---------------------------------------------------------
def evaluate_audio_with_model(audio_model, audio_processor, audio_arrays_with_sr, question_text, metrics_list):
    """
    Sends ALL pooled audio waveforms to Qwen2-Audio for one question, so
    the answer reflects evidence across every audio-bearing file for
    this artist (both standalone audio files AND audio tracks extracted
    from video files) -- not just one arbitrarily chosen "primary" file.

    Per the official Qwen2-Audio processor docs:
      - the prompt text must contain one <|AUDIO|> placeholder token per
        audio clip, built via apply_chat_template over a conversation
        with one {"type": "audio"} content block per clip -- a raw
        f-string prompt with no placeholders does not work correctly
        even once the kwarg name is fixed.
    """
    arrays = [a for a, sr in audio_arrays_with_sr]

    bpm_summary = ", ".join(str(m.get("estimated_bpm", "?")) for m in metrics_list)
    question_with_context = (
        f"Signal analysis per track (BPM): {bpm_summary}. "
        f"Considering ALL provided recordings together, answer directly and "
        f"concisely in 1-3 sentences: {question_text}"
    )

    # One audio content block per clip, then the question as text --
    # this is what apply_chat_template needs to correctly insert one
    # <|AUDIO|> placeholder per clip into the tokenized prompt.
    content = [{"type": "audio"} for _ in arrays]
    content.append({"type": "text", "text": question_with_context})
    conversation = [{"role": "user", "content": content}]

    text_prompt = audio_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = audio_processor(
        text=text_prompt,
        audio=arrays,
        return_tensors="pt",
        padding=True,
    ).to(audio_model.device)

    with torch.no_grad():
        generated_ids = audio_model.generate(**inputs, max_new_tokens=150)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return audio_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def evaluate_video_with_vlm(vlm_model, vlm_processor, all_frames, question_text):
    """Sends frames pooled across ALL of this artist's video files for one question."""
    content = [{"type": "image", "image": frame} for frame in all_frames]
    content.append({
        "type": "text",
        "text": f"Evaluate this musician's performance across all provided video clips, directly and concisely : {question_text} \n Warning: do not describe each evaluation individually",
    })

    messages = [{"role": "user", "content": content}]
    inputs = vlm_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ---------------------------------------------------------
# 5. MAIN PIPELINE LOOP FOR MUSICIANS
# ---------------------------------------------------------
def process_musicians():
    musicians_dir = DATASET_PATH / "musicians"
    if not musicians_dir.exists():
        print(f"[ERROR] Directory not found: {musicians_dir}")
        return

    profiles_dir = REPO_ROOT / "artist_profiles"
    if not profiles_dir.exists():
        print(f"[ERROR] Profiles directory not found: {profiles_dir}")
        return

    print("Loading Models (Qwen2-Audio & Qwen3-VL)...")
    audio_model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        audio_model_id, torch_dtype=torch.float16, device_map="auto"
    )
    audio_processor = AutoProcessor.from_pretrained(audio_model_id)

    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(vlm_model_id, dtype="auto", device_map="auto")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)

    for artist_dir in musicians_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        json_file = profiles_dir / f"{artist_dir.name}.json"
        media_dir = artist_dir / "media"

        if not json_file.exists():
            print(f"[WARNING] No matching profile JSON for '{artist_dir.name}' "
                  f"(expected {json_file}). Skipping -- run docx-to-json.py first.")
            continue

        valid_extensions = AUDIO_ONLY_EXTENSIONS | VIDEO_CONTAINER_EXTENSIONS
        media_files = []
        if media_dir.exists():
            media_files = [
                f for f in media_dir.iterdir()
                if f.suffix.lower() in valid_extensions and not f.name.startswith("~$")
            ]

        if not media_files:
            print(f"Skipping {artist_dir.name}: No valid media found.")
            continue

        print("\n==================================================")
        print(f"Processing Musician: {artist_dir.name} ({len(media_files)} files found)")
        print("==================================================")

        with open(json_file, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # --- Pool audio evidence across ALL media files ---
        # Both pure-audio files AND the audio track of video files count
        # as audio evidence usable audio track for judging musicianship questions. Each
        # file is processed exactly once here; results are reused below
        # for both the model call and the audit-trail metadata.
        all_audio_arrays_with_sr = []
        all_audio_metrics = []
        audio_evidence_filenames = []
        for f in media_files:
            arr, sr, metrics, filename = load_audio_evidence_for_file(f)
            if arr is not None:
                all_audio_arrays_with_sr.append((arr, sr))
                all_audio_metrics.append(metrics)
                audio_evidence_filenames.append(filename)

        # --- Pool video frames across ALL video-container files ---
        video_files = [f for f in media_files if f.suffix.lower() in VIDEO_CONTAINER_EXTENSIONS]
        all_video_frames = []
        for f in video_files:
            all_video_frames.extend(extract_video_frames(f))

        profile_data["audio_signal_metrics"] = all_audio_metrics
        profile_data["media_files_used"] = {
            "audio_evidence_from": audio_evidence_filenames,
            "video_frame_evidence_from": [f.name for f in video_files],
        }

        FAILURE_MARKERS = ("Skipped:", "Evaluation failed:")

        def _needs_retry(q_obj: dict) -> bool:
            answer = q_obj.get("answer")
            if not answer:
                return True
            return any(str(answer).startswith(marker) for marker in FAILURE_MARKERS)

        multimodal_questions = profile_data.get("multimodal_questions", [])
        for idx, q_obj in enumerate(multimodal_questions, start=1):
            if not _needs_retry(q_obj):
                continue
            media_type = q_obj.get("media_type")
            question = q_obj["question"]
            print(f" -> Q{idx}/{len(multimodal_questions)} [{media_type}]: Asking model...")

            try:
                if media_type == "audio" and all_audio_arrays_with_sr:
                    q_obj["answer"] = evaluate_audio_with_model(
                        audio_model, audio_processor, all_audio_arrays_with_sr, question, all_audio_metrics
                    )
                elif media_type == "video" and all_video_frames:
                    q_obj["answer"] = evaluate_video_with_vlm(
                        vlm_model, vlm_processor, all_video_frames, question
                    )
                else:
                    q_obj["answer"] = "Skipped: No compatible media evidence available for this question type."
            except Exception as e:
                q_obj["answer"] = f"Evaluation failed: {str(e)}"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)

        print(f"Saved evaluation updates to {json_file.name}")


if __name__ == "__main__":
    process_musicians()