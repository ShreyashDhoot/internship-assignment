import os
import json
import warnings
from pathlib import Path
import numpy as np
import librosa
import torch
import cv2
from scenedetect import detect, ContentDetector
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Suppress warnings
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
# 2. FREE SIGNAL PROCESSING (Zero Model Calls)
# ---------------------------------------------------------
def analyze_video_structure(video_path):
    """Detects scene cuts, computes pacing stats, and measures cut-to-beat audio alignment."""
    print(f"   [Signal Processing] Analyzing structure for: {video_path.name}")
    
    # --- A. Shot Boundary & Pacing Stats ---
    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
    cut_timestamps = [scene[0].get_seconds() for scene in scene_list[1:]] # Skip initial start
    
    # Calculate Shot Lengths
    if scene_list:
        shot_durations = [scene[1].get_seconds() - scene[0].get_seconds() for scene in scene_list]
    else:
        # Fallback if no cuts detected
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        shot_durations = [duration]

    avg_shot_length = float(np.mean(shot_durations)) if shot_durations else 0.0
    shot_length_variance = float(np.var(shot_durations)) if shot_durations else 0.0

    # Determine Pacing Style
    if avg_shot_length < 2.0:
        pacing_style = "Fast-paced / Rhythmic Montage"
    elif avg_shot_length < 5.0:
        pacing_style = "Balanced Commercial / Narrative"
    else:
        pacing_style = "Slow-paced / Cinematic Long Takes"

    # --- B. Audio Beat Alignment ---
    beat_sync_score = None
    try:
        y, sr = librosa.load(str(video_path), sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Check if cuts fall within 150ms window of a beat
        if len(cut_timestamps) > 0 and len(beat_times) > 0:
            synced_cuts = 0
            for cut_t in cut_timestamps:
                # Find minimum distance to any beat
                min_diff = np.min(np.abs(beat_times - cut_t))
                if min_diff <= 0.15: # 150 milliseconds tolerance
                    synced_cuts += 1
            beat_sync_score = round((synced_cuts / len(cut_timestamps)) * 100, 2)
    except Exception:
        # Audio track missing or unparseable
        beat_sync_score = None

    metrics = {
        "total_cuts": len(cut_timestamps),
        "average_shot_length_seconds": round(avg_shot_length, 2),
        "shot_length_variance": round(shot_length_variance, 2),
        "pacing_style": pacing_style,
        "beat_sync_accuracy_percentage": beat_sync_score,
        "cut_timestamps": [round(ts, 2) for ts in cut_timestamps]
    }
    return metrics

def extract_boundary_frames(video_path, cut_timestamps, num_cuts=3, frames_per_cut=8):
    """Extracts a sequence of frames spanning selected cuts and tracks their exact timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    extracted_frames = []
    frame_timestamps = []

    # Pick cuts spread evenly across the video
    if len(cut_timestamps) > num_cuts:
        selected_cuts = np.linspace(0, len(cut_timestamps) - 1, num_cuts, dtype=int)
        target_cuts = [cut_timestamps[i] for i in selected_cuts]
    else:
        target_cuts = cut_timestamps

    for cut_time in target_cuts:
        # Extract a 2-second window centered on the cut point (-1s to +1s)
        start_frame = max(0, int((cut_time - 1.0) * fps))
        for offset in range(frames_per_cut):
            frame_idx = start_frame + int(offset * (fps / (frames_per_cut / 2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                timestamp = frame_idx / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames.append(rgb_frame)
                frame_timestamps.append(round(timestamp, 2))

    cap.release()
    return extracted_frames, frame_timestamps

# ---------------------------------------------------------
# 3. VLM MODEL INFERENCE (Qwen3-VL)
# ---------------------------------------------------------
def evaluate_video_with_vlm(model, processor, frames, question_text, metrics_context):
    """Passes sequence of cut-spanning frames + signal metrics to the VLM."""
    content = []
    
    # Attach boundary frames sequence
    for frame in frames:
        content.append({"type": "image", "image": frame})

    # Attach prompt with pre-computed structural evidence
    prompt_text = f"""
    You are evaluating a video editor's portfolio clip. 
    Here is pre-computed signal analysis for this video:
    - Average Shot Length: {metrics_context['average_shot_length_seconds']}s
    - Pacing Style: {metrics_context['pacing_style']}
    - Beat Alignment Score: {metrics_context['beat_sync_accuracy_percentage']}%

    Based on the frame sequence spanning the cuts and the structural data above, answer this question concisely:
    {question_text}
    """
    
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

# ---------------------------------------------------------
# 4. MAIN PIPELINE LOOP
# ---------------------------------------------------------
def process_video_editors():
    video_editors_dir = DATASET_PATH / "video_editors"
    if not video_editors_dir.exists():
        print(f"[ERROR] Directory not found: {video_editors_dir}")
        return

    # Load Qwen3-VL model once
    print("Loading Qwen3-VL-4B-Thinking model...")
    MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    for artist_dir in video_editors_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        json_file = artist_dir / "profile.json"
        media_dir = artist_dir / "media"

        if not json_file.exists():
            continue

        video_extensions = {".mp4", ".mov", ".mkv", ".avi"}
        video_files = []
        if media_dir.exists():
            video_files = [
                f for f in media_dir.iterdir()
                if f.suffix.lower() in video_extensions and not f.name.startswith("~$")
            ]

        if not video_files:
            print(f"Skipping {artist_dir.name}: No video files found in media/ directory.")
            continue

        print(f"\n==================================================")
        print(f"Processing Video Editor: {artist_dir.name}")
        print(f"==================================================")

        with open(json_file, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # 1. Run Signal Processing across ALL portfolio videos
        all_cut_timestamps = []
        all_boundary_frames = []
        all_boundary_timestamps = []
        combined_metrics_list = []

        for video_file in video_files:
            print(f"   -> Processing file: {video_file.name}")
            metrics = analyze_video_structure(video_file)
            combined_metrics_list.append(metrics)
            
            # Extract frames for this specific video
            frames, timestamps = extract_boundary_frames(video_file, metrics["cut_timestamps"])
            all_boundary_frames.extend(frames)
            all_boundary_timestamps.extend(timestamps)

        # Aggregate metrics for prompt context
        avg_shot_length = np.mean([m["average_shot_length_seconds"] for m in combined_metrics_list])
        avg_beat_sync = np.mean([m["beat_sync_accuracy_percentage"] for m in combined_metrics_list if m["beat_sync_accuracy_percentage"] is not None])

        aggregate_metrics = {
            "total_videos_analyzed": len(video_files),
            "average_shot_length_seconds": round(float(avg_shot_length), 2),
            "pacing_style": combined_metrics_list[0]["pacing_style"], # Take primary style
            "beat_sync_accuracy_percentage": round(float(avg_beat_sync), 2) if not np.isnan(avg_beat_sync) else None,
            "sampled_boundary_frame_timestamps": all_boundary_timestamps
        }
        
        profile_data["editing_signal_metrics"] = aggregate_metrics

        # 2. Answer Multimodal Questions using frames collected across ALL videos
        multimodal_questions = profile_data.get("multimodal_questions", [])
        for idx, q_obj in enumerate(multimodal_questions, start=1):
            if q_obj.get("media_type") == "video" and not q_obj.get("answer"):
                print(f" -> Q{idx}/10: Asking VLM across {len(video_files)} clips...")
                try:
                    answer = evaluate_video_with_vlm(
                        model, processor, all_boundary_frames, q_obj["question"], aggregate_metrics
                    )
                    q_obj["answer"] = answer
                except Exception as e:
                    q_obj["answer"] = f"Evaluation failed: {str(e)}"

        # 2. Extract Cut-Spanning Frame Sequences & Timestamps
        boundary_frames, boundary_timestamps = extract_boundary_frames(
            primary_video, structure_metrics["cut_timestamps"]
        )

        # Save boundary frame timestamps into signal metrics
        structure_metrics["sampled_boundary_frame_timestamps"] = boundary_timestamps
        profile_data["editing_signal_metrics"] = structure_metrics

        # 3. Answer Multimodal Questions
        multimodal_questions = profile_data.get("multimodal_questions", [])
        for idx, q_obj in enumerate(multimodal_questions, start=1):
            if q_obj.get("media_type") == "video" and not q_obj.get("answer"):
                print(f" -> Q{idx}/10: Asking VLM: '{q_obj['question'][:60]}...'")
                try:
                    answer = evaluate_video_with_vlm(
                        model, processor, boundary_frames, q_obj["question"], structure_metrics
                    )
                    q_obj["answer"] = answer
                except Exception as e:
                    print(f"    [Error]: {e}")
                    q_obj["answer"] = f"Evaluation failed: {str(e)}"

        # 4. Open Discovery Pass (Capturing Unasked Editing Craft)
        print(" -> Running Open Discovery Pass (Editing Craft)...")
        discovery_prompt = (
            "Examine the frame transitions and signal metrics. Identify any notable video editing techniques, "
            "pacing choices, color continuity, motion graphic integration, or style choices not explicitly covered above."
        )
        try:
            profile_data["unasked_artistic_evidence"] = evaluate_video_with_vlm(
                model, processor, boundary_frames, discovery_prompt, structure_metrics
            )
        except Exception as e:
            profile_data["unasked_artistic_evidence"] = f"Discovery pass failed: {str(e)}"

        # 5. Overwrite JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)

        print(f"Successfully saved evaluation to {json_file.name}")

if __name__ == "__main__":
    process_video_editors()