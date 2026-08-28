import json
from pathlib import Path
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------
# 1. MODEL INITIALIZATION (Load once into GPU memory)
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

print(f"Loading model {MODEL_ID}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",  # Or torch.bfloat16 for supported GPUs
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model and processor successfully loaded.")

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def run_vlm_inference(image_paths, question_text):
    """Sends all portfolio images + single question to Qwen3-VL and returns the text response."""
    
    # Construct the multimodal message content
    content = []
    
    # Attach all available images
    for img_path in image_paths:
        content.append({
            "type": "image",
            "image": str(img_path)
        })
        
    # Attach the question text at the end
    content.append({
        "type": "text",
        "text": f"Evaluate the provided portfolio images to provide precise and to the point answers to the given Question: {question_text} /n Warning: Do not describe each picture individually"
    })
    
    messages = [{"role": "user", "content": content}]

    # Prepare inputs using processor chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate response
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
# 3. MAIN EVALUATION LOOP
# ---------------------------------------------------------
def process_photographers(dataset_base_path, profiles_dir):
    base_dir = Path(dataset_base_path)
    photographers_dir = base_dir / "photographers"

    if not photographers_dir.exists():
        print(f"Directory not found: {photographers_dir}")
        return

    profiles_dir = Path(profiles_dir)
    if not profiles_dir.exists():
        print(f"[ERROR] Profiles directory not found: {profiles_dir}")
        return

    # Iterate through all photographer artist directories (e.g., P01_Aanya_Rao)
    for artist_dir in photographers_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        json_file = profiles_dir / f"{artist_dir.name}.json"
        media_dir = artist_dir / "media"

        if not json_file.exists():
            print(f"[WARNING] No matching profile JSON for '{artist_dir.name}' "
                  f"(expected {json_file}). Skipping -- run docx-to-json.py first.")
            continue

        # Find all valid images inside the media folder
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = []
        if media_dir.exists():
            image_files = [
                f for f in media_dir.iterdir() 
                if f.suffix.lower() in valid_extensions and not f.name.startswith("~$")
            ]

        if not image_files:
            print(f"Warning: No images found in {media_dir} for {artist_dir.name}. Skipping VLM evaluation.")
            continue

        print(f"\n==================================================")
        print(f"Processing Photographer: {artist_dir.name} ({len(image_files)} images found)")
        print(f"==================================================")

        # Load profile JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)

        multimodal_questions = profile_data.get("multimodal_questions", [])

        # --- A. ANSWER EACH INDIVIDUAL QUESTION ---
        for idx, q_obj in enumerate(multimodal_questions, start=1):
            # Process only image-based questions that are not answered yet
            if q_obj.get("media_type") == "image" and not q_obj.get("answer"):
                question = q_obj["question"]
                print(f" -> Pass {idx}/10: Asking: '{question[:60]}...'")
                
                try:
                    answer = run_vlm_inference(image_files, question)
                    q_obj["answer"] = answer
                except Exception as e:
                    print(f"    [Error answering question]: {e}")
                    q_obj["answer"] = f"Evaluation failed: {str(e)}"

        # --- B. UNASKED EVIDENCE / SERENDIPITOUS DISCOVERY PASS ---
        print(" -> Running Open Discovery Pass (Capturing unstated artistic evidence)...")
        discovery_prompt = (
            "Examine all images in this portfolio collection. Identify any notable artistic techniques, "
            "hidden visual strengths, composition choices, lighting styles, or technical skill that were NOT "
            "explicitly mentioned in standard descriptions. Summarize what stands out about this photographer's visual signature."
        )
        
        try:
            unasked_evidence = run_vlm_inference(image_files, discovery_prompt)
            profile_data["unasked_artistic_evidence"] = unasked_evidence
        except Exception as e:
            print(f"    [Error running discovery pass]: {e}")
            profile_data["unasked_artistic_evidence"] = f"Discovery pass failed: {str(e)}"

        # --- C. SAVE UPDATED PROFILE BACK TO DISK ---
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=4)
            
        print(f"Saved updated evaluation to: {json_file}")

# ---------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- DYNAMIC RELATIVE PATH RESOLUTION ---
    # Locates the script's directory (.../internship-assignment/processing_dataset)
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Resolves the repository root (.../internship-assignment)
    REPO_ROOT = SCRIPT_DIR.parent
    
    # Path to artist MEDIA (unchanged -- media stays in its original location)
    dataset_path = REPO_ROOT / "Data-set" / "artist_profiles"
    
    # Fallback check if script is executed directly inside the repo root folder
    if not dataset_path.exists():
        dataset_path = SCRIPT_DIR / "Data-set" / "artist_profiles"

    if not dataset_path.exists():
        print(f"[ERROR] Could not locate dataset at: {dataset_path}")
        exit(1)

    # Path to the flat JSON profiles folder written by docx-to-json.py
    # (separate from the media folder above -- see process_photographers)
    profiles_path = REPO_ROOT / "artist_profiles"
    if not profiles_path.exists():
        profiles_path = SCRIPT_DIR / "artist_profiles"

    print(f"Target dataset directory: {dataset_path}")
    print(f"Target profiles directory: {profiles_path}\n")
    
    process_photographers(dataset_path, profiles_path)