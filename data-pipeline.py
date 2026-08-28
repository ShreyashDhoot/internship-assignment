import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    REPO_ROOT / "processing_dataset" / "docx-to-json.py",
    REPO_ROOT / "processing_dataset" / "Question-generator.py",
    REPO_ROOT / "Multimodal" / "image-processor.py",
    REPO_ROOT / "Multimodal" / "music-processor.py",
    REPO_ROOT / "Multimodal" / "video-processor.py",
    REPO_ROOT / "processing_dataset" / "artist-summary.py",
    REPO_ROOT / "processing_dataset" / "artist-profiles-to-jsonl.py",
]


def run_pipeline() -> None:
    for step in PIPELINE_STEPS:
        if not step.exists():
            raise FileNotFoundError(f"Pipeline step not found: {step}")

        print(f"\n{'=' * 70}\nRunning: {step.relative_to(REPO_ROOT)}\n{'=' * 70}")
        subprocess.run([sys.executable, str(step)], cwd=REPO_ROOT, check=True)

    print("\nData pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except subprocess.CalledProcessError as error:
        print(f"\n[ERROR] Pipeline stopped: {error}")
        raise SystemExit(error.returncode)
    except FileNotFoundError as error:
        print(f"\n[ERROR] {error}")
        raise SystemExit(1)
