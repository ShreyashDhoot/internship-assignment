import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROFILES_DIR = REPO_ROOT / "artist_profiles"
OUTPUT_PATH = REPO_ROOT / "submission-files" / "artist_intelligence.jsonl"


def write_artist_intelligence_jsonl(profiles_dir: Path, output_path: Path) -> int:
    json_files = sorted(profiles_dir.glob("*.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as profile_file:
                profile = json.load(profile_file)
            output_file.write(json.dumps(profile, ensure_ascii=False) + "\n")

    return len(json_files)


if __name__ == "__main__":
    if not PROFILES_DIR.exists():
        print(f"[ERROR] Could not locate artist profiles directory at: {PROFILES_DIR}")
        raise SystemExit(1)

    profile_count = write_artist_intelligence_jsonl(PROFILES_DIR, OUTPUT_PATH)
    print(f"Wrote {profile_count} artist profiles to {OUTPUT_PATH}")
