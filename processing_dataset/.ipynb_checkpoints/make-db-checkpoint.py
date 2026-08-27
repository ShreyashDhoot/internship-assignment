import json
import sqlite3
from pathlib import Path

# Field names below match docx-to-json.py's actual output exactly:
# artist_id, name, category, location, work_preference, bio, portfolio.
# (category here is the FOLDER name -- e.g. "musicians", "photographers",
# "video_editors" -- not the docx-stated category text, since that's
# already the ground-truth taxonomy used for filtering.)

# Small, explicitly-stated location grouping -- NOT a geocoding system.
# Deliberate scope limitation (see decision_note.md): exact city match
# first, then this small manual list of known NCR-adjacent names as a
# fallback during relaxation. This is a stated limitation, not something
# we're trying to solve generally -- extend only if the real dataset's
# locations actually need more entries.
NEARBY_CITY_GROUPS = {
    "delhi ncr": {"delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
}


def normalize_city(location_text):
    """Best-effort single lowercase city token for indexed filtering."""
    if not location_text:
        return None
    first_segment = location_text.split(",")[0].split("/")[0].strip().lower()
    return first_segment or None


def city_region(city):
    """
    Which NEARBY_CITY_GROUPS bucket this city falls into, if any.
    Handles two real cases: a specific city name ("gurgaon") that's a
    member of a region, AND a location string that already names the
    region directly ("delhi ncr") -- confirmed necessary because real
    profiles sometimes list "Delhi NCR" itself as the location, not a
    specific city within it, which would otherwise silently fail to
    match any region and get excluded from region-based search.
    """
    if not city:
        return None
    if city in NEARBY_CITY_GROUPS:
        return city
    for region, members in NEARBY_CITY_GROUPS.items():
        if city in members:
            return region
    return None


def setup_sqlite_db(db_path="artists.db"):
    """
    Creates the artists table with real, indexed columns for every field
    the search/filter/relaxation pipeline needs to query on directly
    (category, city, region, work_preference), plus the full profile
    JSON preserved in `data` for anything else (bio text, portfolio,
    later-added evaluation fields like multimodal_questions) that
    doesn't need a WHERE clause.

    Promoting these into real columns means the deterministic filter
    stage can use plain indexed SQL (WHERE category = ? AND
    location_city = ?) instead of json_extract() calls scattered
    through query strings.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS artist_documents (
            artist_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            docx_stated_category TEXT,
            location_raw TEXT,
            location_city TEXT,
            location_region TEXT,
            work_preference TEXT,
            data TEXT CHECK(json_valid(data))
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON artist_documents(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_city ON artist_documents(location_city)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_region ON artist_documents(location_region)')
    conn.commit()
    return conn


def insert_into_db(conn, profile_data):
    """
    Inserts one artist record, populating both the indexed search
    columns AND the full JSON blob (so nothing -- bio, portfolio, any
    evaluation fields added later by music/video processors -- is lost,
    it's just not separately indexed).
    """
    cursor = conn.cursor()

    artist_id = profile_data.get("artist_id", "UNKNOWN")
    name = profile_data.get("name")
    category = profile_data.get("category")
    docx_stated_category = profile_data.get("docx_stated_category")
    location_raw = profile_data.get("location")
    work_preference = profile_data.get("work_preference")

    city = normalize_city(location_raw)
    region = city_region(city)

    cursor.execute('''
        INSERT OR REPLACE INTO artist_documents
            (artist_id, name, category, docx_stated_category, location_raw,
             location_city, location_region, work_preference, data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        artist_id, name, category, docx_stated_category, location_raw,
        city, region, work_preference, json.dumps(profile_data),
    ))
    conn.commit()


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    # JSON profiles now live in the separate flat folder written by
    # docx-to-json.py (one file per artist, e.g. M01_Meera_Arjun.json),
    # not nested under Data-set/artist_profiles/<category>/<artist>/
    # profile.json anymore -- glob for *.json directly in that folder,
    # not profile.json recursively under the old dataset path.
    profiles_path = REPO_ROOT / "artist_profiles"
    if not profiles_path.exists():
        profiles_path = SCRIPT_DIR / "artist_profiles"

    if not profiles_path.exists():
        print(f"[ERROR] Could not locate profiles directory at: {profiles_path}")
        exit(1)

    db_conn = setup_sqlite_db("artists.db")

    json_files = sorted(profiles_path.glob("*.json"))

    if not json_files:
        print("No profile JSON files found. Did you run docx-to-json.py first?")
    else:
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            insert_into_db(db_conn, profile_data)
            print(f"Inserted into DB: {profile_data.get('artist_id')} - {profile_data.get('name')} "
                  f"[category={profile_data.get('category')}, city={normalize_city(profile_data.get('location'))}]")

        db_conn.close()
        print(f"\nDatabase build complete. Inserted {len(json_files)} records into artists.db.")