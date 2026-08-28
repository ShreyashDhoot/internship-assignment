import json
import sqlite3
from pathlib import Path

# Field names below match docx-to-json.py's actual output:
# artist_id, name, category, location, work_preference, bio, portfolio.
#
# IMPORTANT (fixed): the JSON files actually on disk do NOT carry a
# reliable ground-truth "category" -- the `category` key in each profile
# JSON is whatever free text the docx itself stated (e.g. "vocalist and
# Guitarist", "Video Editor", "Photographer / Cinematographer"), and
# `docx_stated_category` doesn't exist in any real profile file. So the
# old code was writing that free text straight into the DB's `category`
# column, which never matched the fixed vocabulary
# ("musician"/"video-editor"/"photographer") the recommender's SQL
# search filters on -- every search silently returned zero rows.
#
# The one thing that IS reliable is the filename prefix (M*/V*/P*),
# which is the actual ground-truth taxonomy used when the profiles were
# generated. So:
#   - `category` (indexed, used for hard filtering) is now derived from
#     the filename prefix: M -> "musician", V -> "video-editor",
#     P -> "photographer".
#   - the free-text category string that used to land in `category` is
#     now stored as `niche` instead (e.g. "vocalist and Guitarist",
#     "Video Editor", "Photographer / Cinematographer") -- it's real,
#     useful info for soft-matching a hirer's specific ask, just not a
#     hard filter value.

FILENAME_PREFIX_TO_CATEGORY = {
    "M": "musician",
    "V": "video-editor",
    "P": "photographer",
}

# Small, explicitly-stated location grouping -- NOT a geocoding system.
# Deliberate scope limitation (see decision_note.md): exact city match
# first, then this small manual list of known NCR-adjacent names as a
# fallback during relaxation. This is a stated limitation, not something
# we're trying to solve generally -- extend only if the real dataset's
# locations actually need more entries.
NEARBY_CITY_GROUPS = {
    "delhi ncr": {"delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
}


def category_from_filename(filename_stem):
    """
    Derive the ground-truth category from the artist_profiles filename,
    not from anything inside the JSON. Filenames look like
    "M01_Meera_Arjun", "PO4_Drift", "VO5_Roshan" -- the first character
    is the reliable signal (some profiles even have a wrong/duplicated
    artist_id *inside* the JSON, e.g. PO4_Drift.json has
    artist_id "V05" -- filename prefix is the one thing that's
    consistent, so that's what we key off of).
    """
    if not filename_stem:
        return "unknown"
    first_char = filename_stem[0].upper()
    return FILENAME_PREFIX_TO_CATEGORY.get(first_char, "unknown")


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
    (category, city, region, niche, work_preference), plus the full
    profile JSON preserved in `data` for anything else (bio text,
    portfolio, later-added evaluation fields like multimodal_questions)
    that doesn't need a WHERE clause.

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
            niche TEXT,
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
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_niche ON artist_documents(niche)')
    conn.commit()
    return conn


def insert_into_db(conn, profile_data, filename_stem):
    """
    Inserts one artist record, populating both the indexed search
    columns AND the full JSON blob (so nothing -- bio, portfolio, any
    evaluation fields added later by music/video processors -- is lost,
    it's just not separately indexed).

    `category` is derived from the filename (ground truth), never from
    the JSON's own `category` key. That JSON key's value is preserved
    as-is under `niche` instead.
    """
    cursor = conn.cursor()

    artist_id = profile_data.get("artist_id", "UNKNOWN")
    name = profile_data.get("name")
    category = category_from_filename(filename_stem)
    niche = profile_data.get("category")  # the docx-stated free text, e.g. "Vocalist & Guitarist"
    location_raw = profile_data.get("location")  # copied exactly as-is, no cleanup
    work_preference = profile_data.get("work_preference")

    city = normalize_city(location_raw)
    region = city_region(city)

    cursor.execute('''
        INSERT OR REPLACE INTO artist_documents
            (artist_id, name, category, niche, location_raw,
             location_city, location_region, work_preference, data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        artist_id, name, category, niche, location_raw,
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

    db_path = REPO_ROOT / "artists.db"
    db_conn = setup_sqlite_db(db_path)

    json_files = sorted(profiles_path.glob("*.json"))

    if not json_files:
        print("No profile JSON files found. Did you run docx-to-json.py first?")
    else:
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            insert_into_db(db_conn, profile_data, json_file.stem)
            derived_category = category_from_filename(json_file.stem)
            print(f"Inserted into DB: {profile_data.get('artist_id')} - {profile_data.get('name')} "
                  f"[file={json_file.stem}, category={derived_category}, "
                  f"niche={profile_data.get('category')!r}, "
                  f"city={normalize_city(profile_data.get('location'))}]")

        db_conn.close()
        print(f"\nDatabase build complete. Inserted {len(json_files)} records into {db_path}.")