# generate_metadata_from_filename.py
import os
import json
import requests
import time

DB_FOLDER = "db"
OUTPUT_FILE = "db_metadata.json"
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "VisualzPlaceIdentifier/1.0 (vortexcode06@gmail.com)"
}

def get_coords_from_place(place_name):
    try:
        response = requests.get(
            GEOCODE_URL,
            params={"q": place_name, "format": "json", "limit": 1},
            headers=HEADERS
        )
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"❌ Geocoding error for '{place_name}': {e}")
    return None, None

def extract_place_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if name.endswith("_general"):
        return None  # Marked as general image
    return name.replace("_", " ")

def generate_metadata():
    metadata = {}
    files = [f for f in os.listdir(DB_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    for fname in files:
        place = extract_place_from_filename(fname)

        if not place:
            metadata[fname] = {"lat": None, "lon": None}
            continue

        lat, lon = get_coords_from_place(place)
        if lat is not None and lon is not None:
            metadata[fname] = {"lat": lat, "lon": lon}
        else:
            metadata[fname] = {"lat": None, "lon": None}

        time.sleep(1)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    generate_metadata()
