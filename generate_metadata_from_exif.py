import os
import json
from PIL import Image
import piexif

DB_FOLDER = "db"
OUTPUT_FILE = "db_metadata.json"

def dms_to_dd(dms, ref):
    degrees, minutes, seconds = dms
    dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        dd *= -1
    return dd

def get_lat_lon(exif_data):
    try:
        gps_info = exif_data.get("GPS", {})
        if not gps_info:
            return None, None

        gps_latitude = gps_info.get(piexif.GPSIFD.GPSLatitude)
        gps_latitude_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef).decode()
        gps_longitude = gps_info.get(piexif.GPSIFD.GPSLongitude)
        gps_longitude_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef).decode()

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = dms_to_dd([val[0] / val[1] for val in gps_latitude], gps_latitude_ref)
            lon = dms_to_dd([val[0] / val[1] for val in gps_longitude], gps_longitude_ref)
            return lat, lon

    except Exception as e:
        print(f"[WARN] EXIF error: {e}")
    return None, None

def generate_metadata():
    metadata = {}
    image_extensions = ('.jpg', '.jpeg', '.png')

    for fname in os.listdir(DB_FOLDER):
        if not fname.lower().endswith(image_extensions):
            continue

        fpath = os.path.join(DB_FOLDER, fname)
        try:
            img = Image.open(fpath)
            exif_dict = piexif.load(img.info.get('exif', b''))

            lat, lon = get_lat_lon(exif_dict)
            if lat is not None and lon is not None:
                print(f"✅ {fname}: ({lat}, {lon})")
                metadata[fname] = {"lat": lat, "lon": lon}
            else:
                print(f"⚠️ {fname}: No GPS data found.")
                metadata[fname] = {"lat": None, "lon": None}

        except Exception as e:
            print(f"❌ {fname}: Error reading EXIF - {e}")
            metadata[fname] = {"lat": None, "lon": None}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved GPS metadata to {OUTPUT_FILE} for {len(metadata)} images.")

if __name__ == "__main__":
    generate_metadata()
