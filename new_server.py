from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, time, json
import faiss
import numpy as np
from clip_utils import get_clip_embedding, predict_place_name
from build_index import build_index
import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from clip_utils import predict_place_name 

# Prevent MKL errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOGETHER_API_KEY"] = "tgp_v1_HDWt-zkoh6PtS5aCanHP-MxOuOJaH7mQC-5DNdCbpXE"
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
INDEX_FILE = 'db_index.faiss'
PATHS_FILE = 'db_paths.json'
META_FILE = 'db_metadata.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

index = faiss.read_index(INDEX_FILE)
with open(PATHS_FILE, 'r') as f:
    db_paths = json.load(f)
with open(META_FILE, 'r') as f:
    db_metadata = json.load(f)

def extract_gps_from_exif(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
                break

        def convert_to_degrees(val):
            d = float(val[0][0]) / float(val[0][1])
            m = float(val[1][0]) / float(val[1][1])
            s = float(val[2][0]) / float(val[2][1])
            return d + (m / 60.0) + (s / 3600.0)

        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = convert_to_degrees(gps_info['GPSLatitude'])
            if gps_info.get('GPSLatitudeRef') == 'S': lat *= -1
            lon = convert_to_degrees(gps_info['GPSLongitude'])
            if gps_info.get('GPSLongitudeRef') == 'W': lon *= -1
            return lat, lon
    except:
        pass
    return None, None

def get_coords_from_place(place_name):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        response = requests.get(url, params={"q": place_name, "format": "json", "limit": 1}, headers={"User-Agent": "VisualzPlaceRecognizer/1.0"})
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    timestamp = int(time.time())
    final_filename = f"{timestamp}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
    image.save(image_path)
    start_time = time.time()
    predicted_place_name = predict_place_name(image_path)
    end_time = time.time()
    print(f"[Together AI] Predicted: {predicted_place_name}")
    print(f"[Together AI] Time taken: {end_time-start_time:.2f}seconds")

    google_maps_link = None
    if predicted_place_name:
        google_maps_link = f"https://www.google.com/maps/search/{predicted_place_name.replace(' ', '+')}"


    embedding = get_clip_embedding(image_path).reshape(1, -1).astype("float32")
    top_k = min(4, index.ntotal)
    D, I = index.search(embedding, top_k)

    matches = []
    for idx_raw, dist_raw in zip(I[0], D[0]):
        idx = int(idx_raw)
        dist = float(dist_raw)
        if idx == -1 or dist < 0.6:
            continue

        image_name = os.path.basename(db_paths[str(idx)])
        meta = db_metadata.get(image_name, {})

        # place_name = meta.get("place", os.path.splitext(image_name)[0])
        # google_maps_link = f"https://www.google.com/maps/search/{place_name.replace(' ', '+')}"

        matches.append({
            "path": image_name,
            "distance": dist,
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            # "place": place_name,
            "map_link": google_maps_link
        })

    return jsonify({
        "message": "Uploaded",
        "filename": final_filename,
        "matches": matches,
        "predicted_place": predicted_place_name,
        "google_maps_link": google_maps_link
    })



@app.route('/uploads/<filename>')
def serve_uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/db/<filename>')
def serve_db(filename):
    return send_from_directory(DB_FOLDER, filename)

@app.route('/api/test')
def test():
    return jsonify({"status": "API is working ✅"})

if __name__ == "__main__":
    app.run(debug=True)