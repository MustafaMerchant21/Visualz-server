from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, time, json
import faiss
from clip_utils import get_clip_embedding, predict_place_name

# Prevent MKL errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOGETHER_API_KEY"] = "tgp_v1_HDWt-zkoh6PtS5aCanHP-MxOuOJaH7mQC-5DNdCbpXE"

app = Flask(__name__)
CORS(app, origins=["https://visualz-server.onrender.com"])

UPLOAD_FOLDER = 'uploads'
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

        image_url_match = db_paths[str(idx)]
        meta = db_metadata.get(os.path.basename(image_url_match), {})

        matches.append({
            "path": image_url_match,
            "distance": dist,
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
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

@app.route('/api/test')
def test():
    return jsonify({"status": "API is working ✅"})

if __name__ == "__main__":
    app.run(debug=True)
