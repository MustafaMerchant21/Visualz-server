from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, time, json
import faiss
import numpy as np
from clip_utils import get_clip_embedding
from build_index import build_index

# build_index()

# Prevent MKL errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

with open("db_metadata.json", "r") as f:
    db_metadata = json.load(f)


UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
INDEX_FILE = 'db_index.faiss'
PATHS_FILE = 'db_paths.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load FAISS index and paths
index = faiss.read_index(INDEX_FILE)
with open(PATHS_FILE, 'r') as f:
    db_paths = json.load(f)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    timestamp = int(time.time())
    final_filename = f"{timestamp}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)

    try:
        image.save(image_path)
        print(f"[INFO] Image saved at {image_path}")

        print("[INFO] Getting CLIP embedding...")
        embedding = get_clip_embedding(image_path).reshape(1, -1).astype("float32")
        print("[INFO] Embedding generated")

        # FAISS search
        top_k = min(4, index.ntotal)
        D, I = index.search(embedding, top_k)
        print(f"[INFO] FAISS search result D: {D}")
        print(f"[INFO] FAISS search result I: {I}")

        matches = []
        threshold = 0.6 
        for idx_raw, dist_raw in zip(I[0], D[0]):
            try:
                idx = int(idx_raw)
                dist = float(dist_raw)
                if idx == -1 or dist == -3.4028235e+38:
                    continue

                # ✅ Only consider as match if distance >= 0.6
                if dist >= threshold and 0 <= idx < len(db_paths):
                    image_name = os.path.basename(db_paths[str(idx)])
                    meta = db_metadata.get(image_name, {})

                    matches.append({
                        "path": image_name,
                        "distance": dist,
                        "lat": meta.get("lat"),
                        "lon": meta.get("lon")
                    })

            except Exception as e:
                print(f"[WARN] Skipping result idx={idx_raw}, error: {e}")

        # Return matches or no match message
        if not matches:
            return jsonify({
                "message": "Uploaded, but no good matches found.",
                "filename": final_filename,
                "matches": []
            }), 200
        print(f"####\n{matches}\n####")
        return jsonify({
            "message": "Uploaded",
            "filename": final_filename,
            "matches": matches
        })

    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/db/<filename>')
def serve_db(filename):
    return send_from_directory(DB_FOLDER, filename)

@app.route('/api/test')
def test():
    return jsonify({"status": "API is working ✅"})

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
