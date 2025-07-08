# build_index.py
import os
import faiss
import json
import numpy as np
from clip_utils import get_clip_embedding
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def build_index():
    DB_FOLDER = "db"
    index = faiss.IndexFlatIP(512)  # CLIP ViT-B-32 = 512 dims
    paths = {}

    print("Building index...")
    num_skipped = 0

    for i, fname in enumerate(os.listdir(DB_FOLDER)):
        fpath = os.path.join(DB_FOLDER, fname)

        try:
            embedding = get_clip_embedding(fpath)
            if embedding is None:
                print(f"⚠️ Skipped (no embedding): {fname}")
                num_skipped += 1
                continue
            print(f"✅ Embedded Successful: {fname}")
            embedding = embedding.astype('float32')
            index.add(embedding.reshape(1, -1))

            upload_result = cloudinary.uploader.upload(fpath)
            image_url = upload_result['secure_url']
            paths[len(paths)] = image_url

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            num_skipped += 1
            continue

    faiss.write_index(index, "db_index.faiss")
    with open("db_paths.json", "w") as f:
        json.dump(paths, f, indent=2)

    print(f"\n✅ Indexed {len(paths)} images.")
    print(f"⚠️ Skipped {num_skipped} images due to errors.")

if __name__ == "__main__":
    build_index()
