# build_index.py
import os
import faiss
import json
import numpy as np
from clip_utils import get_clip_embedding

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
            paths[len(paths)] = fname  # Use len(paths) to keep consistent indexing

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            num_skipped += 1
            continue

    faiss.write_index(index, "db_index.faiss")
    with open("db_paths.json", "w") as f:
        json.dump(paths, f)

    print(f"\n✅ Indexed {len(paths)} images.")
    print(f"⚠️ Skipped {num_skipped} images due to errors.")

if __name__ == "__main__":
    build_index()
