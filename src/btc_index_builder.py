import os
import json
import csv
import time
import numpy as np

BASE_DIR = "/Users/xuannguyen/Desktop/AI-Challenge-2026"
DATA_DIR = os.path.join(BASE_DIR, "data")
CLIP_DIR = os.path.join(DATA_DIR, "clip-features-32")
MAP_DIR = os.path.join(DATA_DIR, "map-keyframes")
KEYFRAMES_DIR = os.path.join(BASE_DIR, "keyframes")

def build_official_btc_index():
    print("=== Building Official BTC Batch 1 CLIP Vector Index ===", flush=True)
    start_time = time.time()

    if not os.path.exists(CLIP_DIR) or not os.path.exists(MAP_DIR):
        print("Error: BTC features or map-keyframes directories not found!", flush=True)
        return

    npy_files = sorted([f for f in os.listdir(CLIP_DIR) if f.endswith(".npy")])
    print(f"Found {len(npy_files)} video feature files in BTC clip-features-32.", flush=True)

    metadata = []
    embeddings_list = []
    total_frames = 0

    for npy_file in npy_files:
        video_id = os.path.splitext(npy_file)[0]
        npy_path = os.path.join(CLIP_DIR, npy_file)
        csv_path = os.path.join(MAP_DIR, f"{video_id}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: Missing mapping CSV for {video_id}, skipping...", flush=True)
            continue

        try:
            vecs = np.load(npy_path).astype(np.float32) # Shape: (N, 512)
            # L2 Normalize vectors
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs_norm = vecs / norms
        except Exception as e:
            print(f"Error loading {npy_file}: {e}", flush=True)
            continue

        # Parse map-keyframes CSV
        frames_info = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames_info.append(row)

        num_vecs = len(vecs_norm)
        num_csv = len(frames_info)

        if num_vecs != num_csv:
            # Handle slight row count mismatch gracefully
            count = min(num_vecs, num_csv)
        else:
            count = num_vecs

        for i in range(count):
            row = frames_info[i]
            frame_n = int(row.get("n", i + 1))
            actual_frame_idx = int(float(row.get("frame_idx", 0)))
            frame_filename = f"{frame_n:03d}.jpg"
            
            rel_path = f"{video_id}/{frame_filename}"
            abs_path = os.path.join(KEYFRAMES_DIR, video_id, frame_filename)

            metadata.append({
                "video_id": video_id,
                "n": frame_n,
                "frame_filename": frame_filename,
                "frame_idx": actual_frame_idx,
                "pts_time": float(row.get("pts_time", 0.0)),
                "fps": float(row.get("fps", 25.0)),
                "rel_path": rel_path,
                "abs_path": abs_path
            })

        embeddings_list.append(vecs_norm[:count])
        total_frames += count

    print(f"Combining {len(embeddings_list)} videos into single matrix...", flush=True)
    all_embeddings = np.vstack(embeddings_list)

    elapsed = time.time() - start_time
    print(f"Combined {total_frames} keyframe vectors. Output shape: {all_embeddings.shape} in {elapsed:.2f}s.", flush=True)

    out_meta = os.path.join(DATA_DIR, "metadata.json")
    out_emb = os.path.join(DATA_DIR, "embeddings.npy")

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    np.save(out_emb, all_embeddings)
    print(f"Saved merged index to '{out_emb}' and metadata to '{out_meta}'.", flush=True)

if __name__ == "__main__":
    build_official_btc_index()
