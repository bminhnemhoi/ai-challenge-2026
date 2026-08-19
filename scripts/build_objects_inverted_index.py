#!/usr/bin/env python3
"""
Build Inverted Index for BTC OpenImages Objects (177k keyframes).
Scans all 873 video folders in data/objects/ and compiles a single inverted index
mapping entity name -> list of [raw_frame_idx, confidence].
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

def process_video_folder(args: Tuple[str, str, Dict[str, int], float]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Processes all JSON files in one video folder.
    Returns: dict mapping entity_name -> [(raw_idx, conf), ...]
    """
    video_id, video_dir_path, key_to_idx, min_conf = args
    local_index: Dict[str, List[Tuple[int, float]]] = {}
    
    if not os.path.exists(video_dir_path):
        return local_index

    try:
        filenames = os.listdir(video_dir_path)
    except OSError:
        return local_index

    for fname in filenames:
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5] # e.g. "001"
        key = f"{video_id}_{stem}"
        if key not in key_to_idx:
            try:
                n_val = int(stem)
                key = f"{video_id}_{n_val:03d}"
            except ValueError:
                pass
            if key not in key_to_idx:
                continue

        raw_idx = key_to_idx[key]
        fpath = os.path.join(video_dir_path, fname)
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        entities = data.get("detection_class_entities", [])
        scores = data.get("detection_scores", [])

        # Deduplicate per frame: keep max confidence per entity
        seen_entities: Dict[str, float] = {}
        for ent, score in zip(entities, scores):
            try:
                s_float = float(score)
            except (ValueError, TypeError):
                continue
            if s_float < min_conf:
                continue
            
            ent_clean = str(ent).strip().lower()
            if not ent_clean or len(ent_clean) < 2:
                continue
                
            if ent_clean not in seen_entities or s_float > seen_entities[ent_clean]:
                seen_entities[ent_clean] = s_float

        for ent_clean, s_float in seen_entities.items():
            if ent_clean not in local_index:
                local_index[ent_clean] = []
            local_index[ent_clean].append((raw_idx, round(s_float, 2)))

    return local_index

def build_inverted_index(data_dir: str = "data", min_confidence: float = 0.35, output_file: str = "data/objects_inverted_index.json"):
    start_time = time.time()
    print(f"🚀 Building BTC Objects Inverted Index from {data_dir} (min_confidence={min_confidence})...")

    # 1. Load metadata.json to map (video_id, stem) -> raw_frame_idx (0..177320)
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Please ensure metadata.json is generated.")

    print("Loading metadata.json...")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    key_to_idx: Dict[str, int] = {}
    for idx, item in enumerate(metadata):
        v_id = item["video_id"]
        fname = item.get("frame_filename", "")
        stem = Path(fname).stem if fname else f"{item.get('n', 1):03d}"
        key_to_idx[f"{v_id}_{stem}"] = idx
        try:
            n_int = int(stem)
            key_to_idx[f"{v_id}_{n_int:03d}"] = idx
        except ValueError:
            pass

    print(f"✅ Indexed {len(key_to_idx)} frame lookup keys for {len(metadata)} total keyframes.")

    # 2. Find all video folders in data/objects/
    objects_dir = os.path.join(data_dir, "objects")
    if not os.path.exists(objects_dir):
        raise FileNotFoundError(f"Missing {objects_dir}")

    video_folders = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]
    print(f"Found {len(video_folders)} video folders to process in parallel...")

    tasks = [
        (v_id, os.path.join(objects_dir, v_id), key_to_idx, min_confidence)
        for v_id in video_folders
    ]

    # 3. Parallel extraction across CPU cores
    global_index: Dict[str, List[Tuple[int, float]]] = {}
    num_workers = min(os.cpu_count() or 4, 16)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_video_folder, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            local_idx = future.result()
            for ent, records in local_idx.items():
                if ent not in global_index:
                    global_index[ent] = []
                global_index[ent].extend(records)
            if idx % 200 == 0 or idx == len(tasks):
                print(f"Processed {idx}/{len(tasks)} videos ({len(global_index)} unique object classes found)...")

    # Sort posting lists by raw_idx
    total_postings = 0
    for ent in global_index:
        global_index[ent].sort(key=lambda x: x[0])
        total_postings += len(global_index[ent])

    # 4. Save to JSON
    print(f"Writing inverted index to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(global_index, f, separators=(',', ':'))

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"🎉 Done in {elapsed:.2f}s! Total {len(global_index)} unique object entities with {total_postings} frame postings.")
    print(f"💾 Inverted Index File Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    output_path = os.path.join(data_dir, "objects_inverted_index.json")
    build_inverted_index(data_dir=data_dir, min_confidence=0.35, output_file=output_path)
