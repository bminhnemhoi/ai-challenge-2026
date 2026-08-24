#!/usr/bin/env python3
"""
Build Inverted Index for BTC OpenImages Objects (177k keyframes).
Compiles a single inverted index mapping entity name -> list of
[raw_frame_idx, confidence].

Reads either the unpacked data/objects/ tree or, if that is absent,
data/objects-aic25-b1.zip directly -- download_data.py leaves the archive
zipped, so requiring the unpacked tree meant this script only ever ran for
whoever had extracted the 610 MB by hand.
"""

import os
import json
import time
import zipfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any


def _accumulate(local_index, raw_idx, data, min_conf):
    """Keep the best confidence per entity for one frame."""
    seen: Dict[str, float] = {}
    for ent, score in zip(
        data.get("detection_class_entities", []), data.get("detection_scores", [])
    ):
        try:
            s = float(score)
        except (ValueError, TypeError):
            continue
        if s < min_conf:
            continue
        ent_clean = str(ent).strip().lower()
        if len(ent_clean) < 2:
            continue
        if ent_clean not in seen or s > seen[ent_clean]:
            seen[ent_clean] = s
    for ent_clean, s in seen.items():
        local_index.setdefault(ent_clean, []).append((raw_idx, round(s, 2)))


def process_zip_chunk(args):
    """One worker opens the archive once and handles a slice of its members."""
    zip_path, members, key_to_idx, min_conf = args
    local_index: Dict[str, List[Tuple[int, float]]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in members:
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            video_id, stem = parts[-2], Path(name).stem
            raw_idx = key_to_idx.get(f"{video_id}_{stem}")
            if raw_idx is None:
                try:
                    raw_idx = key_to_idx.get(f"{video_id}_{int(stem):03d}")
                except ValueError:
                    raw_idx = None
            if raw_idx is None:
                continue
            try:
                data = json.loads(zf.read(name))
            except Exception:
                continue
            _accumulate(local_index, raw_idx, data, min_conf)
    return local_index

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

    # 2. Prefer the unpacked tree; fall back to the archive download_data leaves behind
    objects_dir = os.path.join(data_dir, "objects")
    zip_path = os.path.join(data_dir, "objects-aic25-b1.zip")
    num_workers = min(os.cpu_count() or 4, 16)

    if os.path.isdir(objects_dir):
        video_folders = [
            d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))
        ]
        print(f"Found {len(video_folders)} video folders to process in parallel...")
        worker, tasks = process_video_folder, [
            (v_id, os.path.join(objects_dir, v_id), key_to_idx, min_confidence)
            for v_id in video_folders
        ]
        unit = "videos"
    elif os.path.isfile(zip_path):
        print(f"No {objects_dir}/ — reading {os.path.basename(zip_path)} directly ...")
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".json")]
        by_video = defaultdict(list)
        for n in members:
            parts = Path(n).parts
            if len(parts) >= 2:
                by_video[parts[-2]].append(n)
        # whole videos stay together, so one worker opens the archive once per chunk
        vids = sorted(by_video)
        chunks = [vids[i::num_workers] for i in range(num_workers)]
        worker, tasks = process_zip_chunk, [
            (zip_path, [m for v in chunk for m in by_video[v]], key_to_idx, min_confidence)
            for chunk in chunks
            if chunk
        ]
        print(f"{len(members):,} detection files across {len(vids)} videos, {len(tasks)} workers")
        unit = "chunks"
    else:
        raise FileNotFoundError(
            f"Need either {objects_dir}/ or {zip_path}.\n"
            f"Run: python scripts/download_data.py"
        )

    # 3. Parallel extraction across CPU cores
    global_index: Dict[str, List[Tuple[int, float]]] = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            local_idx = future.result()
            for ent, records in local_idx.items():
                if ent not in global_index:
                    global_index[ent] = []
                global_index[ent].extend(records)
            if idx % max(1, len(tasks) // 10) == 0 or idx == len(tasks):
                print(f"Processed {idx}/{len(tasks)} {unit} ({len(global_index)} unique object classes found)...")

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
    import argparse

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(base_dir, "data"))
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build_inverted_index(
        data_dir=a.data,
        min_confidence=a.min_confidence,
        output_file=a.out or os.path.join(a.data, "objects_inverted_index.json"),
    )
