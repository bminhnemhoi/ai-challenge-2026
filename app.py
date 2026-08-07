import os
import io
import csv
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.task1_kis import TextualKISRetriever
from src.task2_vqa import VisualQAEngine
from src.task3_trake import TRAKEEngine
from src.core import AIC2026Evaluator

app = FastAPI(title="AIC 2026 Unified Search Engine", version="2.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "/Users/xuannguyen/Desktop/AI-Challenge-2026/data"
KEYFRAMES_DIR = "/Users/xuannguyen/Desktop/AI-Challenge-2026/keyframes"

retriever = TextualKISRetriever(data_dir=DATA_DIR)
vqa_engine = VisualQAEngine(data_dir=DATA_DIR)
trake_engine = TRAKEEngine(data_dir=DATA_DIR)

class KISQueryRequest(BaseModel):
    query: str
    top_k: int = 100
    nms_gap: int = 5

class SubmissionExportRequest(BaseModel):
    query_id: str = "query_1"
    predictions: List[dict] # list of {"video_id": ..., "frame_idx": ..., "answer": ...}

@app.on_event("startup")
def startup_event():
    print("Initializing Search Engine Backend...")
    if os.path.exists(os.path.join(DATA_DIR, "embeddings.npy")):
        try:
            retriever.load_index_and_model()
            print("Backend index loaded successfully.")
        except Exception as e:
            print(f"Warning on startup load: {e}")
    else:
        print("Data index not found. Please run `index_builder.py` first.")

from concurrent.futures import ThreadPoolExecutor
image_prefetch_executor = ThreadPoolExecutor(max_workers=16)

def _prefetch_single_image(video_id: str, filename: str):
    image_path = os.path.join(KEYFRAMES_DIR, video_id, filename)
    if os.path.exists(image_path):
        return
    cached_path = os.path.join(CACHE_DIR, video_id, filename)
    if os.path.exists(cached_path):
        return
    
    hf_url = f"{HF_DATASET_RESOLVE_URL}/{video_id}/{filename}"
    try:
        req = urllib.request.Request(
            hf_url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            os.makedirs(os.path.join(CACHE_DIR, video_id), exist_ok=True)
            with open(cached_path, "wb") as f:
                f.write(content)
    except Exception:
        pass

def prefetch_results_images(results: List[dict]):
    for item in results:
        v_id = item.get("video_id")
        f_name = item.get("frame_filename")
        if v_id and f_name:
            image_prefetch_executor.submit(_prefetch_single_image, v_id, f_name)

@app.post("/api/search/kis")
def search_kis(req: KISQueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        results = retriever.search(req.query, top_k=req.top_k, nms_frame_gap=req.nms_gap)
        # Fire 16-worker parallel background prefetching for all search result keyframes
        prefetch_results_images(results)
        return {
            "query": req.query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export_submission")
def export_submission(req: SubmissionExportRequest):
    """
    Exports predictions in official AIC 2026 submission CSV format.
    Format: video_id, frame_idx (or answer for Q&A)
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header format depends on Task
    for p in req.predictions:
        v_id = p.get("video_id", "")
        f_idx = p.get("frame_idx", "")
        ans = p.get("answer", None)
        
        if ans is not None:
            writer.writerow([v_id, f_idx, ans])
        else:
            writer.writerow([v_id, f_idx])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=submission_{req.query_id}.csv"}
    )

import urllib.request

HF_DATASET_RAW_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/raw/main"
CACHE_DIR = os.path.join(DATA_DIR, ".cache_keyframes")
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_placeholder_svg(video_id: str, filename: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="480" height="270" viewBox="0 0 480 270">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#1e1b4b"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>
      </defs>
      <rect width="480" height="270" fill="url(#bg)"/>
      <rect x="12" y="12" width="456" height="246" rx="10" fill="none" stroke="rgba(99, 102, 241, 0.4)" stroke-dasharray="6 4" stroke-width="2"/>
      <g transform="translate(216, 65)">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      </g>
      <text x="240" y="145" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" fill="#f8fafc" text-anchor="middle">{video_id}</text>
      <text x="240" y="175" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#94a3b8" text-anchor="middle">Frame: {filename}</text>
      <text x="240" y="205" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#06b6d4" text-anchor="middle">VECTOR INDEX MATCHED</text>
    </svg>"""

HF_DATASET_RESOLVE_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

@app.get("/api/keyframe/{video_id}/{filename}")
def get_keyframe_image(video_id: str, filename: str):
    # 1. Check direct local keyframe path
    image_path = os.path.join(KEYFRAMES_DIR, video_id, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
        
    # 2. Check cached keyframe path (Instant < 1ms response)
    cached_path = os.path.join(CACHE_DIR, video_id, filename)
    if os.path.exists(cached_path):
        return FileResponse(cached_path)

    # 3. Fetch binary JPEG directly from Hugging Face CDN, cache locally, and serve
    hf_url = f"{HF_DATASET_RESOLVE_URL}/{video_id}/{filename}"
    try:
        req = urllib.request.Request(
            hf_url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            # Save image to local cache for instant future loads
            os.makedirs(os.path.join(CACHE_DIR, video_id), exist_ok=True)
            with open(cached_path, "wb") as f:
                f.write(content)
            return Response(content=content, media_type="image/jpeg")
    except Exception as e:
        # Fallback to SVG placeholder on network timeout/error
        svg_content = generate_placeholder_svg(video_id, filename)
        return Response(content=svg_content, media_type="image/svg+xml")

# Serve static frontend files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
