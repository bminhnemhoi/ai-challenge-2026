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

from src.retriever import TextualKISRetriever
from src.evaluator import AIC2026Evaluator

app = FastAPI(title="AIC 2026 Local Search Engine", version="1.0")

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

@app.post("/api/search/kis")
def search_kis(req: KISQueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        results = retriever.search(req.query, top_k=req.top_k, nms_frame_gap=req.nms_gap)
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

@app.get("/api/keyframe/{video_id}/{filename}")
def get_keyframe_image(video_id: str, filename: str):
    image_path = os.path.join(KEYFRAMES_DIR, video_id, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

# Serve static frontend files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
