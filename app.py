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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KEYFRAMES_DIR = os.path.join(BASE_DIR, "keyframes")

retriever = TextualKISRetriever(data_dir=DATA_DIR, use_siglip_only=True, use_clip_only=False, use_siglip_version="siglip2")
vqa_engine = VisualQAEngine(data_dir=DATA_DIR)
trake_engine = TRAKEEngine(data_dir=DATA_DIR)

class KISQueryRequest(BaseModel):
    query: str
    top_k: int = 100
    nms_gap: int = 5
    max_per_video: int = 2
    use_reranker: bool = False
    use_spatial_grid: bool = True

class SubmissionExportRequest(BaseModel):
    query_id: str = "query_1"
    predictions: List[dict] # list of {"video_id": ..., "frame_idx": ..., "answer": ...}

@app.on_event("startup")
def startup_event():
    print("🚀 Initializing Search Engine Backend (Pre-loading SigLIP 2 & Index into Memory)...")
    try:
        retriever.load_index_and_model()
        print("✅ Backend SigLIP 2 Model & Vector Index 100% Active in RAM/MPS Memory!")
    except Exception as e:
        print(f"❌ Warning on startup load: {e}")

@app.post("/api/search/kis")
def search_kis(req: KISQueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        results = retriever.search(
            req.query,
            top_k=req.top_k,
            nms_frame_gap=req.nms_gap,
            max_per_video=req.max_per_video,
            use_reranker=req.use_reranker,
            use_spatial_grid=req.use_spatial_grid
        )
        return {
            "query": req.query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import asyncio

@app.post("/api/search/kis_stream")
async def search_kis_stream(req: KISQueryRequest):
    """
    Progressive Mini-Batch Streaming Endpoint:
    - Streams 4-image mini-batches over TCP socket for instant left-to-right visual rendering.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    async def stream_generator():
        try:
            loop = asyncio.get_event_loop()

            # 1. Fast Stage 1 Vector Search (<15ms)
            stage1_results = await loop.run_in_executor(
                None,
                lambda: retriever.search(
                    req.query,
                    top_k=req.top_k,
                    nms_frame_gap=req.nms_gap,
                    max_per_video=req.max_per_video,
                    use_reranker=False,
                    use_spatial_grid=req.use_spatial_grid
                )
            )

            # Progressive 4-Image Mini-Batch Streaming (Instant left-to-right rendering!)
            batch_size = 4
            total_items = len(stage1_results)
            for i in range(0, total_items, batch_size):
                sub_batch = stage1_results[i : i + batch_size]
                chunk = {
                    "type": "stage1_batch",
                    "batch_idx": i // batch_size,
                    "total_count": total_items,
                    "results": sub_batch
                }
                yield json.dumps(chunk) + "\n"
                await asyncio.sleep(0.005)  # Immediate TCP Socket Flush for progressive cards!

            # 2. Stage 2 VLM Re-Ranking
            if req.use_reranker:
                final_results = await loop.run_in_executor(
                    None,
                    lambda: retriever.search(
                        req.query,
                        top_k=req.top_k,
                        nms_frame_gap=req.nms_gap,
                        max_per_video=req.max_per_video,
                        use_reranker=True,
                        use_spatial_grid=req.use_spatial_grid
                    )
                )
                prefetch_results_images(final_results)

                # Stream Re-Ranked chunks in 4-image mini-batches
                for i in range(0, len(final_results), batch_size):
                    sub_batch = final_results[i : i + batch_size]
                    chunk2 = {
                        "type": "rerank_batch",
                        "batch_idx": i // batch_size,
                        "total_count": len(final_results),
                        "results": sub_batch
                    }
                    yield json.dumps(chunk2) + "\n"
                    await asyncio.sleep(0.005)

        except Exception as e:
            err_chunk = {"type": "error", "message": str(e)}
            yield json.dumps(err_chunk) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

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

from fastapi.responses import RedirectResponse

HF_DATASET_RESOLVE_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

@app.get("/api/keyframe/{video_id}/{filename}")
def get_keyframe_image(video_id: str, filename: str):
    """
    Pure Cloud Hugging Face Streaming (Zero Local Disk Footprint).
    Redirects browser directly to Hugging Face CDN so browser caches in memory/RAM.
    """
    hf_url = f"{HF_DATASET_RESOLVE_URL}/{video_id}/{filename}"
    return RedirectResponse(url=hf_url, status_code=302)

# Serve static frontend files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
