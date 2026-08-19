from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from .config import DEFAULT_DATA_DIR, MODEL_ID, RETRIEVER_TOP_K
from .vqa_engine_VLM import VisualQAEngine

engine: Optional[VisualQAEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the VLM + retriever index once at startup, not per-request --
    # loading Qwen3-VL takes several seconds and several GB of VRAM.
    global engine
    engine = VisualQAEngine(data_dir=DEFAULT_DATA_DIR)
    engine.load_models()
    yield
    engine = None


app = FastAPI(title="AIC2026 - Task2 VQA Engine", lifespan=lifespan)


class VQARequest(BaseModel):
    question: str
    video_id: Optional[str] = None  # optional filter to a specific video
    top_k: int = RETRIEVER_TOP_K

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question không được để trống")
        return v


class VQAResultItem(BaseModel):
    video_id: str
    frame_idx: int
    answer: str
    score: Optional[float] = None


class VQAResponse(BaseModel):
    results: List[VQAResultItem]


@app.post("/api/search/vqa", response_model=VQAResponse)
def vqa_endpoint(req: VQARequest) -> VQAResponse:
    try:
        records = engine.answer_question(question=req.question, video_id=req.video_id, top_k=req.top_k)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi xử lý truy vấn — xem log server")

    return VQAResponse(results=[
        VQAResultItem(video_id=r["video_id"], frame_idx=r["frame_idx"], answer=r["answer"], score=r.get("score"))
        for r in records
    ])


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": engine is not None and engine._is_loaded}
