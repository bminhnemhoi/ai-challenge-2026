from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

# Ensure parent `src` directory is always in sys.path
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

try:
    from .config import DEFAULT_DATA_DIR, DEFAULT_VIDEO_DIR, MODEL_ID, RETRIEVER_TOP_K
    from .frame_utils import extract_frame, resolve_video_path
    from .postprocessor import clean_answer
    from .vqa_engine_VLM import VisualQAEngine
except ImportError:
    from config import DEFAULT_DATA_DIR, DEFAULT_VIDEO_DIR, MODEL_ID, RETRIEVER_TOP_K
    from frame_utils import extract_frame, resolve_video_path
    from postprocessor import clean_answer
    from vqa_engine_VLM import VisualQAEngine

engine: Optional[VisualQAEngine] = None

# Đặt AIC_USE_MOCK_RETRIEVER=1 khi src/task1_kis (Task 1) chưa sẵn sàng, để
# test riêng luồng Gemini VLM (frame extraction + OCR/object context + VLM).
# Khi Task 1 đã có thật, chỉ cần bỏ biến env này (hoặc set =0) -- không cần
# sửa code, VisualQAEngine sẽ tự import TextualKISRetriever thật.
_USE_MOCK_RETRIEVER = os.environ.get("AIC_USE_MOCK_RETRIEVER", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the VLM + retriever index once at startup, not per-request --
    # loading Qwen3-VL takes several seconds and several GB of VRAM.
    global engine
    retriever = None
    if _USE_MOCK_RETRIEVER:
        try:
            from .mock_retriever import MockRetriever
        except ImportError:
            from mock_retriever import MockRetriever
        retriever = MockRetriever()
        print("[app] AIC_USE_MOCK_RETRIEVER=1 -- dùng MockRetriever, KHÔNG dùng Task 1 thật.")

    engine = VisualQAEngine(data_dir=DEFAULT_DATA_DIR, retriever=retriever)
    engine.load_models()
    yield
    engine = None


app = FastAPI(title="AIC2026 - Task2 VQA Engine", lifespan=lifespan)


class VQARequest(BaseModel):
    question: str
    video_id: Optional[str] = None  # optional filter to a specific video
    top_k: int = RETRIEVER_TOP_K
    # CHỈ dùng để debug/tự kiểm thử: ghi đè câu trả lời của VLM bằng câu bạn
    # tự gõ, để so sánh nhanh với đáp án hệ thống tự sinh ra. KHÔNG dùng
    # trường này khi nộp bài thật -- BTC chấm dựa trên answer do hệ thống
    # TỰ ĐỘNG sinh ra (xem README mục 1, công thức R-Score), không chấp
    # nhận answer gõ tay.
    manual_answer: Optional[str] = None

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
    is_manual_override: bool = False  # True nếu answer bị ghi đè bằng manual_answer (chỉ để debug)


class VQAResponse(BaseModel):
    results: List[VQAResultItem]


@app.post("/api/search/vqa", response_model=VQAResponse)
def vqa_endpoint(req: VQARequest) -> VQAResponse:
    try:
        records = engine.answer_question(question=req.question, video_id=req.video_id, top_k=req.top_k)

        # CHỈ để debug: nếu người dùng tự gõ manual_answer (khác rỗng và khác 'string' mặc định của Swagger)
        is_override = bool(
            req.manual_answer and req.manual_answer.strip() and req.manual_answer.strip() != "string"
        )
        if is_override:
            override_text = clean_answer(req.manual_answer)
            for r in records:
                r["answer"] = override_text


    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi xử lý truy vấn — xem log server")

    return VQAResponse(results=[
        VQAResultItem(
            video_id=r["video_id"],
            frame_idx=r["frame_idx"],
            answer=r["answer"],
            score=r.get("score"),
            is_manual_override=is_override,
        )
        for r in records
    ])


@app.get("/api/frame")
def get_frame(video_id: str, frame_idx: int):
    """Trả về ảnh JPEG của 1 frame cụ thể -- để xem trực tiếp bằng trình
    duyệt (dán URL vào thanh địa chỉ) hoặc qua Swagger UI, phục vụ việc
    tự kiểm tra/nhập câu trả lời thủ công thay vì chỉ tin VLM.

    Ví dụ:
        http://127.0.0.1:8000/api/frame?video_id=L21_V001&frame_idx=150
    """
    try:
        video_path = resolve_video_path(DEFAULT_VIDEO_DIR, video_id)
        image = extract_frame(video_path, frame_idx)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": engine is not None and engine._is_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task2_vqa.app:app", host="0.0.0.0", port=8000, reload=True, app_dir=_SRC_DIR)


