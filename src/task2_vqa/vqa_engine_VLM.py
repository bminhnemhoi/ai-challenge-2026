from __future__ import annotations

from typing import Any, Dict, List, Optional

if __package__:
    from .config import DEFAULT_VIDEO_DIR, VLM_TOP_K
    from .frame_utils import extract_frame_window, resolve_video_path
    from .postprocessor import build_submission_record, clean_answer
else:  # running this file directly from inside src/task2_vqa/
    # NOT try/except ImportError: that turned a missing cv2 into the misleading
    # "No module named 'config'", which hid the real cause for a long time
    from config import DEFAULT_VIDEO_DIR, VLM_TOP_K
    from frame_utils import extract_frame_window, resolve_video_path
    from postprocessor import build_submission_record, clean_answer


class VisualQAEngine:
    """
    Visual Q&A Pipeline Engine for Task 2.
    Integrates VLM (Qwen3-VL) and OCR/object candidate retrieval via Keyframe Retriever.
    """

    def __init__(
        self,
        data_dir: str,
        retriever: Any = None,
        visual_context_engine: Any = None,
        vlm_engine: Any = None,
        video_dir: Optional[str] = None,
        vlm_top_k: int = VLM_TOP_K,
    ):
        self.data_dir = data_dir
        if retriever is None:
            from src.task1_kis import TextualKISRetriever
            retriever = TextualKISRetriever(data_dir=data_dir)
        self.retriever = retriever

        if visual_context_engine is None:
            try:
                from .visual_context import VisualContextEngine
            except ImportError: 
                from visual_context import VisualContextEngine
            visual_context_engine = VisualContextEngine(data_dir=data_dir)
        self.visual_context_engine = visual_context_engine

        self.vlm_engine = vlm_engine
        self.video_dir = video_dir or DEFAULT_VIDEO_DIR
        self.vlm_top_k = vlm_top_k
        self._is_loaded = False

    def get_visual_context(
        self,
        image: Any,
        video_id: Optional[str] = None,
        frame_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return Member 3 OCR/object context for Member 2's VLM prompt."""
        return self.visual_context_engine.analyze(
            image=image,
            video_id=video_id,
            frame_idx=frame_idx,
            **kwargs,
        )

    def load_models(self):
        """Loads VLM, OCR, and underlying candidate keyframe index."""
        if self._is_loaded:
            return
        print("Initializing Task 2: Visual Q&A Engine (Member 2 & 3)...")
        try:
            self.retriever.load_index_and_model()
        except Exception as e:
            print(f"VQA Retriever load note: {e}")

        if self.vlm_engine is None:
            from .gemini_vlm_engine import GeminiVLMEngine, load_gemini_model
            model, processor = load_gemini_model()
            self.vlm_engine = GeminiVLMEngine(model, processor)

        self._is_loaded = True

    def answer_question(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Answers a visual question based on video frame context.

        Args:
            question: Visual question string (e.g., 'What color is the car?')
            video_id: Optional target video ID filter
            top_k: Number of retrieval candidates to consider (cheap, does not
                   run the VLM). The VLM itself only runs on the top
                   `self.vlm_top_k` of these -- see class docstring.

        Returns:
            List of dicts containing video_id, frame_idx, answer, score --
            the exact submission format for this task.
        """
        if not self._is_loaded:
            self.load_models()

        raw_results = self.retriever.search(query=question, top_k=top_k) if self.retriever._is_loaded else []

        candidates = []
        for r in raw_results:
            if video_id and r.get("video_id") != video_id:
                continue
            candidates.append(r)
            if len(candidates) >= top_k:
                break

        if not candidates:
            candidates = [{
                "video_id": video_id or "L21_V001",
                "frame_idx": 150,
                "frame_filename": "006.jpg",
                "score": 0.0,
            }]

        results: List[Dict[str, Any]] = []
        for cand in candidates[: self.vlm_top_k]:
            vid, fidx = cand["video_id"], cand["frame_idx"]
            try:
                video_path = resolve_video_path(self.video_dir, vid)
                frames = extract_frame_window(video_path, fidx)
            except (FileNotFoundError, IndexError) as e:
                print(f"[skipping candidate] {vid} frame {fidx}: {e}")
                continue

            context = self.get_visual_context(frames[len(frames) // 2], video_id=vid, frame_idx=fidx)
            if context.get("warnings"):
                print(f"visual_context warnings ({vid}, {fidx}): {context['warnings']}")

            raw_answer = self.vlm_engine.answer(
                frames, question, video_id=vid, frame_idx=fidx,
                visual_context=context.get("vlm_context"),
            )
            cleaned = clean_answer(raw_answer)
            results.append(
                build_submission_record(video_id=vid, frame_idx=fidx, answer=cleaned, score=cand.get("score"))
            )

        return results
