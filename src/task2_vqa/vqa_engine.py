"""
Task 2: Visual Question Answering (VQA) Module.
Assigned Members: Member 2 (VLM) & Member 3 (OCR & Object Detection)

This module handles video context question answering for AIC 2026.
Format output: video_id, frame_idx, answer text.
"""

from typing import List, Dict, Any, Optional

class VisualQAEngine:
    """
    Visual Q&A Pipeline Engine for Task 2.
    Integrates VLM (LLaVA/Qwen2-VL) and OCR (PaddleOCR/Tesseract).
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._is_loaded = False

    def load_models(self):
        """Loads VLM, OCR, and object detection models."""
        if self._is_loaded:
            return
        print("Initializing Task 2: Visual Q&A Engine (Member 2 & 3)...")
        # TODO: Load VLM and OCR models here
        self._is_loaded = True

    def answer_question(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Answers a visual question based on video frame context.
        
        Args:
            question: Visual question string (e.g., 'What color is the car?')
            video_id: Optional target video ID filter
            top_k: Number of predictions to return
            
        Returns:
            List of dicts containing video_id, frame_idx, answer, confidence score.
        """
        if not self._is_loaded:
            self.load_models()

        # Baseline response format for Task 2
        results = [
            {
                "video_id": video_id or "L21_V001",
                "frame_idx": 150,
                "frame_filename": "006.jpg",
                "answer": "Red car parked near the building",
                "score": 0.85,
                "rel_path": f"{video_id or 'L21_V001'}/006.jpg"
            }
        ]
        return results
