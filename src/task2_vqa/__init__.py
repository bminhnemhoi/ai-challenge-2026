"""
Task 2: Visual Question Answering (VQA) Module.
Assigned Members: Member 2 & Member 3
"""

from .vqa_engine import VisualQAEngine
from .ocr_engine import PaddleOCREngine
from .object_detector import ObjectDetector
from .visual_context import VisualContextEngine
from .postprocessor import build_submission_record, clean_answer

__all__ = [
    "VisualQAEngine",
    "PaddleOCREngine",
    "ObjectDetector",
    "VisualContextEngine",
    "clean_answer",
    "build_submission_record",
]
