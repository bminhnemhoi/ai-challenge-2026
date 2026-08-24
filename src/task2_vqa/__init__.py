"""
Task 2: Visual Question Answering (VQA) Module.
Assigned Members: Member 2 & Member 3

Only `clean_answer` / `build_submission_record` are on the live submission
path; the heavier engines below need optional extras (opencv, paddleocr,
ultralytics) and local .mp4 files. Importing this package must never fail just
because one of those is absent — a missing optional dependency used to abort
`import src` and take Task 1 and Task 3 down with it.

Anything that could not be imported is listed in `UNAVAILABLE` with its reason,
so a caller can report the real cause instead of a misleading one.
"""

import importlib

from .postprocessor import build_submission_record, clean_answer

__all__ = ["clean_answer", "build_submission_record", "UNAVAILABLE"]

#: name -> import error message, for components that need optional extras
UNAVAILABLE: dict = {}

for _name, _module, _attr in (
    ("VisualQAEngine", ".vqa_engine_VLM", "VisualQAEngine"),
    ("PaddleOCREngine", ".ocr_engine", "PaddleOCREngine"),
    ("ObjectDetector", ".object_detector", "ObjectDetector"),
    ("VisualContextEngine", ".visual_context", "VisualContextEngine"),
):
    try:
        # import_module resolves the leading dot against __name__; __import__
        # with level=1 does not, and silently produced "src.task2_vqa." —
        # so every engine looked like a missing optional dependency
        _mod = importlib.import_module(_module, __name__)
        globals()[_name] = getattr(_mod, _attr)
        __all__.append(_name)
    except Exception as _exc:  # noqa: BLE001 - report, never crash the package
        UNAVAILABLE[_name] = f"{type(_exc).__name__}: {_exc}"

del _name, _module, _attr
