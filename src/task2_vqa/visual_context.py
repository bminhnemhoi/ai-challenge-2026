"""Composable OCR and object context for the Task 2 VLM pipeline."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from .object_detector import ObjectDetector
    from .ocr_engine import PaddleOCREngine
except ImportError:  # Allows direct execution of the in-folder test module.
    from object_detector import ObjectDetector
    from ocr_engine import PaddleOCREngine


class VisualContextEngine:
    """Combine local OCR and object detections into VLM-ready context."""

    def __init__(
        self,
        data_dir: str | Path,
        ocr_engine: Any = None,
        object_detector: Any = None,
        max_ocr_texts: int = 20,
        max_object_labels: int = 30,
        request_timeout: float = 15.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ocr_engine = ocr_engine if ocr_engine is not None else PaddleOCREngine()
        self.object_detector = (
            object_detector if object_detector is not None else ObjectDetector(data_dir=data_dir)
        )
        self.max_ocr_texts = int(max_ocr_texts)
        self.max_object_labels = int(max_object_labels)
        self.request_timeout = float(request_timeout)

    def analyze(
        self,
        image: Any,
        video_id: Optional[str] = None,
        frame_idx: Optional[int] = None,
        frame_name: Optional[str] = None,
        objects_json_path: Optional[str | Path] = None,
        use_ocr: bool = True,
        use_objects: bool = True,
        prefer_metadata: bool = True,
    ) -> dict[str, Any]:
        """Analyze one candidate keyframe and return a stable integration payload."""
        prepared_image, inferred_name = self._prepare_image(image)
        frame_name = frame_name or inferred_name
        warnings: list[str] = []
        ocr: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        objects_source: Optional[str] = None

        if use_ocr:
            try:
                ocr = self.ocr_engine.extract(prepared_image)
            except Exception as exc:  # One enrichment must not remove the other.
                warnings.append(f"OCR: {exc}")

        if use_objects:
            try:
                objects, objects_source = self.object_detector.detect(
                    prepared_image,
                    video_id=video_id,
                    frame_name=frame_name,
                    objects_json_path=objects_json_path,
                    prefer_metadata=prefer_metadata,
                )
            except Exception as exc:
                warnings.append(f"Object detection: {exc}")

        ocr_texts = self._unique_texts(ocr)
        object_counts = self._count_objects(objects)
        return {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "ocr": ocr,
            "ocr_texts": ocr_texts,
            "objects": objects,
            "object_counts": object_counts,
            "objects_source": objects_source,
            "vlm_context": self._build_vlm_context(ocr_texts, object_counts),
            "warnings": warnings,
        }

    def _prepare_image(self, image: Any) -> tuple[Any, Optional[str]]:
        if isinstance(image, Path):
            if not image.is_file():
                raise FileNotFoundError(f"Keyframe does not exist: {image}")
            return str(image), image.name
        if isinstance(image, str):
            parsed = urlparse(image)
            if parsed.scheme in ("http", "https"):
                request = Request(image, headers={"User-Agent": "AIC2026-Task2/1.0"})
                with urlopen(request, timeout=self.request_timeout) as response:
                    content = response.read()
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise RuntimeError("Pillow is required to analyze image URLs.") from exc
                loaded = Image.open(io.BytesIO(content)).convert("RGB")
                return loaded, Path(parsed.path).name or None
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Keyframe does not exist: {path}")
            return str(path), path.name
        if image is None:
            raise ValueError("image is required")
        return image, None

    def _unique_texts(self, detections: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        texts: list[str] = []
        for detection in detections:
            text = str(detection.get("text", "")).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            texts.append(text)
            if len(texts) >= self.max_ocr_texts:
                break
        return texts

    def _count_objects(self, objects: list[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(
            str(item.get("label", "unknown")).strip().lower()
            for item in objects
            if str(item.get("label", "")).strip()
        )
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        return dict(ordered[: self.max_object_labels])

    @staticmethod
    def _build_vlm_context(ocr_texts: list[str], object_counts: dict[str, int]) -> str:
        lines = []
        if ocr_texts:
            lines.append(f"Visible text: {' | '.join(ocr_texts)}")
        if object_counts:
            summary = ", ".join(f"{label}: {count}" for label, count in object_counts.items())
            lines.append(f"Detected objects: {summary}")
        return "\n".join(lines) if lines else "No OCR or object context available."
