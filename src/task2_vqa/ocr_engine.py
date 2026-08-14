"""Lazy PaddleOCR adapter for Task 2 visual context extraction."""

from __future__ import annotations

from typing import Any, Callable, Optional


class PaddleOCREngine:
    """Extract normalized text regions without loading PaddleOCR at import time."""

    def __init__(
        self,
        lang: str = "vi",
        min_confidence: float = 0.35,
        use_angle_cls: bool = True,
        detection_model_name: str = "PP-OCRv5_mobile_det",
        recognition_model_name: Optional[str] = None,
        backend: Any = None,
        backend_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.lang = lang
        self.min_confidence = float(min_confidence)
        self.use_angle_cls = use_angle_cls
        self.detection_model_name = detection_model_name
        self.recognition_model_name = recognition_model_name or (
            "latin_PP-OCRv5_mobile_rec" if lang in {"vi", "en", "fr", "de", "es", "pt"}
            else "PP-OCRv5_mobile_rec"
        )
        self._backend = backend
        self._backend_factory = backend_factory

    def _create_backend(self) -> Any:
        if self._backend_factory is not None:
            return self._backend_factory()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddleocr and paddlepaddle "
                "to enable OCR."
            ) from exc

        try:
            return PaddleOCR(
                lang=self.lang,
                text_detection_model_name=self.detection_model_name,
                text_recognition_model_name=self.recognition_model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self.use_angle_cls,
            )
        except (TypeError, ValueError):
            return PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls, show_log=False)

    def _get_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def extract(self, image: Any) -> list[dict[str, Any]]:
        """Return ``text``, ``confidence`` and quadrilateral ``bbox`` records."""
        backend = self._get_backend()
        if hasattr(backend, "predict"):
            raw = backend.predict(image)
        elif hasattr(backend, "ocr"):
            try:
                raw = backend.ocr(image, cls=self.use_angle_cls)
            except TypeError:
                raw = backend.ocr(image)
        else:
            raise TypeError("OCR backend must provide predict() or ocr().")

        detections = self._normalize(raw)
        return [
            item for item in detections
            if item["text"] and item["confidence"] >= self.min_confidence
        ]

    def _normalize(self, raw: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for result in raw or []:
            payload = self._as_mapping(result)
            if payload is not None and "rec_texts" in payload:
                texts = self._to_list(payload.get("rec_texts"))
                scores = self._to_list(payload.get("rec_scores"))
                boxes_value = payload.get("rec_polys")
                if boxes_value is None:
                    boxes_value = payload.get("dt_polys")
                boxes = self._to_list(boxes_value)
                for index, text in enumerate(texts):
                    score = scores[index] if index < len(scores) else 0.0
                    box = boxes[index] if index < len(boxes) else []
                    detections.append(self._record(text, score, box))
                continue

            for line in result or []:
                if not self._is_legacy_line(line):
                    continue
                box, text_score = line
                text, score = text_score
                detections.append(self._record(text, score, box))
        return detections

    @staticmethod
    def _as_mapping(result: Any) -> Optional[dict[str, Any]]:
        if isinstance(result, dict):
            if "res" in result and isinstance(result["res"], dict):
                return result["res"]
            return result
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict):
            return payload.get("res", payload)
        return None

    @staticmethod
    def _is_legacy_line(line: Any) -> bool:
        return (
            isinstance(line, (list, tuple))
            and len(line) >= 2
            and isinstance(line[1], (list, tuple))
            and len(line[1]) >= 2
        )

    @staticmethod
    def _record(text: Any, score: Any, box: Any) -> dict[str, Any]:
        box = PaddleOCREngine._to_list(box)
        normalized_box = [
            [float(point[0]), float(point[1])]
            for point in box
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        return {
            "text": str(text or "").strip(),
            "confidence": float(score or 0.0),
            "bbox": normalized_box,
        }

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)
