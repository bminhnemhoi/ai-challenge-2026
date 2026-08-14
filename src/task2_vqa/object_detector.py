"""BTC Objects metadata reader with a lazy YOLO nano fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


class ObjectDetector:
    """Detect objects cheaply from BTC JSON, falling back to local YOLO."""

    def __init__(
        self,
        data_dir: str | Path,
        model_name: str = "yolo11n.pt",
        min_confidence: float = 0.25,
        image_size: int = 640,
        device: Optional[str] = None,
        yolo_backend: Any = None,
        yolo_backend_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.min_confidence = float(min_confidence)
        self.image_size = int(image_size)
        self.device = device
        self._yolo_backend = yolo_backend
        self._yolo_backend_factory = yolo_backend_factory

    def detect(
        self,
        image: Any,
        video_id: Optional[str] = None,
        frame_name: Optional[str] = None,
        objects_json_path: Optional[str | Path] = None,
        prefer_metadata: bool = True,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return normalized detections and the source used."""
        metadata_path = self._resolve_metadata_path(
            video_id=video_id,
            frame_name=frame_name,
            explicit_path=objects_json_path,
        )
        if prefer_metadata and metadata_path is not None and metadata_path.is_file():
            return self._read_metadata(metadata_path), "btc_metadata"
        return self._detect_yolo(image), "yolo"

    def _resolve_metadata_path(
        self,
        video_id: Optional[str],
        frame_name: Optional[str],
        explicit_path: Optional[str | Path],
    ) -> Optional[Path]:
        if explicit_path is not None:
            return Path(explicit_path)
        if not video_id or not frame_name:
            return None
        json_name = f"{Path(frame_name).stem}.json"
        candidates = (
            self.data_dir / "objects" / video_id / json_name,
            self.data_dir / video_id / json_name,
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    def _read_metadata(self, path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read BTC objects JSON: {path}") from exc
        records = self._metadata_records(payload)
        return [record for record in records if record["confidence"] >= self.min_confidence]

    def _metadata_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            for nested_key in ("objects", "detections", "predictions"):
                if isinstance(payload.get(nested_key), list):
                    return self._records_from_list(payload[nested_key])
            labels = (
                payload.get("detection_class_entities")
                or payload.get("detection_class_names")
                or payload.get("labels")
                or []
            )
            scores = payload.get("detection_scores") or payload.get("scores") or []
            boxes = payload.get("detection_boxes") or payload.get("boxes") or []
            return [
                self._record(
                    labels[index],
                    scores[index] if index < len(scores) else 1.0,
                    boxes[index] if index < len(boxes) else [],
                )
                for index in range(len(labels))
            ]
        if isinstance(payload, list):
            return self._records_from_list(payload)
        return []

    def _records_from_list(self, items: list[Any]) -> list[dict[str, Any]]:
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = next(
                (item[key] for key in ("class_name", "label", "name", "class") if key in item),
                "unknown",
            )
            score = next(
                (item[key] for key in ("score", "confidence", "probability") if key in item),
                1.0,
            )
            box = next(
                (item[key] for key in ("bbox", "box", "bounding_box") if key in item),
                [],
            )
            input_format = str(item.get("bbox_format", "")).lower()
            if isinstance(box, dict):
                safe_box = box
            else:
                values = list(box or [])[:4]
                is_normalized = len(values) == 4 and all(
                    isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
                    for value in values
                )
                if input_format in {"xyxy_normalized", "xyxy_norm"} and is_normalized:
                    safe_box = values
                    input_format = "xyxy_normalized"
                elif input_format in {"yxyx_normalized", "yxyx_norm"} and is_normalized:
                    safe_box = values
                    input_format = "yxyx_normalized"
                elif not input_format and is_normalized:
                    safe_box = values
                    input_format = "yxyx_normalized"
                else:
                    # Pixel coordinates cannot be normalized without image dimensions.
                    safe_box = []
                    input_format = "xyxy_normalized"
            records.append(self._record(label, score, safe_box, input_format=input_format))
        return records

    def _get_yolo_backend(self) -> Any:
        if self._yolo_backend is None:
            if self._yolo_backend_factory is not None:
                self._yolo_backend = self._yolo_backend_factory()
            else:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    raise RuntimeError(
                        "Ultralytics is not installed and BTC objects metadata was not found. "
                        "Install ultralytics to enable YOLO fallback."
                    ) from exc
                self._yolo_backend = YOLO(self.model_name)
        return self._yolo_backend

    def _detect_yolo(self, image: Any) -> list[dict[str, Any]]:
        backend = self._get_yolo_backend()
        kwargs: dict[str, Any] = {
            "conf": self.min_confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = backend.predict(source=image, **kwargs)
        detections: list[dict[str, Any]] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = self._to_list(getattr(boxes, "xyxy", []))
            scores = self._to_list(getattr(boxes, "conf", []))
            classes = self._to_list(getattr(boxes, "cls", []))
            names = getattr(result, "names", None) or getattr(backend, "names", {})
            original_shape = getattr(result, "orig_shape", None)
            if not original_shape or len(original_shape) < 2:
                raise ValueError("YOLO result must include orig_shape for normalized bounding boxes.")
            image_height, image_width = float(original_shape[0]), float(original_shape[1])
            for index, class_id in enumerate(classes):
                score = scores[index] if index < len(scores) else 0.0
                if float(score) < self.min_confidence:
                    continue
                class_number = int(class_id)
                label = names.get(class_number, str(class_number)) if isinstance(names, dict) else names[class_number]
                box = coordinates[index] if index < len(coordinates) else []
                normalized_box = []
                if len(box) >= 4 and image_width > 0 and image_height > 0:
                    normalized_box = [
                        float(box[0]) / image_width,
                        float(box[1]) / image_height,
                        float(box[2]) / image_width,
                        float(box[3]) / image_height,
                    ]
                detections.append(
                    self._record(label, score, normalized_box, input_format="xyxy_normalized")
                )
        return detections

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value or [])

    @staticmethod
    def _record(
        label: Any,
        score: Any,
        box: Any,
        input_format: str = "yxyx_normalized",
    ) -> dict[str, Any]:
        if isinstance(box, dict):
            values = [box.get(key) for key in ("xmin", "ymin", "xmax", "ymax")]
            box = values if all(
                isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
                for value in values
            ) else []
            input_format = "xyxy_normalized"
        normalized_box = [float(value) for value in list(box or [])[:4]]
        if len(normalized_box) == 4 and input_format == "yxyx_normalized":
            ymin, xmin, ymax, xmax = normalized_box
            normalized_box = [xmin, ymin, xmax, ymax]
        return {
            "label": str(label or "unknown").strip().lower(),
            "confidence": float(score or 0.0),
            "bbox": normalized_box,
            "bbox_format": "xyxy_normalized",
        }
