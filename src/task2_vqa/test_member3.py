"""Lightweight tests for Member 3's OCR and object-context modules."""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

TASK2_DIR = Path(__file__).resolve().parent
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from ocr_engine import PaddleOCREngine
from object_detector import ObjectDetector
from visual_context import VisualContextEngine
from postprocessor import build_submission_record, clean_answer
from vqa_engine import VisualQAEngine


class _LegacyOCRBackend:
    def ocr(self, image, cls=True):
        return [[
            [[[0, 0], [10, 0], [10, 5], [0, 5]], ("Xin chào", 0.96)],
            [[[0, 6], [12, 6], [12, 10], [0, 10]], ("mờ", 0.20)],
        ]]


class _ModernOCRBackend:
    def predict(self, image):
        return [{
            "rec_texts": ["AIC 2026", ""],
            "rec_scores": [0.91, 0.99],
            "rec_polys": [
                [[1, 1], [8, 1], [8, 4], [1, 4]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
            ],
        }]


class _ArrayLike:
    """Reproduces NumPy's ambiguous truth value at the OCR boundary."""

    def __init__(self, values):
        self.values = values

    def __bool__(self):
        raise ValueError("truth value of an array is ambiguous")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        return self.values


class _ArrayOCRBackend:
    def predict(self, image):
        return [{
            "rec_texts": _ArrayLike(["Sài Gòn"]),
            "rec_scores": _ArrayLike([0.89]),
            "rec_polys": _ArrayLike([[[0, 0], [2, 0], [2, 1], [0, 1]]]),
        }]


class TestPaddleOCREngine(unittest.TestCase):
    def test_normalizes_legacy_output_and_filters_low_confidence(self):
        engine = PaddleOCREngine(backend=_LegacyOCRBackend(), min_confidence=0.5)

        detections = engine.extract("frame.jpg")

        self.assertEqual(detections, [{
            "text": "Xin chào",
            "confidence": 0.96,
            "bbox": [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        }])

    def test_normalizes_modern_dictionary_output(self):
        engine = PaddleOCREngine(backend=_ModernOCRBackend())

        detections = engine.extract("frame.jpg")

        self.assertEqual(detections[0]["text"], "AIC 2026")
        self.assertAlmostEqual(detections[0]["confidence"], 0.91)
        self.assertEqual(len(detections), 1)

    def test_backend_factory_is_lazy_and_called_once(self):
        calls = []

        def factory():
            calls.append("created")
            return _LegacyOCRBackend()

        engine = PaddleOCREngine(backend_factory=factory)
        self.assertEqual(calls, [])

        engine.extract("one.jpg")
        engine.extract("two.jpg")

        self.assertEqual(calls, ["created"])

    def test_normalizes_array_shaped_modern_output_without_boolean_coercion(self):
        engine = PaddleOCREngine(backend=_ArrayOCRBackend())

        detections = engine.extract("frame.jpg")

        self.assertEqual(detections[0]["text"], "Sài Gòn")

    def test_real_backend_factory_selects_mobile_latin_models(self):
        captured = {}
        fake_module = ModuleType("paddleocr")

        def fake_paddle_ocr(**kwargs):
            captured.update(kwargs)
            return object()

        fake_module.PaddleOCR = fake_paddle_ocr
        engine = PaddleOCREngine(lang="vi")

        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            engine._create_backend()

        self.assertEqual(captured["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(captured["text_recognition_model_name"], "latin_PP-OCRv5_mobile_rec")


class _FakeBoxes:
    xyxy = [[10, 20, 30, 40], [1, 2, 3, 4]]
    conf = [0.88, 0.22]
    cls = [0, 1]


class _FakeYOLOResult:
    boxes = _FakeBoxes()
    names = {0: "person", 1: "car"}
    orig_shape = (100, 100)


class _FakeYOLOBackend:
    names = {0: "person", 1: "car"}

    def predict(self, source, **kwargs):
        return [_FakeYOLOResult()]


class TestObjectDetector(unittest.TestCase):
    def test_reads_openimages_parallel_arrays_and_filters_confidence(self):
        payload = {
            "detection_class_entities": ["Person", "Car"],
            "detection_scores": [0.93, 0.10],
            "detection_boxes": [[0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            detector = ObjectDetector(data_dir=directory, min_confidence=0.25)

            objects, source = detector.detect("frame.jpg", objects_json_path=path)

        self.assertEqual(source, "btc_metadata")
        self.assertEqual(objects, [{
            "label": "person",
            "confidence": 0.93,
            "bbox": [0.2, 0.1, 0.9, 0.8],
            "bbox_format": "xyxy_normalized",
        }])

    def test_resolves_btc_json_from_video_and_frame_name_before_yolo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects" / "L01_V001" / "0001.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([{
                "class_name": "Traffic light",
                "score": 0.81,
                "bbox": [1, 2, 3, 4],
            }]), encoding="utf-8")
            detector = ObjectDetector(
                data_dir=directory,
                yolo_backend_factory=lambda: self.fail("YOLO must not load"),
            )

            objects, source = detector.detect(
                "frame.jpg", video_id="L01_V001", frame_name="0001.jpg"
            )

        self.assertEqual(source, "btc_metadata")
        self.assertEqual(objects[0]["label"], "traffic light")
        self.assertEqual(objects[0]["bbox"], [])
        self.assertEqual(objects[0]["bbox_format"], "xyxy_normalized")

    def test_falls_back_to_lazy_yolo_and_normalizes_results(self):
        calls = []

        def factory():
            calls.append("loaded")
            return _FakeYOLOBackend()

        detector = ObjectDetector(
            data_dir="missing",
            min_confidence=0.25,
            yolo_backend_factory=factory,
        )
        self.assertEqual(calls, [])

        objects, source = detector.detect("frame.jpg")

        self.assertEqual(calls, ["loaded"])
        self.assertEqual(source, "yolo")
        self.assertEqual(objects, [{
            "label": "person",
            "confidence": 0.88,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "bbox_format": "xyxy_normalized",
        }])

    def test_does_not_mislabel_pixel_dictionary_box_as_normalized(self):
        payload = {"objects": [{
            "label": "person",
            "confidence": 0.8,
            "bbox": {"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40},
        }]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            detector = ObjectDetector(data_dir=directory)

            objects, source = detector.detect("frame.jpg", objects_json_path=path)

        self.assertEqual(source, "btc_metadata")
        self.assertEqual(objects[0]["bbox"], [])
        self.assertEqual(objects[0]["bbox_format"], "xyxy_normalized")


class _FakeOCREngine:
    def extract(self, image):
        return [
            {"text": "AIC 2026", "confidence": 0.9, "bbox": []},
            {"text": "AIC 2026", "confidence": 0.8, "bbox": []},
        ]


class _FakeObjectDetector:
    def __init__(self):
        self.received_frame_name = None

    def detect(self, image, **kwargs):
        self.received_frame_name = kwargs["frame_name"]
        return ([
            {"label": "person", "confidence": 0.9, "bbox": []},
            {"label": "person", "confidence": 0.8, "bbox": []},
            {"label": "trophy", "confidence": 0.7, "bbox": []},
        ], "btc_metadata")


class _FailingOCREngine:
    def extract(self, image):
        raise RuntimeError("OCR unavailable")


class TestVisualContextEngine(unittest.TestCase):
    def test_builds_deterministic_context_and_derives_frame_name(self):
        detector = _FakeObjectDetector()
        engine = VisualContextEngine(
            data_dir="data",
            ocr_engine=_FakeOCREngine(),
            object_detector=detector,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "0088.jpg"
            image_path.write_bytes(b"not-decoded-by-fakes")

            context = engine.analyze(
                image=image_path,
                video_id="L05_V005",
                frame_idx=888,
            )

        self.assertEqual(detector.received_frame_name, "0088.jpg")
        self.assertEqual(context["ocr_texts"], ["AIC 2026"])
        self.assertEqual(context["object_counts"], {"person": 2, "trophy": 1})
        self.assertEqual(context["objects_source"], "btc_metadata")
        self.assertEqual(
            context["vlm_context"],
            "Visible text: AIC 2026\nDetected objects: person: 2, trophy: 1",
        )
        self.assertEqual(context["warnings"], [])

    def test_preserves_object_context_when_ocr_fails(self):
        engine = VisualContextEngine(
            data_dir="data",
            ocr_engine=_FailingOCREngine(),
            object_detector=_FakeObjectDetector(),
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            context = engine.analyze(image_file.name, use_ocr=True, use_objects=True)

        self.assertEqual(context["ocr_texts"], [])
        self.assertEqual(context["object_counts"]["person"], 2)
        self.assertIn("OCR unavailable", context["warnings"][0])

    def test_can_disable_both_expensive_engines(self):
        engine = VisualContextEngine(
            data_dir="data",
            ocr_engine=_FailingOCREngine(),
            object_detector=None,
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            context = engine.analyze(image_file.name, use_ocr=False, use_objects=False)

        self.assertEqual(context["ocr"], [])
        self.assertEqual(context["objects"], [])
        self.assertEqual(context["vlm_context"], "No OCR or object context available.")
        self.assertEqual(context["warnings"], [])


class _FakeRetriever:
    _is_loaded = True

    def load_index_and_model(self):
        return None

    def search(self, query, top_k):
        return []


class _FakeVisualContext:
    def analyze(self, image, **kwargs):
        return {"image": image, **kwargs, "vlm_context": "Visible text: AIC 2026"}


class TestPostprocessorAndIntegration(unittest.TestCase):
    def test_cleans_common_vlm_answer_wrappers(self):
        self.assertEqual(clean_answer(' Answer:  "Màu xanh"\n'), "Màu xanh")
        self.assertEqual(clean_answer("Đáp án: Năm."), "Năm")
        self.assertEqual(clean_answer(0), "0")

    def test_rejects_empty_answer_and_builds_submission_record(self):
        with self.assertRaises(ValueError):
            clean_answer("   ")

        record = build_submission_record("L05_V005", 888, " Answer: 5 ", score=0.91)

        self.assertEqual(record, {
            "video_id": "L05_V005",
            "frame_idx": 888,
            "answer": "5",
            "score": 0.91,
        })

    def test_visual_qa_engine_exposes_injected_member3_context(self):
        engine = VisualQAEngine(
            data_dir="data",
            retriever=_FakeRetriever(),
            visual_context_engine=_FakeVisualContext(),
        )

        context = engine.get_visual_context(
            "frame.jpg", video_id="L01_V001", frame_idx=42, use_ocr=False
        )

        self.assertEqual(context["video_id"], "L01_V001")
        self.assertEqual(context["frame_idx"], 42)
        self.assertFalse(context["use_ocr"])

if __name__ == "__main__":
    unittest.main()
