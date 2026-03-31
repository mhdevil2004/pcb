from __future__ import annotations

import base64
import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as default_email_policy
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import pathlib
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

MODEL_PATH = Path(__file__).with_name("pcb_defect_detector.pt")
LOCAL_YOLOV5_REPO = Path(__file__).with_name("yolov5")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_LABELS = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole",
}
DETECTION_COLUMNS = [
    "detection_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]
MODEL_LOCK = threading.Lock()


@dataclass
class PredictionOutput:
    detections: pd.DataFrame
    annotated_image: Image.Image


class ModelBackend:
    def __init__(self, backend_name: str, model: Any, class_labels: dict[int, str]) -> None:
        self.backend_name = backend_name
        self.model = model
        self.class_labels = class_labels

    def predict(self, image: Image.Image, confidence: float, image_size: int) -> PredictionOutput:
        with MODEL_LOCK:
            if self.backend_name == "ultralytics":
                return self._predict_ultralytics(image, confidence, image_size)
            if self.backend_name == "yolov5":
                return self._predict_yolov5(image, confidence, image_size)
        raise RuntimeError(f"Unsupported backend: {self.backend_name}")

    def _predict_ultralytics(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        results = self.model.predict(
            source=np.array(image),
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            detections = empty_detections()
        else:
            detections = build_detection_frame(
                xyxy=boxes.xyxy.cpu().numpy(),
                confidences=boxes.conf.cpu().numpy(),
                class_ids=boxes.cls.cpu().numpy().astype(int),
                class_labels=self.class_labels,
            )

        return PredictionOutput(
            detections=detections,
            annotated_image=Image.fromarray(result.plot()),
        )

    def _predict_yolov5(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        self.model.conf = confidence
        self.model.iou = 0.45
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 1000

        results = self.model(np.array(image), size=image_size)
        frame = results.pandas().xyxy[0].copy()
        rendered_images = results.render()
        if frame.empty:
            detections = empty_detections()
        else:
            frame["class"] = frame["class"].astype(int)
            detections = pd.DataFrame(
                {
                    "detection_id": range(1, len(frame) + 1),
                    "class_id": frame["class"],
                    "class_name": [
                        self.class_labels.get(class_id, str(name))
                        for class_id, name in zip(frame["class"], frame["name"])
                    ],
                    "confidence": frame["confidence"].round(4),
                    "x1": frame["xmin"].round(2),
                    "y1": frame["ymin"].round(2),
                    "x2": frame["xmax"].round(2),
                    "y2": frame["ymax"].round(2),
                },
                columns=DETECTION_COLUMNS,
            )

        return PredictionOutput(
            detections=detections,
            annotated_image=Image.fromarray(rendered_images[0]),
        )


def empty_detections() -> pd.DataFrame:
    return pd.DataFrame(columns=DETECTION_COLUMNS)


def normalize_class_labels(names: Any) -> dict[int, str]:
    if isinstance(names, dict) and names:
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list) and names:
        return {idx: str(name) for idx, name in enumerate(names)}
    return DEFAULT_CLASS_LABELS.copy()


def build_detection_frame(
    xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_labels: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (box, score, class_id) in enumerate(zip(xyxy, confidences, class_ids), start=1):
        rows.append(
            {
                "detection_id": idx,
                "class_id": int(class_id),
                "class_name": class_labels.get(int(class_id), f"class_{class_id}"),
                "confidence": round(float(score), 4),
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
            }
        )
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def is_yolov5_incompatibility(error: Exception) -> bool:
    message = str(error).lower()
    return "yolov5 model" in message or "not forwards compatible" in message


@contextmanager
def windows_checkpoint_compatibility() -> Any:
    original_posix_path = pathlib.PosixPath

    class CompatiblePosixPath(pathlib.PurePosixPath):
        def __new__(cls, *args: Any, **kwargs: Any) -> pathlib.Path:
            return pathlib.WindowsPath(*args, **kwargs)

    pathlib.PosixPath = CompatiblePosixPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path


def load_with_ultralytics(model_path: Path) -> ModelBackend:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = getattr(model.model, "names", None)
    return ModelBackend("ultralytics", model, normalize_class_labels(names))


def load_with_torch_hub(model_path: Path) -> ModelBackend:
    import torch

    try:
        with windows_checkpoint_compatibility():
            if LOCAL_YOLOV5_REPO.exists():
                model = torch.hub.load(
                    str(LOCAL_YOLOV5_REPO),
                    "custom",
                    path=str(model_path),
                    source="local",
                    force_reload=False,
                )
            else:
                model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(model_path),
                    source="github",
                    force_reload=False,
                    trust_repo=True,
                )
    except Exception as exc:
        raise RuntimeError(
            "This model needs a YOLOv5-compatible runtime. Place a local `yolov5` repo "
            "in this folder or allow `torch.hub` to download `ultralytics/yolov5`.\n\n"
            f"Original error: {exc}"
        ) from exc

    names = getattr(model, "names", None)
    return ModelBackend("yolov5", model, normalize_class_labels(names))


@lru_cache(maxsize=1)
def load_model() -> ModelBackend:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        return load_with_ultralytics(MODEL_PATH)
    except Exception as exc:
        if not is_yolov5_incompatibility(exc):
            raise
        return load_with_torch_hub(MODEL_PATH)


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def summarize_detections(detections: pd.DataFrame) -> dict[str, Any]:
    counts = Counter(detections["class_name"]) if not detections.empty else Counter()
    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total_defects": int(len(detections)),
        "detected_classes": int(len(counts)),
        "most_common": sorted_counts[0][0] if sorted_counts else "No defects",
        "class_counts": [
            {"class_name": class_name, "count": count} for class_name, count in sorted_counts
        ],
    }


def parse_uploaded_image(file_item: dict[str, Any]) -> Image.Image:
    if not file_item["content"]:
        raise ValueError("No image file was uploaded.")

    filename = file_item["filename"] or "upload"
    extension = Path(filename).suffix.lower()
    if extension and extension not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image type. Use jpg, jpeg, png, bmp, or webp.")

    raw_bytes = file_item["content"]
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    return Image.open(BytesIO(raw_bytes)).convert("RGB")


def parse_multipart_form_data(headers: Any, body: bytes) -> dict[str, Any]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Content-Type must be multipart/form-data.")

    message = BytesParser(policy=default_email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )

    form: dict[str, Any] = {}
    for part in message.iter_parts():
        content_disposition = part.get("Content-Disposition", "")
        if "form-data" not in content_disposition:
            continue

        name = part.get_param("name", header="Content-Disposition")
        if not name:
            continue

        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            form[name] = {"filename": filename, "content": payload}
        else:
            charset = part.get_content_charset() or "utf-8"
            form[name] = payload.decode(charset).strip()

    return form


def build_prediction_response(
    backend: ModelBackend,
    image: Image.Image,
    confidence: float,
    image_size: int,
) -> dict[str, Any]:
    prediction = backend.predict(image=image, confidence=confidence, image_size=image_size)
    detections = prediction.detections
    summary = summarize_detections(detections)

    return {
        "backend": backend.backend_name,
        "classes": list(backend.class_labels.values()),
        "summary": summary,
        "detections": detections.to_dict(orient="records"),
        "annotated_image": image_to_data_url(prediction.annotated_image),
    }


class PCBRequestHandler(BaseHTTPRequestHandler):
    server_version = "PCBDefectServer/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_path": str(MODEL_PATH),
                },
            )
            return

        if self.path == "/api/classes":
            backend = load_model()
            self._send_json(
                HTTPStatus.OK,
                {
                    "backend": backend.backend_name,
                    "classes": list(backend.class_labels.values()),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return

        try:
            backend = load_model()
            form = self._parse_form_data()
            image_item = form["image"] if "image" in form else None
            if image_item is None:
                raise ValueError("Missing `image` field in form data.")

            confidence = clamp_float(str(form.get("confidence", "0.25")), 0.05, 0.95, 0.25)
            image_size = clamp_int(str(form.get("image_size", "640")), 320, 1280, 640)
            image = parse_uploaded_image(image_item)
            payload = build_prediction_response(
                backend=backend,
                image=image,
                confidence=confidence,
                image_size=image_size,
            )
            self._send_json(HTTPStatus.OK, payload)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _parse_form_data(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Missing Content-Length header.")

        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc

        body = self.rfile.read(length)
        return parse_multipart_form_data(self.headers, body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def clamp_float(value: str, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(parsed, maximum))


def clamp_int(value: str, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(parsed, maximum))


def run_server() -> None:
    host = os.getenv("PCB_BACKEND_HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", os.getenv("PCB_BACKEND_PORT", str(DEFAULT_PORT))))
    server = ThreadingHTTPServer((host, port), PCBRequestHandler)
    print(f"PCB defect backend listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
