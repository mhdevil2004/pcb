from __future__ import annotations

import base64
import gc
import json
import os
import importlib
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
import sys
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).with_name("pcb_defect_detector.pt")
LOCAL_YOLOV5_REPO = Path(__file__).with_name("yolov5")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(8_000_000)))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1280"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
PREDICT_CONCURRENCY = int(os.getenv("PREDICT_CONCURRENCY", "1"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "120"))
ANNOTATION_WIDTH = int(os.getenv("ANNOTATION_WIDTH", "2"))
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
MODEL_STATE_LOCK = threading.Lock()
PREDICT_SEMAPHORE = threading.BoundedSemaphore(max(1, PREDICT_CONCURRENCY))
MODEL_STATUS = "idle"
MODEL_ERROR = ""


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
            if self.backend_name == "yolov5_local":
                return self._predict_yolov5_local(image, confidence, image_size)
        raise RuntimeError(f"Unsupported backend: {self.backend_name}")

    def _predict_ultralytics(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        results = self.model.predict(
            source=np.array(image),
            conf=confidence,
            imgsz=image_size,
            verbose=False,
            max_det=MAX_DETECTIONS,
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
            annotated_image=annotate_image(image, detections),
        )

    def _predict_yolov5(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        self.model.conf = confidence
        self.model.iou = 0.45
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = MAX_DETECTIONS

        results = self.model(np.array(image), size=image_size)
        raw = results.xyxy[0]
        if raw is None or len(raw) == 0:
            detections = empty_detections()
        else:
            raw_np = raw.cpu().numpy()
            detections = build_detection_frame(
                xyxy=raw_np[:, :4],
                confidences=raw_np[:, 4],
                class_ids=raw_np[:, 5].astype(int),
                class_labels=self.class_labels,
            )

        return PredictionOutput(
            detections=detections,
            annotated_image=annotate_image(image, detections),
        )

    def _predict_yolov5_local(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        import torch

        letterbox = importlib.import_module("utils.augmentations").letterbox
        general_utils = importlib.import_module("utils.general")
        non_max_suppression = general_utils.non_max_suppression
        scale_boxes = general_utils.scale_boxes

        image_np = np.array(image)
        stride = int(getattr(self.model, "stride", torch.tensor([32])).max())
        processed, _, _ = letterbox(image_np, new_shape=image_size, auto=False, stride=stride)
        processed = np.ascontiguousarray(processed.transpose((2, 0, 1)))

        tensor = torch.from_numpy(processed).to(next(self.model.parameters()).device)
        tensor = tensor.float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            raw_predictions = self.model(tensor)
            if isinstance(raw_predictions, (list, tuple)):
                raw_predictions = raw_predictions[0]
            predictions = non_max_suppression(
                raw_predictions,
                conf_thres=confidence,
                iou_thres=0.45,
                max_det=MAX_DETECTIONS,
            )

        detections_tensor = predictions[0]
        if detections_tensor is None or len(detections_tensor) == 0:
            detections = empty_detections()
        else:
            detections_tensor[:, :4] = scale_boxes(
                tensor.shape[2:],
                detections_tensor[:, :4],
                image_np.shape,
            ).round()
            detections = build_detection_frame(
                xyxy=detections_tensor[:, :4].cpu().numpy(),
                confidences=detections_tensor[:, 4].cpu().numpy(),
                class_ids=detections_tensor[:, 5].cpu().numpy().astype(int),
                class_labels=self.class_labels,
            )

        return PredictionOutput(
            detections=detections,
            annotated_image=annotate_image(image, detections),
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
    if os.name != "nt":
        yield
        return

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path


def load_with_ultralytics(model_path: Path) -> ModelBackend:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = getattr(model.model, "names", None)
    return ModelBackend("ultralytics", model, normalize_class_labels(names))


def load_with_local_yolov5(model_path: Path) -> ModelBackend:
    import torch

    repo_path = str(LOCAL_YOLOV5_REPO.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    attempt_load = importlib.import_module("models.experimental").attempt_load

    with windows_checkpoint_compatibility():
        model = attempt_load(str(model_path), device=torch.device("cpu"), inplace=True, fuse=True)

    names = getattr(model, "names", None)
    return ModelBackend("yolov5_local", model, normalize_class_labels(names))


def load_with_torch_hub(model_path: Path) -> ModelBackend:
    import torch

    previous_autoinstall = os.environ.get("YOLOv5_AUTOINSTALL")
    try:
        with windows_checkpoint_compatibility():
            os.environ["YOLOv5_AUTOINSTALL"] = "false"
            if LOCAL_YOLOV5_REPO.exists():
                model = torch.hub.load(
                    str(LOCAL_YOLOV5_REPO),
                    "custom",
                    path=str(model_path),
                    source="local",
                    force_reload=False,
                    _verbose=False,
                )
            else:
                model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(model_path),
                    source="github",
                    force_reload=False,
                    trust_repo=True,
                    _verbose=False,
                )
    except Exception as exc:
        raise RuntimeError(
            "This model needs a YOLOv5-compatible runtime. Place a local `yolov5` repo "
            "in this folder or allow `torch.hub` to download `ultralytics/yolov5`.\n\n"
            f"Original error: {exc}"
        ) from exc
    finally:
        if previous_autoinstall is None:
            os.environ.pop("YOLOv5_AUTOINSTALL", None)
        else:
            os.environ["YOLOv5_AUTOINSTALL"] = previous_autoinstall

    names = getattr(model, "names", None)
    return ModelBackend("yolov5", model, normalize_class_labels(names))


@lru_cache(maxsize=1)
def load_model() -> ModelBackend:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if LOCAL_YOLOV5_REPO.exists():
        return load_with_local_yolov5(MODEL_PATH)

    try:
        return load_with_ultralytics(MODEL_PATH)
    except Exception as exc:
        if not is_yolov5_incompatibility(exc):
            raise
        return load_with_torch_hub(MODEL_PATH)


def set_model_state(status: str, error: str = "") -> None:
    global MODEL_STATUS, MODEL_ERROR
    with MODEL_STATE_LOCK:
        MODEL_STATUS = status
        MODEL_ERROR = error


def get_model_state() -> tuple[str, str]:
    with MODEL_STATE_LOCK:
        return MODEL_STATUS, MODEL_ERROR


def warm_model() -> None:
    status, _ = get_model_state()
    if status in {"loading", "ready"}:
        return

    set_model_state("loading")

    def _load() -> None:
        try:
            load_model()
            set_model_state("ready")
        except Exception as exc:
            set_model_state("error", str(exc))

    threading.Thread(target=_load, daemon=True).start()


def image_to_data_url(image: Image.Image) -> str:
    if max(image.size) > MAX_IMAGE_EDGE:
        preview = image.copy()
        preview.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    else:
        preview = image

    buffer = BytesIO()
    # JPEG is much smaller than PNG for photo-like PCB images.
    preview.save(buffer, format="JPEG", quality=72, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    if preview is not image:
        preview.close()
    return f"data:image/jpeg;base64,{encoded}"


def annotate_image(image: Image.Image, detections: pd.DataFrame) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for row in detections.to_dict(orient="records"):
        box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        label = f'{row["class_name"]} {row["confidence"]:.2f}'
        draw.rectangle(box, outline="red", width=ANNOTATION_WIDTH)
        draw.text((row["x1"] + 4, max(0, row["y1"] - 14)), label, fill="red")
    return annotated


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

    image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    if image.width * image.height > MAX_INPUT_PIXELS or max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return image


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
        if self.path == "/":
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": "pcb-defect-backend"})
            return

        if self.path == "/api/health":
            model_status, model_error = get_model_state()
            if model_status == "idle":
                warm_model()
                model_status, model_error = get_model_state()

            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_path": str(MODEL_PATH),
                    "model_status": model_status,
                    "model_error": model_error,
                },
            )
            return

        if self.path == "/api/classes":
            self._send_json(
                HTTPStatus.OK,
                {
                    "backend": "unloaded",
                    "classes": list(DEFAULT_CLASS_LABELS.values()),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return

        if not PREDICT_SEMAPHORE.acquire(blocking=False):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Server is busy with another prediction. Please retry in a few seconds."},
            )
            return

        try:
            model_status, model_error = get_model_state()
            if model_status == "loading":
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "Model is warming up. Please retry in a minute."},
                )
                return
            if model_status == "error":
                raise RuntimeError(model_error or "Model failed to load.")

            backend = load_model()
            form = self._parse_form_data()
            image_item = form["image"] if "image" in form else None
            if image_item is None:
                raise ValueError("Missing `image` field in form data.")

            confidence = clamp_float(str(form.get("confidence", "0.25")), 0.05, 0.95, 0.25)
            image_size = clamp_int(str(form.get("image_size", "640")), 320, MAX_IMAGE_SIZE, 640)
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
        finally:
            gc.collect()
            PREDICT_SEMAPHORE.release()

    def _parse_form_data(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Missing Content-Length header.")

        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Uploaded file is too large ({length} bytes). Limit is {MAX_UPLOAD_BYTES} bytes."
            )

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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Origin, X-Requested-With, Authorization",
        )

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
