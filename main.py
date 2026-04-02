from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import importlib
import socket
import subprocess
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
import tempfile
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw
import psutil

MODEL_PATH = Path(__file__).with_name("pcb_defect_detector.pt")
LOCAL_YOLOV5_REPO = Path(__file__).with_name("yolov5")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(4_000_000)))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1024"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "640"))
PREDICT_CONCURRENCY = int(os.getenv("PREDICT_CONCURRENCY", "1"))
PREDICT_ACQUIRE_TIMEOUT_SEC = float(os.getenv("PREDICT_ACQUIRE_TIMEOUT_SEC", "20"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "60"))
ANNOTATION_WIDTH = int(os.getenv("ANNOTATION_WIDTH", "2"))
MAX_RESPONSE_IMAGE_EDGE = int(os.getenv("MAX_RESPONSE_IMAGE_EDGE", "800"))
RESPONSE_JPEG_QUALITY = int(os.getenv("RESPONSE_JPEG_QUALITY", "60"))
SERVER_REQUEST_TIMEOUT_SEC = float(os.getenv("SERVER_REQUEST_TIMEOUT_SEC", "30"))
MODEL_WARM_ON_STARTUP = os.getenv("MODEL_WARM_ON_STARTUP", "false").strip().lower() not in {
    "0",
    "false",
    "no",
}
MODEL_KEEPALIVE_SEC = float(os.getenv("MODEL_KEEPALIVE_SEC", "0"))
PREDICT_IN_SUBPROCESS = os.getenv("PREDICT_IN_SUBPROCESS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
PREDICT_SUBPROCESS_TIMEOUT_SEC = float(os.getenv("PREDICT_SUBPROCESS_TIMEOUT_SEC", "240"))
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
MODEL_WARM_THREAD: threading.Thread | None = None
MODEL_UNLOAD_TIMER: threading.Timer | None = None


@dataclass
class PredictionOutput:
    detections: list[dict[str, Any]]
    annotated_image: Image.Image


def get_memory_stats() -> dict[str, int]:
    memory_info = psutil.Process(os.getpid()).memory_info()
    return {
        "rss_mb": int(memory_info.rss / (1024 * 1024)),
        "vms_mb": int(memory_info.vms / (1024 * 1024)),
    }


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


def empty_detections() -> list[dict[str, Any]]:
    return []


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (box, score, class_id) in enumerate(zip(xyxy, confidences, class_ids), start=1):
        rows.append({
            "detection_id": idx,
            "class_id": int(class_id),
            "class_name": class_labels.get(int(class_id), f"class_{class_id}"),
            "confidence": round(float(score), 4),
            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),
        })
    return rows


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


def unload_model() -> None:
    global MODEL_UNLOAD_TIMER, MODEL_STATUS, MODEL_ERROR

    with MODEL_STATE_LOCK:
        if MODEL_UNLOAD_TIMER is not None:
            MODEL_UNLOAD_TIMER.cancel()
            MODEL_UNLOAD_TIMER = None
        load_model.cache_clear()
        MODEL_STATUS = "idle"
        MODEL_ERROR = ""

    gc.collect()


def schedule_model_unload() -> None:
    global MODEL_UNLOAD_TIMER

    if MODEL_KEEPALIVE_SEC < 0:
        return

    with MODEL_STATE_LOCK:
        if MODEL_UNLOAD_TIMER is not None:
            MODEL_UNLOAD_TIMER.cancel()
            MODEL_UNLOAD_TIMER = None

        timer = threading.Timer(MODEL_KEEPALIVE_SEC, unload_model)
        timer.daemon = True
        MODEL_UNLOAD_TIMER = timer
        timer.start()


def set_model_state(status: str, error: str = "") -> None:
    global MODEL_STATUS, MODEL_ERROR
    with MODEL_STATE_LOCK:
        MODEL_STATUS = status
        MODEL_ERROR = error


def get_model_state() -> tuple[str, str]:
    with MODEL_STATE_LOCK:
        return MODEL_STATUS, MODEL_ERROR


def warm_model() -> None:
    global MODEL_WARM_THREAD

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
            gc.collect()

    MODEL_WARM_THREAD = threading.Thread(target=_load, name="model-warmup", daemon=True)
    MODEL_WARM_THREAD.start()


def image_to_data_url(image: Image.Image) -> str:
    if max(image.size) > MAX_RESPONSE_IMAGE_EDGE:
        preview = image.copy()
        preview.thumbnail((MAX_RESPONSE_IMAGE_EDGE, MAX_RESPONSE_IMAGE_EDGE), Image.Resampling.LANCZOS)
    else:
        preview = image

    buffer = BytesIO()
    # JPEG is much smaller than PNG for photo-like PCB images.
    preview.save(buffer, format="JPEG", quality=RESPONSE_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    buffer.close()
    if preview is not image:
        preview.close()
    return f"data:image/jpeg;base64,{encoded}"


def annotate_image(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for row in detections:
        box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        label = f'{row["class_name"]} {row["confidence"]:.2f}'
        draw.rectangle(box, outline="red", width=ANNOTATION_WIDTH)
        draw.text((row["x1"] + 4, max(0, row["y1"] - 14)), label, fill="red")
    return annotated


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["class_name"] for item in detections) if detections else Counter()
    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total_defects": len(detections),
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

    source_image = Image.open(BytesIO(raw_bytes))
    image = source_image.convert("RGB")
    source_image.close()
    if image.width * image.height > MAX_INPUT_PIXELS or max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return image


def validate_uploaded_file(file_item: dict[str, Any]) -> tuple[str, bytes]:
    if not file_item["content"]:
        raise ValueError("No image file was uploaded.")

    filename = file_item["filename"] or "upload"
    extension = Path(filename).suffix.lower()
    if extension and extension not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image type. Use jpg, jpeg, png, bmp, or webp.")

    raw_bytes = file_item["content"]
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Uploaded file is too large ({len(raw_bytes)} bytes). Limit is {MAX_UPLOAD_BYTES} bytes."
        )
    return filename, raw_bytes


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
    try:
        detections = prediction.detections
        summary = summarize_detections(detections)

        return {
            "backend": backend.backend_name,
            "classes": list(backend.class_labels.values()),
            "summary": summary,
            "detections": detections,
            "annotated_image": image_to_data_url(prediction.annotated_image),
        }
    finally:
        prediction.annotated_image.close()


def predict_once_from_upload(file_item: dict[str, Any], confidence: float, image_size: int) -> dict[str, Any]:
    image = parse_uploaded_image(file_item)
    try:
        backend = load_model()
        return build_prediction_response(
            backend=backend,
            image=image,
            confidence=confidence,
            image_size=image_size,
        )
    finally:
        image.close()


def predict_via_subprocess(
    file_item: dict[str, Any],
    confidence: float,
    image_size: int,
) -> dict[str, Any]:
    filename, raw_bytes = validate_uploaded_file(file_item)
    suffix = Path(filename).suffix or ".img"

    with tempfile.TemporaryDirectory(prefix="pcb-predict-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / f"input{suffix}"
        output_path = temp_dir_path / "result.json"
        input_path.write_bytes(raw_bytes)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--predict-once",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--confidence",
            str(confidence),
            "--image-size",
            str(image_size),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PREDICT_SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if not message:
                message = f"Prediction worker exited with code {result.returncode}."
            raise RuntimeError(message)
        if not output_path.exists():
            raise RuntimeError("Prediction worker did not produce an output payload.")

        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Prediction worker returned invalid JSON.") from exc


def run_prediction_worker(
    input_path: str,
    output_path: str,
    confidence: float,
    image_size: int,
) -> int:
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction input file not found: {source_path}")

    payload = predict_once_from_upload(
        {
            "filename": source_path.name,
            "content": source_path.read_bytes(),
        },
        confidence=confidence,
        image_size=image_size,
    )
    Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
    gc.collect()
    return 0


class PCBRequestHandler(BaseHTTPRequestHandler):
    server_version = "PCBDefectServer/1.0"
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(SERVER_REQUEST_TIMEOUT_SEC)

    @property
    def request_path(self) -> str:
        return urlparse(self.path).path

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._send_status_only(HTTPStatus.OK)

    def do_GET(self) -> None:
        if self.request_path == "/":
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": "pcb-defect-backend"})
            return

        if self.request_path == "/api/health":
            model_status, model_error = get_model_state()
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "pcb-defect-backend",
                    "model_path": str(MODEL_PATH),
                    "model_status": model_status,
                    "model_error": model_error,
                    "ready": model_status == "ready",
                    "prediction_mode": "subprocess" if PREDICT_IN_SUBPROCESS else "in_process",
                    "memory": get_memory_stats(),
                },
            )
            return

        if self.request_path == "/api/classes":
            self._send_json(
                HTTPStatus.OK,
                {
                    "backend": "unloaded",
                    "model_status": get_model_state()[0],
                    "classes": list(DEFAULT_CLASS_LABELS.values()),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if self.request_path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return

        if not PREDICT_SEMAPHORE.acquire(timeout=PREDICT_ACQUIRE_TIMEOUT_SEC):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Server is busy with another prediction. Please retry in a few seconds."},
            )
            return

        try:
            form = self._parse_form_data()
            image_item = form["image"] if "image" in form else None
            if image_item is None:
                raise ValueError("Missing `image` field in form data.")

            confidence = clamp_float(str(form.get("confidence", "0.25")), 0.05, 0.95, 0.25)
            image_size = clamp_int(str(form.get("image_size", "640")), 320, MAX_IMAGE_SIZE, 640)
            if PREDICT_IN_SUBPROCESS:
                payload = predict_via_subprocess(
                    file_item=image_item,
                    confidence=confidence,
                    image_size=image_size,
                )
            else:
                model_status, model_error = get_model_state()
                if model_status == "loading":
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "Model is warming up. Please retry in a minute."},
                    )
                    return
                if model_status == "error":
                    raise RuntimeError(model_error or "Model failed to load.")
                if model_status == "idle":
                    warm_model()
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "Model is warming up. Please retry in a minute."},
                    )
                    return
                payload = predict_once_from_upload(
                    file_item=image_item,
                    confidence=confidence,
                    image_size=image_size,
                )
                schedule_model_unload()
            self._send_json(HTTPStatus.OK, payload)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        except Exception as exc:
            if not PREDICT_IN_SUBPROCESS:
                schedule_model_unload()
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
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return

    def _send_status_only(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Origin, X-Requested-With, Authorization",
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


class RenderReadyThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32


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
    if MODEL_WARM_ON_STARTUP and not PREDICT_IN_SUBPROCESS:
        warm_model()
    server = RenderReadyThreadingHTTPServer((host, port), PCBRequestHandler)
    print(f"PCB defect backend listening on http://{host}:{port}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCB defect backend")
    parser.add_argument("--predict-once", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict_once:
        if not args.input or not args.output:
            raise SystemExit("--predict-once requires --input and --output")
        raise SystemExit(
            run_prediction_worker(
                input_path=args.input,
                output_path=args.output,
                confidence=clamp_float(str(args.confidence), 0.05, 0.95, 0.25),
                image_size=clamp_int(str(args.image_size), 320, MAX_IMAGE_SIZE, 640),
            )
        )
    run_server()
