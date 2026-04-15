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
import queue
import sys
import tempfile
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw
import psutil

# CRITICAL: Limit threads for Render's limited memory
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Force PyTorch to use minimal memory
import torch
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

MODEL_PATH = Path(__file__).with_name("pcb_defect_detector.pt")
LOCAL_YOLOV5_REPO = Path(__file__).with_name("yolov5")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# REDUCED FOR RENDER FREE TIER (512MB limit)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(3 * 1024 * 1024)))  # 3MB max
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(1_000_000)))  # 1MP max
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "640"))  # Smaller images
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "416"))  # Smaller model input
PREDICT_CONCURRENCY = int(os.getenv("PREDICT_CONCURRENCY", "1"))  # Only 1 at a time
PREDICT_ACQUIRE_TIMEOUT_SEC = float(os.getenv("PREDICT_ACQUIRE_TIMEOUT_SEC", "30"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "30"))  # Fewer detections
ANNOTATION_WIDTH = int(os.getenv("ANNOTATION_WIDTH", "1"))  # Thinner lines
MAX_RESPONSE_IMAGE_EDGE = int(os.getenv("MAX_RESPONSE_IMAGE_EDGE", "400"))  # Smaller preview
RESPONSE_JPEG_QUALITY = int(os.getenv("RESPONSE_JPEG_QUALITY", "40"))  # More compression
SERVER_REQUEST_TIMEOUT_SEC = float(os.getenv("SERVER_REQUEST_TIMEOUT_SEC", "60"))
MODEL_READY_TIMEOUT_SEC = float(os.getenv("MODEL_READY_TIMEOUT_SEC", "120"))
MODEL_WARM_ON_STARTUP = os.getenv("MODEL_WARM_ON_STARTUP", "true").strip().lower() not in {
    "0", "false", "no",
}
# CRITICAL: Aggressive model unloading for memory
MODEL_KEEPALIVE_SEC = float(os.getenv("MODEL_KEEPALIVE_SEC", "10"))  # Unload after 10s idle
PREDICT_IN_SUBPROCESS = os.getenv("PREDICT_IN_SUBPROCESS", "false").strip().lower() not in {
    "0", "false", "no",
}  # Disable subprocess to save memory
PREDICT_SUBPROCESS_TIMEOUT_SEC = float(os.getenv("PREDICT_SUBPROCESS_TIMEOUT_SEC", "120"))
PREFERRED_MODEL_RUNTIME = os.getenv("PREFERRED_MODEL_RUNTIME", "ultralytics").strip().lower()  # Use lighter runtime
INCLUDE_ANNOTATED_IMAGE = os.getenv("INCLUDE_ANNOTATED_IMAGE", "true").strip().lower() not in {
    "0", "false", "no",
}

DEFAULT_CLASS_LABELS = {
    0: "open", 1: "short", 2: "mousebite", 3: "spur", 4: "copper", 5: "pin-hole",
}

MODEL_LOCK = threading.Lock()
MODEL_STATE_LOCK = threading.Lock()
MODEL_STATE_CONDITION = threading.Condition(MODEL_STATE_LOCK)
PREDICT_SEMAPHORE = threading.BoundedSemaphore(max(1, PREDICT_CONCURRENCY))
MODEL_STATUS = "idle"
MODEL_ERROR = ""
MODEL_WARM_THREAD: threading.Thread | None = None
MODEL_UNLOAD_TIMER: threading.Timer | None = None

# Track memory usage for debugging
_last_memory_check = time.time()
_memory_warning_threshold = 400  # MB - warn if RSS exceeds this

def check_memory_pressure() -> bool:
    """Return True if memory usage is high."""
    global _last_memory_check
    now = time.time()
    if now - _last_memory_check < 5:  # Cache for 5 seconds
        return False
    
    _last_memory_check = now
    try:
        mem = psutil.Process(os.getpid()).memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        
        if rss_mb > _memory_warning_threshold:
            print(f"⚠️ High memory usage: {rss_mb:.1f}MB RSS", file=sys.stderr)
            return True
    except:
        pass
    return False

def aggressive_gc():
    """Aggressive garbage collection for memory-constrained environments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force Python to release memory to OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

@dataclass
class PredictionOutput:
    detections: list[dict[str, Any]]

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
            try:
                if self.backend_name == "ultralytics":
                    return self._predict_ultralytics(image, confidence, image_size)
                elif self.backend_name == "yolov5_local":
                    return self._predict_yolov5_local(image, confidence, image_size)
                else:
                    return self._predict_yolov5(image, confidence, image_size)
            finally:
                # Force cleanup after prediction
                if check_memory_pressure():
                    aggressive_gc()

    def _predict_ultralytics(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        # Use lower precision for memory savings
        with torch.inference_mode():
            results = self.model.predict(
                source=np.asarray(image),
                conf=confidence,
                imgsz=image_size,
                verbose=False,
                max_det=MAX_DETECTIONS,
                half=False,  # Don't use half precision on CPU
                device='cpu',
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
        
        # Cleanup
        del results
        aggressive_gc()
        
        return PredictionOutput(detections=detections)

    def _predict_yolov5(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        self.model.conf = confidence
        self.model.iou = 0.45
        self.model.max_det = MAX_DETECTIONS

        with torch.inference_mode():
            results = self.model(np.asarray(image), size=image_size)
        
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
        
        del results, raw
        aggressive_gc()
        
        return PredictionOutput(detections=detections)

    def _predict_yolov5_local(
        self, image: Image.Image, confidence: float, image_size: int
    ) -> PredictionOutput:
        image_np = np.asarray(image)
        stride = int(getattr(self.model, "stride", torch.tensor([32])).max())
        processed, ratio, pad = letterbox_image(image_np, new_shape=image_size, auto=False, stride=stride)
        processed = np.ascontiguousarray(processed.transpose((2, 0, 1)))
        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(processed).unsqueeze(0).to(device=device, dtype=torch.float32)
        tensor.div_(255.0)

        with torch.inference_mode():
            raw_predictions = self.model(tensor)
            if isinstance(raw_predictions, (list, tuple)):
                raw_predictions = raw_predictions[0]
            predictions = single_image_non_max_suppression(
                raw_predictions,
                conf_thres=confidence,
                iou_thres=0.45,
                max_det=MAX_DETECTIONS,
            )

        detections_tensor = predictions[0]
        if detections_tensor is None or len(detections_tensor) == 0:
            detections = empty_detections()
        else:
            detections_tensor[:, :4] = scale_boxes_to_original(
                detections_tensor[:, :4],
                image_shape=image_np.shape[:2],
                resized_shape=tensor.shape[2:],
                ratio=ratio,
                pad=pad,
            ).round()
            detections = build_detection_frame(
                xyxy=detections_tensor[:, :4].cpu().numpy(),
                confidences=detections_tensor[:, 4].cpu().numpy(),
                class_ids=detections_tensor[:, 5].cpu().numpy().astype(int),
                class_labels=self.class_labels,
            )

        del tensor, processed, raw_predictions, predictions
        aggressive_gc()
        
        return PredictionOutput(detections=detections)

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

def load_with_ultralytics(model_path: Path) -> ModelBackend:
    from ultralytics import YOLO
    
    # Load with minimal memory settings
    model = YOLO(str(model_path))
    model.model.eval()
    model.model.requires_grad_(False)
    
    # Move to CPU explicitly
    model.model = model.model.cpu()
    
    names = getattr(model.model, "names", None)
    return ModelBackend("ultralytics", model, normalize_class_labels(names))

def load_with_local_yolov5(model_path: Path) -> ModelBackend:
    repo_path = str(LOCAL_YOLOV5_REPO.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    attempt_load = importlib.import_module("models.experimental").attempt_load
    
    # Load with CPU and no gradients
    model = attempt_load(str(model_path), device=torch.device("cpu"), inplace=True, fuse=True)
    model.eval()
    model.requires_grad_(False)
    
    names = getattr(model, "names", None)
    return ModelBackend("yolov5_local", model, normalize_class_labels(names))

@lru_cache(maxsize=1)
def load_model() -> ModelBackend:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Only try ultralytics for lighter memory footprint
    try:
        return load_with_ultralytics(MODEL_PATH)
    except Exception as exc:
        # Fallback to local YOLOv5 if available
        if LOCAL_YOLOV5_REPO.exists():
            try:
                return load_with_local_yolov5(MODEL_PATH)
            except Exception as exc2:
                raise RuntimeError(f"Failed to load model: {exc} | {exc2}")
        raise

def unload_model() -> None:
    global MODEL_UNLOAD_TIMER, MODEL_STATUS, MODEL_ERROR

    with MODEL_STATE_LOCK:
        if MODEL_UNLOAD_TIMER is not None:
            MODEL_UNLOAD_TIMER.cancel()
            MODEL_UNLOAD_TIMER = None
        
        # Clear model from cache
        load_model.cache_clear()
        MODEL_STATUS = "idle"
        MODEL_ERROR = ""

    # Aggressive cleanup
    aggressive_gc()
    print(f"🧹 Model unloaded - Memory: {get_memory_stats()['rss_mb']}MB RSS", file=sys.stderr)

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
    with MODEL_STATE_CONDITION:
        MODEL_STATUS = status
        MODEL_ERROR = error
        MODEL_STATE_CONDITION.notify_all()

def get_model_state() -> tuple[str, str]:
    with MODEL_STATE_LOCK:
        return MODEL_STATUS, MODEL_ERROR

def ensure_model_ready(timeout_sec: float | None = None) -> ModelBackend:
    global MODEL_STATUS, MODEL_ERROR

    timeout = timeout_sec if timeout_sec is not None else MODEL_READY_TIMEOUT_SEC
    deadline = time.monotonic() + timeout
    should_load = False

    with MODEL_STATE_CONDITION:
        while True:
            if MODEL_STATUS == "ready":
                return load_model()
            if MODEL_STATUS == "error":
                raise RuntimeError(MODEL_ERROR or "Model failed to load.")
            if MODEL_STATUS == "idle":
                MODEL_STATUS = "loading"
                MODEL_ERROR = ""
                should_load = True
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Model warm-up timed out.")
            MODEL_STATE_CONDITION.wait(timeout=remaining)

    if should_load:
        try:
            backend = load_model()
        except Exception as exc:
            set_model_state("error", str(exc))
            aggressive_gc()
            raise
        set_model_state("ready")
        return backend

    raise RuntimeError("Model readiness check failed.")

def warm_model() -> None:
    global MODEL_WARM_THREAD

    status, _ = get_model_state()
    if status in {"loading", "ready"}:
        return

    def _load() -> None:
        try:
            ensure_model_ready()
        except Exception as exc:
            set_model_state("error", str(exc))
            aggressive_gc()

    MODEL_WARM_THREAD = threading.Thread(target=_load, name="model-warmup", daemon=True)
    MODEL_WARM_THREAD.start()

def parse_uploaded_image(file_item: dict[str, Any]) -> Image.Image:
    if not file_item["content"]:
        raise ValueError("No image file was uploaded.")

    filename = file_item["filename"] or "upload"
    extension = Path(filename).suffix.lower()
    if extension and extension not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image type.")

    raw_bytes = file_item["content"]
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    source_image = Image.open(BytesIO(raw_bytes))
    image = source_image.convert("RGB")
    source_image.close()
    
    # More aggressive downscaling for memory
    if image.width * image.height > MAX_INPUT_PIXELS or max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    
    return image

def render_annotated_image_data_url(image: Image.Image, detections: list[dict[str, Any]]) -> str:
    if not INCLUDE_ANNOTATED_IMAGE:
        return ""

    preview = image.copy()
    if max(preview.size) > MAX_RESPONSE_IMAGE_EDGE:
        preview.thumbnail((MAX_RESPONSE_IMAGE_EDGE, MAX_RESPONSE_IMAGE_EDGE), Image.Resampling.LANCZOS)

    scale_x = preview.width / image.width if image.width else 1.0
    scale_y = preview.height / image.height if image.height else 1.0
    draw = ImageDraw.Draw(preview)
    
    for row in detections:
        box = [
            row["x1"] * scale_x,
            row["y1"] * scale_y,
            row["x2"] * scale_x,
            row["y2"] * scale_y,
        ]
        label = f'{row["class_name"]} {row["confidence"]:.2f}'
        draw.rectangle(box, outline="red", width=ANNOTATION_WIDTH)
        draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill="red")

    # Save with compression
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=RESPONSE_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    buffer.close()
    preview.close()
    
    return f"data:image/jpeg;base64,{encoded}"

# [Keep the rest of the helper functions: letterbox_image, single_image_non_max_suppression, 
#  scale_boxes_to_original, summarize_detections, etc. - they remain the same]

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
        "detections": detections,
        "annotated_image": render_annotated_image_data_url(image, detections),
        "memory_mb": get_memory_stats()["rss_mb"],
    }

def predict_once_from_upload(file_item: dict[str, Any], confidence: float, image_size: int) -> dict[str, Any]:
    backend = ensure_model_ready()
    image = parse_uploaded_image(file_item)
    try:
        result = build_prediction_response(
            backend=backend,
            image=image,
            confidence=confidence,
            image_size=image_size,
        )
        schedule_model_unload()  # Schedule unload after prediction
        return result
    finally:
        image.close()
        aggressive_gc()

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

    def do_GET(self) -> None:
        if self.request_path == "/":
            mem_stats = get_memory_stats()
            self._send_json(HTTPStatus.OK, {
                "status": "ok",
                "service": "pcb-defect-backend",
                "memory": mem_stats,
            })
            return

        if self.request_path == "/api/health":
            model_status, model_error = get_model_state()
            mem_stats = get_memory_stats()
            
            # Auto-unload if memory is critically high
            if mem_stats["rss_mb"] > 450 and model_status == "ready":
                print(f"⚠️ Critical memory, forcing model unload", file=sys.stderr)
                unload_model()
                model_status = "idle"
            
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_status": model_status,
                    "model_error": model_error,
                    "ready": model_status == "ready",
                    "memory": mem_stats,
                },
            )
            return

        if self.request_path == "/api/prepare":
            if get_model_state()[0] == "idle":
                warm_model()
            model_status, model_error = get_model_state()
            self._send_json(
                HTTPStatus.ACCEPTED if model_status == "loading" else HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_status": model_status,
                    "ready": model_status == "ready",
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if self.request_path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return

        # Check memory before processing
        if check_memory_pressure():
            unload_model()  # Force unload if memory is high
            aggressive_gc()

        if not PREDICT_SEMAPHORE.acquire(timeout=PREDICT_ACQUIRE_TIMEOUT_SEC):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Server is busy. Please retry."},
            )
            return

        try:
            form = self._parse_form_data()
            image_item = form.get("image")
            if image_item is None:
                raise ValueError("Missing `image` field.")

            confidence = clamp_float(str(form.get("confidence", "0.25")), 0.05, 0.95, 0.25)
            image_size = clamp_int(str(form.get("image_size", "416")), 320, MAX_IMAGE_SIZE, 416)
            
            payload = predict_once_from_upload(
                file_item=image_item,
                confidence=confidence,
                image_size=image_size,
            )
            self._send_json(HTTPStatus.OK, payload)
            
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except TimeoutError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        finally:
            aggressive_gc()
            PREDICT_SEMAPHORE.release()

    def _parse_form_data(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Missing Content-Length header.")

        try:
            length = int(content_length)
        except ValueError:
            raise ValueError("Invalid Content-Length header.")
        
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)}MB.")

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
            pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Origin, X-Requested-With",
        )

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default logging to save memory/CPU
        pass

class RenderReadyThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 5  # Smaller queue for memory

def clamp_float(value: str, minimum: float, maximum: float, fallback: float) -> float:
    try:
        return max(minimum, min(float(value), maximum))
    except (TypeError, ValueError):
        return fallback

def clamp_int(value: str, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return fallback

def run_server() -> None:
    host = os.getenv("PCB_BACKEND_HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    
    # Warm model on startup if configured
    if MODEL_WARM_ON_STARTUP:
        warm_model()
    
    server = RenderReadyThreadingHTTPServer((host, port), PCBRequestHandler)
    print(f"✅ PCB backend listening on http://{host}:{port}")
    print(f"📊 Initial memory: {get_memory_stats()['rss_mb']}MB RSS")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        unload_model()
        server.shutdown()

if __name__ == "__main__":
    run_server()
