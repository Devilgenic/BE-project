import collections
import cv2
from flask import (
    Flask, render_template, request, jsonify,
    Response, send_from_directory
)
import os
import requests
import threading
import time
import json
import logging
import base64
import numpy as np
from datetime import datetime, timezone

from playsound3 import playsound
from flask_socketio import SocketIO, emit

IS_RENDER = os.environ.get("RENDER", "").lower() in ("true", "1")

from config import Config
from database import DetectionDatabase
from detection_engine import RTSPAlertGate, RTSPStreamStabilityFilter, ViolenceDetector
from violence_model import create_violence_detection_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LIVE_SOURCE_TYPES = {"webcam", "rtsp", "browser_webcam"}
LIVE_STATS_WINDOW_SECONDS = 5.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.static_folder = "static"
socketio = SocketIO(app, cors_allowed_origins="*")

config = Config()
db = DetectionDatabase(config.get("database.db_path", "detection_history.db"))

logger.info("Loading violence detection model...")
model = create_violence_detection_model()
if model:
    logger.info("Model loaded successfully.")
else:
    logger.warning("Model failed to load. Detection will be limited.")


def build_detector_config():
    return {
        "cnn_weight": config.get("weights.cnn_weight", 0.50),
        "optical_flow_weight": config.get("weights.optical_flow_weight", 0.30),
        "motion_energy_weight": config.get("weights.motion_energy_weight", 0.20),
        "confidence_threshold": config.get_threshold(),
        "frame_skip": config.get("detection.frame_skip", 3),
        "alert_cooldown_seconds": config.get("detection.alert_cooldown_seconds", 10),
        "window_size": 30,
    }


def build_live_detector_config():
    return {
        "cnn_weight": 0.75,
        "optical_flow_weight": 0.15,
        "motion_energy_weight": 0.10,
        "confidence_threshold": config.get_threshold(),
        "frame_skip": 2,
        "alert_cooldown_seconds": config.get("detection.alert_cooldown_seconds", 10),
        "window_size": 30,
        "suspicious_threshold": 0.50,
        "ema_alpha": 0.5,
        "optical_flow_divisor": 150.0,
        "motion_energy_multiplier": 1.5,
        "min_cnn_for_violence": 0.70,
    }


def build_rtsp_alert_gate():
    return RTSPAlertGate(
        alert_cooldown_seconds=config.get("detection.alert_cooldown_seconds", 10)
    )


def build_webcam_alert_gate():
    return RTSPAlertGate(
        min_sharpness=50.0,
        min_cnn_score=0.70,
        min_optical_flow_score=0.20,
        min_motion_energy_score=0.10,
        min_confidence=0.40,
        required_consecutive_hits=2,
        alert_cooldown_seconds=config.get("detection.alert_cooldown_seconds", 10),
    )


detector = ViolenceDetector(model, build_detector_config())
live_detector = ViolenceDetector(model, build_live_detector_config())
rtsp_stability_filter = RTSPStreamStabilityFilter()
rtsp_alert_gate = build_rtsp_alert_gate()
webcam_alert_gate = build_webcam_alert_gate()

state_lock = threading.Lock()
current_frame = None
current_frame_id = 0
current_frame_timestamp = 0.0
detection_active = False
detection_source = ""
latest_result = {}
last_error = ""
start_time = time.time()

live_state = {
    "raw_frame": None,
    "raw_frame_id": 0,
    "raw_frame_timestamp": 0.0,
    "stream_generation": 0,
    "input_timestamps": collections.deque(),
    "processing_timestamps": collections.deque(),
    "display_timestamps": collections.deque(),
    "dropped_frames": 0,
}

CAPTURES_DIR = "captures"
UPLOADS_DIR = "uploads"
ALERT_SOUND_PATH = os.path.join(BASE_DIR, "static", "alert_alarm.mp3")
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

sound_lock = threading.Lock()
sound_playing = False


def _set_runtime_error(message):
    global last_error
    with state_lock:
        last_error = message


def _clear_runtime_error():
    global last_error
    with state_lock:
        last_error = ""


def play_alert_sound():
    global sound_playing
    if IS_RENDER:
        return
    if not os.path.exists(ALERT_SOUND_PATH):
        logger.warning("Alert sound file not found: %s", ALERT_SOUND_PATH)
        return

    with sound_lock:
        if sound_playing:
            return
        sound_playing = True

    def _play():
        global sound_playing
        try:
            playsound(ALERT_SOUND_PATH, block=True)
        except Exception as exc:
            logger.error("Failed to play alert sound: %s", exc)
        finally:
            with sound_lock:
                sound_playing = False

    threading.Thread(target=_play, daemon=True).start()


def _prune_timestamps_unlocked(queue, now):
    while queue and now - queue[0] > LIVE_STATS_WINDOW_SECONDS:
        queue.popleft()


def _record_live_event_unlocked(key, event_time=None):
    if event_time is None:
        event_time = time.time()
    queue = live_state[key]
    queue.append(event_time)
    _prune_timestamps_unlocked(queue, event_time)


def _fps_from_queue_unlocked(key, now):
    queue = live_state[key]
    _prune_timestamps_unlocked(queue, now)
    if not queue:
        return 0.0
    window_span = min(LIVE_STATS_WINDOW_SECONDS, max(now - queue[0], 1e-6))
    return round(len(queue) / window_span, 2)


def _reset_live_session_state():
    global current_frame, current_frame_id, current_frame_timestamp, latest_result
    with state_lock:
        current_frame = None
        current_frame_id = 0
        current_frame_timestamp = 0.0
        latest_result = {}
        live_state["raw_frame"] = None
        live_state["raw_frame_id"] = 0
        live_state["raw_frame_timestamp"] = 0.0
        live_state["stream_generation"] = 0
        live_state["dropped_frames"] = 0
        live_state["input_timestamps"].clear()
        live_state["processing_timestamps"].clear()
        live_state["display_timestamps"].clear()
    rtsp_stability_filter.reset()
    rtsp_alert_gate.reset()
    webcam_alert_gate.reset()


def _mark_stream_reset_unlocked():
    live_state["raw_frame"] = None
    live_state["raw_frame_id"] = 0
    live_state["raw_frame_timestamp"] = 0.0
    live_state["stream_generation"] += 1


def _update_current_frame(frame, timestamp=None):
    global current_frame, current_frame_id, current_frame_timestamp
    if timestamp is None:
        timestamp = time.time()
    with state_lock:
        current_frame = frame.copy()
        current_frame_id += 1
        current_frame_timestamp = timestamp


def _clear_current_frame():
    global current_frame, current_frame_id, current_frame_timestamp
    with state_lock:
        current_frame = None
        current_frame_id = 0
        current_frame_timestamp = 0.0


def _set_latest_result(result):
    global latest_result
    with state_lock:
        latest_result = dict(result)


def _is_detection_running(source_type=None):
    with state_lock:
        if not detection_active:
            return False
        if source_type and detection_source != source_type:
            return False
        return True


def _get_live_camera_settings():
    return {
        "low_latency_mode": bool(config.get("camera.low_latency_mode", True)),
        "analysis_width": max(160, int(config.get("camera.analysis_width", 320) or 320)),
        "display_width": max(160, int(config.get("camera.display_width", 640) or 640)),
        "stream_fps": max(1.0, float(config.get("camera.stream_fps", 10) or 10)),
        "capture_buffer_size": max(1, int(config.get("camera.capture_buffer_size", 1) or 1)),
    }


def _get_stream_target_fps(source):
    if source in LIVE_SOURCE_TYPES:
        return _get_live_camera_settings()["stream_fps"]
    return 30.0


def _resize_frame_for_width(frame, target_width):
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame.copy()
    scale = target_width / float(width)
    target_height = max(1, int(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _apply_capture_property(cap, prop_name, value):
    prop = getattr(cv2, prop_name, None)
    if prop is None:
        return
    try:
        cap.set(prop, value)
    except Exception:
        logger.debug("Unable to set %s on capture.", prop_name)


def _open_live_capture(source_type, cam_source, live_settings):
    backend = None
    if source_type == "rtsp" and hasattr(cv2, "CAP_FFMPEG"):
        backend = cv2.CAP_FFMPEG

    cap = cv2.VideoCapture(cam_source, backend) if backend is not None else cv2.VideoCapture(cam_source)
    if backend is not None and not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        return None

    if live_settings["low_latency_mode"]:
        _apply_capture_property(cap, "CAP_PROP_BUFFERSIZE", live_settings["capture_buffer_size"])
    _apply_capture_property(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", 2000)
    _apply_capture_property(cap, "CAP_PROP_READ_TIMEOUT_MSEC", 2000)
    return cap


def _store_live_raw_frame(frame, frame_timestamp):
    with state_lock:
        live_state["raw_frame"] = frame
        live_state["raw_frame_id"] += 1
        live_state["raw_frame_timestamp"] = frame_timestamp
        _record_live_event_unlocked("input_timestamps", frame_timestamp)


def _get_live_stats():
    now = time.time()
    with state_lock:
        input_fps = _fps_from_queue_unlocked("input_timestamps", now)
        processing_fps = _fps_from_queue_unlocked("processing_timestamps", now)
        display_fps = _fps_from_queue_unlocked("display_timestamps", now)
        last_frame_age_ms = 0.0
        if current_frame_timestamp:
            last_frame_age_ms = round(max(0.0, (now - current_frame_timestamp) * 1000), 2)
        return {
            "live_input_fps": input_fps,
            "live_processing_fps": processing_fps,
            "live_display_fps": display_fps,
            "dropped_live_frames": int(live_state["dropped_frames"]),
            "last_frame_age_ms": last_frame_age_ms,
        }


def send_telegram_alert(image_path):
    bot_token = config.get("telegram.bot_token", "")
    chat_id = config.get("telegram.chat_id", "")
    if not bot_token or not chat_id:
        logger.info("Telegram not configured. Skipping alert.")
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": chat_id,
                "caption": "VIOLENCE DETECTED! - Violence Detection System Alert"
            }
            resp = requests.post(url, files=files, data=data, timeout=10)
            logger.info("Telegram response: %s", resp.status_code)
            return resp.status_code == 200
    except Exception as e:
        logger.error("Error sending Telegram alert: %s", e)
        return False


def handle_detection_result(result, frame, source):
    global latest_result
    with state_lock:
        latest_result = result

    if result.get("should_alert"):
        play_alert_sound()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        frame_filename = f"violence_{timestamp}.jpg"
        frame_path = os.path.join(CAPTURES_DIR, frame_filename)
        cv2.imwrite(frame_path, frame)

        alert_sent = send_telegram_alert(frame_path)

        db.add_event(
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=result["confidence"],
            cnn_score=result["cnn_score"],
            optical_flow_score=result["optical_flow_score"],
            motion_energy_score=result["motion_energy_score"],
            frame_path=frame_filename,
            source=source,
            alert_sent=1 if alert_sent else 0,
        )
        logger.warning("VIOLENCE DETECTED! Confidence: %.4f | Source: %s", result["confidence"], source)


def process_video(video_path):
    global detection_active, detection_source, latest_result, last_error
    with state_lock:
        detection_active = True
        detection_source = "upload"
        latest_result = {}
        last_error = ""
    detector.reset()
    _clear_current_frame()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        _set_runtime_error("Failed to open uploaded video.")
        with state_lock:
            detection_active = False
            detection_source = ""
        return

    while cap.isOpened():
        with state_lock:
            if not detection_active or detection_source != "upload":
                break
        ret, frame = cap.read()
        if not ret:
            break

        _update_current_frame(frame)

        result = detector.analyze_frame(frame)
        handle_detection_result(result, frame, "upload")
        time.sleep(0.01)

    cap.release()
    with state_lock:
        detection_active = False
        detection_source = ""
    logger.info("Video processing completed.")


def _capture_live_frames(source_type):
    global detection_active, detection_source
    live_settings = _get_live_camera_settings()
    if source_type == "rtsp":
        cam_source = config.get("camera.rtsp_url", "")
    else:
        cam_source = config.get("camera.webcam_index", 0)

    max_open_attempts = 3
    max_read_failures = 3
    open_retry_delay_seconds = 2

    while _is_detection_running(source_type):
        cap = None
        for attempt in range(1, max_open_attempts + 1):
            if not _is_detection_running(source_type):
                return
            candidate_cap = _open_live_capture(source_type, cam_source, live_settings)
            if candidate_cap is not None and candidate_cap.isOpened():
                cap = candidate_cap
                with state_lock:
                    _mark_stream_reset_unlocked()
                live_detector.reset_temporal_state()
                if source_type == "rtsp":
                    rtsp_stability_filter.reset()
                    rtsp_alert_gate.reset()
                elif source_type == "webcam":
                    webcam_alert_gate.reset()
                _clear_runtime_error()
                logger.info("%s stream opened successfully.", source_type.upper())
                break

            if candidate_cap is not None:
                candidate_cap.release()
            logger.error(
                "Failed to open %s source: %s (attempt %s/%s)",
                source_type,
                cam_source,
                attempt,
                max_open_attempts,
            )
            _set_runtime_error(
                f"Failed to open {source_type.upper()} stream (attempt {attempt}/{max_open_attempts})."
            )
            if attempt < max_open_attempts:
                time.sleep(open_retry_delay_seconds)

        if cap is None:
            _set_runtime_error(f"Cannot open {source_type.upper()} stream. Check source/network.")
            with state_lock:
                detection_active = False
                detection_source = ""
            return

        read_failures = 0
        while cap.isOpened():
            if not _is_detection_running(source_type):
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                logger.warning(
                    "Read failure on %s source (count %s/%s)",
                    source_type,
                    read_failures,
                    max_read_failures,
                )
                if read_failures < max_read_failures:
                    _set_runtime_error(
                        f"Temporary {source_type.upper()} stream read failure "
                        f"({read_failures}/{max_read_failures}). Retrying..."
                    )
                    time.sleep(0.1)
                    continue

                _set_runtime_error(f"{source_type.upper()} stream lost. Reconnecting...")
                break

            if read_failures > 0:
                read_failures = 0
                _clear_runtime_error()

            _store_live_raw_frame(frame, time.time())

        cap.release()
        if _is_detection_running(source_type):
            logger.info("Reconnecting %s stream...", source_type.upper())
            time.sleep(open_retry_delay_seconds)


def _process_live_frames(source_type):
    live_settings = _get_live_camera_settings()
    analysis_width = live_settings["analysis_width"]
    display_width = live_settings["display_width"]
    last_frame_id = 0
    last_generation = -1

    while _is_detection_running(source_type):
        with state_lock:
            generation = live_state["stream_generation"]
            frame_id = live_state["raw_frame_id"]
            frame_timestamp = live_state["raw_frame_timestamp"]
            raw_frame = live_state["raw_frame"]

        if generation != last_generation:
            live_detector.reset_temporal_state()
            if source_type == "rtsp":
                rtsp_stability_filter.reset()
                rtsp_alert_gate.reset()
            elif source_type == "webcam":
                webcam_alert_gate.reset()
            last_generation = generation
            last_frame_id = 0

        if raw_frame is None or frame_id == 0 or frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        skipped_frames = max(0, frame_id - last_frame_id - 1) if last_frame_id else 0
        if skipped_frames:
            with state_lock:
                live_state["dropped_frames"] += skipped_frames

        raw_frame_copy = raw_frame.copy()
        analysis_frame = _resize_frame_for_width(raw_frame_copy, analysis_width)
        display_frame = _resize_frame_for_width(raw_frame_copy, display_width)

        if source_type == "rtsp" and not rtsp_stability_filter.allow_frame(analysis_frame):
            live_detector.reset_temporal_state()
            _set_latest_result(live_detector._empty_result())
            _update_current_frame(display_frame, frame_timestamp)
            with state_lock:
                _record_live_event_unlocked("processing_timestamps", time.time())
            if rtsp_stability_filter.last_reason == "global_motion":
                logger.info(
                    "Skipping unstable RTSP frame after camera/stream motion: %s",
                    rtsp_stability_filter.last_metrics,
                )
            last_frame_id = frame_id
            continue

        result = live_detector.analyze_frame(analysis_frame)
        if source_type == "rtsp":
            result = rtsp_alert_gate.filter_result(result, analysis_frame)
            if not result["violence_detected"] and rtsp_alert_gate.last_reason != "confirmed_candidate":
                logger.debug(
                    "RTSP gate blocked frame: %s | %s",
                    rtsp_alert_gate.last_reason,
                    rtsp_alert_gate.last_metrics,
                )
        elif source_type == "webcam":
            result = webcam_alert_gate.filter_result(result, analysis_frame)
            if not result["violence_detected"] and webcam_alert_gate.last_reason != "confirmed_candidate":
                logger.debug(
                    "Webcam gate blocked frame: %s | %s",
                    webcam_alert_gate.last_reason,
                    webcam_alert_gate.last_metrics,
                )
        processed_time = time.time()
        if not _is_detection_running(source_type):
            break

        _update_current_frame(display_frame, frame_timestamp)
        with state_lock:
            _record_live_event_unlocked("processing_timestamps", processed_time)

        handle_detection_result(result, display_frame, source_type)
        last_frame_id = frame_id


def process_camera(source_type="webcam"):
    global detection_active, detection_source
    live_detector.reset()
    _reset_live_session_state()

    capture_thread = threading.Thread(target=_capture_live_frames, args=(source_type,), daemon=True)
    processing_thread = threading.Thread(target=_process_live_frames, args=(source_type,), daemon=True)
    capture_thread.start()
    processing_thread.start()
    capture_thread.join()
    processing_thread.join()

    with state_lock:
        if detection_source == source_type:
            detection_active = False
            detection_source = ""
    logger.info("%s processing stopped.", source_type)


@app.route("/")
def index():
    return render_template("index.html", page="dashboard")


@app.route("/history")
def history_page():
    return render_template("history.html", page="history")


@app.route("/settings")
def settings_page():
    return render_template("settings.html", page="settings", config=config.to_dict())


@app.route("/about")
def about_page():
    return render_template("about.html", page="about")


@app.route("/upload", methods=["POST"])
def upload_video():
    global detection_active, detection_source, latest_result, last_error
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected"}), 400

    filename = os.path.join(UPLOADS_DIR, "uploaded_video.mp4")
    file.save(filename)

    with state_lock:
        detection_active = True
        detection_source = "upload"
        latest_result = {}
        last_error = ""

    thread = threading.Thread(target=process_video, args=(filename,), daemon=True)
    thread.start()
    return jsonify({"message": "Video uploaded. Processing started."})


@app.route("/start_webcam", methods=["POST"])
def start_webcam():
    global detection_active, detection_source, latest_result, last_error
    with state_lock:
        if detection_active:
            return jsonify({"error": "Detection is already running."}), 409
        detection_active = True
        detection_source = "webcam"
        latest_result = {}
        last_error = ""
    thread = threading.Thread(target=process_camera, args=("webcam",), daemon=True)
    thread.start()
    return jsonify({"message": "Webcam detection started."})


@app.route("/start_rtsp", methods=["POST"])
def start_rtsp():
    rtsp_url = str(config.get("camera.rtsp_url", "")).strip()
    if not rtsp_url:
        return jsonify({"error": "RTSP URL not configured."}), 400

    global detection_active, detection_source, latest_result, last_error
    with state_lock:
        if detection_active:
            return jsonify({"error": "Detection is already running."}), 409
        detection_active = True
        detection_source = "rtsp"
        latest_result = {}
        last_error = ""

    thread = threading.Thread(target=process_camera, args=("rtsp",), daemon=True)
    thread.start()
    return jsonify({"message": "RTSP camera detection started."})


@app.route("/start_browser_webcam", methods=["POST"])
def start_browser_webcam():
    global detection_active, detection_source, latest_result, last_error
    with state_lock:
        if detection_active:
            return jsonify({"error": "Detection is already running."}), 409
        detection_active = True
        detection_source = "browser_webcam"
        latest_result = {}
        last_error = ""
    live_detector.reset()
    webcam_alert_gate.reset()
    _clear_current_frame()
    return jsonify({"message": "Browser webcam detection started."})


@socketio.on("browser_frame")
def handle_browser_frame(data):
    global latest_result
    with state_lock:
        if not detection_active or detection_source != "browser_webcam":
            return

    try:
        img_data = data.get("image", "")
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    analysis_frame = _resize_frame_for_width(frame, 320)
    display_frame = _resize_frame_for_width(frame, 640)

    result = live_detector.analyze_frame(analysis_frame)
    result = webcam_alert_gate.filter_result(result, analysis_frame)

    _update_current_frame(display_frame)
    _set_latest_result(result)
    handle_detection_result(result, display_frame, "browser_webcam")

    emit("detection_result", {
        "violence_detected": result["violence_detected"],
        "confidence": result["confidence"],
        "cnn_score": result["cnn_score"],
        "optical_flow_score": result["optical_flow_score"],
        "motion_energy_score": result["motion_energy_score"],
        "fusion_score": result["fusion_score"],
        "smoothed_score": result["smoothed_score"],
        "should_alert": result["should_alert"],
    })


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    global detection_active, detection_source, last_error
    with state_lock:
        detection_active = False
        detection_source = ""
        last_error = ""
    detector.reset()
    live_detector.reset()
    return jsonify({"message": "Detection stopped."})


@app.route("/reset", methods=["POST"])
def reset_system():
    global detection_active, detection_source, latest_result, last_error
    with state_lock:
        detection_active = False
        detection_source = ""
        latest_result = {}
        last_error = ""
    _reset_live_session_state()
    detector.reset()
    live_detector.reset()
    return jsonify({"message": "System reset successfully."})


@app.route("/status")
def status():
    with state_lock:
        result = dict(latest_result) if latest_result else {}
        active = detection_active
        source = detection_source
        error = last_error
    return jsonify({
        "detection_active": active,
        "detection_source": source,
        "last_error": error,
        "violence_detected": result.get("violence_detected", False),
        "confidence": result.get("confidence", 0),
        "cnn_score": result.get("cnn_score", 0),
        "optical_flow_score": result.get("optical_flow_score", 0),
        "motion_energy_score": result.get("motion_energy_score", 0),
        "fusion_score": result.get("fusion_score", 0),
        "smoothed_score": result.get("smoothed_score", 0),
        "should_alert": result.get("should_alert", False),
    })


@app.route("/video_feed")
def video_feed():
    def generate():
        last_sent_frame_id = 0
        last_yield_time = 0.0
        while True:
            with state_lock:
                active = detection_active
                source = detection_source
                frame_id = current_frame_id
                frame_timestamp = current_frame_timestamp
                has_new_frame = frame_id != 0 and frame_id != last_sent_frame_id and current_frame is not None
                frame = current_frame.copy() if has_new_frame else None

            if not active:
                break

            target_fps = _get_stream_target_fps(source)
            min_interval = 1.0 / target_fps if target_fps > 0 else 0.0

            if frame is None:
                time.sleep(max(0.01, min_interval / 2 if min_interval else 0.01))
                continue

            wait_time = (last_yield_time + min_interval) - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                last_sent_frame_id = frame_id
                last_yield_time = time.time()
                if source in LIVE_SOURCE_TYPES and frame_timestamp:
                    with state_lock:
                        _record_live_event_unlocked("display_timestamps", last_yield_time)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    events = db.get_events(limit=limit, offset=offset)
    total = db.get_event_count()
    return jsonify({"events": events, "total": total})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    db.clear_events()
    return jsonify({"message": "History cleared."})


@app.route("/api/stats")
def api_stats():
    db_stats = db.get_stats()
    with state_lock:
        source = detection_source
    if source in LIVE_SOURCE_TYPES:
        det_stats = live_detector.get_stats()
    else:
        det_stats = detector.get_stats()
    uptime = int(time.time() - start_time)
    return jsonify({
        **db_stats,
        **det_stats,
        **_get_live_stats(),
        "uptime_seconds": uptime,
        "model_loaded": model is not None,
    })


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(config.to_dict())


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    for section_key, section_val in data.items():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                config.set(f"{section_key}.{k}", v)
        else:
            config.set(section_key, section_val)

    global detector, live_detector, rtsp_alert_gate, webcam_alert_gate
    detector = ViolenceDetector(model, build_detector_config())
    live_detector = ViolenceDetector(model, build_live_detector_config())
    rtsp_alert_gate = build_rtsp_alert_gate()
    webcam_alert_gate = build_webcam_alert_gate()

    return jsonify({"message": "Settings saved successfully."})


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/api/confidence_stream")
def confidence_stream():
    def generate():
        try:
            while True:
                with state_lock:
                    active = detection_active
                    result = dict(latest_result) if latest_result else {}
                    source = detection_source
                    error = last_error
                data = json.dumps({
                    "confidence": result.get("confidence", 0),
                    "cnn_score": result.get("cnn_score", 0),
                    "optical_flow_score": result.get("optical_flow_score", 0),
                    "motion_energy_score": result.get("motion_energy_score", 0),
                    "fusion_score": result.get("fusion_score", 0),
                    "violence_detected": result.get("violence_detected", False),
                    "should_alert": result.get("should_alert", False),
                    "active": active,
                    "detection_source": source,
                    "last_error": error,
                    "model_loaded": model is not None,
                })
                yield f"data: {data}\n\n"
                time.sleep(0.5)
        except GeneratorExit:
            pass

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    socketio.run(
        app,
        debug=config.get("server.debug", False),
        host=config.get("server.host", "0.0.0.0"),
        port=config.get("server.port", 5000),
    )
