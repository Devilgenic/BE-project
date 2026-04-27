import cv2
import numpy as np
import time
import collections
import threading


class ViolenceDetector:
    def __init__(self, model, config):
        self.model = model
        self.cnn_weight = config.get("cnn_weight", 0.50)
        self.optical_flow_weight = config.get("optical_flow_weight", 0.30)
        self.motion_energy_weight = config.get("motion_energy_weight", 0.20)
        self.confidence_threshold = config.get("confidence_threshold", 0.55)
        self.frame_skip = config.get("frame_skip", 3)
        self.alert_cooldown = config.get("alert_cooldown_seconds", 10)
        window_size = config.get("window_size", 30)

        # Tunable params (defaults match original hardcoded values for backward compat)
        self.suspicious_threshold = config.get("suspicious_threshold", 0.35)
        self.ema_alpha = config.get("ema_alpha", 0.3)
        self.optical_flow_divisor = config.get("optical_flow_divisor", 50.0)
        self.motion_energy_multiplier = config.get("motion_energy_multiplier", 3.0)
        self.min_cnn_for_violence = config.get("min_cnn_for_violence", 0.0)

        self.history = collections.deque(maxlen=window_size)
        self.smoothed_score = 0.0
        self.last_alert_time = 0
        self.frames_processed = 0
        self.frame_counter = 0
        self.prev_gray = None
        self._last_result = self._empty_result()
        self._lock = threading.Lock()

    def analyze_frame(self, frame):
        with self._lock:
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                skipped_result = self._last_result.copy()
                skipped_result["should_alert"] = False
                return skipped_result
            self.frames_processed += 1
            prev_gray = self.prev_gray

        cnn_score = self._compute_cnn_score(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        optical_flow_score = self._compute_optical_flow(prev_gray, curr_gray)
        motion_energy_score = self._compute_motion_energy(prev_gray, curr_gray)

        fusion_score = (
            self.cnn_weight * cnn_score
            + self.optical_flow_weight * optical_flow_score
            + self.motion_energy_weight * motion_energy_score
        )

        with self._lock:
            self.prev_gray = curr_gray
            self.smoothed_score = self.ema_alpha * fusion_score + (1 - self.ema_alpha) * self.smoothed_score

            entry = {
                "fusion_score": fusion_score,
                "cnn_score": cnn_score,
                "optical_flow_score": optical_flow_score,
                "motion_energy_score": motion_energy_score,
                "timestamp": time.time(),
            }
            self.history.append(entry)

            violence_detected, should_alert = self._make_decision()

            result = {
                "violence_detected": violence_detected,
                "confidence": round(self.smoothed_score, 4),
                "cnn_score": round(cnn_score, 4),
                "optical_flow_score": round(optical_flow_score, 4),
                "motion_energy_score": round(motion_energy_score, 4),
                "fusion_score": round(fusion_score, 4),
                "smoothed_score": round(self.smoothed_score, 4),
                "should_alert": should_alert,
            }
            self._last_result = result
            return result

    def reset(self):
        with self._lock:
            self.history.clear()
            self.smoothed_score = 0.0
            self.prev_gray = None
            self.frame_counter = 0
            self.frames_processed = 0
            self._last_result = self._empty_result()

    def reset_temporal_state(self):
        with self._lock:
            self.history.clear()
            self.smoothed_score = 0.0
            self.prev_gray = None
            self.frame_counter = 0
            self._last_result = self._empty_result()

    def get_history(self):
        with self._lock:
            return list(self.history)

    def get_stats(self):
        with self._lock:
            return {
                "frames_processed": self.frames_processed,
                "window_size": self.history.maxlen,
                "smoothed_score": round(self.smoothed_score, 4),
                "history_length": len(self.history),
            }

    def _compute_cnn_score(self, frame):
        if self.model is None:
            return 0.0
        try:
            resized = cv2.resize(frame, (128, 128))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed = np.expand_dims(normalized, axis=0)
            prediction = self.model.predict(processed, verbose=0)
            pred = np.asarray(prediction)
            if pred.ndim >= 2 and pred.shape[1] >= 2:
                score = float(pred[0][1])
            elif pred.ndim >= 2:
                score = float(pred[0][0])
            else:
                score = float(pred.reshape(-1)[0])
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_optical_flow(self, prev_gray, curr_gray):
        if prev_gray is None:
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude)
            score = float(np.clip(mean_mag * std_mag / self.optical_flow_divisor, 0.0, 1.0))
            return score
        except Exception:
            return 0.0

    def _compute_motion_energy(self, prev_gray, curr_gray):
        if prev_gray is None:
            return 0.0
        try:
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            changed_pixels = np.count_nonzero(thresh)
            ratio = changed_pixels / total_pixels
            score = float(np.clip(ratio * self.motion_energy_multiplier, 0.0, 1.0))
            return score
        except Exception:
            return 0.0

    def _make_decision(self):
        if len(self.history) == 0:
            return False, False

        suspicious_count = sum(
            1 for e in self.history if e["fusion_score"] > self.suspicious_threshold
        )
        suspicious_ratio = suspicious_count / len(self.history)

        recent = list(self.history)[-5:]
        recent_peak = max(e["fusion_score"] for e in recent) if recent else 0

        # CNN floor check: require CNN evidence if min_cnn_for_violence > 0
        cnn_ok = True
        if self.min_cnn_for_violence > 0:
            recent_cnn_peak = max(e["cnn_score"] for e in recent) if recent else 0
            cnn_ok = recent_cnn_peak > self.min_cnn_for_violence

        violence_detected = (
            suspicious_ratio > 0.4
            and recent_peak > self.confidence_threshold
            and self.smoothed_score > self.confidence_threshold * 0.8
            and cnn_ok
        )

        should_alert = False
        now = time.time()
        if violence_detected and (now - self.last_alert_time > self.alert_cooldown):
            should_alert = True
            self.last_alert_time = now

        return violence_detected, should_alert

    def _empty_result(self):
        return {
            "violence_detected": False,
            "confidence": 0.0,
            "cnn_score": 0.0,
            "optical_flow_score": 0.0,
            "motion_energy_score": 0.0,
            "fusion_score": 0.0,
            "smoothed_score": 0.0,
            "should_alert": False,
        }


class RTSPStreamStabilityFilter:
    """
    Ignore startup/repositioning frames from phone RTSP streams.

    Phone-based RTSP feeds often produce a burst of full-frame motion while the
    device is being positioned or when the stream drops frames and jumps ahead.
    Those jumps can look like violence to optical-flow and motion-energy logic,
    so this filter only blocks RTSP analysis until the stream settles again.
    """

    def __init__(
        self,
        warmup_frames=5,
        diff_threshold=25,
        min_global_shift_pixels=6.0,
        min_phase_response=0.35,
        min_changed_ratio=0.04,
    ):
        self.warmup_frames = max(0, int(warmup_frames))
        self.diff_threshold = max(1, int(diff_threshold))
        self.min_global_shift_pixels = float(min_global_shift_pixels)
        self.min_phase_response = float(min_phase_response)
        self.min_changed_ratio = float(min_changed_ratio)
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.warmup_remaining = self.warmup_frames
        self.last_reason = "reset"
        self.last_metrics = {
            "shift_magnitude": 0.0,
            "phase_response": 0.0,
            "changed_ratio": 0.0,
        }

    def allow_frame(self, frame):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (9, 9), 0)
        curr_gray_f32 = curr_gray.astype(np.float32)

        if self.prev_gray is None:
            self.prev_gray = curr_gray_f32
            self.warmup_remaining = self.warmup_frames
            self.last_reason = "bootstrap"
            self.last_metrics = {
                "shift_magnitude": 0.0,
                "phase_response": 0.0,
                "changed_ratio": 0.0,
            }
            return False

        shift, response = cv2.phaseCorrelate(self.prev_gray, curr_gray_f32)
        shift_magnitude = float(np.hypot(shift[0], shift[1]))
        diff = cv2.absdiff(self.prev_gray.astype(np.uint8), curr_gray)
        changed_ratio = float(np.count_nonzero(diff > self.diff_threshold)) / float(diff.size)

        self.prev_gray = curr_gray_f32
        self.last_metrics = {
            "shift_magnitude": round(shift_magnitude, 4),
            "phase_response": round(float(response), 4),
            "changed_ratio": round(changed_ratio, 4),
        }

        if (
            shift_magnitude >= self.min_global_shift_pixels
            and float(response) >= self.min_phase_response
            and changed_ratio >= self.min_changed_ratio
        ):
            self.warmup_remaining = self.warmup_frames
            self.last_reason = "global_motion"
            return False

        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            self.last_reason = "warmup"
            return False

        self.last_reason = "stable"
        return True


class RTSPAlertGate:
    """
    Apply stricter confirmation for phone RTSP streams.

    Root causes for RTSP false alarms in this project:
    1. The CNN model can score ordinary phone-camera frames around 0.6-0.7.
    2. Phone motion / blur can drive motion-energy and optical-flow unusually high.

    To avoid false alerts, RTSP requires:
    - a sharper frame,
    - stronger CNN evidence,
    - corroborating motion,
    - repeated confirmation across consecutive analyzed frames.
    """

    def __init__(
        self,
        min_sharpness=80.0,
        min_cnn_score=0.85,
        min_optical_flow_score=0.35,
        min_motion_energy_score=0.20,
        min_confidence=0.45,
        required_consecutive_hits=2,
        alert_cooldown_seconds=10,
    ):
        self.min_sharpness = float(min_sharpness)
        self.min_cnn_score = float(min_cnn_score)
        self.min_optical_flow_score = float(min_optical_flow_score)
        self.min_motion_energy_score = float(min_motion_energy_score)
        self.min_confidence = float(min_confidence)
        self.required_consecutive_hits = max(1, int(required_consecutive_hits))
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)
        self.reset()

    def reset(self):
        self.consecutive_hits = 0
        self.last_alert_time = 0.0
        self.last_reason = "reset"
        self.last_metrics = {
            "sharpness": 0.0,
            "cnn_score": 0.0,
            "optical_flow_score": 0.0,
            "motion_energy_score": 0.0,
            "confidence": 0.0,
            "consecutive_hits": 0,
        }

    def filter_result(self, result, frame):
        filtered = dict(result)
        sharpness = self._compute_sharpness(frame)
        cnn_score = float(filtered.get("cnn_score", 0.0))
        optical_flow_score = float(filtered.get("optical_flow_score", 0.0))
        motion_energy_score = float(filtered.get("motion_energy_score", 0.0))
        confidence = float(filtered.get("confidence", 0.0))

        qualifies = (
            sharpness >= self.min_sharpness
            and cnn_score >= self.min_cnn_score
            and optical_flow_score >= self.min_optical_flow_score
            and motion_energy_score >= self.min_motion_energy_score
            and confidence >= self.min_confidence
        )

        if qualifies:
            self.consecutive_hits += 1
            self.last_reason = "confirmed_candidate"
        else:
            self.consecutive_hits = 0
            if sharpness < self.min_sharpness:
                self.last_reason = "blurry_frame"
            elif cnn_score < self.min_cnn_score:
                self.last_reason = "weak_cnn"
            elif optical_flow_score < self.min_optical_flow_score:
                self.last_reason = "weak_optical_flow"
            elif motion_energy_score < self.min_motion_energy_score:
                self.last_reason = "weak_motion_energy"
            else:
                self.last_reason = "low_confidence"

        self.last_metrics = {
            "sharpness": round(sharpness, 4),
            "cnn_score": round(cnn_score, 4),
            "optical_flow_score": round(optical_flow_score, 4),
            "motion_energy_score": round(motion_energy_score, 4),
            "confidence": round(confidence, 4),
            "consecutive_hits": int(self.consecutive_hits),
        }

        confirmed = qualifies and self.consecutive_hits >= self.required_consecutive_hits
        filtered["violence_detected"] = confirmed
        filtered["should_alert"] = False

        now = time.time()
        if confirmed and (now - self.last_alert_time > self.alert_cooldown_seconds):
            filtered["should_alert"] = True
            self.last_alert_time = now

        return filtered

    def _compute_sharpness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
