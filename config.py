"""
Configuration manager for Flask violence detection system.
Loads and saves configuration from config.json with thread-safe access.
"""

import json
import os
import threading
import copy


DEFAULT_CONFIG = {
    "telegram": {
        "bot_token": "",
        "chat_id": ""
    },
    "detection": {
        "sensitivity": "medium",
        "frame_skip": 3,
        "alert_cooldown_seconds": 10
    },
    "weights": {
        "cnn_weight": 0.50,
        "optical_flow_weight": 0.30,
        "motion_energy_weight": 0.20
    },
    "camera": {
        "source": "webcam",
        "webcam_index": 0,
        "rtsp_url": "",
        "low_latency_mode": True,
        "analysis_width": 320,
        "display_width": 640,
        "stream_fps": 10,
        "capture_buffer_size": 1
    },
    "database": {
        "db_path": "detection_history.db"
    },
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False
    }
}

SENSITIVITY_THRESHOLDS = {
    "low": 0.70,
    "medium": 0.55,
    "high": 0.40
}

CONFIG_PATH = "config.json"


class Config:
    def __init__(self):
        self._lock = threading.Lock()
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    def load(self):
        with self._lock:
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r") as f:
                        loaded = json.load(f)
                    self._deep_merge(self._config, loaded)
                except (json.JSONDecodeError, IOError):
                    pass
            self._save_unlocked()

    def save(self):
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError:
            pass

    def get(self, key, default=None):
        with self._lock:
            return self._nested_get(self._config, key, default)

    def set(self, key, value):
        with self._lock:
            self._nested_set(self._config, key, value)
            self._save_unlocked()

    def get_threshold(self):
        sensitivity = self.get("detection.sensitivity", "medium")
        return SENSITIVITY_THRESHOLDS.get(sensitivity, 0.55)

    def to_dict(self):
        with self._lock:
            return copy.deepcopy(self._config)

    def _nested_get(self, d, key, default=None):
        keys = key.split(".")
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def _nested_set(self, d, key, value):
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def _deep_merge(self, base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
