import unittest

import numpy as np

from detection_engine import RTSPAlertGate, ViolenceDetector


class DummyModel:
    def __init__(self, score=1.0):
        self.score = float(score)

    def predict(self, processed, verbose=0):
        return np.array([[self.score]], dtype=np.float32)


class RTSPAlertGateTests(unittest.TestCase):
    def test_blocks_common_false_positive_patterns(self):
        gate = RTSPAlertGate(
            min_sharpness=80.0,
            min_cnn_score=0.85,
            min_optical_flow_score=0.35,
            min_motion_energy_score=0.20,
            min_confidence=0.45,
            required_consecutive_hits=2,
            alert_cooldown_seconds=10,
        )
        sharp_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        false_patterns = [
            {
                "violence_detected": True,
                "confidence": 0.5386,
                "cnn_score": 0.7799,
                "optical_flow_score": 1.0,
                "motion_energy_score": 1.0,
                "should_alert": True,
            },
            {
                "violence_detected": True,
                "confidence": 0.5161,
                "cnn_score": 0.9323,
                "optical_flow_score": 0.2724,
                "motion_energy_score": 1.0,
                "should_alert": True,
            },
            {
                "violence_detected": True,
                "confidence": 0.4788,
                "cnn_score": 0.3891,
                "optical_flow_score": 0.8467,
                "motion_energy_score": 0.8343,
                "should_alert": True,
            },
        ]

        for pattern in false_patterns:
            filtered = gate.filter_result(pattern, sharp_frame)
            self.assertFalse(filtered["violence_detected"])
            self.assertFalse(filtered["should_alert"])

    def test_requires_two_confirmed_rtsp_hits_before_alerting(self):
        gate = RTSPAlertGate(required_consecutive_hits=2, alert_cooldown_seconds=10)
        sharp_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        strong_result = {
            "violence_detected": True,
            "confidence": 0.72,
            "cnn_score": 0.94,
            "optical_flow_score": 0.64,
            "motion_energy_score": 0.71,
            "should_alert": True,
        }

        first = gate.filter_result(strong_result, sharp_frame)
        self.assertFalse(first["violence_detected"])
        self.assertFalse(first["should_alert"])

        second = gate.filter_result(strong_result, sharp_frame)
        self.assertTrue(second["violence_detected"])
        self.assertTrue(second["should_alert"])

        third = gate.filter_result(strong_result, sharp_frame)
        self.assertTrue(third["violence_detected"])
        self.assertFalse(third["should_alert"])


class ViolenceDetectorSkipAlertTests(unittest.TestCase):
    def test_skipped_frames_do_not_repeat_alert(self):
        detector = ViolenceDetector(
            DummyModel(1.0),
            {
                "frame_skip": 2,
                "confidence_threshold": 0.55,
            },
        )
        detector._last_result = {
            "violence_detected": True,
            "confidence": 0.8,
            "cnn_score": 0.9,
            "optical_flow_score": 0.8,
            "motion_energy_score": 0.8,
            "fusion_score": 0.85,
            "smoothed_score": 0.8,
            "should_alert": True,
        }
        detector.frame_counter = 0
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        skipped = detector.analyze_frame(frame)
        self.assertFalse(skipped["should_alert"])
        self.assertTrue(detector._last_result["should_alert"])


if __name__ == "__main__":
    unittest.main()
