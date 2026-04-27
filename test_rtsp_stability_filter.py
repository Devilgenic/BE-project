import unittest

import cv2
import numpy as np

from detection_engine import RTSPStreamStabilityFilter


def make_base_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (120, 80), (520, 380), (170, 170, 170), -1)
    cv2.circle(frame, (320, 240), 60, (0, 0, 255), -1)
    return frame


class RTSPStreamStabilityFilterTests(unittest.TestCase):
    def test_blocks_startup_and_large_camera_shift(self):
        frame = make_base_frame()
        shifted = np.roll(frame, 30, axis=1)
        stability_filter = RTSPStreamStabilityFilter(warmup_frames=2)

        self.assertFalse(stability_filter.allow_frame(frame))
        self.assertEqual(stability_filter.last_reason, "bootstrap")

        self.assertFalse(stability_filter.allow_frame(frame))
        self.assertEqual(stability_filter.last_reason, "warmup")

        self.assertFalse(stability_filter.allow_frame(frame))
        self.assertEqual(stability_filter.last_reason, "warmup")

        self.assertTrue(stability_filter.allow_frame(frame))
        self.assertEqual(stability_filter.last_reason, "stable")

        self.assertFalse(stability_filter.allow_frame(shifted))
        self.assertEqual(stability_filter.last_reason, "global_motion")
        self.assertGreater(stability_filter.last_metrics["shift_magnitude"], 6.0)

        self.assertFalse(stability_filter.allow_frame(shifted))
        self.assertEqual(stability_filter.last_reason, "warmup")

        self.assertFalse(stability_filter.allow_frame(shifted))
        self.assertEqual(stability_filter.last_reason, "warmup")

        self.assertTrue(stability_filter.allow_frame(shifted))
        self.assertEqual(stability_filter.last_reason, "stable")

    def test_allows_local_subject_motion_after_warmup(self):
        frame = make_base_frame()
        local_action = frame.copy()
        cv2.rectangle(local_action, (250, 180), (390, 320), (255, 255, 255), -1)
        stability_filter = RTSPStreamStabilityFilter(warmup_frames=1)

        self.assertFalse(stability_filter.allow_frame(frame))
        self.assertFalse(stability_filter.allow_frame(frame))
        self.assertTrue(stability_filter.allow_frame(frame))

        self.assertTrue(stability_filter.allow_frame(local_action))
        self.assertEqual(stability_filter.last_reason, "stable")
        self.assertLess(stability_filter.last_metrics["shift_magnitude"], 6.0)


if __name__ == "__main__":
    unittest.main()
