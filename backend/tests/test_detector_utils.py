import unittest

import numpy as np

from detector import YoloDetector


class DetectorUtilityTests(unittest.TestCase):
    def test_tiles_cover_edges(self):
        windows = YoloDetector._tile_windows(2500, 1800, 1280, 0.18)
        self.assertIn((1220, 520, 2500, 1800), windows)
        self.assertEqual(min(w[0] for w in windows), 0)
        self.assertEqual(min(w[1] for w in windows), 0)

    def test_nms_keeps_nonexclusive_overlapping_classes(self):
        dets = [
            (0, 100, 100, 20, 20, 0.9),
            (0, 101, 101, 20, 20, 0.8),
            (11, 101, 101, 20, 20, 0.7),
        ]
        kept = YoloDetector._class_aware_nms(dets)
        self.assertEqual(len(kept), 2)
        self.assertEqual({d[0] for d in kept}, {0, 11})

    def test_nms_suppresses_competing_pitch_classes(self):
        dets = [
            (0, 100, 100, 20, 20, 0.8),
            (4, 100, 100, 20, 20, 0.9),
        ]
        kept = YoloDetector._class_aware_nms(dets)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], 4)

    def test_nms_suppresses_competing_barline_classes(self):
        dets = [
            (32, 100, 100, 8, 50, 0.7),
            (34, 100, 100, 8, 50, 0.85),
        ]
        kept = YoloDetector._class_aware_nms(dets)
        self.assertEqual([item[0] for item in kept], [34])

    def test_score_row_gate_removes_isolated_ocr_candidates(self):
        dets = [
            (index % 7, 40 + index * 30, 100, 12, 18, 0.9)
            for index in range(6)
        ]
        dets.extend([
            (32, 230, 100, 4, 38, 0.9),
            (9, 100, 116, 40, 3, 0.9),
            (5, 80, 200, 12, 18, 0.8),
            (40, 80, 225, 16, 18, 0.9),
        ])
        kept = YoloDetector._filter_score_rows(dets)
        self.assertEqual(sum(0 <= d[0] <= 8 for d in kept), 6)
        self.assertNotIn((5, 80, 200, 12, 18, 0.8), kept)
        self.assertFalse(any(d[0] == 40 for d in kept))

    def test_score_row_gate_keeps_sparse_excerpt_when_uncertain(self):
        dets = [(0, 30, 50, 12, 18, 0.9), (1, 60, 50, 12, 18, 0.9)]
        self.assertEqual(YoloDetector._filter_score_rows(dets), dets)

    def test_adaptive_retry_only_targets_sparse_full_pages(self):
        sparse = [(index % 7, 30 + index * 20, 100, 12, 18, 0.9)
                  for index in range(12)]
        self.assertTrue(YoloDetector._should_retry_full_page(sparse, 1200, 1800))
        self.assertFalse(YoloDetector._should_retry_full_page(sparse, 800, 300))

    def test_adaptive_retry_requires_supported_gain(self):
        baseline = [(index % 7, 30 + index * 20, 100, 12, 18, 0.9)
                    for index in range(10)]
        retry = [(index % 7, 30 + (index % 10) * 20,
                  100 + (index // 10) * 80, 12, 18, 0.6)
                 for index in range(30)]
        retry.extend([(32, 240, 100 + row * 80, 4, 35, 0.6)
                      for row in range(3)])
        self.assertTrue(YoloDetector._prefer_retry(baseline, retry))

    def test_staff_gate_requires_repeated_five_line_systems(self):
        page = np.full((700, 1000), 255, dtype=np.uint8)
        for top in (100, 300):
            for offset in range(0, 35, 8):
                page[top + offset, 100:900] = 0
        self.assertEqual(YoloDetector._staff_line_group_count(page), 2)

    def test_staff_gate_ignores_short_numbered_notation_underlines(self):
        page = np.full((700, 1000), 255, dtype=np.uint8)
        for row in (100, 180, 260, 340, 420, 500):
            page[row, 100:300] = 0
            page[row, 500:700] = 0
        self.assertEqual(YoloDetector._staff_line_group_count(page), 0)


if __name__ == "__main__":
    unittest.main()
