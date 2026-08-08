import tempfile
import unittest
from pathlib import Path

from recognition_history import RecognitionHistory


class RecognitionHistoryTests(unittest.TestCase):
    def test_round_trip_keeps_result_and_uploaded_image(self):
        with tempfile.TemporaryDirectory() as directory:
            history = RecognitionHistory(Path(directory) / "history")
            record_id = history.begin(
                "示例.png", b"png-bytes", width=32, height=24,
                recognizer="fast", confidence=.2,
            )
            history.complete(
                record_id,
                {"recognition_id": record_id, "score": {"title": "示例"}},
                123.4,
            )
            item = history.get(record_id)
            self.assertIsNotNone(item)
            self.assertEqual(item["status"], "succeeded")
            self.assertEqual(item["image_width"], 32)
            self.assertEqual(item["response"]["score"]["title"], "示例")
            self.assertEqual(history.image_path(record_id).read_bytes(), b"png-bytes")
            self.assertEqual(history.list(limit=1)[0]["id"], record_id)

    def test_failed_request_is_visible_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history = RecognitionHistory(Path(directory) / "history")
            record_id = history.begin("bad.png", b"bad")
            history.fail(record_id, "模型超时", 900001)
            item = history.get(record_id)
            self.assertEqual(item["status"], "failed")
            self.assertEqual(item["error"], "模型超时")


if __name__ == "__main__":
    unittest.main()
