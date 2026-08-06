import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image

import main
from accurate_recognizer import (
    AccurateRecognizerBusyError,
    AccurateRecognizerInterruptedError,
    AccurateRecognizerTimeoutError,
)


class _FakeDetector:
    def detect(self, image, conf_threshold):
        return [], image.width, image.height


class _StaffDetector(_FakeDetector):
    @staticmethod
    def is_staff_notation(image):
        return True


class _FailingRecognizer:
    def __init__(self, error):
        self.error = error

    def recognize(self, image, detections):
        raise self.error


class RecognizeErrorMappingTests(unittest.TestCase):
    @staticmethod
    def _upload():
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
        buffer.seek(0)
        return UploadFile(filename="test.png", file=buffer)

    def _request(self, error):
        with patch("main.get_models", return_value=(_FakeDetector(), None)), \
                patch("main.get_accurate_recognizer",
                      return_value=_FailingRecognizer(error)):
            return asyncio.run(main.recognize(
                file=self._upload(), conf=.12, use_transformer=False,
                visual_sequence=False, recognizer="accurate"))

    def test_busy_task_maps_to_conflict(self):
        with self.assertRaises(HTTPException) as caught:
            self._request(AccurateRecognizerBusyError("已有一个精确识别任务正在运行"))
        self.assertEqual(caught.exception.status_code, 409)

    def test_interrupted_task_maps_to_service_unavailable(self):
        with self.assertRaises(HTTPException) as caught:
            self._request(AccurateRecognizerInterruptedError("推理被中断"))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "推理被中断")

    def test_timeout_maps_to_gateway_timeout(self):
        with self.assertRaises(HTTPException) as caught:
            self._request(AccurateRecognizerTimeoutError("推理超时"))
        self.assertEqual(caught.exception.status_code, 504)

    def test_staff_notation_is_rejected_before_vlm(self):
        with patch("main.get_models", return_value=(_StaffDetector(), None)):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(main.recognize(
                    file=self._upload(), conf=.12, use_transformer=False,
                    visual_sequence=False, recognizer="accurate"))
        self.assertEqual(caught.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
