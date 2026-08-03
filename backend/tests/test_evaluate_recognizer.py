import unittest

from scripts.evaluate_recognizer import edit_distance, full_key, pitch_key, similarity


class RecognizerMetricTests(unittest.TestCase):
    def test_edit_similarity(self):
        self.assertEqual(edit_distance([1, 2, 3], [1, 4, 3]), 1)
        self.assertAlmostEqual(similarity([1, 2, 3], [1, 4, 3]), 2 / 3)

    def test_keys_measure_different_semantics(self):
        base = {"pitch": 5, "octave": 1, "duration": 0.5}
        changed = {"pitch": 5, "octave": 1, "duration": 0.25}
        self.assertEqual(pitch_key(base), pitch_key(changed))
        self.assertNotEqual(full_key(base), full_key(changed))


if __name__ == "__main__":
    unittest.main()
