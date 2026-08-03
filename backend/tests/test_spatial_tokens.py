import unittest

from model.spatial_tokens import detections_to_tokens


def det(cls_id, x, y, w=12, h=20, conf=0.9):
    return cls_id, x, y, w, h, conf


class SpatialTokenTests(unittest.TestCase):
    def test_modifiers_attach_to_pitch_in_semantic_order(self):
        detections = [
            det(0, 100, 100), det(1, 150, 100),
            det(12, 100, 75, 5, 5),       # upper dot for P1
            det(9, 150, 120, 15, 2),      # underline for P2
            det(14, 139, 100, 7, 15),     # sharp left of P2
            det(11, 162, 100, 5, 5),      # augmentation dot right of P2
        ]
        self.assertEqual(
            detections_to_tokens(detections),
            ["<BOS>", "P1", "^", "P2", "#", ".", "_", "<EOS>"],
        )

    def test_rows_are_defined_by_digits_not_modifier_height(self):
        detections = [
            det(0, 40, 100), det(1, 80, 100), det(12, 40, 72, 5, 5),
            det(2, 40, 210), det(3, 80, 210), det(13, 80, 238, 5, 5),
            det(32, 110, 100, 2, 40), det(34, 110, 210, 4, 40),
        ]
        self.assertEqual(
            detections_to_tokens(detections),
            ["<BOS>", "P1", "^", "P2", "B|", "<ROW>", "P3", "P4", "v", "B|]", "<EOS>"],
        )

    def test_no_pitch_returns_empty_sequence(self):
        self.assertEqual(detections_to_tokens([det(11, 10, 10, 4, 4)]), ["<BOS>", "<EOS>"])

    def test_grouped_underline_applies_to_every_covered_note(self):
        detections = [
            det(0, 50, 100), det(1, 90, 100), det(2, 130, 100),
            det(9, 90, 120, 100, 2),
        ]
        self.assertEqual(
            detections_to_tokens(detections),
            ["<BOS>", "P1", "_", "P2", "_", "P3", "_", "<EOS>"],
        )


if __name__ == "__main__":
    unittest.main()
