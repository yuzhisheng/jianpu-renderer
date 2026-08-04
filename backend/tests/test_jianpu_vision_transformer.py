import unittest

import torch

from model.jianpu_event_vocabulary import (
    BRANCHES, IGNORE_ID, events_to_targets, events_to_tokens, teacher_inputs,
    tokens_to_events,
)
from model.jianpu_vision_transformer import JianpuVisionTransformer, VisionTransformerConfig
from visual_recognizer import score_row_bands


class JianpuVisionTransformerTests(unittest.TestCase):
    def test_flat_tokens_round_trip_through_event_branches(self):
        tokens = ["P1", "#", "P2", "_", "R0", "-", "B||"]
        self.assertEqual(events_to_tokens(tokens_to_events(tokens)), tokens)

    def test_silver_skeleton_masks_unobserved_modifier_heads(self):
        targets = events_to_targets(
            tokens_to_events(["P1", "B|", "P2"]), skeleton_label=True,
        )
        self.assertNotEqual(targets["kind"][0], IGNORE_ID)
        self.assertEqual(targets["octave"][1], IGNORE_ID)
        self.assertEqual(targets["duration"][1], IGNORE_ID)
        inputs = teacher_inputs(targets)
        self.assertTrue(all(value >= 0 for values in inputs.values() for value in values))

    def test_visual_transformer_shapes_and_generation(self):
        config = VisionTransformerConfig(
            image_height=32, max_width=64, d_model=48, nhead=4,
            decoder_layers=1, dim_feedforward=96, max_seq_len=8,
        )
        model = JianpuVisionTransformer(config).eval()
        images = torch.zeros(2, 1, 32, 64)
        targets = events_to_targets(
            tokens_to_events(["P1", "B|", "P2"]), skeleton_label=True,
        )
        inputs = teacher_inputs(targets)
        tensor_inputs = {
            branch: torch.tensor([values[:-1], values[:-1]])
            for branch, values in inputs.items()
        }
        output = model(images, tensor_inputs)
        self.assertEqual(set(output), set(BRANCHES))
        self.assertEqual(output["kind"].shape[:2], tensor_inputs["kind"].shape)
        generated = model.generate(images, max_len=4)
        self.assertEqual(len(generated), 2)
        ctc = model.ctc_logits(images)
        self.assertEqual(ctc.shape[:2], (2, 4))
        self.assertEqual(len(model.generate_ctc(images)), 2)

    def test_score_row_bands_are_ordered_and_non_overlapping(self):
        detections = [
            (0, 20, 50, 10, 14, .9), (1, 40, 52, 10, 14, .9),
            (2, 20, 180, 10, 14, .9), (3, 40, 178, 10, 14, .9),
        ]
        bands = score_row_bands(detections, 260)
        self.assertEqual(len(bands), 2)
        self.assertLessEqual(bands[0][1], bands[1][0])


if __name__ == "__main__":
    unittest.main()
