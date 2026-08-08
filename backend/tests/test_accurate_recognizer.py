import unittest
import subprocess
from unittest.mock import patch, PropertyMock

from PIL import Image, ImageDraw

from accurate_recognizer import (
    AccurateVLMRecognizer,
    AccurateRecognizerBusyError,
    AccurateRecognizerInterruptedError,
    AccurateRecognizerTimeoutError,
)
from assembler import parse_tokens_to_score


class AccurateRecognizerGeometryTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = AccurateVLMRecognizer()

    def test_grouped_underlines_set_note_durations(self):
        detections = [
            (0, 20, 50, 12, 20, .9), (1, 40, 50, 12, 20, .9),
            (2, 60, 50, 12, 20, .9), (32, 90, 50, 2, 30, .9),
            (9, 40, 68, 52, 4, .9),
        ]
        tokens, *_ = self.recognizer._apply_note_modifiers(
            ["P1", "P2", "P3"], detections, [0, 30, 100, 80], 100)
        score = parse_tokens_to_score(tokens)
        self.assertEqual([note["duration"] for note in score["measures"][0]["notes"]],
                         [.5, .5, .5])

    def test_row_tokens_preserve_original_score_line_breaks(self):
        score = parse_tokens_to_score([
            "P1", "P2", "B|", "<ROW>", "P3", "P5", "B|]",
        ])
        self.assertFalse(score["measures"][0].get("lineBreakBefore", False))
        self.assertTrue(score["measures"][1]["lineBreakBefore"])

    def test_second_vlm_request_is_rejected_instead_of_overlapping(self):
        self.recognizer._vlm_lock.acquire()
        try:
            with patch.object(type(self.recognizer), "available",
                              new_callable=PropertyMock, return_value=True):
                with self.assertRaises(AccurateRecognizerBusyError):
                    self.recognizer._run_vlm(Image.new("RGB", (40, 40), "white"), [])
        finally:
            self.recognizer._vlm_lock.release()

    def test_keyboard_interrupt_is_reported_without_raw_traceback(self):
        completed = subprocess.CompletedProcess(
            args=["vlm"], returncode=130, stdout="", stderr="Traceback\nKeyboardInterrupt\n")
        with patch.object(type(self.recognizer), "available",
                          new_callable=PropertyMock, return_value=True), \
                patch("accurate_recognizer.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(
                    AccurateRecognizerInterruptedError, "推理被服务重启或系统中断"):
                self.recognizer._run_vlm(Image.new("RGB", (40, 40), "white"), [])

    def test_vlm_timeout_has_specific_error(self):
        with patch.object(type(self.recognizer), "available",
                          new_callable=PropertyMock, return_value=True), \
                patch("accurate_recognizer.subprocess.run",
                      side_effect=subprocess.TimeoutExpired("vlm", 900)):
            with self.assertRaises(AccurateRecognizerTimeoutError):
                self.recognizer._run_vlm(Image.new("RGB", (40, 40), "white"), [])

    def test_double_underline_wins_over_single(self):
        detections = [
            (0, 30, 50, 12, 20, .9), (32, 90, 50, 2, 30, .9),
            (9, 30, 68, 14, 4, .9), (10, 30, 71, 14, 4, .8),
        ]
        tokens, *_ = self.recognizer._apply_note_modifiers(
            ["P1"], detections, [0, 30, 100, 80], 100)
        score = parse_tokens_to_score(tokens)
        self.assertEqual(score["measures"][0]["notes"][0]["duration"], .25)

    def test_curve_endpoint_pitch_selects_tie_or_slur(self):
        positions = [20.0, 40.0, 60.0]
        tie = self.recognizer._curve_relations(
            ["P1", "P2", "P1"], positions,
            [(22, 40, 30, 48, 12, .9)], 50, 20)
        slur = self.recognizer._curve_relations(
            ["P1", "P2", "P3"], positions,
            [(22, 40, 30, 48, 12, .9)], 50, 20)
        self.assertEqual(tie, [("tie", 0, 2)])
        self.assertEqual(slur, [("slur", 0, 2)])

    def test_vlm_octave_and_dot_modifiers_follow_target_note(self):
        tokens = self.recognizer._apply_vlm_modifiers(
            ["P1", "P2", "P3"],
            [{"note": 0, "octave": 1, "dot": 1},
             {"note": 2, "octave": -1, "dot": 0}],
        )
        self.assertEqual(tokens, ["P1", "^", ".", "P2", "P3", "v"])

    def test_conflicting_octave_dots_prefer_stricter_upper_evidence(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 29, 32, 33), fill="black")
        draw.ellipse((28, 67, 32, 71), fill="black")
        detections = [(0, 30, 50, 14, 20, .9), (32, 90, 50, 2, 30, .9)]
        modifiers = self.recognizer._visual_dot_modifiers(
            image, [0, 20, 100, 80], [30.0], detections[:1])
        self.assertEqual(modifiers[0], ["^"])

    def test_pixel_geometry_recovers_low_octave_dot_near_digit(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 67, 32, 71), fill="black")
        anchors = [(0, 30, 50, 14, 20, .9)]
        modifiers = self.recognizer._visual_dot_modifiers(
            image, [0, 20, 100, 80], [30.0], anchors)
        self.assertEqual(modifiers[0], ["v"])

    def test_pixel_geometry_scales_octave_dot_with_print_size(self):
        image = Image.new("RGB", (140, 140), "white")
        draw = ImageDraw.Draw(image)
        # Ten-pixel scan dot above a roughly forty-pixel printed digit.
        draw.ellipse((45, 31, 54, 40), fill="black")
        draw.text((42, 61), "3", fill="black")
        anchors = [(2, 50, 78, 28, 40, .9)]
        modifiers = self.recognizer._visual_dot_modifiers(
            image, [0, 15, 140, 125], [50.0], anchors)
        self.assertEqual(modifiers[0], ["^"])

    def test_low_confidence_barline_is_kept_on_music_baseline(self):
        detections = [
            (1, 20, 55, 18, 28, .9), (2, 60, 55, 18, 28, .9),
            (32, 85, 45, 3, 48, .14),
            (32, 30, 130, 3, 30, .8),  # lyric/metadata stroke
        ]
        bars = self.recognizer._dedupe_bars(detections, 20, 150)
        self.assertEqual([(item[0], item[1]) for item in bars], [(32, 85)])

    def test_pixel_repeat_direction_comes_from_colon_side(self):
        image = Image.new("RGB", (180, 110), "white")
        draw = ImageDraw.Draw(image)
        # Repeat start: double bar with two dots on its right.
        draw.rectangle((45, 20, 49, 88), fill="black")
        draw.rectangle((57, 20, 59, 88), fill="black")
        draw.ellipse((69, 38, 76, 45), fill="black")
        draw.ellipse((69, 65, 76, 72), fill="black")
        # Repeat end: the same geometry mirrored.
        draw.ellipse((108, 38, 115, 45), fill="black")
        draw.ellipse((108, 65, 115, 72), fill="black")
        draw.rectangle((125, 20, 127, 88), fill="black")
        draw.rectangle((137, 20, 141, 88), fill="black")
        bars = self.recognizer._pixel_bars(image, [0, 0, 180, 110], 55, 28)
        self.assertEqual([token for token, _ in bars], ["B|:", "B:|"])

    def test_pixel_repeat_keeps_high_colon_dot_on_dense_scan(self):
        image = Image.new("RGB", (120, 110), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 25, 72, 99), fill="black")
        draw.rectangle((78, 25, 85, 99), fill="black")
        # The upper dot is 24.6px above a baseline of 77.1px, a common
        # scaled-scan offset for a 21.4px note glyph.
        draw.rectangle((54, 49, 61, 56), fill="black")
        draw.rectangle((54, 72, 61, 79), fill="black")
        bars = self.recognizer._pixel_bars(image, [0, 0, 120, 110], 77.1, 21.4)
        self.assertEqual([token for token, _ in bars], ["B:|"])

    def test_pixel_repeat_survives_baseline_on_reduction_line(self):
        image = Image.new("RGB", (120, 130), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 25, 72, 106), fill="black")
        draw.rectangle((78, 25, 85, 106), fill="black")
        draw.ellipse((54, 49, 61, 56), fill="black")
        draw.ellipse((54, 72, 61, 79), fill="black")
        # The detector baseline is on the lower reduction-line band, not the
        # digit center; the paired-colon geometry should still win.
        bars = self.recognizer._pixel_bars(image, [0, 0, 120, 130], 85.0, 21.4)
        self.assertEqual([token for token, _ in bars], ["B:|"])

    def test_source_boxes_override_interpolated_note_positions(self):
        image = Image.new("RGB", (180, 110), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((140, 17, 146, 23), fill="black")
        detections = [(0, 30, 55, 16, 30, .9), (1, 80, 55, 16, 30, .9)]
        tokens, positions, *_ = self.recognizer._apply_note_modifiers(
            ["P1", "P2"], detections, [0, 0, 180, 110], 180, image,
            [(30, 55, 24, 36), (143, 55, 24, 36)],
        )
        self.assertEqual(positions, [30, 143])
        self.assertEqual(tokens, ["P1", "P2", "^"])

    def test_visual_glyph_band_recovers_when_detector_hits_reduction_line(self):
        image = Image.new("RGB", (520, 150), "white")
        draw = ImageDraw.Draw(image)
        # Three large printed digits; detector anchors intentionally point at
        # the lower reduction-line band rather than the digit centers.
        for left in (80, 220, 360):
            draw.rectangle((left, 30, left + 27, 85), fill="black")
        draw.ellipse((91, 96, 99, 104), fill="black")
        draw.line((220, 100, 390, 100), fill="black", width=4)
        detections = [
            (5, 93, 112, 17, 17, .8),
            (5, 233, 112, 17, 17, .7),
            (5, 373, 112, 17, 17, .7),
        ]
        tokens, positions, _, baseline, note_height = self.recognizer._apply_note_modifiers(
            ["P1", "P2", "P3"], detections, [0, 0, 520, 150], 520, image)
        self.assertEqual([round(value) for value in positions], [94, 234, 374])
        self.assertAlmostEqual(baseline, 58, delta=1)
        self.assertAlmostEqual(note_height, 56, delta=1)
        self.assertEqual(tokens, ["P1", "v", "P2", "_", "P3", "_"])

    def test_pixel_rhythm_reads_two_lines_below_digit(self):
        image = Image.new("RGB", (120, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.line((20, 72, 80, 72), fill="black", width=3)
        draw.line((20, 80, 80, 80), fill="black", width=3)
        depths = self.recognizer._visual_rhythm_depths(
            image, [0, 0, 120, 120], [(50.0, 50.0, 24.0, 30.0)])
        self.assertEqual(depths, {0: 2})

    def test_pixel_rhythm_preserves_third_reduction_line(self):
        image = Image.new("RGB", (120, 120), "white")
        draw = ImageDraw.Draw(image)
        for y in (72, 80, 88):
            draw.line((20, y, 80, y), fill="black", width=3)
        depths = self.recognizer._visual_rhythm_depths(
            image, [0, 0, 120, 120], [(50.0, 50.0, 24.0, 30.0)])
        self.assertEqual(depths, {0: 3})

    def test_hybrid_bands_split_dense_projection_with_detector_rows(self):
        image = Image.new("RGB", (1000, 220), "white")
        draw = ImageDraw.Draw(image)
        # Thin vertical ink bridges make projection segmentation merge rows.
        draw.rectangle((10, 20, 990, 35), fill="black")
        draw.rectangle((10, 95, 990, 110), fill="black")
        draw.rectangle((10, 170, 990, 185), fill="black")
        draw.line((500, 20, 500, 185), fill="black", width=2)
        detections = [
            (1, 100, 28, 14, 20, .9),
            (2, 100, 103, 14, 20, .9),
            (3, 100, 178, 14, 20, .9),
        ]
        bands = self.recognizer._hybrid_bands(image, detections)
        self.assertEqual(len(bands), 3)

    def test_relations_are_written_to_score_notes(self):
        score = parse_tokens_to_score(["P1", "P2", "P1"])
        self.recognizer._decorate_relations(
            score, [("tie", 0, 2), ("slur", 0, 1), ("triplet", 0, 2)])
        notes = score["measures"][0]["notes"]
        self.assertEqual(notes[0]["tieId"], notes[2]["tieId"])
        self.assertEqual(notes[0]["slurId"], notes[1]["slurId"])
        self.assertEqual(notes[0]["tripletId"], notes[2]["tripletId"])

    def test_repeated_three_note_slurs_are_normalized_as_triplets(self):
        relations = [("slur", 0, 2), ("slur", 3, 5), ("slur", 6, 10)]
        self.assertEqual(
            self.recognizer._normalize_relation_types(relations),
            [("triplet", 0, 2), ("triplet", 3, 5), ("slur", 6, 10)],
        )

    def test_production_policy_keeps_geometry_verified_tie_and_slur(self):
        self.assertEqual(
            self.recognizer._production_relations(
                [("slur", 0, 1), ("tie", 2, 3), ("triplet", 4, 6)]),
            [("slur", 0, 1), ("tie", 2, 3), ("triplet", 4, 6)],
        )

    def test_triplet_requires_small_three_above_span(self):
        positions = [20.0, 40.0, 60.0]
        self.assertFalse(self.recognizer._triplet_marker_visible(
            0, 2, positions, [], 60, 20))
        detections = [(2, 40, 32, 7, 11, .4)]
        self.assertTrue(self.recognizer._triplet_marker_visible(
            0, 2, positions, detections, 60, 20))

    def test_pixel_arch_recovers_slur_endpoints(self):
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.arc((20, 15, 80, 48), 180, 360, fill="black", width=3)
        relations = self.recognizer._visual_curve_relations(
            image, [0, 0, 120, 100], ["P1", "P2", "P3"],
            [20.0, 50.0, 80.0], [], 70, 20)
        self.assertEqual(relations, [("slur", 0, 2)])

    def test_pixel_parentheses_attach_to_nearest_inside_notes(self):
        image = Image.new("RGB", (160, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.arc((20, 20, 42, 80), 90, 270, fill="black", width=4)
        draw.arc((118, 20, 140, 80), 270, 90, fill="black", width=4)
        markers = self.recognizer._visual_parentheses(
            image, [0, 0, 160, 100], [55.0, 80.0, 105.0], 50, 30)
        self.assertEqual(markers, [("left", 0), ("right", 2)])

    def test_repeat_ending_span_must_match_two_barlines(self):
        image = Image.new("RGB", (180, 110), "white")
        draw = ImageDraw.Draw(image)
        draw.line((40, 15, 120, 15), fill="black", width=3)
        draw.line((40, 15, 40, 35), fill="black", width=3)
        endings = self.recognizer._visual_repeat_endings(
            image, [0, 0, 180, 110], 70, 20,
            [("B|", 40.0), ("B:|", 120.0)])
        self.assertEqual(endings, [(1, [1])])

    def test_repeat_endings_skip_empty_short_crop(self):
        image = Image.new("RGB", (180, 40), "white")
        endings = self.recognizer._visual_repeat_endings(
            image, [0, 20, 180, 40], 70, 20,
            [("B|", 40.0), ("B:|", 120.0)])
        self.assertEqual(endings, [])

    def test_score_crop_expands_for_upper_slur_context(self):
        image = Image.new("RGB", (700, 180), "white")
        draw = ImageDraw.Draw(image)
        for left in (60, 150, 240, 330, 420, 510, 600):
            draw.rectangle((left, 55, left + 27, 110), fill="black")
        expanded = self.recognizer._expand_score_crop_context(
            image, [0, 105, 700, 180], 7, [])
        self.assertEqual(expanded[0], 0)
        self.assertLess(expanded[1], 105)
        self.assertEqual(expanded[2:], [700, 180])

    def test_repeat_endings_are_injected_inside_target_measure(self):
        tokens = self.recognizer._inject_repeat_endings(
            ["P1", "B|", "P2", "P3", "B:|"], [(1, [2, 3])])
        self.assertEqual(tokens,
                         ["P1", "B|", "R2", "R3", "P2", "P3", "B:|"])

    def test_parentheses_are_written_to_score_notes(self):
        score = parse_tokens_to_score(["P1", "P2", "P3"])
        self.recognizer._decorate_parentheses(
            score, [("left", 0), ("right", 2)])
        notes = score["measures"][0]["notes"]
        self.assertTrue(notes[0]["parenthesisLeft"])
        self.assertTrue(notes[2]["parenthesisRight"])

    def test_text_layers_align_lyrics_and_local_time_signature(self):
        score = parse_tokens_to_score(["P1", "P2", "B|", "P3", "P5"])
        self.recognizer._decorate_text_layers(
            score,
            [(0, [["我", "的", "心", ""], ["你", "的", "梦", ""]])],
            [(2, 2, 4)],
        )
        first = score["measures"][0]["notes"]
        second = score["measures"][1]["notes"]
        self.assertEqual(first[0]["lyrics"], ["我", "你"])
        self.assertEqual(first[1]["lyrics"], ["的", "的"])
        self.assertEqual(second[0]["lyrics"], ["心", "梦"])
        self.assertEqual(score["measures"][1]["timeSignature"],
                         {"numerator": 2, "denominator": 4})

    def test_grace_note_ornament_is_written_to_target_note(self):
        score = parse_tokens_to_score(["P1", "P2", "P3"])
        self.recognizer._decorate_ornaments(score, [
            (1, {"type": "yinyin", "grace_notes": [5, 6]}),
        ])
        notes = score["measures"][0]["notes"]
        self.assertNotIn("techniques", notes[0])
        self.assertEqual(notes[1]["techniques"], [{
            "type": "yinyin", "graceNotes": [5, 6], "graceOctave": 0,
        }])

    def test_pixel_wave_ornament_attaches_to_nearest_note(self):
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.line([(41, 51), (44, 45), (47, 51), (50, 45), (53, 51)],
                  fill="black", width=4)
        self.assertEqual(
            self.recognizer._visual_boyin(
                image, [0, 0, 120, 100], [20.0, 50.0, 80.0], 70, 20),
            [1],
        )

    def test_navigation_mark_is_written_to_target_measure(self):
        score = parse_tokens_to_score(["P1", "P2", "B|", "P3"])
        self.recognizer._decorate_navigation(score, [
            (2, {"type": "ds", "text": "D.S."}),
        ])
        self.assertEqual(score["measures"][1]["navigationMarks"], [{
            "type": "ds", "text": "D.S.",
        }])


if __name__ == "__main__":
    unittest.main()
