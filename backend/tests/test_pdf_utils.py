import unittest

import fitz

from main import merge_page_scores
from pdf_utils import is_pdf, render_pdf_pages


class PdfUtilsTests(unittest.TestCase):
    def test_pdf_pages_are_rasterized_in_order(self):
        document = fitz.open()
        document.new_page(width=300, height=400)
        document.new_page(width=300, height=400)
        raw = document.tobytes()
        document.close()
        self.assertTrue(is_pdf(raw, "score.pdf"))
        pages = render_pdf_pages(raw, dpi=72)
        self.assertEqual(len(pages), 2)
        self.assertEqual([page.size for page in pages], [(300, 400), (300, 400)])

    def test_page_scores_are_joined_with_a_line_break(self):
        merged = merge_page_scores([
            {"title": "测试", "measures": [{"notes": [{"pitch": 1}]}]},
            {"title": "测试", "measures": [{"notes": [{"pitch": 2}]}]},
        ])
        self.assertEqual(len(merged["measures"]), 2)
        self.assertTrue(merged["measures"][1]["lineBreakBefore"])


if __name__ == "__main__":
    unittest.main()
