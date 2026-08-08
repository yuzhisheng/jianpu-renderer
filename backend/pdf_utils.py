"""PDF page rasterization for the recognition pipeline."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

try:  # PyMuPDF is the normal path; pdftoppm remains a useful local fallback.
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - exercised only on minimal installs
    fitz = None


MAX_PDF_PAGES = 64
DEFAULT_DPI = 160
MAX_RENDER_PIXELS = 3200


def is_pdf(contents: bytes, filename: str | None = None) -> bool:
    return (str(filename or "").lower().endswith(".pdf")
            or contents[:5] == b"%PDF-")


def _fitz_pages(contents: bytes, dpi: int) -> List[Image.Image]:
    document = fitz.open(stream=contents, filetype="pdf")
    try:
        if document.page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF 页数超过 {MAX_PDF_PAGES} 页")
        pages: List[Image.Image] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            scale = dpi / 72.0
            # Avoid allocating enormous bitmaps for posters or malformed PDFs.
            longest = max(float(page.rect.width), float(page.rect.height)) * scale
            if longest > MAX_RENDER_PIXELS:
                scale *= MAX_RENDER_PIXELS / longest
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale), alpha=False,
                colorspace=fitz.csRGB,
            )
            pages.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
        return pages
    finally:
        document.close()


def _pdftoppm_pages(contents: bytes, dpi: int) -> List[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="jianpu-pdf-") as directory:
        root = Path(directory)
        source = root / "source.pdf"
        prefix = root / "page"
        source.write_bytes(contents)
        try:
            subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-png", str(source), str(prefix)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("当前环境缺少 PDF 渲染器，请安装 PyMuPDF") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"PDF 渲染失败: {(exc.stderr or '').strip()[-500:]}") from exc
        paths = sorted(root.glob("page-*.png"))
        if len(paths) > MAX_PDF_PAGES:
            raise ValueError(f"PDF 页数超过 {MAX_PDF_PAGES} 页")
        if not paths:
            raise ValueError("PDF 没有可渲染的页面")
        return [Image.open(path).convert("RGB") for path in paths]


def render_pdf_pages(contents: bytes, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    """Rasterize every PDF page into an RGB PIL image for the existing recognizers."""
    if fitz is not None:
        return _fitz_pages(contents, dpi)
    return _pdftoppm_pages(contents, dpi)
