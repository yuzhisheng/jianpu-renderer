"""Durable local history for image-recognition requests.

The recognizer is intentionally local, so a small SQLite index plus the
original uploaded image is enough to make every result inspectable without
introducing a separate database service.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "recognition_history"
_SAFE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_suffix(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in _SAFE_SUFFIXES else ".bin"


def _safe_filename(filename: Optional[str]) -> str:
    value = Path(filename or "未命名图片").name
    value = re.sub(r"[^\w\-.\u4e00-\u9fff ]+", "_", value).strip()
    return value[:160] or "未命名图片"


class RecognitionHistory:
    """SQLite-backed recognition records with one image per successful upload."""

    def __init__(self, root: Optional[Path] = None):
        configured = os.environ.get("JIANPU_HISTORY_DIR")
        self.root = root or (Path(configured).expanduser() if configured else DEFAULT_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "history.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    image_filename TEXT,
                    image_sha256 TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    recognizer TEXT,
                    confidence REAL,
                    inference_ms REAL,
                    error TEXT,
                    response_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_runs_created "
                "ON recognition_runs(created_at DESC)"
            )

    def begin(
        self,
        filename: Optional[str],
        contents: bytes,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        recognizer: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        record_id = uuid.uuid4().hex
        now = _utc_now()
        image_filename: Optional[str] = None
        digest: Optional[str] = None
        if contents:
            digest = hashlib.sha256(contents).hexdigest()
            image_filename = f"{record_id}{_safe_suffix(filename)}"
            (self.images / image_filename).write_bytes(contents)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recognition_runs (
                    id, created_at, updated_at, status, original_filename,
                    image_filename, image_sha256, image_width, image_height,
                    recognizer, confidence
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id, now, now, _safe_filename(filename), image_filename,
                    digest, width, height, recognizer, confidence,
                ),
            )
        return record_id

    def complete(self, record_id: Optional[str], response: dict[str, Any], inference_ms: float) -> None:
        if not record_id:
            return
        now = _utc_now()
        payload = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                "UPDATE recognition_runs SET updated_at=?, status='succeeded', "
                "inference_ms=?, error=NULL, response_json=? WHERE id=?",
                (now, float(inference_ms), payload, record_id),
            )

    def set_dimensions(self, record_id: Optional[str], width: int, height: int) -> None:
        if not record_id:
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE recognition_runs SET image_width=?, image_height=?, updated_at=? WHERE id=?",
                (int(width), int(height), _utc_now(), record_id),
            )

    def fail(self, record_id: Optional[str], error: str, inference_ms: float) -> None:
        if not record_id:
            return
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE recognition_runs SET updated_at=?, status='failed', "
                "inference_ms=?, error=? WHERE id=?",
                (now, float(inference_ms), str(error)[:4000], record_id),
            )

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        result = {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "original_filename": row["original_filename"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "recognizer": row["recognizer"],
            "confidence": row["confidence"],
            "inference_ms": row["inference_ms"],
            "error": row["error"],
        }
        if row["response_json"]:
            try:
                response = json.loads(row["response_json"])
                score = response.get("score") if isinstance(response, dict) else None
                symbols = response.get("symbol_summary") if isinstance(response, dict) else None
                if isinstance(score, dict) and isinstance(score.get("title"), str):
                    result["title"] = score["title"]
                if isinstance(symbols, dict):
                    result["notes"] = symbols.get("notes")
                    result["lyric_syllables"] = symbols.get("lyric_syllables")
            except (TypeError, json.JSONDecodeError):
                pass
        return result

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        limit = min(max(int(limit), 1), 200)
        offset = max(int(offset), 0)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recognition_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def get(self, record_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM recognition_runs WHERE id=?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        result = self._summary(row)
        result["image_url"] = f"/recognition-history/{record_id}/image" if row["image_filename"] else None
        try:
            result["response"] = json.loads(row["response_json"]) if row["response_json"] else None
        except json.JSONDecodeError:
            result["response"] = None
        return result

    def image_path(self, record_id: str) -> Optional[Path]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_filename FROM recognition_runs WHERE id=?", (record_id,)
            ).fetchone()
        if row is None or not row["image_filename"]:
            return None
        path = (self.images / row["image_filename"]).resolve()
        if path.parent != self.images.resolve() or not path.is_file():
            return None
        return path
