"""Content-addressed on-disk cache of Jev answers (SQLite, stdlib only).

Key = sha256 of (model, state, questions). Re-running an ingest or an
evaluation only pays for inputs that changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def cache_key(model: str, state: Any, questions_wire: Dict[str, Any]) -> str:
    blob = json.dumps({"m": model, "s": state, "q": questions_wire}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, directory: Optional[str]) -> None:
        self.enabled = bool(directory)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if self.enabled:
            path = Path(os.path.expanduser(str(directory)))
            path.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path / "jev_answers.sqlite"), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL)"
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute("SELECT body FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, body: Dict[str, Any]) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers (key, body, created) VALUES (?, ?, ?)",
                (key, json.dumps(body, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
