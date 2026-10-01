"""Local profile database (§18, §35): SQLite, WAL, schema_version migrations.

Everything lives on the user's PC (privacy §8/§78). We store 512-d float
embeddings as BLOBs — never raw face images. Full data deletion is a first
class operation (§55).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("machine.memory.database")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    quality     REAL NOT NULL DEFAULT 0.0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_profile ON embeddings(profile_id);
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    priority    TEXT NOT NULL,
    ts          TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Profile:
    id: int
    name: str
    sample_count: int


class Database:
    """Thread-safe facade (single connection + lock; volume is tiny)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- lifecycle
    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            has_ver = cur.fetchone() is not None
            if not has_ver:
                self._conn.executescript(_SCHEMA)
                self._conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
                self._conn.commit()
                logger.info("database initialized at %s (v%d)", self._path, SCHEMA_VERSION)
                return
            ver = self._conn.execute(
                "SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            if ver < SCHEMA_VERSION:
                # future migrations land here, bumping schema_version (§98)
                self._conn.executescript(_SCHEMA)
                self._conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
                self._conn.commit()
                logger.info("database migrated v%d -> v%d", ver, SCHEMA_VERSION)

    # -------------------------------------------------------------- profiles
    def create_profile(self, name: str) -> int:
        clean = " ".join(name.split())[:60]
        if not clean:
            raise ValueError("profile name required")
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO profiles(name, created_at, updated_at) VALUES (?,?,?)",
                    (clean, _now(), _now()))
            except sqlite3.IntegrityError:
                raise ValueError(f"profile '{clean}' already exists") from None
            self._conn.commit()
            return int(cur.lastrowid)

    def add_embedding(self, profile_id: int, vector_bytes: bytes,
                      quality: float = 0.0) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO embeddings(profile_id, vector, quality, created_at)"
                " VALUES (?,?,?,?)", (profile_id, vector_bytes, quality, _now()))
            self._conn.execute(
                "UPDATE profiles SET updated_at=? WHERE id=?", (_now(), profile_id))
            self._conn.commit()

    def list_profiles(self) -> list[Profile]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.id, p.name, COUNT(e.id) FROM profiles p"
                " LEFT JOIN embeddings e ON e.profile_id=p.id"
                " GROUP BY p.id ORDER BY p.name").fetchall()
        return [Profile(int(r[0]), str(r[1]), int(r[2])) for r in rows]

    def all_embeddings(self) -> list[tuple[int, str, bytes]]:
        """(embedding_id, profile_name, blob) for every stored sample."""
        with self._lock:
            return [(int(r[0]), str(r[1]), bytes(r[2])) for r in self._conn.execute(
                "SELECT e.id, p.name, e.vector FROM embeddings e"
                " JOIN profiles p ON p.id=e.profile_id")]

    def delete_profile(self, profile_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embeddings WHERE profile_id=?", (profile_id,))
            self._conn.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
            self._conn.commit()

    # -------------------------------------------------------------- memories
    def add_memory(self, text: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO memories(text, created_at) VALUES (?,?)",
                               (text[:400], _now()))
            self._conn.commit()

    def list_memories(self, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT text FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [str(r[0]) for r in reversed(rows)]

    def clear_memories(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()

    # ---------------------------------------------------------------- events
    def log_event(self, name: str, payload_json: str, priority: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(name, payload, priority, ts) VALUES (?,?,?,?)",
                (name, payload_json[:2000], priority, _now()))
            # bounded retention (§53): keep last 5000
            self._conn.execute(
                "DELETE FROM events WHERE id NOT IN"
                " (SELECT id FROM events ORDER BY id DESC LIMIT 5000)")
            self._conn.commit()

    def recent_events(self, limit: int = 20) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, name, payload FROM events ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return list(reversed(rows))

    # ------------------------------------------------------------ destruction
    def delete_all_data(self) -> None:
        """§55 — wipe profiles, embeddings, memories, events (settings kept)."""
        with self._lock:
            for table in ("embeddings", "profiles", "memories", "events"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()
        logger.info("ALL USER DATA DELETED")
