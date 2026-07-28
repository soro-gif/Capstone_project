"""Bibliothèque locale : sauvegarde des résumés/interprétations générés par l'utilisateur."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "library.db"


@dataclass
class LibraryEntry:
    id: int
    created_at: str
    kind: str  # "Résumé" ou "Interprétation"
    audience: str
    title: str
    content: str
    source_excerpt: str


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            kind TEXT NOT NULL,
            audience TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source_excerpt TEXT NOT NULL
        )"""
    )
    return conn


def save_entry(kind: str, audience: str, content: str, source_text: str) -> int:
    title = " ".join(source_text.split())[:80] or f"{kind} {audience.capitalize()}"
    excerpt = " ".join(source_text.split())[:300]
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO entries (kind, audience, title, content, source_excerpt) VALUES (?, ?, ?, ?, ?)",
                (kind, audience, title, content, excerpt),
            )
            return cur.lastrowid
    finally:
        conn.close()


def list_entries() -> list[LibraryEntry]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, created_at, kind, audience, title, content, source_excerpt "
            "FROM entries ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [LibraryEntry(*row) for row in rows]


def delete_entry(entry_id: int) -> None:
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    finally:
        conn.close()
