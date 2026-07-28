"""Authentification simple et gestion du plan (free/pro) pour le MVP freemium.

Stockage SQLite local (pas de vrai paiement pour le MVP) : les demandes de
passage au plan "pro" sont enregistrées dans `pro_requests` et activées
manuellement (édition du champ `plan` dans la table `users`).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "app.db"

FREE_DAILY_QUOTA = 5
FREE_AUDIENCES = {"CITOYEN"}


@dataclass
class User:
    email: str
    plan: str  # "free" ou "pro"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free'
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS usage (
            email TEXT NOT NULL,
            day TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (email, day)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pro_requests (
            email TEXT NOT NULL,
            phone TEXT,
            requested_at TEXT NOT NULL
        )"""
    )
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def create_user(email: str, password: str) -> User:
    email = email.strip().lower()
    salt = os.urandom(16)
    password_hash = _hash_password(password, salt)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (email, salt, password_hash, plan) VALUES (?, ?, ?, 'free')",
            (email, salt.hex(), password_hash),
        )
    return User(email=email, plan="free")


def user_exists(email: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return row is not None


def verify_user(email: str, password: str) -> User | None:
    email = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT salt, password_hash, plan FROM users WHERE email = ?", (email,)
        ).fetchone()
    if row is None:
        return None
    salt_hex, password_hash, plan = row
    if _hash_password(password, bytes.fromhex(salt_hex)) != password_hash:
        return None
    return User(email=email, plan=plan)


def get_today_usage(email: str) -> int:
    today = date.today().isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE email = ? AND day = ?", (email, today)
        ).fetchone()
    return row[0] if row else 0


def increment_usage(email: str) -> int:
    today = date.today().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO usage (email, day, count) VALUES (?, ?, 1)
               ON CONFLICT(email, day) DO UPDATE SET count = count + 1""",
            (email, today),
        )
        row = conn.execute(
            "SELECT count FROM usage WHERE email = ? AND day = ?", (email, today)
        ).fetchone()
    return row[0]


def request_pro(email: str, phone: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO pro_requests (email, phone, requested_at) VALUES (?, ?, datetime('now'))",
            (email.strip().lower(), phone.strip()),
        )


def list_pro_requests() -> list[dict]:
    """Demandes de passage Pro les plus récentes en premier, avec le plan actuel de chaque compte."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT pro_requests.email, phone, requested_at, users.plan
               FROM pro_requests
               LEFT JOIN users ON users.email = pro_requests.email
               ORDER BY requested_at DESC"""
        ).fetchall()
    return [
        {"email": email, "phone": phone, "requested_at": requested_at, "plan": plan or "?"}
        for email, phone, requested_at, plan in rows
    ]


def set_plan(email: str, plan: str) -> bool:
    """Active/révoque le plan Pro pour un compte. Retourne False si l'email est inconnu."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET plan = ? WHERE email = ?", (plan, email.strip().lower())
        )
        return cur.rowcount > 0
