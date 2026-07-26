"""
database.py
-----------
Lightweight SQLite logging for the Bean Leaf Disease Classifier API.

Tracks uploads (for retraining), predictions, and retrain events so the
UI monitoring dashboard can show uptime-related stats without needing
an external database.
"""

import os
import sqlite3
import time
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "app_data.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not already exist."""
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                num_new_images INTEGER NOT NULL,
                epochs INTEGER NOT NULL,
                final_val_accuracy REAL,
                model_version TEXT,
                timestamp REAL NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_upload(class_name: str, filename: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO uploads (class_name, filename, timestamp) VALUES (?, ?, ?)",
            (class_name, filename, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def log_prediction(predicted_class: str, confidence: float) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO predictions (predicted_class, confidence, timestamp) VALUES (?, ?, ?)",
            (predicted_class, float(confidence), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def log_retrain_event(
    num_new_images: int,
    epochs: int,
    final_val_accuracy: float | None,
    model_version: str,
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO retrain_events
                (num_new_images, epochs, final_val_accuracy, model_version, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(num_new_images),
                int(epochs),
                final_val_accuracy,
                model_version,
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_prediction_stats() -> dict[str, Any]:
    """
    Returns:
      {
        "total": int,
        "by_class": {class_name: count, ...},
        "recent": [{predicted_class, confidence, timestamp}, ...]  # newest first
      }
    """
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]

        by_class_rows = conn.execute(
            """
            SELECT predicted_class, COUNT(*) AS n
            FROM predictions
            GROUP BY predicted_class
            """
        ).fetchall()
        by_class = {row["predicted_class"]: row["n"] for row in by_class_rows}

        recent_rows = conn.execute(
            """
            SELECT predicted_class, confidence, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 20
            """
        ).fetchall()
        recent = [
            {
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "timestamp": row["timestamp"],
            }
            for row in recent_rows
        ]

        return {"total": total, "by_class": by_class, "recent": recent}
    finally:
        conn.close()


def get_upload_class_distribution() -> dict[str, int]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT class_name, COUNT(*) AS n
            FROM uploads
            GROUP BY class_name
            """
        ).fetchall()
        return {row["class_name"]: row["n"] for row in rows}
    finally:
        conn.close()


def get_retrain_history() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT num_new_images, epochs, final_val_accuracy, model_version, timestamp
            FROM retrain_events
            ORDER BY timestamp DESC
            LIMIT 50
            """
        ).fetchall()
        return [
            {
                "num_new_images": row["num_new_images"],
                "epochs": row["epochs"],
                "final_val_accuracy": row["final_val_accuracy"],
                "model_version": row["model_version"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]
    finally:
        conn.close()
