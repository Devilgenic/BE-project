import sqlite3
import threading
from datetime import datetime, timezone


class DetectionDatabase:
    def __init__(self, db_path="detection_history.db"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    cnn_score REAL NOT NULL,
                    optical_flow_score REAL NOT NULL,
                    motion_energy_score REAL NOT NULL,
                    frame_path TEXT,
                    source TEXT NOT NULL,
                    alert_sent INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._conn.commit()

    def add_event(self, timestamp, confidence, cnn_score, optical_flow_score,
                  motion_energy_score, frame_path, source, alert_sent):
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO detection_events
                   (timestamp, confidence, cnn_score, optical_flow_score,
                    motion_energy_score, frame_path, source, alert_sent)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, confidence, cnn_score, optical_flow_score,
                 motion_energy_score, frame_path, source, alert_sent)
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_events(self, limit=50, offset=0):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM detection_events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_event_count(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM detection_events"
            ).fetchone()
            return row["cnt"]

    def get_stats(self):
        with self._lock:
            row = self._conn.execute("""
                SELECT
                    COUNT(*) as total_events,
                    COALESCE(SUM(alert_sent), 0) as total_alerts,
                    COALESCE(ROUND(AVG(confidence), 4), 0) as avg_confidence
                FROM detection_events
            """).fetchone()

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM detection_events WHERE timestamp LIKE ?",
                (today + "%",)
            ).fetchone()

            return {
                "total_events": row["total_events"],
                "total_alerts": row["total_alerts"],
                "avg_confidence": row["avg_confidence"],
                "events_today": today_row["cnt"]
            }

    def clear_events(self):
        with self._lock:
            self._conn.execute("DELETE FROM detection_events")
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
