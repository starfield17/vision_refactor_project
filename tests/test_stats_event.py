from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from share.kernel.statistics.sqlite_store import get_recent_events, init_stats_db, insert_stats_event
from share.types.stats import StatsEvent


class StatsEventTests(unittest.TestCase):
    def test_stats_event_roundtrip_with_metadata(self) -> None:
        event = StatsEvent.now(
            source_id="edge-001",
            total_detections=2,
            counts_by_class={"person": 2},
            latency_ms=15.5,
            request_id="req-1",
            run_id="run-1",
            model_id="model-1",
            backend="yolo",
            transport_mode="edge-local",
        )
        restored = StatsEvent.from_dict(event.to_dict())

        self.assertEqual(restored.request_id, "req-1")
        self.assertEqual(restored.run_id, "run-1")
        self.assertEqual(restored.model_id, "model-1")
        self.assertEqual(restored.backend, "yolo")
        self.assertEqual(restored.transport_mode, "edge-local")

    def test_sqlite_store_persists_extended_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "stats.db"
            init_stats_db(db_path)
            event = StatsEvent.now(
                source_id="edge-001",
                total_detections=1,
                counts_by_class={"person": 1},
                latency_ms=9.5,
                request_id="req-2",
                run_id="run-2",
                model_id="model-2",
                backend="yolo",
                transport_mode="edge-stream",
            )
            insert_stats_event(db_path, event)

            rows = get_recent_events(db_path, limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["request_id"], "req-2")
            self.assertEqual(rows[0]["run_id"], "run-2")
            self.assertEqual(rows[0]["model_id"], "model-2")
            self.assertEqual(rows[0]["backend"], "yolo")
            self.assertEqual(rows[0]["transport_mode"], "edge-stream")
