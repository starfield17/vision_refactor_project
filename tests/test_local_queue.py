from __future__ import annotations

import unittest
from pathlib import Path

from common.local_gui.queue import (
    LocalQueueTableModel,
    QueueItemStatus,
    create_queue_item,
)


class LocalQueueModelTests(unittest.TestCase):
    def test_queue_model_tracks_status_metrics_and_retry(self) -> None:
        model = LocalQueueTableModel()
        item = create_queue_item(
            name="exp001",
            source="/tmp/dataset",
            output="/tmp/models",
            mode="yolo",
            config_path=Path("train/config/config.example.toml"),
            workdir_override=None,
            overrides=["runtime.dry_run=true"],
        )

        model.add_record(item)
        self.assertEqual(model.metrics().queued, 1)

        model.mark_running(item.item_id)
        self.assertEqual(model.records()[0].status, QueueItemStatus.RUNNING)
        self.assertEqual(model.metrics().running, 1)

        model.mark_failed(item.item_id, "boom")
        self.assertEqual(model.records()[0].status, QueueItemStatus.FAILED)
        self.assertEqual(model.metrics().failed, 1)

        retried = model.retry_rows([0])
        self.assertEqual(retried, 1)
        self.assertEqual(model.records()[0].status, QueueItemStatus.QUEUED)

        removed = model.remove_rows([0])
        self.assertEqual(removed, 1)
        self.assertEqual(model.metrics().total, 0)


if __name__ == "__main__":
    unittest.main()
