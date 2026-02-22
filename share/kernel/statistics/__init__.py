"""Statistics storage helpers."""

from .sqlite_store import (
    get_class_totals,
    get_overview,
    get_recent_events,
    init_stats_db,
    insert_stats_event,
)

__all__ = [
    "get_class_totals",
    "get_overview",
    "get_recent_events",
    "init_stats_db",
    "insert_stats_event",
]
