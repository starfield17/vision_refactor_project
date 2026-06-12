"""Reusable statistics aggregation for APIs and dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.statistics.sqlite_store import get_class_totals, get_overview, get_recent_events


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * max(0.0, min(100.0, p)) / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def filter_events(
    events: list[dict[str, Any]],
    source_id: str = "",
    min_detections: int = 0,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if (not source_id or str(event["source_id"]) == source_id)
        and int(event["total_detections"]) >= int(min_detections)
    ]


def aggregate_class_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in events:
        counts = event.get("counts_by_class", {})
        if not isinstance(counts, dict):
            continue
        for name, count in counts.items():
            try:
                c = int(count)
            except (TypeError, ValueError):
                continue
            totals[str(name)] = totals.get(str(name), 0) + c
    return totals


def build_summary(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {
            "events_total": 0.0,
            "detections_total": 0.0,
            "detections_per_event": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    latencies = [float(e["latency_ms"]) for e in events]
    detections_total = float(sum(int(e["total_detections"]) for e in events))
    events_total = float(len(events))
    return {
        "events_total": events_total,
        "detections_total": detections_total,
        "detections_per_event": detections_total / max(events_total, 1.0),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": max(latencies),
    }


def build_source_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        source = str(event["source_id"])
        entry = grouped.setdefault(
            source,
            {"source_id": source, "events": 0, "detections": 0, "latencies": []},
        )
        entry["events"] += 1
        entry["detections"] += int(event["total_detections"])
        entry["latencies"].append(float(event["latency_ms"]))

    rows: list[dict[str, Any]] = []
    for source, raw in sorted(grouped.items()):
        latencies = raw["latencies"]
        event_count = int(raw["events"])
        detections = int(raw["detections"])
        rows.append(
            {
                "source_id": source,
                "events": event_count,
                "detections": detections,
                "detections_per_event": detections / max(event_count, 1),
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "latency_max_ms": max(latencies) if latencies else 0.0,
            }
        )
    return rows


def build_class_rows(class_totals: dict[str, int]) -> list[dict[str, Any]]:
    total = max(sum(class_totals.values()), 1)
    return [
        {
            "class_name": class_name,
            "count": int(count),
            "share_pct": float(count) * 100.0 / total,
        }
        for class_name, count in sorted(class_totals.items(), key=lambda item: item[1], reverse=True)
    ]


def build_dashboard(
    db_path: Path,
    limit: int = 200,
    source_id: str = "",
    min_detections: int = 0,
) -> dict[str, Any]:
    recent = get_recent_events(db_path, limit=limit)
    filtered = filter_events(recent, source_id=source_id, min_detections=min_detections)
    class_totals = aggregate_class_totals(filtered)
    return {
        "overview": get_overview(db_path),
        "filtered_summary": build_summary(filtered),
        "class_totals": class_totals,
        "class_rows": build_class_rows(class_totals),
        "source_rows": build_source_rows(filtered),
        "events": filtered,
        "sources": sorted({str(e["source_id"]) for e in recent}),
        "all_class_totals": get_class_totals(db_path, limit=limit),
    }
