"""HTTP transport for statistics push events."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from share.types.errors import TransportError
from share.types.stats import StatsEvent


def push_stats_event(
    event: StatsEvent,
    endpoint: str,
    api_key: str,
    timeout_sec: float,
) -> None:
    payload = json.dumps(event.to_dict(), ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(url=endpoint, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            status_code = int(response.status)
            if status_code != 200:
                raise TransportError(f"stats push failed: status_code={status_code}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TransportError(f"stats push failed: status_code={exc.code}, body={detail}") from exc
    except URLError as exc:
        raise TransportError(f"stats push failed: {exc.reason}") from exc
