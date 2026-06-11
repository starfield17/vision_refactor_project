"""Deploy backends for kernel.

Imports are lazy to avoid loading optional model runtimes during service startup.
"""

from __future__ import annotations

__all__ = [
    "run_edge_llm_deploy",
    "run_edge_local_deploy",
    "run_edge_locate_anything_deploy",
    "run_edge_stream_deploy",
    "run_remote_deploy",
]


def __getattr__(name: str):
    if name == "run_edge_llm_deploy":
        from .edge_llm import run_edge_llm_deploy

        return run_edge_llm_deploy
    if name == "run_edge_local_deploy":
        from .edge_local import run_edge_local_deploy

        return run_edge_local_deploy
    if name == "run_edge_locate_anything_deploy":
        from .edge_locate_anything import run_edge_locate_anything_deploy

        return run_edge_locate_anything_deploy
    if name == "run_edge_stream_deploy":
        from .edge_stream import run_edge_stream_deploy

        return run_edge_stream_deploy
    if name == "run_remote_deploy":
        from .remote_server import run_remote_deploy

        return run_remote_deploy
    raise AttributeError(name)
