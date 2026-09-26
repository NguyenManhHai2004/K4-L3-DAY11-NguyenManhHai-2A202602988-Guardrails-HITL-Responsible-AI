"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        req_id = request_id or f"{user_id}_{len(self._open)}"
        self._open[req_id] = {
            "user_id": user_id,
            "input": text,
            "start_time": time.time(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        req_id = request_id or f"{user_id}_{len(self.logs)}"
        start_data = self._open.get(req_id, {})
        start_time = start_data.get("start_time", time.time())
        latency_ms = (time.time() - start_time) * 1000

        log_entry = {
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "input": start_data.get("input", ""),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 2),
        }
        self.logs.append(log_entry)

        # Clean up
        if req_id in self._open:
            del self._open[req_id]

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = filepath or default_audit_log_path()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
