"""Atomic, process-locked research persistence, separate from executable setups."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd


@contextmanager
def exclusive_file(path: Path, timeout: float = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Another range scan or watchlist update is already running.")
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        for attempt in range(10):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


class RectangleStore:
    def __init__(self, root="automation_logs/rectangle_research"):
        self.root = Path(root)

    def read(self, name: str, default):
        path = self.root / f"{name}.json"
        if not path.exists():
            return default
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("version") != 1:
                raise ValueError("Unsupported format")
            if not isinstance(value.get("data"), type(default)):
                raise ValueError("Unexpected data type")
            return value["data"]
        except (ValueError, OSError, TypeError) as exc:
            raise ValueError(f"Could not read rectangle {name}; the saved file was preserved: {exc}") from exc

    def write(self, name: str, value):
        atomic_json(self.root / f"{name}.json", {"version": 1, "data": value})

    def save_watch(self, rectangle, source: str):
        with exclusive_file(self.root / "watchlist.lock"):
            rows = self.read("watchlist", {})
            # One saved original rectangle per ticker; saving again does not redraw it.
            if rectangle.symbol not in rows:
                rows[rectangle.symbol] = {
                    "rectangle": rectangle.record(), "anchor_close": rectangle.last_close,
                    "source": source, "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
                self.write("watchlist", rows)

    def remove_watch(self, symbol: str):
        with exclusive_file(self.root / "watchlist.lock"):
            rows = self.read("watchlist", {})
            rows.pop(symbol, None)
            self.write("watchlist", rows)

    def history_path(self, symbol: str, source: str) -> Path:
        if not symbol or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in symbol):
            raise ValueError("Invalid stock symbol")
        if source not in {"Alpaca IEX", "Yahoo"}:
            raise ValueError("Unknown rectangle data source")
        return self.root / "history" / source.replace(" ", "_") / f"{symbol}.json"

    def read_history(self, symbol: str, source: str):
        path = self.history_path(symbol, source)
        if not path.exists():
            return None, ""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(value["bars"]).set_index("Date")
            frame.index = pd.to_datetime(frame.index)
            return frame, value["bucket"]
        except (ValueError, KeyError, TypeError):
            # A broken download cache is replaceable; durable research is not.
            return None, ""

    def write_history(self, symbol: str, source: str, frame: pd.DataFrame, bucket: str):
        data = frame.copy()
        data.index = data.index.strftime("%Y-%m-%d")
        data.index.name = "Date"
        atomic_json(self.history_path(symbol, source), {
            "bucket": bucket, "bars": data.reset_index().to_dict(orient="records"),
        })
