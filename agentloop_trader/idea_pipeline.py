from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo


PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}(?:/[A-Z]{3,8})?$")


@dataclass(frozen=True)
class TickerResearchIdea:
    ticker: str
    reason_to_research: str
    source: str
    created_at: str


class TickerIdeaSource(Protocol):
    """Research-only source. Implementations have no broker or order interface."""

    def propose(self, limit: int) -> list[TickerResearchIdea]: ...


def normalize_research_ideas(ideas: list[TickerResearchIdea], limit: int = 25) -> list[TickerResearchIdea]:
    normalized_limit = max(0, int(limit))
    if normalized_limit == 0:
        return []
    normalized: list[TickerResearchIdea] = []
    seen: set[str] = set()
    for idea in ideas:
        ticker = str(idea.ticker).strip().upper().replace("-USD", "/USD")
        if not SYMBOL_PATTERN.fullmatch(ticker) or ticker in seen:
            continue
        reason = str(idea.reason_to_research).strip()[:500]
        if not reason:
            continue
        normalized.append(TickerResearchIdea(
            ticker=ticker,
            reason_to_research=reason,
            source=str(idea.source).strip()[:100] or "unknown",
            created_at=str(idea.created_at).strip() or datetime.now(PACIFIC_TIME).isoformat(),
        ))
        seen.add(ticker)
        if len(normalized) >= normalized_limit:
            break
    return normalized


class TickerResearchQueueStore:
    """Durable research queue; intentionally cannot create broker orders."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def replace(self, ideas: list[TickerResearchIdea]) -> None:
        normalized = normalize_research_ideas(ideas)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            temporary.write_text(
                json.dumps([asdict(idea) for idea in normalized], indent=2),
                encoding="utf-8",
            )
            try:
                for attempt in range(10):
                    try:
                        temporary.replace(self.path)
                        return
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(0.02)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def read(self) -> list[TickerResearchIdea]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        rows = []
        for item in payload if isinstance(payload, list) else []:
            try:
                rows.append(TickerResearchIdea(**item))
            except TypeError:
                continue
        return normalize_research_ideas(rows)


def collect_research_ideas(source: TickerIdeaSource, limit: int = 25) -> list[TickerResearchIdea]:
    """Collect and validate proposals without running a strategy or authorizing an order."""
    return normalize_research_ideas(source.propose(max(0, limit)), limit=limit)
