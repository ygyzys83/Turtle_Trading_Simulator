from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agentloop_trader.research_agent import ResearchAgentReport

PACIFIC_TIME = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class ResearchSnapshot:
    created_at: str
    ticker: str
    selected_strategy: str
    best_strategy: str
    final_read: str
    next_action: str
    fit_score: float
    thesis: str
    settings_key: str


class ResearchSnapshotStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, snapshot: ResearchSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(snapshot), sort_keys=True) + "\n")

    def read_recent(self, limit: int = 50) -> list[ResearchSnapshot]:
        if not self.path.exists():
            return []
        rows: list[ResearchSnapshot] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(ResearchSnapshot(**json.loads(line)))
            except (TypeError, json.JSONDecodeError):
                continue
        return rows[-limit:]

    def latest_for_ticker(self, ticker: str, exclude_settings_key: str | None = None) -> ResearchSnapshot | None:
        ticker = ticker.strip().upper()
        for snapshot in reversed(self.read_recent(limit=500)):
            if snapshot.ticker != ticker:
                continue
            if exclude_settings_key and snapshot.settings_key == exclude_settings_key:
                continue
            return snapshot
        return None


def build_research_snapshot(
    report: ResearchAgentReport,
    selected_strategy: str,
    settings_key: str,
) -> ResearchSnapshot:
    best_fit = next((fit for fit in report.strategy_fits if fit.strategy == report.best_strategy), None)
    return ResearchSnapshot(
        created_at=datetime.now(PACIFIC_TIME).isoformat(),
        ticker=report.ticker,
        selected_strategy=selected_strategy,
        best_strategy=report.best_strategy,
        final_read=report.final_read,
        next_action=report.next_action,
        fit_score=best_fit.score if best_fit else 0.0,
        thesis=report.thesis,
        settings_key=settings_key,
    )


def compare_research_snapshots(
    previous: ResearchSnapshot | None,
    current: ResearchSnapshot,
) -> list[dict[str, Any]]:
    if previous is None:
        return [
            {
                "Area": "Research loop",
                "Read": "First saved read",
                "Plain English": "No earlier saved read exists for this ticker yet.",
            }
        ]

    score_change = round(current.fit_score - previous.fit_score, 2)
    if current.final_read == "TRADE" and previous.final_read != "TRADE":
        read = "Improving"
        detail = "The current read moved into TRADE."
    elif current.final_read != "TRADE" and previous.final_read == "TRADE":
        read = "Worsening"
        detail = "The current read moved away from TRADE."
    elif score_change > 0:
        read = "Improving"
        detail = f"Best-fit score improved by {score_change:.2f}."
    elif score_change < 0:
        read = "Worsening"
        detail = f"Best-fit score weakened by {abs(score_change):.2f}."
    elif current.best_strategy != previous.best_strategy:
        read = "Changed"
        detail = f"Best fit changed from {previous.best_strategy} to {current.best_strategy}."
    else:
        read = "Unchanged"
        detail = "The saved read is broadly unchanged from the prior read."

    return [
        {"Area": "Research loop", "Read": read, "Plain English": detail},
        {"Area": "Previous saved read", "Read": previous.final_read, "Plain English": previous.thesis},
    ]


def research_snapshot_records(snapshots: list[ResearchSnapshot]) -> list[dict[str, Any]]:
    return [
        {
            "Saved At": snapshot.created_at,
            "Ticker": snapshot.ticker,
            "Final Read": snapshot.final_read,
            "Best Strategy": snapshot.best_strategy,
            "Fit Score": snapshot.fit_score,
            "Next Action": snapshot.next_action,
        }
        for snapshot in reversed(snapshots)
    ]
