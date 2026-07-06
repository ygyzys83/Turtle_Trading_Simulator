from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.models import PACIFIC_TIME, RiskLimits, StrategyConfig

DEFAULT_RUN_MANIFEST_PATH = Path("audit_logs") / "run_manifests.jsonl"


@dataclass(frozen=True)
class RunManifest:
    session_id: str
    created_at: str
    mode_label: str
    data_source: str
    strategy: dict[str, Any]
    risk_limits: dict[str, Any]
    broker: dict[str, Any]
    account: dict[str, Any]


class RunManifestStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_RUN_MANIFEST_PATH

    def append(self, manifest: RunManifest) -> dict:
        record = run_manifest_record(manifest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read_recent(self, limit: int = 25) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:] if line.strip()]


def build_run_manifest(
    session_id: str,
    mode_label: str,
    data_source: str,
    strategy_config: StrategyConfig,
    risk_limits: RiskLimits,
    alpaca_config: AlpacaConfig,
    account_equity: float,
    paper_cash: float,
) -> RunManifest:
    return RunManifest(
        session_id=session_id,
        created_at=datetime.now(PACIFIC_TIME).isoformat(),
        mode_label=mode_label,
        data_source=data_source,
        strategy={
            "name": strategy_config.name,
            "entry_window": strategy_config.entry_window,
            "exit_window": strategy_config.exit_window,
            "atr_window": strategy_config.atr_window,
            "atr_stop_multiplier": strategy_config.atr_stop_multiplier,
            "risk_per_trade_pct": strategy_config.risk_per_trade_pct,
            "moving_average_window": strategy_config.moving_average_window,
        },
        risk_limits={
            "allowed_symbols": list(risk_limits.allowed_symbols),
            "max_risk_per_trade_pct": risk_limits.max_risk_per_trade_pct,
            "max_position_notional_pct": risk_limits.max_position_notional_pct,
            "max_portfolio_exposure_pct": risk_limits.max_portfolio_exposure_pct,
            "max_symbol_concentration_pct": risk_limits.max_symbol_concentration_pct,
            "max_session_loss_pct": risk_limits.max_session_loss_pct,
            "max_open_positions": risk_limits.max_open_positions,
            "kill_switch_enabled": risk_limits.kill_switch_enabled,
        },
        broker={
            "target": "alpaca",
            "mode": "paper" if alpaca_config.paper else "live",
            "base_url": alpaca_config.base_url,
            "has_credentials": alpaca_config.has_credentials,
            "live_writes_blocked": True,
        },
        account={"account_equity": account_equity, "paper_cash": paper_cash},
    )


def run_manifest_record(manifest: RunManifest) -> dict:
    return {
        "created_at": manifest.created_at,
        "session_id": manifest.session_id,
        "mode_label": manifest.mode_label,
        "data_source": manifest.data_source,
        "strategy": manifest.strategy,
        "risk_limits": manifest.risk_limits,
        "broker": manifest.broker,
        "account": manifest.account,
    }


def run_manifest_records(manifests: list[dict]) -> list[dict]:
    return [
        {
            "Created": record.get("created_at", ""),
            "Session": record.get("session_id", ""),
            "Mode": record.get("mode_label", ""),
            "Data Source": record.get("data_source", ""),
            "Broker Mode": record.get("broker", {}).get("mode", ""),
            "Endpoint": record.get("broker", {}).get("base_url", ""),
            "Live Writes Blocked": record.get("broker", {}).get("live_writes_blocked", True),
        }
        for record in manifests
    ]
