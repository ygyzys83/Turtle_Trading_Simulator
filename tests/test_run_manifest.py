import tempfile
from datetime import datetime

from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.models import RiskLimits, StrategyConfig
from agentloop_trader.run_manifest import (
    RunManifestStore,
    build_run_manifest,
    run_manifest_record,
    run_manifest_records,
)


def test_run_manifest_records_strategy_risk_and_broker_context():
    manifest = build_run_manifest(
        session_id="session-1",
        mode_label="Paper trading",
        data_source="AAPL daily data",
        strategy_config=StrategyConfig(entry_window=10, exit_window=5, risk_per_trade_pct=0.5),
        risk_limits=RiskLimits(allowed_symbols=("AAPL",), kill_switch_enabled=True),
        alpaca_config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        account_equity=50_000,
        paper_cash=49_000,
    )
    record = run_manifest_record(manifest)
    rows = run_manifest_records([record])

    assert record["strategy"]["entry_window"] == 10
    assert record["risk_limits"]["allowed_symbols"] == ["AAPL"]
    assert record["risk_limits"]["kill_switch_enabled"]
    assert record["broker"]["mode"] == "paper"
    assert record["broker"]["live_writes_blocked"]
    assert rows[0]["Endpoint"] == "https://paper-api.alpaca.markets/v2"
    assert not record["created_at"].endswith("+00:00")
    assert datetime.fromisoformat(record["created_at"]).utcoffset().total_seconds() in {-7 * 3600, -8 * 3600}


def test_run_manifest_store_appends_and_reads_recent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = RunManifestStore(f"{tmp_dir}/manifests.jsonl")
        first = build_run_manifest(
            "session-1",
            "Shadow mode",
            "Synthetic",
            StrategyConfig(),
            RiskLimits(),
            AlpacaConfig(api_key=None, api_secret=None),
            50_000,
            50_000,
        )
        second = build_run_manifest(
            "session-2",
            "Paper trading",
            "AAPL",
            StrategyConfig(),
            RiskLimits(),
            AlpacaConfig(api_key="key", api_secret="secret"),
            50_000,
            49_500,
        )

        store.append(first)
        store.append(second)
        rows = store.read_recent(limit=1)

    assert rows[0]["session_id"] == "session-2"
