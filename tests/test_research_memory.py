from agentloop_trader.research_agent import build_research_agent_report
from agentloop_trader.research_memory import (
    ResearchSnapshotStore,
    build_research_snapshot,
    compare_research_snapshots,
    research_snapshot_records,
)


def _report(final_read: str, signal: str, score_return: float):
    return build_research_agent_report(
        ticker="AAPL",
        selected_strategy="Trendline retest continuation",
        strategy_results={
            "Trendline retest continuation": {
                "live": {"signal": signal, "buy_requirements": {"trend": signal == "long", "retest": signal == "long"}},
                "stats": {
                    "return_pct": score_return,
                    "total_trades": 3,
                    "win_rate": 67,
                    "max_drawdown_pct": 4.0,
                    "profit_factor": 1.4,
                },
            }
        },
        setup_rows=[],
        final_read=final_read,
        decision_detail="Research read.",
        next_action="Watch.",
    )


def test_research_snapshot_store_saves_and_reads_recent(tmp_path):
    store = ResearchSnapshotStore(tmp_path / "research.jsonl")
    snapshot = build_research_snapshot(_report("WAIT", "flat", 1.0), "Trendline retest continuation", "key-1")

    store.append(snapshot)

    rows = research_snapshot_records(store.read_recent())
    assert rows[0]["Ticker"] == "AAPL"
    assert rows[0]["Final Read"] == "WAIT"


def test_compare_research_snapshots_reports_improving_when_trade_appears():
    previous = build_research_snapshot(_report("WAIT", "flat", 1.0), "Trendline retest continuation", "key-1")
    current = build_research_snapshot(_report("TRADE", "long", 5.0), "Trendline retest continuation", "key-2")

    rows = compare_research_snapshots(previous, current)

    assert rows[0]["Read"] == "Improving"
    assert "TRADE" in rows[0]["Plain English"]


def test_compare_research_snapshots_reports_first_saved_read():
    current = build_research_snapshot(_report("WAIT", "flat", 1.0), "Trendline retest continuation", "key-1")

    rows = compare_research_snapshots(None, current)

    assert rows[0]["Read"] == "First saved read"
