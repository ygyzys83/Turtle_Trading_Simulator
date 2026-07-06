from datetime import UTC, datetime

from agentloop_trader.audit import build_audit_events, events_to_records
from agentloop_trader.models import AuditEvent, RiskLimits
from agentloop_trader.risk import check_trade_intent, decide_execution


def test_audit_records_include_backtest_risk_and_execution_events():
    risk = check_trade_intent(
        None,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("SYNTH",)),
    )
    decision = decide_execution("backtest_only", risk)

    events = build_audit_events(
        mode_label="Backtest only",
        source_caption="synthetic price data",
        trade_intent=None,
        risk_check=risk,
        execution_decision=decision,
        stats={"final_equity": 50_000, "total_trades": 0, "max_drawdown_pct": 0},
    )
    records = events_to_records(events)

    assert [event.event_type for event in events] == [
        "backtest_completed",
        "trade_intent_absent",
        "risk_check_completed",
        "execution_decision_recorded",
    ]
    assert records[0]["Type"] == "backtest_completed"


def test_audit_records_display_pacific_time_for_utc_events():
    records = events_to_records([
        AuditEvent(
            event_type="timezone",
            message="UTC display should be Pacific",
            created_at=datetime(2026, 7, 6, 2, 7, 44, tzinfo=UTC),
        )
    ])

    assert records[0]["Time"] == "2026-07-05 19:07:44 PDT"
