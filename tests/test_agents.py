from agentloop_trader.agents import build_trade_proposal, generate_research_thesis, proposal_records
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import check_trade_intent, decide_execution


def _live(signal="long"):
    return {
        "signal": signal,
        "last_p": 105.0,
        "don_high": 100.0,
        "don_low": 92.0,
        "last_atr": 2.5,
        "sma_up": True,
    }


def _stats():
    return {
        "return_pct": 4.2,
        "win_rate": 55,
        "max_drawdown_pct": 3.1,
        "profit_factor": 1.4,
    }


def test_research_agent_generates_actionable_thesis_for_trade_intent():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=105, stop_loss=100)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))

    thesis = generate_research_thesis("AAPL", _live(), _stats(), intent, risk)

    assert thesis.symbol == "AAPL"
    assert thesis.generated_by == "rules_research_agent"
    assert "buy setup" in thesis.thesis
    assert any("Breakout distance" in item for item in thesis.data_basis)
    assert any("Volatility" in item for item in thesis.data_basis)


def test_research_agent_generates_no_trade_thesis_when_flat():
    risk = check_trade_intent(None, 50_000, RiskLimits(allowed_symbols=("SYNTH",)))

    thesis = generate_research_thesis("SYNTH", _live(signal="flat"), _stats(), None, risk)

    assert thesis.symbol == "SYNTH"
    assert "not actionable right now" in thesis.thesis
    assert "Reconsider" in thesis.invalidation


def test_trade_proposal_bundles_thesis_risk_and_execution_decision():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=105, stop_loss=100)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    decision = decide_execution("paper", risk)

    proposal = build_trade_proposal("AAPL", _live(), _stats(), intent, risk, decision)
    records = proposal_records(proposal)

    assert proposal.trade_intent == intent
    assert proposal.risk_check.approved
    assert proposal.execution_decision.approved_for_execution
    assert any(row["Field"] == "Loop" for row in records)
