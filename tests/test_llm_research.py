from agentloop_trader.llm_research import LLMResearchConfig, analyze_candidate, llm_research_records
from agentloop_trader.market_data import CompanyResearchContext
from agentloop_trader.scanner import ScanCandidate


def _candidate(decision="WAIT"):
    return ScanCandidate(
        symbol="AAPL",
        decision=decision,
        best_strategy="Trendline retest continuation",
        selected_strategy="Trendline retest continuation",
        fit_score=8.0,
        last_price=200.0,
        atr_percent=1.5,
        liquidity="Good",
        backtest_return_percent=12.0,
        win_rate_percent=48.0,
        profit_factor=1.4,
        max_drawdown_percent=8.0,
        reason="Good setup quality.",
        trade_intent=None,
        scanned_at="2026-07-10T09:00:00-07:00",
    )


def _context():
    return CompanyResearchContext(
        symbol="AAPL",
        event_risk="Not connected",
        event_detail="No event calendar connected.",
        news_status="Not connected",
        fundamentals_status="Not connected",
        headlines=[],
    )


class PromotingClient:
    def analyze(self, prompt):
        return {
            "ticker": "AAPL",
            "recommendation": "TRADE",
            "confidence": 99,
            "best_strategy": "Trendline retest continuation",
            "thesis": "Looks good.",
            "supporting_evidence": ["Strong price action"],
            "opposing_evidence": [],
            "event_risk": "Low",
            "invalidation": "Risk rules fail.",
            "next_action": "Buy now.",
        }


class BrokenClient:
    def analyze(self, prompt):
        raise RuntimeError("model unavailable")


def test_llm_cannot_promote_wait_to_trade():
    result = analyze_candidate(
        _candidate("WAIT"),
        _context(),
        LLMResearchConfig(provider="ollama", model="test"),
        client=PromotingClient(),
    )

    assert result.recommendation == "WAIT"


def test_llm_uses_deterministic_fallback_on_error():
    result = analyze_candidate(
        _candidate("WATCH"),
        _context(),
        LLMResearchConfig(provider="ollama", model="test"),
        client=BrokenClient(),
    )

    assert result.used_fallback is True
    assert result.provider == "deterministic"
    assert result.recommendation == "WATCH"


def test_research_records_use_plain_reason_labels():
    result = analyze_candidate(_candidate("WATCH"), _context(), LLMResearchConfig(provider="deterministic"))

    labels = [row["Area"] for row in llm_research_records(result)]

    assert "Reasons to trade" in labels
    assert "Reasons to wait" in labels
    assert "Supports" not in labels
    assert "Concerns" not in labels
