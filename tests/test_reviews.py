from agentloop_trader.models import TradeThesis
from agentloop_trader.reviews import review_closed_trade, review_records


def _trade(pnl=250, pct=0.5):
    return {
        "trade": 1,
        "symbol": "AAPL",
        "entry_bar": 10,
        "exit_bar": 20,
        "stop": 95,
        "pnl": pnl,
        "pct_acct": pct,
    }


def test_post_trade_review_scores_winning_trade():
    thesis = TradeThesis(
        symbol="AAPL",
        thesis="Trend thesis",
        data_basis=["breakout"],
        invalidation="Stop loss hit",
    )

    review = review_closed_trade(_trade(), thesis)

    assert review.outcome == "Win"
    assert review.rule_following_score == 100
    assert "Aligned" in review.thesis_alignment
    assert review_records(review)


def test_post_trade_review_scores_losing_trade_as_invalidated():
    review = review_closed_trade(_trade(pnl=-150, pct=-0.3))

    assert review.outcome == "Loss"
    assert review.thesis_alignment == "Invalidated or stopped out"
    assert any("Loss was contained" in lesson for lesson in review.lessons)


def test_post_trade_review_penalizes_missing_stop_and_bad_bars():
    trade = _trade()
    trade.pop("stop")
    trade["entry_bar"] = 20
    trade["exit_bar"] = 10

    review = review_closed_trade(trade)

    assert review.rule_following_score < 100
    assert any("Stop level" in lesson for lesson in review.lessons)
    assert any("data integrity" in lesson for lesson in review.lessons)

