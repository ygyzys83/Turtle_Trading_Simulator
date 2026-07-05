from __future__ import annotations

from dataclasses import dataclass

from agentloop_trader.models import TradeThesis


@dataclass(frozen=True)
class PostTradeReview:
    trade_id: int
    symbol: str
    outcome: str
    pnl: float
    pct_account: float
    rule_following_score: int
    thesis_alignment: str
    lessons: list[str]


def review_closed_trade(trade: dict, thesis: TradeThesis | None = None) -> PostTradeReview:
    pnl = float(trade.get("pnl", 0.0))
    pct_account = float(trade.get("pct_acct", 0.0))
    outcome = "Win" if pnl > 0 else "Loss" if pnl < 0 else "Flat"
    symbol = str(trade.get("symbol", thesis.symbol if thesis else "UNKNOWN"))
    lessons = []
    score = 100

    if "stop" not in trade or trade.get("stop") in (None, ""):
        score -= 30
        lessons.append("Stop level was not recorded; require stop data for review.")
    if trade.get("exit_bar", 0) < trade.get("entry_bar", 0):
        score -= 40
        lessons.append("Exit occurred before entry in the trade log; investigate data integrity.")
    if abs(pct_account) > 2.0:
        score -= 15
        lessons.append("Trade impact exceeded 2% of starting account; review sizing discipline.")

    if pnl > 0:
        lessons.append("Winner followed the trend-following thesis; preserve exit discipline.")
        thesis_alignment = "Aligned with trend thesis"
    elif pnl < 0:
        lessons.append("Loss was contained; verify that the exit matched stop/channel rules.")
        thesis_alignment = "Invalidated or stopped out"
    else:
        lessons.append("Flat result; review whether transaction costs/slippage would alter outcome.")
        thesis_alignment = "Neutral"

    if thesis is not None:
        lessons.append(f"Original invalidation: {thesis.invalidation}")

    return PostTradeReview(
        trade_id=int(trade.get("trade", 0)),
        symbol=symbol,
        outcome=outcome,
        pnl=round(pnl, 2),
        pct_account=round(pct_account, 2),
        rule_following_score=max(0, min(100, score)),
        thesis_alignment=thesis_alignment,
        lessons=lessons,
    )


def review_records(review: PostTradeReview) -> list[dict]:
    return [
        {"Field": "Trade", "Value": review.trade_id},
        {"Field": "Symbol", "Value": review.symbol},
        {"Field": "Outcome", "Value": review.outcome},
        {"Field": "P&L", "Value": review.pnl},
        {"Field": "% Account", "Value": review.pct_account},
        {"Field": "Rule Following Score", "Value": review.rule_following_score},
        {"Field": "Thesis Alignment", "Value": review.thesis_alignment},
    ]

