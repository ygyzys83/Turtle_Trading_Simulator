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
        lessons.append("The stop was missing from the record. Do not review or reuse this trade without a stop.")
    if trade.get("exit_bar", 0) < trade.get("entry_bar", 0):
        score -= 40
        lessons.append("The exit appears before the entry in the record. Check the data before trusting this trade.")
    if abs(pct_account) > 2.0:
        score -= 15
        lessons.append("The trade moved the account by more than 2%. Check whether the size was too large.")

    if pnl > 0:
        lessons.append("The winner matched the trend idea. Keep the same exit discipline.")
        thesis_alignment = "Matched trade idea"
    elif pnl < 0:
        lessons.append("The loss was contained. Check that the exit followed the stop or channel rule.")
        thesis_alignment = "Stopped out or invalidated"
    else:
        lessons.append("The result was flat. Check whether costs or slippage would have made it a loss.")
        thesis_alignment = "Neutral"

    if thesis is not None:
        lessons.append(f"What would have made the idea wrong: {thesis.invalidation}")

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
        {"Field": "Rule Score", "Value": review.rule_following_score},
        {"Field": "Matched Trade Idea", "Value": review.thesis_alignment},
    ]
