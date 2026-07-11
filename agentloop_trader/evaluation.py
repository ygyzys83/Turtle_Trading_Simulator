from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agentloop_trader.data import generate_synthetic_prices
from agentloop_trader.models import RiskLimits
from agentloop_trader.strategy_runtime import _run_one


@dataclass(frozen=True)
class WalkForwardResult:
    train_stats: dict
    oos_stats: dict
    verdict: str
    reasons: list[str]
    train_bars: int
    oos_bars: int
    split_index: int
    warmup_bars: int


def synthetic_ohlc_frame(n: int = 700, seed: int | None = None) -> pd.DataFrame:
    close = generate_synthetic_prices(n=n, seed=seed)
    data = pd.DataFrame(
        {
            "Close": close,
            "High": close * 1.005,
            "Low": close * 0.995,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    data.attrs["symbol"] = "SYNTH"
    return data


def _closed_trade_stats(account: float, trade_log: list[dict], eval_bars: int) -> dict:
    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    total_pnl = round(sum(t["pnl"] for t in trade_log), 2)
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    equity = account
    peak = account
    max_drawdown = 0.0
    for trade in trade_log:
        adverse_equity = equity + float(trade.get("max_adverse_pnl", 0))
        if peak > 0:
            max_drawdown = min(max_drawdown, (adverse_equity - peak) / peak)
        equity += trade["pnl"]
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity - peak) / peak)

    exposure_bars = sum(max(0, t["exit_bar"] - t["entry_bar"]) for t in trade_log)
    avg_loss = round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0
    avg_win = round(gross_wins / len(wins), 2) if wins else 0

    return {
        "final_equity": round(account + total_pnl),
        "total_pnl": round(total_pnl),
        "return_pct": round(total_pnl / account * 100, 2),
        "win_rate": round(len(wins) / len(trade_log) * 100) if trade_log else 0,
        "wins": len(wins),
        "losses": len(losses),
        "total_trades": len(trade_log),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss else (99.0 if avg_win > 0 else 0),
        "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses else (99.0 if gross_wins > 0 else 0),
        "max_drawdown_pct": round(abs(max_drawdown) * 100, 2),
        "exposure_pct": round(exposure_bars / eval_bars * 100, 2) if eval_bars else 0,
    }


def evaluate_walk_forward(
    account: float,
    entry_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    seed: int | None = None,
    market_data=None,
    train_fraction: float = 0.65,
    risk_limits: RiskLimits | None = None,
    strategy_type: str = "breakout",
    pullback_w: int = 20,
    momentum_w: int = 10,
) -> WalkForwardResult:
    warmup_bars = max(entry_w, exit_w, ma_w, pullback_w, momentum_w, 14) + 4
    data = market_data.copy() if market_data is not None else synthetic_ohlc_frame(seed=seed)
    if market_data is not None:
        data.attrs["symbol"] = getattr(market_data, "attrs", {}).get("symbol", "MARKET")

    total_bars = len(data)
    split_index = int(total_bars * train_fraction)
    min_required = warmup_bars + 30
    if split_index < warmup_bars or (total_bars - split_index) < 30:
        raise ValueError(
            f"Need at least {min_required} older bars and 30 newer bars; got {total_bars} total bars."
        )

    train_data = data.iloc[:split_index].copy()
    train_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    settings = {
        "strategy_type": strategy_type,
        "entry_window": entry_w,
        "exit_window": exit_w,
        "atr_stop_multiplier": atr_mult,
        "risk_per_trade_pct": risk_pct_dec * 100,
        "moving_average_window": ma_w,
        "pullback_average_length": pullback_w,
        "momentum_turn_length": momentum_w,
    }
    train_trade_log = _run_one(strategy_type, train_data, settings, account, risk_limits)["trade_log"]
    train_stats = _closed_trade_stats(account, train_trade_log, split_index)

    oos_start = max(0, split_index - warmup_bars)
    warmup_offset = split_index - oos_start
    oos_data = data.iloc[oos_start:].copy()
    oos_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    oos_trade_log = _run_one(strategy_type, oos_data, settings, account, risk_limits)["trade_log"]
    oos_trades = [
        trade
        for trade in oos_trade_log
        if trade["entry_bar"] >= warmup_offset
    ]
    oos_stats = _closed_trade_stats(account, oos_trades, total_bars - split_index)

    verdict, reasons = _walk_forward_verdict(train_stats, oos_stats)
    return WalkForwardResult(
        train_stats=train_stats,
        oos_stats=oos_stats,
        verdict=verdict,
        reasons=reasons,
        train_bars=split_index,
        oos_bars=total_bars - split_index,
        split_index=split_index,
        warmup_bars=warmup_bars,
    )


def _walk_forward_verdict(train_stats: dict, oos_stats: dict) -> tuple[str, list[str]]:
    reasons = []
    if oos_stats["total_trades"] == 0:
        reasons.append("No trades happened in the newer test data.")
    if oos_stats["return_pct"] < 0:
        reasons.append("The newer test data lost money.")
    if train_stats["total_trades"] > 0 and oos_stats["total_trades"] > 0:
        if oos_stats["profit_factor"] and train_stats["profit_factor"]:
            if oos_stats["profit_factor"] < train_stats["profit_factor"] * 0.5:
                reasons.append("The newer test data made much less per dollar lost than the older data.")
    if oos_stats["max_drawdown_pct"] > max(5.0, train_stats["max_drawdown_pct"] * 1.5):
        reasons.append("The newer test data had a larger account drop than the older data.")

    if not reasons:
        return "Pass", ["The newer test data looks broadly similar to the older data."]
    if oos_stats["total_trades"] == 0:
        return "Inconclusive", reasons
    return "Needs review", reasons


def walk_forward_records(result: WalkForwardResult) -> list[dict]:
    return [
        {"Metric": "Result", "Older Data": "", "Newer Data": result.verdict},
        {"Metric": "Bars", "Older Data": result.train_bars, "Newer Data": result.oos_bars},
        {"Metric": "Trades", "Older Data": result.train_stats["total_trades"], "Newer Data": result.oos_stats["total_trades"]},
        {"Metric": "Return %", "Older Data": result.train_stats["return_pct"], "Newer Data": result.oos_stats["return_pct"]},
        {"Metric": "Win rate %", "Older Data": result.train_stats["win_rate"], "Newer Data": result.oos_stats["win_rate"]},
        {"Metric": "Profit factor", "Older Data": result.train_stats["profit_factor"], "Newer Data": result.oos_stats["profit_factor"]},
        {"Metric": "Worst drop %", "Older Data": result.train_stats["max_drawdown_pct"], "Newer Data": result.oos_stats["max_drawdown_pct"]},
        {"Metric": "Time in trade %", "Older Data": result.train_stats["exposure_pct"], "Newer Data": result.oos_stats["exposure_pct"]},
    ]
