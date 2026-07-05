from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agentloop_trader.backtest import simulate_turtle_strategy
from agentloop_trader.data import generate_synthetic_prices


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
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
        "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses else 0,
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
) -> WalkForwardResult:
    warmup_bars = max(entry_w, exit_w, ma_w, 14) + 2
    data = market_data.copy() if market_data is not None else synthetic_ohlc_frame(seed=seed)
    if market_data is not None:
        data.attrs["symbol"] = getattr(market_data, "attrs", {}).get("symbol", "MARKET")

    total_bars = len(data)
    split_index = int(total_bars * train_fraction)
    min_required = warmup_bars + 30
    if split_index < warmup_bars or (total_bars - split_index) < 30:
        raise ValueError(
            f"Need at least {min_required} train bars and 30 out-of-sample bars; got {total_bars} total bars."
        )

    train_data = data.iloc[:split_index].copy()
    train_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    _, _, _, _, _, train_stats, _ = simulate_turtle_strategy(
        account, entry_w, exit_w, atr_mult, risk_pct_dec, ma_w, seed, train_data
    )

    oos_start = max(0, split_index - warmup_bars)
    warmup_offset = split_index - oos_start
    oos_data = data.iloc[oos_start:].copy()
    oos_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    _, _, _, oos_trade_log, _, _, _ = simulate_turtle_strategy(
        account, entry_w, exit_w, atr_mult, risk_pct_dec, ma_w, seed, oos_data
    )
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
        reasons.append("No out-of-sample trades were generated.")
    if oos_stats["return_pct"] < 0:
        reasons.append("Out-of-sample return is negative.")
    if train_stats["total_trades"] > 0 and oos_stats["total_trades"] > 0:
        if oos_stats["profit_factor"] and train_stats["profit_factor"]:
            if oos_stats["profit_factor"] < train_stats["profit_factor"] * 0.5:
                reasons.append("Out-of-sample profit factor is materially weaker than training.")
    if oos_stats["max_drawdown_pct"] > max(5.0, train_stats["max_drawdown_pct"] * 1.5):
        reasons.append("Out-of-sample drawdown is elevated versus training.")

    if not reasons:
        return "Pass", ["Out-of-sample behavior is broadly consistent with the training segment."]
    if oos_stats["total_trades"] == 0:
        return "Inconclusive", reasons
    return "Needs review", reasons


def walk_forward_records(result: WalkForwardResult) -> list[dict]:
    return [
        {"Metric": "Verdict", "Training": "", "Out-of-sample": result.verdict},
        {"Metric": "Bars", "Training": result.train_bars, "Out-of-sample": result.oos_bars},
        {"Metric": "Trades", "Training": result.train_stats["total_trades"], "Out-of-sample": result.oos_stats["total_trades"]},
        {"Metric": "Return %", "Training": result.train_stats["return_pct"], "Out-of-sample": result.oos_stats["return_pct"]},
        {"Metric": "Win rate %", "Training": result.train_stats["win_rate"], "Out-of-sample": result.oos_stats["win_rate"]},
        {"Metric": "Profit factor", "Training": result.train_stats["profit_factor"], "Out-of-sample": result.oos_stats["profit_factor"]},
        {"Metric": "Max drawdown %", "Training": result.train_stats["max_drawdown_pct"], "Out-of-sample": result.oos_stats["max_drawdown_pct"]},
        {"Metric": "Exposure %", "Training": result.train_stats["exposure_pct"], "Out-of-sample": result.oos_stats["exposure_pct"]},
    ]

