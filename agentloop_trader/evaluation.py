from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agentloop_trader.assets import normalize_asset_class
from agentloop_trader.data import generate_synthetic_prices
from agentloop_trader.fees import estimate_alpaca_round_trip_fees
from agentloop_trader.models import RiskLimits
from agentloop_trader.performance import allocation_metrics, elapsed_years, ticker_allocated_capital
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


@dataclass(frozen=True)
class PeriodPerformance:
    label: str
    start_date: str
    end_date: str
    bars: int
    stats: dict
    buy_and_hold_return_percent: float
    excess_return_percent: float


@dataclass(frozen=True)
class PeriodPerformanceResult:
    periods: tuple[PeriodPerformance, ...]
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


def _closed_trade_stats(
    account: float,
    trade_log: list[dict],
    eval_bars: int,
    risk_limits: RiskLimits | None = None,
    years: float | None = None,
) -> dict:
    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    total_pnl = round(sum(t["pnl"] for t in trade_log), 2)
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    equity = account
    peak = account
    max_drawdown_dollars = 0.0
    for trade in trade_log:
        adverse_equity = equity + float(trade.get("max_adverse_pnl", 0))
        max_drawdown_dollars = max(max_drawdown_dollars, peak - adverse_equity)
        equity += trade["pnl"]
        peak = max(peak, equity)
        max_drawdown_dollars = max(max_drawdown_dollars, peak - equity)

    exposure_bars = sum(max(0, t["exit_bar"] - t["entry_bar"]) for t in trade_log)
    avg_loss = round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0
    avg_win = round(gross_wins / len(wins), 2) if wins else 0

    capital = allocation_metrics(
        account_equity=account,
        total_pnl=total_pnl,
        max_drawdown_dollars=max_drawdown_dollars,
        risk_limits=risk_limits,
        years=years,
    )
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
        "max_drawdown_pct": round(max_drawdown_dollars / account * 100, 2) if account else 0,
        "exposure_pct": round(exposure_bars / eval_bars * 100, 2) if eval_bars else 0,
        **capital,
    }


def _strategy_settings(
    *,
    strategy_type: str,
    entry_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    pullback_w: int,
    momentum_w: int,
    rsi_entry_filter_enabled: bool,
    rsi_length: int,
    rsi_oversold: float,
    rsi_overbought: float,
    rsi_decline_points: float,
    rsi_rebound_points: float,
    rsi_max_rebound_points: float,
    rsi_sell_recovery_points: float,
    rsi_swing_lookback: int,
    rsi_stop_mode: str,
    rsi_emergency_atr_multiplier: float,
    rsi_max_holding_enabled: bool,
    rsi_max_holding_bars: int,
    rsi_profit_only_exit: bool,
) -> dict:
    return {
        "strategy_type": strategy_type,
        "entry_window": entry_w,
        "exit_window": exit_w,
        "atr_stop_multiplier": atr_mult,
        "risk_per_trade_pct": risk_pct_dec * 100,
        "moving_average_window": ma_w,
        "pullback_average_length": pullback_w,
        "momentum_turn_length": momentum_w,
        "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
        "rsi_length": rsi_length,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "rsi_decline_points": rsi_decline_points,
        "rsi_rebound_points": rsi_rebound_points,
        "rsi_max_rebound_points": rsi_max_rebound_points,
        "rsi_sell_recovery_points": rsi_sell_recovery_points,
        "rsi_swing_lookback": rsi_swing_lookback,
        "rsi_stop_mode": rsi_stop_mode,
        "rsi_emergency_atr_multiplier": rsi_emergency_atr_multiplier,
        "rsi_max_holding_enabled": rsi_max_holding_enabled,
        "rsi_max_holding_bars": rsi_max_holding_bars,
        "rsi_profit_only_exit": rsi_profit_only_exit,
    }


def _buy_and_hold_return_percent(
    data: pd.DataFrame,
    account: float,
    risk_limits: RiskLimits | None,
) -> float:
    close = data["Close"].astype(float).dropna()
    if len(close) < 2 or account <= 0:
        return 0.0
    allocation = ticker_allocated_capital(account, risk_limits)
    if allocation <= 0 or float(close.iloc[0]) <= 0:
        return 0.0
    quantity = allocation / float(close.iloc[0])
    buy_fees, sell_fees = estimate_alpaca_round_trip_fees(
        asset_class=normalize_asset_class(
            data.attrs.get("asset_class"),
            str(data.attrs.get("symbol", "")),
        ),
        quantity=quantity,
        entry_price=float(close.iloc[0]),
        exit_price=float(close.iloc[-1]),
    )
    pnl = (
        (float(close.iloc[-1]) - float(close.iloc[0])) * quantity
        - buy_fees.total
        - sell_fees.total
    )
    return round(pnl / allocation * 100, 2)


def evaluate_period_performance(
    account: float,
    entry_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    seed: int | None = None,
    market_data=None,
    risk_limits: RiskLimits | None = None,
    strategy_type: str = "breakout",
    pullback_w: int = 20,
    momentum_w: int = 10,
    rsi_entry_filter_enabled: bool = False,
    rsi_length: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    rsi_decline_points: float = 40.0,
    rsi_rebound_points: float = 3.0,
    rsi_max_rebound_points: float = 12.0,
    rsi_sell_recovery_points: float = 35.0,
    rsi_swing_lookback: int = 24,
    rsi_stop_mode: str = "standard_atr",
    rsi_emergency_atr_multiplier: float = 5.0,
    rsi_max_holding_enabled: bool = True,
    rsi_max_holding_bars: int = 100,
    rsi_profit_only_exit: bool = False,
    older_fraction: float = 0.55,
    latest_fraction: float = 0.20,
) -> PeriodPerformanceResult:
    if not 0 < older_fraction < 1:
        raise ValueError("Older price fraction must be between 0 and 1.")
    if not 0 < latest_fraction < 1:
        raise ValueError("Latest price fraction must be between 0 and 1.")
    if older_fraction + latest_fraction >= 1:
        raise ValueError("Older and latest price fractions must leave room for the newer price section.")

    warmup_bars = max(entry_w, exit_w, ma_w, pullback_w, momentum_w, rsi_length, rsi_swing_lookback, 14) + 4
    data = market_data.copy() if market_data is not None else synthetic_ohlc_frame(seed=seed)
    if market_data is not None:
        data.attrs.update(getattr(market_data, "attrs", {}))
        data.attrs.setdefault("symbol", "MARKET")
    total_bars = len(data)
    older_end = int(total_bars * older_fraction)
    latest_start = int(total_bars * (1.0 - latest_fraction))
    if older_end < warmup_bars + 30 or latest_start - older_end < 30 or total_bars - latest_start < 30:
        raise ValueError(
            "Not enough price history to show older, newer, and latest performance with at least 30 bars in each section."
        )

    settings = _strategy_settings(
        strategy_type=strategy_type,
        entry_w=entry_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        risk_pct_dec=risk_pct_dec,
        ma_w=ma_w,
        pullback_w=pullback_w,
        momentum_w=momentum_w,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
        rsi_length=rsi_length,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        rsi_decline_points=rsi_decline_points,
        rsi_rebound_points=rsi_rebound_points,
        rsi_max_rebound_points=rsi_max_rebound_points,
        rsi_sell_recovery_points=rsi_sell_recovery_points,
        rsi_swing_lookback=rsi_swing_lookback,
        rsi_stop_mode=rsi_stop_mode,
        rsi_emergency_atr_multiplier=rsi_emergency_atr_multiplier,
        rsi_max_holding_enabled=rsi_max_holding_enabled,
        rsi_max_holding_bars=rsi_max_holding_bars,
        rsi_profit_only_exit=rsi_profit_only_exit,
    )
    newer_fraction = 1.0 - older_fraction - latest_fraction
    boundaries = (
        (f"Older {older_fraction:.0%}", 0, older_end),
        (f"Newer {newer_fraction:.0%}", older_end, latest_start),
        (f"Latest {latest_fraction:.0%}", latest_start, total_bars),
    )
    periods: list[PeriodPerformance] = []
    for label, start, end in boundaries:
        load_start = max(0, start - warmup_bars)
        warmup_offset = start - load_start
        section = data.iloc[load_start:end].copy()
        section.attrs.update(data.attrs)
        section.attrs["_evaluation_start_bar"] = warmup_offset
        trade_log = _run_one(strategy_type, section, settings, account, risk_limits)["trade_log"]
        trades = [trade for trade in trade_log if trade["entry_bar"] >= warmup_offset]
        evaluation_data = data.iloc[start:end].copy()
        evaluation_data.attrs.update(data.attrs)
        stats = _closed_trade_stats(
            account,
            trades,
            end - start,
            risk_limits,
            elapsed_years(data.index, start=start, end=end),
        )
        benchmark_return = _buy_and_hold_return_percent(evaluation_data, account, risk_limits)
        periods.append(
            PeriodPerformance(
                label=label,
                start_date=_display_timestamp(evaluation_data.index[0]),
                end_date=_display_timestamp(evaluation_data.index[-1]),
                bars=end - start,
                stats=stats,
                buy_and_hold_return_percent=benchmark_return,
                excess_return_percent=round(stats["allocated_return_pct"] - benchmark_return, 2),
            )
        )
    return PeriodPerformanceResult(periods=tuple(periods), warmup_bars=warmup_bars)


def _display_timestamp(value) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.hour or timestamp.minute or timestamp.second:
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return timestamp.strftime("%Y-%m-%d")


def period_performance_records(result: PeriodPerformanceResult) -> list[dict]:
    return [
        {
            "Price section": period.label,
            "Dates": f"{period.start_date} to {period.end_date}",
            "Completed trades": period.stats["total_trades"],
            "Strategy return": f"{period.stats['allocated_return_pct']:+.2f}%",
            "Buy and hold": f"{period.buy_and_hold_return_percent:+.2f}%",
            "Difference": f"{period.excess_return_percent:+.2f}%",
            "Profit factor": period.stats["profit_factor"],
            "Worst drop": f"{period.stats['allocated_max_drawdown_pct']:.2f}%",
        }
        for period in result.periods
    ]


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
    rsi_entry_filter_enabled: bool = False,
    rsi_length: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    rsi_decline_points: float = 40.0,
    rsi_rebound_points: float = 3.0,
    rsi_max_rebound_points: float = 12.0,
    rsi_sell_recovery_points: float = 35.0,
    rsi_swing_lookback: int = 24,
    rsi_stop_mode: str = "standard_atr",
    rsi_emergency_atr_multiplier: float = 5.0,
    rsi_max_holding_enabled: bool = True,
    rsi_max_holding_bars: int = 100,
    rsi_profit_only_exit: bool = False,
) -> WalkForwardResult:
    warmup_bars = max(entry_w, exit_w, ma_w, pullback_w, momentum_w, rsi_length, rsi_swing_lookback, 14) + 4
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
    settings = _strategy_settings(
        strategy_type=strategy_type,
        entry_w=entry_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        risk_pct_dec=risk_pct_dec,
        ma_w=ma_w,
        pullback_w=pullback_w,
        momentum_w=momentum_w,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
        rsi_length=rsi_length,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        rsi_decline_points=rsi_decline_points,
        rsi_rebound_points=rsi_rebound_points,
        rsi_max_rebound_points=rsi_max_rebound_points,
        rsi_sell_recovery_points=rsi_sell_recovery_points,
        rsi_swing_lookback=rsi_swing_lookback,
        rsi_stop_mode=rsi_stop_mode,
        rsi_emergency_atr_multiplier=rsi_emergency_atr_multiplier,
        rsi_max_holding_enabled=rsi_max_holding_enabled,
        rsi_max_holding_bars=rsi_max_holding_bars,
        rsi_profit_only_exit=rsi_profit_only_exit,
    )
    train_trade_log = _run_one(strategy_type, train_data, settings, account, risk_limits)["trade_log"]
    train_stats = _closed_trade_stats(
        account,
        train_trade_log,
        split_index,
        risk_limits,
        elapsed_years(train_data.index),
    )

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
    oos_stats = _closed_trade_stats(
        account,
        oos_trades,
        total_bars - split_index,
        risk_limits,
        elapsed_years(data.index, start=split_index),
    )

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
    if oos_stats["allocated_return_pct"] < 0:
        reasons.append("The newer test data lost money.")
    if train_stats["total_trades"] > 0 and oos_stats["total_trades"] > 0:
        if oos_stats["profit_factor"] and train_stats["profit_factor"]:
            if oos_stats["profit_factor"] < train_stats["profit_factor"] * 0.5:
                reasons.append("The newer test data made much less per dollar lost than the older data.")
    if oos_stats["allocated_max_drawdown_pct"] > max(5.0, train_stats["allocated_max_drawdown_pct"] * 1.5):
        reasons.append("The newer test data had a larger allocated-capital drop than the older data.")

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
        {"Metric": "Allocated return %", "Older Data": result.train_stats["allocated_return_pct"], "Newer Data": result.oos_stats["allocated_return_pct"]},
        {"Metric": "Account return %", "Older Data": result.train_stats["return_pct"], "Newer Data": result.oos_stats["return_pct"]},
        {"Metric": "Win rate %", "Older Data": result.train_stats["win_rate"], "Newer Data": result.oos_stats["win_rate"]},
        {"Metric": "Profit factor", "Older Data": result.train_stats["profit_factor"], "Newer Data": result.oos_stats["profit_factor"]},
        {"Metric": "Allocated worst drop %", "Older Data": result.train_stats["allocated_max_drawdown_pct"], "Newer Data": result.oos_stats["allocated_max_drawdown_pct"]},
        {"Metric": "Time in trade %", "Older Data": result.train_stats["exposure_pct"], "Newer Data": result.oos_stats["exposure_pct"]},
    ]
