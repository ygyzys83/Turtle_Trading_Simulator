from __future__ import annotations

import numpy as np

from agentloop_trader.data import generate_synthetic_prices
from agentloop_trader.indicators import calc_atr, calc_sma
from agentloop_trader.models import BacktestResult, StrategyConfig, TradeIntent


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak)
    return round(abs(max_dd) * 100, 2)


def _build_stats(
    account: float,
    final_balance: float,
    trade_log: list[dict],
    equity_curve: list[float],
    exposure_bars: int,
    total_bars: int,
) -> dict:
    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trade_log)
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    win_rate = round(len(wins) / len(trade_log) * 100) if trade_log else 0
    avg_win = round(gross_wins / len(wins), 2) if wins else 0
    avg_loss = round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0
    rr = round(abs(avg_win / avg_loss), 2) if avg_loss else 0
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses else 0
    exposure_pct = round(exposure_bars / total_bars * 100, 2) if total_bars else 0

    result = BacktestResult(
        starting_equity=account,
        final_equity=round(final_balance, 2),
        total_pnl=round(total_pnl, 2),
        return_pct=round(total_pnl / account * 100, 2),
        total_trades=len(trade_log),
        win_rate=win_rate,
        wins=len(wins),
        losses=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss,
        reward_to_risk=rr,
        profit_factor=profit_factor,
        max_drawdown_pct=_max_drawdown_pct(equity_curve),
        exposure_pct=exposure_pct,
    )

    return {
        "final_equity": round(result.final_equity),
        "total_pnl": round(result.total_pnl),
        "return_pct": result.return_pct,
        "win_rate": result.win_rate,
        "wins": result.wins,
        "losses": result.losses,
        "total_trades": result.total_trades,
        "rr_ratio": result.reward_to_risk,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
        "exposure_pct": result.exposure_pct,
        "result": result,
    }


def simulate_turtle_strategy(
    account: float,
    entry_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    seed: int | None = None,
    market_data=None,
):
    config = StrategyConfig(
        entry_window=entry_w,
        exit_window=exit_w,
        atr_stop_multiplier=atr_mult,
        risk_per_trade_pct=risk_pct_dec * 100,
        moving_average_window=ma_w,
    )

    if market_data is None:
        n_bars = 400
        prices = generate_synthetic_prices(n_bars, seed)
        highs = lows = None
        labels = [f"Day {i + 1}" for i in range(n_bars)]
        symbol = "SYNTH"
    else:
        prices = market_data["Close"].to_numpy(dtype=float)
        highs = market_data["High"].to_numpy(dtype=float)
        lows = market_data["Low"].to_numpy(dtype=float)
        n_bars = len(prices)
        labels = market_data.index.strftime("%Y-%m-%d").tolist()
        symbol = str(getattr(market_data, "attrs", {}).get("symbol", "MARKET"))

    min_bars = max(entry_w, exit_w, ma_w, config.atr_window) + 2
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, config.atr_window, highs, lows)
    smas = calc_sma(prices, ma_w)

    trade_log = []
    equity_curve = [float(account)]
    exposure_bars = 0
    in_trade = False
    entry_price = stop_price = shares = entry_bar = 0
    balance = float(account)
    start = max(entry_w, exit_w, ma_w)

    live_bar = n_bars - 1
    for i in range(start, live_bar):
        p = prices[i]
        sma = smas[i]
        atr = atrs[i]
        if sma is None or atr is None:
            equity_curve.append(balance)
            continue

        don_high = float(np.max(prices[i - entry_w:i]))
        don_low = float(np.min(prices[i - exit_w:i]))
        ma_up = sma > (smas[i - 1] if smas[i - 1] is not None else sma)

        if not in_trade:
            if p > don_high and ma_up:
                stop = p - atr_mult * atr
                risk = p - stop
                size = int((balance * risk_pct_dec) / risk) if risk > 0 else 0
                if size > 0:
                    in_trade = True
                    entry_price = p
                    stop_price = stop
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            mark_to_market = balance + (p - entry_price) * shares
            if p <= stop_price or p <= don_low:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append({
                    "trade": len(trade_log) + 1,
                    "symbol": symbol,
                    "entry_date": labels[entry_bar],
                    "exit_date": labels[i],
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry": round(entry_price, 2),
                    "exit": round(p, 2),
                    "shares": shares,
                    "stop": round(stop_price, 2),
                    "pnl": round(pnl, 2),
                    "pct_acct": round(pnl / account * 100, 2),
                })
                in_trade = False
                equity_curve.append(balance)
                continue
            stop_price = max(stop_price, p - atr_mult * atr)
            equity_curve.append(mark_to_market)
            continue

        equity_curve.append(balance)

    last_p = float(prices[-1])
    last_atr = atrs[-1]
    last_sma = smas[-1]
    prev_sma = next((smas[i] for i in range(len(smas) - 2, -1, -1) if smas[i] is not None), last_sma)
    sma_up = bool(last_sma and prev_sma and last_sma > prev_sma)
    dh_last = float(np.max(prices[-1 - entry_w:-1]))
    dl_last = float(np.min(prices[-1 - exit_w:-1]))
    open_position_value = (last_p - entry_price) * shares if in_trade else 0.0
    live_balance = balance + open_position_value
    pos_size = int((balance * risk_pct_dec) / (atr_mult * last_atr)) if last_atr else 0

    signal = "flat"
    if not in_trade and last_p > dh_last and sma_up:
        signal = "long"
    elif in_trade and (last_p <= stop_price or last_p <= dl_last):
        signal = "exit"

    proposed_trade_intent = None
    if signal == "long" and pos_size > 0:
        proposed_trade_intent = TradeIntent(
            symbol=symbol,
            side="buy",
            quantity=pos_size,
            entry_price=last_p,
            stop_loss=round(last_p - atr_mult * last_atr, 2) if last_atr else None,
            max_holding_bars=exit_w,
            rationale=f"{entry_w}-bar breakout with upward {ma_w}-bar SMA filter.",
            source_signals=[
                f"close_above_{entry_w}_bar_high",
                f"sma_{ma_w}_sloping_up",
                "atr_position_sizing",
            ],
        )

    live = {
        "last_p": last_p,
        "last_atr": last_atr,
        "last_sma": last_sma,
        "sma_up": sma_up,
        "don_high": dh_last,
        "don_low": dl_last,
        "pos_size": pos_size,
        "stop_from_entry": round(atr_mult * last_atr, 2) if last_atr else 0,
        "balance": live_balance,
        "signal": signal,
        "trade_intent": proposed_trade_intent,
    }

    stats = _build_stats(account, balance, trade_log, equity_curve, exposure_bars, n_bars)
    return prices, smas, atrs, trade_log, live, stats, labels
