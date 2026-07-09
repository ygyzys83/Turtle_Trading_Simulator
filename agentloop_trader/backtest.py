from __future__ import annotations

import numpy as np

from agentloop_trader.data import generate_synthetic_prices
from agentloop_trader.indicators import calc_atr, calc_rsi, calc_sma
from agentloop_trader.models import BacktestResult, RiskLimits, StrategyConfig, TradeIntent


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
    account = float(account)
    final_balance = float(final_balance)
    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    total_pnl = final_balance - account
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


def _backtest_limited_quantity(
    *,
    raw_quantity: int,
    account_equity: float,
    entry_price: float,
    stop_price: float,
    session_pnl: float,
    limits: RiskLimits | None,
) -> int:
    if raw_quantity <= 0:
        return 0
    if limits is None:
        return raw_quantity
    if limits.kill_switch_enabled:
        return 0
    max_session_loss = account_equity * limits.max_session_loss_pct / 100
    if session_pnl <= -max_session_loss:
        return 0
    if entry_price <= 0:
        return 0

    max_quantities = [raw_quantity, limits.max_quantity]
    risk_per_share = abs(entry_price - stop_price)
    max_risk_dollars = account_equity * limits.max_risk_per_trade_pct / 100
    max_quantities.append(int(max_risk_dollars // risk_per_share) if risk_per_share > 0 else 0)

    max_position_notional = account_equity * limits.max_position_notional_pct / 100
    max_portfolio_notional = account_equity * limits.max_portfolio_exposure_pct / 100
    max_symbol_notional = account_equity * limits.max_symbol_concentration_pct / 100
    max_quantities.extend([
        int(max_position_notional // entry_price),
        int(max_portfolio_notional // entry_price),
        int(max_symbol_notional // entry_price),
        int(max(0.0, account_equity) // entry_price),
    ])
    return max(0, min(max_quantities))


def _trade_record(
    *,
    trade_number: int,
    symbol: str,
    labels: list[str],
    entry_bar: int,
    exit_bar: int,
    entry_price: float,
    exit_price: float,
    shares: int,
    stop_price: float,
    account: float,
    exit_rule: str = "Exit rule",
) -> dict:
    pnl = (exit_price - entry_price) * shares
    notional = entry_price * shares
    risk_dollars = abs(entry_price - stop_price) * shares
    return {
        "trade": trade_number,
        "symbol": symbol,
        "entry_date": labels[entry_bar],
        "exit_date": labels[exit_bar],
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "entry": round(entry_price, 2),
        "exit": round(exit_price, 2),
        "shares": shares,
        "notional": round(notional, 2),
        "risk_dollars": round(risk_dollars, 2),
        "risk_pct": round(risk_dollars / account * 100, 2) if account else 0,
        "stop": round(stop_price, 2),
        "exit_rule": exit_rule,
        "pnl": round(pnl, 2),
        "pct_acct": round(pnl / account * 100, 2) if account else 0,
    }


def _profit_protection_stop(
    *,
    entry_price: float,
    initial_stop_price: float,
    current_stop_price: float,
    current_price: float,
    current_atr: float | None,
    high_since_entry: float,
    strategy_exit_price: float | None,
    breakeven_after_r: float = 1.0,
    trail_after_r: float = 2.0,
    trailing_atr_multiplier: float = 3.0,
) -> tuple[str, float]:
    initial_risk = entry_price - initial_stop_price
    candidates = [("saved stop", current_stop_price), ("original stop", initial_stop_price)]
    if strategy_exit_price is not None:
        candidates.append(("strategy exit", strategy_exit_price))
    if initial_risk > 0:
        profit_r = (current_price - entry_price) / initial_risk
        if profit_r >= breakeven_after_r:
            candidates.append(("break-even stop", entry_price))
        if profit_r >= trail_after_r and current_atr is not None:
            candidates.append(("ATR trail", high_since_entry - trailing_atr_multiplier * current_atr))
    return max(candidates, key=lambda item: item[1])


def _market_arrays(seed: int | None, market_data=None):
    if market_data is None:
        n_bars = 400
        prices = generate_synthetic_prices(n_bars, seed)
        highs = lows = volumes = None
        labels = [f"Day {i + 1}" for i in range(n_bars)]
        symbol = "SYNTH"
    else:
        prices = market_data["Close"].to_numpy(dtype=float)
        highs = market_data["High"].to_numpy(dtype=float)
        lows = market_data["Low"].to_numpy(dtype=float)
        volumes = market_data["Volume"].to_numpy(dtype=float) if "Volume" in market_data else None
        n_bars = len(prices)
        labels = market_data.index.strftime("%Y-%m-%d").tolist()
        symbol = str(getattr(market_data, "attrs", {}).get("symbol", "MARKET"))
    return prices, highs, lows, volumes, n_bars, labels, symbol


def _volume_status(volumes, index: int, window: int = 20) -> tuple[str, bool | None]:
    if volumes is None or index < window:
        return "Unknown", None
    avg = float(np.mean(volumes[index - window:index]))
    current = float(volumes[index])
    if avg <= 0:
        return "Unknown", None
    ratio = current / avg
    if ratio >= 1.5:
        return "Strong", True
    if ratio >= 1.0:
        return "Normal", True
    return "Light", False


def _rsi_status(rsi: float | None) -> str:
    if rsi is None:
        return "Unknown"
    if rsi < 50:
        return "Weak"
    if rsi <= 70:
        return "Good"
    if rsi <= 75:
        return "Strong"
    return "Extended"


def _liquidity_status(prices, volumes, index: int, window: int = 20) -> str:
    if volumes is None or index < window:
        return "Unknown"
    avg_volume = float(np.mean(volumes[index - window + 1:index + 1]))
    dollar_volume = avg_volume * float(prices[index])
    if dollar_volume >= 20_000_000:
        return "Good"
    if dollar_volume >= 5_000_000:
        return "Usable"
    return "Thin"


def _recent_descending_trendline(prices, end_index: int, lookback: int, pivot_window: int = 2) -> tuple[float, float, tuple[int, int]] | None:
    start = max(pivot_window, end_index - lookback)
    swing_highs: list[tuple[int, float]] = []
    for idx in range(start, end_index - pivot_window):
        left = prices[idx - pivot_window:idx]
        right = prices[idx + 1:idx + 1 + pivot_window]
        if len(left) < pivot_window or len(right) < pivot_window:
            continue
        value = float(prices[idx])
        if value >= max(left) and value >= max(right):
            swing_highs.append((idx, value))

    for right_idx in range(len(swing_highs) - 1, 0, -1):
        x2, y2 = swing_highs[right_idx]
        for left_idx in range(right_idx - 1, -1, -1):
            x1, y1 = swing_highs[left_idx]
            if x2 <= x1 or y2 >= y1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                return y1, slope, (x1, x2)
    return None


def _trendline_value(intercept_at_x1: float, slope: float, x1: int, index: int) -> float:
    return float(intercept_at_x1 + slope * (index - x1))


def _base_live_fields(
    *,
    prices,
    smas,
    atrs,
    rsis,
    volumes,
    index: int,
    entry_level: float,
    exit_level: float,
    stop_distance: float,
    balance: float,
    pos_size: int,
    signal: str,
    trade_intent: TradeIntent | None,
    strategy_name: str,
    setup_type: str,
    pullback_depth_pct: float | None = None,
    momentum_turn: bool | None = None,
) -> dict:
    last_p = float(prices[index])
    last_atr = atrs[index]
    last_sma = smas[index]
    prev_sma = next((smas[i] for i in range(index - 1, -1, -1) if smas[i] is not None), last_sma)
    sma_up = bool(last_sma and prev_sma and last_sma > prev_sma)
    volume_status, volume_confirmed = _volume_status(volumes, index)
    rsi = rsis[index] if rsis else None
    return {
        "strategy_name": strategy_name,
        "setup_type": setup_type,
        "last_p": last_p,
        "last_atr": last_atr,
        "last_sma": last_sma,
        "sma_up": sma_up,
        "don_high": entry_level,
        "don_low": exit_level,
        "entry_level": entry_level,
        "exit_level": exit_level,
        "pos_size": pos_size,
        "stop_from_entry": round(stop_distance, 2) if stop_distance else 0,
        "balance": balance,
        "signal": signal,
        "trade_intent": trade_intent,
        "volume_status": volume_status,
        "volume_confirmed": volume_confirmed,
        "rsi": rsi,
        "rsi_status": _rsi_status(rsi),
        "liquidity_status": _liquidity_status(prices, volumes, index),
        "pullback_depth_pct": pullback_depth_pct,
        "momentum_turn": momentum_turn,
        "market_condition": "Unknown",
        "relative_strength": "Unknown",
        "event_risk": "Unknown",
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
    risk_limits: RiskLimits | None = None,
):
    config = StrategyConfig(
        entry_window=entry_w,
        exit_window=exit_w,
        atr_stop_multiplier=atr_mult,
        risk_per_trade_pct=risk_pct_dec * 100,
        moving_average_window=ma_w,
    )

    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)

    min_bars = max(entry_w, exit_w, ma_w, config.atr_window) + 2
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, config.atr_window, highs, lows)
    smas = calc_sma(prices, ma_w)
    rsis = calc_rsi(prices, 14)

    trade_log = []
    equity_curve = [float(account)]
    exposure_bars = 0
    in_trade = False
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = 0.0
    exit_rule = "Exit rule"
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
                raw_size = int((balance * risk_pct_dec) / risk) if risk > 0 else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=float(p),
                    stop_price=float(stop),
                    session_pnl=balance - account,
                    limits=risk_limits,
                )
                if size > 0:
                    in_trade = True
                    entry_price = p
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            mark_to_market = balance + (p - entry_price) * shares
            high_value = float(highs[i]) if highs is not None else float(p)
            high_since_entry = max(high_since_entry, high_value)
            exit_rule, stop_price = _profit_protection_stop(
                entry_price=float(entry_price),
                initial_stop_price=float(initial_stop_price),
                current_stop_price=float(stop_price),
                current_price=float(p),
                current_atr=float(atr) if atr is not None else None,
                high_since_entry=float(high_since_entry),
                strategy_exit_price=float(don_low),
            )
            if p <= stop_price:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=float(p),
                    shares=int(shares),
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                ))
                in_trade = False
                equity_curve.append(balance)
                continue
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
    raw_pos_size = int((balance * risk_pct_dec) / (atr_mult * last_atr)) if last_atr else 0
    live_stop = last_p - atr_mult * last_atr if last_atr else last_p
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=live_balance,
        entry_price=last_p,
        stop_price=live_stop,
        session_pnl=live_balance - account,
        limits=risk_limits,
    )

    entry_setup_ready = bool(last_p > dh_last and sma_up)
    strategy_exit_ready = bool(last_p <= dl_last)
    simulated_exit_ready = bool(in_trade and strategy_exit_ready)

    signal = "flat"
    if entry_setup_ready:
        signal = "long"
    elif simulated_exit_ready:
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

    buy_requirements = {
        f"Price above {entry_w}-bar high": last_p > dh_last,
        f"{ma_w}-bar trend filter rising": sma_up,
        "Position size above zero": pos_size > 0,
    }
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif last_p <= dh_last:
        no_trade_reason = f"No BUY because price is not above the {entry_w}-bar entry level."
    elif not sma_up:
        no_trade_reason = f"No BUY because the {ma_w}-bar trend filter is not rising."
    elif pos_size <= 0:
        no_trade_reason = "No BUY because calculated share size is zero."
    else:
        no_trade_reason = "No BUY because the selected strategy rules are not fully met."

    live = _base_live_fields(
        prices=prices,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        entry_level=dh_last,
        exit_level=dl_last,
        stop_distance=atr_mult * last_atr if last_atr else 0,
        balance=live_balance,
        pos_size=pos_size,
        signal=signal,
        trade_intent=proposed_trade_intent,
        strategy_name="Breakout continuation",
        setup_type="breakout",
    )
    live["buy_requirements"] = buy_requirements
    live["no_trade_reason"] = no_trade_reason
    live["in_simulated_trade"] = in_trade
    live["exit_ready"] = strategy_exit_ready
    live["exit_reason"] = (
        f"Exit now because price is at or below the {exit_w}-bar exit level."
        if strategy_exit_ready
        else f"Hold because price is above the {exit_w}-bar exit level."
    )
    live["sell_requirements"] = {
        f"Price at or below {exit_w}-bar exit level": strategy_exit_ready,
    }

    final_equity = equity_curve[-1] if equity_curve else balance
    stats = _build_stats(account, final_equity, trade_log, equity_curve, exposure_bars, n_bars)
    return prices, smas, atrs, trade_log, live, stats, labels


def simulate_trend_pullback_strategy(
    account: float,
    pullback_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    trend_w: int,
    momentum_w: int = 10,
    seed: int | None = None,
    market_data=None,
    risk_limits: RiskLimits | None = None,
):
    config = StrategyConfig(
        name="Trend Pullback Continuation",
        entry_window=pullback_w,
        exit_window=exit_w,
        atr_stop_multiplier=atr_mult,
        risk_per_trade_pct=risk_pct_dec * 100,
        moving_average_window=trend_w,
    )
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    min_bars = max(pullback_w, exit_w, trend_w, momentum_w, config.atr_window) + 3
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, config.atr_window, highs, lows)
    trend_smas = calc_sma(prices, trend_w)
    pullback_smas = calc_sma(prices, pullback_w)
    exit_smas = calc_sma(prices, exit_w)
    momentum_smas = calc_sma(prices, momentum_w)
    rsis = calc_rsi(prices, 14)

    trade_log = []
    equity_curve = [float(account)]
    exposure_bars = 0
    in_trade = False
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    start = max(pullback_w, exit_w, trend_w, momentum_w, config.atr_window)
    live_bar = n_bars - 1

    def setup_at(i: int) -> tuple[bool, float | None, bool, bool, bool]:
        p = float(prices[i])
        trend_sma = trend_smas[i]
        pullback_sma = pullback_smas[i]
        momentum_sma = momentum_smas[i]
        prev_momentum_sma = momentum_smas[i - 1] if i > 0 else None
        if trend_sma is None or pullback_sma is None or momentum_sma is None or prev_momentum_sma is None:
            return False, None, False, False, False
        prev_trend = trend_smas[i - 1] if i > 0 else trend_sma
        trend_ok = bool(p > trend_sma and trend_sma >= (prev_trend or trend_sma))
        recent_low = float(np.min(prices[max(0, i - pullback_w):i + 1]))
        pullback_depth = (p - recent_low) / p * 100 if p else None
        touched_pullback = recent_low <= pullback_sma * 1.02
        momentum_turn = bool(p > momentum_sma and momentum_sma >= prev_momentum_sma and p > prices[i - 1])
        return trend_ok and touched_pullback and momentum_turn, pullback_depth, trend_ok, touched_pullback, momentum_turn

    for i in range(start, live_bar):
        p = float(prices[i])
        atr = atrs[i]
        if atr is None:
            equity_curve.append(balance)
            continue
        setup_ready, _, _, _, _ = setup_at(i)
        exit_sma = exit_smas[i]

        if not in_trade:
            if setup_ready:
                stop = min(float(np.min(prices[i - pullback_w:i + 1])), p - atr_mult * atr)
                risk = p - stop
                raw_size = int((balance * risk_pct_dec) / risk) if risk > 0 else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=float(p),
                    stop_price=float(stop),
                    session_pnl=balance - account,
                    limits=risk_limits,
                )
                if size > 0:
                    in_trade = True
                    entry_price = p
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            mark_to_market = balance + (p - entry_price) * shares
            high_value = float(highs[i]) if highs is not None else float(p)
            high_since_entry = max(high_since_entry, high_value)
            exit_rule, stop_price = _profit_protection_stop(
                entry_price=float(entry_price),
                initial_stop_price=float(initial_stop_price),
                current_stop_price=float(stop_price),
                current_price=float(p),
                current_atr=float(atr) if atr is not None else None,
                high_since_entry=float(high_since_entry),
                strategy_exit_price=float(exit_sma) if exit_sma is not None else None,
            )
            exit_hit = p <= stop_price
            if exit_hit:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=float(p),
                    shares=int(shares),
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                ))
                in_trade = False
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    last_p = float(prices[-1])
    last_atr = atrs[-1]
    pullback_level = float(pullback_smas[-1]) if pullback_smas[-1] is not None else last_p
    exit_level = float(exit_smas[-1]) if exit_smas[-1] is not None else last_p
    setup_ready, pullback_depth, trend_ok, touched_pullback, momentum_turn = setup_at(live_bar)
    open_position_value = (last_p - entry_price) * shares if in_trade else 0.0
    live_balance = balance + open_position_value
    stop_distance = atr_mult * last_atr if last_atr else 0
    raw_pos_size = int((balance * risk_pct_dec) / stop_distance) if stop_distance else 0
    live_stop = last_p - stop_distance if stop_distance else last_p
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=live_balance,
        entry_price=last_p,
        stop_price=live_stop,
        session_pnl=live_balance - account,
        limits=risk_limits,
    )

    strategy_exit_ready = bool(last_p < exit_level)
    simulated_exit_ready = bool(in_trade and strategy_exit_ready)

    signal = "flat"
    if setup_ready:
        signal = "long"
    elif simulated_exit_ready:
        signal = "exit"

    proposed_trade_intent = None
    if signal == "long" and pos_size > 0:
        proposed_trade_intent = TradeIntent(
            symbol=symbol,
            side="buy",
            quantity=pos_size,
            entry_price=last_p,
            stop_loss=round(last_p - stop_distance, 2) if stop_distance else None,
        max_holding_bars=exit_w,
            rationale=f"Trend pullback: price is above the {trend_w}-bar trend filter and momentum turned back up after a pullback.",
            source_signals=[
                f"above_{trend_w}_bar_trend_filter",
                f"pullback_near_{pullback_w}_bar_average",
                f"momentum_turn_{momentum_w}_bar_average",
                "atr_position_sizing",
            ],
        )

    buy_requirements = {
        f"Price above {trend_w}-bar trend filter": trend_ok,
        f"Pullback touched {pullback_w}-bar average": touched_pullback,
        f"Momentum turned up above {momentum_w}-bar average": momentum_turn,
        "Position size above zero": pos_size > 0,
    }
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif not trend_ok:
        no_trade_reason = f"No BUY because price is not above the rising {trend_w}-bar trend filter."
    elif not touched_pullback:
        no_trade_reason = f"No BUY because price has not pulled back near the {pullback_w}-bar average."
    elif not momentum_turn:
        no_trade_reason = f"No BUY because momentum has not turned back up above the {momentum_w}-bar average."
    elif pos_size <= 0:
        no_trade_reason = "No BUY because calculated share size is zero."
    else:
        no_trade_reason = "No BUY because the selected strategy rules are not fully met."

    live = _base_live_fields(
        prices=prices,
        smas=trend_smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        entry_level=pullback_level,
        exit_level=exit_level,
        stop_distance=stop_distance,
        balance=live_balance,
        pos_size=pos_size,
        signal=signal,
        trade_intent=proposed_trade_intent,
        strategy_name="Trend pullback continuation",
        setup_type="pullback",
        pullback_depth_pct=pullback_depth,
        momentum_turn=momentum_turn,
    )
    live["pullback_ready"] = setup_ready
    live["trend_ok"] = trend_ok
    live["touched_pullback"] = touched_pullback
    live["buy_requirements"] = buy_requirements
    live["no_trade_reason"] = no_trade_reason
    live["in_simulated_trade"] = in_trade
    live["exit_ready"] = strategy_exit_ready
    live["exit_reason"] = (
        f"Exit now because price is below the {exit_w}-bar exit average."
        if strategy_exit_ready
        else f"Hold because price is above the {exit_w}-bar exit average."
    )
    live["sell_requirements"] = {
        f"Price below {exit_w}-bar exit average": strategy_exit_ready,
    }

    final_equity = equity_curve[-1] if equity_curve else balance
    stats = _build_stats(account, final_equity, trade_log, equity_curve, exposure_bars, n_bars)
    return prices, trend_smas, atrs, trade_log, live, stats, labels


def simulate_trendline_breakout_strategy(
    account: float,
    trendline_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    seed: int | None = None,
    market_data=None,
    risk_limits: RiskLimits | None = None,
):
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    min_bars = max(trendline_w, exit_w, ma_w, 14) + 4
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, 14, highs, lows)
    smas = calc_sma(prices, ma_w)
    rsis = calc_rsi(prices, 14)
    trade_log = []
    equity_curve = [float(account)]
    exposure_bars = 0
    in_trade = False
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    start = max(trendline_w, exit_w, ma_w, 14)
    live_bar = n_bars - 1

    for i in range(start, live_bar):
        p = float(prices[i])
        atr = atrs[i]
        sma = smas[i]
        if atr is None or sma is None:
            equity_curve.append(balance)
            continue
        line = _recent_descending_trendline(prices, i, trendline_w)
        trendline_level = None
        if line is not None:
            intercept, slope, (x1, _) = line
            trendline_level = _trendline_value(intercept, slope, x1, i)
        prev_sma = smas[i - 1] if smas[i - 1] is not None else sma
        trend_ok = bool(p > sma and sma >= prev_sma)
        exit_level = float(np.min(prices[i - exit_w:i]))

        if not in_trade:
            if trendline_level is not None and p > trendline_level and trend_ok:
                stop = p - atr_mult * atr
                raw_size = int((balance * risk_pct_dec) / (p - stop)) if p > stop else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=p,
                    stop_price=stop,
                    session_pnl=balance - account,
                    limits=risk_limits,
                )
                if size > 0:
                    in_trade = True
                    entry_price = p
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            mark_to_market = balance + (p - entry_price) * shares
            high_value = float(highs[i]) if highs is not None else float(p)
            high_since_entry = max(high_since_entry, high_value)
            exit_rule, stop_price = _profit_protection_stop(
                entry_price=float(entry_price),
                initial_stop_price=float(initial_stop_price),
                current_stop_price=float(stop_price),
                current_price=float(p),
                current_atr=float(atr) if atr is not None else None,
                high_since_entry=float(high_since_entry),
                strategy_exit_price=float(exit_level),
            )
            if p <= stop_price:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=p,
                    shares=int(shares),
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                ))
                in_trade = False
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    live = _trendline_live_fields(
        prices=prices,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        lookback=trendline_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        balance=balance + ((float(prices[-1]) - entry_price) * shares if in_trade else 0.0),
        account=account,
        risk_pct_dec=risk_pct_dec,
        risk_limits=risk_limits,
        symbol=symbol,
        strategy_name="Trendline breakout",
        setup_type="trendline",
        require_retest=False,
    )
    live["in_simulated_trade"] = in_trade
    final_equity = equity_curve[-1] if equity_curve else balance
    stats = _build_stats(account, final_equity, trade_log, equity_curve, exposure_bars, n_bars)
    return prices, smas, atrs, trade_log, live, stats, labels


def simulate_trendline_retest_strategy(
    account: float,
    trendline_w: int,
    exit_w: int,
    atr_mult: float,
    risk_pct_dec: float,
    ma_w: int,
    momentum_w: int = 5,
    seed: int | None = None,
    market_data=None,
    risk_limits: RiskLimits | None = None,
):
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    min_bars = max(trendline_w, exit_w, ma_w, momentum_w, 14) + 4
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, 14, highs, lows)
    smas = calc_sma(prices, ma_w)
    momentum_smas = calc_sma(prices, momentum_w)
    rsis = calc_rsi(prices, 14)
    trade_log = []
    equity_curve = [float(account)]
    exposure_bars = 0
    in_trade = False
    waiting_retest = False
    retest_seen = False
    breakout_line: tuple[float, float, int] | None = None
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    start = max(trendline_w, exit_w, ma_w, momentum_w, 14)
    live_bar = n_bars - 1

    for i in range(start, live_bar):
        p = float(prices[i])
        atr = atrs[i]
        sma = smas[i]
        momentum_sma = momentum_smas[i]
        if atr is None or sma is None or momentum_sma is None:
            equity_curve.append(balance)
            continue
        line = _recent_descending_trendline(prices, i, trendline_w)
        prev_sma = smas[i - 1] if smas[i - 1] is not None else sma
        trend_ok = bool(p > sma and sma >= prev_sma)
        exit_level = float(np.min(prices[i - exit_w:i]))

        if not in_trade:
            if not waiting_retest and line is not None:
                intercept, slope, (x1, _) = line
                level = _trendline_value(intercept, slope, x1, i)
                if p > level and trend_ok:
                    waiting_retest = True
                    retest_seen = False
                    breakout_line = (intercept, slope, x1)
            elif waiting_retest and breakout_line is not None:
                intercept, slope, x1 = breakout_line
                level = _trendline_value(intercept, slope, x1, i)
                low_value = float(lows[i]) if lows is not None else p
                if low_value <= level + atr * 0.5 and p >= level:
                    retest_seen = True
                momentum_turn = bool(retest_seen and p > momentum_sma and p > float(prices[i - 1]) and trend_ok)
                if momentum_turn:
                    stop = min(level - atr * 0.25, p - atr_mult * atr)
                    raw_size = int((balance * risk_pct_dec) / (p - stop)) if p > stop else 0
                    size = _backtest_limited_quantity(
                        raw_quantity=raw_size,
                        account_equity=balance,
                        entry_price=p,
                        stop_price=stop,
                        session_pnl=balance - account,
                        limits=risk_limits,
                    )
                    if size > 0:
                        in_trade = True
                        waiting_retest = False
                        retest_seen = False
                        breakout_line = None
                        entry_price = p
                        stop_price = stop
                        initial_stop_price = stop
                        high_since_entry = float(p)
                        shares = size
                        entry_bar = i
        else:
            exposure_bars += 1
            mark_to_market = balance + (p - entry_price) * shares
            high_value = float(highs[i]) if highs is not None else float(p)
            high_since_entry = max(high_since_entry, high_value)
            exit_rule, stop_price = _profit_protection_stop(
                entry_price=float(entry_price),
                initial_stop_price=float(initial_stop_price),
                current_stop_price=float(stop_price),
                current_price=float(p),
                current_atr=float(atr) if atr is not None else None,
                high_since_entry=float(high_since_entry),
                strategy_exit_price=float(exit_level),
            )
            if p <= stop_price:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=p,
                    shares=int(shares),
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                ))
                in_trade = False
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    live = _trendline_live_fields(
        prices=prices,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        lookback=trendline_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        balance=balance + ((float(prices[-1]) - entry_price) * shares if in_trade else 0.0),
        account=account,
        risk_pct_dec=risk_pct_dec,
        risk_limits=risk_limits,
        symbol=symbol,
        strategy_name="Trendline retest continuation",
        setup_type="trendline_retest",
        require_retest=True,
        momentum_w=momentum_w,
    )
    live["in_simulated_trade"] = in_trade
    final_equity = equity_curve[-1] if equity_curve else balance
    stats = _build_stats(account, final_equity, trade_log, equity_curve, exposure_bars, n_bars)
    return prices, smas, atrs, trade_log, live, stats, labels


def _trendline_live_fields(
    *,
    prices,
    smas,
    atrs,
    rsis,
    volumes,
    index: int,
    lookback: int,
    exit_w: int,
    atr_mult: float,
    balance: float,
    account: float,
    risk_pct_dec: float,
    risk_limits: RiskLimits | None,
    symbol: str,
    strategy_name: str,
    setup_type: str,
    require_retest: bool,
    momentum_w: int = 5,
) -> dict:
    last_p = float(prices[index])
    last_atr = atrs[index]
    last_sma = smas[index]
    line = _recent_descending_trendline(prices, index, lookback)
    trendline_level = None
    trendline_slope = None
    if line is not None:
        intercept, slope, (x1, _) = line
        trendline_level = _trendline_value(intercept, slope, x1, index)
        trendline_slope = slope
    prev_sma = next((smas[i] for i in range(index - 1, -1, -1) if smas[i] is not None), last_sma)
    trend_ok = bool(last_sma and prev_sma and last_p > last_sma and last_sma >= prev_sma)
    trendline_break = bool(trendline_level is not None and last_p > trendline_level and trend_ok)
    retest_ready = False
    if require_retest and trendline_level is not None and last_atr:
        low_value = float(prices[index])
        retest_ready = bool(low_value <= trendline_level + last_atr * 0.5 and last_p >= trendline_level and trend_ok)
    entry_ready = retest_ready if require_retest else trendline_break
    stop_distance = atr_mult * last_atr if last_atr else 0
    raw_pos_size = int((balance * risk_pct_dec) / stop_distance) if stop_distance else 0
    stop_loss = last_p - stop_distance if stop_distance else last_p
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=balance,
        entry_price=last_p,
        stop_price=stop_loss,
        session_pnl=balance - account,
        limits=risk_limits,
    )
    signal = "long" if entry_ready else "flat"
    proposed_trade_intent = None
    if signal == "long" and pos_size > 0:
        proposed_trade_intent = TradeIntent(
            symbol=symbol,
            side="buy",
            quantity=pos_size,
            entry_price=last_p,
            stop_loss=round(stop_loss, 2) if stop_distance else None,
            max_holding_bars=exit_w,
            rationale=(
                f"{lookback}-bar descending trendline retest continued upward."
                if require_retest
                else f"Close broke above a descending trendline built from the last {lookback} bars."
            ),
            source_signals=[
                "descending_trendline_detected",
                "close_above_trendline" if not require_retest else "trendline_retest_continuation",
                f"sma_{lookback}_trend_context",
                "atr_position_sizing",
            ],
        )

    buy_requirements = {
        f"Descending trendline found in last {lookback} bars": trendline_level is not None,
        "Price above trendline": trendline_break,
        f"{lookback}-bar trendline slopes down": bool(trendline_slope is not None and trendline_slope < 0),
        "Retest held trendline": True if not require_retest else retest_ready,
        "Position size above zero": pos_size > 0,
    }
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif trendline_level is None:
        no_trade_reason = f"No BUY because a clean descending trendline was not found in the last {lookback} bars."
    elif not trendline_break:
        no_trade_reason = "No BUY because price has not broken above the descending trendline."
    elif require_retest and not retest_ready:
        no_trade_reason = "No BUY because price has not retested the broken trendline and turned back up."
    elif pos_size <= 0:
        no_trade_reason = "No BUY because calculated share size is zero."
    else:
        no_trade_reason = "No BUY because the selected strategy rules are not fully met."

    exit_level = float(np.min(prices[index - exit_w:index])) if index >= exit_w else last_p
    live = _base_live_fields(
        prices=prices,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=index,
        entry_level=trendline_level if trendline_level is not None else last_p,
        exit_level=exit_level,
        stop_distance=stop_distance,
        balance=balance,
        pos_size=pos_size,
        signal=signal,
        trade_intent=proposed_trade_intent,
        strategy_name=strategy_name,
        setup_type=setup_type,
        momentum_turn=retest_ready if require_retest else trendline_break,
    )
    live["trendline_level"] = trendline_level
    live["trendline_slope"] = trendline_slope
    live["trendline_break"] = trendline_break
    live["retest_ready"] = retest_ready
    live["buy_requirements"] = buy_requirements
    live["no_trade_reason"] = no_trade_reason
    live["exit_ready"] = bool(last_p <= exit_level)
    live["exit_reason"] = (
        f"Exit now because price is at or below the {exit_w}-bar exit level."
        if live["exit_ready"]
        else f"Hold because price is above the {exit_w}-bar exit level."
    )
    live["sell_requirements"] = {
        f"Price at or below {exit_w}-bar exit level": live["exit_ready"],
    }
    return live


def strategy_comparison_records(results: dict[str, dict]) -> list[dict]:
    rows = []
    for name, stats in results.items():
        rows.append({
            "Strategy": name,
            "Return": f"{stats.get('return_pct', 0)}%",
            "Trades": stats.get("total_trades", 0),
            "Win Rate": f"{stats.get('win_rate', 0)}%",
            "Worst Drop": f"{stats.get('max_drawdown_pct', 0)}%",
            "Profit Factor": stats.get("profit_factor", 0),
        })
    return rows
