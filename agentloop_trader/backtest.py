from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentloop_trader.assets import floor_quantity, normalize_asset_class
from agentloop_trader.data import generate_synthetic_prices
from agentloop_trader.fees import (
    estimate_alpaca_order_fees,
    estimate_alpaca_round_trip_fees,
    fee_adjusted_break_even_price,
)
from agentloop_trader.indicators import calc_atr, calc_rsi, calc_sma
from agentloop_trader.market_data import validate_price_bars
from agentloop_trader.models import BacktestResult, RiskLimits, StrategyConfig, TradeIntent
from agentloop_trader.performance import allocation_metrics, elapsed_years


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


def _max_drawdown_dollars(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, float(equity))
        worst = max(worst, peak - float(equity))
    return round(worst, 2)


def _build_stats(
    account: float,
    final_balance: float,
    trade_log: list[dict],
    equity_curve: list[float],
    exposure_bars: int,
    total_bars: int,
    risk_limits: RiskLimits | None = None,
    market_data=None,
    open_entry_fee: float = 0.0,
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
    rr = round(abs(avg_win / avg_loss), 2) if avg_loss else (99.0 if avg_win > 0 else 0)
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses else (99.0 if gross_wins > 0 else 0)
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
    period_years = (
        elapsed_years(market_data.index)
        if market_data is not None and hasattr(market_data, "index")
        else total_bars / 252.0 if total_bars > 0 else None
    )
    capital = allocation_metrics(
        account_equity=account,
        total_pnl=total_pnl,
        max_drawdown_dollars=_max_drawdown_dollars(equity_curve),
        risk_limits=risk_limits,
        years=period_years,
    )

    return {
        "final_equity": round(result.final_equity, 2),
        "total_pnl": round(result.total_pnl, 2),
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
        "estimated_alpaca_fees": round(
            sum(float(trade.get("estimated_alpaca_fees", 0.0)) for trade in trade_log) + float(open_entry_fee),
            2,
        ),
        "result": result,
        **capital,
    }


def _backtest_session_keys(market_data, labels: list[str], asset_class: str) -> list[str]:
    """Return the trading date used to reset historical daily-loss controls."""
    if market_data is None:
        return list(labels)
    timezone = "UTC" if asset_class == "crypto" else "America/New_York"
    try:
        index = market_data.index
        localized = index.tz_localize(timezone) if getattr(index, "tz", None) is None else index.tz_convert(timezone)
        return [timestamp.date().isoformat() for timestamp in localized]
    except (AttributeError, TypeError, ValueError):
        return [label[:10] if len(label) >= 10 and label[4:5] == "-" else label for label in labels]


@dataclass
class _BacktestSessionRisk:
    session_keys: list[str]
    current_session: str | None = None
    starting_equity: float | None = None
    session_pnl: float = 0.0

    def update(self, bar_index: int, account_equity: float) -> float:
        session = self.session_keys[bar_index]
        if session != self.current_session or self.starting_equity is None:
            self.current_session = session
            self.starting_equity = float(account_equity)
        self.session_pnl = float(account_equity) - self.starting_equity
        return self.session_pnl


def _backtest_limited_quantity(
    *,
    raw_quantity: float,
    account_equity: float,
    entry_price: float,
    stop_price: float,
    session_pnl: float,
    limits: RiskLimits | None,
    asset_class: str = "equity",
) -> float:
    if raw_quantity <= 0:
        return 0
    if limits is None:
        return raw_quantity
    if limits.kill_switch_enabled:
        return 0
    session_start_equity = max(0.0, account_equity - session_pnl)
    max_session_loss = session_start_equity * limits.max_session_loss_pct / 100
    if session_pnl < -max_session_loss:
        return 0
    if entry_price <= 0:
        return 0

    max_quantities = [raw_quantity, limits.max_quantity]
    risk_per_share = abs(entry_price - stop_price)
    max_risk_dollars = account_equity * limits.max_risk_per_trade_pct / 100
    risk_limited_quantity = floor_quantity(max_risk_dollars / risk_per_share, asset_class) if risk_per_share > 0 else 0
    while risk_limited_quantity > 0:
        buy_fees, sell_fees = estimate_alpaca_round_trip_fees(
            asset_class=asset_class,
            quantity=risk_limited_quantity,
            entry_price=entry_price,
            exit_price=stop_price,
        )
        total_stop_risk = risk_per_share * risk_limited_quantity + buy_fees.total + sell_fees.total
        if total_stop_risk <= max_risk_dollars:
            break
        risk_limited_quantity = floor_quantity(
            risk_limited_quantity - (0.0001 if asset_class == "crypto" else 1), asset_class,
        )
    max_quantities.append(risk_limited_quantity)

    max_position_notional = account_equity * limits.max_position_notional_pct / 100
    max_portfolio_notional = account_equity * limits.max_portfolio_exposure_pct / 100
    max_symbol_notional = account_equity * limits.max_symbol_concentration_pct / 100
    max_quantities.extend([
        floor_quantity(max_position_notional / entry_price, asset_class),
        floor_quantity(max_portfolio_notional / entry_price, asset_class),
        floor_quantity(max_symbol_notional / entry_price, asset_class),
        floor_quantity(max(0.0, account_equity) / entry_price, asset_class),
    ])
    return max(0, min(max_quantities))


def _backtest_fixed_notional_quantity(
    *,
    account_equity: float,
    entry_price: float,
    session_pnl: float,
    limits: RiskLimits | None,
    asset_class: str = "equity",
) -> float:
    """Size a no-stop research trade from capital allocation, not invented stop risk."""
    if account_equity <= 0 or entry_price <= 0:
        return 0
    if limits is None:
        allocation_pct = 5.0
        max_quantity = 100_000
    else:
        if limits.kill_switch_enabled:
            return 0
        session_start_equity = max(0.0, account_equity - session_pnl)
        max_session_loss = session_start_equity * limits.max_session_loss_pct / 100
        if session_pnl < -max_session_loss:
            return 0
        allocation_pct = min(
            limits.max_position_notional_pct,
            limits.max_portfolio_exposure_pct,
            limits.max_symbol_concentration_pct,
        )
        max_quantity = limits.max_quantity
    allocation_dollars = account_equity * allocation_pct / 100
    return max(0, min(
        floor_quantity(allocation_dollars / entry_price, asset_class),
        max_quantity,
        floor_quantity(account_equity / entry_price, asset_class),
    ))


def _trade_record(
    *,
    trade_number: int,
    symbol: str,
    labels: list[str],
    entry_bar: int,
    exit_bar: int,
    entry_price: float,
    exit_price: float,
    shares: float,
    stop_price: float,
    account: float,
    exit_rule: str = "Exit rule",
    lowest_price_since_entry: float | None = None,
    asset_class: str = "equity",
) -> dict:
    entry_price = round(float(entry_price), 2)
    exit_price = round(float(exit_price), 2)
    stop_price = round(float(stop_price), 2)
    gross_pnl = (exit_price - entry_price) * shares
    buy_fees, sell_fees = estimate_alpaca_round_trip_fees(
        asset_class=asset_class,
        quantity=shares,
        entry_price=entry_price,
        exit_price=exit_price,
    )
    estimated_fees = buy_fees.total + sell_fees.total
    pnl = gross_pnl - estimated_fees
    notional = entry_price * shares
    price_risk_dollars = abs(entry_price - stop_price) * shares
    risk_buy_fees, risk_sell_fees = estimate_alpaca_round_trip_fees(
        asset_class=asset_class,
        quantity=shares,
        entry_price=entry_price,
        exit_price=stop_price,
    )
    risk_dollars = price_risk_dollars + risk_buy_fees.total + risk_sell_fees.total
    adverse_price = min(entry_price, float(lowest_price_since_entry)) if lowest_price_since_entry is not None else entry_price
    max_adverse_pnl = (adverse_price - entry_price) * shares
    return {
        "trade": trade_number,
        "symbol": symbol,
        "entry_date": labels[entry_bar],
        "exit_date": labels[exit_bar],
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "entry": entry_price,
        "exit": exit_price,
        "shares": shares,
        "asset_class": asset_class,
        "notional": round(notional, 2),
        "risk_dollars": round(risk_dollars, 2),
        "price_risk_dollars": round(price_risk_dollars, 2),
        "risk_pct": round(risk_dollars / account * 100, 2) if account else 0,
        "max_adverse_pnl": round(max_adverse_pnl, 2),
        "max_adverse_pct": round(max_adverse_pnl / account * 100, 2) if account else 0,
        "stop": stop_price,
        "exit_rule": exit_rule,
        "gross_pnl": round(gross_pnl, 2),
        "estimated_buy_fees": buy_fees.total,
        "estimated_sell_fees": sell_fees.total,
        "estimated_alpaca_fees": round(estimated_fees, 2),
        "pnl": round(pnl, 2),
        "pct_acct": round(pnl / account * 100, 2) if account else 0,
    }


def _entry_fee(price: float, shares: float, asset_class: str = "equity") -> float:
    return estimate_alpaca_order_fees(
        asset_class=asset_class, side="buy", quantity=shares, price=price,
    ).total


def _exit_balance_change(entry_price: float, exit_price: float, shares: float, asset_class: str = "equity") -> float:
    gross_pnl = (round(float(exit_price), 2) - round(float(entry_price), 2)) * float(shares)
    sell_fee = estimate_alpaca_order_fees(
        asset_class=asset_class,
        side="sell",
        quantity=shares,
        price=round(float(exit_price), 2),
    ).total
    return gross_pnl - sell_fee


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
        highest_profit_r = (high_since_entry - entry_price) / initial_risk
        if highest_profit_r >= breakeven_after_r:
            candidates.append(("break-even stop", entry_price))
        if highest_profit_r >= trail_after_r and current_atr is not None:
            candidates.append(("ATR trail", high_since_entry - trailing_atr_multiplier * current_atr))
    return max(candidates, key=lambda item: item[1])


def _bar_open_prices(prices, market_data=None):
    if market_data is not None and "Open" in market_data.columns:
        clean = validate_price_bars(market_data, str(getattr(market_data, "attrs", {}).get("symbol", "MARKET")))
        return clean["Open"].to_numpy(dtype=float)
    return np.asarray(prices, dtype=float)


def _protective_stop_fill(open_price: float, low_price: float, stop_price: float) -> float | None:
    """Fill at the stop, or at the open when the bar gaps below it."""
    if low_price > stop_price:
        return None
    return float(open_price if open_price < stop_price else stop_price)


def _market_arrays(seed: int | None, market_data=None):
    if market_data is None:
        n_bars = 400
        prices = generate_synthetic_prices(n_bars, seed)
        highs = lows = volumes = None
        labels = [f"Day {i + 1}" for i in range(n_bars)]
        symbol = "SYNTH"
    else:
        market_data = validate_price_bars(market_data, str(getattr(market_data, "attrs", {}).get("symbol", "MARKET")))
        prices = market_data["Close"].to_numpy(dtype=float)
        highs = market_data["High"].to_numpy(dtype=float)
        lows = market_data["Low"].to_numpy(dtype=float)
        volumes = market_data["Volume"].to_numpy(dtype=float) if "Volume" in market_data else None
        n_bars = len(prices)
        has_intraday_times = any(
            getattr(timestamp, "hour", 0) or getattr(timestamp, "minute", 0)
            for timestamp in market_data.index[: min(len(market_data), 100)]
        )
        labels = market_data.index.strftime("%Y-%m-%d %H:%M" if has_intraday_times else "%Y-%m-%d").tolist()
        symbol = str(getattr(market_data, "attrs", {}).get("symbol", "MARKET"))
    return prices, highs, lows, volumes, n_bars, labels, symbol


def _trendline_crossed(prices, line: tuple[float, float, tuple[int, int]] | None, index: int) -> tuple[bool, float | None]:
    if line is None or index <= 0:
        return False, None
    intercept, slope, (x1, _) = line
    current_level = _trendline_value(intercept, slope, x1, index)
    previous_level = _trendline_value(intercept, slope, x1, index - 1)
    crossed = float(prices[index - 1]) <= previous_level and float(prices[index]) > current_level
    return bool(crossed), current_level


def _live_retest_context(
    prices,
    highs,
    lows,
    smas,
    atrs,
    momentum_smas,
    index: int,
    lookback: int,
) -> tuple[bool, float | None, float | None]:
    """Reconstruct a prior trendline breakout and later retest for the live bar."""
    source = highs if highs is not None else prices
    start = max(lookback, index - lookback)
    latest: tuple[int, float, float, int] | None = None
    for breakout_index in range(start, index):
        line = _recent_descending_trendline(source, breakout_index, lookback)
        crossed, _ = _trendline_crossed(prices, line, breakout_index)
        sma = smas[breakout_index]
        prev_sma = smas[breakout_index - 1] if breakout_index > 0 else sma
        trend_ok = bool(sma is not None and prev_sma is not None and prices[breakout_index] > sma and sma >= prev_sma)
        if crossed and trend_ok and line is not None:
            intercept, slope, (x1, _) = line
            latest = (breakout_index, intercept, slope, x1)
    if latest is None:
        return False, None, None
    breakout_index, intercept, slope, x1 = latest
    retest_seen = False
    for retest_index in range(breakout_index + 1, index + 1):
        atr = atrs[retest_index]
        if atr is None:
            continue
        level = _trendline_value(intercept, slope, x1, retest_index)
        low_value = float(lows[retest_index]) if lows is not None else float(prices[retest_index])
        if low_value <= level + float(atr) * 0.5 and float(prices[retest_index]) >= level:
            retest_seen = True
    current_level = _trendline_value(intercept, slope, x1, index)
    momentum_sma = momentum_smas[index]
    momentum_turn = bool(
        retest_seen
        and momentum_sma is not None
        and float(prices[index]) > float(momentum_sma)
        and float(prices[index]) > float(prices[index - 1])
    )
    return momentum_turn and float(prices[index]) >= current_level, current_level, slope


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


RSI_ENTRY_LENGTH = 14
RSI_ENTRY_MIN = 50.0
RSI_ENTRY_MAX = 70.0


def _rsi_entry_allowed(rsis, index: int, enabled: bool) -> bool:
    """Allow long entries only when 14-bar RSI is healthy but not extended."""
    if not enabled:
        return True
    value = rsis[index] if rsis and 0 <= index < len(rsis) else None
    return bool(value is not None and RSI_ENTRY_MIN <= float(value) <= RSI_ENTRY_MAX)


def _rsi_entry_requirement(rsis, index: int) -> tuple[str, bool]:
    value = rsis[index] if rsis and 0 <= index < len(rsis) else None
    label = (
        f"RSI({RSI_ENTRY_LENGTH}) between {RSI_ENTRY_MIN:.0f} and {RSI_ENTRY_MAX:.0f}"
        if value is None
        else f"RSI({RSI_ENTRY_LENGTH}) is {float(value):.1f}; required {RSI_ENTRY_MIN:.0f}-{RSI_ENTRY_MAX:.0f}"
    )
    return label, _rsi_entry_allowed(rsis, index, True)


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


def _rsi_scalp_live_setup(
    prices,
    rsis,
    index: int,
    *,
    oversold: float,
    decline_points: float,
    rebound_points: float,
    swing_lookback: int,
) -> tuple[bool, bool, float | None, float | None, float | None, float | None]:
    """Rebuild the current RSI setup from completed bars without carrying hidden state."""
    start = max(1, index - max(2, swing_lookback) + 1)
    armed = False
    setup_low: float | None = None
    recent_high: float | None = None
    decline: float | None = None
    rebound: float | None = None
    ready = False
    for i in range(start, index + 1):
        value = rsis[i]
        if value is None:
            continue
        value = float(value)
        lookback_values = [
            float(item)
            for item in rsis[max(0, i - swing_lookback + 1):i + 1]
            if item is not None
        ]
        recent_high = max(lookback_values) if lookback_values else value
        decline = recent_high - value
        if value <= oversold or decline >= decline_points:
            if not armed:
                armed = True
                setup_low = value
            else:
                setup_low = min(float(setup_low), value) if setup_low is not None else value
        if not armed or setup_low is None:
            continue
        setup_low = min(setup_low, value)
        rebound = value - setup_low
        confirmation = rebound >= rebound_points and float(prices[i]) > float(prices[i - 1])
        if confirmation:
            if i == index:
                ready = True
            else:
                armed = False
                setup_low = None
                rebound = None
    return ready, armed, setup_low, recent_high, decline, rebound


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
    rsi_entry_filter_enabled: bool = False,
):
    config = StrategyConfig(
        entry_window=entry_w,
        exit_window=exit_w,
        atr_stop_multiplier=atr_mult,
        risk_per_trade_pct=risk_pct_dec * 100,
        moving_average_window=ma_w,
    )

    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    asset_class = normalize_asset_class(getattr(market_data, "attrs", {}).get("asset_class"), symbol)
    session_risk = _BacktestSessionRisk(_backtest_session_keys(market_data, labels, asset_class))
    opens = _bar_open_prices(prices, market_data)
    entry_source = highs if highs is not None else prices
    exit_source = lows if lows is not None else prices

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
    low_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    open_entry_fee = 0.0
    start = max(entry_w, exit_w, ma_w)

    live_bar = n_bars - 1
    for i in range(start, live_bar):
        p = prices[i]
        current_equity = balance + ((float(p) - entry_price) * shares if in_trade else 0.0)
        session_pnl = session_risk.update(i, current_equity)
        sma = smas[i]
        atr = atrs[i]
        if sma is None or atr is None:
            equity_curve.append(balance)
            continue

        don_high = float(np.max(entry_source[i - entry_w:i]))
        don_low = float(np.min(exit_source[i - exit_w:i]))
        prior_sma = smas[i - 1] if smas[i - 1] is not None else sma
        ma_up = bool(p > sma and sma > prior_sma)

        if not in_trade:
            if p > don_high and ma_up and _rsi_entry_allowed(rsis, i, rsi_entry_filter_enabled):
                stop = round(float(p - atr_mult * atr), 2)
                risk = p - stop
                raw_size = floor_quantity((balance * risk_pct_dec) / risk, asset_class) if risk > 0 else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=float(p),
                    stop_price=float(stop),
                    session_pnl=session_pnl,
                    limits=risk_limits,
                    asset_class=asset_class,
                )
                if size > 0:
                    in_trade = True
                    entry_price = round(float(p), 2)
                    open_entry_fee = _entry_fee(entry_price, size, asset_class)
                    balance -= open_entry_fee
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    low_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            low_value = float(lows[i]) if lows is not None else float(p)
            protective_fill = _protective_stop_fill(float(opens[i]), low_value, float(stop_price))
            if protective_fill is not None:
                low_since_entry = min(low_since_entry, protective_fill)
                equity_curve.append(balance + (low_since_entry - entry_price) * shares)
                balance += _exit_balance_change(entry_price, protective_fill, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=protective_fill, shares=shares, stop_price=float(initial_stop_price),
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            low_since_entry = min(low_since_entry, low_value)
            equity_curve.append(balance + (low_since_entry - entry_price) * shares)
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
                balance += _exit_balance_change(entry_price, p, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=float(p),
                    shares=shares,
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                    lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue

        equity_curve.append(balance)

    last_p = float(prices[-1])
    last_atr = atrs[-1]
    last_sma = smas[-1]
    prev_sma = next((smas[i] for i in range(len(smas) - 2, -1, -1) if smas[i] is not None), last_sma)
    sma_up = bool(last_sma and prev_sma and last_p > last_sma and last_sma > prev_sma)
    dh_last = float(np.max(entry_source[-1 - entry_w:-1]))
    dl_last = float(np.min(exit_source[-1 - exit_w:-1]))
    open_position_value = (last_p - entry_price) * shares if in_trade else 0.0
    live_balance = balance + open_position_value
    live_session_pnl = session_risk.update(live_bar, live_balance)
    raw_pos_size = floor_quantity((balance * risk_pct_dec) / (atr_mult * last_atr), asset_class) if last_atr else 0
    live_stop = last_p - atr_mult * last_atr if last_atr else last_p
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=live_balance,
        entry_price=last_p,
        stop_price=live_stop,
        session_pnl=live_session_pnl,
        limits=risk_limits,
        asset_class=asset_class,
    )

    rsi_entry_ready = _rsi_entry_allowed(rsis, live_bar, rsi_entry_filter_enabled)
    entry_setup_ready = bool(last_p > dh_last and sma_up and rsi_entry_ready)
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
            asset_class=asset_class,
            time_in_force="gtc" if asset_class == "crypto" else "day",
            entry_price=last_p,
            stop_loss=round(last_p - atr_mult * last_atr, 2) if last_atr else None,
            max_holding_bars=exit_w,
            rationale=f"{entry_w}-bar breakout with rising {ma_w}-bar trend filter.",
            source_signals=[
                f"close_above_{entry_w}_bar_high",
                f"sma_{ma_w}_sloping_up",
                "atr_position_sizing",
            ] + (["rsi_14_between_50_and_70"] if rsi_entry_filter_enabled else []),
        )

    buy_requirements = {
        f"Price above {entry_w}-bar high": last_p > dh_last,
        f"Price above rising {ma_w}-bar trend filter": sma_up,
        "Position size above zero": pos_size > 0,
    }
    if rsi_entry_filter_enabled:
        rsi_label, rsi_passed = _rsi_entry_requirement(rsis, live_bar)
        buy_requirements[rsi_label] = rsi_passed
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif last_p <= dh_last:
        no_trade_reason = f"No BUY because price is not above the {entry_w}-bar entry level."
    elif not sma_up:
        no_trade_reason = f"No BUY because the {ma_w}-bar trend filter is not rising."
    elif not rsi_entry_ready:
        no_trade_reason = f"No BUY because RSI({RSI_ENTRY_LENGTH}) is outside the required {RSI_ENTRY_MIN:.0f}-{RSI_ENTRY_MAX:.0f} range."
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
    live.update({
        "entry_window": entry_w,
        "trend_window": ma_w,
        "trend_filter_level": last_sma,
        "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
    })
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

    if not equity_curve or equity_curve[-1] != live_balance:
        equity_curve.append(live_balance)
    final_equity = live_balance
    stats = _build_stats(
        account, final_equity, trade_log, equity_curve, exposure_bars, n_bars,
        risk_limits, market_data, open_entry_fee,
    )
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
    rsi_entry_filter_enabled: bool = False,
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
    asset_class = normalize_asset_class(getattr(market_data, "attrs", {}).get("asset_class"), symbol)
    session_risk = _BacktestSessionRisk(_backtest_session_keys(market_data, labels, asset_class))
    opens = _bar_open_prices(prices, market_data)
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
    low_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    open_entry_fee = 0.0
    start = max(pullback_w, exit_w, trend_w, momentum_w, config.atr_window)
    live_bar = n_bars - 1

    def setup_at(i: int) -> tuple[bool, float | None, bool, bool, bool, bool]:
        p = float(prices[i])
        trend_sma = trend_smas[i]
        pullback_sma = pullback_smas[i]
        momentum_sma = momentum_smas[i]
        prev_momentum_sma = momentum_smas[i - 1] if i > 0 else None
        if trend_sma is None or pullback_sma is None or momentum_sma is None or prev_momentum_sma is None:
            return False, None, False, False, False, False
        prev_trend = trend_smas[i - 1] if i > 0 else trend_sma
        trend_ok = bool(p > trend_sma and trend_sma >= (prev_trend or trend_sma))
        recent_low = float(np.min(prices[max(0, i - pullback_w):i + 1]))
        pullback_depth = (p - recent_low) / p * 100 if p else None
        touched_pullback = recent_low <= pullback_sma * 1.02
        momentum_turn = bool(p > momentum_sma and momentum_sma >= prev_momentum_sma and p > prices[i - 1])
        rsi_ok = _rsi_entry_allowed(rsis, i, rsi_entry_filter_enabled)
        return trend_ok and touched_pullback and momentum_turn and rsi_ok, pullback_depth, trend_ok, touched_pullback, momentum_turn, rsi_ok

    for i in range(start, live_bar):
        p = float(prices[i])
        current_equity = balance + ((p - entry_price) * shares if in_trade else 0.0)
        session_pnl = session_risk.update(i, current_equity)
        atr = atrs[i]
        if atr is None:
            equity_curve.append(balance)
            continue
        setup_ready, _, _, _, _, _ = setup_at(i)
        exit_sma = exit_smas[i]

        if not in_trade:
            if setup_ready:
                stop = round(min(float(np.min(prices[i - pullback_w:i + 1])), p - atr_mult * atr), 2)
                risk = p - stop
                raw_size = floor_quantity((balance * risk_pct_dec) / risk, asset_class) if risk > 0 else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=float(p),
                    stop_price=float(stop),
                    session_pnl=session_pnl,
                    limits=risk_limits,
                    asset_class=asset_class,
                )
                if size > 0:
                    in_trade = True
                    entry_price = round(float(p), 2)
                    open_entry_fee = _entry_fee(entry_price, size, asset_class)
                    balance -= open_entry_fee
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    low_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            low_value = float(lows[i]) if lows is not None else float(p)
            protective_fill = _protective_stop_fill(float(opens[i]), low_value, float(stop_price))
            if protective_fill is not None:
                low_since_entry = min(low_since_entry, protective_fill)
                equity_curve.append(balance + (low_since_entry - entry_price) * shares)
                balance += _exit_balance_change(entry_price, protective_fill, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=protective_fill, shares=shares, stop_price=float(initial_stop_price),
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            low_since_entry = min(low_since_entry, low_value)
            equity_curve.append(balance + (low_since_entry - entry_price) * shares)
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
                balance += _exit_balance_change(entry_price, p, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=float(p),
                    shares=shares,
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                    lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    last_p = float(prices[-1])
    last_atr = atrs[-1]
    pullback_level = float(pullback_smas[-1]) if pullback_smas[-1] is not None else last_p
    exit_level = float(exit_smas[-1]) if exit_smas[-1] is not None else last_p
    setup_ready, pullback_depth, trend_ok, touched_pullback, momentum_turn, rsi_entry_ready = setup_at(live_bar)
    open_position_value = (last_p - entry_price) * shares if in_trade else 0.0
    live_balance = balance + open_position_value
    live_session_pnl = session_risk.update(live_bar, live_balance)
    atr_stop = last_p - atr_mult * last_atr if last_atr else last_p
    recent_close_low = float(np.min(prices[-pullback_w:]))
    live_stop = min(recent_close_low, atr_stop)
    stop_distance = max(0.0, last_p - live_stop)
    raw_pos_size = floor_quantity((balance * risk_pct_dec) / stop_distance, asset_class) if stop_distance else 0
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=live_balance,
        entry_price=last_p,
        stop_price=live_stop,
        session_pnl=live_session_pnl,
        limits=risk_limits,
        asset_class=asset_class,
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
            asset_class=asset_class,
            time_in_force="gtc" if asset_class == "crypto" else "day",
            entry_price=last_p,
            stop_loss=round(live_stop, 2) if stop_distance else None,
            max_holding_bars=exit_w,
            rationale=f"Trend pullback: price is above the {trend_w}-bar trend filter and momentum turned back up after a pullback.",
            source_signals=[
                f"above_{trend_w}_bar_trend_filter",
                f"pullback_near_{pullback_w}_bar_average",
                f"momentum_turn_{momentum_w}_bar_average",
                "atr_position_sizing",
            ] + (["rsi_14_between_50_and_70"] if rsi_entry_filter_enabled else []),
        )

    buy_requirements = {
        f"Price above {trend_w}-bar trend filter": trend_ok,
        f"Pullback touched {pullback_w}-bar average": touched_pullback,
        f"Momentum turned up above {momentum_w}-bar average": momentum_turn,
        "Position size above zero": pos_size > 0,
    }
    if rsi_entry_filter_enabled:
        rsi_label, rsi_passed = _rsi_entry_requirement(rsis, live_bar)
        buy_requirements[rsi_label] = rsi_passed
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif not trend_ok:
        no_trade_reason = f"No BUY because price is not above the rising {trend_w}-bar trend filter."
    elif not touched_pullback:
        no_trade_reason = f"No BUY because price has not pulled back near the {pullback_w}-bar average."
    elif not momentum_turn:
        no_trade_reason = f"No BUY because momentum has not turned back up above the {momentum_w}-bar average."
    elif not rsi_entry_ready:
        no_trade_reason = f"No BUY because RSI({RSI_ENTRY_LENGTH}) is outside the required {RSI_ENTRY_MIN:.0f}-{RSI_ENTRY_MAX:.0f} range."
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
    live.update({
        "trend_window": trend_w,
        "trend_filter_level": trend_smas[live_bar],
        "pullback_window": pullback_w,
        "pullback_average_level": pullback_level,
        "pullback_zone_high": pullback_level * 1.02,
        "recent_pullback_low": recent_close_low,
        "momentum_window": momentum_w,
        "momentum_average_level": momentum_smas[live_bar],
        "prior_p": float(prices[live_bar - 1]),
        "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
    })
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

    if not equity_curve or equity_curve[-1] != live_balance:
        equity_curve.append(live_balance)
    final_equity = live_balance
    stats = _build_stats(
        account, final_equity, trade_log, equity_curve, exposure_bars, n_bars,
        risk_limits, market_data, open_entry_fee,
    )
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
    rsi_entry_filter_enabled: bool = False,
):
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    asset_class = normalize_asset_class(getattr(market_data, "attrs", {}).get("asset_class"), symbol)
    session_risk = _BacktestSessionRisk(_backtest_session_keys(market_data, labels, asset_class))
    opens = _bar_open_prices(prices, market_data)
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
    low_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    open_entry_fee = 0.0
    start = max(trendline_w, exit_w, ma_w, 14)
    live_bar = n_bars - 1

    for i in range(start, live_bar):
        p = float(prices[i])
        current_equity = balance + ((p - entry_price) * shares if in_trade else 0.0)
        session_pnl = session_risk.update(i, current_equity)
        atr = atrs[i]
        sma = smas[i]
        if atr is None or sma is None:
            equity_curve.append(balance)
            continue
        trendline_source = highs if highs is not None else prices
        line = _recent_descending_trendline(trendline_source, i, trendline_w)
        trendline_level = None
        crossed_trendline = False
        if line is not None:
            crossed_trendline, trendline_level = _trendline_crossed(prices, line, i)
        prev_sma = smas[i - 1] if smas[i - 1] is not None else sma
        trend_ok = bool(p > sma and sma >= prev_sma)
        exit_source = lows if lows is not None else prices
        exit_level = float(np.min(exit_source[i - exit_w:i]))

        if not in_trade:
            if trendline_level is not None and crossed_trendline and trend_ok and _rsi_entry_allowed(rsis, i, rsi_entry_filter_enabled):
                stop = round(float(p - atr_mult * atr), 2)
                raw_size = floor_quantity((balance * risk_pct_dec) / (p - stop), asset_class) if p > stop else 0
                size = _backtest_limited_quantity(
                    raw_quantity=raw_size,
                    account_equity=balance,
                    entry_price=p,
                    stop_price=stop,
                    session_pnl=session_pnl,
                    limits=risk_limits,
                    asset_class=asset_class,
                )
                if size > 0:
                    in_trade = True
                    entry_price = round(float(p), 2)
                    open_entry_fee = _entry_fee(entry_price, size, asset_class)
                    balance -= open_entry_fee
                    stop_price = stop
                    initial_stop_price = stop
                    high_since_entry = float(p)
                    low_since_entry = float(p)
                    shares = size
                    entry_bar = i
        else:
            exposure_bars += 1
            low_value = float(lows[i]) if lows is not None else float(p)
            protective_fill = _protective_stop_fill(float(opens[i]), low_value, float(stop_price))
            if protective_fill is not None:
                low_since_entry = min(low_since_entry, protective_fill)
                equity_curve.append(balance + (low_since_entry - entry_price) * shares)
                balance += _exit_balance_change(entry_price, protective_fill, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=protective_fill, shares=shares, stop_price=float(initial_stop_price),
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            low_since_entry = min(low_since_entry, low_value)
            equity_curve.append(balance + (low_since_entry - entry_price) * shares)
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
                balance += _exit_balance_change(entry_price, p, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=p,
                    shares=shares,
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                    lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    live_balance = balance + ((float(prices[-1]) - entry_price) * shares if in_trade else 0.0)
    live_session_pnl = session_risk.update(live_bar, live_balance)
    live = _trendline_live_fields(
        prices=prices,
        highs=highs,
        lows=lows,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        lookback=trendline_w,
        ma_w=ma_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        balance=live_balance,
        account=account,
        session_pnl=live_session_pnl,
        risk_pct_dec=risk_pct_dec,
        risk_limits=risk_limits,
        symbol=symbol,
        strategy_name="Trendline breakout",
        setup_type="trendline",
        require_retest=False,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
    live["in_simulated_trade"] = in_trade
    live_balance = float(live["balance"])
    if not equity_curve or equity_curve[-1] != live_balance:
        equity_curve.append(live_balance)
    final_equity = live_balance
    stats = _build_stats(
        account, final_equity, trade_log, equity_curve, exposure_bars, n_bars,
        risk_limits, market_data, open_entry_fee,
    )
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
    rsi_entry_filter_enabled: bool = False,
):
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    asset_class = normalize_asset_class(getattr(market_data, "attrs", {}).get("asset_class"), symbol)
    session_risk = _BacktestSessionRisk(_backtest_session_keys(market_data, labels, asset_class))
    opens = _bar_open_prices(prices, market_data)
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
    breakout_bar: int | None = None
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = 0.0
    low_since_entry = 0.0
    exit_rule = "Exit rule"
    balance = float(account)
    open_entry_fee = 0.0
    start = max(trendline_w, exit_w, ma_w, momentum_w, 14)
    live_bar = n_bars - 1

    for i in range(start, live_bar):
        p = float(prices[i])
        current_equity = balance + ((p - entry_price) * shares if in_trade else 0.0)
        session_pnl = session_risk.update(i, current_equity)
        atr = atrs[i]
        sma = smas[i]
        momentum_sma = momentum_smas[i]
        if atr is None or sma is None or momentum_sma is None:
            equity_curve.append(balance)
            continue
        trendline_source = highs if highs is not None else prices
        line = _recent_descending_trendline(trendline_source, i, trendline_w)
        prev_sma = smas[i - 1] if smas[i - 1] is not None else sma
        trend_ok = bool(p > sma and sma >= prev_sma)
        exit_source = lows if lows is not None else prices
        exit_level = float(np.min(exit_source[i - exit_w:i]))

        if not in_trade:
            if waiting_retest and breakout_bar is not None and i - breakout_bar > trendline_w:
                waiting_retest = False
                retest_seen = False
                breakout_line = None
                breakout_bar = None
            if not waiting_retest and line is not None:
                crossed_trendline, level = _trendline_crossed(prices, line, i)
                intercept, slope, (x1, _) = line
                if crossed_trendline and trend_ok:
                    waiting_retest = True
                    retest_seen = False
                    breakout_line = (intercept, slope, x1)
                    breakout_bar = i
            elif waiting_retest and breakout_line is not None:
                intercept, slope, x1 = breakout_line
                level = _trendline_value(intercept, slope, x1, i)
                low_value = float(lows[i]) if lows is not None else p
                if low_value <= level + atr * 0.5 and p >= level:
                    retest_seen = True
                momentum_turn = bool(retest_seen and p > momentum_sma and p > float(prices[i - 1]) and trend_ok)
                if momentum_turn and _rsi_entry_allowed(rsis, i, rsi_entry_filter_enabled):
                    stop = round(min(level - atr * 0.25, p - atr_mult * atr), 2)
                    raw_size = floor_quantity((balance * risk_pct_dec) / (p - stop), asset_class) if p > stop else 0
                    size = _backtest_limited_quantity(
                        raw_quantity=raw_size,
                        account_equity=balance,
                        entry_price=p,
                        stop_price=stop,
                        session_pnl=session_pnl,
                        limits=risk_limits,
                        asset_class=asset_class,
                    )
                    if size > 0:
                        in_trade = True
                        waiting_retest = False
                        retest_seen = False
                        breakout_line = None
                        breakout_bar = None
                        entry_price = round(float(p), 2)
                        open_entry_fee = _entry_fee(entry_price, size, asset_class)
                        balance -= open_entry_fee
                        stop_price = stop
                        initial_stop_price = stop
                        high_since_entry = float(p)
                        low_since_entry = float(p)
                        shares = size
                        entry_bar = i
        else:
            exposure_bars += 1
            low_value = float(lows[i]) if lows is not None else float(p)
            protective_fill = _protective_stop_fill(float(opens[i]), low_value, float(stop_price))
            if protective_fill is not None:
                low_since_entry = min(low_since_entry, protective_fill)
                equity_curve.append(balance + (low_since_entry - entry_price) * shares)
                balance += _exit_balance_change(entry_price, protective_fill, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=protective_fill, shares=shares, stop_price=float(initial_stop_price),
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            low_since_entry = min(low_since_entry, low_value)
            equity_curve.append(balance + (low_since_entry - entry_price) * shares)
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
                balance += _exit_balance_change(entry_price, p, shares, asset_class)
                trade_log.append(_trade_record(
                    trade_number=len(trade_log) + 1,
                    symbol=symbol,
                    labels=labels,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    entry_price=float(entry_price),
                    exit_price=p,
                    shares=shares,
                    stop_price=float(initial_stop_price),
                    account=float(account),
                    exit_rule=exit_rule,
                    lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                ))
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            equity_curve.append(mark_to_market)
            continue
        equity_curve.append(balance)

    live_balance = balance + ((float(prices[-1]) - entry_price) * shares if in_trade else 0.0)
    live_session_pnl = session_risk.update(live_bar, live_balance)
    live = _trendline_live_fields(
        prices=prices,
        highs=highs,
        lows=lows,
        smas=smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        lookback=trendline_w,
        ma_w=ma_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        balance=live_balance,
        account=account,
        session_pnl=live_session_pnl,
        risk_pct_dec=risk_pct_dec,
        risk_limits=risk_limits,
        symbol=symbol,
        strategy_name="Trendline retest continuation",
        setup_type="trendline_retest",
        require_retest=True,
        momentum_w=momentum_w,
        momentum_smas=momentum_smas,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
    live["in_simulated_trade"] = in_trade
    live_balance = float(live["balance"])
    if not equity_curve or equity_curve[-1] != live_balance:
        equity_curve.append(live_balance)
    final_equity = live_balance
    stats = _build_stats(
        account, final_equity, trade_log, equity_curve, exposure_bars, n_bars,
        risk_limits, market_data, open_entry_fee,
    )
    return prices, smas, atrs, trade_log, live, stats, labels


def _trendline_live_fields(
    *,
    prices,
    highs,
    lows,
    smas,
    atrs,
    rsis,
    volumes,
    index: int,
    lookback: int,
    ma_w: int,
    exit_w: int,
    atr_mult: float,
    balance: float,
    account: float,
    session_pnl: float,
    risk_pct_dec: float,
    risk_limits: RiskLimits | None,
    symbol: str,
    strategy_name: str,
    setup_type: str,
    require_retest: bool,
    momentum_w: int = 5,
    momentum_smas=None,
    rsi_entry_filter_enabled: bool = False,
) -> dict:
    asset_class = normalize_asset_class(symbol=symbol)
    last_p = float(prices[index])
    last_atr = atrs[index]
    last_sma = smas[index]
    trendline_source = highs if highs is not None else prices
    line = _recent_descending_trendline(trendline_source, index, lookback)
    trendline_level = None
    trendline_slope = None
    if line is not None:
        intercept, slope, (x1, _) = line
        trendline_level = _trendline_value(intercept, slope, x1, index)
        trendline_slope = slope
    prev_sma = next((smas[i] for i in range(index - 1, -1, -1) if smas[i] is not None), last_sma)
    trend_ok = bool(last_sma and prev_sma and last_p > last_sma and last_sma >= prev_sma)
    crossed_trendline, _ = _trendline_crossed(prices, line, index)
    trendline_break = bool(trendline_level is not None and crossed_trendline and trend_ok)
    retest_ready = False
    if require_retest and momentum_smas is not None:
        retest_ready, retest_level, retest_slope = _live_retest_context(
            prices, highs, lows, smas, atrs, momentum_smas, index, lookback
        )
        if retest_level is not None:
            trendline_level = retest_level
            trendline_slope = retest_slope
        retest_ready = bool(retest_ready and trend_ok)
    structural_entry_ready = retest_ready if require_retest else trendline_break
    rsi_entry_ready = _rsi_entry_allowed(rsis, index, rsi_entry_filter_enabled)
    entry_ready = bool(structural_entry_ready and rsi_entry_ready)
    atr_stop = last_p - atr_mult * last_atr if last_atr else last_p
    stop_loss = (
        min(float(trendline_level) - float(last_atr) * 0.25, atr_stop)
        if require_retest and trendline_level is not None and last_atr is not None
        else atr_stop
    )
    stop_distance = max(0.0, last_p - stop_loss)
    raw_pos_size = floor_quantity((balance * risk_pct_dec) / stop_distance, asset_class) if stop_distance else 0
    pos_size = _backtest_limited_quantity(
        raw_quantity=raw_pos_size,
        account_equity=balance,
        entry_price=last_p,
        stop_price=stop_loss,
        session_pnl=session_pnl,
        limits=risk_limits,
        asset_class=asset_class,
    )
    signal = "long" if entry_ready else "flat"
    proposed_trade_intent = None
    if signal == "long" and pos_size > 0:
        proposed_trade_intent = TradeIntent(
            symbol=symbol,
            side="buy",
            quantity=pos_size,
            asset_class=asset_class,
            time_in_force="gtc" if asset_class == "crypto" else "day",
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
                f"sma_{ma_w}_trend_context",
                "atr_position_sizing",
            ] + (["rsi_14_between_50_and_70"] if rsi_entry_filter_enabled else []),
        )

    buy_requirements = {
        f"Descending trendline found in last {lookback} bars": trendline_level is not None,
        "Price above trendline": trendline_break,
        f"{lookback}-bar trendline slopes down": bool(trendline_slope is not None and trendline_slope < 0),
        "Retest held trendline": True if not require_retest else retest_ready,
        "Position size above zero": pos_size > 0,
    }
    if rsi_entry_filter_enabled:
        rsi_label, rsi_passed = _rsi_entry_requirement(rsis, index)
        buy_requirements[rsi_label] = rsi_passed
    if proposed_trade_intent is not None:
        no_trade_reason = "BUY intent is present."
    elif trendline_level is None:
        no_trade_reason = f"No BUY because a clean descending trendline was not found in the last {lookback} bars."
    elif structural_entry_ready and not rsi_entry_ready:
        no_trade_reason = f"No BUY because RSI({RSI_ENTRY_LENGTH}) is outside the required {RSI_ENTRY_MIN:.0f}-{RSI_ENTRY_MAX:.0f} range."
    elif not require_retest and not trendline_break:
        no_trade_reason = "No BUY because price has not broken above the descending trendline."
    elif require_retest and not retest_ready:
        no_trade_reason = "No BUY because price has not retested the broken trendline and turned back up."
    elif pos_size <= 0:
        no_trade_reason = "No BUY because calculated share size is zero."
    else:
        no_trade_reason = "No BUY because the selected strategy rules are not fully met."

    exit_source = lows if lows is not None else prices
    exit_level = float(np.min(exit_source[index - exit_w:index])) if index >= exit_w else last_p
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
    live.update({
        "entry_window": lookback,
        "trend_window": ma_w,
        "trend_filter_level": last_sma,
        "trend_ok": trend_ok,
        "momentum_window": momentum_w,
        "momentum_average_level": momentum_smas[index] if momentum_smas is not None else None,
        "prior_p": float(prices[index - 1]),
        "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
        "retest_description": "Retest held" if retest_ready else "Retest and upward turn not complete",
    })
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


def simulate_rsi_mean_reversion_strategy(
    account: float,
    atr_mult: float,
    risk_pct_dec: float,
    rsi_length: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    rsi_decline_points: float = 40.0,
    rsi_rebound_points: float = 3.0,
    rsi_sell_recovery_points: float = 35.0,
    rsi_swing_lookback: int = 24,
    rsi_stop_mode: str = "standard_atr",
    rsi_emergency_atr_multiplier: float = 5.0,
    rsi_max_holding_enabled: bool = True,
    rsi_max_holding_bars: int = 100,
    rsi_profit_only_exit: bool = False,
    seed: int | None = None,
    market_data=None,
    risk_limits: RiskLimits | None = None,
):
    """Long-only RSI mean-reversion scalp with selectable price protection."""
    if rsi_stop_mode not in {"standard_atr", "emergency_atr", "no_price_stop"}:
        raise ValueError(f"Unknown RSI stop mode: {rsi_stop_mode}")
    active_stop_multiplier = (
        atr_mult if rsi_stop_mode == "standard_atr"
        else rsi_emergency_atr_multiplier if rsi_stop_mode == "emergency_atr"
        else None
    )
    prices, highs, lows, volumes, n_bars, labels, symbol = _market_arrays(seed, market_data)
    asset_class = normalize_asset_class(getattr(market_data, "attrs", {}).get("asset_class"), symbol)
    session_risk = _BacktestSessionRisk(_backtest_session_keys(market_data, labels, asset_class))
    opens = _bar_open_prices(prices, market_data)
    min_bars = max(14, rsi_length, rsi_swing_lookback) + 3
    if n_bars < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for these settings; got {n_bars}.")

    atrs = calc_atr(prices, 14, highs, lows)
    rsis = calc_rsi(prices, rsi_length)
    display_smas = calc_sma(prices, max(2, rsi_length))
    trade_log: list[dict] = []
    equity_curve = [float(account)]
    exposure_bars = 0
    balance = float(account)
    open_entry_fee = 0.0
    in_trade = False
    armed = False
    armed_bar = -1
    setup_low: float | None = None
    entry_setup_low: float | None = None
    entry_rsi: float | None = None
    entry_price = stop_price = initial_stop_price = shares = entry_bar = 0
    high_since_entry = low_since_entry = 0.0
    exit_rule = "Exit rule"
    start = max(14, rsi_length, rsi_swing_lookback)
    live_bar = n_bars - 1

    for i in range(start, live_bar):
        price = float(prices[i])
        current_equity = balance + ((price - entry_price) * shares if in_trade else 0.0)
        session_pnl = session_risk.update(i, current_equity)
        atr = atrs[i]
        rsi = rsis[i]
        if atr is None or rsi is None:
            equity_curve.append(balance)
            continue
        rsi = float(rsi)
        recent_values = [
            float(value)
            for value in rsis[max(0, i - rsi_swing_lookback + 1):i + 1]
            if value is not None
        ]
        recent_high = max(recent_values) if recent_values else rsi
        decline = recent_high - rsi

        if not in_trade:
            if armed and i - armed_bar > rsi_swing_lookback:
                armed = False
                setup_low = None
            if rsi <= rsi_oversold or decline >= rsi_decline_points:
                if not armed:
                    armed = True
                    armed_bar = i
                    setup_low = rsi
                else:
                    setup_low = min(float(setup_low), rsi) if setup_low is not None else rsi
            if armed and setup_low is not None:
                setup_low = min(setup_low, rsi)
                rebound = rsi - setup_low
                price_turned_up = price > float(prices[i - 1])
                if rebound >= rsi_rebound_points and price_turned_up:
                    stop = (
                        round(price - active_stop_multiplier * float(atr), 2)
                        if active_stop_multiplier is not None
                        else None
                    )
                    if stop is None:
                        size = _backtest_fixed_notional_quantity(
                            account_equity=balance,
                            entry_price=price,
                            session_pnl=session_pnl,
                            limits=risk_limits,
                            asset_class=asset_class,
                        )
                    else:
                        risk = price - stop
                        raw_size = floor_quantity((balance * risk_pct_dec) / risk, asset_class) if risk > 0 else 0
                        size = _backtest_limited_quantity(
                            raw_quantity=raw_size,
                            account_equity=balance,
                            entry_price=price,
                            stop_price=stop,
                            session_pnl=session_pnl,
                            limits=risk_limits,
                            asset_class=asset_class,
                        )
                    if size > 0:
                        in_trade = True
                        entry_price = round(price, 2)
                        open_entry_fee = _entry_fee(entry_price, size, asset_class)
                        balance -= open_entry_fee
                        stop_price = initial_stop_price = stop
                        shares = size
                        entry_bar = i
                        high_since_entry = low_since_entry = price
                        entry_setup_low = setup_low
                        entry_rsi = rsi
                        armed = False
                        setup_low = None
        else:
            exposure_bars += 1
            low_value = float(lows[i]) if lows is not None else price
            protective_fill = (
                _protective_stop_fill(float(opens[i]), low_value, float(stop_price))
                if stop_price is not None
                else None
            )
            if protective_fill is not None:
                low_since_entry = min(low_since_entry, protective_fill)
                equity_curve.append(balance + (low_since_entry - entry_price) * shares)
                balance += _exit_balance_change(entry_price, protective_fill, shares, asset_class)
                record = _trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=protective_fill, shares=shares, stop_price=float(initial_stop_price),
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                )
                record.update({"entry_rsi": entry_rsi, "entry_rsi_setup_low": entry_setup_low, "exit_rsi": rsi})
                trade_log.append(record)
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue

            low_since_entry = min(low_since_entry, low_value)
            equity_curve.append(balance + (low_since_entry - entry_price) * shares)
            high_value = float(highs[i]) if highs is not None else price
            high_since_entry = max(high_since_entry, high_value)
            if stop_price is not None and initial_stop_price is not None:
                exit_rule, stop_price = _profit_protection_stop(
                    entry_price=float(entry_price),
                    initial_stop_price=float(initial_stop_price),
                    current_stop_price=float(stop_price),
                    current_price=price,
                    current_atr=float(atr),
                    high_since_entry=float(high_since_entry),
                    strategy_exit_price=None,
                )
            rsi_sell_level = min(rsi_overbought, float(entry_setup_low or rsi_oversold) + rsi_sell_recovery_points)
            rsi_exit_signal = rsi >= rsi_sell_level
            fee_break_even = fee_adjusted_break_even_price(
                asset_class=asset_class,
                quantity=shares,
                entry_price=entry_price,
            )
            rsi_profit_ready = not rsi_profit_only_exit or price > fee_break_even
            rsi_exit = rsi_exit_signal and rsi_profit_ready
            time_exit = bool(rsi_max_holding_enabled and i - entry_bar >= rsi_max_holding_bars)
            price_exit = bool(stop_price is not None and price <= stop_price)
            if rsi_exit or time_exit or price_exit:
                if rsi_exit:
                    exit_rule = f"RSI recovered to {rsi_sell_level:.1f}"
                elif time_exit:
                    exit_rule = f"Maximum hold of {rsi_max_holding_bars} bars"
                balance += _exit_balance_change(entry_price, price, shares, asset_class)
                record = _trade_record(
                    trade_number=len(trade_log) + 1, symbol=symbol, labels=labels,
                    entry_bar=entry_bar, exit_bar=i, entry_price=float(entry_price),
                    exit_price=price, shares=shares,
                    stop_price=float(initial_stop_price) if initial_stop_price is not None else 0.0,
                    account=float(account), exit_rule=exit_rule, lowest_price_since_entry=low_since_entry,
                    asset_class=asset_class,
                )
                record.update({"entry_rsi": entry_rsi, "entry_rsi_setup_low": entry_setup_low, "exit_rsi": rsi})
                trade_log.append(record)
                in_trade = False
                open_entry_fee = 0.0
                equity_curve.append(balance)
                continue
            equity_curve.append(balance + (price - entry_price) * shares)
            continue
        equity_curve.append(balance)

    last_price = float(prices[live_bar])
    last_atr = atrs[live_bar]
    last_rsi = float(rsis[live_bar]) if rsis[live_bar] is not None else None
    setup_ready, setup_armed, live_setup_low, recent_high, decline, rebound = _rsi_scalp_live_setup(
        prices, rsis, live_bar,
        oversold=rsi_oversold,
        decline_points=rsi_decline_points,
        rebound_points=rsi_rebound_points,
        swing_lookback=rsi_swing_lookback,
    )
    open_value = (last_price - entry_price) * shares if in_trade else 0.0
    live_balance = balance + open_value
    live_session_pnl = session_risk.update(live_bar, live_balance)
    live_stop = (
        last_price - active_stop_multiplier * float(last_atr)
        if last_atr is not None and active_stop_multiplier is not None
        else None
    )
    stop_distance = max(0.0, last_price - live_stop) if live_stop is not None else 0.0
    if live_stop is None:
        pos_size = _backtest_fixed_notional_quantity(
            account_equity=live_balance,
            entry_price=last_price,
            session_pnl=live_session_pnl,
            limits=risk_limits,
            asset_class=asset_class,
        )
    else:
        raw_size = floor_quantity((balance * risk_pct_dec) / stop_distance, asset_class) if stop_distance else 0
        pos_size = _backtest_limited_quantity(
            raw_quantity=raw_size,
            account_equity=live_balance,
            entry_price=last_price,
            stop_price=live_stop,
            session_pnl=live_session_pnl,
            limits=risk_limits,
            asset_class=asset_class,
        )
    proposed_trade_intent = None
    if setup_ready and pos_size > 0:
        proposed_trade_intent = TradeIntent(
            symbol=symbol,
            side="buy",
            quantity=pos_size,
            asset_class=asset_class,
            time_in_force="gtc" if asset_class == "crypto" else "day",
            entry_price=last_price,
            stop_loss=round(live_stop, 2) if live_stop is not None else None,
            max_holding_bars=rsi_max_holding_bars if rsi_max_holding_enabled else None,
            rationale=(
                f"RSI mean-reversion scalp: RSI rebounded {float(rebound or 0):.1f} points "
                f"from a setup low of {float(live_setup_low or 0):.1f}, with price turning up."
            ),
            source_signals=[
                f"rsi_{rsi_length}_setup_armed",
                f"rsi_rebound_{rsi_rebound_points:g}_points",
                "close_above_prior_bar",
                "fixed_notional_sizing" if live_stop is None else "atr_position_sizing",
            ],
        )

    saved_sell_level = min(rsi_overbought, float(entry_setup_low or rsi_oversold) + rsi_sell_recovery_points)
    simulated_rsi_exit_signal = bool(in_trade and last_rsi is not None and last_rsi >= saved_sell_level)
    live_fee_break_even = (
        fee_adjusted_break_even_price(
            asset_class=asset_class,
            quantity=shares,
            entry_price=entry_price,
        )
        if in_trade
        else None
    )
    simulated_rsi_profit_ready = bool(
        not rsi_profit_only_exit
        or (live_fee_break_even is not None and last_price > live_fee_break_even)
    )
    simulated_rsi_exit = simulated_rsi_exit_signal and simulated_rsi_profit_ready
    simulated_time_exit = bool(
        rsi_max_holding_enabled and in_trade and live_bar - entry_bar >= rsi_max_holding_bars
    )
    signal = "long" if proposed_trade_intent is not None else "exit" if simulated_rsi_exit or simulated_time_exit else "flat"
    live = _base_live_fields(
        prices=prices,
        smas=display_smas,
        atrs=atrs,
        rsis=rsis,
        volumes=volumes,
        index=live_bar,
        entry_level=last_price,
        exit_level=last_price,
        stop_distance=stop_distance,
        balance=live_balance,
        pos_size=pos_size,
        signal=signal,
        trade_intent=proposed_trade_intent,
        strategy_name="RSI mean-reversion scalp",
        setup_type="rsi_scalp",
        momentum_turn=setup_ready,
    )
    live.update({
        "entry_level": None,
        "exit_level": None,
        "rsi_setup_armed": setup_armed,
        "rsi_setup_low": live_setup_low,
        "rsi_recent_high": recent_high,
        "rsi_decline_points": decline,
        "rsi_rebound_points": rebound,
        "required_rsi_rebound_points": rsi_rebound_points,
        "prior_p": float(prices[live_bar - 1]),
        "rsi_sell_level": min(rsi_overbought, float(live_setup_low or rsi_oversold) + rsi_sell_recovery_points),
        "rsi_length": rsi_length,
        "rsi_oversold": rsi_oversold,
        "required_rsi_decline_points": rsi_decline_points,
        "rsi_stop_mode": rsi_stop_mode,
        "rsi_emergency_atr_multiplier": rsi_emergency_atr_multiplier,
        "rsi_profit_only_exit": rsi_profit_only_exit,
        "rsi_fee_adjusted_break_even": live_fee_break_even,
        "rsi_profit_condition_ready": simulated_rsi_profit_ready,
        "buy_requirements": {
            f"RSI({rsi_length}) reached {rsi_oversold:g} or fell {rsi_decline_points:g} points": setup_armed or setup_ready,
            f"RSI rebounded {rsi_rebound_points:g} points from the setup low": bool(rebound is not None and rebound >= rsi_rebound_points),
            "Price closed above the prior completed bar": last_price > float(prices[live_bar - 1]),
            "Position size above zero": pos_size > 0,
        },
        "sell_requirements": {
            f"RSI reached setup low + {rsi_sell_recovery_points:g} points, capped at {rsi_overbought:g}": simulated_rsi_exit_signal,
            (
                "Completed-bar close is above estimated fee-adjusted break-even"
                if rsi_profit_only_exit
                else "Profit-only RSI exit is off"
            ): simulated_rsi_profit_ready,
            f"Position reached maximum hold of {rsi_max_holding_bars} bars": simulated_time_exit,
            "Price protection reached": False,
        },
        "in_simulated_trade": in_trade,
        "exit_ready": simulated_rsi_exit or simulated_time_exit,
        "exit_reason": (
            f"Exit now because RSI reached {saved_sell_level:.1f}."
            if simulated_rsi_exit
            else (
                f"Hold. RSI reached {saved_sell_level:.1f}, but ${last_price:,.2f} is not above the estimated "
                f"fee-adjusted break-even price of ${live_fee_break_even:,.2f}. Check again after the next completed bar."
            )
            if simulated_rsi_exit_signal and rsi_profit_only_exit and live_fee_break_even is not None
            else f"Exit now because the {rsi_max_holding_bars}-bar maximum hold was reached."
            if simulated_time_exit
            else f"Hold until RSI reaches the saved recovery level, price protection is hit, or {rsi_max_holding_bars} bars pass."
            if rsi_max_holding_enabled
            else "Hold until RSI reaches the saved recovery level or price protection is hit."
        ),
        "no_trade_reason": (
            "BUY intent is present."
            if proposed_trade_intent is not None
            else f"No BUY because RSI has not rebounded {rsi_rebound_points:g} points from an armed setup low with price turning up."
        ),
    })
    if not equity_curve or equity_curve[-1] != live_balance:
        equity_curve.append(live_balance)
    stats = _build_stats(
        account, live_balance, trade_log, equity_curve, exposure_bars, n_bars,
        risk_limits, market_data, open_entry_fee,
    )
    return prices, display_smas, atrs, trade_log, live, stats, labels


def strategy_comparison_records(results: dict[str, dict]) -> list[dict]:
    rows = []
    for name, stats in results.items():
        rows.append({
            "Strategy": name,
            "Allocated Return": f"{stats.get('allocated_return_pct', stats.get('return_pct', 0))}%",
            "Annualized Return": (
                "Not shown (period is 1 year or less)"
                if stats.get("annualized_allocated_return_pct") is None
                else f"{stats['annualized_allocated_return_pct']}%"
            ),
            "Account Return": f"{stats.get('return_pct', 0)}%",
            "Trades": stats.get("total_trades", 0),
            "Win Rate": f"{stats.get('win_rate', 0)}%",
            "Allocated Worst Drop": f"{stats.get('allocated_max_drawdown_pct', stats.get('max_drawdown_pct', 0))}%",
            "Profit Factor": stats.get("profit_factor", 0),
        })
    return rows
