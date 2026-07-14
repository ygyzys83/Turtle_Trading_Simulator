from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from agentloop_trader.models import PACIFIC_TIME


DEFAULT_BUY_WATCHLIST_PATH = Path("automation_logs") / "buy_watchlist.json"
MAX_BUY_WATCHLIST_ITEMS = 10


@dataclass(frozen=True)
class BuyWatchPlan:
    plan_id: str
    symbol: str
    interval: str
    history: str
    price_data_source: str
    strategy_label: str
    asset_class: str = "equity"
    strategy_settings: dict[str, Any] = field(default_factory=dict)
    risk_limits: dict[str, Any] = field(default_factory=dict)
    order_style: str = "Limit at current price"
    limit_adjustment_pct: float = 0.0
    custom_limit_price: float = 0.0
    repeat_after_exit: bool = False
    enabled: bool = True
    status: str = "Waiting for BUY"
    detail: str = "Waiting for the saved strategy's required BUY rules."
    cycle_state: str = "waiting_for_buy"
    last_cycle_completed_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_checked_at: str = ""
    order_sent_at: str = ""
    latest_price: float | None = None
    next_buy_level: float | None = None
    distance_to_buy_pct: float | None = None
    buy_requirement_levels: list[dict[str, Any]] = field(default_factory=list)


def buy_watch_plan_id(symbol: str, interval: str, strategy_label: str, asset_class: str = "equity") -> str:
    identity = f"{asset_class.strip().lower()}|{symbol.strip().upper()}|{interval.strip().lower()}|{strategy_label.strip()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


class BuyWatchlistStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_BUY_WATCHLIST_PATH

    def read(self) -> list[BuyWatchPlan]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        allowed = {field.name for field in BuyWatchPlan.__dataclass_fields__.values()}
        rows = payload if isinstance(payload, list) else []
        return [
            BuyWatchPlan(**{key: value for key, value in row.items() if key in allowed})
            for row in rows
            if isinstance(row, dict) and row.get("plan_id")
        ]

    def replace_all(self, plans: list[BuyWatchPlan]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(plan) for plan in plans], indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def upsert(self, plan: BuyWatchPlan) -> BuyWatchPlan:
        plans = self.read()
        existing = next((row for row in plans if row.plan_id == plan.plan_id), None)
        if existing is None and len(plans) >= MAX_BUY_WATCHLIST_ITEMS:
            raise ValueError(f"Buy watchlist is limited to {MAX_BUY_WATCHLIST_ITEMS} setups.")
        now = datetime.now(PACIFIC_TIME).isoformat()
        saved = replace(
            plan,
            symbol=plan.symbol.strip().upper(),
            created_at=existing.created_at if existing and existing.created_at else (plan.created_at or now),
            updated_at=now,
            last_checked_at="" if existing is None else existing.last_checked_at,
            order_sent_at="",
        )
        updated = [saved if row.plan_id == saved.plan_id else row for row in plans]
        if existing is None:
            updated.append(saved)
        self.replace_all(updated)
        return saved

    def update(self, plan_id: str, **changes: Any) -> BuyWatchPlan | None:
        plans = self.read()
        updated_plan = None
        updated = []
        for plan in plans:
            if plan.plan_id == plan_id:
                updated_plan = replace(plan, **changes, updated_at=datetime.now(PACIFIC_TIME).isoformat())
                updated.append(updated_plan)
            else:
                updated.append(plan)
        if updated_plan is not None:
            self.replace_all(updated)
        return updated_plan

    def remove(self, plan_id: str) -> bool:
        plans = self.read()
        updated = [plan for plan in plans if plan.plan_id != plan_id]
        if len(updated) == len(plans):
            return False
        self.replace_all(updated)
        return True


def buy_watchlist_records(plans: list[BuyWatchPlan]) -> list[dict[str, Any]]:
    return [
        {
            "Ticker": plan.symbol,
            "Asset Type": plan.asset_class.title(),
            "Interval": plan.interval,
            "Strategy": plan.strategy_label,
            "Repeat After Exit": "On" if plan.repeat_after_exit else "Off",
            "Enabled": "Yes" if plan.enabled else "No",
            "Status": plan.status,
            "Current Price": f"${plan.latest_price:,.2f}" if plan.latest_price is not None else "Not checked",
            "Next BUY Level": f"${plan.next_buy_level:,.2f}" if plan.next_buy_level is not None else "Pattern / indicator rules",
            "Distance To BUY": f"{plan.distance_to_buy_pct:+.2f}%" if plan.distance_to_buy_pct is not None else "Depends on rule",
            "Last checked": plan.last_checked_at or "Not checked yet",
            "Plain English": plan.detail,
        }
        for plan in plans
    ]


def buy_watch_plan_detail_records(plan: BuyWatchPlan) -> list[dict[str, str]]:
    """Return every saved input that can affect a queued BUY or its initial exit plan."""
    settings = dict(plan.strategy_settings or {})
    limits = dict(plan.risk_limits or {})

    def yes_no(value: Any) -> str:
        return "On" if bool(value) else "Off"

    def number(key: str, suffix: str = "", default: Any = "Not saved") -> str:
        value = settings.get(key, default)
        return f"{value}{suffix}" if value != "Not saved" else str(value)

    def risk_number(key: str, suffix: str = "", default: Any = "Not saved") -> str:
        value = limits.get(key, default)
        return f"{value}{suffix}" if value != "Not saved" else str(value)

    rows = [
        {"Area": "Setup", "Input": "Ticker", "Saved Value": plan.symbol},
        {"Area": "Setup", "Input": "Asset type", "Saved Value": plan.asset_class.title()},
        {"Area": "Setup", "Input": "Price source", "Saved Value": plan.price_data_source},
        {"Area": "Setup", "Input": "Price interval", "Saved Value": plan.interval},
        {"Area": "Setup", "Input": "History period", "Saved Value": plan.history},
        {"Area": "Setup", "Input": "Strategy", "Saved Value": plan.strategy_label},
        {"Area": "Strategy", "Input": "Buy breakout / trendline lookback", "Saved Value": number("entry_window", " bars")},
        {"Area": "Strategy", "Input": "Sell exit length", "Saved Value": number("exit_window", " bars")},
        {"Area": "Strategy", "Input": "Stop distance", "Saved Value": number("atr_stop_multiplier", " ATR")},
        {"Area": "Strategy", "Input": "Strategy risk per trade", "Saved Value": number("risk_per_trade_pct", "%")},
        {"Area": "Strategy", "Input": "Trend filter length", "Saved Value": number("moving_average_window", " bars")},
        {"Area": "Strategy", "Input": "Pullback average length", "Saved Value": number("pullback_average_length", " bars")},
        {"Area": "Strategy", "Input": "Momentum turn length", "Saved Value": number("momentum_turn_length", " bars")},
        {"Area": "Strategy", "Input": "RSI 50-70 BUY rule", "Saved Value": yes_no(settings.get("rsi_entry_filter_enabled", False))},
        {"Area": "RSI scalp", "Input": "RSI length", "Saved Value": number("rsi_length", " bars")},
        {"Area": "RSI scalp", "Input": "Arm at or below RSI", "Saved Value": number("rsi_oversold")},
        {"Area": "RSI scalp", "Input": "Arm after RSI decline", "Saved Value": number("rsi_decline_points", " points")},
        {"Area": "RSI scalp", "Input": "Buy after RSI rebound", "Saved Value": number("rsi_rebound_points", " points")},
        {"Area": "RSI scalp", "Input": "Sell after RSI recovery", "Saved Value": number("rsi_sell_recovery_points", " points")},
        {"Area": "RSI scalp", "Input": "RSI sell cap", "Saved Value": number("rsi_overbought")},
        {
            "Area": "RSI scalp",
            "Input": "Require profit for RSI exit",
            "Saved Value": "On" if settings.get("rsi_profit_only_exit", False) else "Off",
        },
        {"Area": "RSI scalp", "Input": "RSI swing lookback", "Saved Value": number("rsi_swing_lookback", " bars")},
        {"Area": "RSI scalp", "Input": "Stop protection", "Saved Value": str(settings.get("rsi_stop_mode", "standard_atr")).replace("_", " ").title()},
        {
            "Area": "RSI scalp",
            "Input": "Emergency stop distance",
            "Saved Value": (
                number("rsi_emergency_atr_multiplier", " ATR")
                if settings.get("rsi_stop_mode") == "emergency_atr"
                else "Not used"
            ),
        },
        {
            "Area": "RSI scalp",
            "Input": "Maximum holding period",
            "Saved Value": (
                number("rsi_max_holding_bars", " bars")
                if settings.get("rsi_max_holding_enabled", True)
                else "Off"
            ),
        },
        {"Area": "Risk", "Input": "Allowed symbols", "Saved Value": ", ".join(limits.get("allowed_symbols") or ()) or "Any"},
        {"Area": "Risk", "Input": "Max risk per trade", "Saved Value": risk_number("max_risk_per_trade_pct", "%")},
        {"Area": "Risk", "Input": "Max new order size", "Saved Value": risk_number("max_position_notional_pct", "%")},
        {"Area": "Risk", "Input": "Max portfolio exposure", "Saved Value": risk_number("max_portfolio_exposure_pct", "%")},
        {"Area": "Risk", "Input": "Max symbol concentration", "Saved Value": risk_number("max_symbol_concentration_pct", "%")},
        {"Area": "Risk", "Input": "Max daily loss", "Saved Value": risk_number("max_session_loss_pct", "%")},
        {"Area": "Risk", "Input": "Max open positions", "Saved Value": risk_number("max_open_positions")},
        {"Area": "Risk", "Input": "Max share quantity", "Saved Value": risk_number("max_quantity")},
        {"Area": "Risk", "Input": "Allow adding to existing position", "Saved Value": yes_no(limits.get("allow_add_to_existing_position", False))},
        {"Area": "Risk", "Input": "Require stop loss", "Saved Value": yes_no(limits.get("require_stop_loss", True))},
        {"Area": "Order", "Input": "Paper buy price", "Saved Value": plan.order_style},
    ]
    if plan.order_style == "Limit below current price":
        rows.append({"Area": "Order", "Input": "Buy limit discount", "Saved Value": f"{plan.limit_adjustment_pct}%"})
    elif plan.order_style == "Limit above current price":
        rows.append({"Area": "Order", "Input": "Buy limit cushion", "Saved Value": f"{plan.limit_adjustment_pct}%"})
    elif plan.order_style == "Custom limit price":
        rows.append({"Area": "Order", "Input": "Custom limit price", "Saved Value": f"${plan.custom_limit_price:,.2f}"})
    rows.extend([
        {"Area": "Order", "Input": "Auto-cancel old limit buys", "Saved Value": yes_no(settings.get("auto_cancel_stale_limit_orders", False))},
        {"Area": "Order", "Input": "Cancel unfilled limit buy after", "Saved Value": number("stale_limit_order_minutes", " minutes")},
        {"Area": "Order", "Input": "Allow limit buys outside market hours", "Saved Value": yes_no(settings.get("allow_limit_buys_outside_market_hours", False))},
        {"Area": "Automation", "Input": "Check automation every", "Saved Value": number("automation_refresh_seconds", " seconds")},
        {"Area": "Automation", "Input": "Repeat after exit", "Saved Value": yes_no(plan.repeat_after_exit)},
        {"Area": "Automation", "Input": "Max automatic buys this session", "Saved Value": number("max_auto_buys_per_session")},
        {"Area": "Automation", "Input": "Wait after an exit before re-buying", "Saved Value": number("reentry_cooldown_minutes", " minutes")},
        {"Area": "Automation", "Input": "Order sizing account", "Saved Value": "Current Alpaca paper account at execution"},
        {"Area": "Automation", "Input": "Order reference price", "Saved Value": "Latest available Alpaca IEX trade at execution"},
        {"Area": "Initial exit plan", "Input": "Move stop to break-even after", "Saved Value": number("breakeven_after_r", "R")},
        {"Area": "Initial exit plan", "Input": "Start ATR trail after", "Saved Value": number("trail_after_r", "R")},
        {"Area": "Initial exit plan", "Input": "Trailing ATR distance", "Saved Value": number("trailing_atr_multiplier", " ATR")},
        {"Area": "Initial exit plan", "Input": "Confirm exit on bar close", "Saved Value": yes_no(settings.get("confirm_exit_on_bar_close", True))},
    ])
    return rows
