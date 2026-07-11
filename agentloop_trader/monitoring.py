from __future__ import annotations

from dataclasses import dataclass

from agentloop_trader.execution import PaperBroker
from agentloop_trader.models import RiskLimits


@dataclass(frozen=True)
class MonitoringResult:
    status: str
    alerts: list[str]
    metrics: dict[str, float | int | bool]

    @property
    def has_breach(self) -> bool:
        return self.status == "BREACH"


def monitor_paper_session(
    broker: PaperBroker,
    starting_cash: float,
    account_equity: float,
    limits: RiskLimits,
) -> MonitoringResult:
    exposure = sum(position.market_value for position in broker.positions.values())
    paper_equity = broker.cash + exposure
    session_pnl = paper_equity - starting_cash
    exposure_pct = exposure / account_equity * 100 if account_equity else 0.0
    session_loss_pct = abs(session_pnl) / account_equity * 100 if account_equity and session_pnl < 0 else 0.0

    alerts = []
    status = "OK"
    if limits.kill_switch_enabled:
        status = "BREACH"
        alerts.append("Kill Switch is on.")
    if session_loss_pct > limits.max_session_loss_pct:
        status = "BREACH"
        alerts.append(
            f"Daily loss {session_loss_pct:.2f}% exceeds max {limits.max_session_loss_pct:.2f}%."
        )
    if exposure_pct > limits.max_portfolio_exposure_pct:
        status = "BREACH"
        alerts.append(
            f"Portfolio exposure {exposure_pct:.2f}% exceeds max {limits.max_portfolio_exposure_pct:.2f}%."
        )
    if len(broker.positions) > limits.max_open_positions:
        status = "BREACH"
        alerts.append(
            f"Open positions {len(broker.positions)} exceeds max {limits.max_open_positions}."
        )

    if status == "OK" and (exposure_pct > limits.max_portfolio_exposure_pct * 0.8 or session_loss_pct > limits.max_session_loss_pct * 0.8):
        status = "WARN"
        alerts.append("The account is approaching a configured risk limit.")

    return MonitoringResult(
        status=status,
        alerts=alerts or ["The account is within configured monitoring limits."],
        metrics={
            "paper_cash": round(broker.cash, 2),
            "paper_equity": round(paper_equity, 2),
            "session_pnl": round(session_pnl, 2),
            "session_loss_pct": round(session_loss_pct, 2),
            "portfolio_exposure": round(exposure, 2),
            "portfolio_exposure_pct": round(exposure_pct, 2),
            "open_positions": len(broker.positions),
            "kill_switch_enabled": limits.kill_switch_enabled,
        },
    )


def monitoring_records(result: MonitoringResult) -> list[dict]:
    return [
        {"Metric": key.replace("_", " ").title(), "Value": value}
        for key, value in result.metrics.items()
    ]


def daily_risk_records(
    local_order_records: list[dict],
    tracked_alpaca_orders: list[dict],
    account_equity: float,
    session_pnl: float,
    portfolio_exposure: float,
    limits: RiskLimits,
) -> list[dict]:
    local_filled = [order for order in local_order_records if str(order.get("Status", "")).lower() == "filled"]
    alpaca_submitted = [
        order for order in tracked_alpaca_orders
        if _enum_value(order.get("status", "")) in {"accepted", "new", "pending_new", "partially_filled", "filled"}
    ]
    local_notional = sum(_as_float(order.get("Notional")) for order in local_filled)
    alpaca_quantity = sum(_as_float(order.get("filled_quantity")) for order in alpaca_submitted)
    session_loss_pct = abs(session_pnl) / account_equity * 100 if account_equity and session_pnl < 0 else 0.0
    exposure_pct = portfolio_exposure / account_equity * 100 if account_equity else 0.0
    return [
        {"Metric": "Filled app paper orders", "Value": len(local_filled)},
        {"Metric": "Saved Alpaca active/filled orders", "Value": len(alpaca_submitted)},
        {"Metric": "Filled app order value", "Value": round(local_notional, 2)},
        {"Metric": "Filled Alpaca shares", "Value": _format_number(alpaca_quantity)},
        {"Metric": "Portfolio exposure", "Value": round(portfolio_exposure, 2)},
        {"Metric": "Portfolio exposure %", "Value": round(exposure_pct, 2)},
        {"Metric": "Max portfolio exposure %", "Value": limits.max_portfolio_exposure_pct},
        {"Metric": "Daily P&L", "Value": round(session_pnl, 2)},
        {"Metric": "Daily loss %", "Value": round(session_loss_pct, 2)},
        {"Metric": "Max daily loss %", "Value": limits.max_session_loss_pct},
        {"Metric": "Kill Switch on", "Value": limits.kill_switch_enabled},
        {"Metric": "Max open positions", "Value": limits.max_open_positions},
    ]


def broker_heartbeat_records(
    broker_connected: bool,
    broker_state_stale: bool,
    broker_reasons: list[str],
    market_advisory: dict,
) -> list[dict]:
    return [
        {
            "Check": "Alpaca connected",
            "Passed": broker_connected,
            "Policy": "Alpaca must be connected before paper orders can be sent.",
            "Detail": "Connected." if broker_connected else "Alpaca account is disconnected.",
        },
        {
            "Check": "Alpaca data current",
            "Passed": not broker_state_stale,
            "Policy": "Positions and orders must be refreshed before buy, cancel, exit, or automation checks.",
            "Detail": "; ".join(broker_reasons) if broker_reasons else "Positions and orders are available.",
        },
        {
            "Check": "Market open",
            "Passed": bool(market_advisory.get("Open", False)),
            "Policy": "Closed-market paper orders may be accepted but may not fill until the market opens.",
            "Detail": market_advisory.get("Message", ""),
        },
    ]


def risk_halt_records(
    monitoring_result: MonitoringResult,
    broker_connected: bool,
    broker_state_stale: bool,
    automation_ready_rows: list[dict],
) -> list[dict]:
    rows = []
    if monitoring_result.status == "BREACH":
        for alert in monitoring_result.alerts:
            rows.append({"Block": "Risk limit hit", "Active": True, "Detail": alert})
    if not broker_connected:
        rows.append({"Block": "Alpaca disconnected", "Active": True, "Detail": "Alpaca account data is unavailable."})
    if broker_state_stale:
        rows.append({"Block": "Refresh Alpaca", "Active": True, "Detail": "Refresh Alpaca positions and orders."})
    for row in automation_ready_rows:
        if row.get("Check") in {"Alpaca data current", "Kill Switch off"} and not row.get("Passed"):
            rows.append({"Block": row.get("Check", ""), "Active": True, "Detail": row.get("Detail", "")})
    if not rows:
        rows.append({"Block": "None", "Active": False, "Detail": "No active blocks."})
    return rows


def _enum_value(value) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")
