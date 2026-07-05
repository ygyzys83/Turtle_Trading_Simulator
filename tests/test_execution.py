from agentloop_trader.execution import PaperBroker
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import check_trade_intent, decide_execution


def _approved_buy_intent():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_position_notional_pct=25),
    )
    return intent, decide_execution("paper", risk)


def test_paper_broker_fills_approved_buy_order_and_updates_cash_position():
    broker = PaperBroker(cash=50_000)
    intent, decision = _approved_buy_intent()

    order = broker.submit_order(intent, decision)

    assert order.status == "filled"
    assert broker.cash == 49_000
    assert broker.positions["AAPL"].quantity == 10
    assert broker.positions["AAPL"].average_price == 100


def test_paper_broker_blocks_when_execution_decision_blocks():
    broker = PaperBroker(cash=50_000)
    intent, _ = _approved_buy_intent()
    risk = check_trade_intent(intent, account_equity=50_000, limits=RiskLimits(allowed_symbols=("AAPL",)))
    decision = decide_execution("backtest_only", risk)

    order = broker.submit_order(intent, decision)

    assert order.status == "blocked"
    assert broker.cash == 50_000
    assert broker.positions == {}


def test_paper_broker_rejects_order_with_insufficient_cash():
    broker = PaperBroker(cash=500)
    intent, decision = _approved_buy_intent()

    order = broker.submit_order(intent, decision)

    assert order.status == "rejected"
    assert order.message == "Insufficient paper cash."
    assert broker.cash == 500
    assert broker.positions == {}

