import os
import tempfile
from types import SimpleNamespace

from agentloop_trader.brokers import (
    AlpacaBrokerAdapterStub,
    AlpacaConfig,
    PaperBrokerAdapter,
    alpaca_config_validation_records,
    build_alpaca_cancel_preview,
    alpaca_cancel_preview_records,
    alpaca_tracked_order_from_broker_order,
    alpaca_tracked_order_records,
    build_alpaca_order_preview,
    broker_status_records,
)
from agentloop_trader.execution import PaperBroker
from agentloop_trader.models import ExecutionDecision, RiskCheckResult, RiskLimits, TradeIntent
from agentloop_trader.risk import check_trade_intent, decide_execution


class FakeAlpacaClient:
    def __init__(self, order_status="filled"):
        self.submitted_orders = []
        self.canceled_orders = []
        self.order_status = order_status

    def get_account(self):
        return SimpleNamespace(
            status="ACTIVE",
            cash="100000",
            buying_power="200000",
            portfolio_value="100000",
            equity="100000",
            currency="USD",
        )

    def get_all_positions(self):
        return [
            SimpleNamespace(
                symbol="AAPL",
                qty="10",
                market_value="1000",
                avg_entry_price="100",
                unrealized_pl="0",
            )
        ]

    def get_orders(self):
        return [
            SimpleNamespace(
                id="abcdef123456",
                symbol="AAPL",
                side="buy",
                qty="10",
                status="filled",
                submitted_at="2026-01-01T14:30:00Z",
                filled_at="2026-01-01T14:31:00Z",
                filled_qty="10",
                filled_avg_price="100.50",
            )
        ]

    def get_order_by_id(self, order_id):
        return SimpleNamespace(
            id=order_id,
            symbol="AAPL",
            side="buy",
            qty="10",
            status=self.order_status,
            submitted_at="2026-01-01T14:30:00Z",
            filled_at="2026-01-01T14:31:00Z" if self.order_status == "filled" else "",
            filled_qty="10" if self.order_status == "filled" else "0",
            filled_avg_price="100.50" if self.order_status == "filled" else "",
        )

    def submit_order(self, order_data):
        self.submitted_orders.append(order_data)
        return SimpleNamespace(id="paper-order-1", status="accepted", symbol="AAPL")

    def cancel_order_by_id(self, order_id):
        self.canceled_orders.append(order_id)
        return SimpleNamespace(id=order_id, status="canceled")


def _order_field(order, field):
    if isinstance(order, dict):
        return order.get(field)
    return getattr(order, field, None)


def _approved_intent_and_decision():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    return intent, decide_execution("paper", risk)


def test_paper_broker_adapter_reports_ready_and_submits_order():
    adapter = PaperBrokerAdapter(PaperBroker(cash=50_000))
    intent, decision = _approved_intent_and_decision()

    status = adapter.status()
    order = adapter.submit_order(intent, decision)

    assert status.connected
    assert status.can_submit_orders
    assert order.status == "filled"
    assert adapter.position_records()[0]["Symbol"] == "AAPL"


def test_alpaca_adapter_reports_missing_credentials_and_blocks_orders():
    adapter = AlpacaBrokerAdapterStub(AlpacaConfig(api_key=None, api_secret=None))
    status = adapter.status()

    assert not status.connected
    assert not status.can_submit_orders
    assert "credentials not configured" in status.message


def test_alpaca_read_only_adapter_reports_fake_account_data():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", base_url="https://paper-api.alpaca.markets"),
        trading_client=FakeAlpacaClient(),
    )

    status = adapter.status()

    assert status.connected
    assert not status.can_submit_orders
    assert "Alpaca account connected" in status.message
    assert adapter.account_records()[0]["Field"] == "Status"
    assert adapter.position_records()[0]["Symbol"] == "AAPL"
    assert adapter.order_records()[0]["Order ID"] == "abcdef12"
    assert adapter.order_records()[0]["Filled Qty"] == "10"


def test_alpaca_adapter_blocks_submission_when_manual_gate_disabled():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret"),
        trading_client=FakeAlpacaClient(),
        allow_order_submission=False,
    )
    intent, decision = _approved_intent_and_decision()

    try:
        adapter.submit_order(intent, decision)
    except RuntimeError as exc:
        assert "turned off in the sidebar" in str(exc)
        return
    raise AssertionError("Expected Alpaca paper adapter to block order submission.")


def test_alpaca_adapter_submits_paper_order_when_all_gates_enabled():
    client = FakeAlpacaClient()
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=client,
        allow_order_submission=True,
    )
    intent, decision = _approved_intent_and_decision()

    preview = build_alpaca_order_preview(intent, decision, adapter.config)
    order = adapter.submit_order(intent, decision, expected_preview_hash=preview.preview_hash)

    assert order.status == "accepted"
    assert client.submitted_orders


def test_alpaca_adapter_submits_limit_order_when_intent_uses_limit_price():
    client = FakeAlpacaClient()
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=client,
        allow_order_submission=True,
    )
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, order_type="limit", limit_price=101.25, entry_price=101.25, stop_loss=96)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    decision = decide_execution("paper", risk)

    preview = build_alpaca_order_preview(intent, decision, adapter.config)
    order = adapter.submit_order(intent, decision, expected_preview_hash=preview.preview_hash)

    assert order.status == "accepted"
    assert preview.order["order_type"] == "limit"
    assert preview.order["limit_price"] == 101.25
    submitted = client.submitted_orders[0]
    assert _order_field(submitted, "limit_price") == 101.25


def test_alpaca_adapter_submits_paper_exit_when_all_gates_enabled():
    client = FakeAlpacaClient()
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=client,
        allow_order_submission=True,
    )
    intent = TradeIntent(symbol="AAPL", side="sell", quantity=10)
    decision = ExecutionDecision(
        mode="paper",
        approved_for_execution=True,
        requires_manual_approval=False,
        reason="Exit preview approved.",
        risk_check=RiskCheckResult(approved=True, rejected_reasons=[], checks={"exit": True}),
    )
    preview = build_alpaca_order_preview(intent, decision, adapter.config)

    order = adapter.submit_order(intent, decision, expected_preview_hash=preview.preview_hash)

    assert order.status == "accepted"
    assert client.submitted_orders


def test_alpaca_preview_blocks_synthetic_symbols():
    intent = TradeIntent(symbol="SYNTH", side="buy", quantity=10, entry_price=100, stop_loss=95)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("SYNTH",)))
    decision = decide_execution("paper", risk)

    preview = build_alpaca_order_preview(intent, decision, AlpacaConfig(api_key="key", api_secret="secret"))

    assert not preview.valid
    assert "Synthetic symbols cannot be sent to Alpaca." in preview.blocked_reasons


def test_alpaca_adapter_blocks_stale_preview_hash():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=FakeAlpacaClient(),
        allow_order_submission=True,
    )
    intent, decision = _approved_intent_and_decision()

    try:
        adapter.submit_order(intent, decision, expected_preview_hash="stale")
    except RuntimeError as exc:
        assert "preview changed" in str(exc)
        return
    raise AssertionError("Expected stale preview hash to block Alpaca paper submission.")


def test_alpaca_tracked_order_records_include_lifecycle_fields():
    order = SimpleNamespace(
        id="order-123456",
        symbol="AAPL",
        side="buy",
        qty="40",
        type="limit",
        limit_price="211.50",
        status="filled",
        submitted_at="2026-07-05T16:37:41Z",
        filled_at="2026-07-05T16:37:42Z",
        filled_qty="40",
        filled_avg_price="212.34",
    )

    tracked = alpaca_tracked_order_from_broker_order(order, preview_hash="hash123")
    record = alpaca_tracked_order_records([tracked])[0]

    assert record["Order ID"] == "order-12"
    assert record["Alpaca Order ID"] == "order-123456"
    assert record["Review ID"] == "hash123"
    assert record["Order Type"] == "limit"
    assert record["Limit Price"] == "211.50"
    assert record["Status"] == "filled"
    assert record["Filled Qty"] == "40"
    assert record["Avg Fill"] == "212.34"


def test_alpaca_adapter_refreshes_tracked_order_by_id():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=FakeAlpacaClient(),
    )

    record = adapter.tracked_order_record("order-123456", preview_hash="hash123")

    assert record["Order ID"] == "order-12"
    assert record["Alpaca Order ID"] == "order-123456"
    assert record["Review ID"] == "hash123"
    assert record["Status"] == "filled"


def test_alpaca_adapter_refreshes_tracked_order_records_by_id():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=FakeAlpacaClient(order_status="canceled"),
    )

    records = adapter.refreshed_tracked_order_records([
        {"broker_order_id": "order-123456", "preview_hash": "hash123"}
    ])

    assert records[0]["Alpaca Order ID"] == "order-123456"
    assert records[0]["Review ID"] == "hash123"
    assert records[0]["Status"] == "canceled"


def test_alpaca_order_records_request_all_statuses_when_supported():
    class FilterClient(FakeAlpacaClient):
        def __init__(self):
            super().__init__()
            self.received_filter = None

        def get_orders(self, filter=None):
            self.received_filter = filter
            return super().get_orders()

    client = FilterClient()
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=client,
    )
    request = object()
    adapter._build_all_orders_request = lambda: request

    records = adapter.order_records()

    assert client.received_filter is request
    assert records[0]["Alpaca Order ID"] == "abcdef123456"


def test_alpaca_cancel_preview_blocks_closed_orders():
    record = {
        "Broker Order ID": "order-123456",
        "Symbol": "AAPL",
        "Side": "buy",
        "Quantity": "10",
        "Status": "filled",
    }

    preview = build_alpaca_cancel_preview(record, AlpacaConfig(api_key="key", api_secret="secret", paper=True))
    rows = alpaca_cancel_preview_records(preview)

    assert not preview.valid
    assert "Only open Alpaca paper orders can be cancelled" in preview.blocked_reasons[0]
    assert rows[-1]["Field"] == "Ready To Cancel"


def test_alpaca_cancel_preview_blocks_live_mode():
    record = {
        "Broker Order ID": "order-123456",
        "Symbol": "AAPL",
        "Side": "buy",
        "Quantity": "10",
        "Status": "accepted",
    }

    preview = build_alpaca_cancel_preview(record, AlpacaConfig(api_key="key", api_secret="secret", paper=False))

    assert not preview.valid
    assert "Alpaca live order cancellation is blocked. Use paper mode only." in preview.blocked_reasons


def test_alpaca_cancel_preview_hash_canonicalizes_sdk_enum_values():
    config = AlpacaConfig(api_key="key", api_secret="secret", paper=True)
    ui_record = {
        "Broker Order ID": "order-123456",
        "Symbol": "AAPL",
        "Side": "buy",
        "Quantity": "40",
        "Status": "accepted",
    }
    refreshed_record = {
        "Broker Order ID": "order-123456",
        "Symbol": "AAPL",
        "Side": "OrderSide.BUY",
        "Quantity": "40.0",
        "Status": "OrderStatus.ACCEPTED",
    }

    ui_preview = build_alpaca_cancel_preview(ui_record, config)
    refreshed_preview = build_alpaca_cancel_preview(refreshed_record, config)

    assert ui_preview.preview_hash == refreshed_preview.preview_hash
    assert refreshed_preview.cancel["side"] == "buy"
    assert refreshed_preview.cancel["status"] == "accepted"
    assert refreshed_preview.cancel["quantity"] == "40"


def test_alpaca_adapter_blocks_stale_cancel_preview_hash():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=FakeAlpacaClient(order_status="accepted"),
        allow_order_submission=True,
    )

    try:
        adapter.cancel_order("order-123456", expected_cancel_hash="stale")
    except RuntimeError as exc:
        assert "cancel preview changed" in str(exc)
        return
    raise AssertionError("Expected stale cancel preview hash to block Alpaca paper cancellation.")


def test_alpaca_adapter_cancels_paper_order_when_all_gates_enabled():
    client = FakeAlpacaClient(order_status="accepted")
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        trading_client=client,
        allow_order_submission=True,
    )
    order_record = adapter.tracked_order_record("order-123456", preview_hash="")
    preview = build_alpaca_cancel_preview(order_record, adapter.config)

    result = adapter.cancel_order("order-123456", expected_cancel_hash=preview.preview_hash)

    assert result.status == "canceled"
    assert client.canceled_orders == ["order-123456"]


def test_alpaca_adapter_blocks_live_cancel_even_with_gate_enabled():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=False),
        trading_client=FakeAlpacaClient(order_status="accepted"),
        allow_order_submission=True,
    )

    try:
        adapter.cancel_order("order-123456")
    except RuntimeError as exc:
        assert "live order cancellation is blocked" in str(exc)
        return
    raise AssertionError("Expected live Alpaca order cancellation to be blocked.")


def test_alpaca_adapter_blocks_live_order_submission_even_with_gate_enabled():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=False),
        trading_client=FakeAlpacaClient(),
        allow_order_submission=True,
    )
    intent, decision = _approved_intent_and_decision()

    try:
        adapter.submit_order(intent, decision)
    except RuntimeError as exc:
        assert "live order submission is blocked" in str(exc)
        return
    raise AssertionError("Expected live Alpaca order submission to be blocked.")


def test_alpaca_config_can_load_environment_variable_names():
    old_key = os.environ.get("APCA_API_KEY_ID")
    old_secret = os.environ.get("APCA_API_SECRET_KEY")
    try:
        os.environ["APCA_API_KEY_ID"] = "key"
        os.environ["APCA_API_SECRET_KEY"] = "secret"
        config = AlpacaConfig.from_env()
    finally:
        if old_key is None:
            os.environ.pop("APCA_API_KEY_ID", None)
        else:
            os.environ["APCA_API_KEY_ID"] = old_key
        if old_secret is None:
            os.environ.pop("APCA_API_SECRET_KEY", None)
        else:
            os.environ["APCA_API_SECRET_KEY"] = old_secret

    assert config.has_credentials


def test_alpaca_config_loads_dotenv_file_without_overriding_existing_env():
    old_cwd = os.getcwd()
    env_names = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL", "ALPACA_PAPER"]
    old_env = {name: os.environ.get(name) for name in env_names}
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            with open(".env", "w", encoding="utf-8") as env_file:
                env_file.write(
                    "\n".join(
                        [
                            "APCA_API_KEY_ID=dotenv_key",
                            "APCA_API_SECRET_KEY=dotenv_secret",
                            "APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2",
                            "ALPACA_PAPER=true",
                        ]
                    )
                )
            for name in env_names:
                os.environ.pop(name, None)

            config = AlpacaConfig.from_env()
            os.chdir(old_cwd)
    finally:
        os.chdir(old_cwd)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert config.api_key == "dotenv_key"
    assert config.api_secret == "dotenv_secret"
    assert config.base_url == "https://paper-api.alpaca.markets/v2"
    assert config.paper


def test_alpaca_config_defaults_to_v2_paper_endpoint():
    old_base = os.environ.get("APCA_API_BASE_URL")
    old_alt_base = os.environ.get("ALPACA_API_BASE_URL")
    try:
        os.environ.pop("APCA_API_BASE_URL", None)
        os.environ.pop("ALPACA_API_BASE_URL", None)
        config = AlpacaConfig(api_key=None, api_secret=None)
        from_env = AlpacaConfig.from_env()
    finally:
        if old_base is None:
            os.environ.pop("APCA_API_BASE_URL", None)
        else:
            os.environ["APCA_API_BASE_URL"] = old_base
        if old_alt_base is None:
            os.environ.pop("ALPACA_API_BASE_URL", None)
        else:
            os.environ["ALPACA_API_BASE_URL"] = old_alt_base

    assert config.base_url == "https://paper-api.alpaca.markets/v2"
    assert from_env.base_url == "https://paper-api.alpaca.markets/v2"


def test_broker_status_records_are_display_ready():
    statuses = [
        PaperBrokerAdapter(PaperBroker(cash=50_000)).status(),
        AlpacaBrokerAdapterStub(AlpacaConfig(api_key=None, api_secret=None)).status(),
    ]

    records = broker_status_records(statuses)

    assert records[0]["Broker"] == "PaperBroker"
    assert records[1]["Broker"] == "AlpacaPaperAdapter"


def test_alpaca_config_validation_records_accept_paper_v2_config():
    records = alpaca_config_validation_records(
        AlpacaConfig(
            api_key="key",
            api_secret="secret",
            paper=True,
            base_url="https://paper-api.alpaca.markets/v2",
        )
    )
    checks = {row["Check"]: row for row in records}

    assert checks["API keys found"]["Passed"]
    assert checks["Using paper account"]["Passed"]
    assert checks["Paper account URL"]["Passed"]
    assert checks["Live account URL blocked"]["Passed"]
    assert checks["Alpaca URL includes /v2"]["Passed"]


def test_alpaca_config_validation_records_reject_live_endpoint():
    records = alpaca_config_validation_records(
        AlpacaConfig(
            api_key="key",
            api_secret="secret",
            paper=False,
            base_url="https://api.alpaca.markets/v2",
        )
    )
    checks = {row["Check"]: row for row in records}

    assert not checks["Using paper account"]["Passed"]
    assert not checks["Paper account URL"]["Passed"]
    assert not checks["Live account URL blocked"]["Passed"]
