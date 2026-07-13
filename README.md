# AgentLoop Trader

A governed trading simulator, paper-trading console, and research lab for testing simple technical strategies before any real-money use. The app supports synthetic data, Alpaca price data, and Yahoo Finance price data; compares four deterministic strategies; generates structured trade intents; and passes every proposed order through explicit risk controls before paper execution can act on it.

## Features

- Simulates breakout entries using an N-bar high.
- Uses a trend filter before entering trades.
- Calculates ATR-based stops and position sizing.
- Exits on stop loss or an N-bar low.
- Shows performance metrics, current signal state, live rule values, and an interactive Plotly chart.
- Reports max drawdown, profit factor, and market exposure.
- Generates a structured trade intent when the strategy produces a current entry signal.
- Runs each trade intent through deterministic risk checks.
- Supports explicit execution modes: backtest only, paper trading, shadow mode, live with approval, and automated live.
- Adds an out-of-sample walk-forward evaluation panel to compare training and test-period behavior.
- Adds a bounded strategy-input search that favors stable nearby settings, rolling periods, an untouched final period, and realistic execution-cost stress.
- Can apply the selected settings unchanged to other tickers and report whether the result generalizes.
- Adds portfolio-aware risk policy and execution preflight checks.
- Adds a broker adapter interface with a local simulator and Alpaca paper-order support.
- Adds durable JSONL audit logs and shadow-mode decision recording.
- Adds local-only automation readiness checks for market data freshness, restart recovery, scheduler preview, and paper account health.
- Supports synthetic data runs plus optional stock data through Alpaca or `yfinance`.
- Adds a ticker Ideas page that scans a watchlist, ranks current setup quality, and creates a concise research read.
- Adds optional deterministic, Ollama, or Gemini research summaries. The LLM cannot send orders or override deterministic risk rules.
- Adds an optional paper-only background worker for automation checks outside the Streamlit page refresh.
- Adds a durable Buy watchlist capped at 10 independent setups. Each row saves its ticker, interval, strategy, inputs, risk limits, and order instructions; the worker records whether it is waiting, blocked, paused, or sent and disables it after submission.
- Queued BUY signals use cached completed bars, while one lightweight batched Alpaca latest-trades request supplies current IEX prices for order repricing and sizing on each worker cycle. A missing latest price blocks submission rather than falling back to stale history.
- Includes a selectable trade log that highlights entries and exits on the chart.

## Product Direction

This project is evolving from a standalone simulator into AgentLoop Trader: a governed agentic trading lab. The core product rule is:

```text
The LLM may propose a trade. Deterministic code decides whether it is allowed.
```

The app should be treated as a personal paper-trading and strategy-research environment until the live checklist is deliberately completed. The LLM can explain and rank ideas, but deterministic strategy and risk code controls order eligibility.

This repository is now being developed toward eventual real-money Alpaca use, but live unattended trading is not considered safe until the controls in `docs/PRODUCTION_SAFETY.md` and `docs/LIVE_DEPLOYMENT_CHECKLIST.md` are satisfied.

## UI Product Rule

The daily workflow should stay simple enough for an expert operator to use quickly:

- Daily Trading Screen shows open positions, ideas, current signal, trade plan, risk decision, and broker actions.
- Full Records and Evidence holds the detailed audit trail, lifecycle tables, readiness reports, and deployment proof.
- Hard blockers prevent obvious mechanical mistakes. Warnings inform the operator without adding ceremony.
- Button labels should describe the actual action in plain language, such as review, send, refresh, track, exit, or cancel.
- The visual system uses a deep-navy trading-console theme with compact controls and semantic color: green for approved actions and positive states, blue for research and selection, amber for warnings and strategy exits, and red for losses, blocks, stops, and the Kill Switch.
- Shared UI colors and component styling live in `agentloop_trader/ui_theme.py`; Streamlit theme defaults live in `.streamlit/config.toml`.

## Backtest Assumptions

- Indicators and entry signals use completed bars.
- Breakout continuation compares the completed-bar close with prior bar highs for entry and prior bar lows for exit.
- Historical entries use the signal-bar close.
- Protective stops fill at the stop price, or at the next bar open after a gap below the stop.
- Break-even and ATR trailing protection turn on when the highest price since entry reaches the saved R threshold; once active, protection does not loosen.
- Results include unrealized profit or loss from a simulated position still open on the final bar.
- Newer-data and optimizer comparisons use completed trades on both sides of the split so open-position P&L cannot skew one side.
- Normal backtest results do not deduct commission, spread, market impact, or slippage. The strategy-input recommendation separately shows 5, 10, and 20 basis-point-per-side stress results; paper and live fills remain the final execution test.
- For real tickers, the strategy-input search ranks daily, 4-hour, and 1-hour results over the same latest two-year calendar window (or the shorter common window available from every interval). It then tests each interval's winning settings without changes on its longer available history: daily up to 10 years, 4-hour up to 5 years, and 1-hour up to 2 years with Alpaca. A result is labeled ready for paper testing only when it beats equal-capital buy-and-hold in both the newer-data and untouched locked periods and passes the minimum trade, rolling-period, and slippage checks.
- `Require RSI 50-70 for BUY` is the only optional setup-quality read that currently becomes a hard entry rule. When enabled, all four strategies require 14-bar RSI between 50 and 70 in historical and current BUY decisions. The optimizer tests every bounded setting combination with this rule off and on; RSI does not control exits.
- Strategy quality is measured against a stable per-ticker capital allocation set by `Max symbol concentration`. The UI keeps whole-account return for portfolio impact, while allocated return, allocated worst drop, and equal-capital buy-and-hold are used for strategy comparison and optimizer evidence. Strategy and buy-and-hold worst drops both divide the largest peak-to-trough dollar decline by the original ticker allocation, making the displayed percentages directly comparable. Annualized allocated and buy-and-hold returns use actual timestamps and appear only when the measured period exceeds one year. Return on average capital deployed is intentionally not used.
- Alpaca and Yahoo price data are validated, sorted, deduplicated, and checked for impossible OHLC values before use.

## Current Modules

```text
agentloop_trader/
  backtest.py     # deterministic turtle-strategy simulation and metrics
  automation_runtime.py # background worker control and heartbeat files
  brokers.py      # broker adapter contract, paper adapter, and Alpaca integration
  data.py         # synthetic data generation
  execution.py    # thin paper broker/order skeleton
  evaluation.py   # train/test split and walk-forward evaluation
  parameter_loop.py # bounded parameter evaluation and recommendations
  indicators.py   # ATR and SMA calculations
  models.py       # structured trading, risk, and audit contracts
  llm_research.py # deterministic/Ollama/Gemini research writer adapter
  market_data.py  # Alpaca/yfinance bars plus Alpaca news context
  ops_readiness.py # local-only paper automation readiness checks
  scanner.py      # deterministic watchlist scanner and candidate store
  risk.py         # deterministic risk checks and execution preflight policy
```

## Alpaca Paper Trading

The Alpaca adapter can read account, position, and order state. In paper mode it can submit paper buys, paper exits, and paper cancels through the app gates. Live order wiring exists only behind explicit live environment variables and live confirmation; unattended live automation is not enabled.

Paper environment variables:

```powershell
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2
ALPACA_PAPER=true
```

## Optional LLM Research

The app works without an LLM. The built-in deterministic writer summarizes the current setup from strategy data. Optional adapters can call Ollama or Gemini for a concise research read using structured JSON.

```powershell
RESEARCH_LLM_PROVIDER=deterministic
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
```

The LLM cannot promote a deterministic `WAIT` into a paper-trade `TRADE`, cannot change risk settings, and cannot submit orders.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run tests:

```powershell
pytest -q
```

## Run

Start the app with:

```powershell
streamlit run turtle_trading.py
```

The app will open in your browser. Use the sidebar controls to switch between synthetic data and stock data, adjust the strategy windows, risk settings, and ATR stop multiplier.

Paper automation:

Selecting `Paper trading - send orders to Alpaca paper` automatically uses the configured Alpaca paper account; there is no second account checkbox. With Streamlit open, the page can check the loaded ticker and saved exits on its refresh timer. Use `Start Worker` when monitoring must continue after Streamlit closes or when the durable Buy watchlist should be active. `Allow automatic paper buys` appears only for `Auto entries and exits`; it does not control automatic exits. The worker is paper-only and uses the same saved controls, risk limits, Alpaca paper account, and broker-state file as Streamlit.

You can still start it manually for debugging:

```powershell
python -m agentloop_trader.worker
```

## Notes

Yahoo Finance intraday data can be delayed and may have period limits depending on the selected interval. Alpaca free market data uses its available feed and can also be delayed. This simulator is for research and education only, not financial advice. Live trading should remain disabled until paper trading has been tested over time and the live deployment checklist is satisfied.

The current Alpaca adapter can read paper account data and can submit paper orders behind manual or paper-automation gates. Alpaca live order submission is intentionally restricted by environment variables, confirmation text, risk checks, and live setup records.
Shadow mode records would-have-traded decisions without submitting orders. Audit logs are written to `audit_logs/agentloop_audit.jsonl` by default.

## Production Safety Docs

- `docs/PRODUCTION_SAFETY.md`
- `docs/LIVE_DEPLOYMENT_CHECKLIST.md`
