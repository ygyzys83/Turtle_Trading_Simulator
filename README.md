# AgentLoop Trader

A governed trading simulator, paper-trading console, and research lab for testing simple technical strategies before any real-money use. The app supports synthetic data, Alpaca stock and crypto data, and Yahoo Finance stock data; compares five deterministic strategies; generates structured trade intents; and passes every proposed order through explicit risk controls before execution can act on it.

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
- Always shows how the exact selected strategy and inputs performed in the older 55%, newer 25%, and latest 20% of the loaded price history.
- Adds a bounded strategy-input search that favors stable nearby settings, separates older/newer/latest price sections, and includes realistic execution-cost stress.
- Describes each winning result across deterministic price-behavior periods and reports whether performance depended on one favorable environment.
- Adds an optional bounded analyst, skeptical-reviewer, and decision-editor loop over hashed deterministic evidence. Invalid or invented claims fall back to the built-in decision.
- Can apply the selected settings unchanged to other tickers and report whether the result generalizes.
- Adds portfolio-aware risk policy and execution preflight checks.
- Adds a broker adapter interface with a local simulator and Alpaca paper-order support.
- Adds durable JSONL audit logs and shadow-mode decision recording.
- Adds local-only automation readiness checks for market data freshness, restart recovery, scheduler preview, and paper account health.
- Supports synthetic data runs plus optional stock data through Alpaca or `yfinance`.
- Supports Alpaca crypto pairs such as `BTC/USD` with 24/7 completed-bar logic, fractional sizing, GTC/IOC order rules, and Alpaca maker/taker fee estimates.
- Adds a ticker Ideas page that scans a watchlist, ranks current setup quality, and creates a concise research read.
- Adds optional deterministic, Ollama, or Gemini research summaries. The LLM cannot send orders or override deterministic risk rules.
- Adds an optional paper-only background worker for automation checks outside the Streamlit page refresh.
- Adds a durable Buy watchlist capped at 10 independent setups. Save the current researched setup from New Trade, then manage queue status, repeat behavior, saved details, pausing, and removal from Positions & Queue.
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
- RSI mean-reversion scalp arms after RSI reaches the selected low level or falls the selected number of points from a recent RSI high. It buys only after RSI rebounds from the setup low and price closes above the prior completed bar. It supports a standard ATR stop, a wider emergency ATR stop, or a no-price-stop research backtest. No-price-stop mode uses fixed ticker allocation and cannot be queued for automated orders. RSI recovery and the optional 100-bar maximum remain available in every mode.
- Historical entries use the signal-bar close.
- Protective stops fill at the stop price, or at the next bar open after a gap below the stop.
- Break-even and ATR trailing protection turn on when the highest price since entry reaches the saved R threshold; once active, protection does not loosen.
- Results include unrealized profit or loss from a simulated position still open on the final bar.
- Newer-data and optimizer comparisons use completed trades on both sides of the split so open-position P&L cannot skew one side.
- Backtest results deduct estimated Alpaca trading fees. Stock tests use the current U.S. equity regulatory-fee model; crypto tests conservatively use Alpaca's Tier 1 taker fee on each side. Spread, market impact, taxes, and idle-cash interest are not included. The strategy-input recommendation separately shows 5, 10, and 20 basis-point-per-side stress results; paper and live fills remain the final execution test.
- A normal trading-screen refresh calculates only the strategy selected in the sidebar. The optional all-strategy comparison runs only after `Compare All Strategies` is clicked and remains cached until a material input changes.
- Each strategy-input search uses one interval chosen by the operator and compares the four trend strategies on that same dataset. Alpaca equity search history remains 1-hour/2-year, 4-hour/5-year, or daily/10-year; Alpaca crypto remains 1-hour/1-year, 4-hour/2-year, or daily/5-year. The older 55% supplies the historical sample, the newer 25% selects among the historical finalists, and the latest 20% reports what happened without choosing replacement settings.
- The search samples 32 broad combinations per strategy, then up to 16 intentional nearby combinations around the strongest regions. Chronological validation is limited to three finalists per strategy, and price-behavior checks reuse the same older/newer/latest sections instead of running six extra periods.
- Historical bars are cached until another bar can complete. A lightweight Alpaca latest-trade request updates current pricing every 15 seconds without repeatedly downloading complete history.
- RSI remains available as current setup information and as the separate RSI mean-reversion strategy. RSI rules are intentionally excluded from the four-strategy input search so the search stays focused and less vulnerable to combinatorial overfitting.
- Price-behavior labels describe the ticker path rather than claiming to identify an economic business cycle. Direction is classified as strong/mild uptrend, sideways, or mild/strong downtrend; path is classified as persistent, mixed, or choppy. The result also reports performance across six chronological periods and whether profits were concentrated in one type of price behavior.
- Strategy quality is measured against a stable per-ticker capital allocation set by `Max symbol concentration`. The UI keeps whole-account return for portfolio impact, while allocated return, allocated worst drop, and equal-capital buy-and-hold are used for strategy comparison and optimizer evidence. Strategy and buy-and-hold worst drops both divide the largest peak-to-trough dollar decline by the original ticker allocation, making the displayed percentages directly comparable. Annualized allocated and buy-and-hold returns use actual timestamps and appear only when the measured period exceeds one year. Return on average capital deployed is intentionally not used.
- Alpaca and Yahoo price data are validated, sorted, deduplicated, and checked for impossible OHLC values before use.
- Crypto bars are completed on elapsed UTC time rather than stock-session boundaries, so the worker can evaluate crypto entries and exits on weekends and outside stock market hours.

## Current Modules

```text
agentloop_trader/
  backtest.py     # deterministic strategy simulations and metrics
  automation_runtime.py # background worker control and heartbeat files
  brokers.py      # broker adapter contract, paper adapter, and Alpaca integration
  data.py         # synthetic data generation
  execution.py    # thin paper broker/order skeleton
  evaluation.py   # time-period consistency checks for exact selected settings
  parameter_loop.py # bounded parameter evaluation and recommendations
  indicators.py   # ATR and SMA calculations
  models.py       # structured trading, risk, and audit contracts
  llm_research.py # deterministic/Ollama/Gemini research writer adapter
  price_regime.py # deterministic price-path classification and regime-dependence evidence
  strategy_recommendation.py # bounded strategy analyst/reviewer/editor loop
  idea_pipeline.py # research-only ticker source contract and durable future queue
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

The LLM cannot promote a built-in recommendation, change strategy calculations or risk settings, or submit orders. The strategy-search review runs at most one analyst draft, one skeptical review, and one edited decision. Every final claim must cite a supplied evidence ID; failed validation uses the built-in recommendation and records the failure.

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
python run_app.py
```

The supervised launcher opens the app at `http://127.0.0.1:8501`, prevents a second UI from starting on another port, and cleans up the complete Streamlit process tree when you press `Ctrl+C` in the terminal. The app will open in your browser. Use the sidebar controls to switch between synthetic data and stock data, adjust the strategy windows, risk settings, and ATR stop multiplier.

Paper automation:

Selecting `Paper trading - send orders to Alpaca paper` automatically uses the configured Alpaca paper account; there is no second account checkbox. The ticker currently open in Streamlit is research/manual-only and is never an automatic BUY source. Automatic entries can come only from enabled Buy watchlist setups, and only the Background Worker monitors that queue. Select `Auto exits and queued buys`, check `Allow queued buys`, and start the worker to enable them. Automatic exits remain independent of queued-buy permission and can run from the open page or worker. The worker is paper-only and uses the same saved controls, risk limits, Alpaca paper account, and broker-state file as Streamlit.

When Streamlit opens while a worker record exists, the sidebar restores the worker's saved Automation mode, Kill Switch, and queued-buy permission before writing control state. The sidebar separately reports worker-process state, heartbeat freshness, and whether automation actions are On or Off. Windows Sleep suspends monitoring until the computer wakes; reopening Streamlit no longer resets a resumed worker to Manual.

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
