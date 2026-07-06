# AgentLoop Trader

A governed trading simulator and research lab built from a classic turtle-style trend-following strategy. The current app supports synthetic price data or market data from Yahoo Finance, runs a deterministic backtest, generates structured trade intents, and passes proposed trades through explicit risk controls before any execution mode can act on them.

## Features

- Simulates breakout entries using an N-bar high.
- Uses an SMA trend filter before entering trades.
- Calculates ATR-based stops and position sizing.
- Exits on stop loss or an N-bar low.
- Shows performance metrics, current signal state, live rule values, and an interactive Plotly chart.
- Reports max drawdown, profit factor, and market exposure.
- Generates a structured trade intent when the strategy produces a current entry signal.
- Runs each trade intent through deterministic risk checks.
- Supports explicit execution modes: backtest only, paper trading, shadow mode, live with approval, and automated live.
- Adds an out-of-sample walk-forward evaluation panel to compare training and test-period behavior.
- Adds a bounded parameter loop that ranks allowed strategy settings without changing risk or execution code.
- Adds portfolio-aware risk policy and execution preflight checks.
- Adds a broker adapter interface with a local paper adapter and non-trading Alpaca stub.
- Adds durable JSONL audit logs and shadow-mode decision recording.
- Adds local-only automation readiness checks for market data freshness, restart recovery, scheduler preview, and paper account health.
- Supports synthetic data runs and optional stock data through `yfinance`.
- Includes a selectable trade log that highlights entries and exits on the chart.

## Product Direction

This project is evolving from a standalone simulator into AgentLoop Trader: a governed agentic trading lab. The core product rule is:

```text
The LLM may propose a trade. Deterministic code decides whether it is allowed.
```

The app should not be framed as an autonomous trading bot. It is a research, backtesting, paper-trading, and risk-governed execution environment.

This repository is now being developed toward eventual real-money Alpaca use, but live unattended trading is not considered safe until the controls in `docs/PRODUCTION_SAFETY.md` and `docs/LIVE_DEPLOYMENT_CHECKLIST.md` are satisfied.

## Portfolio Narrative

AgentLoop Trader is designed to demonstrate a practical agentic workflow with human-in-the-loop guardrails:

1. Observe market state and strategy signals.
2. Propose a structured trade thesis and intent.
3. Gate the proposal through deterministic risk policy.
4. Require a human to arm one exact broker preview before paper execution.
5. Reconcile broker state after the action.
6. Review outcomes and preserve audit evidence.
7. Rehearse paper automation with local dry-run queues before any unattended behavior exists.

The default Streamlit workspace is a daily operator view for running a small personal account. The `Portfolio Evidence` workspace exposes the deeper artifacts that make the AI TPM story visible: run manifests, approval ledgers, dry-run automation evidence, readiness reports, and deployment guardrails.

The product intent is not to maximize automation authority. It is to show how an AI system can propose actions while deterministic software and human approval retain control over capital.

## UI Product Rule

The daily workflow should stay simple enough for an expert operator to use quickly:

- Daily Operator shows the research, current signal, trade plan, risk decision, and broker actions.
- Portfolio Evidence holds the detailed audit trail, lifecycle tables, readiness reports, and deployment proof.
- Hard blockers prevent obvious mechanical mistakes. Warnings inform the operator without adding ceremony.
- Button labels should describe the actual action in plain language, such as review, send, refresh, track, exit, or cancel.

## Architecture Roadmap

1. Refactor the simulator into reusable modules for data, indicators, strategy, backtesting, risk, execution, agents, and UI.
2. Preserve the turtle trend-following strategy as Strategy 1.
3. Add structured models for trade intents, theses, backtest results, risk checks, execution decisions, session configs, and audit events.
4. Expand the backtest harness with richer metrics, walk-forward testing, and out-of-sample evaluation.
5. Add deterministic risk controls for allowed symbols, stop-loss requirements, max risk per trade, max position size, portfolio concentration, duplicate positions, and kill-switch behavior.
6. Add an AI research/thesis layer that produces bounded trade proposals.
7. Add a paper broker adapter with orders, fills, positions, cash, and P&L.
8. Add broker adapters later, targeting Alpaca first because its API-first paper/live workflow is faster to operationalize.
9. Add monitoring and post-trade review so closed trades can be evaluated against their original thesis.

## Current Modules

```text
agentloop_trader/
  backtest.py     # deterministic turtle-strategy simulation and metrics
  brokers.py      # broker adapter contract, paper adapter, and Alpaca stub
  data.py         # synthetic data generation
  execution.py    # thin paper broker/order skeleton
  evaluation.py   # train/test split and walk-forward evaluation
  parameter_loop.py # bounded parameter evaluation and recommendations
  indicators.py   # ATR and SMA calculations
  models.py       # structured trading, risk, and audit contracts
  ops_readiness.py # local-only paper automation readiness checks
  risk.py         # deterministic risk checks and execution preflight policy
```

## Alpaca Adapter Target

The current Alpaca adapter is a non-trading stub. It checks configuration readiness but intentionally blocks API order submission until live controls are mature.

Expected future environment variables:

```powershell
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
APCA_API_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
```

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

## Notes

Yahoo Finance intraday data can be delayed and may have period limits depending on the selected interval. This simulator is for research and education only, not financial advice. Live trading should remain disabled until broker adapters, audit logs, kill-switch behavior, and manual approval controls are mature.

The current Alpaca adapter can read paper account data and can submit paper orders only behind manual gates. Alpaca live order submission is intentionally blocked in code.
Shadow mode records would-have-traded decisions without submitting orders. Audit logs are written to `audit_logs/agentloop_audit.jsonl` by default.

## Production Safety Docs

- `docs/PRODUCTION_SAFETY.md`
- `docs/LIVE_DEPLOYMENT_CHECKLIST.md`
