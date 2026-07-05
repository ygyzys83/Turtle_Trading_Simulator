# Production Safety Plan

AgentLoop Trader is being built with the assumption that it may eventually manage real capital through Alpaca. The system must remain conservative by default.

## Non-Negotiable Controls

- The LLM or rules agent may propose trades, but deterministic code must approve or reject them.
- Risk limits, broker credentials, order submission code, and kill-switch behavior are not agent-modifiable.
- Live order submission must remain disabled until paper trading and shadow-mode evidence are reviewed.
- Alpaca live mode must require separate configuration from Alpaca paper mode.
- Every proposal, risk decision, preflight result, order attempt, and emergency disable event must be auditable.
- Audit events must be persisted to durable JSONL logs before live capital is used.
- Shadow-mode decisions must be reviewed before manual live trading.

## Promotion Path

1. Backtest only.
2. Walk-forward and bounded parameter evaluation.
3. Local paper broker simulation.
4. Alpaca read-only paper account connection.
5. Alpaca paper order submission with manual confirmation.
6. Shadow mode against live market/account data.
7. Live trading with manual approval and tiny capital.
8. Unattended live trading only after explicit review of logs, failures, and risk events.

## Live Capital Guardrails

- Max daily/session loss.
- Max position notional.
- Max portfolio exposure.
- Max symbol concentration.
- Max open positions.
- Stop loss required.
- Kill switch always visible.
- Broker reconciliation required before order submission.
- No market order submission outside an approved session.

## Stop Conditions

The system should stop proposing or submitting orders when:

- Kill switch is enabled.
- Broker account cannot be reconciled.
- Alpaca API returns unexpected order/account state.
- Session loss cap is breached.
- Portfolio exposure cap is breached.
- Audit logging is unavailable.
- Time, market, or data status is unknown.
