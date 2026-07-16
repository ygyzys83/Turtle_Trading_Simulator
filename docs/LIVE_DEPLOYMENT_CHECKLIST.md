# Live Deployment Checklist

Strategy-search or LLM recommendations are research evidence only. Completing an agent review never satisfies an execution, risk, or broker-readiness item below.

Use this checklist before enabling any real-money order path.

## Alpaca Account

- Alpaca paper account tested.
- Alpaca live account reviewed separately.
- API keys stored outside source control.
- Paper and live base URLs are visibly distinct.
- Account equity, cash, buying power, positions, and open orders reconcile before trading.

## Strategy Evidence

- Backtest completed.
- Walk-forward evaluation completed.
- Bounded parameter loop reviewed.
- Paper trading logs reviewed.
- Shadow-mode results reviewed.
- Post-trade reviews show rule-following discipline.

## Session Controls

- Execution mode confirmed.
- Manual approval requirement confirmed.
- Kill switch visible and tested.
- `Emergency disable session` tested and visible in the audit log.
- Max session loss configured.
- Max position size configured.
- Max portfolio exposure configured.
- Allowed symbols configured.
- Stop loss required.
- Audit logging active.
- `Create Live Mode Lockfile` completed.
- `Live mode lockfile` shows `Live Mode Locked`.

## Deployment Environment

- `.env.example` exists and contains no secrets.
- `.env` is present only on the deployment host and ignored by git.
- `Audit log path` points to durable storage.
- `Broker state path` points to durable storage.
- `Automation dry-run path` points to durable storage.
- `Evidence export path` points to durable storage.
- Restart procedure preserves audit logs, broker state, and automation snapshots.
- Evidence export has been generated after the latest paper test.

## First Live Session

- Tiny capital only.
- One symbol or tightly bounded watchlist.
- One order at a time.
- Manual approval only.
- No unattended operation.
- Review broker fills and audit log after every order.

## Unattended Live Criteria

- Multiple successful paper sessions.
- Multiple successful shadow sessions.
- Multiple successful manual-live sessions.
- No unexplained order behavior.
- No unreconciled account state.
- Tested emergency disable path.
- Tested restart/recovery path.
