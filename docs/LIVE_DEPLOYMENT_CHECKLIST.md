# Live Deployment Checklist

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
- Max session loss configured.
- Max position size configured.
- Max portfolio exposure configured.
- Allowed symbols configured.
- Stop loss required.
- Audit logging active.

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

