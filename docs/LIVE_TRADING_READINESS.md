# Live Trading Readiness

This app is still paper-first. Alpaca live broker writes are blocked in code, and paper broker writes remain manually gated.

## Required Evidence Before Live

- `Submit Alpaca Paper Order` has placed a paper order in Alpaca.
- `Cancel Alpaca Paper Order` has canceled a paper order in Alpaca.
- `Submit Alpaca Paper Exit` has been inspected and tested in Alpaca paper.
- `Refresh Alpaca Paper Order State` has reconciled at least one filled paper order.
- `Alpaca paper position lifecycle` has shown the expected filled position.
- `Record Paper Automation Dry Run` has captured durable dry-run evidence.
- `Paper performance dashboard` has been reviewed with `Record Paper Performance Review`.
- `Emergency disable session` has been tested and recorded.
- `Record Run Manifest` has captured the active strategy, risk, broker, and account context.
- `Approval ledger` shows preview hashes for every Alpaca paper arm/submit/cancel/exit action.
- `Paper automation supervisor dry-run` shows zero broker writes submitted.
- `Risk halt reasons` shows no active halt before any manual paper submission.
- `Export Evidence Package` produces a local review file.
- The pre-live readiness report shows every required check as complete.

## Live Constraints

- Do not run unattended live trading until live broker writes have a separate manual approval gate.
- Do not allow the agent to modify API credentials, execution mode, risk limits, or kill-switch behavior.
- Start any future live pilot with small capital and a hard external broker-side buying power limit.
- Keep the Alpaca paper account as the default environment.

## Current Safe Status

- Alpaca paper account reads can connect when credentials are present.
- Alpaca paper order, cancel, and exit paths require explicit UI confirmation.
- Live Alpaca submissions remain blocked by the adapter safety invariant.
- Evidence, manifest, approval ledger, and supervisor controls are local-only and do not contact Alpaca.
