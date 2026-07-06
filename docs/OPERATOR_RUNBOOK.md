# Operator Runbook

Use this runbook for paper testing, emergency halt, recovery, and evidence export.

## Before Each Session

1. Confirm `.env` exists locally and is not committed.
2. Confirm `Alpaca config validation` passes paper mode and `/v2` endpoint checks.
3. Open `Run manifest` and click `Record Run Manifest`.
4. Open `Deployment readiness` and confirm `Live mode lockfile` is present.
5. Open `Broker heartbeat and staleness policy` and confirm Alpaca is connected and state is fresh.

## Market-Open Paper Test

1. Generate a risk-approved real-ticker trade intent.
2. Check `Enable Alpaca paper orders`.
3. Check `Confirm next Alpaca paper order`.
4. Click `Arm Alpaca Paper Order`.
5. Confirm `Alpaca paper order preview is armed and unchanged`.
6. Click `Submit Alpaca Paper Order`.
7. Confirm the order appears in Alpaca.
8. After fill, click `Refresh Alpaca Paper Order State`.
9. Confirm `Alpaca paper order lifecycle` and `Alpaca paper position lifecycle` reconcile the fill.

## Exit Test

1. Review `Alpaca paper exit preview`.
2. Check `Confirm next Alpaca paper exit`.
3. Click `Arm Alpaca Paper Exit`.
4. Stop and inspect the preview.
5. Click `Submit Alpaca Paper Exit` only after manual approval.
6. Export evidence after the exit test.

## Local Simulator Test

1. Open `Local paper lifecycle simulators`.
2. Click `Simulate Alpaca Paper Fill`.
3. Confirm simulated lifecycle tables update.
4. Click `Record Simulated Exit Readiness`.
5. Treat simulator evidence as UI evidence only. It does not replace real Alpaca paper fill evidence.

## Emergency Halt

1. Click `Emergency disable session`.
2. Confirm `Risk halt reasons` shows an active halt.
3. Confirm `Preflight` is blocked.
4. Export evidence.
5. Restart only after reviewing the audit log and resetting the paper broker/session intentionally.

## Recovery

1. Restart Streamlit.
2. Confirm `Audit log path`, `Broker state path`, and `Automation dry-run path` point to the expected files.
3. Confirm tracked Alpaca paper orders reload.
4. Click `Refresh Alpaca Paper Order State`.
5. Export a fresh evidence package.

## Evidence Export

1. Open `Evidence dashboard`.
2. Review `Approval ledger`.
3. Click `Export Evidence Package`.
4. Keep exported evidence local unless it has been reviewed for account details.
