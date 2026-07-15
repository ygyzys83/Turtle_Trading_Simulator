# Alpaca Paper Test Plan

Use these checks before any live-trading work. The labels below match the Streamlit UI.

## Market-Open Fill Check

1. Confirm Alpaca paper credentials are loaded and the account is connected.
2. Generate a risk-approved real-ticker proposal, such as `AAPL`.
3. Check `Enable Alpaca paper orders`.
4. Check `Confirm next Alpaca paper order`.
5. Click `Arm Alpaca Paper Order`.
6. Confirm the green banner says `Alpaca paper order preview is armed and unchanged`.
7. Click `Submit Alpaca Paper Order`.
8. Confirm the order appears in the Alpaca Orders dashboard.
9. After the order fills, click `Refresh Alpaca Paper Order State`.
10. Confirm `Alpaca paper order lifecycle` shows the order as filled.
11. Confirm `Alpaca paper position lifecycle` shows the position.

## Cancel Check

1. Submit a paper order outside market hours or another order that remains open.
2. Check `Confirm next Alpaca paper cancel`.
3. Click `Arm Alpaca Paper Cancel`.
4. Confirm the green banner says `Alpaca paper cancel preview is armed and unchanged`.
5. Click `Cancel Alpaca Paper Order`.
6. Confirm Alpaca Orders shows the order canceled.
7. Click `Refresh Alpaca Paper Order State`.
8. Confirm `Alpaca paper order lifecycle` shows `canceled_at_alpaca`.

## Exit Preview Check

1. Wait until `Alpaca paper position lifecycle` shows an open position.
2. Review `Alpaca paper exit preview`.
3. Click `Arm Alpaca Paper Exit`.
4. Do not click `Submit Alpaca Paper Exit` until the preview has been manually inspected.

## Position Lifecycle Check

1. Fill a real Alpaca paper BUY and refresh the app.
2. Confirm the position's cycle ID and basis BUY match the current Alpaca fill.
3. Confirm the average entry and initial stop are based on Alpaca's actual average fill.
4. Fully exit the position and then re-enter the same ticker.
5. Confirm the re-entry starts a new cycle and does not inherit the prior cycle's high-water mark or profit protection.
6. Confirm a position opened directly in Alpaca remains unmanaged until exit settings are explicitly saved.

## Automation Evidence Check

1. Open `Paper automation dry-run`.
2. Review `Paper automation readiness`.
3. Review `Paper automation candidate queue`.
4. Review `Paper automation supervisor dry-run`.
5. Confirm `Broker Writes Submitted` is `0`.
6. Click `Record Paper Automation Dry Run`.
7. Confirm `Automation evidence dashboard` increments `Dry Run Snapshots`.

## Evidence Package Check

1. Open `Run manifest`.
2. Click `Record Run Manifest`.
3. Open `Evidence dashboard`.
4. Review `Approval ledger`.
5. Click `Export Evidence Package`.
6. Confirm the package is written to `Evidence export path`.

## Halt Check

1. Review `Broker heartbeat and staleness policy`.
2. Review `Risk halt reasons`.
3. Toggle `Kill switch`.
4. Confirm `Risk halt reasons` shows an active halt.
5. Turn `Kill switch` off before any paper-order test.

## Readiness Check

1. Review `Paper performance dashboard`.
2. Click `Record Paper Performance Review`.
3. Click `Emergency disable session` once during a safe test run.
4. Open `Pre-live readiness report`.
5. Confirm incomplete items remain blocked until the matching evidence exists.
