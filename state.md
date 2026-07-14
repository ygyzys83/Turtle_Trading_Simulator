# Trading Simulator - Conversation State Handoff

Last updated: 2026-07-10 (Pacific Time)

Purpose: give a future Codex conversation enough context to continue this project without replaying the full conversation. This is a structured handoff, not a verbatim transcript. Read this file before proposing or implementing additional work.

## Project Location

Primary project directory:

`C:\Users\godma\PycharmProjects\Trading_Simulator`

Primary Streamlit entry point:

`turtle_trading.py`

Python package:

`agentloop_trader\`

The user manually creates Git commits and pushes them. When the user says "let's commit," provide a commit message only. Do not commit or push on the user's behalf unless they explicitly change that instruction.

## Product Goal

Build a simple, credible automated trading application for the user's own small Alpaca account while also making it an impressive AI Technical Product Manager portfolio project.

The intended product story is:

1. The user provides a ticker, or eventually receives candidate tickers from a scanner/recommendation loop.
2. Deterministic research and optional LLM explanation assess the setup.
3. The app compares strategies and strategy inputs.
4. Historical simulation and newer-data testing show how the selected rules behaved.
5. The user tests the workflow through Alpaca paper trading.
6. The user normally approves entries manually and lets the background worker manage exits. When desired, the Buy watchlist can monitor specifically queued paper-entry setups.
7. Full paper automation is available for testing but is not the default daily workflow.
8. Live trading is wired but remains deliberately restricted until paper validation is complete.

The immediate objective is not additional feature expansion. It is to run paper trading for at least two weeks, observe real behavior, and correct issues found under actual market conditions.

## User Preferences And Working Style

- Use simple, natural, explicit language throughout the UI.
- Do not use ambiguous jargon when a plain description is available.
- Match instructions to exact UI labels.
- When giving manual test instructions, always state:
  - Workspace: Daily Trading Screen or Full Records and Evidence.
  - Command-center page: Positions & Queue, Ideas, New Trade, Alpaca, or Paper Review.
  - Order mode: Backtest only, Paper trading, Practice mode, or live mode.
  - Automation setting: Manual, Auto exits, or Auto exits and queued buys.
  - Exact buttons, checkboxes, and expected messages.
- The user is an experienced trader/investor. Do not add ceremony merely to protect paper trading.
- Keep essential deterministic controls, audit records, and a Kill Switch, but do not clutter the daily workflow with repetitive checklists.
- Detailed records belong in Full Records and Evidence or collapsed expanders.
- Do not remove completed functionality casually. Hide or consolidate detailed evidence when it is useful but not needed daily.
- Prioritize research quality, strategy quality, correct math, and reliable automation over governance theater.
- The user wants decisive implementation and large safe batches rather than artificial batching.
- The user controls Git commits.

## Intended Daily Workflow

### Sidebar

The sidebar is organized in this order:

1. Kill Switch.
2. Background Automation worker controls.
3. Navigation / Workspace.
4. Automation level.
5. Paper trading and order behavior.
6. Ticker and price data.
7. Strategy settings.
8. Risk limits.
9. Research options.
10. Setup quality checks.
11. Advanced files/reset controls in a collapsed section.

### Main Command Center

The main pages are:

- Positions & Queue: primary daily surface for open-position exits and saved automatic-BUY setups.
- Ideas: scans a bounded ticker list and creates deterministic or optional LLM research reads.
- New Trade: research, BUY decision, backtest, strategy comparison, and order preparation.
- Alpaca: account summary, waiting orders, order actions, and automation status.
- Paper Review: daily paper-trading journal, progress, and performance review.

Two workspaces exist:

- Daily Trading Screen: streamlined daily operation.
- Full Records and Evidence *: adds detailed rules, logs, lifecycle tables, audit evidence, and deployment records. An asterisk marks Full Records-only content.

## Current Strategy Set

The app supports four long strategies.

### 1. Breakout continuation

- Entry: completed-bar close moves above the highest prior bar high over the selected buy lookback.
- Trend requirement: price is above a rising trend-filter SMA.
- Exit line: lowest prior bar low over the selected sell exit length.
- Initial stop: selected ATR multiplier below entry.

### 2. Trend pullback continuation

- Trend requirement: price is above a rising long-term trend-filter SMA.
- Pullback requirement: recent price action reaches the selected pullback-average area.
- Momentum requirement: price turns back up above the selected short momentum average.
- Historical and live initial stop: lower of the recent pullback low or the ATR stop.
- Strategy exit: selected sell-length moving average.

### 3. Trendline breakout

- Finds recent descending trendlines from confirmed swing highs.
- Requires a true close crossing from below the line to above it, not merely a price already above the line.
- Requires price above a rising trend-filter SMA.
- Initial stop: selected ATR multiplier.
- Strategy exit: selected prior-low channel.

### 4. Trendline retest continuation

- Requires a recent descending-trendline breakout.
- Waits for a retest near the broken line and a momentum turn upward.
- A pending breakout expires after the configured trendline lookback, preventing stale retests.
- Historical and live initial stop: lower of the structural retest stop and the ATR stop.
- Strategy exit: selected prior-low channel.

## Indicator And Market-Data Rules

- ATR uses the standard Wilder-smoothed 14-bar true range.
- ATR is interval-specific. A 1-hour ATR and daily ATR are different measurements.
- Strategy signals and indicators use completed bars only.
- The newest forming bar is preserved separately for current price, high, and order repricing.
- Alpaca and Yahoo data are sorted, deduplicated, converted to numeric OHLCV values, and validated for impossible ranges.
- Alpaca is the preferred real-ticker data source.
- Yahoo remains available as a second option.
- Synthetic data remains the app default so startup is immediate and does not depend on an API call.
- Alpaca free market data may use the IEX feed and therefore may not represent full consolidated-market volume.
- Event-calendar/fundamental data is not fully connected. Event risk is informational unless a future implementation explicitly changes it.

## Backtest Contract

- Historical entries occur at the completed signal bar's close.
- Protective stops fill at the stop price.
- If a bar opens below a long stop, the simulated fill occurs at that bar's open to model the gap.
- Strategy exits based on a completed close fill at that close.
- Position sizing is recomputed from account equity, selected strategy risk, stop distance, and configured risk caps.
- Historical trade records include:
  - Entry and exit.
  - Quantity and notional.
  - Initial risk dollars and account-risk percentage.
  - Stop.
  - Exit rule.
  - P&L.
  - Maximum adverse movement during the trade.
- Final equity includes unrealized P&L from a simulated position still open on the final bar.
- Completed-trade metrics such as win rate and profit factor exclude that open trade.
- Maximum drawdown includes adverse intratrade movement, not only closed-trade endpoints.
- Commission, spread, market impact, and slippage are not modeled. Paper fills are required to evaluate execution realism.
- Research Inputs are setup-quality descriptions. They do not change historical P&L unless they are explicitly implemented as strategy rules.
- Strategy Settings and Risk Limits do affect simulated orders and historical results.

## Risk And Position-Sizing Hierarchy

The strategy proposes raw quantity from:

`strategy risk dollars / stop risk per share`

Then deterministic sizing applies the most restrictive quantity allowed by:

1. Raw strategy quantity.
2. Maximum quantity.
3. Max risk per trade.
4. Max new order size.
5. Remaining portfolio exposure.
6. Remaining symbol concentration.
7. Available cash.
8. Daily-loss state.

Relevant distinctions:

- Strategy risk per trade controls the strategy's desired sizing.
- Max risk per trade is the hard account-level ceiling.
- Max new order size caps each new order's notional value.
- Max symbol concentration caps existing ticker exposure plus the proposed order.
- Max portfolio exposure includes positions and remaining unfilled BUY-order exposure.
- Max open positions includes symbols represented by waiting BUY orders.
- Daily loss is calculated from Alpaca current portfolio value versus Alpaca prior-day equity.
- The daily-loss dollar threshold is based on prior-day equity, not the already-reduced current equity.
- Adding to an existing filled position may be enabled.
- A second waiting BUY for the same ticker remains blocked even when adding to a filled position is allowed.
- Exits are not blocked by entry risk sizing because reducing exposure must remain possible.

## Per-Position Exit Management

Alpaca combines multiple buys for one ticker into one broker position. The app manages that combined position with one saved exit plan.

When an additional lot changes the Alpaca average entry:

- The position's original risk distance is preserved.
- The protective stop is rebased from the new average entry.
- The app uses the most recent filled/adopted/manual position exit settings, not an unrelated waiting add-on order.

The active sell trigger is the highest valid value among:

1. Original protective stop.
2. Current strategy exit line.
3. Break-even stop, once enabled.
4. ATR trailing stop, once enabled.
5. Previously saved trigger.

This means protection can tighten but cannot loosen.

### R-based profit protection

`1R = entry price - original stop price` for a long position.

Defaults:

- Move stop to break-even after the highest price since entry reaches +1R.
- Start ATR trailing after the highest price since entry reaches +2R.
- Trail by 3 ATR from the highest high since entry.

Important behavior:

- Activation uses the highest price reached since entry, not current price.
- Once a threshold has been reached, the protection remains active after a pullback.
- The position table displays current profit in R and highest profit reached in R separately.
- The exact interval, sell exit length, trend filter, pullback average, momentum length, ATR multiplier, entry ATR, initial stop, strategy exit, trailing stop, and active trigger are saved/displayed when available.
- Positions opened before those fields were introduced may honestly display "Not recorded."

## Alpaca Integration

- Paper credentials are stored in `.env`; never print, commit, or copy their values into documentation.
- `.env` is ignored by Git.
- Paper endpoint: `https://paper-api.alpaca.markets/v2`.
- Market-data endpoint is handled separately by the Alpaca data API.
- Paper orders can be market or limit orders.
- Limit choices include below current price, at current price, above current price, and a custom limit price.
- Limit BUY orders may optionally be submitted outside regular market hours.
- Old unfilled limit BUY orders can be canceled automatically after 5 minutes, 10 minutes, 15 minutes, 30 minutes, 1 hour, 2 hours, 4 hours, or 8 hours.
- Default stale-order time is 1 hour.
- Alpaca's broker clock is authoritative for regular-market status when available.
- Every submitted order uses a deterministic `client_order_id` derived from the reviewed order, providing an additional duplicate-order barrier.
- Broker reads fail closed in the worker. A positions/orders API failure cannot be mistaken for an empty account.
- Live order wiring requires live credentials, the live endpoint, `ALPACA_PAPER=false`, the explicit live environment switch, and the exact live confirmation string.
- Automated live submission is not the current validated operating mode.

## Background Worker

The sidebar has Start Worker and Stop Worker buttons below the Kill Switch.

The worker can continue after the Streamlit browser is closed.

It currently supports:

- Automatic paper exits for saved per-position exit plans.
- Automatic paper entries for the currently configured ticker when full paper automation is selected and explicitly enabled.
- Automatic stale limit-BUY cancellation.
- Alpaca order/position reconciliation.
- Audit logging.
- Atomic broker-state updates across the Streamlit app and worker.

The worker:

- Is paper-only in its current unattended path.
- Uses Alpaca portfolio value and cash for sizing.
- Uses Alpaca prior-day equity for daily P&L.
- Includes positions and open BUY orders in exposure calculations.
- Refuses new buys when account data cannot be read.
- Refuses new buys after the daily-loss threshold.
- Blocks duplicate waiting buys.
- Blocks duplicate waiting sells.
- Records an exit reason and all trigger components when it sends an exit.
- Continues to retry safely after transient blocked/error cycles.

The worker cannot discover random tickers by itself yet. Auto entry only evaluates the ticker/settings saved in the automation control file. The Ideas scanner is the start of the future discovery loop.

## Research And Agent Loop

The deterministic research loop is authoritative for trade eligibility:

1. Load and validate price bars.
2. Run all four strategy simulations.
3. Evaluate current required strategy rules.
4. Apply deterministic risk sizing and hard risk limits.
5. Produce a plain-English research read.
6. Recommend the best current strategy fit.
7. Present BUY, WAIT, or BLOCK.
8. Record the result for comparison and audit evidence.

The concise Research read includes:

- Final answer.
- Selected strategy, using the exact sidebar name, and a separate best current fit across all four exactly named strategies.
- Trend.
- Setup.
- Volatility.
- Liquidity.
- Risk/reward.
- Event risk status.
- Next action.

Optional LLM adapters:

- Built-in deterministic writer.
- Local Ollama.
- Gemini.

LLM boundaries:

- The LLM receives structured deterministic facts.
- It may explain, summarize, lower confidence, or recommend waiting.
- It may not promote a deterministic WAIT to TRADE.
- It cannot modify credentials, risk limits, execution code, order mode, or the Kill Switch.
- It cannot submit orders.
- If an LLM fails, the app falls back to deterministic research and reports the fallback.

Future vision: use a scanner/recommendation loop to propose company/ticker candidates, then run the deterministic strategy/risk process before any order becomes eligible.

## Strategy Input Optimizer

The optimizer:

- Searches a bounded neighborhood of settings across all four strategies.
- For real tickers, ranks daily, 4-hour, and 1-hour searches over the same latest two-year calendar period (or the shorter period common to all available intervals). It separately checks each interval's winning settings, unchanged, over its longer available history: daily up to 10 years, 4-hour up to 5 years, and 1-hour up to 2 years with Alpaca.
- Varies entry/trendline lookback, sell exit length, ATR multiplier, trend filter, pullback average, and momentum length as relevant to each strategy.
- Uses older and newer portions of the selected ticker history.
- Compares strategy return and drawdown with adjusted-price buy-and-hold over the exact same newer and locked bars.
- Uses completed trades consistently on both sides of the split.
- Penalizes too few trades, negative newer returns, excessive drawdown, deterioration, and returns concentrated in one newer subperiod.
- Caps profit-factor bonuses so a single no-loss sample cannot dominate.
- Reports newer-period return, trades, profit factor, drawdown, profitable subperiods, confidence, and the main concern.
- Suggests strategy risk only within the user's existing risk ceiling.
- Never changes risk limits, credentials, order mode, or broker access automatically.

The concise recommendation is labeled `Ready for paper test` only when it has Medium/High confidence, enough newer-data trades, profitable results in at least half of rolling periods, a passing 10-basis-point-per-side stress test, and positive excess return versus buy-and-hold in both the newer-data and untouched locked periods. Otherwise it is labeled `Research only`; the candidate remains inspectable and can still be applied deliberately. Applying a real-ticker recommendation also applies its recommended interval and supported history period.

Optimizer reporting distinguishes three periods explicitly: the complete shared interval-comparison period, the middle validation period used to rank candidates, and the untouched final period. The banner leads with complete-period strategy return, equal-capital buy-and-hold return, and excess return; it then reports validation and final-period excess separately. This prevents a favorable validation slice from appearing to contradict or outweigh a weak complete backtest.

Most setup-quality reads remain research and explanation inputs. RSI is the one explicit exception: `Require RSI 50-70 for BUY` applies a fixed 14-bar RSI rule to historical entries, current BUY intents, scanner/worker strategy runs, walk-forward evaluation, and paper-entry automation. It does not control exits. The optimizer compares every bounded strategy-setting combination with this RSI rule off and on, accounts for the larger number of trials, and displays the winning RSI choice in the existing recommendation table. RSI length and thresholds are intentionally fixed to avoid multiplying the search space. Volume, market condition, liquidity, and the other setup reads remain informational.

Strategy performance now separates account impact from strategy quality. `Max symbol concentration` defines the stable capital allocation for one ticker and defaults to 5%, matching the default `Max new order size`. Backtests show account return, allocated-capital return, allocated-capital worst drop, equal-capital buy-and-hold return, and excess return. Optimizer scoring, rolling evidence, locked-period checks, slippage stress, regime results, bootstrap results, and cross-ticker tests use allocated-capital return and drawdown. Buy-and-hold invests the same ticker allocation while the rest of the account remains idle in both views. Strategy and buy-and-hold worst-drop percentages now use the same denominator: the original ticker allocation. Annualized allocated and buy-and-hold returns use CAGR over actual timestamps and appear only for periods longer than one year; annualization is unavailable when the allocated sleeve is exhausted. Return on average capital deployed was deliberately excluded because highly variable time in position makes it harder to interpret.

Interpretation: this is bounded historical evidence for paper testing, not proof of the highest future profit or probability of success. The same newer sample is used to rank candidates, so recommendations still require paper validation and should not be described as an unbiased guarantee.

## Most Recent Full Quality Audit

Completed on 2026-07-10.

The audit reviewed market data, indicator math, all four strategies, live/backtest parity, risk sizing, optimizer logic, Alpaca previews/submission, duplicate prevention, exits, worker behavior, persistence, research, and UI organization.

Important corrections made during the audit:

- Replaced simple ATR with Wilder ATR.
- Added strict market-data validation and completed-bar handling.
- Corrected Donchian channels to prior highs/lows.
- Required price above a rising SMA for breakout entries.
- Added true trendline crossings and recent-breakout expiration for retests.
- Aligned live structural stops with historical pullback/retest stops.
- Added gap-aware stop fills and intratrade drawdown tracking.
- Included final open-position mark-to-market P&L.
- Made walk-forward evaluation honor the selected strategy.
- Made optimizer older/newer calculations closed-trade consistent.
- Corrected Alpaca daily-loss enforcement in the UI and worker.
- Included waiting BUY exposure in risk calculations.
- Prevented duplicate waiting orders while allowing optional additions to filled positions.
- Made broker reads fail closed.
- Added atomic cross-process broker-state persistence that preserves exit settings.
- Made break-even/trailing activation depend on highest profit reached.
- Reorganized strategy-specific chart lines.
- Removed duplicated daily UI panels and unrelated cross-page guidance.
- Made Streamlit dataframe values Arrow-safe.
- Removed the unwanted portfolio-story section from the README.
- Pinned the requirements file to the tested environment versions.

Final automated verification:

`238 passed in 1.31s`

Additional verification:

- Python compilation passed.
- Financial invariants passed for all four strategies.
- Walk-forward smoke checks passed for all four strategies.
- Desktop UI check passed with no horizontal overflow or Streamlit exceptions.
- Compact 390-pixel UI check passed with no clipped metrics or horizontal overflow.
- `git diff --check` passed; Windows line-ending notices were informational only.
- `.env`, `automation_logs`, `broker_state`, and `audit_logs` are ignored and untracked.
- No broker orders were submitted during the audit.

The commit message supplied for this audit was:

`Audit and harden trading logic, risk controls, automation, and UI`

## What Has Been Manually Confirmed Previously

Across the conversation, the user manually confirmed at various times:

- Alpaca paper account connection.
- Paper BUY submission.
- Paper limit BUY submission.
- Paper order cancellation.
- Automatic stale limit-order cancellation while Streamlit was closed.
- Filled paper position reconciliation.
- Manual paper exit flow.
- An automatic paper BUY.
- An automatic paper SELL.
- Per-position saved exit settings.
- Post-trade review.
- Shadow-decision recording.
- Paper journal/performance surfaces.
- Worker start/stop behavior after fixes.

Some of those confirmations occurred before the latest quality-audit changes, so the current audited build still needs the planned multi-week paper run.

## Known Limitations And Honest Uncertainty

No software review can provide literal 100% certainty for unattended trading.

Remaining real-world uncertainty includes:

- Alpaca outages, rate limits, delayed responses, and SDK behavior.
- Partial fills and fill timing.
- Spread, slippage, market impact, and gap behavior beyond OHLC assumptions.
- Free IEX market-data coverage versus consolidated volume.
- Computer sleep, Windows restarts, internet loss, and process termination.
- Multiple weeks of worker restart/recovery behavior.
- Strategy performance changing after the historical sample.
- Earnings/calendar risk is not fully connected.
- The scanner is bounded, not an autonomous market-wide recommender.
- Automated live entries have not been validated and should remain off.

## Current Next Phase

Run live Alpaca paper trading for at least two weeks.

## Robust Strategy-Input Search Added (2026-07-11)

The recommended strategy-input search was strengthened to reduce overfitting without adding daily workflow clutter:

- Candidate settings are rewarded when nearby settings are also profitable, instead of rewarding one isolated best value.
- The development data is checked in separate chronological rolling periods.
- The final 20% of price history is locked while strategy and settings are selected. Only the selected winner is evaluated on that final period.
- Completed trades are stressed at 0, 5, 10, and 20 basis points of slippage per side.
- Results are broken down by rising, sideways, and falling trends and by higher/lower volatility.
- Completed trade outcomes are resampled with a deterministic bootstrap to estimate a weak-case return, loss probability, and drawdown.
- Confidence is reduced to account for the number of setting combinations searched. This is an approximate trade-level statistical check and is labeled as insufficient when there are too few completed trades.
- A manual other-ticker test can apply the selected settings unchanged to up to eight real tickers. It never re-optimizes those comparison tickers.

The normal recommendation remains one compact table. Detailed robustness evidence is under `Why this recommendation`; other-ticker validation is under `Test these settings on other tickers`. The search does not change account risk limits, broker access, order mode, credentials, the Kill Switch, or settings automatically. RSI remains a possible future deterministic strategy variable and was not added in this batch.

The optimizer is now run explicitly with `Run Strategy Input Search` instead of a persistent checkbox. Its result is saved through ordinary Streamlit reruns and marked stale when the ticker, source, interval, history, market bars, strategy inputs, material account-equity bucket, risk limits, older-data split, or candidate count changes. A stale recommendation cannot be applied or tested on other tickers until the search is run again. Alpaca history choices now extend through `2y` for `1h` bars and `5y` for `4h` bars; Yahoo intraday choices remain shorter.

New Trade includes `Add or Update Buy Setup`, which saves the exact ticker, interval, strategy, strategy inputs, risk limits, price source, and paper-order instructions currently being researched. Positions & Queue contains the durable Buy watchlist management surface: queue status, selected-setup controls, Repeat after exit, saved details, pausing, resuming, and removal. Ticker + interval + strategy identifies a row, so two strategies can monitor the same ticker. The worker independently records `Waiting for BUY`, `Blocked`, `Paused`, or `Order sent`, and disables a one-time row after sending its order. A bounded in-memory bar cache prevents full price histories from being downloaded every 15 seconds. `Max automatic buys this session` still limits submissions. `Allowed symbols` remains an optional whitelist and does not create watchlist rows.

Selecting a queued row opens `Saved setup details`, an expanded table showing the complete saved price-data setup, strategy inputs, RSI rule, risk limits, order type and limit instructions, cancellation and re-entry behavior, automation limits, and the initial break-even/ATR trailing exit plan. It also states that sizing uses the current Alpaca paper account and order repricing uses the latest available Alpaca IEX trade at execution.

Queued automatic BUY setups require `Ticker (Alpaca)`. Strategy signals are calculated from cached completed bars, with history refreshes bounded by interval. Each worker cycle makes one lightweight batched Alpaca latest-trades request for all enabled queue symbols and uses those current available IEX trade prices to reprice and size any BUY intent. If a symbol has no valid latest trade or the request fails, that queued order is blocked instead of using a stale cached price.

The New Trade `Automation readiness` table now treats the Buy watchlist as a background-worker workflow. It explicitly reports whether the worker is running, how many saved setups are enabled, their current statuses, and the worker's actual last check/action. The Streamlit page's 15-second rerun is not presented as a queue check. Closed-market limit orders are allowed when `Allow limit buys outside market hours` is checked and the selected order style is not `Market`; this permission does not bypass the saved strategy's required BUY rules.

Sidebar automation controls were simplified. Selecting `Paper trading - send orders to Alpaca paper` now automatically enables the configured Alpaca paper account, so the redundant `Use Alpaca paper account` checkbox was removed. `Enable Automation` was renamed `Allow automatic paper buys` and moved into `Background Automation`; it controls BUY permission only, not exits. The sidebar now explains that an open Streamlit page can check the loaded ticker and saved exits, while the Background Worker is required for monitoring after Streamlit closes and for the durable Buy watchlist.

Open-position stop labels distinguish the order-time plan from the operational stop: `Planned stop before fill` is calculated from the strategy reference price, while `Fill-adjusted initial stop` applies the saved entry ATR distance to Alpaca's actual average fill. The position editor's `Initial stop ATR multiplier` now genuinely recalculates the saved initial-risk distance, planned stop, fill-adjusted stop, and R thresholds. It shows the projected fill-adjusted stop before saving. Explicitly saving a wider multiplier may loosen the active stop; tightening and loosening quick actions use the same calculation.

Alpaca's 4-hour bars may arrive in small pages even when a large page limit is requested. The market-data fetcher now follows up to 200 pages and refuses to use a silently truncated dataset. This fixed an optimizer failure where five-year 4-hour history stopped in 2023, causing the shared interval-comparison period to display as 0.0 years and produce no candidates.

## Visual Design System (2026-07-11)

The app uses a professional trading-console design system built around a deep-navy background, compact spacing, thin borders, restrained six-pixel corners, and one semantic palette. Green is used for approved actions and positive states; blue for research and selection; amber for warnings and strategy exits; red for blocks, losses, stops, and the Kill Switch; teal for ATR information. Main command navigation reads as tabs, the daily account state uses one compact status strip, cards and controls use dense typography, and chart colors follow the same semantics. The sidebar remains a slim home for global controls while ticker-, trade-, and position-specific work stays on the relevant main page. Numbered workflow headings were removed where tabs and visual hierarchy already provide orientation. Shared styling and chart colors live in `agentloop_trader/ui_theme.py`; base Streamlit colors live in `.streamlit/config.toml`. The overhaul intentionally changed presentation rather than trading, risk, or broker logic.

Recommended operating mode:

- Workspace: Daily Trading Screen.
- Main page for management: Positions & Queue.
- Order mode: Paper trading - send orders to Alpaca paper.
- Automation: Auto exits - app sells paper positions.
- Background worker: Running during the period positions should be managed.
- Entries: normally reviewed and sent manually.
- Kill Switch: off during intended operation; turn it on to stop new automation actions.

Daily checks:

1. Positions & Queue: confirm each position's exit plan and each queued setup's saved strategy, status, and repeat setting.
2. Alpaca: compare app positions and waiting orders with the Alpaca dashboard.
3. Paper Review: inspect daily activity, fills, exits, cancellations, and performance.
4. Full Records and Evidence only when investigating a decision, trigger, or mismatch.
5. Confirm audit logs include the exact reason for every automated action.

Scenarios that should be deliberately observed during paper testing:

- Automatic exit from the original stop.
- Automatic exit from the strategy exit line.
- Break-even activation after highest price reaches +1R.
- ATR trailing activation after highest price reaches +2R.
- Trigger remains tightened after price pulls back.
- Limit BUY expiration and auto-cancel.
- Partial fill handling.
- App closed while worker continues.
- Worker stop/start and computer restart recovery.
- Temporary Alpaca/data failure causes a blocked cycle, not an order based on empty state.
- Multiple positions with independent saved exit plans.
- Optional addition to an existing filled ticker position without duplicate waiting orders.

After the paper period:

1. Review every unexpected or missing action.
2. Compare expected trigger prices with actual order timestamps/fills.
3. Review optimizer recommendations against realized paper behavior.
4. Select conservative default strategy and risk settings.
5. Run another full automated and manual regression check.
6. Complete a live-readiness review.
7. If proceeding, begin with very small capital, manual BUY approval, and automated exits.
8. Do not enable automatic live entries until manual-live and auto-exit behavior have been observed reliably.

## Strategy Candidate Verdicts (July 12, 2026)

The strategy input search now assigns one deterministic verdict to its recommended strategy and settings:

- Strong Candidate: broad support across unseen buy-and-hold comparisons, after-cost expectancy, nearby settings, trading-cost stress, best-trade dependence, account drawdown, interval-adjusted trade count, rolling periods, return per drawdown, and market conditions.
- Promising Candidate: enough support for paper testing, with modest or incomplete evidence but no serious contradiction.
- Research Only: too few trades or evidence that remains incomplete or inconsistent.
- Reject: a core test clearly failed, such as negative after-cost expectancy, material buy-and-hold underperformance in both unseen periods, fragile nearby settings, or dependence on one exceptional trade.

The trade-count thresholds adapt to bar interval: 1-hour requires 20 trades for Promising and 40 for Strong; 4-hour requires 12 and 25; daily requires 8 and 15. Missing evidence is treated as uncertain rather than failed. The LLM may explain this result later, but it does not choose or override the verdict.

The main UI shows only the verdict and next step. Detailed supporting, uncertain, and contradictory evidence stays inside `Why this recommendation`.

## Alpaca Equity Fee Accounting (July 13, 2026)

The app uses Alpaca's U.S. equity brokerage fee schedule revised July 1, 2026:

- Direct self-directed API commission assumption: 0%.
- SEC transaction fee on sells: 0.0000206 times trade value.
- FINRA Trading Activity Fee on sells: $0.000195 per share, capped at $9.79 per trade.
- FINRA CAT fee on buys and sells: $0.000003 per executed equivalent share for NMS equities.

Alpaca aggregates each fee type by account and trading day, rounds each daily fee total up to the nearest cent, and posts the charge at day-end. Since a preview or backtest cannot know unrelated account activity, the app uses a conservative per-order estimate that rounds each applicable fee component upward.

Fee behavior in the app:

- Backtest trade P&L, final equity, returns, drawdown, profit factor, strategy comparisons, optimizer results, and buy-and-hold benchmarks are net of estimated Alpaca fees.
- Historical trade rows retain gross P&L, estimated fees, and net P&L.
- Risk sizing and dollars-at-risk include estimated fees if the stop is hit.
- Available-cash checks include the estimated buy fee.
- Paper and live Alpaca order previews show estimated order value, fees, and cash needed or net proceeds.
- Paper Review estimates what filled paper orders would have cost live.
- Alpaca paper balances remain unchanged because Alpaca paper trading does not deduct regulatory fees. The app labels paper fee figures as live-equivalent estimates.
- Spread, slippage, market impact, taxes, ADR custody fees, margin interest, partner-specific commissions, and optional Alpaca Elite routing charges are not silently included in the base fee model.

The full suite passed after this integration: 286 tests.

## Alpaca Crypto Support (July 13, 2026)

The app now treats crypto as an explicit asset type rather than pretending it is a stock:

- The daily UI can select Stocks or Crypto. Crypto uses Alpaca and normalizes Bitcoin to `BTC/USD`.
- Alpaca crypto history and latest trades use the official `/v1beta3/crypto/us` data endpoints.
- Completed crypto bars use elapsed UTC time and run 24/7; stock bars keep stock-session completion rules.
- The four deterministic strategies, backtests, optimizer, risk sizing, buy watchlist, broker previews, paper orders, and background exits support fractional crypto quantities.
- Crypto order previews require GTC or IOC and show a conservative Tier 1 taker fee plus the possible maker fee for limit orders.
- Crypto backtests and buy-and-hold comparisons use conservative Tier 1 taker fees on both sides. Actual crypto rates vary with maker/taker status and trailing 30-day volume.
- The worker batches stock and crypto latest-price reads separately. Stock orders respect stock market hours; crypto orders and exits do not.
- Live crypto orders use the existing Alpaca live-account wiring and remain subject to the same live environment gates. The background worker remains paper-only.

The crypto integration checkpoint passed 296 tests. No broker order was submitted while building or testing it.

## Repeating Buy Watchlist Setups (July 13, 2026)

Buy watchlist setups now have an explicit `Repeat After Exit` setting, shown as On or Off in the queue and in Saved setup details.

- Off preserves the original one-time behavior: the setup disables itself after sending one buy order.
- On remains durably enabled across repeated buy, position, and exit cycles.
- While its buy order or position is active, the worker waits and does not submit another buy for that setup.
- After the order or position finishes, the prior BUY signal must first turn off. This prevents immediate re-entry from the same unchanged signal or completed bar.
- Once the signal resets and the saved re-entry cooldown has elapsed, the setup waits for a new BUY signal and may trade again.
- The repeating setup remains On until the user pauses, updates, or removes it. Existing risk limits, duplicate-order checks, the Kill Switch, automation mode, and the worker-session automatic-buy cap still apply.

The repeat lifecycle and one-time fallback passed the full 299-test suite. No broker order was submitted during testing.

## Queued-Only Automatic Entries (July 13, 2026)

Automatic entry sources are now deliberately separated from research:

- The ticker currently open in the sidebar is research/manual-only. A TRADE result can still be sent with the explicit `Send Paper Buy to Alpaca` button, but it can never trigger an automatic BUY.
- Automatic BUY orders can originate only from enabled setups explicitly saved in the Buy watchlist.
- The visible automation choice is `Auto exits and queued buys`, with a separate `Allow queued buys` permission. This permission does not control automatic exits.
- Only the Background Worker monitors queued buys. If the Buy watchlist is empty, the worker reports that no automatic buys are monitored and does not fall back to its last saved sidebar ticker.
- The open Streamlit page timer manages saved exits and stale limit-order cancellation only. It has no automatic BUY submission path.
- Auto exits remain independent and continue to use each open position's saved exit plan.

Regression tests prove that an empty queue cannot call the worker's entry sender and that the Streamlit page contains no automatic-entry submission event.

## Positions And Queue Workspace (July 13, 2026)

The command-center workflow now separates setup creation from ongoing management:

- `Open Positions` was renamed `Positions & Queue`.
- New Trade keeps only `Add or Update Buy Setup`, because that action saves the ticker and inputs currently being researched.
- Positions & Queue contains the Buy watchlist table, `Manage queued setup`, per-setup Repeat after exit control, Saved setup details, pause/resume, removal, and queue automation status.
- The queue remains visible and manageable even when there are no open Alpaca positions.
- Open-position exit management remains on the same page, making Positions & Queue the daily surface for everything already being monitored by automation.

## Worker State Restoration After Sleep (July 13, 2026)

A fresh Streamlit session previously rendered the Automation selector at its Manual default before reading the existing worker control file. The page then wrote `Manual review only` and `enabled: false` back to the worker, leaving the Python process alive and heartbeating but in `Watching only` mode. Windows Sleep exposed this because the worker resumed normally, then reopening Streamlit disabled its actions.

The app now reads persisted worker control and heartbeat state before rendering sidebar automation widgets. When a worker record is present, a fresh UI restores its saved Automation mode, Kill Switch, and queued-buy permission. With no worker record, the safe default remains Manual. A stale-but-present worker record suppresses the open-page automation timer and displays `Needs attention` until the heartbeat resumes or the worker is stopped. The sidebar now states three separate facts: worker-process state, heartbeat freshness/timestamp, and whether automation actions are On or Off.

## Manual Paper Orders (July 13, 2026)

New Trade now has a collapsed `Manual paper order - no BUY signal required` ticket for discretionary entries. It is separate from a strategy-generated BUY:

- The user chooses dollar amount or quantity, market or exact limit price, and the initial ATR stop multiplier.
- Strategy entry rules are intentionally bypassed, but the Kill Switch, account risk, order-size, portfolio exposure, symbol concentration, available-cash, duplicate-position, and open-order checks still apply.
- The order uses the selected ticker or crypto pair and supports fractional crypto quantities with GTC time in force.
- The preview clearly shows the risk-adjusted quantity, planned buy price, initial stop, order value, and estimated risk.
- The user may save Auto exit On or Off. When On, the manual entry saves the current interval, selected exit strategy, sell exit length, ATR stop, break-even, and trailing-stop settings for the worker.
- The Alpaca page's existing buy action was relabeled `Approved strategy paper buy` so it is clear that it still requires a strategy-generated trade intent.

The manual-order checkpoint passed 308 tests. No broker order was submitted during development or testing.

## Backtest Daily-Loss Reset Fix (July 13, 2026)

The historical strategies previously passed cumulative profit and loss from the beginning of the entire test into the Max daily loss control. After cumulative losses crossed that limit, calculated quantity stayed at zero for every later bar. This appeared most clearly in a BTC/USD 5-minute, one-month test: trades stopped permanently at trade 82 around June 17 even though price history continued through July 13.

All four backtests now reset daily profit and loss by trading date before applying the daily-loss limit:

- Stocks use the America/New_York trading date.
- Crypto uses the UTC calendar date because it trades continuously.
- The limit is calculated from that day's starting equity, matching the live deterministic risk logic.
- Cumulative account results still flow normally into equity, drawdown, sizing, fees, and performance statistics; only the daily-loss pause resets.

A read-only replay of the exact BTC/USD history and displayed pullback settings reproduced the old result at 82 trades ending June 17. The corrected logic generated 503 trades through the end of the available data. The full checkpoint passed 310 tests. No broker order was submitted.

## RSI Mean-Reversion Scalp Strategy (July 13, 2026)

A fifth deterministic strategy, `RSI mean-reversion scalp`, was added for short-interval trading, especially 5-minute and 15-minute bars.

- Long-only v1 uses completed bars.
- A possible buy is armed when RSI reaches the selected low level (default 30) or falls the selected number of points from its recent RSI high (default 40 points over 24 bars).
- The app buys only after RSI rebounds from the lowest RSI reached while the setup was armed (default 3 points) and price closes above the prior completed bar.
- The RSI exit is the lower of the saved setup low plus the selected recovery (default 35 points) or the selected RSI sell cap (default 70).
- Standard and emergency modes retain ATR initial-stop, break-even, and trailing-stop protection. The optional maximum holding period defaults to 100 bars and can be turned off.
- RSI mean-reversion scalp now offers Standard ATR stop, Emergency ATR stop, and No price stop backtest modes. The normal screen runs only the selected mode. A separate comparison button runs all three against cached bars. No price stop is research-only, sizes from ticker allocation, and cannot be queued for automated execution.
- The RSI setup low, RSI at entry, calculated RSI exit, and all strategy inputs are saved with queued and submitted entries so the background worker can manage each position without relying on current sidebar settings.
- The strategy is included in Strategy comparison, walk-forward evaluation, the bounded input optimizer, research reads, saved queue details, charts, and per-position exit management.
- When this strategy is selected, Run Strategy Input Search compares 5-minute, 15-minute, and 1-hour data. The other strategies continue to use daily, 4-hour, and 1-hour comparisons.
- The separate `Require RSI 50-70 for BUY` option remains available only to the four trend strategies; it is not layered onto the RSI scalp.

No broker order was submitted during development or testing.

## Manual Position Exit Ownership And BUY-Level Visibility (July 14, 2026)

Manual paper buys no longer silently inherit the selected strategy's sell line. A manual order now defaults to `ATR protection only`, which uses the saved initial ATR stop, break-even protection, and ATR trailing stop. The manual-order form and each open position also allow an explicit `Strategy exit + ATR protection` choice. Existing saved positions without the new field retain their prior strategy-managed behavior for backward compatibility.

When a strategy is attached to an open position, the app shows its current required BUY conditions and blocks saving a strategy exit that would immediately sell the position because price is already below that strategy line. New Trade and queued setups now show current values, exact thresholds where the strategy has a real price threshold, distance to that threshold, and honest state descriptions for path-dependent or RSI rules. The detached worker persists those queue snapshots while continuing to cache completed historical bars and use a lightweight latest Alpaca price for order pricing.

Regression coverage includes the observed manual-position failure mode: an ATR-only IBM position around $218 must ignore an unrelated strategy exit near $288 and retain its ATR stop near $206. Existing strategy-managed positions still use their strategy exit unless the user changes the exit method.

## Entry-Candle Profit Protection Correction (July 14, 2026)

A manual WYFI position exposed an intrabar timing defect. The position filled partway through a 4-hour candle, but the exit evaluator treated that candle's full high, including price movement before the fill, as profit earned after entry. This falsely activated the +1R break-even stop even though the position was near 0R.

The shared UI/worker evaluator now excludes the full entry candle when its timestamp begins before the recorded fill. Until a post-entry bar is available, its high-water mark uses only the actual fill and prices observed after the fill. Any previously saved break-even or ATR-trail trigger that is unsupported by the corrected R calculation is discarded, and corrected state is allowed to move downward instead of only ratcheting upward. Manual entry records also persist the actual fill timestamp and initialize their high-water mark from the planned entry rather than a pre-fill quote.

Regression coverage reproduces the WYFI case: average fill $32.09, initial risk $3.53, current price $32.13, and a pre-fill candle high of $38.87. The corrected result is approximately 0.01R, no break-even stop, and a $28.56 fill-adjusted initial stop. The full checkpoint passed 322 tests. No broker order was submitted.

## Instructions For The Next Codex Conversation

1. Read this entire file before making recommendations.
2. Inspect the current Git status and current code; do not assume this state file supersedes newer code.
3. Do not overwrite or revert user changes.
4. Run tests after meaningful logic changes.
5. For strategy/risk changes, add mathematical and financial-invariant tests.
6. For broker/worker changes, test fail-closed behavior, duplicate prevention, and persistence.
7. Keep the daily UI concise. Put investigation records in Full Records and Evidence or collapsed sections.
8. Use exact UI labels in all test instructions.
9. Never expose `.env` values.
10. Never submit an Alpaca order while testing unless the user explicitly authorizes that exact paper/live action.
11. Treat paper trading as a practical test environment, not as a reason to build repetitive approval ceremony.
12. Be honest about residual uncertainty before live capital.
