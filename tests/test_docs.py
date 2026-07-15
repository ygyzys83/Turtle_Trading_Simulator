from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_safety_doc_exists_and_names_non_negotiable_controls():
    text = (ROOT / "docs" / "PRODUCTION_SAFETY.md").read_text(encoding="utf-8")

    assert "Non-Negotiable Controls" in text
    assert "kill-switch" in text
    assert "not agent-modifiable" in text


def test_live_deployment_checklist_exists_and_blocks_unattended_first_session():
    text = (ROOT / "docs" / "LIVE_DEPLOYMENT_CHECKLIST.md").read_text(encoding="utf-8")

    assert "First Live Session" in text
    assert "No unattended operation" in text
    assert "Unattended Live Criteria" in text
    assert "Create Live Mode Lockfile" in text


def test_operator_runbook_names_halt_recovery_and_evidence_export():
    text = (ROOT / "docs" / "OPERATOR_RUNBOOK.md").read_text(encoding="utf-8")

    assert "Emergency Halt" in text
    assert "Recovery" in text
    assert "Evidence Export" in text
    assert "Local Simulator Test" in text


def test_paper_test_plan_includes_local_simulator_check():
    text = (ROOT / "docs" / "PAPER_TEST_PLAN.md").read_text(encoding="utf-8")

    assert "Local Simulator Check" in text
    assert "Simulate Alpaca Paper Fill" in text
    assert "Record Simulated Exit Readiness" in text


def test_readme_explains_product_and_backtest_contract_without_ui_story_copy():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Portfolio Narrative" not in text
    assert "## Backtest Assumptions" in text
    assert "## Current Modules" in text
    assert "deterministic strategy and risk code controls order eligibility" in text
    assert "Daily Trading Screen" in text
    assert "Full Records and Evidence" in text


def test_strategy_input_search_is_explicit_and_alpaca_intraday_history_is_bounded():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert '"Run Strategy Input Search"' in text
    assert '"Find recommended strategy inputs"' not in text
    assert '["1mo", "3mo", "6mo", "1y", "2y", "5y"]' in text
    assert '["1mo", "3mo", "6mo", "1y", "2y"]' in text
    assert '"Inputs changed. Run the search again."' in text


def test_ui_theme_uses_one_semantic_palette_and_streamlit_theme_file():
    theme_module = (ROOT / "agentloop_trader" / "ui_theme.py").read_text(encoding="utf-8")
    streamlit_theme = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert "TRADING_CONSOLE_CSS" in theme_module
    assert '"price": "#D8E0E9"' in theme_module
    assert '"entry": "#35C46A"' in theme_module
    assert '"exit": "#E3AA42"' in theme_module
    assert '"sell": "#FF6262"' in theme_module
    assert 'primaryColor = "#35C46A"' in streamlit_theme
    assert 'backgroundColor = "#0B1420"' in streamlit_theme
    assert "baseFontSize = 13" in streamlit_theme
    assert '--sidebar-bg: #0D1824' in theme_module
    assert 'background: #1F7A45' in theme_module
    assert '[data-testid="stToolbar"]' in theme_module
    assert 'height: 0 !important' in theme_module
    assert '[data-testid="stSidebarHeader"]' in theme_module
    assert 'initial_sidebar_state="expanded"' in (ROOT / "turtle_trading.py").read_text(encoding="utf-8")
    assert '[data-testid="stSidebarCollapsedControl"]' in theme_module
    assert 'transform: none !important' in theme_module
    assert '[class*="st-"]' not in theme_module
    assert '[data-testid="stIconMaterial"]' in theme_module
    assert 'font-family: "Material Symbols Rounded"' in theme_module
    assert 'overscroll-behavior-y: contain !important' in theme_module


def test_readme_uses_supervised_streamlit_launcher():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python run_app.py" in readme
    assert "prevents a second UI" in readme


def test_long_price_load_shows_three_explicit_progress_stages():
    app_text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "Step 1 of 3 - Download price history." in app_text
    assert "Step 2 of 3 - Run backtests." in app_text
    assert "Step 3 of 3 - Prepare results." in app_text
    assert "Wait for the finished results before changing sidebar inputs." in app_text


def test_daily_workspace_titles_are_not_numbered():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "import re" in text
    assert 'page_section("1. Daily command center"' not in text
    assert 'sub_section("1.1 Open positions"' not in text
    assert 'sub_section("1.3 New trade"' not in text
    assert 'page_section("2. Backtest"' not in text


def test_main_navigation_is_sticky_and_precedes_status_strip():
    app_text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")
    theme_text = (ROOT / "agentloop_trader" / "ui_theme.py").read_text(encoding="utf-8")

    assert 'with st.container(key="top_navigation")' in app_text
    assert app_text.index('with st.container(key="top_navigation")') < app_text.index("status_rows = compact_status_records(")
    assert app_text.count('"Command center page"') == 1
    assert ".st-key-top_navigation" in theme_text
    assert "position: sticky" in theme_text
    assert '[data-testid="stLayoutWrapper"]:has(> .st-key-top_navigation)' in theme_text
    assert '[data-testid="stMarkdownContainer"]:has(h1)' in theme_text


def test_strategy_decision_sections_have_plain_language_helpers():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "It answers which of those five exact strategies fits the ticker now; it does not search for better settings." in text
    assert "This table does not search for better settings" in text
    assert "The older 55% of prices finds useful regions" in text
    assert "the newer 25% chooses among those regions" in text
    assert "the latest 20% " in text
    assert "only reports what happened afterward" in text
    assert "RSI rules are not part of this search" in text


def test_long_ticker_load_shows_numerical_progress_through_backtests():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "load_progress_bar = st.progress(" in text
    assert "Downloading completed price bars (step 1 of 3)" in text
    assert "Running five backtests across" in text
    assert "Building decisions, tables, and charts (step 3 of 3)" in text
    assert 'load_progress_bar.progress(1.0, text=f"{ticker} is ready")' in text


def test_sidebar_risk_ranges_allow_smaller_limits_without_changing_defaults():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert '"Strategy risk per trade (%)",\n    0.25,\n    3.0,\n    0.5,\n    step=0.25' in text
    assert '"Max risk per trade (%)",\n    0.10,\n    5.0,\n    0.5,\n    step=0.05' in text
    assert '"Max new order size (%)",\n    2.0,\n    100.0,\n    5.0,\n    step=1.0' in text
    assert '"Max symbol concentration (%)",\n    2.0,\n    100.0,\n    5.0,\n    step=1.0' in text
    assert '"Max daily loss (%)",\n    0.25,\n    10.0,\n    2.0,\n    step=0.25' in text
    assert '"Max open positions",\n    1,\n    30,\n    20,\n    step=1' in text
    assert 'max(0.25, min(3.0, round(optimizer_risk_pct * 4) / 4))' in text


def test_daily_loss_replaces_automatic_buy_count_cap():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")
    worker = (ROOT / "agentloop_trader" / "worker.py").read_text(encoding="utf-8")

    assert "Max automatic buys this session" not in text
    assert "max_auto_buys_per_session" not in text
    assert "max_auto_buys_per_session" not in worker
    assert "Stops new BUY orders when today's Alpaca account loss reaches this percentage" in text
    assert "It does not turn on the Kill Switch" in text


def test_buy_watchlist_readiness_distinguishes_worker_state_from_market_hours():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "The Streamlit page timer does not monitor queued setups." in text
    assert "Only the background worker monitors the durable Buy watchlist." in text
    assert 'allow_limit_buys_outside_market_hours\n    and paper_buy_order_style != "Market"' in text


def test_sidebar_automation_controls_state_their_exact_scope():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert '"Allow queued buys"' in text
    assert '"Auto exits and queued buys"' in text
    assert '"Use Alpaca paper account"' not in text
    assert 'enable_alpaca_paper_orders = bool(execution_mode == "paper" and alpaca_config.paper)' in text
    assert "It does not control automatic exits." in text
    assert "The ticker currently open for research is never bought automatically." in text
    assert "Worker process: Running. Heartbeat: current" in text
    assert "Automation actions: {'On' if worker_actions_enabled else 'Off'}" in text


def test_fresh_streamlit_session_restores_running_worker_controls_before_rendering_widgets():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    control_read = text.index("saved_sidebar_control = sidebar_control_store.read()")
    automation_widget = text.index('automation_level_label = st.sidebar.selectbox(')
    assert control_read < automation_widget
    assert "automation_mode_for_new_ui_session(saved_sidebar_control, sidebar_worker_present)" in text
    assert "index=list(automation_level_options.keys()).index(saved_automation_label)" in text
    assert "saved_sidebar_control.full_automation_enabled and sidebar_worker_present" in text
    assert 'st.session_state["background_worker_enabled"] = sidebar_worker_present' in text


def test_streamlit_page_has_no_automatic_buy_submission_path():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert "elif auto_entry_status.ready:" not in text
    assert 'event_type="auto_paper_entry_submitted"' not in text


def test_buy_watchlist_creation_and_management_are_on_their_intended_pages():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert '["Positions & Queue", "Ideas", "New Trade", "Alpaca", "Paper Review"]' in text
    assert 'if command_center_view == "Positions & Queue":' in text
    assert 'if command_center_view == "Open Positions":' not in text
    assert text.count("render_current_setup_watchlist_action()") == 2
    assert text.count("render_buy_watchlist_manager()") == 2
    positions_panel = text[text.index("def render_open_positions_panel()") : text.index('if command_center_view == "Positions & Queue":')]
    assert positions_panel.index("render_buy_watchlist_manager()") < positions_panel.index("if not alpaca_positions:")


def test_position_stop_labels_distinguish_planned_price_from_fill_adjusted_stop():
    text = (ROOT / "turtle_trading.py").read_text(encoding="utf-8")

    assert '"Planned stop before fill"' in text
    assert '"Fill-adjusted initial stop"' in text
    assert '"Initial stop ATR multiplier"' in text
    assert "Projected fill-adjusted initial stop:" in text
    assert '"Stop loss at entry"' not in text
    assert '"Original stop"' not in text
