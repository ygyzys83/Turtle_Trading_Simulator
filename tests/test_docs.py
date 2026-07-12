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

    assert "It answers which of those four exact strategies fits the ticker now; it does not search for better settings." in text
    assert "This table does not search for better settings" in text
    assert "Use this as the strongest candidate for paper testing" in text
