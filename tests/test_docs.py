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
