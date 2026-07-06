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


def test_readme_explains_agentic_portfolio_narrative():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Portfolio Narrative" in text
    assert "human-in-the-loop guardrails" in text
    assert "deterministic risk policy" in text
    assert "Portfolio Evidence" in text
