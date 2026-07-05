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

