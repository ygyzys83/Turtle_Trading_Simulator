import pandas as pd

from agentloop_trader.evaluation import synthetic_ohlc_frame
from agentloop_trader.llm_research import LLMResearchConfig
from agentloop_trader.parameter_loop import optimize_strategy_families
from agentloop_trader.strategy_recommendation import (
    ACTION_LEVEL,
    RecommendationLoopStore,
    build_strategy_search_evidence,
    recommendation_summary_records,
    recommendation_run_matches,
    run_strategy_recommendation_loop,
)


CURRENT_SETTINGS = {
    "strategy_type": "pullback",
    "entry_window": 20,
    "exit_window": 10,
    "atr_stop_multiplier": 1.5,
    "moving_average_window": 50,
    "pullback_average_length": 20,
    "momentum_turn_length": 5,
    "risk_per_trade_pct": 0.5,
}


def _search_result():
    data = synthetic_ohlc_frame(n=500, seed=73)
    data.index = pd.date_range("2023-01-01", periods=len(data), freq="4h")
    return optimize_strategy_families(
        market_data_by_interval={"4h": ("500 bars", data)},
        strategy_intervals={
            "breakout": ("4h",),
            "pullback": ("4h",),
            "trendline": ("4h",),
            "trendline_retest": ("4h",),
        },
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=1,
        max_local_candidates_per_strategy=0,
        bootstrap_samples=10,
    )


class ScriptedClient:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, schema):
        self.calls += 1
        if self.calls == 2:
            return {
                "accepted": False,
                "primary_objection": "The result may depend on a narrow environment.",
                "evidence_ids": ["C1"],
                "required_change": "Use more cautious language.",
            }
        return {
            "action": "USE FOR PAPER TEST",
            "recommended_candidate_id": "C1",
            "alternative_candidate_id": "C2",
            "assessment": "This candidate has the clearest support across the supplied historical evidence.",
            "why_it_may_work": "Its result appears across more than one part of the history and does not rely only on one exact setting.",
            "primary_concern": "Historical behavior may not repeat.",
            "alternative_view": "The alternative may be useful if its weaker return comes with more consistent behavior.",
            "next_action": "Review the current setup before a paper test.",
            "evidence_ids": ["C1", "C2", "M1", "M2"],
        }


def test_built_in_recommendation_uses_hashed_deterministic_evidence():
    result = _search_result()
    evidence = build_strategy_search_evidence("TEST", result)
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )

    assert len(evidence.evidence_hash) == 20
    assert len(evidence.candidates) >= 4
    assert run.recommendation.evidence_hash == evidence.evidence_hash
    assert run.recommendation.action in ACTION_LEVEL
    assert set(run.recommendation.evidence_ids) <= set(evidence.facts)


def test_agent_loop_runs_one_draft_review_and_revision_and_cannot_promote():
    result = _search_result()
    built_in = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )
    client = ScriptedClient()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="ollama", model="test"),
        client,
    )

    assert client.calls == 3
    assert ACTION_LEVEL[run.recommendation.action] <= ACTION_LEVEL["WATCH FOR A BETTER SETUP"]
    assert run.recommendation.candidate_id == "C1"
    assert not run.recommendation.used_fallback
    assert run.reviewer_response["accepted"] is False


def test_invalid_agent_evidence_falls_back_and_store_keeps_audit_record(tmp_path):
    class InvalidClient:
        def generate_json(self, prompt, schema):
            return {
                "action": "USE FOR PAPER TEST",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "",
                "assessment": "The candidate appears useful.",
                "why_it_may_work": "The supplied evidence is supportive.",
                "primary_concern": "None.",
                "alternative_view": "No alternative was selected.",
                "next_action": "Trade.",
                "evidence_ids": ["INVENTED"],
            }

    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="ollama", model="test"),
        InvalidClient(),
    )
    store = RecommendationLoopStore(tmp_path / "recommendations.jsonl")
    store.append(run)

    assert run.recommendation.used_fallback
    assert run.recommendation.provider == "deterministic"
    assert store.read_recent(1)[0]["recommendation"]["evidence_hash"] == run.recommendation.evidence_hash


def test_unknown_provider_falls_back_without_calling_an_agent():
    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="unexpected", model="test")
    )

    assert run.recommendation.used_fallback
    assert run.recommendation.provider == "deterministic"
    assert "Unsupported research provider" in run.recommendation.error


def test_execution_language_or_invented_numbers_cannot_reach_the_final_recommendation():
    class UnsafeClient:
        def generate_json(self, prompt, schema):
            return {
                "action": "USE FOR PAPER TEST",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "",
                "assessment": "Buy now because the result was 99 percent.",
                "why_it_may_work": "The result looks strong.",
                "primary_concern": "None.",
                "alternative_view": "No alternative was selected.",
                "next_action": "Submit the order.",
                "evidence_ids": ["C1"],
            }

    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="ollama", model="test"),
        UnsafeClient(),
    )

    assert run.recommendation.used_fallback
    assert run.recommendation.provider == "deterministic"
    assert "Strategy Analyst" in run.recommendation.error
    assert "99" in run.recommendation.error
    assert run.analyst_draft["recommended_candidate_id"] == "C1"


def test_model_prompt_uses_facts_instead_of_the_high_precision_internal_catalog():
    class PromptInspectingClient(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_json(self, prompt, schema):
            self.prompts.append(prompt)
            return super().generate_json(prompt, schema)

    result = _search_result()
    client = PromptInspectingClient()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="gemini", model="test"),
        client,
    )

    assert not run.recommendation.used_fallback
    assert '"candidate_index"' in client.prompts[0]
    assert '"candidate_catalog"' not in client.prompts[0]
    assert '"nearby_top_percent"' not in client.prompts[0]
    assert '"complete_history_trades"' not in client.prompts[0]


def test_reviewer_invented_number_is_rejected_and_saved_for_diagnosis():
    class UnsafeReviewerClient(ScriptedClient):
        def generate_json(self, prompt, schema):
            self.calls += 1
            if self.calls == 2:
                return {
                    "accepted": False,
                    "primary_objection": "The candidate has a 99 percent failure rate.",
                    "evidence_ids": ["C1"],
                    "required_change": "Use more cautious language.",
                }
            return {
                "action": "WATCH FOR A BETTER SETUP",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "C2",
                "assessment": "This candidate has some support, but it still needs confirmation.",
                "why_it_may_work": "The cited evidence includes more than one price section.",
                "primary_concern": "Historical behavior may not repeat.",
                "alternative_view": "The alternative deserves comparison.",
                "next_action": "Review the current setup before a paper test.",
                "evidence_ids": ["C1", "C2"],
            }

    result = _search_result()
    client = UnsafeReviewerClient()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="gemini", model="test"),
        client,
    )

    assert run.recommendation.used_fallback
    assert "Skeptical Reviewer" in run.recommendation.error
    assert "99" in run.recommendation.error
    assert run.reviewer_response["accepted"] is False


def test_agent_may_quote_numbers_that_exist_in_cited_evidence():
    class EvidenceNumberClient:
        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, schema):
            self.calls += 1
            if self.calls == 2:
                return {
                    "accepted": True,
                    "primary_objection": "No material concern was omitted.",
                    "evidence_ids": ["C1"],
                    "required_change": "None.",
                }
            return {
                "action": "WATCH FOR A BETTER SETUP",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "C2",
                "assessment": "The candidate used a 4 hour interval, but the result still needs confirmation.",
                "why_it_may_work": "The cited evidence includes more than one price section.",
                "primary_concern": "Historical behavior may not repeat.",
                "alternative_view": "The alternative deserves comparison.",
                "next_action": "Review the current setup before a paper test.",
                "evidence_ids": ["C1", "C2"],
            }

    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="ollama", model="test"),
        EvidenceNumberClient(),
    )

    assert not run.recommendation.used_fallback
    assert run.recommendation.action == "WATCH FOR A BETTER SETUP"


def test_rejected_skip_draft_cannot_be_promoted_by_editor():
    class RejectedSkipClient:
        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, schema):
            self.calls += 1
            if self.calls == 2:
                return {
                    "accepted": False,
                    "primary_objection": "The evidence is weaker than the draft suggests.",
                    "evidence_ids": ["C1"],
                    "required_change": "Keep the final action at skip.",
                }
            return {
                "action": "WATCH FOR A BETTER SETUP",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "C2",
                "assessment": "The candidate is not convincing enough for a paper test.",
                "why_it_may_work": "Some historical evidence is supportive.",
                "primary_concern": "The result may not be repeatable.",
                "alternative_view": "The alternative also has unresolved concerns.",
                "next_action": "Skip this result and review another setup.",
                "evidence_ids": ["C1", "C2"],
            } if self.calls == 3 else {
                "action": "SKIP",
                "recommended_candidate_id": "C1",
                "alternative_candidate_id": "C2",
                "assessment": "The candidate is not convincing enough for a paper test.",
                "why_it_may_work": "Some historical evidence is supportive.",
                "primary_concern": "The result may not be repeatable.",
                "alternative_view": "The alternative also has unresolved concerns.",
                "next_action": "Skip this result and review another setup.",
                "evidence_ids": ["C1", "C2"],
            }

    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST",
        result,
        LLMResearchConfig(provider="ollama", model="test"),
        RejectedSkipClient(),
    )

    assert not run.recommendation.used_fallback
    assert run.recommendation.action == "SKIP"


def test_recommendation_summary_keeps_alternative_explanation_out_of_value_column():
    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )

    alternative = next(
        row for row in recommendation_summary_records(run) if row["Item"] == "Best alternative"
    )

    assert alternative["Value"] != run.recommendation.alternative_view
    assert run.recommendation.alternative_view in alternative["Plain English"]


def test_recommendation_match_rejects_stale_streamlit_values_and_accepts_current_run():
    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )

    assert recommendation_run_matches(run, run.recommendation.evidence_hash)
    assert not recommendation_run_matches(run, "different")
    assert not recommendation_run_matches({"recommendation": {}}, run.recommendation.evidence_hash)


def test_recommendation_store_safe_append_reports_success(tmp_path):
    result = _search_result()
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )
    store = RecommendationLoopStore(tmp_path / "recommendations.jsonl")

    assert store.append_safely(run) == ""
    assert len(store.read_recent()) == 1
    assert store.read_recent(0) == []
