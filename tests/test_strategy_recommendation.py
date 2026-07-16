import pandas as pd

from agentloop_trader.evaluation import synthetic_ohlc_frame
from agentloop_trader.llm_research import LLMResearchConfig
from agentloop_trader.parameter_loop import optimize_strategy_families
from agentloop_trader.strategy_recommendation import (
    ACTION_LEVEL,
    RecommendationLoopStore,
    build_strategy_search_evidence,
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
                "evidence_ids": ["E7"],
                "required_change": "Use more cautious language.",
            }
        return {
            "action": "USE FOR PAPER TEST",
            "thesis": "The supplied evidence supports a bounded paper test.",
            "primary_concern": "Historical behavior may not repeat.",
            "next_action": "Review the current setup before a paper test.",
            "evidence_ids": ["E1", "E2", "E5", "E7"],
        }


def test_built_in_recommendation_uses_hashed_deterministic_evidence():
    result = _search_result()
    evidence = build_strategy_search_evidence("TEST", result)
    run = run_strategy_recommendation_loop(
        "TEST", result, LLMResearchConfig(provider="deterministic")
    )

    assert len(evidence.evidence_hash) == 20
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
    assert ACTION_LEVEL[run.recommendation.action] <= max(
        0, ACTION_LEVEL[built_in.recommendation.action] - 1
    )
    assert not run.recommendation.used_fallback
    assert run.reviewer_response["accepted"] is False


def test_invalid_agent_evidence_falls_back_and_store_keeps_audit_record(tmp_path):
    class InvalidClient:
        def generate_json(self, prompt, schema):
            return {
                "action": "USE FOR PAPER TEST",
                "thesis": "Looks good.",
                "primary_concern": "None.",
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
                "thesis": "Buy now because the result was 99 percent.",
                "primary_concern": "None.",
                "next_action": "Submit the order.",
                "evidence_ids": ["E1"],
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
