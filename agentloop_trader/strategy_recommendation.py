from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from agentloop_trader.llm_research import LLMResearchConfig
from agentloop_trader.parameter_loop import MultiStrategySearchResult, StrategySearchResult


PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
ALLOWED_ACTIONS = ("USE FOR PAPER TEST", "WATCH FOR A BETTER SETUP", "SKIP")
ACTION_LEVEL = {"SKIP": 0, "WATCH FOR A BETTER SETUP": 1, "USE FOR PAPER TEST": 2}

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
        "recommended_candidate_id": {"type": "string"},
        "alternative_candidate_id": {"type": "string"},
        "assessment": {"type": "string"},
        "why_it_may_work": {"type": "string"},
        "primary_concern": {"type": "string"},
        "alternative_view": {"type": "string"},
        "next_action": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": [
        "action",
        "recommended_candidate_id",
        "alternative_candidate_id",
        "assessment",
        "why_it_may_work",
        "primary_concern",
        "alternative_view",
        "next_action",
        "evidence_ids",
    ],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "boolean"},
        "primary_objection": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "required_change": {"type": "string"},
    },
    "required": ["accepted", "primary_objection", "evidence_ids", "required_change"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class StrategySearchEvidence:
    ticker: str
    strategy: str
    interval: str
    settings: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    facts: dict[str, str]
    evidence_hash: str


@dataclass(frozen=True)
class StrategyRecommendation:
    created_at: str
    ticker: str
    strategy: str
    interval: str
    action: str
    candidate_id: str
    alternative_candidate_id: str
    assessment: str
    why_it_may_work: str
    primary_concern: str
    alternative_view: str
    next_action: str
    evidence_ids: tuple[str, ...]
    evidence_hash: str
    provider: str
    model: str
    used_fallback: bool = False
    error: str = ""


@dataclass(frozen=True)
class RecommendationLoopRun:
    recommendation: StrategyRecommendation
    facts: dict[str, str]
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    analyst_draft: dict[str, Any] = field(default_factory=dict)
    reviewer_response: dict[str, Any] = field(default_factory=dict)
    editor_response: dict[str, Any] = field(default_factory=dict)


class RecommendationLLM(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class OllamaRecommendationLLM:
    def __init__(self, config: LLMResearchConfig):
        self.config = config

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Use only the supplied evidence. Do not calculate or invent values. "
                        "You cannot authorize or submit an order. Return only the requested JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.config.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["message"]["content"])


class GeminiRecommendationLLM:
    def __init__(self, config: LLMResearchConfig):
        self.config = config

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not self.config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [{"text": "Use only supplied evidence. Never invent values or authorize an order."}]
            },
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.config.gemini_api_key},
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["candidates"][0]["content"]["parts"][0]["text"])


def _best_strategy_result(result: MultiStrategySearchResult) -> tuple[StrategySearchResult, Any, Any]:
    selected = result.best_strategy
    if selected is None or selected.best_result.best is None:
        raise ValueError("No strategy search result is available to review.")
    interval_result = next(
        row for row in selected.interval_results if row.interval == selected.best_interval
    )
    return selected, interval_result, selected.best_result.best


def _candidate_settings_text(settings: dict[str, Any]) -> str:
    labels = (
        ("entry_window", "buy lookback"),
        ("exit_window", "sell exit"),
        ("atr_stop_multiplier", "stop ATR"),
        ("moving_average_window", "trend filter"),
        ("pullback_average_length", "pullback average"),
        ("momentum_turn_length", "momentum turn"),
    )
    return "; ".join(f"{label} {settings[key]}" for key, label in labels if key in settings)


def _candidate_catalog(result: MultiStrategySearchResult) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    candidate_number = 1
    for strategy_result in sorted(result.strategy_results, key=lambda row: row.strategy_label):
        interval_result = strategy_result.interval_results[0]
        for within_strategy_rank, candidate in enumerate(strategy_result.best_result.candidates[:2], start=1):
            is_primary = within_strategy_rank == 1
            robustness = strategy_result.best_result.robustness
            locked = robustness.locked_test if robustness and is_primary else None
            diagnostics = robustness.diagnostics if robustness and is_primary else None
            dependency = robustness.regime_dependency if robustness and is_primary else {}
            candidate_id = f"C{candidate_number}"
            catalog[candidate_id] = {
                "strategy": strategy_result.strategy_label,
                "interval": strategy_result.best_interval,
                "within_strategy_rank": within_strategy_rank,
                "settings": dict(candidate.settings),
                "settings_text": _candidate_settings_text(candidate.settings),
                "older_trades": candidate.train_trades,
                "older_excess_percent": candidate.train_excess_return_percent,
                "newer_trades": candidate.test_trades,
                "newer_excess_percent": candidate.excess_return_percent,
                "latest_trades": locked.trades if locked else None,
                "latest_excess_percent": locked.excess_return_percent if locked else None,
                "complete_history_trades": interval_result.durability_trades if is_primary else None,
                "complete_history_annualized_excess_percent": (
                    interval_result.durability_annualized_excess_percent if is_primary else None
                ),
                "complete_history_maximum_decline_percent": (
                    interval_result.durability_max_drawdown_percent if is_primary else None
                ),
                "nearby_top_count": candidate.nearby_top_count,
                "nearby_settings_checked": candidate.plateau_neighbors,
                "nearby_top_percent": candidate.nearby_top_percent,
                "best_trade_removed_return_percent": (
                    diagnostics.best_trade_removed_return_percent if diagnostics else None
                ),
                "latest_price_behavior": dependency.get("current_regime"),
                "latest_behavior_match": dependency.get("current_match"),
                "price_behavior_summary": dependency.get("summary"),
            }
            candidate_number += 1
    return catalog


def build_strategy_search_evidence(ticker: str, result: MultiStrategySearchResult) -> StrategySearchEvidence:
    selected, _, candidate = _best_strategy_result(result)
    candidates = _candidate_catalog(result)
    facts: dict[str, str] = {}
    for candidate_id, row in candidates.items():
        latest_text = (
            "Latest-period results were not calculated for this secondary candidate."
            if row["latest_excess_percent"] is None
            else (
                f"The latest price section had {row['latest_trades']} completed trades and "
                f"{row['latest_excess_percent']:+.2f}% versus buy and hold."
            )
        )
        annualized_text = (
            "Complete-history results were not calculated for this secondary candidate."
            if row["complete_history_annualized_excess_percent"] is None
            else (
                "Complete-history annualized return was "
                f"{row['complete_history_annualized_excess_percent']:+.2f}% versus buy and hold."
            )
        )
        decline_text = (
            "Complete-history maximum decline was not calculated for this secondary candidate."
            if row["complete_history_maximum_decline_percent"] is None
            else (
                "Complete-history maximum decline was "
                f"{row['complete_history_maximum_decline_percent']:.2f}%."
            )
        )
        best_trade_text = (
            "The best-trade removal test was not calculated for this candidate."
            if row["best_trade_removed_return_percent"] is None
            else (
                "After removing the single best trade, allocated return was "
                f"{row['best_trade_removed_return_percent']:+.2f}%."
            )
        )
        behavior_text = (
            "Detailed price-behavior evidence was not calculated for this secondary candidate."
            if not row["price_behavior_summary"]
            else (
                f"The latest price behavior was {row['latest_price_behavior']}; its historical match was "
                f"{row['latest_behavior_match']}. {row['price_behavior_summary']}"
            )
        )
        facts[candidate_id] = (
            f"{row['strategy']} using {row['interval']} prices with {row['settings_text']}. "
            f"The older price section had {row['older_trades']} completed trades and "
            f"{row['older_excess_percent']:+.2f}% versus buy and hold. "
            f"The newer price section had {row['newer_trades']} completed trades and "
            f"{row['newer_excess_percent']:+.2f}% versus buy and hold. "
            f"{latest_text} {annualized_text} {decline_text} {best_trade_text} "
            f"{row['nearby_top_count']} of {row['nearby_settings_checked']} checked nearby settings ranked in "
            f"the top third ({row['nearby_top_percent']:.0f}%). {behavior_text}"
        )
    facts["M1"] = (
        "Every candidate was produced by the same deterministic search process. The list contains up to two "
        "high-ranked input regions from each searched strategy; the wording does not declare any candidate reliable."
    )
    facts["M2"] = (
        "The search includes estimated Alpaca fees. Separate detailed records contain additional price-slippage checks "
        "for the strongest result within each strategy."
    )
    identity = {"ticker": ticker.strip().upper(), "candidates": candidates, "facts": facts}
    evidence_hash = sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    return StrategySearchEvidence(
        ticker=ticker.strip().upper(),
        strategy=selected.strategy_label,
        interval=selected.best_interval,
        settings=dict(candidate.settings),
        candidates=candidates,
        facts=facts,
        evidence_hash=evidence_hash,
    )


def _deterministic_action(
    result: MultiStrategySearchResult,
    evidence: StrategySearchEvidence,
) -> tuple[str, str, str, str, str, str, str, str, tuple[str, ...]]:
    selected, interval_result, candidate = _best_strategy_result(result)
    robustness = selected.best_result.robustness
    locked = robustness.locked_test if robustness else None
    dependency = robustness.regime_dependency if robustness else {}
    latest_excess = locked.excess_return_percent if locked else 0.0
    selected_id = next(
        (
            candidate_id
            for candidate_id, row in evidence.candidates.items()
            if row["strategy"] == selected.strategy_label and row["within_strategy_rank"] == 1
        ),
        next(iter(evidence.candidates), ""),
    )
    alternative_id = next(
        (
            candidate_id
            for candidate_id, row in evidence.candidates.items()
            if row["strategy"] != selected.strategy_label and row["within_strategy_rank"] == 1
        ),
        "",
    )
    evidence_ids = tuple(item for item in (selected_id, alternative_id, "M1", "M2") if item)
    alternative_text = (
        "The next-ranked strategy is available as an alternative, but its tradeoffs should be reviewed before choosing it."
        if alternative_id
        else "No separate strategy produced an alternative candidate with enough information to compare."
    )
    enough_history = candidate.train_trades >= 35
    positive_newer = candidate.excess_return_percent > 0
    positive_latest = latest_excess > 0
    stable_inputs = candidate.nearby_stability != "Isolated result"
    current_match = dependency.get("current_match", "Not enough evidence")

    if (
        positive_newer
        and positive_latest
        and enough_history
        and stable_inputs
        and current_match in {"Strong match", "Direction matches"}
    ):
        return (
            "USE FOR PAPER TEST",
            selected_id,
            alternative_id,
            "This candidate is suitable for a limited paper test because its historical advantage continued into newer prices, nearby inputs remained competitive, and the latest price behavior was compatible with the conditions in which it previously worked.",
            "The result did not depend only on the oldest prices or one exact input combination.",
            "Historical evidence can still fail in live paper trading.",
            alternative_text,
            "Load the saved inputs, review the current BUY requirements, and collect paper results before considering live use.",
            evidence_ids,
        )
    if candidate.excess_return_percent < 0 and latest_excess < 0 and interval_result.durability_excess_return_percent < 0:
        return (
            "SKIP",
            selected_id,
            alternative_id,
            "The highest-ranked historical candidate should be skipped because it failed to maintain an advantage over buying and holding in both recent prices and the complete-history comparison.",
            "There is no clear evidence that this candidate improved on simply holding the ticker.",
            "Changing one exact input combination would risk chasing noise.",
            alternative_text,
            "Skip this result and review another strategy or ticker.",
            evidence_ids,
        )
    return (
        "WATCH FOR A BETTER SETUP",
        selected_id,
        alternative_id,
        "The search found a potentially useful candidate, but the historical evidence is not consistent enough to treat it as ready for a paper test.",
        "Some price sections or nearby settings support the candidate, so it may still deserve continued observation.",
        "The main concern is recent consistency, input stability, sample size, or the latest price behavior.",
        alternative_text,
        "Keep this as a research candidate and wait for stronger evidence or a better current setup.",
        evidence_ids,
    )


def _fallback_recommendation(
    evidence: StrategySearchEvidence,
    result: MultiStrategySearchResult,
    *,
    error: str = "",
) -> StrategyRecommendation:
    (
        action,
        candidate_id,
        alternative_candidate_id,
        assessment,
        why_it_may_work,
        concern,
        alternative_view,
        next_action,
        evidence_ids,
    ) = _deterministic_action(result, evidence)
    candidate = evidence.candidates.get(candidate_id, {})
    return StrategyRecommendation(
        created_at=datetime.now(PACIFIC_TIME).isoformat(),
        ticker=evidence.ticker,
        strategy=str(candidate.get("strategy", evidence.strategy)),
        interval=str(candidate.get("interval", evidence.interval)),
        action=action,
        candidate_id=candidate_id,
        alternative_candidate_id=alternative_candidate_id,
        assessment=assessment,
        why_it_may_work=why_it_may_work,
        primary_concern=concern,
        alternative_view=alternative_view,
        next_action=next_action,
        evidence_ids=evidence_ids,
        evidence_hash=evidence.evidence_hash,
        provider="deterministic",
        model="built-in",
        used_fallback=bool(error),
        error=error,
    )


def _validated_agent_payload(
    payload: dict[str, Any],
    evidence: StrategySearchEvidence,
    maximum_action: str,
    *,
    stage: str = "The agent",
) -> dict[str, Any]:
    action = str(payload.get("action", "SKIP")).upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError("The agent returned an unsupported action.")
    if ACTION_LEVEL[action] > ACTION_LEVEL[maximum_action]:
        action = maximum_action
    candidate_id = str(payload.get("recommended_candidate_id", "")).strip()
    alternative_candidate_id = str(payload.get("alternative_candidate_id", "")).strip()
    if candidate_id not in evidence.candidates:
        raise ValueError("The agent selected a candidate that was not supplied.")
    if alternative_candidate_id and alternative_candidate_id not in evidence.candidates:
        raise ValueError("The agent selected an alternative that was not supplied.")
    if alternative_candidate_id == candidate_id:
        alternative_candidate_id = ""
    evidence_ids = tuple(dict.fromkeys(str(item) for item in payload.get("evidence_ids", ())))[:10]
    if not evidence_ids or any(item not in evidence.facts for item in evidence_ids):
        raise ValueError("The agent cited evidence that was not supplied.")
    if candidate_id not in evidence_ids:
        raise ValueError("The agent did not cite the evidence for its selected candidate.")
    values = {
        "action": action,
        "candidate_id": candidate_id,
        "alternative_candidate_id": alternative_candidate_id,
        "assessment": str(payload.get("assessment", "")).strip()[:1400],
        "why_it_may_work": str(payload.get("why_it_may_work", "")).strip()[:900],
        "primary_concern": str(payload.get("primary_concern", "")).strip()[:500],
        "alternative_view": str(payload.get("alternative_view", "")).strip()[:700],
        "next_action": str(payload.get("next_action", "")).strip()[:500],
        "evidence_ids": evidence_ids,
    }
    if not all(
        values[key]
        for key in ("assessment", "why_it_may_work", "primary_concern", "alternative_view", "next_action")
    ):
        raise ValueError("The agent omitted a required explanation.")
    narrative = " ".join(
        str(values[key]).lower()
        for key in ("assessment", "why_it_may_work", "primary_concern", "alternative_view", "next_action")
    )
    cited_text = " ".join(evidence.facts[item] for item in evidence_ids)
    supplied_numbers = {_normalized_number(token) for token in _number_tokens(cited_text)}
    unsupported_numbers = {
        token
        for token in _number_tokens(narrative)
        if _normalized_number(token) not in supplied_numbers
    }
    if unsupported_numbers:
        unsupported_text = ", ".join(sorted(unsupported_numbers))
        raise ValueError(
            f"{stage} used number(s) that were not present in its cited evidence: {unsupported_text}."
        )
    forbidden = (
        "buy now",
        "sell now",
        "send the order",
        "submit the order",
        "enable automation",
        "authorize the order",
        "place the order",
    )
    if any(phrase in narrative for phrase in forbidden):
        raise ValueError("The agent attempted to authorize an execution action.")
    return values


def _number_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", str(value)))


def _normalized_number(value: str) -> str:
    number = str(value).strip().replace("%", "").replace("+", "")
    try:
        return f"{float(number):.12g}"
    except ValueError:
        return number


def _validated_review_payload(
    payload: dict[str, Any],
    evidence: StrategySearchEvidence,
) -> dict[str, Any]:
    evidence_ids = tuple(dict.fromkeys(str(item) for item in payload.get("evidence_ids", ())))[:5]
    if not evidence_ids or any(item not in evidence.facts for item in evidence_ids):
        raise ValueError("The Skeptical Reviewer cited evidence that was not supplied.")
    accepted = payload.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("The Skeptical Reviewer did not return a true/false decision.")
    review = {
        "accepted": accepted,
        "primary_objection": str(payload.get("primary_objection", "")).strip()[:500],
        "evidence_ids": evidence_ids,
        "required_change": str(payload.get("required_change", "")).strip()[:500],
    }
    if not review["primary_objection"]:
        raise ValueError("The Skeptical Reviewer omitted its main objection or conclusion.")
    if not review["accepted"] and not review["required_change"]:
        raise ValueError("The Skeptical Reviewer rejected the draft without explaining the required change.")

    narrative = f"{review['primary_objection']} {review['required_change']}".lower()
    cited_text = " ".join(evidence.facts[item] for item in evidence_ids)
    supplied_numbers = {_normalized_number(token) for token in _number_tokens(cited_text)}
    unsupported_numbers = {
        token
        for token in _number_tokens(narrative)
        if _normalized_number(token) not in supplied_numbers
    }
    if unsupported_numbers:
        unsupported_text = ", ".join(sorted(unsupported_numbers))
        raise ValueError(
            "The Skeptical Reviewer used number(s) that were not present in its cited evidence: "
            f"{unsupported_text}."
        )
    return review


def _evidence_prompt(evidence: StrategySearchEvidence) -> str:
    # Facts are the single canonical numeric source. The full internal candidate
    # catalog contains higher-precision and diagnostic values that are not all
    # included in the trader-facing evidence sentences.
    candidate_index = {
        candidate_id: {
            "strategy": candidate.get("strategy"),
            "interval": candidate.get("interval"),
            "settings": candidate.get("settings_text"),
        }
        for candidate_id, candidate in evidence.candidates.items()
    }
    return json.dumps({
        "ticker": evidence.ticker,
        "candidate_index": candidate_index,
        "evidence": evidence.facts,
        "rules": [
            "Cite only evidence IDs that appear above.",
            "Do not invent or recalculate numbers.",
            "Compare the supplied candidates fairly. Their order does not prove that one is reliable.",
            "Select the most defensible supplied candidate even when the action is SKIP; use the assessment to explain that none is convincing.",
            "Use an empty alternative_candidate_id only when no meaningful alternative exists.",
            "You may conclude that no candidate is convincing or that two candidates are too close to distinguish.",
            "Explain the result in complete sentences using clear, simple language.",
            "Avoid unexplained terms such as walk-forward, regime dependency, and plateau stability.",
            "You may repeat a useful number only when it appears in evidence you cite. Do not invent or recalculate values.",
            "This is research only and cannot authorize an order.",
        ],
    }, indent=2, default=str)


def run_strategy_recommendation_loop(
    ticker: str,
    result: MultiStrategySearchResult,
    config: LLMResearchConfig | None = None,
    client: RecommendationLLM | None = None,
) -> RecommendationLoopRun:
    evidence = build_strategy_search_evidence(ticker, result)
    fallback = _fallback_recommendation(evidence, result)
    selected = config or LLMResearchConfig.from_env()
    if selected.provider == "deterministic":
        return RecommendationLoopRun(fallback, evidence.facts, evidence.candidates)
    if selected.provider not in {"ollama", "gemini"}:
        fallback = _fallback_recommendation(
            evidence,
            result,
            error=f"Unsupported research provider: {selected.provider}",
        )
        return RecommendationLoopRun(fallback, evidence.facts, evidence.candidates)

    draft_raw: dict[str, Any] = {}
    review_raw: dict[str, Any] = {}
    editor_raw: dict[str, Any] = {}
    try:
        provider = client
        if provider is None:
            provider = OllamaRecommendationLLM(selected) if selected.provider == "ollama" else GeminiRecommendationLLM(selected)
        facts = _evidence_prompt(evidence)
        draft_raw = provider.generate_json(
            "Act as the Strategy Analyst. Compare every supplied candidate, identify the most credible input region, "
            "and identify the best alternative. Do not assume the first candidate is best. Explain whether the result "
            "looks repeatable across price sections and nearby settings, or whether it may be an anomaly. Use clear, "
            "complete sentences.\n" + facts,
            RECOMMENDATION_SCHEMA,
        )
        draft = _validated_agent_payload(
            draft_raw,
            evidence,
            "USE FOR PAPER TEST",
            stage="The Strategy Analyst",
        )
        review_raw = provider.generate_json(
            "Act as the Skeptical Reviewer. Review the full candidate set, not just the analyst's selection. Challenge "
            "overfitting, weak samples, dependence on one type of price movement, one exceptional trade, unstable nearby "
            "settings, and trading friction. Also challenge the analyst if a different candidate is more defensible. "
            "Return accepted=true only when no material concern was missed.\n"
            + facts + "\nANALYST DRAFT:\n" + json.dumps(draft, default=str),
            REVIEW_SCHEMA,
        )
        review = _validated_review_payload(review_raw, evidence)
        editor_maximum_action = draft["action"]
        if not review["accepted"]:
            lower_level = max(0, ACTION_LEVEL[editor_maximum_action] - 1)
            editor_maximum_action = next(
                action for action, level in ACTION_LEVEL.items() if level == lower_level
            )
        editor_raw = provider.generate_json(
            "Act as the Decision Editor. Produce the final assessment after considering the analyst draft, skeptical "
            "review, and complete candidate evidence. Write an intelligible, well-reasoned explanation in clear, simple "
            "language and complete sentences. Focus on the recommendation, why it may work, the main concern, the best "
            "alternative, the price behavior in which it historically worked or struggled, and the practical next step. "
            "Focused does not mean abbreviated: include enough explanation that a trader can understand why the conclusion "
            "was reached. Do not use unexplained industry jargon.\n"
            + facts + "\nANALYST DRAFT:\n" + json.dumps(draft, default=str)
            + "\nREVIEW:\n" + json.dumps(review, default=str),
            RECOMMENDATION_SCHEMA,
        )
        editor = _validated_agent_payload(
            editor_raw,
            evidence,
            editor_maximum_action,
            stage="The Decision Editor",
        )
        selected_candidate = evidence.candidates[editor["candidate_id"]]
        recommendation = StrategyRecommendation(
            created_at=datetime.now(PACIFIC_TIME).isoformat(),
            ticker=evidence.ticker,
            strategy=str(selected_candidate["strategy"]),
            interval=str(selected_candidate["interval"]),
            action=editor["action"],
            candidate_id=editor["candidate_id"],
            alternative_candidate_id=editor["alternative_candidate_id"],
            assessment=editor["assessment"],
            why_it_may_work=editor["why_it_may_work"],
            primary_concern=editor["primary_concern"],
            alternative_view=editor["alternative_view"],
            next_action=editor["next_action"],
            evidence_ids=editor["evidence_ids"],
            evidence_hash=evidence.evidence_hash,
            provider=selected.provider,
            model=selected.model,
        )
        return RecommendationLoopRun(
            recommendation,
            evidence.facts,
            evidence.candidates,
            draft,
            review,
            editor,
        )
    except Exception as exc:
        fallback = _fallback_recommendation(evidence, result, error=str(exc))
        return RecommendationLoopRun(
            fallback,
            evidence.facts,
            evidence.candidates,
            draft_raw,
            review_raw,
            editor_raw,
        )


class RecommendationLoopStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def append(self, run: RecommendationLoopRun) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(run), sort_keys=True, default=str) + "\n")

    def append_safely(self, run: RecommendationLoopRun) -> str:
        try:
            self.append(run)
            return ""
        except OSError as exc:
            return str(exc)

    def read_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        normalized_limit = max(0, int(limit))
        if normalized_limit == 0 or not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
        return records[-normalized_limit:]


def recommendation_summary_records(run: RecommendationLoopRun) -> list[dict[str, str]]:
    result = run.recommendation
    selected = run.candidates.get(result.candidate_id, {})
    alternative = run.candidates.get(result.alternative_candidate_id, {})
    return [
        {"Item": "Recommendation", "Value": result.action, "Plain English": result.assessment},
        {
            "Item": "Best candidate",
            "Value": f"{result.strategy}, {result.interval}",
            "Plain English": (
                f"Inputs to paper test: {selected.get('settings_text')}."
                if selected else "The selected candidate details are available in Full Records."
            ),
        },
        {"Item": "Why it may work", "Value": result.why_it_may_work, "Plain English": "The strongest support for this conclusion."},
        {"Item": "Main concern", "Value": result.primary_concern, "Plain English": "The strongest reason to remain cautious."},
        {
            "Item": "Best alternative",
            "Value": (
                f"{alternative.get('strategy')}, {alternative.get('interval')}"
                if alternative else "None selected"
            ),
            "Plain English": (
                f"{result.alternative_view} Inputs to compare: {alternative.get('settings_text')}."
                if alternative else result.alternative_view
            ),
        },
        {
            "Item": "Next step",
            "Value": result.next_action,
            "Plain English": "Research guidance only. Deterministic order rules still control every order.",
        },
    ]


def recommendation_audit_records(run: RecommendationLoopRun) -> list[dict[str, str]]:
    result = run.recommendation
    return [
        {"Item": "Created", "Value": result.created_at},
        {"Item": "Provider", "Value": f"{result.provider} / {result.model}"},
        {"Item": "Evidence ID", "Value": result.evidence_hash},
        {"Item": "Selected candidate", "Value": result.candidate_id},
        {"Item": "Alternative candidate", "Value": result.alternative_candidate_id or "None"},
        {"Item": "Evidence used", "Value": ", ".join(result.evidence_ids)},
        {"Item": "Fallback used", "Value": "Yes" if result.used_fallback else "No"},
        {"Item": "Error", "Value": result.error or "None"},
    ]


def recommendation_run_matches(value: Any, evidence_hash: str) -> bool:
    """Reject stale dictionaries and pre-reload class instances without raising in Streamlit."""
    return bool(
        isinstance(value, RecommendationLoopRun)
        and value.recommendation.evidence_hash == evidence_hash
    )
