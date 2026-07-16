from __future__ import annotations

import json
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
        "thesis": {"type": "string"},
        "primary_concern": {"type": "string"},
        "next_action": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": ["action", "thesis", "primary_concern", "next_action", "evidence_ids"],
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
    facts: dict[str, str]
    evidence_hash: str


@dataclass(frozen=True)
class StrategyRecommendation:
    created_at: str
    ticker: str
    strategy: str
    interval: str
    action: str
    thesis: str
    primary_concern: str
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


def build_strategy_search_evidence(ticker: str, result: MultiStrategySearchResult) -> StrategySearchEvidence:
    selected, interval_result, candidate = _best_strategy_result(result)
    robustness = selected.best_result.robustness
    locked = robustness.locked_test if robustness else None
    diagnostics = robustness.diagnostics if robustness else None
    dependency = robustness.regime_dependency if robustness else {}
    latest_excess = locked.excess_return_percent if locked else 0.0
    latest_trades = locked.trades if locked else 0
    annualized_excess = interval_result.durability_annualized_excess_percent
    older_percent = int(round(selected.best_result.train_fraction * 100))
    latest_percent = int(round(selected.best_result.locked_fraction * 100))
    newer_percent = max(0, 100 - older_percent - latest_percent)
    facts = {
        "E1": (
            f"The older {older_percent}% produced {candidate.train_trades} completed trades and "
            f"{candidate.train_excess_return_percent:+.2f}% versus buy and hold."
        ),
        "E2": (
            f"The newer {newer_percent}% produced {candidate.test_trades} completed trades and "
            f"{candidate.excess_return_percent:+.2f}% versus buy and hold."
        ),
        "E3": (
            f"The latest {latest_percent}% produced {latest_trades} completed trades and "
            f"{latest_excess:+.2f}% versus buy and hold."
        ),
        "E4": (
            "Complete-history annualized excess return was not available."
            if annualized_excess is None
            else f"Complete-history annualized return was {annualized_excess:+.2f}% versus buy and hold."
        ),
        "E5": (
            f"Input stability was {candidate.nearby_stability}: {candidate.nearby_top_count} of "
            f"{candidate.plateau_neighbors} nearby settings ranked in the top third."
        ),
        "E6": (
            f"Latest price-section behavior was {dependency.get('current_regime', 'not enough evidence')}; "
            f"the match to the strongest historical behavior was "
            f"{dependency.get('current_match', 'not enough evidence')}."
        ),
        "E7": dependency.get(
            "summary",
            "There was not enough evidence to judge dependence on one type of price behavior.",
        ),
        "E8": (
            "The best-trade removal test was not available."
            if diagnostics is None
            else (
                f"After removing the single best trade, allocated return was "
                f"{diagnostics.best_trade_removed_return_percent:+.2f}%."
            )
        ),
        "E9": (
            "Trading-cost stress evidence was not available."
            if not robustness or not robustness.stress_rows
            else "The detailed result includes Alpaca fees and separate price-slippage stress tests."
        ),
    }
    identity = {
        "ticker": ticker.strip().upper(),
        "strategy": selected.strategy_label,
        "interval": selected.best_interval,
        "settings": candidate.settings,
        "facts": facts,
    }
    evidence_hash = sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    return StrategySearchEvidence(
        ticker=ticker.strip().upper(),
        strategy=selected.strategy_label,
        interval=selected.best_interval,
        settings=dict(candidate.settings),
        facts=facts,
        evidence_hash=evidence_hash,
    )


def _deterministic_action(result: MultiStrategySearchResult) -> tuple[str, str, str, str, tuple[str, ...]]:
    selected, interval_result, candidate = _best_strategy_result(result)
    robustness = selected.best_result.robustness
    locked = robustness.locked_test if robustness else None
    dependency = robustness.regime_dependency if robustness else {}
    latest_excess = locked.excess_return_percent if locked else 0.0
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
            "The historical result repeated across newer prices, nearby inputs, and a compatible latest price section.",
            "Historical evidence can still fail in live paper trading.",
            "Load the saved inputs, review the current BUY requirements, and collect paper results before considering live use.",
            ("E1", "E2", "E3", "E5", "E6"),
        )
    if candidate.excess_return_percent < 0 and latest_excess < 0 and interval_result.durability_excess_return_percent < 0:
        return (
            "SKIP",
            "The strategy did not keep an advantage over buying and holding in the important recent and complete-history comparisons.",
            "Changing one exact input combination would risk chasing noise.",
            "Skip this result and review another strategy or ticker.",
            ("E2", "E3", "E4", "E5"),
        )
    return (
        "WATCH FOR A BETTER SETUP",
        "The result has useful historical evidence, but at least one important check is weak or incomplete.",
        "The main concern is recent consistency, input stability, sample size, or the latest price-section fit.",
        "Keep this as a research candidate and wait for stronger evidence or a better current setup.",
        ("E1", "E2", "E3", "E5", "E6", "E7"),
    )


def _fallback_recommendation(
    evidence: StrategySearchEvidence,
    result: MultiStrategySearchResult,
    *,
    error: str = "",
) -> StrategyRecommendation:
    action, thesis, concern, next_action, evidence_ids = _deterministic_action(result)
    return StrategyRecommendation(
        created_at=datetime.now(PACIFIC_TIME).isoformat(),
        ticker=evidence.ticker,
        strategy=evidence.strategy,
        interval=evidence.interval,
        action=action,
        thesis=thesis,
        primary_concern=concern,
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
) -> dict[str, Any]:
    action = str(payload.get("action", "SKIP")).upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError("The agent returned an unsupported action.")
    if ACTION_LEVEL[action] > ACTION_LEVEL[maximum_action]:
        action = maximum_action
    evidence_ids = tuple(dict.fromkeys(str(item) for item in payload.get("evidence_ids", ())))[:6]
    if not evidence_ids or any(item not in evidence.facts for item in evidence_ids):
        raise ValueError("The agent cited evidence that was not supplied.")
    values = {
        "action": action,
        "thesis": str(payload.get("thesis", "")).strip()[:600],
        "primary_concern": str(payload.get("primary_concern", "")).strip()[:500],
        "next_action": str(payload.get("next_action", "")).strip()[:500],
        "evidence_ids": evidence_ids,
    }
    if not all(values[key] for key in ("thesis", "primary_concern", "next_action")):
        raise ValueError("The agent omitted a required explanation.")
    narrative = " ".join(str(values[key]).lower() for key in ("thesis", "primary_concern", "next_action"))
    if any(character.isdigit() for character in narrative):
        raise ValueError("The agent restated a number instead of citing deterministic evidence.")
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


def _evidence_prompt(evidence: StrategySearchEvidence) -> str:
    return json.dumps({
        "ticker": evidence.ticker,
        "strategy": evidence.strategy,
        "interval": evidence.interval,
        "settings": evidence.settings,
        "evidence": evidence.facts,
        "rules": [
            "Cite only evidence IDs that appear above.",
            "Do not invent or recalculate numbers.",
            "Do not recommend a different strategy or interval.",
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
        return RecommendationLoopRun(fallback, evidence.facts)
    if selected.provider not in {"ollama", "gemini"}:
        fallback = _fallback_recommendation(
            evidence,
            result,
            error=f"Unsupported research provider: {selected.provider}",
        )
        return RecommendationLoopRun(fallback, evidence.facts)

    try:
        provider = client
        if provider is None:
            provider = OllamaRecommendationLLM(selected) if selected.provider == "ollama" else GeminiRecommendationLLM(selected)
        facts = _evidence_prompt(evidence)
        draft_raw = provider.generate_json(
            "Act as the Strategy Analyst. Draft a concise recommendation from this evidence.\n" + facts,
            RECOMMENDATION_SCHEMA,
        )
        draft = _validated_agent_payload(draft_raw, evidence, fallback.action)
        review_raw = provider.generate_json(
            "Act as the Skeptical Reviewer. Challenge overfitting, weak samples, dependence on one price environment, "
            "one exceptional trade, and trading friction. Return accepted=true only when no material concern was missed.\n"
            + facts + "\nANALYST DRAFT:\n" + json.dumps(draft, default=str),
            REVIEW_SCHEMA,
        )
        review_ids = tuple(dict.fromkeys(str(item) for item in review_raw.get("evidence_ids", ())))[:5]
        if not review_ids or any(item not in evidence.facts for item in review_ids):
            raise ValueError("The reviewer cited evidence that was not supplied.")
        accepted = review_raw.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("The reviewer did not return a true/false decision.")
        review = {
            "accepted": accepted,
            "primary_objection": str(review_raw.get("primary_objection", "")).strip()[:500],
            "evidence_ids": review_ids,
            "required_change": str(review_raw.get("required_change", "")).strip()[:500],
        }
        if not review["accepted"] and (not review["primary_objection"] or not review["required_change"]):
            raise ValueError("The reviewer rejected the draft without explaining the required change.")
        editor_maximum_action = fallback.action
        if not review["accepted"]:
            lower_level = max(0, ACTION_LEVEL[editor_maximum_action] - 1)
            editor_maximum_action = next(
                action for action, level in ACTION_LEVEL.items() if level == lower_level
            )
        editor_raw = provider.generate_json(
            "Act as the Decision Editor. Revise the draft after the skeptical review. Be concise. You may make the action "
            "more cautious but never more aggressive than the built-in action.\n"
            + facts + "\nANALYST DRAFT:\n" + json.dumps(draft, default=str)
            + "\nREVIEW:\n" + json.dumps(review, default=str),
            RECOMMENDATION_SCHEMA,
        )
        editor = _validated_agent_payload(editor_raw, evidence, editor_maximum_action)
        recommendation = StrategyRecommendation(
            created_at=datetime.now(PACIFIC_TIME).isoformat(),
            ticker=evidence.ticker,
            strategy=evidence.strategy,
            interval=evidence.interval,
            action=editor["action"],
            thesis=editor["thesis"],
            primary_concern=editor["primary_concern"],
            next_action=editor["next_action"],
            evidence_ids=editor["evidence_ids"],
            evidence_hash=evidence.evidence_hash,
            provider=selected.provider,
            model=selected.model,
        )
        return RecommendationLoopRun(recommendation, evidence.facts, draft, review, editor)
    except Exception as exc:
        fallback = _fallback_recommendation(evidence, result, error=str(exc))
        return RecommendationLoopRun(fallback, evidence.facts)


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
    cited = [run.facts[item] for item in result.evidence_ids if item in run.facts]
    return [
        {"Item": "Decision", "Value": result.action, "Plain English": result.thesis},
        {"Item": "Strategy and interval", "Value": f"{result.strategy}, {result.interval}", "Plain English": "This review does not change the optimizer's selected strategy or interval."},
        {"Item": "Main concern", "Value": result.primary_concern, "Plain English": "The strongest reason to remain cautious."},
        {"Item": "Next step", "Value": result.next_action, "Plain English": "This is research guidance only; deterministic order rules still control every order."},
        {"Item": "Evidence used", "Value": ", ".join(result.evidence_ids), "Plain English": " ".join(cited)},
    ]


def recommendation_audit_records(run: RecommendationLoopRun) -> list[dict[str, str]]:
    result = run.recommendation
    return [
        {"Item": "Created", "Value": result.created_at},
        {"Item": "Provider", "Value": f"{result.provider} / {result.model}"},
        {"Item": "Evidence ID", "Value": result.evidence_hash},
        {"Item": "Fallback used", "Value": "Yes" if result.used_fallback else "No"},
        {"Item": "Error", "Value": result.error or "None"},
    ]


def recommendation_run_matches(value: Any, evidence_hash: str) -> bool:
    """Reject stale dictionaries and pre-reload class instances without raising in Streamlit."""
    return bool(
        isinstance(value, RecommendationLoopRun)
        and value.recommendation.evidence_hash == evidence_hash
    )
