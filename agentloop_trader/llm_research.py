from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

from agentloop_trader.market_data import CompanyResearchContext
from agentloop_trader.scanner import ScanCandidate


RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "recommendation": {"type": "string", "enum": ["TRADE", "WATCH", "WAIT"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "best_strategy": {"type": "string"},
        "thesis": {"type": "string"},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "opposing_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "event_risk": {"type": "string"},
        "invalidation": {"type": "string"},
        "next_action": {"type": "string"},
    },
    "required": ["ticker", "recommendation", "confidence", "best_strategy", "thesis", "supporting_evidence", "opposing_evidence", "event_risk", "invalidation", "next_action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMResearchConfig:
    provider: str = "deterministic"
    model: str = ""
    ollama_url: str = "http://localhost:11434"
    gemini_api_key: str = ""
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMResearchConfig":
        selected = str(provider or os.getenv("RESEARCH_LLM_PROVIDER", "deterministic")).strip().lower()
        default_model = os.getenv("OLLAMA_MODEL", "qwen3:8b") if selected == "ollama" else os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        return cls(
            provider=selected,
            model=default_model,
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            timeout_seconds=int(os.getenv("RESEARCH_LLM_TIMEOUT_SECONDS", "60")),
        )


@dataclass(frozen=True)
class LLMResearchResult:
    ticker: str
    recommendation: str
    confidence: float
    best_strategy: str
    thesis: str
    supporting_evidence: list[str]
    opposing_evidence: list[str]
    event_risk: str
    invalidation: str
    next_action: str
    provider: str
    model: str
    used_fallback: bool = False
    error: str = ""


class ResearchLLM(Protocol):
    def analyze(self, prompt: str) -> dict[str, Any]: ...


class OllamaResearchLLM:
    def __init__(self, config: LLMResearchConfig):
        self.config = config

    def analyze(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Analyze only the supplied trading facts. Do not invent prices, news, fundamentals, or order authority. Return the requested JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": RESEARCH_SCHEMA,
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


class GeminiResearchLLM:
    def __init__(self, config: LLMResearchConfig):
        self.config = config

    def analyze(self, prompt: str) -> dict[str, Any]:
        if not self.config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": "Analyze only supplied facts. Never invent market data or authorize an order."}]},
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": RESEARCH_SCHEMA,
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
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)


def _deterministic_result(candidate: ScanCandidate, context: CompanyResearchContext, error: str = "") -> LLMResearchResult:
    recommendation = candidate.decision if candidate.decision in {"TRADE", "WATCH", "WAIT"} else "WAIT"
    supporting = [
        f"Best deterministic fit: {candidate.best_strategy} ({candidate.fit_score:.2f}).",
        f"Backtest return {candidate.backtest_return_percent:.2f}% and profit factor {candidate.profit_factor:.2f}.",
    ]
    opposing = []
    if candidate.max_drawdown_percent:
        opposing.append(f"Historical worst drop was {candidate.max_drawdown_percent:.2f}%.")
    if context.fundamentals_status == "Not connected":
        opposing.append("Company fundamentals are not connected yet.")
    return LLMResearchResult(
        ticker=candidate.symbol,
        recommendation=recommendation,
        confidence=min(95.0, max(10.0, candidate.fit_score * 10)),
        best_strategy=candidate.best_strategy,
        thesis=candidate.reason,
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        event_risk=context.event_risk,
        invalidation="The setup is invalid if its deterministic strategy or risk rules stop passing.",
        next_action="Open this ticker in New Trade and run the full risk check." if recommendation != "WAIT" else "Keep watching for a complete setup.",
        provider="deterministic",
        model="built-in",
        used_fallback=bool(error),
        error=error,
    )


def _validate(payload: dict[str, Any], candidate: ScanCandidate, config: LLMResearchConfig) -> LLMResearchResult:
    recommendation = str(payload.get("recommendation", "WAIT")).upper()
    if recommendation not in {"TRADE", "WATCH", "WAIT"}:
        recommendation = "WAIT"
    # The LLM may be more cautious, but it may not promote a deterministic WAIT into a TRADE.
    if candidate.decision != "TRADE" and recommendation == "TRADE":
        recommendation = "WATCH" if candidate.decision == "WATCH" else "WAIT"
    return LLMResearchResult(
        ticker=candidate.symbol,
        recommendation=recommendation,
        confidence=min(100.0, max(0.0, float(payload.get("confidence", 0)))),
        best_strategy=str(payload.get("best_strategy") or candidate.best_strategy),
        thesis=str(payload.get("thesis") or candidate.reason)[:1200],
        supporting_evidence=[str(item)[:400] for item in list(payload.get("supporting_evidence") or [])[:5]],
        opposing_evidence=[str(item)[:400] for item in list(payload.get("opposing_evidence") or [])[:5]],
        event_risk=str(payload.get("event_risk") or "Unknown")[:400],
        invalidation=str(payload.get("invalidation") or "Re-run deterministic checks.")[:600],
        next_action=str(payload.get("next_action") or "Review in New Trade.")[:400],
        provider=config.provider,
        model=config.model,
    )


def build_research_prompt(candidate: ScanCandidate, context: CompanyResearchContext) -> str:
    facts = {
        "candidate": asdict(candidate),
        "company_research": {
            "event_risk": context.event_risk,
            "event_detail": context.event_detail,
            "news_status": context.news_status,
            "fundamentals_status": context.fundamentals_status,
            "headlines": [asdict(item) for item in context.headlines[:8]],
        },
        "constraints": [
            "The deterministic Current Read is authoritative for order eligibility.",
            "You may lower confidence or recommend waiting, but may not turn WAIT into TRADE.",
            "News is recent context, not proof of an upcoming earnings date.",
            "The answer is research only and cannot submit an order.",
        ],
    }
    return "Analyze this ticker candidate and return the required JSON schema.\n" + json.dumps(facts, indent=2, default=str)


def analyze_candidate(
    candidate: ScanCandidate,
    context: CompanyResearchContext,
    config: LLMResearchConfig | None = None,
    client: ResearchLLM | None = None,
) -> LLMResearchResult:
    selected = config or LLMResearchConfig.from_env()
    if selected.provider == "deterministic":
        return _deterministic_result(candidate, context)
    try:
        provider = client
        if provider is None:
            provider = OllamaResearchLLM(selected) if selected.provider == "ollama" else GeminiResearchLLM(selected)
        return _validate(provider.analyze(build_research_prompt(candidate, context)), candidate, selected)
    except Exception as exc:
        return _deterministic_result(candidate, context, error=str(exc))


def llm_research_records(result: LLMResearchResult) -> list[dict[str, str]]:
    return [
        {"Area": "Recommendation", "Read": result.recommendation, "Plain English": result.thesis},
        {"Area": "Best strategy", "Read": result.best_strategy, "Plain English": f"Source: {result.provider} / {result.model}."},
        {"Area": "Confidence", "Read": f"{result.confidence:.0f}%", "Plain English": "Research confidence only; it does not change order rules."},
        {"Area": "Event risk", "Read": result.event_risk, "Plain English": "Review recent news and confirm the earnings calendar separately."},
        {"Area": "Reasons to trade", "Read": str(len(result.supporting_evidence)), "Plain English": "; ".join(result.supporting_evidence) or "None listed."},
        {"Area": "Reasons to wait", "Read": str(len(result.opposing_evidence)), "Plain English": "; ".join(result.opposing_evidence) or "None listed."},
        {"Area": "Invalidation", "Read": "Defined", "Plain English": result.invalidation},
        {"Area": "Next action", "Read": result.next_action, "Plain English": result.error if result.used_fallback else "The LLM cannot submit orders."},
    ]
