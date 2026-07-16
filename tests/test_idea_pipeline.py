from agentloop_trader.idea_pipeline import (
    TickerResearchIdea,
    TickerResearchQueueStore,
    collect_research_ideas,
)


class Source:
    def propose(self, limit):
        return [
            TickerResearchIdea(" aapl ", "Liquid trend candidate", "test", "2026-07-15T10:00:00-07:00"),
            TickerResearchIdea("AAPL", "duplicate", "test", "2026-07-15T10:00:00-07:00"),
            TickerResearchIdea("btc-usd", "Crypto candidate", "test", "2026-07-15T10:00:00-07:00"),
            TickerResearchIdea("BAD SYMBOL!", "invalid", "test", "2026-07-15T10:00:00-07:00"),
        ]


def test_future_idea_source_only_produces_a_validated_research_queue(tmp_path):
    ideas = collect_research_ideas(Source(), limit=10)
    store = TickerResearchQueueStore(tmp_path / "ideas.json")
    store.replace(ideas)

    assert [idea.ticker for idea in store.read()] == ["AAPL", "BTC/USD"]
    assert not hasattr(store, "submit_order")


def test_zero_idea_limit_returns_an_empty_queue(tmp_path):
    ideas = collect_research_ideas(Source(), limit=0)
    store = TickerResearchQueueStore(tmp_path / "ideas.json")
    store.replace(ideas)

    assert ideas == []
    assert store.read() == []
