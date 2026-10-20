"""Stable universe identities shared by scanner storage and presentation."""

UNIVERSE_LABELS = {
    "nasdaq": "Nasdaq common stocks (Composite-style)",
    "sp500": "S&P 500",
}
UNIVERSE_NOTES = {
    "nasdaq": "Public Nasdaq listing directory, filtered for common-stock types. "
              "This is a Composite-style universe, not a licensed, verified Composite constituent list.",
    "sp500": "S&P 500 universe uses published equity holdings of iShares Core S&P 500 ETF (IVV). "
             "Cash, funds and derivatives are excluded. This is a holdings-based proxy, not a licensed "
             "index constituent feed; membership can differ temporarily around index changes.",
}


def validate_universe_id(universe_id: str) -> str:
    if universe_id not in UNIVERSE_LABELS:
        raise ValueError("Rectangle universe must be Nasdaq (nasdaq) or S&P 500 (sp500).")
    return universe_id


def universe_cache_name(universe_id="nasdaq") -> str:
    validate_universe_id(universe_id)
    return "universe" if universe_id == "nasdaq" else "universe_sp500"
