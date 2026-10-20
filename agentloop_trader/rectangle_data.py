"""Public stock universes and bounded batched historical downloads."""
from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from agentloop_trader.rectangle_timeframes import latest_completed_date
from agentloop_trader.rectangle_universes import UNIVERSE_LABELS, UNIVERSE_NOTES, validate_universe_id


DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
SP500_HOLDINGS_URL = "https://www.ishares.com/us/products/239726/ishares-core-s-p-500-etf/latest-holdings.csv"
HISTORY_DAYS = 1100  # Three years: 24-month ranges plus ATR and pivot warmup.
UNIVERSE_LABEL = UNIVERSE_LABELS["nasdaq"]  # Backward-compatible exports.
UNIVERSE_NOTE = UNIVERSE_NOTES["nasdaq"]


class ScanCancelled(Exception):
    pass


def parse_nasdaq_directory(text: str) -> tuple[list[str], str]:
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    required = {"Symbol", "Security Name", "Test Issue", "ETF", "Financial Status"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Nasdaq returned an unrecognized listing directory.")
    symbols = set()
    excluded = re.compile(r"\b(warrants?|rights?|preferred|preference|debentures?|notes?|closed.end fund)\b", re.I)
    common = re.compile(r"\b(common (stock|shares)|ordinary shares|depositary (shares|receipts)|beneficial interest|limited partnership)\b", re.I)
    for row in reader:
        symbol, name = row.get("Symbol", "").strip(), row.get("Security Name", "")
        if row.get("ETF") != "N" or row.get("Test Issue") != "N" or row.get("Financial Status") != "N":
            continue
        if row.get("NextShares", "N") == "Y" or not re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol):
            continue
        # Ordinary SPAC units are excluded; units of beneficial interest can qualify.
        if excluded.search(name) or (re.search(r"\bunits?\b", name, re.I) and "beneficial interest" not in name.lower()):
            continue
        if common.search(name):
            symbols.add(symbol)
    if not symbols:
        raise ValueError("No eligible Nasdaq common stocks were returned.")
    match = re.search(r"File Creation Time: ([^|\r\n]+)", text)
    return sorted(symbols), match.group(1).strip() if match else "Not supplied"


def fetch_nasdaq_universe() -> dict:
    request = Request(DIRECTORY_URL, headers={"User-Agent": "AgentLoopTrader/1.0 (personal research)"})
    with urlopen(request, timeout=25) as response:
        raw = response.read().decode("utf-8-sig")
    symbols, directory_date = parse_nasdaq_directory(raw)
    return {"symbols": symbols, "directory_date": directory_date,
            "fetched_at": datetime.now(UTC).isoformat(), "source": DIRECTORY_URL, "universe_id": "nasdaq"}


def parse_sp500_holdings(text: str) -> tuple[list[str], str]:
    """Read the provider's metadata-prefixed CSV, including separate share classes.

    Never treat a partial response or malformed equity ticker as a smaller valid
    universe. Fund holdings are explicitly described as an index proxy in the UI.
    """
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    required = {"Ticker", "Name", "Asset Class", "Exchange"}
    header = next((i for i, row in enumerate(rows) if required.issubset(row)), None)
    if header is None or not rows or rows[0] != ["iShares Core S&P 500 ETF"]:
        raise ValueError("iShares returned unrecognized S&P 500 holdings.")
    date = next((r[1].strip() for r in rows[:header] if len(r) > 1 and r[0].strip() == "Fund Holdings as of"), "")
    try:
        as_of = datetime.strptime(date, "%b %d, %Y").date().isoformat()
    except ValueError:
        raise ValueError("S&P 500 holdings are missing a valid as-of date.") from None
    symbols = set()
    for fields in rows[header+1:]:
        if len(fields) < len(rows[header]):
            if any(value.strip() for value in fields):
                # Provider footer is separated from rows by a blank line.
                # An equity-looking truncated row must never be silently lost.
                asset_column = rows[header].index("Asset Class")
                if len(fields) > asset_column and fields[asset_column].strip() == "Equity":
                    raise ValueError("S&P 500 holdings contain a truncated equity row.")
            continue
        row = dict(zip(rows[header], fields))
        if row["Asset Class"].strip() != "Equity":
            continue
        if row["Exchange"].strip().upper() == "NO MARKET (E.G. UNLISTED)":
            continue  # Residual positions after delisting are not tradable stocks.
        if row["Exchange"].strip().upper() not in {"NASDAQ", "NYSE", "NEW YORK STOCK EXCHANGE INC.", "CBOE BZX"}:
            raise ValueError("S&P 500 holdings contain an unrecognized equity exchange.")
        symbol = re.sub(r"[ .-]([A-Z])$", r".\1", row["Ticker"].strip())
        if not re.fullmatch(r"[A-Z][A-Z0-9]*(?:\.[A-Z])?", symbol) or len(symbol) > 10:
            raise ValueError("S&P 500 holdings contain an unrecognized equity symbol.")
        symbols.add(symbol)
    if not symbols:
        raise ValueError("No S&P 500 equity holdings were returned.")
    return sorted(symbols), as_of


def fetch_sp500_universe() -> dict:
    request = Request(SP500_HOLDINGS_URL, headers={
        "User-Agent": "Mozilla/5.0 (compatible; AgentLoopTrader/1.0; personal research)",
    })
    with urlopen(request, timeout=25) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("S&P 500 holdings response exceeded the expected size.")
    symbols, as_of = parse_sp500_holdings(raw.decode("utf-8-sig"))
    # A top-ten-only page or truncated download cannot replace a full universe.
    if not 490 <= len(symbols) <= 520:
        raise ValueError("S&P 500 holdings count is incomplete or unexpected; previous results were kept.")
    age = (datetime.now(UTC).date()-datetime.fromisoformat(as_of).date()).days
    if not -1 <= age <= 10:
        raise ValueError("S&P 500 holdings date is stale or unexpected; previous results were kept.")
    return {"symbols": symbols, "directory_date": as_of, "universe_id": "sp500",
            "fetched_at": datetime.now(UTC).isoformat(), "source": SP500_HOLDINGS_URL}


def fetch_universe(universe_id="nasdaq") -> dict:
    validate_universe_id(universe_id)
    return fetch_nasdaq_universe() if universe_id == "nasdaq" else fetch_sp500_universe()


class DailyHistoryProvider:
    def __init__(self, api_key=None, api_secret=None, cancel: threading.Event | None = None):
        self.api_key, self.api_secret = api_key, api_secret
        self.source = "Alpaca IEX" if api_key and api_secret else "Yahoo"
        self.cancel = cancel or threading.Event()
        self.last_request = 0.0

    def check_cancel(self):
        if self.cancel.is_set():
            raise ScanCancelled("Scan canceled; previous results were kept.")

    def _json(self, params: dict):
        for attempt in range(3):
            self.check_cancel()
            if self.cancel.wait(max(0, .35 - (time.monotonic()-self.last_request))):
                self.check_cancel()
            request = Request("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params), headers={
                "APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret,
                "Accept": "application/json",
            })
            self.last_request = time.monotonic()
            try:
                with urlopen(request, timeout=20) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise PermissionError("Alpaca historical data access failed. Check the configured credentials and IEX access.") from None
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    # Do not persist provider URLs, headers, or credential-bearing exceptions.
                    raise RuntimeError(f"Historical download failed (HTTP {exc.code}).") from None
                delay = min(15, max(1, int(exc.headers.get("Retry-After", 2**(attempt+1)))))
                if self.cancel.wait(delay):
                    self.check_cancel()
        raise RuntimeError("Historical download failed.")

    def fetch(self, symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
        self.check_cancel()
        if self.source == "Yahoo":
            import yfinance as yf
            aliases = {symbol.replace(".", "-"): symbol for symbol in symbols}
            data = yf.download(tickers=list(aliases), start=start, interval="1d", auto_adjust=True,
                               progress=False, threads=False, prepost=False, timeout=20,
                               group_by="ticker", multi_level_index=True)
            result = {}
            for alias, symbol in aliases.items():
                if data is not None and not data.empty and alias in data.columns.get_level_values(0):
                    frame = data[alias].dropna(subset=["Open", "High", "Low", "Close"])
                    if not frame.empty:
                        result[symbol] = frame  # Session calendar excludes forming bars in the scan runner.
            return result
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start,
                  "feed": "iex", "adjustment": "all", "limit": 10000}
        bars: dict[str, list] = {s: [] for s in symbols}
        tokens = set()
        for _ in range(200):
            payload = self._json(params)
            for symbol, rows in payload.get("bars", {}).items():
                if symbol in bars:
                    bars[symbol].extend(rows)
            token = payload.get("next_page_token")
            if not token:
                break
            if token in tokens:
                raise RuntimeError("Historical pagination repeated; partial data was discarded.")
            tokens.add(token)
            params["page_token"] = token
        else:
            raise RuntimeError("Historical pagination exceeded its limit; partial data was discarded.")
        result = {}
        for symbol, rows in bars.items():
            if not rows:
                continue
            frame = pd.DataFrame(rows).rename(columns={"t": "Date", "o": "Open", "h": "High",
                                                     "l": "Low", "c": "Close", "v": "Volume"})
            frame["Date"] = pd.to_datetime(frame.Date, utc=True)
            result[symbol] = frame.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return result


def history_bucket() -> str:
    return latest_completed_date("1d").strftime("%Y-%m-%d")


def full_history_start() -> str:
    return (datetime.now(UTC) - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")


def merge_history(old: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame | None:
    """Return None when overlap changed, requiring a full corporate-action refresh."""
    if old is None or old.empty:
        return fresh
    overlap = old.index.intersection(fresh.index)
    if overlap.empty:
        return None
    for column in ("Open", "High", "Low", "Close"):
        if not np.allclose(old.loc[overlap, column], fresh.loc[overlap, column], rtol=1e-6, atol=1e-6):
            return None
    combined = pd.concat([old, fresh])
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    return combined.loc[combined.index >= pd.Timestamp(full_history_start())]
