"""Cancelable, research-only scan jobs; never executed by the broker worker."""
from __future__ import annotations

import threading
from dataclasses import field, dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from agentloop_trader.rectangle_data import (
    DailyHistoryProvider, ScanCancelled, fetch_universe,
    HISTORY_DAYS, full_history_start, history_bucket, merge_history,
)
from agentloop_trader.rectangle_scanner import DETECTOR_VERSION, daily_bars, detect_rectangles
from agentloop_trader.rectangle_store import RectangleStore, exclusive_file
from agentloop_trader.rectangle_timeframes import analysis_bars, latest_completed_date, validate_interval, TIMEFRAME_LABELS
from agentloop_trader.rectangle_universes import UNIVERSE_LABELS, universe_cache_name, validate_universe_id


@dataclass
class ScanJob:
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    message: str = "Loading stock universe..."
    progress: float = 0.0
    error: str = ""
    canceled: bool = False

    def update(self, **values):
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self):
        with self.lock:
            return {key: getattr(self, key) for key in ("done", "message", "progress", "error", "canceled")}


def run_range_scan(store: RectangleStore, provider: DailyHistoryProvider, job: ScanJob,
                   universe_loader=None, *, interval="1d", universe_id="nasdaq"):
    """Commit a completed snapshot only; cancel/fatal failure keeps previous results.

    Per-ticker failures remain explicit. New-candidate comparison retains the last
    known result for failed symbols so an outage cannot create false new arrivals.
    """
    try:
        validate_interval(interval)
        validate_universe_id(universe_id)
        with exclusive_file(store.root / "scan.lock", timeout=.1):
            provider.check_cancel()
            previous = store.read_scan(interval, universe_id)
            watches = store.read("watchlist", {})
            cache_name = universe_cache_name(universe_id)
            cached_universe = store.read(cache_name, {})
            if cached_universe and cached_universe.get("universe_id", "nasdaq") != universe_id:
                raise ValueError("Cached stock universe has a different identity; previous results were kept.")
            today = datetime.now(UTC).date().isoformat()
            if cached_universe.get("fetched_at", "")[:10] == today:
                universe = cached_universe
            else:
                universe = universe_loader() if universe_loader is not None else fetch_universe(universe_id)
                if universe.get("universe_id", universe_id) != universe_id:
                    raise ValueError("Downloaded stock universe has a different identity; previous results were kept.")
                universe = {**universe, "universe_id": universe_id}
                store.write(cache_name, universe)
            symbols = list(dict.fromkeys(universe["symbols"]))
            if not symbols:
                raise ValueError(f"{UNIVERSE_LABELS[universe_id]} returned no eligible stock symbols.")
            watched = {(w["rectangle"]["symbol"], w["source"]) for w in watches.values()}
            watch_extra = sorted((s, source) for s, source in watched if s not in symbols or source != provider.source)
            total = len(symbols) + len(watch_extra)
            finished = 0
            rectangles, errors = [], []
            observed = dict(previous.get("observed", {})) if previous.get("source") == provider.source else {}
            new_symbols, skipped = [], 0
            dates = []
            successful = 0
            bucket = history_bucket()

            def load_batch(batch, selected_provider):
                cached, result, missing = {}, {}, []
                for symbol in batch:
                    old, old_bucket = store.read_history(symbol, selected_provider.source)
                    covered = old is not None and old.attrs.get("history_days", 0) >= HISTORY_DAYS
                    if covered and old_bucket == bucket:
                        result[symbol] = old
                    else:
                        cached[symbol] = old
                        missing.append(symbol)
                if not missing:
                    return result
                # Incrementally refresh only when every requested ticker has recent history.
                oldest = min((f.index[-1] for f in cached.values() if f is not None and not f.empty), default=None)
                incremental = all(cached[s] is not None and not cached[s].empty for s in missing)
                incremental &= all(cached[s] is not None and cached[s].attrs.get("history_days", 0) >= HISTORY_DAYS for s in missing)
                incremental &= oldest is not None and (pd.Timestamp.now().normalize()-oldest).days < 25
                start = (oldest-pd.Timedelta(days=7)).strftime("%Y-%m-%d") if incremental else full_history_start()
                fresh = selected_provider.fetch(missing, start)
                full_refresh = []
                for symbol in missing:
                    selected_provider.check_cancel()
                    if symbol not in fresh:
                        continue
                    try:
                        frame = daily_bars(fresh[symbol])
                        merged = merge_history(cached[symbol], frame) if incremental else frame
                        if merged is None:
                            full_refresh.append(symbol)
                            continue
                        actual_bucket = min(bucket, merged.index[-1].strftime("%Y-%m-%d"))
                        store.write_history(symbol, selected_provider.source, merged, actual_bucket, history_days=HISTORY_DAYS)
                        result[symbol] = merged
                    except ValueError:
                        continue
                if full_refresh:
                    refreshed = selected_provider.fetch(full_refresh, full_history_start())
                    for symbol, frame in refreshed.items():
                        frame = daily_bars(frame)
                        actual_bucket = min(bucket, frame.index[-1].strftime("%Y-%m-%d"))
                        store.write_history(symbol, selected_provider.source, frame, actual_bucket, history_days=HISTORY_DAYS)
                        result[symbol] = frame
                return result

            for offset in range(0, len(symbols), 20):
                provider.check_cancel()
                batch = symbols[offset:offset+20]
                job.update(message=f"{UNIVERSE_LABELS[universe_id]} · {TIMEFRAME_LABELS[interval]} · Checking {finished:,} of {total:,} stocks · {len(set(r.symbol for r in rectangles)):,} candidates",
                           progress=finished/max(total, 1))
                try:
                    frames = load_batch(batch, provider)
                except (PermissionError, ScanCancelled):
                    raise
                except Exception as exc:
                    # Exception messages must never include configured secrets.
                    errors.extend({"Ticker": s, "Problem": f"Historical download failed ({type(exc).__name__})."} for s in batch)
                    finished += len(batch)
                    continue
                for symbol in batch:
                    provider.check_cancel()
                    try:
                        frame = frames.get(symbol)
                        if frame is None or frame.empty:
                            raise ValueError("No completed daily history returned.")
                        if frame.index[-1] < latest_completed_date("1d"):
                            raise ValueError("Daily history is stale; previous result was not treated as a fresh scan.")
                        successful += 1
                        bars = analysis_bars(frame, interval)
                        if bars.empty or bars.index[-1] < latest_completed_date(interval):
                            raise ValueError(f"Latest completed {TIMEFRAME_LABELS[interval].lower()} candle is missing expected trading sessions.")
                        dates.append(bars.index[-1].strftime("%Y-%m-%d"))
                        # Feed-specific basic activity floor; not consolidated-market volume.
                        liquid = len(bars) >= (63 if interval == "1d" else 13) and frame.Close.iloc[-1] >= 1 and "Volume" in frame
                        if liquid:
                            liquid = float((frame.Close*frame.Volume).tail(20).mean()) >= 100_000
                            liquid &= float((frame.Volume.tail(20) > 0).mean()) >= .9
                        found = detect_rectangles(symbol, frame, interval=interval) if liquid else []
                        if not liquid:
                            skipped += 1
                        if found and not observed.get(symbol, False):
                            new_symbols.append(symbol)
                        observed[symbol] = bool(found)
                        rectangles.extend(found)
                    except Exception as exc:
                        errors.append({"Ticker": symbol, "Problem": str(exc) if isinstance(exc, ValueError) else f"Range calculation failed ({type(exc).__name__})."})
                    finished += 1
                job.update(progress=finished/max(total, 1))
            for symbol, source in watch_extra:
                provider.check_cancel()
                selected = provider if source == provider.source else DailyHistoryProvider(cancel=provider.cancel)
                try:
                    if source != selected.source:
                        raise ValueError("The saved watchlist data source is unavailable.")
                    frames = load_batch([symbol], selected)
                    if symbol not in frames:
                        raise ValueError("No updated watchlist history returned.")
                except ScanCancelled:
                    raise
                except Exception:
                    errors.append({"Ticker": symbol, "Problem": "Saved watchlist history could not be refreshed; its previous data was kept."})
                finished += 1
                job.update(progress=finished/max(total, 1))
            provider.check_cancel()
            if not dates:
                raise ValueError("No usable daily history was returned. Previous scan results were kept.")
            payload = {
                "detector_version": DETECTOR_VERSION, "interval": interval, "universe_id": universe_id,
                "scanned_at": datetime.now(UTC).isoformat(),
                "source": provider.source, "universe": universe, "total": len(symbols),
                "successful": successful, "skipped": skipped, "errors": errors,
                "rectangles": [r.record() for r in rectangles], "new_symbols": new_symbols,
                "observed": observed, "data_start": min(dates), "data_end": max(dates),
                "first_scan": not bool(previous) or previous.get("source") != provider.source,
            }
            store.write_scan(interval, payload, universe_id)
            job.update(message="Scan complete", progress=1.0)
    except ScanCancelled:
        job.update(canceled=True, message="Scan canceled; previous results were kept.")
    except Exception as exc:
        safe = str(exc) if isinstance(exc, (ValueError, PermissionError, RuntimeError)) else f"Scan failed ({type(exc).__name__}); previous results were kept."
        for secret in (getattr(provider, "api_key", None), getattr(provider, "api_secret", None)):
            if secret:
                safe = safe.replace(secret, "[redacted]")
        job.update(error=safe)
    finally:
        job.update(done=True)


def start_range_scan(store, provider, *, interval="1d", universe_id="nasdaq") -> ScanJob:
    validate_interval(interval)
    validate_universe_id(universe_id)
    job = ScanJob(cancel=provider.cancel)
    threading.Thread(target=run_range_scan, args=(store, provider, job),
                     kwargs={"interval": interval, "universe_id": universe_id}, daemon=True,
                     name="rectangle-research-scan").start()
    return job
