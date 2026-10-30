"""
async_market_ingestor.py
========================
Module 2 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Responsibilities
----------------
1. Per-endpoint async token-bucket rate limiting and a retrying aiohttp client
   (exponential backoff with full jitter on 429 / 5xx / timeouts / connection errors).
2. Full pagination of the Polymarket Gamma `/events` endpoint, with robust parsing of
   double-encoded JSON fields (`outcomes`, `outcomePrices`, `clobTokenIds`).
3. Batched CLOB order book retrieval (`POST /books`, chunks of 100) with a per-token
   `GET /book` fallback, strict ladder sorting, and microstructure snapshots.
4. Exact USD-vs-shares VWAP and slippage computation for configurable clip sizes.
5. Sharp bookmaker odds from The-Odds-API plus a deterministic fuzzy matcher that pairs
   Polymarket markets with sharp events for Module 3's de-vigging engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import aiohttp

from config_and_database import (
    AppConfig,
    DatabaseManager,
    MarketRecord,
    OrderbookSnapshot,
)

logger = logging.getLogger("pm_engine.ingestor")

CLOB_BATCH_SIZE = 100
GAMMA_PAGE_SIZE = 100
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}
SHARP_BOOK_PRIORITY: Tuple[str, ...] = ("pinnacle", "betfair_ex_eu", "betfair_ex_uk", "matchbook", "smarkets")


# =============================================================================
# Rate limiting
# =============================================================================

class TokenBucket:
    """Async token bucket: `rate` tokens/second refill, burst up to `capacity`."""

    def __init__(self, rate: float, capacity: Optional[float] = None, name: str = "bucket") -> None:
        if rate <= 0:
            raise ValueError("TokenBucket rate must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate))
        self.name = name
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds bucket capacity {self.capacity}")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self.rate)

    def penalize(self, seconds: float) -> None:
        """Drain the bucket so the next acquire waits roughly `seconds` (used after a 429)."""
        self._refill()
        self._tokens = min(self._tokens, 0.0) - seconds * self.rate


class HTTPRequestError(RuntimeError):
    def __init__(self, status: Optional[int], url: str, body: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


class AsyncHTTPClient:
    """aiohttp wrapper with per-endpoint token buckets, a global concurrency cap and retries."""

    def __init__(self, cfg: AppConfig) -> None:
        rl = cfg.rate_limits
        self.cfg = cfg
        self.buckets: Dict[str, TokenBucket] = {
            "gamma": TokenBucket(rl.gamma_rps, name="gamma"),
            "clob": TokenBucket(rl.clob_rps, name="clob"),
            "odds": TokenBucket(rl.odds_api_rps, name="odds"),
        }
        self._sem = asyncio.Semaphore(rl.http_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None
        self.stats: Dict[str, int] = {"requests": 0, "retries": 0, "failures": 0}
        self.last_headers: Dict[str, Dict[str, str]] = {}

    async def __aenter__(self) -> "AsyncHTTPClient":
        await self.open()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.rate_limits.request_timeout_s)
            connector = aiohttp.TCPConnector(limit=self.cfg.rate_limits.http_concurrency, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                timeout=timeout, connector=connector,
                headers={"User-Agent": "pm-fair-value-engine/1.0", "Accept": "application/json"},
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        rl = self.cfg.rate_limits
        if retry_after is not None:
            return min(rl.backoff_cap_s, retry_after) + random.uniform(0, rl.backoff_base_s)
        ceiling = min(rl.backoff_cap_s, rl.backoff_base_s * (2 ** attempt))
        return random.uniform(0, ceiling)  # full jitter

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> Optional[float]:
        raw = headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    async def request_json(self, method: str, url: str, bucket: str, *,
                           params: Optional[Mapping[str, Any]] = None,
                           json_body: Any = None) -> Any:
        await self.open()
        assert self._session is not None
        max_retries = self.cfg.rate_limits.max_retries
        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            await self.buckets[bucket].acquire()
            retry_after: Optional[float] = None
            try:
                async with self._sem:
                    self.stats["requests"] += 1
                    async with self._session.request(method, url, params=params, json=json_body) as resp:
                        self.last_headers[bucket] = dict(resp.headers)
                        if resp.status == 200:
                            return await resp.json(content_type=None)
                        body = await resp.text()
                        if resp.status not in RETRYABLE_STATUS:
                            raise HTTPRequestError(resp.status, url, body)
                        retry_after = self._retry_after(resp.headers)
                        if resp.status == 429:
                            self.buckets[bucket].penalize(retry_after or 1.0)
                        last_exc = HTTPRequestError(resp.status, url, body)
            except HTTPRequestError as exc:
                if exc.status not in RETRYABLE_STATUS:
                    self.stats["failures"] += 1
                    raise
                last_exc = exc
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                last_exc = exc
            if attempt < max_retries:
                delay = self._backoff(attempt, retry_after)
                self.stats["retries"] += 1
                logger.debug("Retry %d/%d for %s in %.2fs (%s)", attempt + 1, max_retries, url, delay, last_exc)
                await asyncio.sleep(delay)
        self.stats["failures"] += 1
        raise HTTPRequestError(getattr(last_exc, "status", None), url, f"exhausted retries: {last_exc}")

    async def get_json(self, url: str, bucket: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return await self.request_json("GET", url, bucket, params=params)

    async def post_json(self, url: str, bucket: str, body: Any) -> Any:
        return await self.request_json("POST", url, bucket, json_body=body)


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_json_list(value: Any) -> List[Any]:
    """Parse a Gamma field that may be a native list, a JSON string, or a double-encoded JSON string."""
    current = value
    for _ in range(3):
        if current is None:
            return []
        if isinstance(current, (list, tuple)):
            return list(current)
        if isinstance(current, (bytes, bytearray)):
            current = current.decode("utf-8", errors="replace")
        if isinstance(current, str):
            text = current.strip()
            if text == "":
                return []
            try:
                current = json.loads(text)
                continue
            except json.JSONDecodeError:
                if "," in text and not text.startswith("["):
                    return [part.strip().strip('"') for part in text.split(",") if part.strip()]
                return [text]
        return [current]
    return list(current) if isinstance(current, (list, tuple)) else []


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_gamma_market(raw: Mapping[str, Any], event: Mapping[str, Any]) -> Optional[MarketRecord]:
    """Convert one Gamma market dict (nested under an event) to a MarketRecord."""
    condition_id = raw.get("conditionId") or raw.get("condition_id")
    if not condition_id:
        return None
    outcomes = [str(o) for o in parse_json_list(raw.get("outcomes"))]
    token_ids = [str(t) for t in parse_json_list(raw.get("clobTokenIds"))]
    if not token_ids or len(token_ids) != len(outcomes):
        return None
    if raw.get("enableOrderBook") is False or raw.get("closed") is True:
        return None
    tags = event.get("tags") or []
    category = raw.get("category") or event.get("category") or (
        tags[0].get("label", "") if tags and isinstance(tags[0], Mapping) else "")
    return MarketRecord(
        condition_id=str(condition_id),
        question=str(raw.get("question") or event.get("title") or ""),
        slug=str(raw.get("slug") or ""),
        event_slug=str(event.get("slug") or ""),
        category=str(category or ""),
        description=str(raw.get("description") or event.get("description") or ""),
        outcomes=outcomes,
        token_ids=token_ids,
        end_date_iso=raw.get("endDate") or event.get("endDate"),
        volume_24h=to_float(raw.get("volume24hr")),
        volume_total=to_float(raw.get("volumeNum", raw.get("volume"))),
        liquidity=to_float(raw.get("liquidityNum", raw.get("liquidity"))),
        active=bool(raw.get("active", True)),
        neg_risk=bool(raw.get("negRisk", event.get("negRisk", False))),
    )


# =============================================================================
# Order book math
# =============================================================================

Level = Tuple[float, float]  # (price, size_in_shares)


def normalize_ladder(levels: Iterable[Any], side: str) -> List[Level]:
    """Coerce raw CLOB levels to floats, drop invalid ones, and sort: asks ASC, bids DESC."""
    out: List[Level] = []
    for lvl in levels or []:
        if isinstance(lvl, Mapping):
            price, size = to_float(lvl.get("price"), -1.0), to_float(lvl.get("size"), -1.0)
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            price, size = to_float(lvl[0], -1.0), to_float(lvl[1], -1.0)
        else:
            continue
        if 0.0 < price < 1.0 and size > 0.0:
            out.append((price, size))
    if side == "asks":
        out.sort(key=lambda x: x[0])                 # lowest ask first
    elif side == "bids":
        out.sort(key=lambda x: x[0], reverse=True)   # highest bid first
    else:
        raise ValueError("side must be 'asks' or 'bids'")
    return out


def ladder_depth_usd(levels: Sequence[Level]) -> float:
    return sum(p * s for p, s in levels)


def calculate_effective_entry_price(asks_ladder: Sequence[Level],
                                    target_stake_usd: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Walk an ASCENDING asks ladder spending `target_stake_usd`.
    Each level costs price * size USD; shares bought at a level = usd_spent_at_level / price.
    Returns (vwap, slippage) where vwap = target_stake_usd / total_shares and
    slippage = vwap - best_ask. Returns (None, None) if book depth cannot fill the stake.
    """
    if target_stake_usd <= 0 or not asks_ladder:
        return None, None
    ladder = sorted(asks_ladder, key=lambda x: x[0])
    if ladder_depth_usd(ladder) + 1e-9 < target_stake_usd:
        return None, None
    best_ask = ladder[0][0]
    remaining = target_stake_usd
    shares = 0.0
    for price, size in ladder:
        level_usd = price * size
        spend = min(remaining, level_usd)
        shares += spend / price
        remaining -= spend
        if remaining <= 1e-9:
            break
    if shares <= 0:
        return None, None
    vwap = target_stake_usd / shares
    return vwap, vwap - best_ask


def order_book_imbalance(bids: Sequence[Level], asks: Sequence[Level],
                         mid: Optional[float], band: float = 0.10) -> Tuple[float, float, Optional[float]]:
    """USD depth within +/- band of mid on each side, and OBI = (bid - ask) / (bid + ask)."""
    if mid is None:
        bid_usd, ask_usd = ladder_depth_usd(bids), ladder_depth_usd(asks)
    else:
        bid_usd = sum(p * s for p, s in bids if p >= mid - band)
        ask_usd = sum(p * s for p, s in asks if p <= mid + band)
    total = bid_usd + ask_usd
    return bid_usd, ask_usd, ((bid_usd - ask_usd) / total if total > 0 else None)


def build_snapshot(token_id: str, condition_id: str, raw_book: Mapping[str, Any],
                   clip_sizes: Sequence[float] = (100.0, 500.0, 1_000.0)) -> OrderbookSnapshot:
    asks = normalize_ladder(raw_book.get("asks", []), "asks")
    bids = normalize_ladder(raw_book.get("bids", []), "bids")
    best_ask = asks[0][0] if asks else None
    best_bid = bids[0][0] if bids else None
    spread = (best_ask - best_bid) if best_ask is not None and best_bid is not None else None
    mid = (best_ask + best_bid) / 2.0 if spread is not None else None
    clips = list(clip_sizes) + [100.0, 500.0, 1_000.0]
    vw: Dict[float, Tuple[Optional[float], Optional[float]]] = {
        c: calculate_effective_entry_price(asks, c) for c in clips}
    bid_usd, ask_usd, obi = order_book_imbalance(bids, asks, mid)
    return OrderbookSnapshot(
        token_id=token_id, condition_id=condition_id,
        best_bid=best_bid, best_ask=best_ask, spread=spread, mid=mid,
        vwap_100=vw[100.0][0], vwap_500=vw[500.0][0], vwap_1000=vw[1_000.0][0],
        slippage_100=vw[100.0][1], slippage_500=vw[500.0][1], slippage_1000=vw[1_000.0][1],
        bid_depth_usd=bid_usd, ask_depth_usd=ask_usd, imbalance=obi,
    )


# =============================================================================
# Sharp odds & matching
# =============================================================================

@dataclass
class SharpEvent:
    event_id: str
    sport_key: str
    commence_time: Optional[datetime]
    home_team: str
    away_team: str
    book_odds: Dict[str, Dict[str, float]]          # bookmaker -> {outcome_name: decimal_odds}
    consensus_odds: Dict[str, float] = field(default_factory=dict)  # median across sharp books
    primary_book: Optional[str] = None


@dataclass
class MatchedPair:
    market: MarketRecord
    sharp: SharpEvent
    score: float
    outcome_map: Dict[str, str]        # polymarket outcome -> sharp outcome or "NOT:<sharp outcome>"
    token_map: Dict[str, str]          # polymarket outcome -> CLOB token id
    sharp_decimal_odds: List[float]    # full sharp book outcomes for de-vigging (consistent order)
    sharp_outcome_names: List[str]

    def to_devig_input(self) -> Dict[str, Any]:
        return {
            "condition_id": self.market.condition_id,
            "book": self.sharp.primary_book or "consensus",
            "outcome_names": self.sharp_outcome_names,
            "decimal_odds": self.sharp_decimal_odds,
            "outcome_map": self.outcome_map,
            "token_map": self.token_map,
            "match_score": self.score,
        }


_STOPWORDS = {
    "will", "the", "a", "an", "of", "in", "on", "to", "vs", "v", "versus", "win", "wins", "beat", "beats",
    "game", "match", "fc", "sc", "cf", "afc", "at", "and", "or", "be", "by", "for", "is", "who", "which",
    "team", "over", "against", "defeat", "defeats", "draw", "yes", "no", "2024", "2025", "2026", "2027",
}


def normalize_tokens(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def entity_coverage(entity: str, haystack_tokens: Sequence[str]) -> float:
    """Fraction of an entity's tokens found in the haystack (prefix match for abbreviations >= 4 chars)."""
    ent = normalize_tokens(entity)
    if not ent:
        return 0.0
    hay = set(haystack_tokens)
    hits = 0
    for tok in ent:
        if tok in hay or (len(tok) >= 4 and any(h.startswith(tok) or tok.startswith(h) for h in hay if len(h) >= 4)):
            hits += 1
    return hits / len(ent)


def team_match_score(team: str, haystack_tokens: Sequence[str]) -> float:
    """Coverage of a team name, boosted to 0.85 when its distinctive final token (nickname) is present."""
    ent = normalize_tokens(team)
    if not ent:
        return 0.0
    cov = entity_coverage(team, haystack_tokens)
    return max(cov, 0.85) if ent[-1] in set(haystack_tokens) else cov


def name_similarity(a: str, b: str) -> float:
    """Symmetric similarity: best of either name's coverage inside the other (with nickname boost)."""
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    return max(team_match_score(a, tb), team_match_score(b, ta))


def match_polymarket_to_sharp_event(markets: Sequence[MarketRecord], sharp_events: Sequence[SharpEvent],
                                    min_score: float = 0.75, max_days_apart: float = 3.0) -> List[MatchedPair]:
    """Deterministically pair Polymarket markets with sharp events; one best sharp event per market."""
    pairs: List[MatchedPair] = []
    for m in markets:
        hay = normalize_tokens(" ".join([m.question, m.event_slug.replace("-", " "), " ".join(m.outcomes)]))
        q_tokens = normalize_tokens(m.question)
        m_end = parse_iso(m.end_date_iso)
        best: Optional[Tuple[float, SharpEvent]] = None
        for ev in sharp_events:
            if not ev.consensus_odds:
                continue
            if m_end and ev.commence_time and abs((m_end - ev.commence_time).total_seconds()) > max_days_apart * 86_400:
                continue
            score = (team_match_score(ev.home_team, hay) + team_match_score(ev.away_team, hay)) / 2.0
            if score >= min_score and (best is None or score > best[0]
                                       or (score == best[0] and ev.event_id < best[1].event_id)):
                best = (score, ev)
        if best is None:
            continue
        score, ev = best
        names = sorted(ev.consensus_odds.keys())
        outcome_map: Dict[str, str] = {}
        lowered = [o.lower() for o in m.outcomes]
        if set(lowered) == {"yes", "no"}:
            # The focus team is the one named first in the question ("Will X beat Y?")
            named = [n for n in names if n != "Draw" and team_match_score(n, q_tokens) >= 0.5]
            if not named:
                continue
            def first_pos(n: str) -> int:
                toks = normalize_tokens(n)
                return min((q_tokens.index(t) for t in toks if t in q_tokens), default=10_000)
            focus = min(named, key=lambda n: (first_pos(n), n))
            if first_pos(focus) == 10_000:
                continue
            for o in m.outcomes:
                outcome_map[o] = focus if o.lower() == "yes" else f"NOT:{focus}"
        else:
            for o in m.outcomes:
                cand = max(names, key=lambda n: (name_similarity(n, o), n))
                if name_similarity(cand, o) >= 0.5 or o.lower() == cand.lower():
                    outcome_map[o] = cand
            if len(outcome_map) != len(m.outcomes) or len(set(outcome_map.values())) != len(outcome_map):
                continue
        pairs.append(MatchedPair(
            market=m, sharp=ev, score=round(score, 4), outcome_map=outcome_map,
            token_map=dict(zip(m.outcomes, m.token_ids)),
            sharp_decimal_odds=[ev.consensus_odds[n] for n in names], sharp_outcome_names=names,
        ))
    return pairs


# =============================================================================
# Ingestor
# =============================================================================

class MarketIngestor:
    def __init__(self, cfg: AppConfig, http: AsyncHTTPClient, db: Optional[DatabaseManager] = None) -> None:
        self.cfg = cfg
        self.http = http
        self.db = db

    # ------------------------------------------------------------------ Gamma
    async def fetch_active_events(self, max_pages: int = 200) -> List[Dict[str, Any]]:
        url = f"{self.cfg.endpoints.gamma_base_url}/events"
        events: List[Dict[str, Any]] = []
        for page in range(max_pages):
            params = {"active": "true", "closed": "false", "limit": GAMMA_PAGE_SIZE, "offset": page * GAMMA_PAGE_SIZE}
            batch = await self.http.get_json(url, "gamma", params=params)
            if isinstance(batch, Mapping):
                batch = batch.get("data") or batch.get("events") or []
            if not batch:
                break
            events.extend(batch)
            if len(batch) < GAMMA_PAGE_SIZE:
                break
        logger.info("Gamma: fetched %d active events", len(events))
        return events

    async def fetch_markets(self, max_pages: int = 200) -> List[MarketRecord]:
        events = await self.fetch_active_events(max_pages=max_pages)
        seen: Dict[str, MarketRecord] = {}
        for ev in events:
            for raw in ev.get("markets") or []:
                rec = parse_gamma_market(raw, ev)
                if rec and rec.active:
                    seen[rec.condition_id] = rec
        markets = list(seen.values())
        logger.info("Gamma: parsed %d tradable markets", len(markets))
        return markets

    # ------------------------------------------------------------------- CLOB
    async def fetch_book_single(self, token_id: str) -> Optional[Dict[str, Any]]:
        try:
            return await self.http.get_json(f"{self.cfg.endpoints.clob_base_url}/book", "clob",
                                            params={"token_id": token_id})
        except HTTPRequestError as exc:
            if exc.status == 404:
                return None
            logger.warning("CLOB /book failed for %s: %s", token_id, exc)
            return None

    async def fetch_books(self, token_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Batched POST /books in chunks of 100; falls back to GET /book for missing tokens."""
        url = f"{self.cfg.endpoints.clob_base_url}/books"
        unique = list(dict.fromkeys(token_ids))
        chunks = [unique[i:i + CLOB_BATCH_SIZE] for i in range(0, len(unique), CLOB_BATCH_SIZE)]

        async def run_chunk(chunk: List[str]) -> List[Dict[str, Any]]:
            try:
                res = await self.http.post_json(url, "clob", [{"token_id": t} for t in chunk])
                return res if isinstance(res, list) else []
            except HTTPRequestError as exc:
                logger.warning("CLOB /books batch failed (%d tokens): %s", len(chunk), exc)
                return []

        books: Dict[str, Dict[str, Any]] = {}
        for result in await asyncio.gather(*(run_chunk(c) for c in chunks)):
            for book in result:
                tid = str(book.get("asset_id") or book.get("token_id") or "")
                if tid:
                    books[tid] = book
        missing = [t for t in unique if t not in books]
        if missing:
            logger.info("CLOB: %d tokens missing from batch, using single /book fallback", len(missing))
            singles = await asyncio.gather(*(self.fetch_book_single(t) for t in missing))
            for tid, book in zip(missing, singles):
                if book:
                    books[tid] = book
        return books

    async def snapshot_markets(self, markets: Sequence[MarketRecord]) -> List[OrderbookSnapshot]:
        token_to_cond = {t: m.condition_id for m in markets for t in m.token_ids}
        books = await self.fetch_books(list(token_to_cond.keys()))
        clips = self.cfg.strategy.vwap_clip_sizes_usd
        return [build_snapshot(tid, token_to_cond[tid], book, clips)
                for tid, book in books.items() if tid in token_to_cond]

    # ------------------------------------------------------------- sharp odds
    async def fetch_sharp_odds(self, sport: str, regions: str = "eu,uk",
                               markets: str = "h2h") -> List[SharpEvent]:
        key = self.cfg.keys.odds_api_key
        if not key:
            logger.info("ODDS_API_KEY not set; skipping sharp odds for %s", sport)
            return []
        url = f"{self.cfg.endpoints.odds_api_base_url}/sports/{sport}/odds"
        params = {"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "decimal"}
        try:
            data = await self.http.get_json(url, "odds", params=params)
        except HTTPRequestError as exc:
            logger.warning("Odds API failed for %s: %s", sport, exc)
            return []
        remaining = self.http.last_headers.get("odds", {}).get("x-requests-remaining")
        if remaining is not None:
            logger.info("Odds API quota remaining: %s", remaining)
        return [ev for ev in (self._parse_odds_event(e) for e in data or []) if ev is not None]

    @staticmethod
    def _parse_odds_event(raw: Mapping[str, Any]) -> Optional[SharpEvent]:
        book_odds: Dict[str, Dict[str, float]] = {}
        for bk in raw.get("bookmakers") or []:
            for mk in bk.get("markets") or []:
                if mk.get("key") != "h2h":
                    continue
                prices = {str(o["name"]): float(o["price"]) for o in mk.get("outcomes") or []
                          if o.get("name") and to_float(o.get("price")) > 1.0}
                if len(prices) >= 2:
                    book_odds[str(bk.get("key"))] = prices
        if not book_odds:
            return None
        sharp_books = [b for b in SHARP_BOOK_PRIORITY if b in book_odds]
        source_books = sharp_books or list(book_odds.keys())
        outcome_names = set.intersection(*(set(book_odds[b]) for b in source_books))
        consensus: Dict[str, float] = {}
        for name in outcome_names:
            vals = sorted(book_odds[b][name] for b in source_books)
            n = len(vals)
            consensus[name] = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        return SharpEvent(
            event_id=str(raw.get("id")), sport_key=str(raw.get("sport_key")),
            commence_time=parse_iso(raw.get("commence_time")),
            home_team=str(raw.get("home_team") or ""), away_team=str(raw.get("away_team") or ""),
            book_odds=book_odds, consensus_odds=consensus,
            primary_book=sharp_books[0] if sharp_books else None,
        )

    # -------------------------------------------------------------- pipeline
    async def run_ingestion_cycle(self, sports: Sequence[str] = (),
                                  max_pages: int = 200) -> Dict[str, Any]:
        t0 = time.monotonic()
        markets = await self.fetch_markets(max_pages=max_pages)
        s = self.cfg.strategy
        tradable = [m for m in markets if m.liquidity >= s.min_market_liquidity_usd
                    and m.volume_24h >= s.min_24h_volume_usd]
        snapshots = await self.snapshot_markets(tradable)
        sharp_events: List[SharpEvent] = []
        for sport in sports:
            sharp_events.extend(await self.fetch_sharp_odds(sport))
        pairs = match_polymarket_to_sharp_event(tradable, sharp_events) if sharp_events else []
        if self.db is not None:
            await self.db.upsert_markets(markets)
            await self.db.deactivate_missing_markets(m.condition_id for m in markets)
            await self.db.insert_orderbook_snapshots(snapshots)
        summary = {
            "markets_total": len(markets), "markets_tradable": len(tradable),
            "snapshots": len(snapshots), "sharp_events": len(sharp_events), "matched_pairs": len(pairs),
            "elapsed_s": round(time.monotonic() - t0, 2), "http": dict(self.http.stats),
        }
        logger.info("Ingestion cycle: %s", summary)
        return {"summary": summary, "markets": tradable, "all_markets": markets,
                "snapshots": snapshots, "pairs": pairs}


# =============================================================================
# Self-test
# =============================================================================

async def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --- double-encoded JSON parsing
    assert parse_json_list('["Yes", "No"]') == ["Yes", "No"]
    assert parse_json_list(["Yes", "No"]) == ["Yes", "No"]
    assert parse_json_list(json.dumps('["0.42", "0.58"]')) == ["0.42", "0.58"]
    assert parse_json_list(None) == [] and parse_json_list("") == []
    rec = parse_gamma_market({"conditionId": "0xc", "question": "Q?", "outcomes": '["Yes","No"]',
                              "clobTokenIds": json.dumps(json.dumps(["11", "22"])), "active": True},
                             {"slug": "ev", "tags": [{"label": "Sports"}]})
    assert rec and rec.token_ids == ["11", "22"] and rec.category == "Sports"

    # --- ladder sorting (deliberately unsorted input, string prices as from CLOB)
    raw_book = {"asks": [{"price": "0.55", "size": "1000"}, {"price": "0.50", "size": "200"},
                         {"price": "0.52", "size": "500"}],
                "bids": [{"price": "0.45", "size": "300"}, {"price": "0.48", "size": "100"}]}
    asks = normalize_ladder(raw_book["asks"], "asks")
    bids = normalize_ladder(raw_book["bids"], "bids")
    assert asks[0][0] == 0.50 and bids[0][0] == 0.48

    # --- VWAP: $100 fits entirely in the 0.50 level ($100 of depth)
    vwap, slip = calculate_effective_entry_price(asks, 100.0)
    assert abs(vwap - 0.50) < 1e-12 and abs(slip) < 1e-12
    # $500: $100 @0.50 (200 sh) + $260 @0.52 (500 sh) + $140 @0.55 (254.545 sh)
    vwap, slip = calculate_effective_entry_price(asks, 500.0)
    expected = 500.0 / (200 + 500 + 140 / 0.55)
    assert abs(vwap - expected) < 1e-12 and abs(slip - (expected - 0.50)) < 1e-12
    # Insufficient depth: total = 100 + 260 + 550 = $910 < $1,000
    assert calculate_effective_entry_price(asks, 1_000.0) == (None, None)
    assert calculate_effective_entry_price([], 100.0) == (None, None)

    snap = build_snapshot("tok", "0xc", raw_book)
    assert snap.best_ask == 0.50 and snap.best_bid == 0.48 and abs(snap.spread - 0.02) < 1e-12
    assert snap.vwap_1000 is None and snap.slippage_1000 is None and snap.vwap_500 is not None
    print(f"snapshot: mid={snap.mid:.3f} vwap500={snap.vwap_500:.4f} slip500={snap.slippage_500:.4f} "
          f"obi={snap.imbalance:.3f}")

    # --- fuzzy matcher
    ev = SharpEvent("e1", "basketball_nba", parse_iso("2026-11-01T00:00:00Z"), "Los Angeles Lakers",
                    "Boston Celtics", {"pinnacle": {"Los Angeles Lakers": 2.10, "Boston Celtics": 1.80}},
                    {"Los Angeles Lakers": 2.10, "Boston Celtics": 1.80}, "pinnacle")
    m1 = MarketRecord("0x1", "Lakers vs. Celtics", outcomes=["Lakers", "Celtics"], token_ids=["a", "b"],
                      end_date_iso="2026-11-01T03:00:00Z", event_slug="nba-lal-bos-2026-10-31")
    m2 = MarketRecord("0x2", "Will the Boston Celtics beat the Los Angeles Lakers?", outcomes=["Yes", "No"],
                      token_ids=["c", "d"], end_date_iso="2026-11-01T03:00:00Z")
    m3 = MarketRecord("0x3", "Will Bitcoin hit $200k?", outcomes=["Yes", "No"], token_ids=["e", "f"])
    pairs = {p.market.condition_id: p for p in match_polymarket_to_sharp_event([m1, m2, m3], [ev])}
    assert "0x3" not in pairs
    assert pairs["0x1"].outcome_map == {"Lakers": "Los Angeles Lakers", "Celtics": "Boston Celtics"}
    assert pairs["0x2"].outcome_map == {"Yes": "Boston Celtics", "No": "NOT:Boston Celtics"}
    print("matched:", {k: v.outcome_map for k, v in pairs.items()})
    print("devig input:", pairs["0x1"].to_devig_input())

    # --- optional live read-only test
    if os.getenv("RUN_LIVE_TEST") == "1":
        from config_and_database import load_config
        cfg = load_config(env_file=None)
        async with AsyncHTTPClient(cfg) as http:
            ing = MarketIngestor(cfg, http)
            markets = await ing.fetch_markets(max_pages=2)
            top = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:5]
            snaps = await ing.snapshot_markets(top)
            for s in snaps[:6]:
                print(f"LIVE {s.token_id[:12]}… bid={s.best_bid} ask={s.best_ask} "
                      f"vwap100={s.vwap_100} vwap1000={s.vwap_1000} obi={s.imbalance}")
            print("http stats:", http.stats)
    print("Self-test passed.")


if __name__ == "__main__":
    asyncio.run(_self_test())
