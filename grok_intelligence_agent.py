"""
grok_intelligence_agent.py
==========================
Module 4 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Responsibilities
----------------
1. Cost-aware pre-filter that picks which markets deserve a (paid) Grok live-search call.
2. Async Grok client: official `xai_sdk` when installed, otherwise native aiohttp against
   `/v1/responses` with `/v1/chat/completions` fallback, both with server-side
   `x_search` / `web_search` tools, a concurrency semaphore and exponential backoff.
3. Strict Pydantic schema (`GrokMarketAssessment`, `EvidenceSignal`) plus a code-level
   calibration guardrail that clamps likelihood ratios and weights by source tier.
4. Bridge into Module 3: prior -> Grok evidence -> log-odds posterior -> EV / Kelly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import aiohttp
from pydantic import BaseModel, Field, ValidationError, field_validator

from config_and_database import AppConfig, DatabaseManager, MarketRecord, OrderbookSnapshot
from async_market_ingestor import MatchedPair, parse_iso
from quant_math_engine import (
    EVEvaluation,
    compute_posterior_fair_value,
    evaluate_market_ev,
    hours_to_expiry,
    microstructure_prior,
    prior_from_matched_pair,
)

try:  # Official SDK is optional; the aiohttp path is always available.
    from xai_sdk import AsyncClient as XAIAsyncClient
    from xai_sdk.chat import system as xai_system, user as xai_user
    from xai_sdk.tools import web_search as xai_web_search, x_search as xai_x_search
    HAS_XAI_SDK = True
except Exception:  # pragma: no cover - depends on environment
    HAS_XAI_SDK = False

logger = logging.getLogger("pm_engine.grok")

XAI_BASE_URL = "https://api.x.ai/v1"
SourceTier = Literal["tier1_official", "tier2_verified_reporter", "tier3_secondary", "rumor"]
Direction = Literal["YES", "NO", "NEUTRAL"]

# (lr_min, lr_max, max_weight) per tier. Rumors carry no evidential weight at all.
TIER_CAPS: Dict[str, Tuple[float, float, float]] = {
    "tier1_official": (0.05, 20.0, 1.0),
    "tier2_verified_reporter": (0.25, 4.0, 0.8),
    "tier3_secondary": (0.5, 2.0, 0.5),
    "rumor": (1.0, 1.0, 0.0),
}
STALE_AFTER_DAYS = 14.0
MAX_SIGNALS = 10


# =============================================================================
# Schema
# =============================================================================

class EvidenceSignal(BaseModel):
    claim: str = Field(min_length=1, max_length=600)
    source_handle_or_domain: str = Field(default="unknown", max_length=200)
    source_tier: SourceTier = "tier3_secondary"
    published_at_iso: Optional[str] = None
    direction: Direction = "NEUTRAL"
    likelihood_ratio: float = 1.0
    confidence_weight: float = 0.0
    url: Optional[str] = None

    @field_validator("source_tier", mode="before")
    @classmethod
    def _norm_tier(cls, v: Any) -> str:
        s = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"tier1": "tier1_official", "official": "tier1_official",
                   "tier2": "tier2_verified_reporter", "verified_reporter": "tier2_verified_reporter",
                   "tier3": "tier3_secondary", "secondary": "tier3_secondary", "unverified": "rumor"}
        s = aliases.get(s, s)
        return s if s in TIER_CAPS else "tier3_secondary"

    @field_validator("direction", mode="before")
    @classmethod
    def _norm_direction(cls, v: Any) -> str:
        s = str(v or "NEUTRAL").strip().upper()
        return s if s in {"YES", "NO", "NEUTRAL"} else "NEUTRAL"

    @field_validator("likelihood_ratio", "confidence_weight", mode="before")
    @classmethod
    def _to_float(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return f


class GrokMarketAssessment(BaseModel):
    condition_id: str = ""
    summary: str = ""
    catalyst_summary: str = ""
    resolution_risk_score: int = Field(default=5)
    resolution_trap_warning: str = ""
    evidence_signals: List[EvidenceSignal] = Field(default_factory=list)
    model_probability_estimate: Optional[float] = None  # logged only, never used in the math
    citations: List[str] = Field(default_factory=list)
    raw_text: str = Field(default="", exclude=True)
    transport: str = ""

    @field_validator("resolution_risk_score", mode="before")
    @classmethod
    def _risk(cls, v: Any) -> int:
        try:
            return int(min(10, max(1, round(float(v)))))
        except (TypeError, ValueError):
            return 5


ASSESSMENT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["summary", "resolution_risk_score", "resolution_trap_warning", "evidence_signals"],
    "properties": {
        "summary": {"type": "string"},
        "catalyst_summary": {"type": "string"},
        "resolution_risk_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "resolution_trap_warning": {"type": "string"},
        "model_probability_estimate": {"type": ["number", "null"]},
        "evidence_signals": {"type": "array", "items": {
            "type": "object",
            "required": ["claim", "source_handle_or_domain", "source_tier", "direction",
                         "likelihood_ratio", "confidence_weight"],
            "properties": {
                "claim": {"type": "string"},
                "source_handle_or_domain": {"type": "string"},
                "source_tier": {"enum": list(TIER_CAPS.keys())},
                "published_at_iso": {"type": ["string", "null"]},
                "direction": {"enum": ["YES", "NO", "NEUTRAL"]},
                "likelihood_ratio": {"type": "number", "exclusiveMinimum": 0},
                "confidence_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "url": {"type": ["string", "null"]},
            }}},
    },
}

SYSTEM_PROMPT = """You are a Quantitative Superforecaster and Resolution Rule Auditor on an algorithmic
prediction-market desk. You never output a trade recommendation or a price target. Your job is to
(1) audit the contract's resolution rules and (2) extract discrete pieces of NEW evidence from live X
posts and web news, each scored as a calibrated Bayesian likelihood ratio.

Definitions:
- "YES" means the FIRST listed outcome resolves true. "NO" means it does not.
- likelihood_ratio = P(this evidence would exist | YES) / P(this evidence would exist | NO).
  1.0 = uninformative. 2.0 = twice as likely under YES. 0.5 = twice as likely under NO.
  Use values near 1.0 unless the evidence is direct and decisive. Do not double count: if several
  outlets report the same underlying fact, emit ONE signal citing the primary source.
- Evidence already priced in long ago (older than two weeks, widely known) should have LR near 1.0.
- source_tier: tier1_official (governing body, official account, court, exchange, the principal),
  tier2_verified_reporter (named beat reporter or established outlet), tier3_secondary (aggregators,
  blogs, commentary), rumor (anonymous, unsourced, or engagement bait).
- confidence_weight in [0, 1]: how sure you are the claim is real and relevant to THIS contract.

Resolution audit: read the rules text carefully for oracle/source mismatch, timezone or date cutoffs,
ambiguous wording, "Other"/catch-all outcomes, early-resolution clauses, and edge cases where the
headline question and the rules disagree. Score resolution_risk_score 1 (airtight) to 10 (trap) and
describe the main trap in resolution_trap_warning (empty string if none).

Respond with ONE JSON object only, no prose and no markdown, matching this JSON schema:
""" + json.dumps(ASSESSMENT_JSON_SCHEMA)


def build_user_prompt(market: MarketRecord, prices: Mapping[str, Optional[float]], prior: Mapping[str, float],
                      now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    hrs = hours_to_expiry(market.end_date_iso, now)
    lines = [
        f"Current UTC time: {now.isoformat(timespec='minutes')}",
        f"Market question: {market.question}",
        f"Outcomes (first = YES): {json.dumps(market.outcomes)}",
        f"Category: {market.category or 'unknown'}",
        f"Scheduled end: {market.end_date_iso or 'unknown'}"
        + (f" (~{hrs:.1f}h from now)" if hrs is not None else ""),
        "Current Polymarket ask prices: " + json.dumps({k: v for k, v in prices.items()}),
        "Baseline prior (from sharp books or order book): " + json.dumps({k: round(v, 4) for k, v in prior.items()}),
        "", "Resolution rules (audit verbatim):", market.description[:6000] or "(none provided)", "",
        "Search X (prioritize official accounts, governing bodies and named beat reporters) and the web "
        "for developments from the last 14 days that bear on this outcome. Return the JSON object.",
    ]
    return "\n".join(lines)


# =============================================================================
# Parsing & calibration guardrail
# =============================================================================

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of raw LLM text (handles markdown fences and surrounding prose)."""
    if not text or not text.strip():
        raise ValueError("Empty model output")
    candidates: List[str] = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        start = cand.find("{")
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cand)):
            ch = cand[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = cand[start:i + 1]
                    try:
                        obj = json.loads(snippet)
                    except json.JSONDecodeError:
                        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", snippet))  # trailing commas
                    if isinstance(obj, dict):
                        return obj
                    break
    raise ValueError("No JSON object found in model output")


def parse_assessment(text: str, condition_id: str, citations: Sequence[str] = (),
                     transport: str = "") -> GrokMarketAssessment:
    data = extract_json_object(text)
    if "evidence_signals" not in data:
        for alias in ("signals", "evidence", "evidenceSignals"):
            if alias in data:
                data["evidence_signals"] = data.pop(alias)
                break
    good: List[Dict[str, Any]] = []
    for raw in data.get("evidence_signals") or []:
        try:
            good.append(EvidenceSignal.model_validate(raw).model_dump())
        except ValidationError as exc:
            logger.debug("Dropping malformed signal %r: %s", raw, exc)
    data["evidence_signals"] = good
    data["condition_id"] = condition_id
    data["citations"] = list(dict.fromkeys(list(data.get("citations") or []) + list(citations)))
    data["transport"] = transport
    assessment = GrokMarketAssessment.model_validate(data)
    assessment.raw_text = text
    return assessment


def sanitize_and_calibrate_signals(signals: Sequence[EvidenceSignal],
                                   now: Optional[datetime] = None) -> List[EvidenceSignal]:
    """
    Code-level guardrail so model output can never blow up the Bayesian update:
    - rumors -> LR 1.0, weight 0.0
    - LR and weight clamped to per-tier bounds; NaN/inf/<=0 LR -> neutral
    - direction must agree with LR side, else neutralized; NEUTRAL -> LR 1.0
    - stale (> STALE_AFTER_DAYS) or future-dated evidence has its weight halved
    - duplicates (same source + claim) removed; at most MAX_SIGNALS kept, strongest first
    """
    now = now or datetime.now(timezone.utc)
    out: List[EvidenceSignal] = []
    seen: set = set()
    for sig in signals:
        s = sig.model_copy()
        lo, hi, wmax = TIER_CAPS.get(s.source_tier, TIER_CAPS["tier3_secondary"])
        lr = s.likelihood_ratio if math.isfinite(s.likelihood_ratio) and s.likelihood_ratio > 0 else 1.0
        w = s.confidence_weight if math.isfinite(s.confidence_weight) else 0.0
        lr = min(max(lr, lo), hi)
        w = min(max(w, 0.0), wmax)
        if s.direction == "NEUTRAL" or (s.direction == "YES" and lr < 1.0) or (s.direction == "NO" and lr > 1.0):
            lr, w = 1.0, 0.0
        published = parse_iso(s.published_at_iso)
        if published is not None:
            age_days = (now - published).total_seconds() / 86_400
            if age_days > STALE_AFTER_DAYS or age_days < -1.0:
                w *= 0.5
        key = (s.source_handle_or_domain.strip().lower(), re.sub(r"\W+", " ", s.claim.lower()).strip()[:120])
        if key in seen:
            continue
        seen.add(key)
        s.likelihood_ratio, s.confidence_weight = float(lr), float(w)
        out.append(s)
    out.sort(key=lambda s: -abs(s.confidence_weight * math.log(s.likelihood_ratio)))
    return out[:MAX_SIGNALS]


# =============================================================================
# Pre-filter
# =============================================================================

@dataclass
class GrokCandidate:
    market: MarketRecord
    snapshots: Dict[str, OrderbookSnapshot]      # outcome -> snapshot
    prior_by_outcome: Dict[str, float]
    prior_source: str
    pair: Optional[MatchedPair]
    volume_velocity: Optional[float]
    hours_to_expiry: Optional[float]
    divergence: float
    score: float
    reasons: List[str] = field(default_factory=list)


def _binary_priors(market: MarketRecord, snaps: Dict[str, OrderbookSnapshot], pair: Optional[MatchedPair],
                   velocity: Optional[float], cfg: AppConfig) -> Tuple[Optional[Dict[str, float]], str]:
    if pair is not None:
        try:
            return prior_from_matched_pair(pair, cfg.strategy.devig_method), f"sharp:{pair.sharp.primary_book or 'consensus'}"
        except (KeyError, ValueError) as exc:
            logger.debug("Sharp prior failed for %s: %s", market.condition_id, exc)
    first, second = market.outcomes[0], market.outcomes[1]
    snap = snaps.get(first)
    mp = microstructure_prior(snap, market, velocity, cfg.strategy.max_spread) if snap else None
    if mp is None and second in snaps:
        mp2 = microstructure_prior(snaps[second], market, velocity, cfg.strategy.max_spread)
        if mp2 is not None:
            return {first: 1.0 - mp2.prior_prob, second: mp2.prior_prob}, "microstructure"
    if mp is None:
        return None, ""
    return {first: mp.prior_prob, second: 1.0 - mp.prior_prob}, "microstructure"


async def prefilter_markets_for_grok(cfg: AppConfig, markets: Sequence[MarketRecord],
                                     snapshots: Mapping[str, OrderbookSnapshot],
                                     pairs: Sequence[MatchedPair] = (),
                                     db: Optional[DatabaseManager] = None,
                                     velocities: Optional[Mapping[str, float]] = None,
                                     now: Optional[datetime] = None) -> List[GrokCandidate]:
    """Score binary markets and return the top `max_markets_per_cycle` worth a live Grok call."""
    g = cfg.grok
    pair_by_cid = {p.market.condition_id: p for p in pairs}
    candidates: List[GrokCandidate] = []
    for m in markets:
        if len(m.outcomes) != 2 or len(m.token_ids) != 2:
            continue
        snaps = {o: snapshots[t] for o, t in zip(m.outcomes, m.token_ids) if t in snapshots}
        first_snap = snaps.get(m.outcomes[0])
        if first_snap is None or first_snap.best_ask is None or first_snap.best_bid is None:
            continue  # dead / one-sided book
        if velocities is not None and m.condition_id in velocities:
            velocity: Optional[float] = velocities[m.condition_id]
        elif db is not None:
            velocity = await db.get_volume_velocity(m.condition_id)
        else:
            velocity = None
        pair = pair_by_cid.get(m.condition_id)
        priors, source = _binary_priors(m, snaps, pair, velocity, cfg)
        if priors is None:
            continue
        divergence = max(abs(priors[o] - s.best_ask) for o, s in snaps.items() if s.best_ask is not None)
        if source == "microstructure":
            divergence *= 0.5  # order-book prior is weaker evidence of mispricing than a sharp line
        hrs = hours_to_expiry(m.end_date_iso, now)
        spread = first_snap.spread or 0.0
        obi = abs(first_snap.imbalance or 0.0)
        reasons: List[str] = []
        if divergence >= cfg.strategy.min_edge_pct:
            reasons.append(f"prior divergence {divergence:.3f} ({source})")
        if velocity is not None and velocity >= g.prefilter_min_volume_velocity:
            reasons.append(f"volume velocity {velocity:.2f}x")
        if spread >= g.prefilter_min_spread:
            reasons.append(f"wide spread {spread:.3f}")
        catalyst = hrs is not None and 0 < hrs <= g.prefilter_catalyst_window_h
        if catalyst:
            reasons.append(f"resolves in {hrs:.1f}h")
        if obi >= 0.5:
            reasons.append(f"book imbalance {first_snap.imbalance:+.2f}")
        if not reasons:
            continue
        vel_norm = min(max((velocity or 0.0) - 1.0, 0.0) / 3.0, 1.0)
        score = (3.0 * min(divergence / 0.15, 1.0) + 2.0 * vel_norm
                 + 1.5 * (1.0 - hrs / g.prefilter_catalyst_window_h if catalyst and hrs is not None else 0.0)
                 + 1.0 * obi + 0.5 * min(spread / max(g.prefilter_min_spread * 3, 1e-9), 1.0))
        candidates.append(GrokCandidate(m, snaps, priors, source, pair, velocity, hrs, divergence,
                                        round(score, 4), reasons))
    candidates.sort(key=lambda c: (-c.score, c.market.condition_id))
    selected = candidates[: g.max_markets_per_cycle]
    logger.info("Grok pre-filter: %d/%d markets selected", len(selected), len(markets))
    return selected


# =============================================================================
# Grok client
# =============================================================================

class GrokAPIError(RuntimeError):
    """Raised when every Grok transport (SDK, /v1/responses, /v1/chat/completions) fails."""


class GrokIntelligenceClient:
    def __init__(self, cfg: AppConfig, use_sdk: Optional[bool] = None, cache_ttl_s: float = 1_800.0) -> None:
        self.cfg = cfg
        self.api_key = cfg.keys.xai_api_key
        self.use_sdk = HAS_XAI_SDK if use_sdk is None else (use_sdk and HAS_XAI_SDK)
        self._sem = asyncio.Semaphore(cfg.rate_limits.xai_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, GrokMarketAssessment]] = {}
        self.cache_ttl_s = cache_ttl_s
        self.stats: Dict[str, int] = {"calls": 0, "cache_hits": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "GrokIntelligenceClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=max(120.0, self.cfg.rate_limits.request_timeout_s * 6)),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
        return self._session

    def _responses_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        if self.cfg.grok.enable_web_search:
            tools.append({"type": "web_search"})
        if self.cfg.grok.enable_x_search:
            tools.append({"type": "x_search"})
        return tools

    async def _post_with_retry(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        rl = self.cfg.rate_limits
        last: Optional[str] = None
        for attempt in range(rl.max_retries + 1):
            try:
                async with self._session_get().post(url, json=body) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        return json.loads(text)
                    last = f"HTTP {resp.status}: {text[:300]}"
                    if resp.status not in (408, 409, 429, 500, 502, 503, 504):
                        raise GrokAPIError(last)
                    retry_after = resp.headers.get("Retry-After")
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                last, retry_after = repr(exc), None
            if attempt < rl.max_retries:
                ceiling = min(rl.backoff_cap_s, rl.backoff_base_s * 2 ** (attempt + 1))
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() \
                    else random.uniform(0, ceiling)
                await asyncio.sleep(delay)
        raise GrokAPIError(f"Exhausted retries for {url}: {last}")

    @staticmethod
    def _text_from_responses(data: Mapping[str, Any]) -> Tuple[str, List[str]]:
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            text = data["output_text"]
        else:
            parts: List[str] = []
            for item in data.get("output") or []:
                if item.get("type") == "message":
                    for c in item.get("content") or []:
                        if c.get("type") in ("output_text", "text") and c.get("text"):
                            parts.append(c["text"])
            text = "\n".join(parts)
        cites = [c if isinstance(c, str) else c.get("url", "") for c in data.get("citations") or []]
        for item in data.get("output") or []:
            for c in item.get("content") or []:
                for ann in c.get("annotations") or []:
                    if ann.get("url"):
                        cites.append(ann["url"])
        return text, [c for c in cites if c]

    async def _call_responses(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str]]:
        body = {"model": self.cfg.grok.model, "temperature": self.cfg.grok.temperature,
                "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "tools": self._responses_tools()}
        data = await self._post_with_retry(f"{XAI_BASE_URL}/responses", body)
        usage = data.get("usage") or {}
        self.stats["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        self.stats["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        return self._text_from_responses(data)

    async def _call_chat_completions(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str]]:
        sources = ([{"type": "web"}] if self.cfg.grok.enable_web_search else []) + \
                  ([{"type": "x"}] if self.cfg.grok.enable_x_search else [])
        body: Dict[str, Any] = {
            "model": self.cfg.grok.model, "temperature": self.cfg.grok.temperature,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
        }
        if sources:
            body["search_parameters"] = {"mode": "auto", "sources": sources, "return_citations": True}
        data = await self._post_with_retry(f"{XAI_BASE_URL}/chat/completions", body)
        usage = data.get("usage") or {}
        self.stats["input_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self.stats["output_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return text, [str(c) for c in data.get("citations") or []]

    async def _call_sdk(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str]]:
        client = XAIAsyncClient(api_key=self.api_key)
        tools = []
        if self.cfg.grok.enable_web_search:
            tools.append(xai_web_search())
        if self.cfg.grok.enable_x_search:
            tools.append(xai_x_search(from_date=datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)))
        chat = client.chat.create(model=self.cfg.grok.model, tools=tools or None,
                                  temperature=self.cfg.grok.temperature)
        chat.append(xai_system(system_prompt))
        chat.append(xai_user(user_prompt))
        response = await chat.sample()
        return response.content or "", list(getattr(response, "citations", []) or [])

    async def _call_model(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str], str]:
        """Try SDK -> /responses -> /chat/completions. Returns (text, citations, transport)."""
        errors: List[str] = []
        if self.use_sdk:
            try:
                text, cites = await self._call_sdk(system_prompt, user_prompt)
                if text.strip():
                    return text, cites, "xai_sdk"
            except Exception as exc:  # SDK surface can change between versions
                errors.append(f"sdk: {exc!r}")
        try:
            text, cites = await self._call_responses(system_prompt, user_prompt)
            if text.strip():
                return text, cites, "responses"
        except GrokAPIError as exc:
            errors.append(f"responses: {exc}")
        text, cites = await self._call_chat_completions(system_prompt, user_prompt)
        if not text.strip():
            raise GrokAPIError(f"Empty output from all transports: {errors}")
        return text, cites, "chat_completions"

    async def assess_market(self, candidate: GrokCandidate) -> GrokMarketAssessment:
        m = candidate.market
        cached = self._cache.get(m.condition_id)
        if cached and time.time() - cached[0] < self.cache_ttl_s:
            self.stats["cache_hits"] += 1
            return cached[1]
        prices = {o: (s.best_ask if s else None) for o, s in candidate.snapshots.items()}
        user_prompt = build_user_prompt(m, prices, candidate.prior_by_outcome)
        async with self._sem:
            self.stats["calls"] += 1
            try:
                text, cites, transport = await self._call_model(SYSTEM_PROMPT, user_prompt)
                assessment = parse_assessment(text, m.condition_id, cites, transport)
            except (GrokAPIError, ValueError, ValidationError) as exc:
                self.stats["errors"] += 1
                logger.warning("Grok assessment failed for %s: %s", m.condition_id, exc)
                return GrokMarketAssessment(condition_id=m.condition_id, summary=f"assessment failed: {exc}",
                                            resolution_risk_score=self.cfg.strategy.max_resolution_risk + 1,
                                            resolution_trap_warning="Grok unavailable; rules not audited.",
                                            transport="error")
        assessment.evidence_signals = sanitize_and_calibrate_signals(assessment.evidence_signals)
        self._cache[m.condition_id] = (time.time(), assessment)
        return assessment


# =============================================================================
# Bridge to Module 3
# =============================================================================

def build_rationale(a: GrokMarketAssessment, max_sources: int = 3) -> str:
    srcs = [f"{s.source_handle_or_domain} ({s.source_tier}, LR {s.likelihood_ratio:.2f})"
            for s in a.evidence_signals if s.confidence_weight > 0][:max_sources]
    parts = [a.catalyst_summary or a.summary]
    if srcs:
        parts.append("Sources: " + "; ".join(srcs))
    if a.resolution_trap_warning:
        parts.append(f"Resolution risk {a.resolution_risk_score}/10: {a.resolution_trap_warning}")
    return " | ".join(p for p in parts if p)[:1500]


async def evaluate_candidates_with_grok(cfg: AppConfig, candidates: Sequence[GrokCandidate],
                                        client: GrokIntelligenceClient,
                                        arb_roi_by_token: Optional[Mapping[str, float]] = None
                                        ) -> List[Tuple[EVEvaluation, GrokMarketAssessment]]:
    """Run Grok on each candidate, then evaluate both outcomes through Module 3's math."""
    async def one(c: GrokCandidate) -> List[Tuple[EVEvaluation, GrokMarketAssessment]]:
        if client.enabled:
            assessment = await client.assess_market(c)
        else:
            assessment = GrokMarketAssessment(condition_id=c.market.condition_id,
                                              summary="Grok disabled (no XAI_API_KEY); prior only.",
                                              transport="disabled")
        signals = [s for s in assessment.evidence_signals if s.confidence_weight > 0]
        yes_lrs = [s.likelihood_ratio for s in signals]
        weights = [s.confidence_weight for s in signals]
        rationale = build_rationale(assessment)
        results: List[Tuple[EVEvaluation, GrokMarketAssessment]] = []
        for idx, outcome in enumerate(c.market.outcomes):
            snap = c.snapshots.get(outcome)
            prior = c.prior_by_outcome.get(outcome)
            if snap is None or prior is None or not 0.0 < prior < 1.0:
                continue
            lrs = yes_lrs if idx == 0 else [1.0 / lr for lr in yes_lrs]  # complement inverts evidence
            posterior = compute_posterior_fair_value(prior, lrs, weights)
            ev = evaluate_market_ev(cfg, c.market, outcome, snap, posterior, c.prior_source, lrs, weights,
                                    resolution_risk=assessment.resolution_risk_score, rationale=rationale,
                                    arb_roi=(arb_roi_by_token or {}).get(snap.token_id))
            if ev is not None:
                results.append((ev, assessment))
        return results

    nested = await asyncio.gather(*(one(c) for c in candidates), return_exceptions=True)
    out: List[Tuple[EVEvaluation, GrokMarketAssessment]] = []
    for c, res in zip(candidates, nested):
        if isinstance(res, BaseException):
            logger.error("Evaluation failed for %s: %r", c.market.condition_id, res)
            continue
        out.extend(res)
    return out


# =============================================================================
# Self-test
# =============================================================================

class _MockGrokClient(GrokIntelligenceClient):
    """Exercises the real parse + sanitize path with canned model output."""

    def __init__(self, cfg: AppConfig, canned: str) -> None:
        super().__init__(cfg, use_sdk=False)
        self.api_key = "mock"
        self.canned = canned

    async def _call_model(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str], str]:
        assert "Resolution rules" in user_prompt and "likelihood_ratio" in system_prompt
        return self.canned, ["https://example.org/mock"], "mock"


async def _self_test() -> None:
    from config_and_database import load_config
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(env_file=None)
    now = datetime.now(timezone.utc)
    soon = (now + timedelta(hours=20)).isoformat()
    later = (now + timedelta(days=60)).isoformat()

    def snap(tok: str, cid: str, bid: Optional[float], ask: Optional[float], obi: float = 0.0) -> OrderbookSnapshot:
        if bid is None or ask is None:
            return OrderbookSnapshot(tok, cid, bid, ask, None, None, None, None, None, None, None, None, 0, 0, None)
        return OrderbookSnapshot(tok, cid, bid, ask, ask - bid, (ask + bid) / 2, ask, ask + .005, ask + .01,
                                 0.0, .005, .01, 4000, 4000, obi)

    # --- pre-filter
    dead = MarketRecord("0xdead", "Dead market?", outcomes=["Yes", "No"], token_ids=["d1", "d2"], end_date_iso=later)
    quiet = MarketRecord("0xquiet", "Quiet market?", outcomes=["Yes", "No"], token_ids=["q1", "q2"], end_date_iso=later)
    hot = MarketRecord("0xhot", "Will the Fed cut rates in December?", outcomes=["Yes", "No"],
                       token_ids=["h1", "h2"], end_date_iso=soon, description="Resolves per FOMC statement.")
    snaps = {"d1": snap("d1", "0xdead", None, None), "q1": snap("q1", "0xquiet", .49, .50),
             "q2": snap("q2", "0xquiet", .50, .51), "h1": snap("h1", "0xhot", .30, .36, 0.7),
             "h2": snap("h2", "0xhot", .64, .70, -0.7)}
    cands = await prefilter_markets_for_grok(cfg, [dead, quiet, hot], snaps,
                                             velocities={"0xquiet": 0.9, "0xhot": 3.2}, now=now)
    assert [c.market.condition_id for c in cands] == ["0xhot"], [c.market.condition_id for c in cands]
    print(f"Prefilter: selected {cands[0].market.condition_id} score={cands[0].score} reasons={cands[0].reasons}")

    # --- JSON extraction & guardrails
    raw = """Here is my analysis:
```json
{"summary": "FOMC minutes and Fed speakers lean dovish.", "catalyst_summary": "Powell speech Friday",
 "resolution_risk_score": 12, "resolution_trap_warning": "Statement timing vs ET cutoff.",
 "evidence_signals": [
  {"claim": "Fed Chair signals openness to December cut", "source_handle_or_domain": "@federalreserve",
   "source_tier": "tier1_official", "published_at_iso": "%s", "direction": "YES",
   "likelihood_ratio": 500, "confidence_weight": 1.7},
  {"claim": "Reporter: two governors favor a cut", "source_handle_or_domain": "@NickTimiraos",
   "source_tier": "tier2_verified_reporter", "published_at_iso": "%s", "direction": "YES",
   "likelihood_ratio": 9.0, "confidence_weight": 0.9},
  {"claim": "Anon account says cut is locked", "source_handle_or_domain": "@leaks123",
   "source_tier": "rumor", "direction": "YES", "likelihood_ratio": 50, "confidence_weight": 1.0},
  {"claim": "Strong jobs print reduces cut odds", "source_handle_or_domain": "bls.gov",
   "source_tier": "tier1_official", "published_at_iso": "%s", "direction": "NO",
   "likelihood_ratio": 3.0, "confidence_weight": 0.8},
  {"claim": "Reporter: two governors favor a cut", "source_handle_or_domain": "@NickTimiraos",
   "source_tier": "tier2_verified_reporter", "direction": "YES", "likelihood_ratio": 2.0, "confidence_weight": 0.5},
 ]
}
```
Hope this helps.""" % (now.isoformat(), now.isoformat(), (now - timedelta(days=30)).isoformat())
    a = parse_assessment(raw, "0xhot")
    assert a.resolution_risk_score == 10 and len(a.evidence_signals) == 5
    clean = sanitize_and_calibrate_signals(a.evidence_signals, now)
    by_src = {s.source_handle_or_domain: s for s in clean}
    assert by_src["@federalreserve"].likelihood_ratio == 20.0 and by_src["@federalreserve"].confidence_weight == 1.0
    assert by_src["@NickTimiraos"].likelihood_ratio == 4.0 and by_src["@NickTimiraos"].confidence_weight == 0.8
    assert by_src["@leaks123"].likelihood_ratio == 1.0 and by_src["@leaks123"].confidence_weight == 0.0
    assert by_src["bls.gov"].likelihood_ratio == 1.0  # direction NO with LR > 1 is contradictory -> neutral
    assert len(clean) == 4  # duplicate reporter claim removed
    try:
        extract_json_object("no json here")
        raise AssertionError("expected failure")
    except ValueError as exc:
        assert "JSON" in str(exc) or "json" in str(exc), exc
    print("Guardrails:", [(s.source_handle_or_domain, s.likelihood_ratio, s.confidence_weight) for s in clean])

    # --- full bridge with mocked Grok output
    mock_json = json.dumps({
        "summary": "Dovish shift", "catalyst_summary": "Fed speakers signal a December cut",
        "resolution_risk_score": 3, "resolution_trap_warning": "",
        "evidence_signals": [
            {"claim": "Chair signals cut", "source_handle_or_domain": "@federalreserve",
             "source_tier": "tier1_official", "published_at_iso": now.isoformat(), "direction": "YES",
             "likelihood_ratio": 3.0, "confidence_weight": 0.9},
            {"claim": "Two governors favor cut", "source_handle_or_domain": "@NickTimiraos",
             "source_tier": "tier2_verified_reporter", "published_at_iso": now.isoformat(), "direction": "YES",
             "likelihood_ratio": 2.0, "confidence_weight": 0.7}]})
    client = _MockGrokClient(cfg, mock_json)
    results = await evaluate_candidates_with_grok(cfg, cands, client)
    assert len(results) == 2
    yes_ev = next(e for e, _ in results if e.record.outcome == "Yes")
    no_ev = next(e for e, _ in results if e.record.outcome == "No")
    py, pn = yes_ev.record.posterior_prob, no_ev.record.posterior_prob
    assert abs(py + pn - 1.0) < 1e-9, (py, pn)  # complementary posteriors stay coherent
    assert py > cands[0].prior_by_outcome["Yes"] and yes_ev.record.signal_type in ("ev", "high_conviction")
    assert no_ev.record.signal_type == "pass"
    for e, _ in results:
        r = e.record
        print(f"  {r.outcome}: prior={r.prior_prob:.4f} -> P_true={r.posterior_prob:.4f} entry={r.entry_vwap:.3f} "
              f"ROI={r.expected_roi:+.2%} kelly={r.kelly_stake_pct:.4f} signal={r.signal_type}")
    cached = await client.assess_market(cands[0])
    assert client.stats["cache_hits"] == 1 and cached.condition_id == "0xhot"

    # --- optional live check
    if os.getenv("RUN_LIVE_GROK_TEST") == "1" and cfg.keys.xai_api_key:
        async with GrokIntelligenceClient(cfg) as live:
            res = await live.assess_market(cands[0])
            print(f"LIVE via {res.transport}: risk={res.resolution_risk_score} signals={len(res.evidence_signals)} "
                  f"citations={len(res.citations)} stats={live.stats}")
    elif os.getenv("RUN_LIVE_GROK_TEST") == "1":
        print("Live Grok test skipped: XAI_API_KEY not set.")
    print(f"xai_sdk installed: {HAS_XAI_SDK}")
    print("Self-test passed.")


if __name__ == "__main__":
    asyncio.run(_self_test())
