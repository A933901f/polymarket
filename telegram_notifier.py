"""
telegram_notifier.py
====================
Module 5 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Responsibilities
----------------
1. Strict HTML escaping and tag-aware chunking for Telegram's HTML parse mode.
2. Rich trade cards: +EV / high-conviction, pure arbitrage, and operational heartbeat.
3. Async dispatcher with token-bucket rate limiting, 429 `retry_after` handling,
   exponential backoff on 5xx / network errors, plain-text fallback on HTML 400s,
   dry-run mode, and alert deduplication through Module 1's `alert_history`.
"""

from __future__ import annotations

import asyncio
import html
import logging
import math
import os
import random
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiohttp

from config_and_database import AppConfig, DatabaseManager, FairValueRecord, MarketRecord, OrderbookSnapshot
from async_market_ingestor import TokenBucket
from quant_math_engine import ArbLeg, ArbOpportunity, EVEvaluation, hours_to_expiry
from grok_intelligence_agent import EvidenceSignal, GrokMarketAssessment

logger = logging.getLogger("pm_engine.telegram")

TELEGRAM_MAX_LEN = 4096
DEFAULT_CHUNK_LEN = 3900
ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler")
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*)>")

BANNERS = {
    "high_conviction": "🚨🚨 <b>HIGH-CONVICTION +EV (ROI ≥ 100%)</b> 🚨🚨",
    "ev": "📈 <b>+EV MISPRICING</b>",
    "arb": "🔒 <b>RISK-FREE ARBITRAGE</b>",
    "pass": "ℹ️ <b>MONITOR (below thresholds)</b>",
}
TIER_ICONS = {"tier1_official": "🏛", "tier2_verified_reporter": "📰", "tier3_secondary": "🔗", "rumor": "❔"}


# =============================================================================
# Escaping & chunking
# =============================================================================

def escape_html(value: Any) -> str:
    """Escape any dynamic value for Telegram HTML (handles <, >, &, and quotes)."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def safe_url(url: Optional[str]) -> str:
    """Only allow http(s) links; escape for use inside href."""
    if not url:
        return ""
    url = str(url).strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return ""
    return escape_html(url)


def link(url: Optional[str], label: str) -> str:
    href = safe_url(url)
    return f'<a href="{href}">{escape_html(label)}</a>' if href else escape_html(label)


def strip_html(text: str) -> str:
    """Convert Telegram HTML to plain text (for the 400 fallback)."""
    text = re.sub(r'<a\s+href="([^"]*)"\s*>(.*?)</a>', lambda m: f"{m.group(2)} ({m.group(1)})", text, flags=re.S)
    text = _TAG_RE.sub("", text)
    return html.unescape(text)


def _open_tag_stack(text: str) -> List[Tuple[str, str]]:
    """Return unclosed (tag_name, full_open_tag) pairs at the end of `text`."""
    stack: List[Tuple[str, str]] = []
    for m in _TAG_RE.finditer(text):
        closing, name = m.group(1) == "/", m.group(2).lower()
        if name not in ALLOWED_TAGS:
            continue
        if not closing:
            stack.append((name, m.group(0)))
        else:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    del stack[i]
                    break
    return stack


def _safe_cut(text: str, limit: int) -> int:
    """Largest cut index <= limit that is not inside a tag or an HTML entity."""
    cut = limit
    lt, gt = text.rfind("<", 0, cut), text.rfind(">", 0, cut)
    if lt > gt:
        cut = lt
    amp = text.rfind("&", 0, cut)
    if amp != -1 and ";" not in text[amp:cut] and cut - amp < 10:
        cut = amp
    space = text.rfind(" ", 0, cut)
    if space > cut * 0.6:
        cut = space + 1
    return max(cut, 1)


def chunk_html_message(text: str, max_len: int = DEFAULT_CHUNK_LEN) -> List[str]:
    """
    Split on paragraph, then line, then safe character boundaries. Tags left open at a cut
    are closed at the end of the chunk and reopened at the start of the next one.
    """
    if len(text) <= max_len:
        return [text]
    budget = max_len - 200  # room for re-closing / re-opening tags
    pieces: List[str] = []
    for para in text.split("\n\n"):
        if len(para) <= budget:
            pieces.append(para)
            continue
        for line in para.split("\n"):
            while len(line) > budget:
                cut = _safe_cut(line, budget)
                pieces.append(line[:cut])
                line = line[cut:]
            pieces.append(line)
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        sep = "\n\n" if current and "\n\n" in text else "\n"
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        chunks.append(current)
    balanced: List[str] = []
    carry: List[Tuple[str, str]] = []
    for chunk in chunks:
        body = "".join(tag for _, tag in carry) + chunk
        carry = _open_tag_stack(body)
        body += "".join(f"</{name}>" for name, _ in reversed(carry))
        balanced.append(body)
    return balanced


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_pct(x: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def fmt_price(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def fmt_duration(hours: Optional[float]) -> str:
    if hours is None:
        return "unknown"
    if hours < 1:
        return f"{hours * 60:.0f}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def market_url(cfg: AppConfig, market: MarketRecord) -> str:
    return f"{cfg.endpoints.polymarket_event_url}/{market.event_slug or market.slug}"


# =============================================================================
# Cards
# =============================================================================

def format_ev_card(ev: EVEvaluation, market: MarketRecord, snap: Optional[OrderbookSnapshot],
                   assessment: Optional[GrokMarketAssessment], cfg: AppConfig, max_signals: int = 5) -> str:
    r: FairValueRecord = ev.record
    hrs = hours_to_expiry(market.end_date_iso)
    is_complement = bool(market.outcomes) and r.outcome != market.outcomes[0]
    lines: List[str] = [
        BANNERS.get(r.signal_type, BANNERS["ev"]),
        "",
        f"<b>{escape_html(market.question)}</b>",
        f"🎯 Buy <b>{escape_html(r.outcome)}</b> · {link(market_url(cfg, market), 'Open on Polymarket')}",
        f"⏳ Resolves in <b>{fmt_duration(hrs)}</b>" + (f" ({escape_html(market.end_date_iso)})" if market.end_date_iso else ""),
        "",
        "<b>💹 Pricing &amp; Microstructure</b>",
    ]
    if snap is not None:
        lines += [
            f"Bid / Ask: <code>{fmt_price(snap.best_bid)}</code> / <code>{fmt_price(snap.best_ask)}</code> "
            f"· Spread <code>{fmt_price(snap.spread)}</code>",
            f"OBI: <code>{(snap.imbalance if snap.imbalance is not None else 0):+.2f}</code> "
            f"· Depth bid {fmt_usd(snap.bid_depth_usd)} / ask {fmt_usd(snap.ask_depth_usd)}",
            f"VWAP $100 <code>{fmt_price(snap.vwap_100)}</code> · $500 <code>{fmt_price(snap.vwap_500)}</code> "
            f"· $1000 <code>{fmt_price(snap.vwap_1000)}</code>",
        ]
    tier = f"${ev.clip_tier:,.0f} clip" if ev.clip_tier else "top of book"
    lines.append(f"Executable entry: <b><code>{fmt_price(r.entry_vwap)}</code></b> ({tier})"
                 + (" ⚠️ <i>depth-limited</i>" if ev.depth_limited else ""))
    lines += ["", "<b>🧮 Bayesian Fair Value</b>",
              f"Prior: <code>{fmt_pct(r.prior_prob)}</code> <i>({escape_html(r.prior_source)})</i>"]
    signals: List[EvidenceSignal] = [s for s in (assessment.evidence_signals if assessment else [])
                                     if s.confidence_weight > 0][:max_signals]
    if signals:
        lines.append("Evidence" + (" <i>(LRs inverted for this outcome)</i>" if is_complement else "") + ":")
        for s in signals:
            lr = 1.0 / s.likelihood_ratio if is_complement else s.likelihood_ratio
            lines.append(
                f"  {TIER_ICONS.get(s.source_tier, '•')} {escape_html(s.source_handle_or_domain)} "
                f"<i>[{escape_html(s.source_tier)}]</i> {escape_html(s.direction)} "
                f"LR <code>{lr:.2f}</code> w <code>{s.confidence_weight:.2f}</code>")
            lines.append(f"     <i>{escape_html(s.claim[:220])}</i>")
    else:
        lines.append("Evidence: <i>none weighted (prior only)</i>")
    lines.append(f"Posterior P_true: <b><code>{fmt_pct(r.posterior_prob)}</code></b> "
                 f"· Fair odds <code>{r.fair_decimal_odds:.3f}</code>")
    if assessment and (assessment.catalyst_summary or assessment.summary):
        lines.append(f"🗞 {escape_html((assessment.catalyst_summary or assessment.summary)[:400])}")
    lines += [
        "", "<b>💰 Edge &amp; Sizing</b>",
        f"Expected ROI: <b>{fmt_pct(r.expected_roi, signed=True)}</b> · Edge <b>{fmt_pct(r.edge_pct, signed=True)}</b>",
        f"Stake ({cfg.strategy.kelly_fraction:g}× Kelly): <b>{fmt_usd(ev.stake_usd)}</b> "
        f"= {fmt_pct(r.kelly_stake_pct)} of {fmt_usd(cfg.strategy.bankroll_usd)}",
        "", "<b>⚖️ Resolution Audit</b>",
    ]
    risk = r.resolution_risk
    risk_icon = "🟢" if risk <= 3 else "🟡" if risk <= 6 else "🔴"
    lines.append(f"{risk_icon} Risk score: <b>{risk}/10</b>")
    if assessment and assessment.resolution_trap_warning:
        lines.append(f"⚠️ {escape_html(assessment.resolution_trap_warning[:500])}")
    cites = [c for c in (assessment.citations if assessment else []) if safe_url(c)][:3]
    if cites:
        lines.append("Sources: " + " · ".join(link(c, f"[{i + 1}]") for i, c in enumerate(cites)))
    lines.append(f"<i>cid {escape_html(market.condition_id[:18])}… · token {escape_html(r.token_id[:12])}…</i>")
    return "\n".join(lines)


def format_arb_card(arb: ArbOpportunity, markets_by_cid: Mapping[str, MarketRecord], cfg: AppConfig) -> str:
    kind_label = {"intra_market": "Intra-market book arb", "neg_risk_event": "Neg-risk event arb",
                  "cross_venue": "Cross-venue (Polymarket vs sharp book)"}.get(arb.kind, arb.kind)
    lines = [BANNERS["arb"], "", f"Type: <b>{escape_html(kind_label)}</b>"]
    for cid in arb.condition_ids[:5]:
        m = markets_by_cid.get(cid)
        if m:
            lines.append(f"• {escape_html(m.question)} · {link(market_url(cfg, m), 'Polymarket')}")
    lines += [
        "",
        f"Implied sum: <code>{arb.implied_sum:.4f}</code>",
        f"Guaranteed ROI: <b>{fmt_pct(arb.roi, signed=True)}</b>",
        f"Outlay: <b>{fmt_usd(arb.outlay_usd)}</b> → Profit <b>{fmt_usd(arb.guaranteed_profit_usd)}</b>",
        "", "<b>Legs</b>",
    ]
    for i, leg in enumerate(arb.legs, 1):
        lines.append(
            f"{i}. <b>{escape_html(leg.venue)}</b> · {escape_html(leg.outcome)}\n"
            f"   price <code>{leg.price:.4f}</code> · odds <code>{leg.decimal_odds:.3f}</code> · "
            f"stake <b>{fmt_usd(leg.stake_usd)}</b> · pays {fmt_usd(leg.payout_usd)}")
    if arb.notes:
        lines += ["", f"<i>{escape_html(arb.notes)}</i>"]
    return "\n".join(lines)


def format_heartbeat_card(cycle_summary: Mapping[str, Any], db_stats: Mapping[str, int],
                          clv_stats: Mapping[str, Any], grok_stats: Mapping[str, int]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"💓 <b>Engine heartbeat</b> · {escape_html(now)}", "", "<b>Last cycle</b>"]
    for k, v in cycle_summary.items():
        if isinstance(v, Mapping):
            v = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
        lines.append(f"  {escape_html(k)}: <code>{escape_html(v)}</code>")
    lines += ["", "<b>Database</b>"] + [f"  {escape_html(k)}: <code>{v:,}</code>" for k, v in db_stats.items()]
    lines += ["", "<b>Closing line value</b>"]
    if clv_stats:
        for sig, row in clv_stats.items():
            avg = row.get("avg_clv")
            beat = row.get("beat_rate")
            lines.append(f"  {escape_html(sig)}: n=<code>{row.get('n', 0)}</code> avg CLV "
                         f"<code>{(avg or 0):+.4f}</code> beat <code>{fmt_pct(beat, 1)}</code>")
    else:
        lines.append("  <i>no closed positions yet</i>")
    lines += ["", "<b>Grok usage</b>"] + [f"  {escape_html(k)}: <code>{v:,}</code>" for k, v in grok_stats.items()]
    return "\n".join(lines)


# =============================================================================
# Dispatcher
# =============================================================================

class TelegramNotifier:
    def __init__(self, cfg: AppConfig, db: Optional[DatabaseManager] = None) -> None:
        self.cfg = cfg
        self.db = db
        self.bucket = TokenBucket(cfg.rate_limits.telegram_rps, capacity=1.0, name="telegram")
        self._session: Optional[aiohttp.ClientSession] = None
        self.sent_log: List[str] = []
        self.stats: Dict[str, int] = {"sent": 0, "suppressed": 0, "failed": 0, "plain_fallbacks": 0, "retries": 0}

    @property
    def dry_run(self) -> bool:
        k = self.cfg.keys
        return self.cfg.runtime.dry_run or not k.telegram_bot_token or not k.telegram_chat_id

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "TelegramNotifier":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.cfg.rate_limits.request_timeout_s))
        return self._session

    async def _post(self, text: str, html_mode: bool) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.cfg.endpoints.telegram_base_url}/bot{self.cfg.keys.telegram_bot_token}/sendMessage"
        body: Dict[str, Any] = {"chat_id": self.cfg.keys.telegram_chat_id, "text": text,
                                "disable_web_page_preview": True}
        if html_mode:
            body["parse_mode"] = "HTML"
        async with self._session_get().post(url, json=body) as resp:
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = {"ok": False, "description": (await resp.text())[:300]}
            return resp.status, data if isinstance(data, dict) else {}

    async def _send_chunk(self, text: str) -> bool:
        rl = self.cfg.rate_limits
        html_mode = True
        for attempt in range(rl.max_retries + 1):
            await self.bucket.acquire()
            try:
                status, data = await self._post(text if html_mode else strip_html(text), html_mode)
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                status, data = 0, {"description": repr(exc)}
            if status == 200 and data.get("ok", True):
                return True
            desc = str(data.get("description", ""))
            if status == 429:
                retry_after = float((data.get("parameters") or {}).get("retry_after", 1.0))
                self.bucket.penalize(retry_after)
                self.stats["retries"] += 1
                logger.warning("Telegram 429, retrying after %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                continue
            if status == 400 and html_mode:
                logger.warning("Telegram rejected HTML (%s); retrying as plain text", desc)
                html_mode = False
                self.stats["plain_fallbacks"] += 1
                continue
            if status == 0 or status >= 500:
                delay = random.uniform(0, min(rl.backoff_cap_s, rl.backoff_base_s * 2 ** (attempt + 1)))
                self.stats["retries"] += 1
                await asyncio.sleep(delay)
                continue
            logger.error("Telegram send failed (HTTP %s): %s", status, desc)
            return False
        return False

    async def send_message(self, text: str) -> bool:
        chunks = chunk_html_message(text)
        if self.dry_run:
            for c in chunks:
                logger.info("[DRY-RUN telegram]\n%s", strip_html(c))
                self.sent_log.append(c)
            self.stats["sent"] += 1
            return True
        ok = True
        for c in chunks:
            if await self._send_chunk(c):
                self.sent_log.append(c)
            else:
                ok = False
        self.stats["sent" if ok else "failed"] += 1
        return ok

    async def notify_ev(self, ev: EVEvaluation, market: MarketRecord, snap: Optional[OrderbookSnapshot],
                        assessment: Optional[GrokMarketAssessment]) -> str:
        r = ev.record
        if r.signal_type == "pass":
            return "skipped"
        s = self.cfg.strategy
        if self.db is not None and not await self.db.should_send_alert(
                r.condition_id, r.token_id, r.signal_type, r.expected_roi, s.alert_roi_delta, s.alert_cooldown_s):
            self.stats["suppressed"] += 1
            return "suppressed"
        if not await self.send_message(format_ev_card(ev, market, snap, assessment, self.cfg)):
            return "failed"
        if self.db is not None:
            await self.db.record_alert(r.condition_id, r.token_id, r.signal_type, r.expected_roi, r.entry_vwap)
        return "sent"

    @staticmethod
    def _arb_identity(arb: ArbOpportunity) -> Tuple[str, str, str]:
        refs = "|".join(sorted(f"{l.venue}:{l.ref}:{l.outcome}" for l in arb.legs))[:500]
        return ",".join(arb.condition_ids)[:500], refs, f"arb_{arb.kind}"

    async def notify_arb(self, arb: ArbOpportunity, markets_by_cid: Mapping[str, MarketRecord]) -> str:
        cid, ref, sig = self._arb_identity(arb)
        s = self.cfg.strategy
        if self.db is not None and not await self.db.should_send_alert(
                cid, ref, sig, arb.roi, s.alert_roi_delta, s.alert_cooldown_s):
            self.stats["suppressed"] += 1
            return "suppressed"
        if not await self.send_message(format_arb_card(arb, markets_by_cid, self.cfg)):
            return "failed"
        if self.db is not None:
            await self.db.record_alert(cid, ref, sig, arb.roi, arb.implied_sum)
        return "sent"

    async def send_heartbeat(self, cycle_summary: Mapping[str, Any], grok_stats: Mapping[str, int]) -> bool:
        db_stats = await self.db.health_stats() if self.db else {}
        clv = await self.db.clv_summary() if self.db else {}
        return await self.send_message(format_heartbeat_card(cycle_summary, db_stats, clv, grok_stats))


# =============================================================================
# Self-test
# =============================================================================

async def _self_test() -> None:
    from pathlib import Path
    from aiohttp import web
    from config_and_database import load_config
    from quant_math_engine import compute_posterior_fair_value, evaluate_market_ev
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(env_file=None)

    # --- escaping
    evil = '<script>alert("x")</script> BTC > $100k & ETH < $5k \'quoted\''
    esc = escape_html(evil)
    assert "<" not in esc and ">" not in esc and "&amp;" in esc and "&quot;" in esc and "&#x27;" in esc
    assert safe_url("javascript:alert(1)") == "" and "&amp;" in safe_url("https://x.com/a?b=1&c=2")
    assert strip_html(f"<b>{esc}</b>") == evil

    # --- EV card (+123% high conviction) with adversarial strings everywhere
    market = MarketRecord("0xhc1234567890abcdef", f"Will {evil} happen?", event_slug="btc-100k-<test>",
                          outcomes=["Yes", "No"], token_ids=["hy", "hn"], end_date_iso="2026-12-31T23:59:00Z")
    snap = OrderbookSnapshot("hy", market.condition_id, .16, .18, .02, .17, .18, .185, .19, 0, .005, .01, 6000, 3000, -.3)
    post = compute_posterior_fair_value(0.22, [2.5, 1.6], [0.8, 0.6])
    ev = evaluate_market_ev(cfg, market, "Yes", snap, post, "microstructure", [2.5, 1.6], [0.8, 0.6],
                            resolution_risk=3)
    assert ev and ev.record.signal_type == "high_conviction" and ev.record.expected_roi > 1.0
    assessment = GrokMarketAssessment(
        condition_id=market.condition_id, summary="s", catalyst_summary=f"Catalyst: {evil}",
        resolution_risk_score=3, resolution_trap_warning="Rules say 'ET' & close <midnight>",
        citations=["https://example.com/a?x=1&y=2", "javascript:bad()"],
        evidence_signals=[EvidenceSignal(claim=evil, source_handle_or_domain="@<handle>&co",
                                         source_tier="tier1_official", direction="YES",
                                         likelihood_ratio=2.5, confidence_weight=0.8)])
    card = format_ev_card(ev, market, snap, assessment, cfg)
    assert "<script>" not in card and "HIGH-CONVICTION" in card and "javascript" not in card
    assert set(t for t, _ in _open_tag_stack(card)) == set()  # all tags balanced
    print(f"EV card: {len(card)} chars, ROI {ev.record.expected_roi:+.2%}")

    # --- arb card
    legs = [ArbLeg("polymarket", "Lakers", "l", 0.405, 1 / 0.405), ArbLeg("pinnacle", "Boston Celtics", "e2", 1 / 1.8, 1.8)]
    implied = sum(l.price for l in legs)
    stakes = [500 * l.price / implied for l in legs]
    for l, st in zip(legs, stakes):
        l.stake_usd, l.payout_usd = st, 500 / implied
    arb = ArbOpportunity("cross_venue", "0xnba:Lakers", ["0xnba"], legs, implied, 1 / implied - 1, 500,
                         500 / implied - 500, "Execute the book leg first & confirm <fills>.")
    m_nba = MarketRecord("0xnba", "Lakers vs. Celtics", event_slug="nba-lal-bos", outcomes=["Lakers", "Celtics"],
                         token_ids=["l", "c"])
    arb_card = format_arb_card(arb, {"0xnba": m_nba}, cfg)
    assert "Cross-venue" in arb_card and "&lt;fills&gt;" in arb_card and not _open_tag_stack(arb_card)

    # --- heartbeat
    hb = format_heartbeat_card({"markets_total": 614, "http": {"requests": 3}}, {"alert_history": 2},
                               {"ev": {"n": 4, "avg_clv": 0.031, "beat_rate": 0.75}}, {"calls": 12})
    assert "heartbeat" in hb and not _open_tag_stack(hb)

    # --- chunking a >5000-char message with tags spanning boundaries
    big = "<b>" + "\n".join(f"Line {i}: {escape_html(evil)}" for i in range(120)) + "</b>\n\n" + card
    chunks = chunk_html_message(big)
    assert len(big) > 5000 and len(chunks) >= 2
    assert all(len(c) <= DEFAULT_CHUNK_LEN for c in chunks) and all(not _open_tag_stack(c) for c in chunks)
    assert strip_html("".join(chunks)).count("Line ") == 120
    print(f"Chunking: {len(big)} chars -> {len(chunks)} chunks {[len(c) for c in chunks]}")

    # --- dedup with temp DB (dry-run)
    path = Path("/tmp/pm_tg_selftest.sqlite3")
    if path.exists():
        path.unlink()
    async with DatabaseManager(path) as db:
        notifier = TelegramNotifier(cfg, db)
        assert notifier.dry_run
        assert await notifier.notify_ev(ev, market, snap, assessment) == "sent"
        assert await notifier.notify_ev(ev, market, snap, assessment) == "suppressed"
        ev.record.expected_roi += cfg.strategy.alert_roi_delta
        assert await notifier.notify_ev(ev, market, snap, assessment) == "sent"
        assert await notifier.notify_arb(arb, {"0xnba": m_nba}) == "sent"
        assert await notifier.notify_arb(arb, {"0xnba": m_nba}) == "suppressed"
        assert await notifier.send_heartbeat({"cycle": 1}, {"calls": 0})
        print(f"Dedup: stats={notifier.stats} alert rows={(await db.health_stats())['alert_history']}")

    # --- mock Telegram server: 429 -> HTML 400 -> plain text 200, then a 500 -> 200
    calls: List[Dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        calls.append(body)
        n = len(calls)
        if n == 1:
            return web.json_response({"ok": False, "error_code": 429, "parameters": {"retry_after": 0.2}}, status=429)
        if n == 2:
            return web.json_response({"ok": False, "description": "Bad Request: can't parse entities"}, status=400)
        if n == 4:
            return web.json_response({"ok": False, "description": "Internal"}, status=500)
        return web.json_response({"ok": True, "result": {"message_id": n}})

    app = web.Application()
    app.router.add_post("/bot{token}/sendMessage", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8765)
    await site.start()
    try:
        live_cfg = replace(cfg, endpoints=replace(cfg.endpoints, telegram_base_url="http://127.0.0.1:8765"),
                           runtime=replace(cfg.runtime, dry_run=False),
                           keys=replace(cfg.keys, telegram_bot_token="TEST", telegram_chat_id="1"),
                           rate_limits=replace(cfg.rate_limits, telegram_rps=50.0, backoff_base_s=0.01))
        async with TelegramNotifier(live_cfg) as tn:
            assert await tn.send_message(card)
            assert "parse_mode" not in calls[2] and "<b>" not in calls[2]["text"]
            assert await tn.send_message(arb_card)
            print(f"Mock Telegram: {len(calls)} HTTP calls, stats={tn.stats}")
    finally:
        await runner.cleanup()

    # --- optional live test
    if os.getenv("RUN_LIVE_TELEGRAM_TEST") == "1":
        if cfg.keys.telegram_bot_token and cfg.keys.telegram_chat_id:
            async with TelegramNotifier(replace(cfg, runtime=replace(cfg.runtime, dry_run=False))) as tn:
                print("LIVE Telegram:", await tn.send_message(hb))
        else:
            print("Live Telegram test skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
    print("Self-test passed.")


if __name__ == "__main__":
    asyncio.run(_self_test())
