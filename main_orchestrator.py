"""
main_orchestrator.py
====================
Module 6 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Wires Modules 1-5 + Module 7 (100,000-Unit Paper Trading Engine) into an async pipeline:
    --mode single-scan     one full cycle (cron / GitHub Actions)
    --mode continuous      loop every SCAN_INTERVAL_S with graceful SIGINT/SIGTERM shutdown
    --mode paper-report    send an immediate 4-hour 100k-unit paper bankroll & bets report
    --mode backtest-audit  CLV stamping and signal-performance report from SQLite

Pipeline per cycle:
    1 ingest -> 2 closing lines & paper MTM/settle -> 3 arbitrage (+ paper arb execution)
    -> 4 Grok pre-filter (+ sharp prior-only) -> 5 Grok + Bayesian evaluation
    -> 6 portfolio Kelly + persist + paper bet execution -> 7 alerts -> 8 4h unit report & heartbeat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config_and_database import AppConfig, DatabaseManager, MarketRecord, OrderbookSnapshot, load_config
from async_market_ingestor import AsyncHTTPClient, MarketIngestor, MatchedPair
from quant_math_engine import (
    ArbOpportunity,
    EVEvaluation,
    allocate_portfolio_kelly,
    hours_to_expiry,
    scan_arbitrage_opportunities,
)
from grok_intelligence_agent import (
    GrokCandidate,
    GrokIntelligenceClient,
    GrokMarketAssessment,
    _binary_priors,
    evaluate_candidates_with_grok,
    prefilter_markets_for_grok,
)
from telegram_notifier import TelegramNotifier, escape_html
from paper_trading_engine import DEFAULT_INITIAL_UNITS, REPORT_INTERVAL_S, PaperTradingEngine

logger = logging.getLogger("pm_engine.orchestrator")

DEFAULT_SPORTS = "basketball_nba,soccer_epl,icehockey_nhl,americanfootball_nfl"


# =============================================================================
# Logging
# =============================================================================

def setup_logging(cfg: AppConfig) -> None:
    cfg.runtime.log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.runtime.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)
    file_handler = RotatingFileHandler(cfg.runtime.log_dir / "pm_engine.log", maxBytes=10 * 1024 * 1024,
                                       backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("aiohttp.access", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
# Orchestrator
# =============================================================================

class PredictionMarketOrchestrator:
    def __init__(self, cfg: AppConfig, *, http: Optional[AsyncHTTPClient] = None,
                 grok: Optional[GrokIntelligenceClient] = None, min_roi: Optional[float] = None,
                 force_heartbeat: bool = False, force_paper_report: bool = False,
                 initial_units: float = DEFAULT_INITIAL_UNITS,
                 report_interval_s: int = REPORT_INTERVAL_S) -> None:
        self.cfg = cfg
        self.min_roi = min_roi
        self.force_heartbeat = force_heartbeat
        self.force_paper_report = force_paper_report
        self.db = DatabaseManager(cfg.runtime.db_path)
        self.http = http or AsyncHTTPClient(cfg)
        self.ingestor = MarketIngestor(cfg, self.http, self.db)
        self.grok = grok or GrokIntelligenceClient(cfg)
        self.prior_only = GrokIntelligenceClient(replace(cfg, keys=replace(cfg.keys, xai_api_key="")))
        self.notifier = TelegramNotifier(cfg, self.db)
        self.paper = PaperTradingEngine(self.db, initial_units=initial_units, report_interval_s=report_interval_s)
        self.stop_event = asyncio.Event()
        self._last_heartbeat = time.time()
        self.cycle_count = 0
        self.last_summary: Dict[str, Any] = {}

    async def __aenter__(self) -> "PredictionMarketOrchestrator":
        await self.db.connect()
        await self.paper.initialize()
        await self.http.open()
        snap = await self.paper.get_portfolio_snapshot()
        logger.info("Orchestrator started (dry_run=%s, grok=%s, paper_equity=%.2f units)",
                    self.cfg.runtime.dry_run, self.grok.enabled, snap.total_equity_units)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        for closer in (self.notifier.close, self.grok.close, self.prior_only.close, self.http.close, self.db.close):
            try:
                await closer()
            except Exception as exc:  # shutdown must never raise
                logger.warning("Error during shutdown: %r", exc)
        logger.info("Orchestrator stopped after %d cycles", self.cycle_count)

    # ----------------------------------------------------------- closing lines
    async def update_closing_lines(self, markets: Sequence[MarketRecord],
                                   snapshots_by_token: Mapping[str, OrderbookSnapshot]) -> int:
        stamped = 0
        for m in markets:
            hrs = hours_to_expiry(m.end_date_iso)
            for tok in m.token_ids:
                snap = snapshots_by_token.get(tok)
                if snap is None:
                    continue
                expiring = hrs is not None and hrs <= 1.0
                decided = (snap.best_bid is not None and snap.best_bid >= 0.98) or \
                          (snap.best_ask is not None and snap.best_ask <= 0.02)
                if not (expiring or decided):
                    continue
                price = snap.mid if snap.mid is not None else (snap.best_bid if snap.best_bid is not None
                                                               else snap.best_ask)
                if price is not None:
                    stamped += await self.db.record_closing_line(tok, float(price))
        if stamped:
            logger.info("Stamped closing lines on %d fair-value rows", stamped)
        return stamped

    # ------------------------------------------------------- prior-only adds
    def _sharp_prior_candidates(self, pairs: Sequence[MatchedPair], snaps: Mapping[str, OrderbookSnapshot],
                                exclude: set) -> List[GrokCandidate]:
        out: List[GrokCandidate] = []
        for pair in pairs:
            m = pair.market
            if m.condition_id in exclude or len(m.outcomes) != 2:
                continue
            by_outcome = {o: snaps[t] for o, t in zip(m.outcomes, m.token_ids) if t in snaps}
            if len(by_outcome) != 2:
                continue
            priors, source = _binary_priors(m, by_outcome, pair, None, self.cfg)
            if priors is None:
                continue
            div = max(abs(priors[o] - s.best_ask) for o, s in by_outcome.items() if s.best_ask is not None)
            out.append(GrokCandidate(m, by_outcome, priors, source, pair, None, hours_to_expiry(m.end_date_iso),
                                     div, 0.0, ["sharp-matched (prior only, no Grok cost)"]))
        return out

    # ------------------------------------------------------------- one cycle
    async def run_single_cycle(self, sports: Sequence[str], max_pages: int = 200) -> Dict[str, Any]:
        t0 = time.monotonic()
        self.cycle_count += 1
        summary: Dict[str, Any] = {"cycle": self.cycle_count,
                                   "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

        # 1. Ingest
        ingest = await self.ingestor.run_ingestion_cycle(sports=sports, max_pages=max_pages)
        markets: List[MarketRecord] = ingest["markets"]
        snaps: Dict[str, OrderbookSnapshot] = {s.token_id: s for s in ingest["snapshots"]}
        pairs: List[MatchedPair] = ingest["pairs"]
        markets_by_cid = {m.condition_id: m for m in markets}
        summary.update({k: ingest["summary"][k] for k in
                        ("markets_total", "markets_tradable", "snapshots", "sharp_events", "matched_pairs")})

        # 2. Closing lines & Mark-to-Market / Settlement of Open Paper Positions
        summary["closing_lines_stamped"] = await self.update_closing_lines(markets, snaps)
        mtm_stats = await self.paper.mark_and_settle_positions(markets_by_cid, snaps)
        summary["paper_marked"] = mtm_stats["marked"]
        summary["paper_settled"] = mtm_stats["settled"]

        # 3. Arbitrage (+ Paper Arb Execution)
        event_counts: Dict[str, int] = {}
        for m in ingest.get("all_markets", markets):
            if m.event_slug:
                event_counts[m.event_slug] = event_counts.get(m.event_slug, 0) + 1
        arbs: List[ArbOpportunity] = scan_arbitrage_opportunities(self.cfg, markets, snaps, pairs,
                                                                  event_market_counts=event_counts)
        paper_arbs = await self.paper.execute_arb_opportunities(self.cfg, arbs, markets_by_cid)
        arb_roi_by_token: Dict[str, float] = {}
        arb_sent = 0
        for arb in arbs:
            for leg in arb.legs:
                if leg.venue == "polymarket" and arb.kind == "cross_venue":
                    arb_roi_by_token[leg.ref] = max(arb.roi, arb_roi_by_token.get(leg.ref, 0.0))
            if await self.notifier.notify_arb(arb, markets_by_cid) == "sent":
                arb_sent += 1
        summary.update({"arbs_found": len(arbs), "arb_alerts": arb_sent, "paper_arbs_opened": len(paper_arbs)})

        # 4. Candidates: Grok-worthy + sharp-matched prior-only
        grok_cands = await prefilter_markets_for_grok(self.cfg, markets, snaps, pairs, self.db)
        prior_cands = self._sharp_prior_candidates(pairs, snaps, {c.market.condition_id for c in grok_cands})
        summary.update({"grok_candidates": len(grok_cands), "prior_only_candidates": len(prior_cands)})

        # 5. Evaluate
        results: List[Tuple[EVEvaluation, GrokMarketAssessment]] = []
        results += await evaluate_candidates_with_grok(self.cfg, grok_cands, self.grok, arb_roi_by_token)
        results += await evaluate_candidates_with_grok(self.cfg, prior_cands, self.prior_only, arb_roi_by_token)
        assessments = {id(ev): a for ev, a in results}
        evaluations = [ev for ev, _ in results]

        # 6. Portfolio Kelly, persist, and Execute Paper Trades from 100k-unit account
        actionable = allocate_portfolio_kelly(self.cfg, evaluations)
        if self.min_roi is not None:
            filtered_actionable = [
                ev for ev in actionable
                if ev.record.signal_type == "arb" or ev.record.expected_roi >= self.min_roi
            ]
        else:
            filtered_actionable = actionable

        for ev in evaluations:
            await self.db.insert_fair_value(ev.record)

        paper_ev_trades = await self.paper.execute_ev_evaluations(self.cfg, filtered_actionable, markets_by_cid)
        portfolio_snap = await self.paper.get_portfolio_snapshot()
        summary.update({
            "evaluations": len(evaluations),
            "actionable": len(filtered_actionable),
            "total_exposure_pct": round(sum(e.record.kelly_stake_pct for e in filtered_actionable), 4),
            "paper_ev_opened": len(paper_ev_trades),
            "paper_equity_units": round(portfolio_snap.total_equity_units, 2),
            "paper_cash_units": round(portfolio_snap.cash_units, 2),
            "paper_open_positions": portfolio_snap.open_count,
        })

        # 7. Alerts
        ev_sent = 0
        for ev in filtered_actionable:
            r = ev.record
            market = markets_by_cid.get(r.condition_id)
            if market is None:
                continue
            status = await self.notifier.notify_ev(ev, market, snaps.get(r.token_id), assessments.get(id(ev)))
            ev_sent += status == "sent"
        summary["ev_alerts"] = ev_sent

        # 8. 4-Hour Paper Portfolio Report & Heartbeat
        if await self.paper.should_send_report(force=self.force_paper_report):
            grok_note = await self.paper.generate_grok_portfolio_note(self.grok)
            report_msg = await self.paper.format_4h_report(grok_note=grok_note)
            if await self.notifier.send_message(report_msg):
                await self.paper.mark_report_sent()
                summary["paper_4h_report_sent"] = True
            self.force_paper_report = False

        summary["elapsed_s"] = round(time.monotonic() - t0, 2)
        if self.force_heartbeat or time.time() - self._last_heartbeat >= self.cfg.runtime.heartbeat_interval_s:
            summary["pruned_snapshots"] = await self.db.prune_snapshots(self.cfg.runtime.snapshot_retention_days)
            await self.notifier.send_heartbeat(summary, self.grok.stats)
            self._last_heartbeat = time.time()
            self.force_heartbeat = False
        self.last_summary = summary
        logger.info("Cycle %d complete: %s", self.cycle_count, json.dumps(summary, default=str))
        return summary

    # ------------------------------------------------------------ continuous
    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop, sig.name)
            except (NotImplementedError, RuntimeError):  # e.g. Windows / non-main thread
                signal.signal(sig, lambda *_a, n=sig.name: loop.call_soon_threadsafe(self._request_stop, n))

    def _request_stop(self, name: str) -> None:
        if not self.stop_event.is_set():
            logger.warning("Received %s: finishing current step and shutting down", name)
            self.stop_event.set()

    async def run_continuous(self, sports: Sequence[str], max_pages: int = 200,
                             max_cycles: Optional[int] = None) -> None:
        self._install_signal_handlers()
        cycles = 0
        while not self.stop_event.is_set():
            try:
                await self.run_single_cycle(sports, max_pages)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Cycle failed: %r", exc)
                await self.notifier.send_message(f"⚠️ <b>Cycle error</b>\n<code>{escape_html(repr(exc)[:800])}</code>")
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.cfg.runtime.scan_interval_s)
            except asyncio.TimeoutError:
                logger.debug("Sleep interval elapsed; starting next cycle")
        logger.info("Continuous loop exited cleanly after %d cycles", cycles)

    # ---------------------------------------------------------- backtest
    async def _stamp_closing_from_snapshots(self) -> int:
        """Stamp CLV for open rows whose market has expired, deactivated, or is effectively decided."""
        query = """
            SELECT DISTINCT f.token_id, m.end_date_iso, m.active
            FROM fair_value_models f LEFT JOIN markets_catalog m ON m.condition_id = f.condition_id
            WHERE f.clv IS NULL
        """
        async with self.db.conn.execute(query) as cur:
            rows = await cur.fetchall()
        stamped = 0
        for row in rows:
            snap = await self.db.latest_snapshot(row["token_id"])
            if not snap:
                continue
            hrs = hours_to_expiry(row["end_date_iso"])
            decided = (snap["best_bid"] or 0) >= 0.98 or (snap["best_ask"] if snap["best_ask"] is not None else 1) <= 0.02
            if (hrs is not None and hrs <= 1.0) or row["active"] == 0 or decided:
                price = snap["mid"] if snap["mid"] is not None else (snap["best_bid"] or snap["best_ask"])
                if price is not None:
                    stamped += await self.db.record_closing_line(row["token_id"], float(price))
        return stamped

    async def run_backtest_audit(self, send_telegram: bool = True) -> Dict[str, Any]:
        stamped = await self._stamp_closing_from_snapshots()
        health = await self.db.health_stats()
        clv = await self.db.clv_summary()
        paper_snap = await self.paper.get_portfolio_snapshot()
        async with self.db.conn.execute("""
            SELECT signal_type, COUNT(*) AS n, AVG(expected_roi) AS avg_roi, AVG(edge_pct) AS avg_edge,
                   AVG(kelly_stake_pct) AS avg_kelly, AVG(resolution_risk) AS avg_risk,
                   SUM(CASE WHEN clv IS NOT NULL THEN 1 ELSE 0 END) AS closed
            FROM fair_value_models GROUP BY signal_type ORDER BY n DESC
        """) as cur:
            breakdown = [dict(r) for r in await cur.fetchall()]
        alerts = await self.db.recent_alerts(10)

        lines = [f"{'signal':<16}{'n':>6}{'closed':>8}{'avg ROI':>10}{'avg edge':>10}{'avg kelly':>11}"
                 f"{'avg CLV':>10}{'beat%':>8}"]
        for b in breakdown:
            c = clv.get(b["signal_type"], {})
            lines.append(f"{b['signal_type']:<16}{b['n']:>6}{b['closed']:>8}{(b['avg_roi'] or 0):>+10.2%}"
                         f"{(b['avg_edge'] or 0):>+10.3f}{(b['avg_kelly'] or 0):>11.4f}"
                         f"{(c.get('avg_clv') or 0):>+10.4f}{(c.get('beat_rate') or 0):>8.1%}")
        table = "\n".join(lines)
        report = {"stamped": stamped, "health": health, "clv": clv, "breakdown": breakdown,
                  "recent_alerts": alerts, "table": table, "paper_equity_units": paper_snap.total_equity_units}
        print("\n=== BACKTEST AUDIT ===")
        print(f"Closing lines newly stamped: {stamped}")
        print(f"Paper Account Equity: {paper_snap.total_equity_units:,.2f} u "
              f"(Cash: {paper_snap.cash_units:,.2f} u, Open Bets: {paper_snap.open_count})")
        print("Tables:", json.dumps(health))
        print(table)
        if alerts:
            print("\nRecent alerts:")
            for a in alerts:
                ts = datetime.fromtimestamp(a["last_sent_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                print(f"  {ts}  {a['signal_type']:<20} ROI {a['last_roi']:+.2%}  x{a['alert_count']}  "
                      f"{a['condition_id'][:24]}")
        if send_telegram:
            await self.notifier.send_message(
                f"📊 <b>Backtest audit</b> · stamped {stamped}\n<pre>{escape_html(table)}</pre>")
            await self.notifier.send_message(await self.paper.format_4h_report())
        return report


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket Fair-Value (+EV), Arbitrage & 100k-Unit Paper Trading Engine")
    p.add_argument("--mode", choices=("single-scan", "continuous", "paper-report", "backtest-audit"),
                   default="single-scan")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--sports", default=DEFAULT_SPORTS, help="Comma-separated The-Odds-API sport keys")
    p.add_argument("--max-pages", type=int, default=200, help="Max Gamma pages of 100 events")
    p.add_argument("--max-cycles", type=int, default=None, help="Stop continuous mode after N cycles")
    p.add_argument("--min-roi", type=float, default=None, help="Override EV ROI threshold, e.g. 1.0 for +100%%")
    p.add_argument("--initial-units", type=float, default=float(os.getenv("PAPER_INITIAL_UNITS", "100000.0")),
                   help="Starting virtual units for paper account (default 100,000)")
    p.add_argument("--report-interval-hours", type=float, default=float(os.getenv("PAPER_REPORT_HOURS", "4.0")),
                   help="Hours between automatic paper bankroll reports (default 4.0)")
    p.add_argument("--live", action="store_true", help="Send real Telegram alerts (sets DRY_RUN=false)")
    p.add_argument("--send-heartbeat", action="store_true", help="Send a heartbeat after this cycle")
    p.add_argument("--send-paper-report", action="store_true", help="Force sending the 4h paper bankroll report")
    p.add_argument("--self-test", action="store_true", help="Run the offline end-to-end integration test")
    return p.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    cfg = load_config(args.env_file)
    if args.live:
        cfg = replace(cfg, runtime=replace(cfg.runtime, dry_run=False))
    if args.min_roi is not None:
        cfg = replace(cfg, strategy=replace(cfg.strategy, ev_roi_threshold=args.min_roi))
    # Synchronize strategy bankroll with paper initial units so clip scaling aligns cleanly
    cfg = replace(cfg, strategy=replace(cfg.strategy, bankroll_usd=args.initial_units))
    setup_logging(cfg)
    logger.info("Config: %s", json.dumps(cfg.summary(), default=str))
    if args.live:
        cfg.validate(require_keys=("telegram_bot_token", "telegram_chat_id"))
    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    report_interval_s = int(max(60.0, args.report_interval_hours * 3600.0))
    async with PredictionMarketOrchestrator(
        cfg,
        min_roi=args.min_roi,
        force_heartbeat=args.send_heartbeat,
        force_paper_report=args.send_paper_report,
        initial_units=args.initial_units,
        report_interval_s=report_interval_s,
    ) as orch:
        if args.mode == "single-scan":
            await orch.run_single_cycle(sports, args.max_pages)
        elif args.mode == "continuous":
            await orch.run_continuous(sports, args.max_pages, args.max_cycles)
        elif args.mode == "paper-report":
            grok_note = await orch.paper.generate_grok_portfolio_note(orch.grok)
            msg = await orch.paper.format_4h_report(grok_note=grok_note)
            await orch.notifier.send_message(msg)
            await orch.paper.mark_report_sent()
        else:
            await orch.run_backtest_audit()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        asyncio.run(_self_test())
        return 0
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


# =============================================================================
# End-to-end integration self-test (fully offline)
# =============================================================================

class _MockHTTP(AsyncHTTPClient):
    """Routes Gamma / CLOB / Odds API calls to synthetic fixtures."""

    def __init__(self, cfg: AppConfig, events: List[Dict[str, Any]], books: Dict[str, Dict[str, Any]],
                 odds: List[Dict[str, Any]]) -> None:
        super().__init__(cfg)
        self.events, self.books, self.odds = events, books, odds
        self.calls: List[str] = []

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_json(self, url: str, bucket: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        self.calls.append(url)
        if url.endswith("/events"):
            return self.events if int((params or {}).get("offset", 0)) == 0 else []
        if url.endswith("/book"):
            return self.books.get(str((params or {}).get("token_id")))
        if "/sports/" in url:
            return self.odds
        raise AssertionError(f"Unexpected GET {url}")

    async def post_json(self, url: str, bucket: str, body: Any) -> Any:
        self.calls.append(url)
        assert url.endswith("/books")
        return [self.books[b["token_id"]] for b in body if b["token_id"] in self.books]


class _MockGrok(GrokIntelligenceClient):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg, use_sdk=False)
        self.api_key = "mock"

    async def _call_model(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[str], str]:
        now = datetime.now(timezone.utc).isoformat()
        if "Fed" in user_prompt:
            payload = {
                "summary": "FOMC leaning to cut", "catalyst_summary": "Chair confirms December cut is on the table",
                "resolution_risk_score": 2, "resolution_trap_warning": "",
                "evidence_signals": [
                    {"claim": "Chair: cut appropriate in December", "source_handle_or_domain": "@federalreserve",
                     "source_tier": "tier1_official", "published_at_iso": now, "direction": "YES",
                     "likelihood_ratio": 4.0, "confidence_weight": 0.9},
                    {"claim": "Two governors back a cut", "source_handle_or_domain": "@NickTimiraos",
                     "source_tier": "tier2_verified_reporter", "published_at_iso": now, "direction": "YES",
                     "likelihood_ratio": 3.0, "confidence_weight": 0.8}]}
        else:
            payload = {"summary": "No material news", "resolution_risk_score": 3,
                       "resolution_trap_warning": "", "evidence_signals": []}
        return json.dumps(payload), ["https://www.federalreserve.gov/"], "mock"


def _book(tok: str, bid: float, ask: float, size: float = 20_000) -> Dict[str, Any]:
    return {"asset_id": tok, "bids": [{"price": f"{bid:.3f}", "size": str(size)}],
            "asks": [{"price": f"{ask + 0.01:.3f}", "size": str(size)}, {"price": f"{ask:.3f}", "size": str(size)}]}


async def _self_test() -> None:
    import tempfile
    from datetime import timedelta
    tmp = Path(tempfile.mkdtemp(prefix="pm_e2e_"))
    base = load_config(env_file=None)
    cfg = replace(base, runtime=replace(base.runtime, db_path=tmp / "e2e.sqlite3", log_dir=tmp / "logs",
                                        dry_run=True, log_level="WARNING"),
                  strategy=replace(base.strategy, bankroll_usd=100_000.0),
                  keys=replace(base.keys, odds_api_key="mock-odds-key", xai_api_key="mock"))
    setup_logging(cfg)
    now = datetime.now(timezone.utc)

    def gm(cid: str, q: str, outcomes: List[str], toks: List[str], end: datetime) -> Dict[str, Any]:
        return {"conditionId": cid, "question": q, "slug": cid, "outcomes": json.dumps(outcomes),
                "clobTokenIds": json.dumps(toks), "endDate": end.isoformat(), "volume24hr": 50_000,
                "volumeNum": 900_000, "liquidityNum": 40_000, "active": True, "enableOrderBook": True,
                "description": f"Resolves per official source. {q}"}

    events = [
        {"slug": "arb-event", "markets": [gm("0xarb", "Synthetic binary arb?", ["Yes", "No"], ["ay", "an"],
                                              now + timedelta(days=10))]},
        {"slug": "nba-lal-bos", "markets": [gm("0xnba", "Lakers vs. Celtics", ["Lakers", "Celtics"], ["l", "c"],
                                                now + timedelta(hours=26))]},
        {"slug": "fed-dec-cut", "markets": [gm("0xfed", "Will the Fed cut rates in December?", ["Yes", "No"],
                                                ["fy", "fn"], now + timedelta(minutes=40))]},
        {"slug": "dead", "markets": [gm("0xdead", "Dead market?", ["Yes", "No"], ["dy", "dn"],
                                         now + timedelta(days=90))]},
    ]
    books = {"ay": _book("ay", .45, .47), "an": _book("an", .48, .50),
             "l": _book("l", .48, .50), "c": _book("c", .48, .52),
             "fy": _book("fy", .16, .18), "fn": _book("fn", .80, .84),
             "dy": {"asset_id": "dy", "bids": [], "asks": []}}
    odds = [{"id": "e1", "sport_key": "basketball_nba", "commence_time": (now + timedelta(hours=24)).isoformat(),
             "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics",
             "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                 {"name": "Los Angeles Lakers", "price": 1.60}, {"name": "Boston Celtics", "price": 2.45}]}]}]}]

    http = _MockHTTP(cfg, events, books, odds)
    async with PredictionMarketOrchestrator(cfg, http=http, grok=_MockGrok(cfg),
                                            force_heartbeat=True, force_paper_report=True,
                                            initial_units=100_000.0) as orch:
        summary = await orch.run_single_cycle(["basketball_nba"], max_pages=2)
        print("Cycle summary:", json.dumps(summary, default=str))
        health = await orch.db.health_stats()
        print("DB:", health)
        assert all(health[t] > 0 for t in ("markets_catalog", "orderbook_snapshots", "fair_value_models", "alert_history"))
        assert summary["matched_pairs"] == 1 and summary["arbs_found"] >= 2
        assert summary["paper_ev_opened"] >= 1 and summary["paper_4h_report_sent"] is True

        sent = orch.notifier.sent_log
        assert any("4-HOUR PAPER BANKROLL" in m for m in sent), "4h paper report missing"
        assert any("100,000.00 u" in m for m in sent), "100k unit starting balance missing"

        report = await orch.run_backtest_audit(send_telegram=True)
        assert report["stamped"] >= 1, "Fed market (40 min to expiry) should be stamped"
        print("Paper Snapshot after cycle:", await orch.paper.get_portfolio_snapshot())

    print("End-to-end self-test (with 100k-unit paper trading & 4h reporting) passed.")


if __name__ == "__main__":
    sys.exit(main())
