"""
paper_trading_engine.py
=======================
Module 7: Automated Paper Trading & 100,000-Unit Bankroll Ledger (with Grok AI Desk Execution).

Responsibilities
----------------
1. Maintain a persistent SQLite ledger (`paper_account` and `paper_positions`) starting with
   100,000.00 units of virtual capital.
2. Send candidate alerts to the Grok AI Portfolio Manager (when `XAI_API_KEY` is configured)
   or deterministic Kelly engine to execute virtual bets with fake capital, sizing dynamically
   from CURRENT total equity.
3. Mark open positions to market (MTM) on every scan using live orderbook bids/mids and
   automatically settle winning/losing positions when markets resolve (bid >= 0.98 or ask <= 0.02
   or past expiry).
4. Generate and send a comprehensive 4-hour Unit & Portfolio Report to Telegram (including
   Grok's AI Desk Commentary on open and settled bets).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiosqlite

from config_and_database import AppConfig, DatabaseManager, MarketRecord, OrderbookSnapshot, load_config
from quant_math_engine import ArbOpportunity, EVEvaluation, hours_to_expiry
from grok_intelligence_agent import GrokIntelligenceClient, GrokMarketAssessment
from telegram_notifier import TelegramNotifier, escape_html, fmt_pct, fmt_price

logger = logging.getLogger("pm_engine.paper")

DEFAULT_INITIAL_UNITS = 100_000.0
REPORT_INTERVAL_S = 4 * 3600  # 4 hours


PAPER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_account (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    initial_units       REAL NOT NULL DEFAULT 100000.0,
    cash_units          REAL NOT NULL DEFAULT 100000.0,
    realized_pnl_units  REAL NOT NULL DEFAULT 0.0,
    total_trades        INTEGER NOT NULL DEFAULT 0,
    winning_trades      INTEGER NOT NULL DEFAULT 0,
    losing_trades       INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_report_at      REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    position_key        TEXT NOT NULL,
    condition_id        TEXT NOT NULL,
    token_id            TEXT NOT NULL,
    question            TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    signal_type         TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    shares              REAL NOT NULL,
    stake_units         REAL NOT NULL,
    expected_roi        REAL NOT NULL,
    posterior_prob      REAL NOT NULL,
    mark_price          REAL NOT NULL,
    unrealized_pnl      REAL NOT NULL DEFAULT 0.0,
    status              TEXT NOT NULL DEFAULT 'OPEN', -- OPEN | WON | LOST | CLOSED
    exit_price          REAL,
    payout_units        REAL,
    realized_pnl        REAL,
    opened_at           REAL NOT NULL,
    closed_at           REAL,
    notes               TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_paper_pos_status ON paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_paper_pos_token ON paper_positions(token_id, status);
"""


@dataclass
class PaperPortfolioSnapshot:
    initial_units: float
    cash_units: float
    open_stake_units: float
    open_mark_value_units: float
    unrealized_pnl_units: float
    realized_pnl_units: float
    total_equity_units: float
    net_pnl_units: float
    roi_pct: float
    total_trades: int
    open_count: int
    winning_trades: int
    losing_trades: int
    last_report_at: float


class PaperTradingEngine:
    """Manages the 100,000-unit virtual bankroll, trade execution, MTM & 4-hour reporting."""

    def __init__(self, db: DatabaseManager, initial_units: float = DEFAULT_INITIAL_UNITS,
                 report_interval_s: int = REPORT_INTERVAL_S) -> None:
        self.db = db
        self.initial_units = float(initial_units)
        self.report_interval_s = int(report_interval_s)

    async def initialize(self) -> None:
        async with self.db._write_lock:
            await self.db.conn.executescript(PAPER_SCHEMA_SQL)
            now = time.time()
            await self.db.conn.execute("""
                INSERT OR IGNORE INTO paper_account
                    (id, initial_units, cash_units, realized_pnl_units, total_trades,
                     winning_trades, losing_trades, created_at, updated_at, last_report_at)
                VALUES (1, ?, ?, 0.0, 0, 0, 0, ?, ?, 0.0)
            """, (self.initial_units, self.initial_units, now, now))
            await self.db.conn.commit()

    # ----------------------------------------------------------------- portfolio state
    async def get_portfolio_snapshot(self) -> PaperPortfolioSnapshot:
        async with self.db.conn.execute("SELECT * FROM paper_account WHERE id = 1") as cur:
            acct = await cur.fetchone()
        if acct is None:
            await self.initialize()
            return await self.get_portfolio_snapshot()

        async with self.db.conn.execute("""
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(stake_units), 0.0) AS staked,
                   COALESCE(SUM(shares * mark_price), 0.0) AS mark_val,
                   COALESCE(SUM(unrealized_pnl), 0.0) AS upnl
            FROM paper_positions WHERE status = 'OPEN'
        """) as cur:
            pos = await cur.fetchone()

        initial = float(acct["initial_units"])
        cash = float(acct["cash_units"])
        staked = float(pos["staked"] if pos else 0.0)
        mark_val = float(pos["mark_val"] if pos else 0.0)
        upnl = float(pos["upnl"] if pos else 0.0)
        rpnl = float(acct["realized_pnl_units"])
        equity = cash + mark_val
        net_pnl = equity - initial
        roi_pct = (net_pnl / initial) if initial > 0 else 0.0

        return PaperPortfolioSnapshot(
            initial_units=initial,
            cash_units=cash,
            open_stake_units=staked,
            open_mark_value_units=mark_val,
            unrealized_pnl_units=upnl,
            realized_pnl_units=rpnl,
            total_equity_units=equity,
            net_pnl_units=net_pnl,
            roi_pct=roi_pct,
            total_trades=int(acct["total_trades"]),
            open_count=int(pos["n"] if pos else 0),
            winning_trades=int(acct["winning_trades"]),
            losing_trades=int(acct["losing_trades"]),
            last_report_at=float(acct["last_report_at"]),
        )

    async def list_open_positions(self, limit: int = 25) -> List[Dict[str, Any]]:
        async with self.db.conn.execute(
            "SELECT * FROM paper_positions WHERE status = 'OPEN' ORDER BY opened_at DESC LIMIT ?",
            (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def list_recent_closed_positions(self, since_ts: float = 0.0, limit: int = 15) -> List[Dict[str, Any]]:
        async with self.db.conn.execute(
            "SELECT * FROM paper_positions WHERE status != 'OPEN' AND closed_at >= ? "
            "ORDER BY closed_at DESC LIMIT ?",
            (since_ts, limit)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # --------------------------------------------------------- execution of +EV & Arb bets
    async def execute_ev_evaluations(self, cfg: AppConfig, evaluations: Sequence[EVEvaluation],
                                     markets_by_cid: Mapping[str, MarketRecord]) -> List[Dict[str, Any]]:
        """
        Execute virtual bets for actionable EVEvaluations that do not already have an OPEN position.
        Sizes dynamically against current total equity (in units) while respecting available cash.
        """
        snap = await self.get_portfolio_snapshot()
        equity = snap.total_equity_units
        cash = snap.cash_units
        max_exposure = cfg.strategy.max_portfolio_exposure_pct * equity
        remaining_exposure = max(0.0, max_exposure - snap.open_stake_units)

        executed: List[Dict[str, Any]] = []
        now = time.time()

        for ev in evaluations:
            r = ev.record
            if r.signal_type == "pass" or r.kelly_stake_pct <= 0:
                continue
            pos_key = f"ev:{r.condition_id}:{r.token_id}"
            async with self.db.conn.execute(
                "SELECT id FROM paper_positions WHERE position_key = ? AND status = 'OPEN'", (pos_key,)
            ) as cur:
                if await cur.fetchone() is not None:
                    continue

            desired_units = r.kelly_stake_pct * equity
            if ev.clip_tier is not None and cfg.strategy.bankroll_usd > 0:
                unit_scale = equity / cfg.strategy.bankroll_usd
                desired_units = min(desired_units, ev.clip_tier * unit_scale)

            stake_units = min(desired_units, cash, remaining_exposure)
            if stake_units < 10.0:  # minimum 10 units per trade
                continue

            entry_price = float(r.entry_vwap or r.market_price)
            if not (0.0 < entry_price < 1.0):
                continue
            shares = stake_units / entry_price
            market = markets_by_cid.get(r.condition_id)
            question = market.question if market else r.condition_id

            async with self.db._write_lock:
                cur = await self.db.conn.execute("""
                    INSERT INTO paper_positions (
                        position_key, condition_id, token_id, question, outcome, signal_type,
                        entry_price, shares, stake_units, expected_roi, posterior_prob,
                        mark_price, unrealized_pnl, status, opened_at, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,0.0,'OPEN',?,?)
                """, (
                    pos_key, r.condition_id, r.token_id, question, r.outcome, r.signal_type,
                    entry_price, shares, stake_units, r.expected_roi, r.posterior_prob,
                    entry_price, now, r.rationale[:400]
                ))
                pos_id = int(cur.lastrowid or 0)
                cash -= stake_units
                remaining_exposure -= stake_units
                await self.db.conn.execute("""
                    UPDATE paper_account
                    SET cash_units = ?, total_trades = total_trades + 1, updated_at = ?
                    WHERE id = 1
                """, (cash, now))
                await self.db.conn.commit()

            executed.append({
                "id": pos_id, "condition_id": r.condition_id, "token_id": r.token_id,
                "question": question, "outcome": r.outcome, "signal_type": r.signal_type,
                "entry_price": entry_price, "shares": shares, "stake_units": stake_units,
                "expected_roi": r.expected_roi, "posterior_prob": r.posterior_prob,
            })
            logger.info("PAPER BET EXECUTED: %s [%s] %.2f units @ %.3f (ROI %+.2f%%)",
                        question[:50], r.outcome, stake_units, entry_price, r.expected_roi * 100)
        return executed

    async def execute_arb_opportunities(self, cfg: AppConfig, arbs: Sequence[ArbOpportunity],
                                        markets_by_cid: Mapping[str, MarketRecord]) -> List[Dict[str, Any]]:
        """Execute virtual multi-leg arbitrage packages when an arb has not yet been opened."""
        snap = await self.get_portfolio_snapshot()
        cash = snap.cash_units
        executed: List[Dict[str, Any]] = []
        now = time.time()

        for arb in arbs:
            pos_key = f"arb:{arb.kind}:{arb.key}"
            async with self.db.conn.execute(
                "SELECT id FROM paper_positions WHERE position_key = ? AND status = 'OPEN'", (pos_key,)
            ) as cur:
                if await cur.fetchone() is not None:
                    continue

            outlay_units = min(self.initial_units * 0.02, cash)
            if outlay_units < 50.0:
                continue

            cid = arb.condition_ids[0] if arb.condition_ids else arb.key
            market = markets_by_cid.get(cid)
            question = market.question if market else f"Arb {arb.key}"
            legs_desc = " + ".join(f"{l.venue}:{l.outcome}@{l.price:.3f}" for l in arb.legs)
            shares = outlay_units / arb.implied_sum

            async with self.db._write_lock:
                cur = await self.db.conn.execute("""
                    INSERT INTO paper_positions (
                        position_key, condition_id, token_id, question, outcome, signal_type,
                        entry_price, shares, stake_units, expected_roi, posterior_prob,
                        mark_price, unrealized_pnl, status, opened_at, notes
                    ) VALUES (?,?,?,?,?,'arb', ?,?,?,?,1.0, ?,?, 'OPEN', ?, ?)
                """, (
                    pos_key, cid, arb.legs[0].ref, question, legs_desc[:120],
                    arb.implied_sum, shares, outlay_units, arb.roi,
                    1.0, shares - outlay_units, now, arb.notes[:300]
                ))
                pos_id = int(cur.lastrowid or 0)
                cash -= outlay_units
                await self.db.conn.execute("""
                    UPDATE paper_account
                    SET cash_units = ?, total_trades = total_trades + 1, updated_at = ?
                    WHERE id = 1
                """, (cash, now))
                await self.db.conn.commit()

            executed.append({
                "id": pos_id, "condition_id": cid, "question": question,
                "outcome": legs_desc, "signal_type": "arb", "entry_price": arb.implied_sum,
                "shares": shares, "stake_units": outlay_units, "expected_roi": arb.roi,
            })
        return executed

    # --------------------------------------------------------- mark-to-market & settlement
    async def mark_and_settle_positions(self, markets_by_cid: Mapping[str, MarketRecord],
                                        snapshots_by_token: Mapping[str, OrderbookSnapshot]) -> Dict[str, int]:
        """Update mark_price and unrealized_pnl on open positions; settle resolved ones."""
        open_positions = await self.list_open_positions(limit=1000)
        marked = 0
        settled = 0
        now = time.time()

        for pos in open_positions:
            pid = pos["id"]
            tok = pos["token_id"]
            cid = pos["condition_id"]
            shares = float(pos["shares"])
            stake = float(pos["stake_units"])

            if pos["signal_type"] == "arb":
                market = markets_by_cid.get(cid)
                hrs = hours_to_expiry(market.end_date_iso) if market else None
                if hrs is not None and hrs <= 0.0:
                    payout = shares * 1.0
                    pnl = payout - stake
                    await self._settle_position(pid, "WON", 1.0, payout, pnl, now)
                    settled += 1
                continue

            snap = snapshots_by_token.get(tok)
            if snap is None:
                continue

            mark = snap.mid if snap.mid is not None else (
                snap.best_bid if snap.best_bid is not None else float(pos["mark_price"])
            )
            mark = float(mark)
            upnl = shares * mark - stake

            if snap.best_bid is not None and snap.best_bid >= 0.98:
                payout = shares * 1.0
                pnl = payout - stake
                await self._settle_position(pid, "WON", 1.0, payout, pnl, now)
                settled += 1
            elif snap.best_ask is not None and snap.best_ask <= 0.02:
                await self._settle_position(pid, "LOST", 0.0, 0.0, -stake, now)
                settled += 1
            else:
                async with self.db._write_lock:
                    await self.db.conn.execute("""
                        UPDATE paper_positions
                        SET mark_price = ?, unrealized_pnl = ?
                        WHERE id = ?
                    """, (mark, upnl, pid))
                    await self.db.conn.commit()
                marked += 1

        return {"marked": marked, "settled": settled}

    async def _settle_position(self, pos_id: int, status: str, exit_price: float,
                               payout_units: float, realized_pnl: float, now: float) -> None:
        win_inc = 1 if realized_pnl > 0 else 0
        loss_inc = 1 if realized_pnl <= 0 else 0
        async with self.db._write_lock:
            await self.db.conn.execute("""
                UPDATE paper_positions
                SET status = ?, mark_price = ?, exit_price = ?, payout_units = ?,
                    unrealized_pnl = 0.0, realized_pnl = ?, closed_at = ?
                WHERE id = ? AND status = 'OPEN'
            """, (status, exit_price, exit_price, payout_units, realized_pnl, now, pos_id))
            await self.db.conn.execute("""
                UPDATE paper_account
                SET cash_units = cash_units + ?,
                    realized_pnl_units = realized_pnl_units + ?,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    updated_at = ?
                WHERE id = 1
            """, (payout_units, realized_pnl, win_inc, loss_inc, now))
            await self.db.conn.commit()

    # --------------------------------------------------------- Grok Desk Commentary
    async def generate_grok_portfolio_note(self, grok: Optional[GrokIntelligenceClient]) -> str:
        """Ask Grok to write a concise 2-3 sentence quantitative desk update on the 100k-unit portfolio."""
        if grok is None or not grok.enabled:
            return ""
        snap = await self.get_portfolio_snapshot()
        open_pos = await self.list_open_positions(limit=8)
        closed_pos = await self.list_recent_closed_positions(since_ts=snap.last_report_at, limit=5)
        sys_prompt = (
            "You are the Head Quantitative Portfolio Manager for a 100,000-unit Prediction Market Fund. "
            "Write a sharp, 2-to-3 sentence desk update summarizing current unit equity, PnL drivers, "
            "and risk exposure across active positions. Plain text only, no markdown."
        )
        payload = {
            "total_equity_units": round(snap.total_equity_units, 2),
            "net_pnl_units": round(snap.net_pnl_units, 2),
            "roi_pct": round(snap.roi_pct * 100, 2),
            "open_positions": [
                {"question": p["question"], "outcome": p["outcome"], "stake_units": round(p["stake_units"], 1),
                 "entry": p["entry_price"], "mark": p["mark_price"], "upnl": round(p["unrealized_pnl"], 1)}
                for p in open_pos
            ],
            "recently_settled": [
                {"question": p["question"], "status": p["status"], "realized_pnl": round(p["realized_pnl"] or 0, 1)}
                for p in closed_pos
            ],
        }
        try:
            text, _cites, _tr = await grok._call_model(sys_prompt, json.dumps(payload))
            return text.strip()
        except Exception as exc:
            logger.debug("Grok portfolio note skipped: %r", exc)
            return ""

    # --------------------------------------------------------- 4-Hour Report Generation
    async def should_send_report(self, force: bool = False) -> bool:
        if force:
            return True
        snap = await self.get_portfolio_snapshot()
        return (time.time() - snap.last_report_at) >= self.report_interval_s

    async def mark_report_sent(self) -> None:
        async with self.db._write_lock:
            await self.db.conn.execute(
                "UPDATE paper_account SET last_report_at = ?, updated_at = ? WHERE id = 1",
                (time.time(), time.time())
            )
            await self.db.conn.commit()

    async def format_4h_report(self, grok_note: str = "") -> str:
        snap = await self.get_portfolio_snapshot()
        open_pos = await self.list_open_positions(limit=10)
        closed_pos = await self.list_recent_closed_positions(since_ts=snap.last_report_at, limit=10)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        win_total = snap.winning_trades + snap.losing_trades
        win_rate = (snap.winning_trades / win_total) if win_total > 0 else None

        lines: List[str] = [
            f"🏦 <b>4-HOUR PAPER BANKROLL &amp; BET REPORT</b> · {escape_html(now_str)}",
            "",
            "<b>💰 Account Summary (100,000-Unit Fund)</b>",
            f"  Starting Capital: <code>{snap.initial_units:,.2f} u</code>",
            f"  <b>Current Total Equity: <code>{snap.total_equity_units:,.2f} u</code></b> "
            f"(<b>{fmt_pct(snap.roi_pct, signed=True)}</b>)",
            f"  Available Cash: <code>{snap.cash_units:,.2f} u</code>",
            f"  Capital in Open Bets: <code>{snap.open_stake_units:,.2f} u</code> "
            f"(MTM value: <code>{snap.open_mark_value_units:,.2f} u</code>)",
            f"  Realized PnL: <code>{snap.realized_pnl_units:+,.2f} u</code> · "
            f"Unrealized PnL: <code>{snap.unrealized_pnl_units:+,.2f} u</code>",
            f"  Total Bets Executed: <code>{snap.total_trades}</code> "
            f"(Open: <code>{snap.open_count}</code> · Won: <code>{snap.winning_trades}</code> · "
            f"Lost: <code>{snap.losing_trades}</code>"
            + (f" · Win rate: <code>{fmt_pct(win_rate, 1)}</code>)" if win_rate is not None else ")"),
            "",
            f"<b>📂 Active Open Positions ({snap.open_count})</b>",
        ]
        if open_pos:
            for i, p in enumerate(open_pos, 1):
                upnl = float(p["unrealized_pnl"])
                lines.append(
                    f"{i}. <b>{escape_html(p['question'][:65])}</b>\n"
                    f"   🎯 <b>{escape_html(p['outcome'][:45])}</b> [{escape_html(p['signal_type'])}] · "
                    f"Stake <code>{p['stake_units']:,.1f} u</code> @ <code>{p['entry_price']:.3f}</code> "
                    f"→ Mark <code>{p['mark_price']:.3f}</code> "
                    f"(PnL <b><code>{upnl:+,.1f} u</code></b> | exp ROI {fmt_pct(p['expected_roi'], signed=True)})"
                )
        else:
            lines.append("  <i>No open positions right now.</i>")

        if closed_pos:
            lines += ["", "<b>✅ Settled Since Last Report</b>"]
            for p in closed_pos:
                rpnl = float(p["realized_pnl"] or 0.0)
                badge = "🟢 WON" if p["status"] == "WON" else "🔴 LOST"
                lines.append(
                    f"  {badge} <b>{escape_html(p['question'][:55])}</b> ({escape_html(p['outcome'][:30])}): "
                    f"Stake <code>{p['stake_units']:,.1f} u</code> → PnL <b><code>{rpnl:+,.1f} u</code></b>"
                )

        if grok_note:
            lines += ["", "<b>🤖 Grok AI Desk Note</b>", f"<i>{escape_html(grok_note[:600])}</i>"]

        return "\n".join(lines)


async def _self_test() -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(env_file=None)
    db_path = Path("/tmp/pm_paper_selftest.sqlite3")
    if db_path.exists():
        db_path.unlink()

    from quant_math_engine import compute_posterior_fair_value, evaluate_market_ev
    async with DatabaseManager(db_path) as db:
        paper = PaperTradingEngine(db, initial_units=100_000.0)
        await paper.initialize()
        s0 = await paper.get_portfolio_snapshot()
        assert s0.total_equity_units == 100_000.0 and s0.cash_units == 100_000.0

        m = MarketRecord("0xfed", "Will the Fed cut rates in December?", outcomes=["Yes", "No"], token_ids=["fy", "fn"])
        snap = OrderbookSnapshot("fy", "0xfed", 0.16, 0.18, 0.02, 0.17, 0.18, 0.185, 0.19, 0, .005, .01, 6000, 3000, 0.2)
        post = compute_posterior_fair_value(0.22, [3.0, 2.0], [0.9, 0.7])
        ev = evaluate_market_ev(cfg, m, "Yes", snap, post, "sharp:pinnacle", [3.0, 2.0], [0.9, 0.7], resolution_risk=2)
        assert ev is not None

        trades = await paper.execute_ev_evaluations(cfg, [ev], {"0xfed": m})
        assert len(trades) == 1
        assert len(await paper.execute_ev_evaluations(cfg, [ev], {"0xfed": m})) == 0

        snap_up = replace(snap, best_bid=0.34, best_ask=0.36, mid=0.35)
        await paper.mark_and_settle_positions({"0xfed": m}, {"fy": snap_up})
        s1 = await paper.get_portfolio_snapshot()
        assert s1.unrealized_pnl_units > 0 and s1.total_equity_units > 100_000.0

        report_html = await paper.format_4h_report(grok_note="Fed position in strong profit.")
        assert "100,000.00 u" in report_html and "Will the Fed cut" in report_html

        snap_won = replace(snap, best_bid=0.99, best_ask=1.00, mid=0.995)
        await paper.mark_and_settle_positions({"0xfed": m}, {"fy": snap_won})
        s2 = await paper.get_portfolio_snapshot()
        assert s2.open_count == 0 and s2.winning_trades == 1 and s2.realized_pnl_units > 0
        print(f"Paper Engine self-test passed: Final Equity = {s2.total_equity_units:,.2f} units "
              f"(PnL {s2.realized_pnl_units:+,.2f} u, ROI {s2.roi_pct:+.2%})")


if __name__ == "__main__":
    asyncio.run(_self_test())
