"""
config_and_database.py
======================
Module 1 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Responsibilities
----------------
1. Load and validate all runtime configuration from environment variables / `.env`
   into strongly typed, immutable dataclasses.
2. Provide a fully asynchronous SQLite persistence layer (`aiosqlite`) with
   versioned schema migrations and CRUD helpers for:
       - markets_catalog
       - orderbook_snapshots
       - fair_value_models
       - alert_history
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

logger = logging.getLogger("pm_engine.config_db")


# =============================================================================
# Environment helpers
# =============================================================================

class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env_str(key: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        raise ConfigError(f"Missing required environment variable: {key}")
    return (value or "").strip()


def _env_float(key: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    raw = os.getenv(key)
    try:
        value = float(raw) if raw not in (None, "") else float(default)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a float, got {raw!r}") from exc
    if lo is not None and value < lo:
        raise ConfigError(f"{key}={value} is below minimum {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{key}={value} is above maximum {hi}")
    return value


def _env_int(key: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    raw = os.getenv(key)
    try:
        value = int(raw) if raw not in (None, "") else int(default)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an int, got {raw!r}") from exc
    if lo is not None and value < lo:
        raise ConfigError(f"{key}={value} is below minimum {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{key}={value} is above maximum {hi}")
    return value


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(key: str, default: Sequence[str]) -> Tuple[str, ...]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_float_list(key: str, default: Sequence[float]) -> Tuple[float, ...]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return tuple(float(x) for x in default)
    try:
        return tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be a comma-separated list of floats") from exc


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass(frozen=True)
class APIKeys:
    xai_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    odds_api_key: str
    polymarket_api_key: str
    polymarket_api_secret: str
    polymarket_passphrase: str
    polymarket_private_key: str

    def redacted(self) -> Dict[str, str]:
        def mask(v: str) -> str:
            return "" if not v else f"{v[:4]}…{v[-2:]}" if len(v) > 8 else "****"
        return {k: mask(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class EndpointConfig:
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    telegram_base_url: str = "https://api.telegram.org"
    polymarket_event_url: str = "https://polymarket.com/event"


@dataclass(frozen=True)
class RateLimitConfig:
    gamma_rps: float = 5.0
    clob_rps: float = 10.0
    odds_api_rps: float = 1.0
    telegram_rps: float = 0.9
    xai_concurrency: int = 4
    http_concurrency: int = 32
    max_retries: int = 6
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    request_timeout_s: float = 20.0


@dataclass(frozen=True)
class StrategyConfig:
    arb_roi_threshold: float = 0.02
    ev_roi_threshold: float = 0.10
    high_conviction_roi_threshold: float = 1.00
    min_edge_pct: float = 0.03
    kelly_fraction: float = 0.25
    max_stake_pct_per_market: float = 0.05
    max_portfolio_exposure_pct: float = 0.30
    bankroll_usd: float = 10_000.0
    vwap_clip_sizes_usd: Tuple[float, ...] = (100.0, 500.0, 1_000.0)
    min_market_liquidity_usd: float = 2_000.0
    min_24h_volume_usd: float = 1_000.0
    max_spread: float = 0.15
    max_resolution_risk: int = 7
    alert_roi_delta: float = 0.05
    alert_cooldown_s: int = 3_600
    devig_method: str = "shin"


@dataclass(frozen=True)
class GrokConfig:
    model: str = "grok-4.5"
    enable_x_search: bool = True
    enable_web_search: bool = True
    prefilter_min_volume_velocity: float = 1.5
    prefilter_min_spread: float = 0.04
    prefilter_catalyst_window_h: float = 72.0
    max_markets_per_cycle: int = 25
    temperature: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path = Path("data/pm_engine.sqlite3")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    scan_interval_s: int = 300
    heartbeat_interval_s: int = 3_600
    snapshot_retention_days: int = 30
    dry_run: bool = True


@dataclass(frozen=True)
class AppConfig:
    keys: APIKeys
    endpoints: EndpointConfig
    rate_limits: RateLimitConfig
    strategy: StrategyConfig
    grok: GrokConfig
    runtime: RuntimeConfig

    def validate(self, require_keys: Iterable[str] = ()) -> None:
        for name in require_keys:
            if not getattr(self.keys, name, ""):
                raise ConfigError(f"API key '{name}' is required for this run mode")
        s = self.strategy
        if not 0 < s.kelly_fraction <= 1:
            raise ConfigError("KELLY_FRACTION must be in (0, 1]")
        if s.devig_method not in {"multiplicative", "additive", "power", "shin"}:
            raise ConfigError(f"Unknown DEVIG_METHOD '{s.devig_method}'")
        if s.arb_roi_threshold >= s.ev_roi_threshold:
            logger.warning("ARB_ROI_THRESHOLD >= EV_ROI_THRESHOLD; arb tier will swallow +EV tier")
        if any(c <= 0 for c in s.vwap_clip_sizes_usd):
            raise ConfigError("VWAP_CLIP_SIZES_USD must all be positive")

    def summary(self) -> Dict[str, Any]:
        return {
            "keys": self.keys.redacted(),
            "strategy": asdict(self.strategy),
            "grok": asdict(self.grok),
            "rate_limits": asdict(self.rate_limits),
            "runtime": {k: str(v) for k, v in asdict(self.runtime).items()},
        }


def load_config(env_file: Optional[str] = ".env") -> AppConfig:
    """Load configuration from the environment (optionally seeded by a .env file)."""
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    keys = APIKeys(
        xai_api_key=_env_str("XAI_API_KEY"),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env_str("TELEGRAM_CHAT_ID"),
        odds_api_key=_env_str("ODDS_API_KEY"),
        polymarket_api_key=_env_str("POLYMARKET_API_KEY"),
        polymarket_api_secret=_env_str("POLYMARKET_API_SECRET"),
        polymarket_passphrase=_env_str("POLYMARKET_PASSPHRASE"),
        polymarket_private_key=_env_str("POLYMARKET_PRIVATE_KEY"),
    )
    rl = RateLimitConfig(
        gamma_rps=_env_float("GAMMA_RPS", 5.0, lo=0.1),
        clob_rps=_env_float("CLOB_RPS", 10.0, lo=0.1),
        odds_api_rps=_env_float("ODDS_API_RPS", 1.0, lo=0.05),
        telegram_rps=_env_float("TELEGRAM_RPS", 0.9, lo=0.05),
        xai_concurrency=_env_int("XAI_CONCURRENCY", 4, lo=1, hi=64),
        http_concurrency=_env_int("HTTP_CONCURRENCY", 32, lo=1, hi=512),
        max_retries=_env_int("MAX_RETRIES", 6, lo=0, hi=20),
        backoff_base_s=_env_float("BACKOFF_BASE_S", 0.5, lo=0.01),
        backoff_cap_s=_env_float("BACKOFF_CAP_S", 30.0, lo=0.1),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 20.0, lo=1.0),
    )
    strat = StrategyConfig(
        arb_roi_threshold=_env_float("ARB_ROI_THRESHOLD", 0.02, lo=0.0),
        ev_roi_threshold=_env_float("EV_ROI_THRESHOLD", 0.10, lo=0.0),
        high_conviction_roi_threshold=_env_float("HIGH_CONVICTION_ROI_THRESHOLD", 1.00, lo=0.0),
        min_edge_pct=_env_float("MIN_EDGE_PCT", 0.03, lo=0.0, hi=1.0),
        kelly_fraction=_env_float("KELLY_FRACTION", 0.25, lo=0.0, hi=1.0),
        max_stake_pct_per_market=_env_float("MAX_STAKE_PCT_PER_MARKET", 0.05, lo=0.0, hi=1.0),
        max_portfolio_exposure_pct=_env_float("MAX_PORTFOLIO_EXPOSURE_PCT", 0.30, lo=0.0, hi=1.0),
        bankroll_usd=_env_float("BANKROLL_USD", 10_000.0, lo=0.0),
        vwap_clip_sizes_usd=_env_float_list("VWAP_CLIP_SIZES_USD", (100.0, 500.0, 1_000.0)),
        min_market_liquidity_usd=_env_float("MIN_MARKET_LIQUIDITY_USD", 2_000.0, lo=0.0),
        min_24h_volume_usd=_env_float("MIN_24H_VOLUME_USD", 1_000.0, lo=0.0),
        max_spread=_env_float("MAX_SPREAD", 0.15, lo=0.0, hi=1.0),
        max_resolution_risk=_env_int("MAX_RESOLUTION_RISK", 7, lo=1, hi=10),
        alert_roi_delta=_env_float("ALERT_ROI_DELTA", 0.05, lo=0.0),
        alert_cooldown_s=_env_int("ALERT_COOLDOWN_S", 3_600, lo=0),
        devig_method=_env_str("DEVIG_METHOD", "shin").lower(),
    )
    grok = GrokConfig(
        model=_env_str("GROK_MODEL", "grok-4.5"),
        enable_x_search=_env_bool("GROK_ENABLE_X_SEARCH", True),
        enable_web_search=_env_bool("GROK_ENABLE_WEB_SEARCH", True),
        prefilter_min_volume_velocity=_env_float("PREFILTER_MIN_VOLUME_VELOCITY", 1.5, lo=0.0),
        prefilter_min_spread=_env_float("PREFILTER_MIN_SPREAD", 0.04, lo=0.0, hi=1.0),
        prefilter_catalyst_window_h=_env_float("PREFILTER_CATALYST_WINDOW_H", 72.0, lo=0.0),
        max_markets_per_cycle=_env_int("GROK_MAX_MARKETS_PER_CYCLE", 25, lo=0),
        temperature=_env_float("GROK_TEMPERATURE", 0.1, lo=0.0, hi=2.0),
    )
    runtime = RuntimeConfig(
        db_path=Path(_env_str("DB_PATH", "data/pm_engine.sqlite3")),
        log_dir=Path(_env_str("LOG_DIR", "logs")),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        scan_interval_s=_env_int("SCAN_INTERVAL_S", 300, lo=10),
        heartbeat_interval_s=_env_int("HEARTBEAT_INTERVAL_S", 3_600, lo=60),
        snapshot_retention_days=_env_int("SNAPSHOT_RETENTION_DAYS", 30, lo=1),
        dry_run=_env_bool("DRY_RUN", True),
    )
    cfg = AppConfig(keys, EndpointConfig(), rl, strat, grok, runtime)
    cfg.validate()
    return cfg


# =============================================================================
# Row dataclasses
# =============================================================================

@dataclass
class MarketRecord:
    condition_id: str
    question: str
    slug: str = ""
    event_slug: str = ""
    category: str = ""
    description: str = ""
    outcomes: List[str] = field(default_factory=list)
    token_ids: List[str] = field(default_factory=list)
    end_date_iso: Optional[str] = None
    volume_24h: float = 0.0
    volume_total: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    neg_risk: bool = False


@dataclass
class OrderbookSnapshot:
    token_id: str
    condition_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    mid: Optional[float]
    vwap_100: Optional[float]
    vwap_500: Optional[float]
    vwap_1000: Optional[float]
    slippage_100: Optional[float]
    slippage_500: Optional[float]
    slippage_1000: Optional[float]
    bid_depth_usd: float
    ask_depth_usd: float
    imbalance: Optional[float]
    captured_at: float = field(default_factory=time.time)


@dataclass
class FairValueRecord:
    condition_id: str
    token_id: str
    outcome: str
    prior_prob: float
    prior_source: str
    likelihood_ratios: List[float]
    confidence_weights: List[float]
    posterior_prob: float
    fair_decimal_odds: float
    market_price: float
    entry_vwap: Optional[float]
    expected_roi: float
    edge_pct: float
    kelly_stake_pct: float
    resolution_risk: int
    signal_type: str  # "arb" | "ev" | "high_conviction"
    rationale: str = ""
    closing_price: Optional[float] = None
    clv: Optional[float] = None
    created_at: float = field(default_factory=time.time)


# =============================================================================
# Schema migrations
# =============================================================================

MIGRATIONS: List[Tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS markets_catalog (
        condition_id   TEXT PRIMARY KEY,
        question       TEXT NOT NULL,
        slug           TEXT,
        event_slug     TEXT,
        category       TEXT,
        description    TEXT,
        outcomes_json  TEXT NOT NULL DEFAULT '[]',
        token_ids_json TEXT NOT NULL DEFAULT '[]',
        end_date_iso   TEXT,
        volume_24h     REAL DEFAULT 0,
        volume_total   REAL DEFAULT 0,
        liquidity      REAL DEFAULT 0,
        active         INTEGER DEFAULT 1,
        neg_risk       INTEGER DEFAULT 0,
        first_seen_at  REAL NOT NULL,
        updated_at     REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS market_volume_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id TEXT NOT NULL REFERENCES markets_catalog(condition_id) ON DELETE CASCADE,
        volume_24h   REAL,
        volume_total REAL,
        liquidity    REAL,
        recorded_at  REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_volhist_cond_time ON market_volume_history(condition_id, recorded_at);
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        token_id      TEXT NOT NULL,
        condition_id  TEXT NOT NULL,
        best_bid      REAL, best_ask REAL, spread REAL, mid REAL,
        vwap_100      REAL, vwap_500 REAL, vwap_1000 REAL,
        slippage_100  REAL, slippage_500 REAL, slippage_1000 REAL,
        bid_depth_usd REAL, ask_depth_usd REAL, imbalance REAL,
        captured_at   REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ob_token_time ON orderbook_snapshots(token_id, captured_at);
    CREATE TABLE IF NOT EXISTS fair_value_models (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_id       TEXT NOT NULL,
        token_id           TEXT NOT NULL,
        outcome            TEXT,
        prior_prob         REAL NOT NULL,
        prior_source       TEXT,
        likelihood_ratios  TEXT NOT NULL DEFAULT '[]',
        confidence_weights TEXT NOT NULL DEFAULT '[]',
        posterior_prob     REAL NOT NULL,
        fair_decimal_odds  REAL NOT NULL,
        market_price       REAL NOT NULL,
        entry_vwap         REAL,
        expected_roi       REAL NOT NULL,
        edge_pct           REAL NOT NULL,
        kelly_stake_pct    REAL NOT NULL,
        resolution_risk    INTEGER,
        signal_type        TEXT,
        rationale          TEXT,
        closing_price      REAL,
        clv                REAL,
        created_at         REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_fv_token_time ON fair_value_models(token_id, created_at);
    CREATE TABLE IF NOT EXISTS alert_history (
        alert_key    TEXT PRIMARY KEY,
        condition_id TEXT NOT NULL,
        token_id     TEXT NOT NULL,
        signal_type  TEXT NOT NULL,
        last_roi     REAL NOT NULL,
        last_price   REAL,
        alert_count  INTEGER NOT NULL DEFAULT 1,
        first_sent_at REAL NOT NULL,
        last_sent_at  REAL NOT NULL
    );
    """),
    (2, """
    ALTER TABLE fair_value_models ADD COLUMN model_version TEXT DEFAULT 'v1';
    CREATE INDEX IF NOT EXISTS idx_fv_unresolved ON fair_value_models(condition_id) WHERE clv IS NULL;
    CREATE INDEX IF NOT EXISTS idx_catalog_active_end ON markets_catalog(active, end_date_iso);
    """),
]


# =============================================================================
# Database manager
# =============================================================================

class DatabaseManager:
    """Async SQLite manager with WAL mode, migrations, and a write lock."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

    # ---------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path.as_posix())
        self._conn.row_factory = aiosqlite.Row
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON",
                       "busy_timeout=5000", "temp_store=MEMORY"):
            await self._conn.execute(f"PRAGMA {pragma};")
        await self._migrate()
        logger.info("Database connected at %s", self.db_path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "DatabaseManager":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("DatabaseManager is not connected")
        return self._conn

    async def _migrate(self) -> None:
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at REAL)"
        )
        async with self.conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version") as cur:
            row = await cur.fetchone()
        current = int(row[0]) if row else 0
        for version, sql in MIGRATIONS:
            if version <= current:
                continue
            logger.info("Applying schema migration v%d", version)
            async with self._write_lock:
                try:
                    await self.conn.executescript(sql)
                except aiosqlite.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
                    logger.warning("Migration v%d partially applied already: %s", version, exc)
                await self.conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)", (version, time.time())
                )
                await self.conn.commit()

    # ------------------------------------------------------------ markets_catalog
    async def upsert_markets(self, markets: Sequence[MarketRecord]) -> int:
        if not markets:
            return 0
        now = time.time()
        rows = [(
            m.condition_id, m.question, m.slug, m.event_slug, m.category, m.description,
            json.dumps(m.outcomes), json.dumps(m.token_ids), m.end_date_iso,
            m.volume_24h, m.volume_total, m.liquidity, int(m.active), int(m.neg_risk), now, now,
        ) for m in markets]
        hist = [(m.condition_id, m.volume_24h, m.volume_total, m.liquidity, now) for m in markets]
        async with self._write_lock:
            await self.conn.executemany("""
                INSERT INTO markets_catalog (condition_id, question, slug, event_slug, category, description,
                    outcomes_json, token_ids_json, end_date_iso, volume_24h, volume_total, liquidity,
                    active, neg_risk, first_seen_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(condition_id) DO UPDATE SET
                    question=excluded.question, slug=excluded.slug, event_slug=excluded.event_slug,
                    category=excluded.category, description=excluded.description,
                    outcomes_json=excluded.outcomes_json, token_ids_json=excluded.token_ids_json,
                    end_date_iso=excluded.end_date_iso, volume_24h=excluded.volume_24h,
                    volume_total=excluded.volume_total, liquidity=excluded.liquidity,
                    active=excluded.active, neg_risk=excluded.neg_risk, updated_at=excluded.updated_at
            """, rows)
            await self.conn.executemany(
                "INSERT INTO market_volume_history (condition_id, volume_24h, volume_total, liquidity, recorded_at)"
                " VALUES (?,?,?,?,?)", hist)
            await self.conn.commit()
        return len(rows)

    @staticmethod
    def _row_to_market(row: aiosqlite.Row) -> MarketRecord:
        return MarketRecord(
            condition_id=row["condition_id"], question=row["question"], slug=row["slug"] or "",
            event_slug=row["event_slug"] or "", category=row["category"] or "",
            description=row["description"] or "", outcomes=json.loads(row["outcomes_json"]),
            token_ids=json.loads(row["token_ids_json"]), end_date_iso=row["end_date_iso"],
            volume_24h=row["volume_24h"] or 0.0, volume_total=row["volume_total"] or 0.0,
            liquidity=row["liquidity"] or 0.0, active=bool(row["active"]), neg_risk=bool(row["neg_risk"]),
        )

    async def get_market(self, condition_id: str) -> Optional[MarketRecord]:
        async with self.conn.execute(
            "SELECT * FROM markets_catalog WHERE condition_id = ?", (condition_id,)
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_market(row) if row else None

    async def list_active_markets(self, min_liquidity: float = 0.0, limit: int = 5_000) -> List[MarketRecord]:
        async with self.conn.execute(
            "SELECT * FROM markets_catalog WHERE active = 1 AND liquidity >= ? "
            "ORDER BY volume_24h DESC LIMIT ?", (min_liquidity, limit)
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_market(r) for r in rows]

    async def deactivate_missing_markets(self, live_condition_ids: Iterable[str]) -> int:
        live = set(live_condition_ids)
        async with self.conn.execute("SELECT condition_id FROM markets_catalog WHERE active = 1") as cur:
            stale = [r[0] for r in await cur.fetchall() if r[0] not in live]
        if not stale:
            return 0
        async with self._write_lock:
            await self.conn.executemany(
                "UPDATE markets_catalog SET active = 0, updated_at = ? WHERE condition_id = ?",
                [(time.time(), cid) for cid in stale])
            await self.conn.commit()
        return len(stale)

    async def get_volume_velocity(self, condition_id: str, window_days: int = 7) -> Optional[float]:
        """Ratio of the latest 24h volume to the trailing average of 24h volume readings."""
        since = time.time() - window_days * 86_400
        async with self.conn.execute(
            "SELECT volume_24h FROM market_volume_history WHERE condition_id = ? AND recorded_at >= ? "
            "ORDER BY recorded_at ASC", (condition_id, since)
        ) as cur:
            vols = [r[0] for r in await cur.fetchall() if r[0] is not None]
        if len(vols) < 2:
            return None
        baseline = sum(vols[:-1]) / len(vols[:-1])
        return vols[-1] / baseline if baseline > 0 else None

    # --------------------------------------------------------- orderbook_snapshots
    async def insert_orderbook_snapshots(self, snaps: Sequence[OrderbookSnapshot]) -> int:
        if not snaps:
            return 0
        rows = [(
            s.token_id, s.condition_id, s.best_bid, s.best_ask, s.spread, s.mid,
            s.vwap_100, s.vwap_500, s.vwap_1000, s.slippage_100, s.slippage_500, s.slippage_1000,
            s.bid_depth_usd, s.ask_depth_usd, s.imbalance, s.captured_at,
        ) for s in snaps]
        async with self._write_lock:
            await self.conn.executemany("""
                INSERT INTO orderbook_snapshots (token_id, condition_id, best_bid, best_ask, spread, mid,
                    vwap_100, vwap_500, vwap_1000, slippage_100, slippage_500, slippage_1000,
                    bid_depth_usd, ask_depth_usd, imbalance, captured_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            await self.conn.commit()
        return len(rows)

    async def latest_snapshot(self, token_id: str) -> Optional[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM orderbook_snapshots WHERE token_id = ? ORDER BY captured_at DESC LIMIT 1",
            (token_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def snapshot_history(self, token_id: str, since_ts: float) -> List[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM orderbook_snapshots WHERE token_id = ? AND captured_at >= ? ORDER BY captured_at",
            (token_id, since_ts)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def prune_snapshots(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86_400
        async with self._write_lock:
            cur = await self.conn.execute("DELETE FROM orderbook_snapshots WHERE captured_at < ?", (cutoff,))
            await self.conn.execute("DELETE FROM market_volume_history WHERE recorded_at < ?", (cutoff,))
            await self.conn.commit()
            return cur.rowcount or 0

    # ----------------------------------------------------------- fair_value_models
    async def insert_fair_value(self, fv: FairValueRecord, model_version: str = "v1") -> int:
        async with self._write_lock:
            cur = await self.conn.execute("""
                INSERT INTO fair_value_models (condition_id, token_id, outcome, prior_prob, prior_source,
                    likelihood_ratios, confidence_weights, posterior_prob, fair_decimal_odds, market_price,
                    entry_vwap, expected_roi, edge_pct, kelly_stake_pct, resolution_risk, signal_type,
                    rationale, closing_price, clv, created_at, model_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                fv.condition_id, fv.token_id, fv.outcome, fv.prior_prob, fv.prior_source,
                json.dumps(fv.likelihood_ratios), json.dumps(fv.confidence_weights), fv.posterior_prob,
                fv.fair_decimal_odds, fv.market_price, fv.entry_vwap, fv.expected_roi, fv.edge_pct,
                fv.kelly_stake_pct, fv.resolution_risk, fv.signal_type, fv.rationale,
                fv.closing_price, fv.clv, fv.created_at, model_version,
            ))
            await self.conn.commit()
            return int(cur.lastrowid or 0)

    async def latest_fair_value(self, token_id: str) -> Optional[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM fair_value_models WHERE token_id = ? ORDER BY created_at DESC LIMIT 1", (token_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        out = dict(row)
        out["likelihood_ratios"] = json.loads(out["likelihood_ratios"])
        out["confidence_weights"] = json.loads(out["confidence_weights"])
        return out

    async def record_closing_line(self, token_id: str, closing_price: float) -> int:
        """Stamp closing price and CLV (closing price minus entry price, in probability points)."""
        async with self._write_lock:
            cur = await self.conn.execute("""
                UPDATE fair_value_models
                SET closing_price = ?, clv = ? - COALESCE(entry_vwap, market_price)
                WHERE token_id = ? AND clv IS NULL
            """, (closing_price, closing_price, token_id))
            await self.conn.commit()
            return cur.rowcount or 0

    async def clv_summary(self, since_ts: float = 0.0) -> Dict[str, Any]:
        async with self.conn.execute("""
            SELECT signal_type, COUNT(*) AS n, AVG(clv) AS avg_clv,
                   SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS beat_rate,
                   AVG(expected_roi) AS avg_expected_roi
            FROM fair_value_models WHERE clv IS NOT NULL AND created_at >= ?
            GROUP BY signal_type
        """, (since_ts,)) as cur:
            return {r["signal_type"]: dict(r) for r in await cur.fetchall()}

    # --------------------------------------------------------------- alert_history
    @staticmethod
    def make_alert_key(condition_id: str, token_id: str, signal_type: str) -> str:
        return f"{condition_id}:{token_id}:{signal_type}"

    async def should_send_alert(self, condition_id: str, token_id: str, signal_type: str,
                                roi: float, roi_delta: float, cooldown_s: int) -> bool:
        """True if never alerted, or ROI improved by >= roi_delta, or cooldown elapsed."""
        key = self.make_alert_key(condition_id, token_id, signal_type)
        async with self.conn.execute(
            "SELECT last_roi, last_sent_at FROM alert_history WHERE alert_key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return True
        if roi >= row["last_roi"] + roi_delta:
            return True
        return cooldown_s > 0 and (time.time() - row["last_sent_at"]) >= cooldown_s and roi >= row["last_roi"]

    async def record_alert(self, condition_id: str, token_id: str, signal_type: str,
                           roi: float, price: Optional[float]) -> None:
        key = self.make_alert_key(condition_id, token_id, signal_type)
        now = time.time()
        async with self._write_lock:
            await self.conn.execute("""
                INSERT INTO alert_history (alert_key, condition_id, token_id, signal_type, last_roi,
                    last_price, alert_count, first_sent_at, last_sent_at)
                VALUES (?,?,?,?,?,?,1,?,?)
                ON CONFLICT(alert_key) DO UPDATE SET
                    last_roi = excluded.last_roi, last_price = excluded.last_price,
                    alert_count = alert_history.alert_count + 1, last_sent_at = excluded.last_sent_at
            """, (key, condition_id, token_id, signal_type, roi, price, now, now))
            await self.conn.commit()

    async def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM alert_history ORDER BY last_sent_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # -------------------------------------------------------------------- health
    async def health_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for table in ("markets_catalog", "orderbook_snapshots", "fair_value_models", "alert_history"):
            async with self.conn.execute(f"SELECT COUNT(*) FROM {table}") as cur:
                row = await cur.fetchone()
                stats[table] = int(row[0]) if row else 0
        return stats

    async def vacuum(self) -> None:
        async with self._write_lock:
            await self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            await self.conn.execute("VACUUM;")


# =============================================================================
# Self-test
# =============================================================================

async def _self_test() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(env_file=None)
    db_path = Path("/tmp/pm_engine_selftest.sqlite3")
    if db_path.exists():
        db_path.unlink()
    async with DatabaseManager(db_path) as db:
        m = MarketRecord("0xabc", "Will X happen?", outcomes=["Yes", "No"], token_ids=["1", "2"], liquidity=5000)
        await db.upsert_markets([m])
        assert (await db.get_market("0xabc")).token_ids == ["1", "2"]
        snap = OrderbookSnapshot("1", "0xabc", 0.40, 0.42, 0.02, 0.41, 0.421, 0.43, 0.44,
                                 0.001, 0.01, 0.02, 3000, 2500, 0.09)
        await db.insert_orderbook_snapshots([snap])
        fv = FairValueRecord("0xabc", "1", "Yes", 0.45, "shin", [1.3], [0.8], 0.52, 1 / 0.52,
                             0.42, 0.43, 0.209, 0.09, 0.03, 3, "ev")
        await db.insert_fair_value(fv)
        await db.record_closing_line("1", 0.50)
        assert await db.should_send_alert("0xabc", "1", "ev", 0.2, 0.05, 3600)
        await db.record_alert("0xabc", "1", "ev", 0.2, 0.42)
        assert not await db.should_send_alert("0xabc", "1", "ev", 0.22, 0.05, 3600)
        assert await db.should_send_alert("0xabc", "1", "ev", 0.26, 0.05, 3600)
        print("health:", await db.health_stats())
        print("clv:", await db.clv_summary())
    print("config:", json.dumps(cfg.summary()["strategy"], default=str))
    print("Self-test passed.")


if __name__ == "__main__":
    asyncio.run(_self_test())
