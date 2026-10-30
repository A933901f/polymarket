"""
quant_math_engine.py
====================
Module 3 of the Quantitative Fair-Value (+EV) & Arbitrage Engine.

Pure quantitative mathematics (numpy / scipy), no network I/O:
1. De-vigging (multiplicative, additive, power, Shin 1993) and priors from MatchedPair.
2. Microstructure fallback prior from OrderbookSnapshot (micro-price, spread, velocity, theta).
3. Log-odds Bayesian fair-value updating with correlation dampening.
4. Intra-market and sharp-vs-Polymarket pure arbitrage scanning with equalized stake splits.
5. Binary EV / ROI / fractional Kelly evaluation and portfolio-level Kelly scaling.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from config_and_database import AppConfig, FairValueRecord, MarketRecord, OrderbookSnapshot
from async_market_ingestor import MatchedPair, parse_iso

logger = logging.getLogger("pm_engine.quant")

EPS = 1e-9
PROB_FLOOR = 1e-6
DEVIG_METHODS = ("multiplicative", "additive", "power", "shin")


# =============================================================================
# 1. De-vigging
# =============================================================================

def implied_probabilities(decimal_odds: Sequence[float]) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or odds.size < 2:
        raise ValueError("Need at least two outcomes to de-vig")
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError(f"Decimal odds must be finite and > 1.0, got {decimal_odds}")
    return 1.0 / odds


def overround(decimal_odds: Sequence[float]) -> float:
    return float(implied_probabilities(decimal_odds).sum())


def _devig_multiplicative(q: np.ndarray, s: float) -> np.ndarray:
    return q / s


def _devig_additive(q: np.ndarray, s: float) -> np.ndarray:
    """Subtract equal margin from each outcome; iteratively drop outcomes that would go negative."""
    p = q.copy()
    active = np.ones_like(p, dtype=bool)
    for _ in range(q.size):
        margin = (p[active].sum() - 1.0) / active.sum()
        trial = np.where(active, p - margin, 0.0)
        negative = active & (trial < 0)
        if not negative.any():
            return trial
        active &= ~negative
        p = np.where(active, p, 0.0)
    return q / s


def _devig_power(q: np.ndarray, s: float) -> np.ndarray:
    """Find k > 1 such that sum(q_i ** k) = 1; inflates favorites relative to longshots."""
    f = lambda k: float(np.sum(q ** k) - 1.0)
    hi = 2.0
    while f(hi) > 0 and hi < 1e4:
        hi *= 2.0
    k = brentq(f, 1.0, hi, xtol=1e-14, maxiter=500)
    return q ** k


def shin_probabilities(q: np.ndarray, s: float, z: float) -> np.ndarray:
    return (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * (q ** 2) / s) - z) / (2.0 * (1.0 - z))


def _devig_shin(q: np.ndarray, s: float) -> Tuple[np.ndarray, float]:
    """Solve Shin (1993) for insider share z in (0, 1) such that probabilities sum to 1."""
    f = lambda z: float(shin_probabilities(q, s, z).sum() - 1.0)
    lo, hi = 0.0, 0.999999
    if f(lo) <= 0:  # sum(q)/sqrt(S) = sqrt(S) <= 1 only when no margin
        return q / s, 0.0
    if f(hi) > 0:
        logger.warning("Shin root not bracketed (S=%.4f); falling back to multiplicative", s)
        return q / s, float("nan")
    z = brentq(f, lo, hi, xtol=1e-14, maxiter=500)
    return shin_probabilities(q, s, z), float(z)


def devig_odds(decimal_odds: Sequence[float], method: str = "shin") -> np.ndarray:
    """Return fair probabilities summing to 1. If overround S <= 1 (zero margin or arb), normalize."""
    method = method.lower()
    if method not in DEVIG_METHODS:
        raise ValueError(f"Unknown de-vig method '{method}', choose from {DEVIG_METHODS}")
    q = implied_probabilities(decimal_odds)
    s = float(q.sum())
    if s <= 1.0 + EPS:
        if s < 1.0 - EPS:
            logger.info("Odds %s have overround %.4f < 1 (cross-book arb); normalizing", list(decimal_odds), s)
        return q / s
    if method == "multiplicative":
        p = _devig_multiplicative(q, s)
    elif method == "additive":
        p = _devig_additive(q, s)
    elif method == "power":
        p = _devig_power(q, s)
    else:
        p, _z = _devig_shin(q, s)
    p = np.clip(p, 0.0, 1.0)
    return p / p.sum()


def shin_insider_share(decimal_odds: Sequence[float]) -> float:
    q = implied_probabilities(decimal_odds)
    s = float(q.sum())
    return 0.0 if s <= 1.0 else _devig_shin(q, s)[1]


def prior_from_matched_pair(pair: MatchedPair, method: str = "shin") -> Dict[str, float]:
    """Map each Polymarket outcome to a de-vigged prior; 'NOT:<X>' resolves to 1 - P(X)."""
    data = pair.to_devig_input()
    names: List[str] = data["outcome_names"]
    probs = devig_odds(data["decimal_odds"], method)
    by_name = dict(zip(names, probs.tolist()))
    priors: Dict[str, float] = {}
    for pm_outcome, sharp_name in data["outcome_map"].items():
        if sharp_name.startswith("NOT:"):
            base = sharp_name[4:]
            if base not in by_name:
                raise KeyError(f"Sharp outcome '{base}' missing for {pair.market.condition_id}")
            priors[pm_outcome] = 1.0 - by_name[base]
        else:
            if sharp_name not in by_name:
                raise KeyError(f"Sharp outcome '{sharp_name}' missing for {pair.market.condition_id}")
            priors[pm_outcome] = by_name[sharp_name]
    return {k: float(np.clip(v, PROB_FLOOR, 1.0 - PROB_FLOOR)) for k, v in priors.items()}


# =============================================================================
# 2. Microstructure fallback prior
# =============================================================================

@dataclass
class MicrostructurePrior:
    prior_prob: float
    micro_price: float
    mid: float
    spread_penalty: float
    velocity_factor: float
    theta_floor: float
    hours_to_expiry: Optional[float]
    confidence: float  # 0..1, how much to trust this prior versus a sharp line


def hours_to_expiry(end_date_iso: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    end = parse_iso(end_date_iso)
    if end is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (end - now).total_seconds() / 3600.0)


def volume_velocity_factor(velocity: Optional[float]) -> float:
    """Map 24h/7d-average volume ratio to a 0.5..1.5 trust multiplier for order-flow signals."""
    if velocity is None or not math.isfinite(velocity) or velocity <= 0:
        return 1.0
    return float(np.clip(1.0 + 0.5 * math.tanh(math.log(velocity)), 0.5, 1.5))


def microstructure_prior(snapshot: OrderbookSnapshot, market: Optional[MarketRecord] = None,
                         volume_velocity: Optional[float] = None, max_spread: float = 0.15,
                         now: Optional[datetime] = None) -> Optional[MicrostructurePrior]:
    """
    Order-book-informed baseline when no sharp line exists.
    micro_price = mid + OBI * spread / 2 (depth-weighted toward the thin side's price),
    with the OBI tilt shrunk by spread penalty and scaled by volume velocity, then clamped
    by a theta floor that tightens toward 0/1 as expiry approaches.
    """
    if snapshot.best_bid is None or snapshot.best_ask is None or snapshot.best_ask <= snapshot.best_bid:
        return None
    mid = (snapshot.best_bid + snapshot.best_ask) / 2.0
    spread = snapshot.best_ask - snapshot.best_bid
    obi = snapshot.imbalance if snapshot.imbalance is not None else 0.0
    spread_penalty = float(np.clip(spread / max(max_spread, EPS), 0.0, 1.0))
    v_factor = volume_velocity_factor(volume_velocity)
    tilt = obi * (spread / 2.0) * (1.0 - spread_penalty) * v_factor
    micro = float(np.clip(mid + tilt, snapshot.best_bid, snapshot.best_ask))
    hrs = hours_to_expiry(market.end_date_iso if market else None, now)
    if hrs is None:
        floor = 0.02
    else:
        floor = float(np.clip(0.02 * math.sqrt(hrs / 720.0), 0.001, 0.02))
    # The theta floor keeps priors off exact 0/1 but must never drag the prior outside the quoted market.
    prior = float(np.clip(micro, min(floor, snapshot.best_bid), max(1.0 - floor, snapshot.best_ask)))
    depth = snapshot.bid_depth_usd + snapshot.ask_depth_usd
    depth_conf = 1.0 - math.exp(-depth / 5_000.0)
    confidence = float(np.clip(depth_conf * (1.0 - spread_penalty) * min(v_factor, 1.0), 0.0, 1.0))
    return MicrostructurePrior(prior, micro, mid, spread_penalty, v_factor, floor, hrs, confidence)


# =============================================================================
# 3. Bayesian log-odds updating
# =============================================================================

def logit(p: float) -> float:
    p = min(max(p, PROB_FLOOR), 1.0 - PROB_FLOOR)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class PosteriorResult:
    prior_prob: float
    posterior_prob: float
    fair_decimal_odds: float
    prior_logit: float
    posterior_logit: float
    total_log_shift: float
    effective_weights: List[float] = field(default_factory=list)


def compute_posterior_fair_value(prior_prob: float, likelihood_ratios: Sequence[float],
                                 confidence_weights: Optional[Sequence[float]] = None,
                                 correlation_dampening: float = 0.35,
                                 max_log_shift: float = 4.0) -> PosteriorResult:
    """
    logit(P_post) = logit(P_prior) + sum_k w_k_eff * ln(LR_k).
    Same-direction signals are sorted by |w ln LR| and the j-th gets (1 - rho)^j extra damping,
    so correlated news echoes don't compound like independent evidence. Total shift is capped.
    """
    if not 0.0 < prior_prob < 1.0:
        raise ValueError(f"prior_prob must be in (0, 1), got {prior_prob}")
    lrs = list(likelihood_ratios)
    weights = list(confidence_weights) if confidence_weights is not None else [1.0] * len(lrs)
    if len(weights) != len(lrs):
        raise ValueError("likelihood_ratios and confidence_weights must have equal length")
    rho = float(np.clip(correlation_dampening, 0.0, 1.0))
    contributions: List[Tuple[int, float]] = []
    for i, (lr, w) in enumerate(zip(lrs, weights)):
        if lr is None or not math.isfinite(lr) or lr <= 0:
            logger.debug("Skipping invalid likelihood ratio %r", lr)
            continue
        contributions.append((i, float(np.clip(w, 0.0, 1.0)) * math.log(lr)))
    eff = [0.0] * len(lrs)
    shift = 0.0
    for sign in (1, -1):
        same = sorted([c for c in contributions if c[1] * sign > 0], key=lambda c: -abs(c[1]))
        for rank, (idx, contrib) in enumerate(same):
            damp = (1.0 - rho) ** rank
            shift += contrib * damp
            eff[idx] = float(np.clip(weights[idx], 0.0, 1.0)) * damp
    shift = float(np.clip(shift, -max_log_shift, max_log_shift))
    prior_l = logit(prior_prob)
    post_l = prior_l + shift
    post = float(np.clip(sigmoid(post_l), PROB_FLOOR, 1.0 - PROB_FLOOR))
    return PosteriorResult(prior_prob, post, 1.0 / post, prior_l, post_l, shift, eff)


# =============================================================================
# 4. Arbitrage scanning
# =============================================================================

@dataclass
class ArbLeg:
    venue: str           # "polymarket" or a bookmaker key
    outcome: str
    ref: str             # token_id or event_id
    price: float         # probability price paid (1 / decimal odds)
    decimal_odds: float
    stake_usd: float = 0.0
    payout_usd: float = 0.0


@dataclass
class ArbOpportunity:
    kind: str            # "intra_market" | "neg_risk_event" | "cross_venue"
    key: str
    condition_ids: List[str]
    legs: List[ArbLeg]
    implied_sum: float
    roi: float
    outlay_usd: float
    guaranteed_profit_usd: float
    notes: str = ""


def equalized_stakes(decimal_odds: Sequence[float], outlay_usd: float) -> Tuple[List[float], float]:
    """Stakes s_i = X * (1/o_i) / sum(1/o) so every outcome pays X / sum(1/o)."""
    inv = np.array([1.0 / o for o in decimal_odds])
    total = float(inv.sum())
    stakes = (outlay_usd * inv / total).tolist()
    return stakes, outlay_usd / total


def _leg_price(snap: OrderbookSnapshot, stake_usd: float) -> Optional[float]:
    """Executable price for a leg: the smallest VWAP tier covering the stake, else best ask."""
    for tier, vwap in ((100.0, snap.vwap_100), (500.0, snap.vwap_500), (1_000.0, snap.vwap_1000)):
        if stake_usd <= tier:
            return vwap
    return None


def _build_arb(kind: str, key: str, cids: List[str], legs: List[ArbLeg], outlay: float,
               notes: str = "") -> ArbOpportunity:
    stakes, payout = equalized_stakes([l.decimal_odds for l in legs], outlay)
    for leg, st in zip(legs, stakes):
        leg.stake_usd, leg.payout_usd = st, payout
    implied = sum(l.price for l in legs)
    return ArbOpportunity(kind, key, cids, legs, implied, 1.0 / implied - 1.0, outlay, payout - outlay, notes)


def _book_arb(kind: str, key: str, cids: List[str], items: List[Tuple[str, str, OrderbookSnapshot]],
              outlay: float, threshold: float, notes: str) -> Optional[ArbOpportunity]:
    """Two-pass: size with best asks, then re-price each leg at its VWAP tier and re-check."""
    if any(s.best_ask is None for _, _, s in items):
        return None
    prices = [s.best_ask for _, _, s in items]
    if sum(prices) >= 1.0 - threshold / (1.0 + threshold):
        return None
    for _ in range(2):
        stakes, _ = equalized_stakes([1.0 / p for p in prices], outlay)
        new_prices = [_leg_price(s, st) for (_, _, s), st in zip(items, stakes)]
        if any(p is None for p in new_prices):
            return None
        prices = [float(p) for p in new_prices]
    if 1.0 / sum(prices) - 1.0 < threshold:
        return None
    legs = [ArbLeg("polymarket", o, t, p, 1.0 / p) for (o, t, _), p in zip(items, prices)]
    return _build_arb(kind, key, cids, legs, outlay, notes)


def scan_arbitrage_opportunities(cfg: AppConfig, markets: Sequence[MarketRecord],
                                 snapshots: Mapping[str, OrderbookSnapshot],
                                 pairs: Sequence[MatchedPair] = (),
                                 outlay_usd: float = 500.0,
                                 event_market_counts: Optional[Mapping[str, int]] = None) -> List[ArbOpportunity]:
    """
    `event_market_counts` maps event_slug -> number of live markets in the FULL catalog. A neg-risk
    event is only scanned when every one of its markets is present (otherwise the YES-sum is
    trivially < 1 because outcomes were filtered out, which is not an arbitrage).
    """
    threshold = cfg.strategy.arb_roi_threshold
    opps: List[ArbOpportunity] = []

    # Intra-market: all outcomes of one market are mutually exclusive and exhaustive.
    for m in markets:
        items = [(o, t, snapshots[t]) for o, t in zip(m.outcomes, m.token_ids) if t in snapshots]
        if len(items) == len(m.token_ids) >= 2:
            arb = _book_arb("intra_market", m.condition_id, [m.condition_id], items, outlay_usd, threshold,
                            "Buy every outcome; one share set pays $1.")
            if arb:
                opps.append(arb)

    # Neg-risk events: buy YES on every market in the event.
    groups: Dict[str, List[MarketRecord]] = defaultdict(list)
    for m in markets:
        if m.neg_risk and m.event_slug and len(m.outcomes) == 2:
            groups[m.event_slug].append(m)
    for slug, ms in groups.items():
        if len(ms) < 2:
            continue
        expected = (event_market_counts or {}).get(slug)
        if expected is None or len(ms) != expected:
            continue  # incomplete outcome set: cannot guarantee a payout
        items = []
        for m in ms:
            yes_idx = next((i for i, o in enumerate(m.outcomes) if o.lower() == "yes"), 0)
            tok = m.token_ids[yes_idx]
            if tok not in snapshots:
                items = []
                break
            items.append((m.question, tok, snapshots[tok]))
        if items:
            arb = _book_arb("neg_risk_event", slug, [m.condition_id for m in ms], items, outlay_usd, threshold,
                            "Assumes the listed markets cover every outcome; verify no unlisted 'Other'.")
            if arb:
                opps.append(arb)

    # Cross-venue: Polymarket outcome vs the sharp book's complement legs at best available odds.
    for pair in pairs:
        best: Dict[str, Tuple[float, str]] = {}
        for book, prices in pair.sharp.book_odds.items():
            for name, odds in prices.items():
                if name not in best or odds > best[name][0]:
                    best[name] = (odds, book)
        for pm_outcome, sharp_name in pair.outcome_map.items():
            tok = pair.token_map.get(pm_outcome)
            snap = snapshots.get(tok) if tok else None
            if snap is None or snap.best_ask is None:
                continue
            if sharp_name.startswith("NOT:"):
                hedge_names = [sharp_name[4:]]
            else:
                hedge_names = [n for n in pair.sharp_outcome_names if n != sharp_name]
            if not hedge_names or any(n not in best for n in hedge_names):
                continue
            hedge_inv = sum(1.0 / best[n][0] for n in hedge_names)
            pm_stake = outlay_usd * snap.best_ask / (snap.best_ask + hedge_inv)
            pm_price = _leg_price(snap, pm_stake)
            if pm_price is None:
                continue
            implied = pm_price + hedge_inv
            if 1.0 / implied - 1.0 < threshold:
                continue
            legs = [ArbLeg("polymarket", pm_outcome, tok, pm_price, 1.0 / pm_price)]
            legs += [ArbLeg(best[n][1], n, pair.sharp.event_id, 1.0 / best[n][0], best[n][0]) for n in hedge_names]
            opps.append(_build_arb("cross_venue", f"{pair.market.condition_id}:{pm_outcome}",
                                   [pair.market.condition_id], legs, outlay_usd,
                                   "Sharp odds can move before both legs fill; execute the book leg first."))
    opps.sort(key=lambda a: a.roi, reverse=True)
    return opps


# =============================================================================
# 5. EV, ROI, Kelly
# =============================================================================

def binary_kelly(p_true: float, price: float, fraction: float, cap: float) -> float:
    """Fractional Kelly for a $1-payout contract bought at `price`: f = frac * (p - c) / (1 - c)."""
    if not 0.0 < price < 1.0:
        return 0.0
    full = (p_true - price) / (1.0 - price)
    return float(np.clip(fraction * full, 0.0, cap))


@dataclass
class EVEvaluation:
    record: FairValueRecord
    stake_usd: float
    clip_tier: Optional[float]
    depth_limited: bool
    event_key: str = ""


def select_entry_price(snap: OrderbookSnapshot, provisional_stake_usd: float,
                       allow_top_of_book: bool = False) -> Tuple[Optional[float], Optional[float], bool]:
    """Pick the VWAP tier that covers the stake. Returns (price, tier, depth_limited)."""
    tiers = [(100.0, snap.vwap_100), (500.0, snap.vwap_500), (1_000.0, snap.vwap_1000)]
    for tier, vwap in tiers:
        if provisional_stake_usd <= tier and vwap is not None:
            return vwap, tier, False
    for tier, vwap in reversed(tiers):  # depth can't cover the stake: largest tier that fills
        if vwap is not None:
            return vwap, tier, True
    if allow_top_of_book and snap.best_ask is not None:
        return snap.best_ask, None, True
    return None, None, True


def classify_signal(cfg: AppConfig, roi: float, edge: float, resolution_risk: int,
                    arb_roi: Optional[float] = None) -> str:
    s = cfg.strategy
    if arb_roi is not None and arb_roi >= s.arb_roi_threshold and edge > 0:
        return "arb"
    if edge < s.min_edge_pct or resolution_risk > s.max_resolution_risk:
        return "pass"
    if roi >= s.high_conviction_roi_threshold:
        return "high_conviction"
    if roi >= s.ev_roi_threshold:
        return "ev"
    return "pass"


def evaluate_market_ev(cfg: AppConfig, market: MarketRecord, outcome: str, snapshot: OrderbookSnapshot,
                       posterior: PosteriorResult, prior_source: str,
                       likelihood_ratios: Sequence[float], confidence_weights: Sequence[float],
                       resolution_risk: int = 5, rationale: str = "", arb_roi: Optional[float] = None,
                       allow_top_of_book: bool = False) -> Optional[EVEvaluation]:
    s = cfg.strategy
    p = posterior.posterior_prob
    if snapshot.best_ask is None:
        return None
    provisional = binary_kelly(p, snapshot.best_ask, s.kelly_fraction, s.max_stake_pct_per_market) * s.bankroll_usd
    entry, tier, limited = select_entry_price(snapshot, max(provisional, 1.0), allow_top_of_book)
    if entry is None:
        return None
    kelly = binary_kelly(p, entry, s.kelly_fraction, s.max_stake_pct_per_market)
    stake = kelly * s.bankroll_usd
    if tier is not None and stake > tier:
        stake = tier  # never size beyond the depth the VWAP was computed for
        kelly = stake / s.bankroll_usd if s.bankroll_usd > 0 else 0.0
    roi = p / entry - 1.0
    edge = p - entry
    signal = classify_signal(cfg, roi, edge, resolution_risk, arb_roi)
    record = FairValueRecord(
        condition_id=market.condition_id, token_id=snapshot.token_id, outcome=outcome,
        prior_prob=posterior.prior_prob, prior_source=prior_source,
        likelihood_ratios=[float(x) for x in likelihood_ratios],
        confidence_weights=[float(x) for x in confidence_weights],
        posterior_prob=p, fair_decimal_odds=posterior.fair_decimal_odds,
        market_price=snapshot.best_ask, entry_vwap=entry, expected_roi=roi, edge_pct=edge,
        kelly_stake_pct=kelly if signal != "pass" else 0.0, resolution_risk=int(resolution_risk),
        signal_type=signal, rationale=rationale,
    )
    return EVEvaluation(record, stake if signal != "pass" else 0.0, tier, limited,
                        market.event_slug or market.condition_id)


def allocate_portfolio_kelly(cfg: AppConfig, evaluations: Sequence[EVEvaluation]) -> List[EVEvaluation]:
    """
    Simultaneous Kelly scaling:
    1. Bets sharing an event are correlated: cap each event's total at max_stake_pct_per_market.
    2. If the sum across events exceeds max_portfolio_exposure_pct, scale every stake proportionally.
    Mutates and returns the actionable evaluations, sorted by expected ROI.
    """
    s = cfg.strategy
    active = [e for e in evaluations if e.record.signal_type != "pass" and e.record.kelly_stake_pct > 0]
    by_event: Dict[str, List[EVEvaluation]] = defaultdict(list)
    for e in active:
        by_event[e.event_key].append(e)
    for group in by_event.values():
        total = sum(e.record.kelly_stake_pct for e in group)
        if total > s.max_stake_pct_per_market:
            scale = s.max_stake_pct_per_market / total
            for e in group:
                e.record.kelly_stake_pct *= scale
    grand = sum(e.record.kelly_stake_pct for e in active)
    if grand > s.max_portfolio_exposure_pct > 0:
        scale = s.max_portfolio_exposure_pct / grand
        for e in active:
            e.record.kelly_stake_pct *= scale
    for e in active:
        e.stake_usd = e.record.kelly_stake_pct * s.bankroll_usd
    return sorted(active, key=lambda e: e.record.expected_roi, reverse=True)


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> None:
    from config_and_database import load_config
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(env_file=None)

    # --- de-vigging on 2-way and 3-way books
    nba = [1.25, 4.00]
    soccer = [1.50, 4.20, 7.00]  # Home / Draw / Away
    for odds, label in ((nba, "NBA"), (soccer, "1X2")):
        mult = devig_odds(odds, "multiplicative")
        print(f"{label} odds={odds} overround={overround(odds):.4f} shin_z={shin_insider_share(odds):.5f}")
        for method in DEVIG_METHODS:
            p = devig_odds(odds, method)
            assert abs(p.sum() - 1.0) < 1e-9, (method, p.sum())
            print(f"  {method:<14} {np.round(p, 6).tolist()}  sum={p.sum():.6f}")
            if method in ("shin", "power"):
                assert p[0] > mult[0] and p[-1] < mult[-1], f"{method} should favor favorites"
    assert np.allclose(devig_odds([2.10, 2.10], "shin"), [0.5, 0.5])
    arb_odds = [2.10, 2.05]  # overround < 1
    assert abs(devig_odds(arb_odds, "shin").sum() - 1.0) < 1e-12

    # --- NOT:<Team> on 3-way soccer
    from async_market_ingestor import SharpEvent
    ev = SharpEvent("s1", "soccer_epl", None, "Arsenal", "Chelsea",
                    {"pinnacle": {"Arsenal": 1.50, "Draw": 4.20, "Chelsea": 7.00}},
                    {"Arsenal": 1.50, "Draw": 4.20, "Chelsea": 7.00}, "pinnacle")
    mkt = MarketRecord("0xsoc", "Will Arsenal beat Chelsea?", outcomes=["Yes", "No"], token_ids=["y", "n"])
    names = sorted(ev.consensus_odds)
    pair = MatchedPair(mkt, ev, 1.0, {"Yes": "Arsenal", "No": "NOT:Arsenal"}, {"Yes": "y", "No": "n"},
                       [ev.consensus_odds[n] for n in names], names)
    priors = prior_from_matched_pair(pair, "shin")
    p_ars = dict(zip(names, devig_odds(pair.sharp_decimal_odds, "shin")))["Arsenal"]
    assert abs(priors["Yes"] - p_ars) < 1e-9 and abs(priors["No"] - (1 - p_ars)) < 1e-9
    print(f"NOT resolution: Yes={priors['Yes']:.4f} No={priors['No']:.4f} (No includes Draw + Chelsea)")

    # --- Bayesian updating
    post = compute_posterior_fair_value(0.30, [2.0, 1.8, 0.9], [0.9, 0.7, 0.5])
    no_damp = compute_posterior_fair_value(0.30, [2.0, 1.8, 0.9], [0.9, 0.7, 0.5], correlation_dampening=0.0)
    manual = sigmoid(logit(0.30) + 0.9 * math.log(2.0) + 0.7 * math.log(1.8) + 0.5 * math.log(0.9))
    assert abs(no_damp.posterior_prob - manual) < 1e-12 and post.posterior_prob < no_damp.posterior_prob
    assert compute_posterior_fair_value(0.42, []).posterior_prob == 0.42
    print(f"Bayes: prior 0.30 -> {post.posterior_prob:.4f} (undamped {no_damp.posterior_prob:.4f}), "
          f"fair odds {post.fair_decimal_odds:.3f}")

    # --- microstructure prior
    snap_ms = OrderbookSnapshot("t", "c", 0.40, 0.44, 0.04, 0.42, 0.44, 0.45, 0.46, 0, .01, .02,
                                8000, 2000, 0.6)
    mp = microstructure_prior(snap_ms, None, volume_velocity=2.0)
    assert mp and 0.42 < mp.prior_prob <= 0.44
    print(f"Microstructure prior: {mp.prior_prob:.4f} (mid {mp.mid}), confidence {mp.confidence:.3f}")

    # --- intra-market arb
    m_arb = MarketRecord("0xarb", "Binary arb?", outcomes=["Yes", "No"], token_ids=["ya", "na"])
    snaps = {
        "ya": OrderbookSnapshot("ya", "0xarb", 0.45, 0.47, .02, .46, 0.47, 0.475, 0.48, 0, .005, .01, 5000, 5000, 0),
        "na": OrderbookSnapshot("na", "0xarb", 0.48, 0.50, .02, .49, 0.50, 0.505, 0.51, 0, .005, .01, 5000, 5000, 0),
    }
    arbs = scan_arbitrage_opportunities(cfg, [m_arb], snaps, outlay_usd=500.0)
    assert arbs and arbs[0].kind == "intra_market"
    a = arbs[0]
    assert abs(a.legs[0].payout_usd - a.legs[1].payout_usd) < 1e-9
    print(f"Arb: sum={a.implied_sum:.3f} roi={a.roi:.4%} stakes={[round(l.stake_usd, 2) for l in a.legs]} "
          f"profit=${a.guaranteed_profit_usd:.2f}")

    # --- neg-risk event: only scanned when the event's full market set is present
    nr = [MarketRecord(f"0xn{i}", f"Will C{i} win?", event_slug="race", outcomes=["Yes", "No"],
                       token_ids=[f"ny{i}", f"nn{i}"], neg_risk=True) for i in range(3)]
    nr_snaps = {f"ny{i}": OrderbookSnapshot(f"ny{i}", f"0xn{i}", .28, .30, .02, .29, .30, .30, .30, 0, 0, 0,
                                            9000, 9000, 0) for i in range(3)}
    assert scan_arbitrage_opportunities(cfg, nr, nr_snaps, event_market_counts={"race": 3})[0].kind == "neg_risk_event"
    assert not scan_arbitrage_opportunities(cfg, nr, nr_snaps, event_market_counts={"race": 5})
    assert not scan_arbitrage_opportunities(cfg, nr, nr_snaps)
    lo = OrderbookSnapshot("t", "c", 0.001, 0.003, .002, .002, .003, .003, .003, 0, 0, 0, 500, 500, 0)
    assert microstructure_prior(lo).prior_prob <= 0.003  # floor never lifts prior above the ask

    # --- cross-venue arb (Polymarket Lakers @0.40 vs Pinnacle Celtics @1.80)
    ev2 = SharpEvent("e2", "basketball_nba", None, "Los Angeles Lakers", "Boston Celtics",
                     {"pinnacle": {"Los Angeles Lakers": 2.20, "Boston Celtics": 1.80}},
                     {"Los Angeles Lakers": 2.20, "Boston Celtics": 1.80}, "pinnacle")
    m2 = MarketRecord("0xnba", "Lakers vs. Celtics", outcomes=["Lakers", "Celtics"], token_ids=["l", "c"])
    pair2 = MatchedPair(m2, ev2, 0.85, {"Lakers": "Los Angeles Lakers", "Celtics": "Boston Celtics"},
                        {"Lakers": "l", "Celtics": "c"}, [1.80, 2.20], ["Boston Celtics", "Los Angeles Lakers"])
    snaps2 = {"l": OrderbookSnapshot("l", "0xnba", .38, .40, .02, .39, .40, .405, .41, 0, .005, .01, 4000, 4000, 0)}
    cross = scan_arbitrage_opportunities(cfg, [], snaps2, [pair2])
    assert cross and cross[0].kind == "cross_venue"
    print(f"Cross-venue arb: roi={cross[0].roi:.4%} legs={[(l.venue, l.outcome, round(l.stake_usd, 2)) for l in cross[0].legs]}")

    # --- high-conviction (+100% ROI) EV with Kelly
    m_hc = MarketRecord("0xhc", "Will candidate X win?", outcomes=["Yes", "No"], token_ids=["hy", "hn"],
                        event_slug="election-x")
    snap_hc = OrderbookSnapshot("hy", "0xhc", 0.16, 0.18, .02, .17, 0.18, 0.185, 0.19, 0, .005, .01, 6000, 3000, -0.3)
    post_hc = compute_posterior_fair_value(0.22, [2.5, 1.6], [0.8, 0.6])
    ev_hc = evaluate_market_ev(cfg, m_hc, "Yes", snap_hc, post_hc, "microstructure", [2.5, 1.6], [0.8, 0.6],
                               resolution_risk=3, rationale="Primary-source catalyst")
    assert ev_hc is not None
    r = ev_hc.record
    expected_k = min(cfg.strategy.kelly_fraction * (r.posterior_prob - r.entry_vwap) / (1 - r.entry_vwap),
                     cfg.strategy.max_stake_pct_per_market)
    assert r.expected_roi >= 1.0 and r.signal_type == "high_conviction"
    assert abs(r.kelly_stake_pct - min(expected_k, ev_hc.clip_tier / cfg.strategy.bankroll_usd)) < 1e-12
    print(f"High conviction: P_true={r.posterior_prob:.4f} entry={r.entry_vwap} ROI={r.expected_roi:.2%} "
          f"edge={r.edge_pct:.4f} kelly={r.kelly_stake_pct:.4f} stake=${ev_hc.stake_usd:.2f} tier={ev_hc.clip_tier}")

    # --- portfolio scaling
    many = []
    for i in range(10):
        mk = MarketRecord(f"0x{i}", "q", event_slug=f"ev{i % 4}", outcomes=["Yes", "No"], token_ids=[f"t{i}", "x"])
        e = evaluate_market_ev(cfg, mk, "Yes", snap_hc, post_hc, "test", [2.5], [0.8], resolution_risk=3)
        many.append(e)
    alloc = allocate_portfolio_kelly(cfg, many)
    total = sum(e.record.kelly_stake_pct for e in alloc)
    assert total <= cfg.strategy.max_portfolio_exposure_pct + 1e-12
    print(f"Portfolio: {len(alloc)} bets, total exposure {total:.4f} (cap {cfg.strategy.max_portfolio_exposure_pct})")
    import asyncio
    from pathlib import Path
    from config_and_database import DatabaseManager

    async def _persist() -> Dict[str, Any]:
        path = Path("/tmp/pm_quant_selftest.sqlite3")
        if path.exists():
            path.unlink()
        async with DatabaseManager(path) as db:
            await db.insert_fair_value(r)
            return await db.latest_fair_value("hy")
    stored = asyncio.run(_persist())
    assert stored and stored["signal_type"] == "high_conviction"
    print("FairValueRecord persisted via db.insert_fair_value():", stored["signal_type"],
          round(stored["expected_roi"], 4))
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()
