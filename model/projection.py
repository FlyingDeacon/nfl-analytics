"""Pure fantasy-points projection engine (no Streamlit, no module-level globals).

Extracted from app/pages/7_Fantasy_Predictions.py so the same computation can be
driven from the Streamlit page (as_of_season=None, uses all available data) and
from scripts/backtest_model.py (as_of_season=S, walk-forward: trains only on
season pairs < S and projects season S off season-S-1 features).

Position-specific 2026 roster overlays (expert team corrections, injury removals,
player multipliers, age curve, force-included starters, VOR replacement levels)
are NOT part of this module — they are hand-curated for the current season and
don't apply retroactively to historical holdout seasons in the backtest. They
stay in the Streamlit page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Default model constants (single source of truth — imported by both the
# Streamlit page and scripts/backtest_model.py so they can never drift apart).
# ══════════════════════════════════════════════════════════════════════════════

NFL_GAMES = 17

# Per-game features only — normalises out "played more games = more counting stats".
# game_rate (games/NFL_GAMES) is always appended and captures injury-proneness / role depth.
POSITION_FEATURES: dict[str, list[str]] = {
    "QB": ["passing_yards", "passing_tds", "interceptions",
           "rushing_yards", "rushing_tds", "completions", "attempts"],
    "RB": ["rushing_yards", "rushing_tds", "carries",
           "receptions", "receiving_yards", "receiving_tds", "targets"],
    "WR": ["receiving_yards", "receiving_tds", "targets", "receptions", "rushing_yards"],
    "TE": ["receiving_yards", "receiving_tds", "targets", "receptions"],
}

# QBs need 12 full games to qualify as a genuine starter (not a backup fill-in).
# Skill positions use 6 — role players and injury-return candidates are valid.
MIN_GAMES_BY_POS: dict[str, int] = {"QB": 12, "RB": 6, "WR": 6, "TE": 6}

# Ridge penalty prevents wild extrapolation from small samples.
RIDGE_ALPHA = 4.0
# Per-year recency decay: the most recent training pair is weighted highest.
DECAY = 0.35
# Prior-year PPG blend weight (RB/WR/TE only — QB uses its own weighted-PPG path).
PPG_BLEND_WEIGHT = 0.45

# Backtest-calibrated expected-games used as the PER-POSITION CENTRE of the
# player-specific availability model below (a perfectly average-availability
# player projects at this many games).
#
# Recalibrated against the 2022-2025 walk-forward backtest: the prior 15/15/15/16
# centres over-projected season totals by ~+28 pts on average (training pairs only
# ever include players who survived into season N+1, so the fit skews to the
# durable). Scaling the skill-position centres ~0.90 and pulling QB down to 13
# (its +44 bias was the worst — nearly a full extra game of counting stats) cut
# the overall bias to ~+21 and QB's to ~+10, improved MAE and top-50 hit rate,
# and left the rank correlation (spearman) essentially unchanged.
PPG_BASELINE_GAMES: dict[str, float] = {"QB": 13, "RB": 13.5, "WR": 13.5, "TE": 14.4}

# ── Player-specific availability (expected games) ─────────────────────────────
# Replaces the flat positional games constant with a per-player estimate:
# recency-weighted recent-season games, regressed toward the positional
# population mean. Centred on that mean so the aggregate scale (and the
# backtest-calibrated totals) is preserved — points are only REDISTRIBUTED from
# fragile players to durable ones, not inflated or deflated in aggregate.
# SHRINK < 1 because a single injury season is a noisy signal (research: a
# player who missed time has <50% chance of a clean next season, vs ~62% for a
# player coming off a full season — real but far from deterministic).
AVAIL_SHRINK = 0.35                       # conservative damping of one-season availability swings
AVAIL_GAMES_FLOOR = 10.0                  # projected starters don't fall below ~10 games
AVAIL_GAMES_CEILING = 17.0                # full season
AVAIL_RECENCY_WEIGHTS = (0.5, 0.3, 0.2)   # most-recent season weighted highest


@dataclass
class ProjectionConfig:
    """Explicit config for build_predictions_core — no reads from module globals."""
    target_col: str
    name_col: str
    pos_col: str
    team_col: str | None
    track_col: str
    position_features: dict[str, list[str]] = field(default_factory=lambda: dict(POSITION_FEATURES))
    min_games_by_pos: dict[str, int] = field(default_factory=lambda: dict(MIN_GAMES_BY_POS))
    ridge_alpha: float = RIDGE_ALPHA
    decay: float = DECAY
    ppg_blend_weight: float = PPG_BLEND_WEIGHT
    ppg_baseline_games: dict[str, int] = field(default_factory=lambda: dict(PPG_BASELINE_GAMES))
    nfl_games: int = NFL_GAMES
    avail_shrink: float = AVAIL_SHRINK
    avail_floor: float = AVAIL_GAMES_FLOOR
    avail_ceiling: float = AVAIL_GAMES_CEILING
    avail_recency_w: tuple[float, ...] = AVAIL_RECENCY_WEIGHTS


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE REGRESSION (pure numpy — no sklearn dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float,
               weights: np.ndarray | None = None):
    """Weighted ridge regression via the regularised normal equation."""
    n = X.shape[0]
    Xb = np.column_stack([np.ones(n), X])
    if weights is not None:
        w = weights / weights.sum() * n      # normalise so sum = n
        W = np.diag(w)
        XtW = Xb.T @ W
        A, bv = XtW @ Xb, XtW @ y
    else:
        A, bv = Xb.T @ Xb, Xb.T @ y
    reg = np.eye(A.shape[0]) * alpha
    reg[0, 0] = 0                           # do not regularise the intercept
    theta = np.linalg.solve(A + reg, bv)
    yp = Xb @ theta
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    return theta[1:], theta[0], r2, rmse


# ══════════════════════════════════════════════════════════════════════════════
# AVAILABILITY (expected games)
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_durability(games_by_season: dict[int, float],
                          recency_w: tuple[float, ...]) -> float:
    """Recency-weighted average games played over a player's most recent seasons.

    Uses ALL of a player's seasons (including sub-starter, injury-shortened ones)
    so a missed year actually pulls the estimate down. Returns NaN for players
    with no history (they fall back to the positional baseline).
    """
    if not games_by_season:
        return float("nan")
    seasons = sorted(games_by_season, reverse=True)[:len(recency_w)]
    ws  = recency_w[:len(seasons)]
    num = sum(w * games_by_season[s] for w, s in zip(ws, seasons))
    den = sum(ws)
    return num / den if den else float("nan")


def _expected_games(durability: float, pop_mean: float, base_games: float,
                     shrink: float, floor: float, ceiling: float) -> float:
    """Player-specific expected games, regressed toward the positional mean.

    A player whose availability equals the population mean maps exactly to
    base_games, so the cohort mean — and thus the backtest-calibrated aggregate
    scale — is preserved. `shrink` damps how far one season's availability moves
    a player off that centre; the result is clipped to a sane [floor, ceiling].
    """
    if durability is None or not np.isfinite(durability):
        return float(base_games)
    return float(np.clip(base_games + shrink * (durability - pop_mean), floor, ceiling))


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_predictions_core(weekly_df: pd.DataFrame, config: ProjectionConfig,
                            as_of_season: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project season-N+1 fantasy points from season-N per-game features.

    as_of_season=None: use every season present in weekly_df (the live-app call —
        matches the original build_predictions() behaviour exactly).
    as_of_season=S: restrict all season aggregates to season < S before deriving
        "latest season" and training pairs. This makes the "latest" observed
        season S-1, so training only ever sees pairs strictly before S and the
        function projects season S off S-1 features — walk-forward, no leakage.
    """
    name_col, pos_col, team_col, track_col = (
        config.name_col, config.pos_col, config.team_col, config.track_col
    )
    target_col         = config.target_col
    position_features  = config.position_features
    min_games_by_pos    = config.min_games_by_pos
    nfl_games           = config.nfl_games
    decay               = config.decay
    ppg_blend_weight    = config.ppg_blend_weight
    ppg_baseline_games  = config.ppg_baseline_games
    ridge_alpha         = config.ridge_alpha
    avail_shrink        = config.avail_shrink
    avail_floor         = config.avail_floor
    avail_ceiling       = config.avail_ceiling
    avail_recency_w     = config.avail_recency_w

    # ── Regular season only ──────────────────────────────────────────────────
    reg = weekly_df.copy()
    if "season_type" in reg.columns:
        reg = reg[reg["season_type"] == "REG"]
    reg = reg[reg[pos_col].isin(position_features)]

    # ── Season aggregates ────────────────────────────────────────────────────
    all_feat_cols = list({c for feats in position_features.values() for c in feats})
    stat_cols     = [target_col] + [c for c in all_feat_cols if c in reg.columns]
    group_keys    = [track_col, name_col, pos_col, "season"]
    if team_col:
        group_keys.append(team_col)

    agg = reg.groupby(group_keys, as_index=False)[stat_cols].sum()
    gp  = reg.groupby(group_keys, as_index=False)[target_col].count()
    gp.rename(columns={target_col: "games"}, inplace=True)
    agg = agg.merge(gp, on=group_keys, how="left")

    # Per-game rates (the core features the model trains on)
    for c in stat_cols:
        if c in agg.columns:
            agg[f"{c}_pg"] = (agg[c] / agg["games"].clip(lower=1)).round(4)
    # game_rate encodes starter reliability / injury history
    agg["game_rate"] = (agg["games"] / nfl_games).clip(upper=1.0).round(4)

    if as_of_season is not None:
        agg = agg[agg["season"] < as_of_season]

    all_seasons = sorted(agg["season"].dropna().unique().astype(int))
    if not all_seasons:
        return pd.DataFrame(), pd.DataFrame()
    latest_szn = all_seasons[-1]

    predictions_list = []

    for pos, base_feats in position_features.items():
        min_g    = min_games_by_pos[pos]
        pg_feats = [f"{c}_pg" for c in base_feats if f"{c}_pg" in agg.columns] + ["game_rate"]

        pos_df = agg[(agg[pos_col] == pos) & (agg["games"] >= min_g)].copy()
        if pos_df.empty:
            continue

        # ── Training: season-N per-game features → season-N+1 total points ──
        train_X, train_y, train_w = [], [], []
        for szn in all_seasons[:-1]:
            nxt = szn + 1
            if nxt not in all_seasons:
                continue
            cur_d = pos_df[pos_df["season"] == szn].set_index(track_col)
            nxt_d = pos_df[pos_df["season"] == nxt].set_index(track_col)
            common = cur_d.index.intersection(nxt_d.index)
            # Recency weight: most-recent season pairs weighted highest
            w = float(np.exp(decay * (szn - (latest_szn - 1))))
            for pid in common:
                rc = (cur_d.loc[pid] if isinstance(cur_d.loc[pid], pd.Series)
                      else cur_d.loc[pid].iloc[-1])
                rn = (nxt_d.loc[pid] if isinstance(nxt_d.loc[pid], pd.Series)
                      else nxt_d.loc[pid].iloc[-1])
                train_X.append([float(rc.get(f, 0) or 0) for f in pg_feats])
                train_y.append(float(rn[target_col]))
                train_w.append(w)

        if len(train_X) < 20:
            continue

        Xtr = np.array(train_X, dtype=np.float64)
        ytr = np.array(train_y, dtype=np.float64)
        wtr = np.array(train_w, dtype=np.float64)

        # Standardise for numerical stability
        mu    = Xtr.mean(0)
        sigma = Xtr.std(0); sigma[sigma == 0] = 1.0
        coefs, intercept, r2, rmse = _ridge_fit((Xtr - mu) / sigma, ytr, alpha=ridge_alpha, weights=wtr)

        # ── Predict on latest available season ──────────────────────────────
        lat = pos_df[pos_df["season"] == latest_szn].copy()
        if lat.empty:
            continue

        Xlat     = (lat[pg_feats].fillna(0).values.astype(np.float64) - mu) / sigma
        raw_pred = np.clip(Xlat @ coefs + intercept, 0, None)

        games_lat  = lat["games"].values.astype(float)
        proj_games = np.full(len(lat), float(nfl_games))   # full season by default

        adj_factor = np.clip(
            1.0 + (proj_games / games_lat.clip(min=1) - 1.0) * 0.60,
            a_min=1.00, a_max=1.20
        )
        model_pts = np.clip(raw_pred * adj_factor, 0, None)

        if pos == "QB":
            # Explicit season weights: most-recent=65%, 2nd-most-recent=25%, 3rd=10%.
            # Backtest-validated best (softening it hurts QB rank accuracy — recent
            # QB performance predicts better than betting on bounce-backs).
            _szn_sorted = sorted(all_seasons)
            _n = len(_szn_sorted)
            _recency_w = {0: 0.65, 1: 0.25, 2: 0.10}
            _szn_weight = {
                s: _recency_w.get(_n - 1 - i, 0.0)
                for i, s in enumerate(_szn_sorted)
            }

            wtd_ppg = np.zeros(len(lat))
            wtd_sum = np.zeros(len(lat))
            for szn in all_seasons:
                szn_df = pos_df[pos_df["season"] == szn].set_index(track_col)
                w_szn  = _szn_weight.get(szn, 0.10)
                for idx, row in lat.iterrows():
                    pid = row[track_col]
                    if pid in szn_df.index:
                        ppg_szn = szn_df.loc[pid, f"{target_col}_pg"] if isinstance(
                            szn_df.loc[pid], pd.Series) else szn_df.loc[pid].iloc[-1][f"{target_col}_pg"]
                        if ppg_szn and ppg_szn > 0:
                            lat_idx = lat.index.get_loc(idx)
                            wtd_ppg[lat_idx] += float(ppg_szn) * w_szn
                            wtd_sum[lat_idx] += w_szn
            wtd_sum = np.where(wtd_sum == 0, 1.0, wtd_sum)
            qb_weighted_ppg = wtd_ppg / wtd_sum
            fallback_ppg = lat[f"{target_col}_pg"].fillna(0).values
            qb_weighted_ppg = np.where(qb_weighted_ppg > 0, qb_weighted_ppg, fallback_ppg)
            qb_baseline_games = float(ppg_baseline_games.get("QB", nfl_games))
            pred_pts = np.clip(qb_weighted_ppg * qb_baseline_games, 0, None)
        else:
            # ── Prior-year PPG baseline blend for RB/WR/TE ───────────────────
            prior_ppg    = lat[f"{target_col}_pg"].fillna(0).values
            ppg_proj_g   = float(ppg_baseline_games.get(pos, nfl_games))
            ppg_baseline = np.clip(prior_ppg * ppg_proj_g, 0, None)
            pred_pts = (model_pts * (1.0 - ppg_blend_weight)
                        + ppg_baseline * ppg_blend_weight)

        # ── Player-specific availability (expected games) ────────────────────
        # pred_pts above is a full-season-equivalent total built on the flat
        # positional games centre (base_g). Back out the implied per-game rate,
        # then re-scale by each player's OWN expected games so durable players
        # rise and fragile ones fall — without changing the cohort's aggregate
        # scale (expected games is centred on the positional population mean).
        base_g = float(ppg_baseline_games.get(pos, nfl_games))
        active_ppg = pred_pts / base_g if base_g > 0 else pred_pts

        games_by_pid: dict = {}
        pos_all = agg[agg[pos_col] == pos]
        for pid, szn, g in zip(pos_all[track_col], pos_all["season"], pos_all["games"]):
            games_by_pid.setdefault(pid, {})[int(szn)] = float(g)
        durab = {pid: _weighted_durability(h, avail_recency_w)
                 for pid, h in games_by_pid.items()}
        lat_durab = np.array([durab.get(pid, np.nan) for pid in lat[track_col]], dtype=float)
        pop_mean = float(np.nanmean(lat_durab)) if np.isfinite(lat_durab).any() else base_g
        exp_games = np.array([
            _expected_games(d, pop_mean, base_g, avail_shrink, avail_floor, avail_ceiling)
            for d in lat_durab
        ], dtype=float)
        pred_pts = np.clip(active_ppg * exp_games, 0, None)

        lat = lat.copy()
        lat["predicted_pts"] = pred_pts.round(1)
        lat["proj_games"]    = np.round(exp_games, 1)
        lat["pred_ppg"]      = np.round(active_ppg, 2)
        lat["rmse"]          = round(rmse, 1)

        # ── Auto injury-risk flag (QB only): avg games/yr < 14.5 over last 3 seasons.
        # Hand-curated injury overlays (INJURY_RISK_MAP) are NOT applied here — that
        # dict is 2026-specific roster intelligence and stays in the Streamlit page,
        # which ORs it into this auto-flag after calling build_predictions_core.
        if pos == "QB":
            avg_g_map: dict = {}
            for szn in all_seasons[-3:]:
                szn_g = pos_df[pos_df["season"] == szn].set_index(track_col)["games"]
                for pid, g in szn_g.items():
                    avg_g_map.setdefault(pid, []).append(float(g))
            lat["injury_risk"] = lat.apply(
                lambda r: "      Yes      " if (
                    len(avg_g_map.get(r[track_col], [])) > 0 and
                    sum(avg_g_map.get(r[track_col], [17])) /
                    len(avg_g_map.get(r[track_col], [17])) < 14.5
                ) else "",
                axis=1
            )
        else:
            lat["injury_risk"] = ""

        predictions_list.append(lat)

    if not predictions_list:
        return pd.DataFrame(), pd.DataFrame()

    all_preds = pd.concat(predictions_list, ignore_index=True)

    # Historical per-season totals for the trajectory line chart (include games for PPG tooltip)
    hist = (agg.groupby([track_col, name_col, pos_col, "season"], as_index=False)
               [[target_col, "games"]].sum())
    if team_col and team_col in agg.columns:
        tl = agg.groupby([track_col, "season"], as_index=False)[team_col].first()
        hist = hist.merge(tl, on=[track_col, "season"], how="left")

    return all_preds, hist


# ══════════════════════════════════════════════════════════════════════════════
# VALUE OVER REPLACEMENT — dynamic, roster-slot-derived baseline.
#
# Not wired into the live app in this commit (the app still uses the static
# REPLACEMENT_LEVEL dict — that's Phase 2). Used here only so the backtest can
# score a holdout season's ACTUAL VOR against a fair, self-derived baseline
# instead of a hardcoded constant. Phase 2 will point the live app at this same
# function for PROJECTED VOR so the logic is never duplicated.
# ══════════════════════════════════════════════════════════════════════════════

def derive_replacement_baseline(
    points_df: pd.DataFrame, pos_col: str, points_col: str,
    roster_slots: dict[str, int], flex_eligible: set[str], league_size: int,
    name_col: str | None = None, max_iter: int = 3,
) -> tuple[dict[str, float], dict[str, list[tuple]]]:
    """Derive a per-position replacement points baseline from a ranked player pool.

    starters[pos] = league_size * roster_slots[pos]; players ranked below their
    positional cutoff who are flex-eligible are pooled and sorted by points; the
    top (league_size * roster_slots["FLEX"]) fill flex and are added back into
    starters[pos]; repeat until stable. Baseline = median points of the 3 players
    immediately below the final cutoff (the realistic waiver-wire replacement).

    Returns (baseline, debug) where debug[pos] lists the (name, points) — or just
    (points,) if name_col is None — of the players the baseline was derived from,
    for sanity-checking that the source players are plausible.
    """
    positions = [p for p in roster_slots if p != "FLEX"]
    starters = {p: league_size * roster_slots[p] for p in positions}
    flex_slots = league_size * roster_slots.get("FLEX", 0)

    ranked = {
        p: points_df[points_df[pos_col] == p]
                    .sort_values(points_col, ascending=False)
                    .reset_index(drop=True)
        for p in positions
    }

    for _ in range(max_iter):
        flex_pool = []
        for p in positions:
            if p not in flex_eligible:
                continue
            for _, row in ranked[p].iloc[starters[p]:].iterrows():
                flex_pool.append((p, float(row[points_col])))
        flex_pool.sort(key=lambda t: t[1], reverse=True)

        new_starters = {p: league_size * roster_slots[p] for p in positions}
        for p, _ in flex_pool[:flex_slots]:
            new_starters[p] += 1

        if new_starters == starters:
            break
        starters = new_starters

    baseline: dict[str, float] = {}
    debug: dict[str, list[tuple]] = {}
    for p in positions:
        n = starters[p]
        window = ranked[p].iloc[n:n + 3]
        if not window.empty:
            baseline[p] = float(window[points_col].median())
        elif not ranked[p].empty:
            baseline[p] = float(ranked[p].iloc[-1][points_col])
        else:
            baseline[p] = 0.0
        if name_col and name_col in window.columns:
            debug[p] = list(zip(window[name_col].tolist(), window[points_col].round(1).tolist()))
        else:
            debug[p] = list(zip(window[points_col].round(1).tolist()))
    return baseline, debug
