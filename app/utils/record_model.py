"""2026 season projection model — team record & expected finish.

Approach (transparent, explainable):

1. Predicted offensive PPG from the roster.  Every skill player gets a
   regressed 2025 per-game PPR value (small samples pulled toward positional
   replacement).  Each team's projected starters — read from the updated
   post-offseason depth chart, so **acquisitions and losses are baked in** —
   are combined into an offensive "roster index".  A regression fit on 2025
   (actual offensive PPG vs the same index) maps that index to a predicted
   2026 offensive PPG.

2. Predicted defensive PPG allowed, built the same way from the defensive
   roster.  Each defender gets a regressed 2025 per-game value from a composite
   box-score stat (sacks, TFL, QB hits, INTs, pass breakups, forced fumbles,
   tackles).  Projected 2026 DL/LB/DB starters form a defensive index that a
   2025 regression maps to points allowed, so defensive acquisitions and losses
   move the projection too (teams with no index fall back to prior-year OPPG).

3. Power rating (points of margin vs a league-average team) = predicted net
   PPG, centered on the league.

4. Game model: P(home win) = Phi((power_home - power_away + HOME_ADV) / GAME_SD).

5. The 272-game slate is simulated 20,000 times for win-total distributions,
   division-title odds, playoff odds (4 division winners + 3 wild cards per
   conference, ties broken at random) and expected finish.
"""
from __future__ import annotations

from math import erf

import numpy as np
import pandas as pd

# ── Model constants ──────────────────────────────────────────────────────────
# Replacement-level per-game PPR by position (2025 starter distribution).
REPL_PG = {"QB": 13.6, "RB": 5.3, "WR": 6.2, "TE": 4.8}
REG_GAMES = 4.0          # games of replacement-level prior (low-sample shrink)
ROOKIE_FACTOR = 0.85     # unknown/rookie starter = fraction of replacement

# Per-game weight of each projected offensive starter in the roster index.
STARTER_WEIGHTS = {
    ("QB", 1): 1.00,
    ("RB", 1): 0.60, ("RB", 2): 0.30,
    ("WR", 1): 0.70, ("WR", 2): 0.50, ("WR", 3): 0.35,
    ("TE", 1): 0.45,
}
SKILL = ("QB", "RB", "WR", "TE")

OFF_REGRESS = 0.60       # roster-index offense regressed toward league mean.
                         # Walk-forward backtest (scripts/backtest_record_model.py)
                         # shows ~0.5-0.6 shrinkage of a team's scoring signal is
                         # optimal; previously offense was taken at full face value
                         # while defense was already regressed (asymmetric bias).
DEF_REGRESS = 0.60       # 2025 points-allowed regression toward league mean
                         # (fallback for teams with no defensive roster index)

# ── Defense (roster-based, mirrors offense) ─────────────────────────────────
DEF_BUCKETS = ("DL", "LB", "DB")
# Map granular 2025 stat positions into the same DL/LB/DB buckets the 2026
# depth chart uses, so the 2025 and 2026 indices are built on one basis.
DEF_POS_BUCKET = {
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "OLB": "LB", "MLB": "LB", "ILB": "LB", "EDGE": "LB",
    "CB": "DB", "SAF": "DB", "FS": "DB", "SS": "DB", "S": "DB",
    "DB": "DB", "NB": "DB",
}
# Composite per-play defensive value (sacks/turnovers weighted heaviest).
DEF_WEIGHTS = {
    "def_sacks": 2.0, "def_tackles_for_loss": 1.0, "def_qb_hits": 0.5,
    "def_interceptions": 3.5, "def_pass_defended": 1.0,
    "def_fumbles_forced": 2.5, "def_fumble_recovery_opp": 1.5,
    "def_tds": 6.0, "def_tackles_solo": 0.35, "def_tackle_assists": 0.15,
}
DEF_REPL_PG = {"DL": 1.1, "LB": 1.2, "DB": 1.2}   # replacement-level per game
DEF_ROOKIE_FACTOR = 0.85
# Per-game weight of each projected defensive starter in the roster index.
DEF_STARTER_WEIGHTS = {
    ("DL", 1): 0.55, ("DL", 2): 0.45, ("DL", 3): 0.35, ("DL", 4): 0.25,
    ("LB", 1): 0.50, ("LB", 2): 0.40, ("LB", 3): 0.25,
    ("DB", 1): 0.45, ("DB", 2): 0.40, ("DB", 3): 0.30, ("DB", 4): 0.25, ("DB", 5): 0.15,
}
DEF_DISP = {"DL": 4, "LB": 3, "DB": 5}   # depth cutoffs shown as "starters"

HOME_ADV = 1.6           # home-field advantage, points
GAME_SD = 13.2           # single-game point-margin standard deviation
N_SIMS = 20_000
GAMES = 17

# Blend the model power rating toward the market's implied rating.  Vegas win
# totals are the single most accurate publicly available preseason signal — the
# backtest shows any prior-year model tops out around 2.4 wins MAE, while good
# win-total models reach ~1.7-2.0 — so the market is the backbone and the roster
# model supplies the rest (and drives the offseason-impact narrative).
# 0 => ignore the market entirely.
MARKET_WEIGHT = 0.80

_SQRT2 = np.sqrt(2.0)


def _norm_cdf(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    flat = [0.5 * (1.0 + erf(float(v) / _SQRT2)) for v in arr.ravel()]
    return np.array(flat, dtype=np.float64).reshape(arr.shape)


def _norm_ppf(p) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    out = []
    for v in np.ravel(np.asarray(p, dtype=np.float64)):
        v = min(max(float(v), 1e-6), 1 - 1e-6)
        if v < 0.02425:
            q = np.sqrt(-2 * np.log(v))
            x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif v <= 0.97575:
            q = v - 0.5
            r = q*q
            x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        else:
            q = np.sqrt(-2 * np.log(1 - v))
            x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        out.append(x)
    return np.array(out, dtype=np.float64)


# ── Player value ─────────────────────────────────────────────────────────────
def _player_values(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player 2025 regular-season production, regressed to per-game value."""
    w = weekly[weekly["season"] == 2025]
    if "season_type" in w.columns:
        w = w[w["season_type"] == "REG"]
    w = w[w["position"].isin(SKILL)]
    g = w.groupby(["player_id", "player_display_name", "position", "recent_team"]).agg(
        gp=("week", "nunique"),
        ppr=("fantasy_points_ppr", "sum"),
    ).reset_index()
    repl = g["position"].map(REPL_PG).fillna(5.0)
    g["reg_pg"] = (g["ppr"] + repl * REG_GAMES) / (g["gp"] + REG_GAMES)
    return g


def _aggregate_by_id(players: pd.DataFrame) -> pd.DataFrame:
    """Collapse mid-season trades to one row per player (primary team = most games)."""
    primary = (players.sort_values("gp", ascending=False)
               .drop_duplicates("player_id")[["player_id", "recent_team"]]
               .rename(columns={"recent_team": "team_2025"}))
    agg = players.groupby("player_id").agg(
        player_display_name=("player_display_name", "first"),
        position=("position", "first"),
        gp=("gp", "sum"),
        ppr=("ppr", "sum"),
    ).reset_index().merge(primary, on="player_id", how="left")
    repl = agg["position"].map(REPL_PG).fillna(5.0)
    agg["reg_pg"] = (agg["ppr"] + repl * REG_GAMES) / (agg["gp"] + REG_GAMES)
    return agg.set_index("player_id")


def _team_index_2025(players: pd.DataFrame) -> dict:
    """Reconstruct each team's 2025 starters and sum their weighted per-game value."""
    out = {}
    for team, tdf in players.groupby("recent_team"):
        s = 0.0
        for pos in SKILL:
            ranked = tdf[tdf["position"] == pos].sort_values("ppr", ascending=False)
            for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
                s += STARTER_WEIGHTS.get((pos, rank), 0.0) * r["reg_pg"]
        out[team] = s
    return out


def _team_index_2026(depth: pd.DataFrame, pg_by_id: dict) -> tuple[dict, dict]:
    """Sum weighted per-game value of each team's projected 2026 starters.

    Returns (index_by_team, starter_rows) where starter_rows[team] lists the
    weighted contributors for transparency.
    """
    off = depth[(depth["side"] == "offense") & (depth["position"].isin(SKILL))]
    index, rows = {}, {}
    for team, tdf in off.groupby("team"):
        s = 0.0
        rlist = []
        for _, r in tdf.iterrows():
            pos, order = r["position"], int(r["depth_order"])
            w = STARTER_WEIGHTS.get((pos, order), 0.0)
            if w == 0.0:
                continue
            gid = r["gsis_id"]
            if pd.notna(gid) and gid in pg_by_id:
                pg = pg_by_id[gid]
            else:
                pg = REPL_PG.get(pos, 5.0) * ROOKIE_FACTOR
            s += w * pg
            rlist.append((r["player_name"], pos, order, pg, r["gsis_id"]))
        index[team] = s
        rows[team] = rlist
    return index, rows


def roster_changes(depth: pd.DataFrame, players: pd.DataFrame) -> dict:
    """Per-team acquisitions, losses and rookie starters vs 2025."""
    disp = {"QB": 2, "RB": 3, "WR": 4, "TE": 2}   # depth cutoffs shown as "starters"
    off = depth[(depth["side"] == "offense") & (depth["position"].isin(SKILL))]
    by_id = _aggregate_by_id(players)
    id_team_2025 = by_id["team_2025"].to_dict()
    changes = {}
    for team, tdf in off.groupby("team"):
        roster_ids = set(tdf["gsis_id"].dropna())
        adds, rookies = [], []
        for _, r in tdf.iterrows():
            if int(r["depth_order"]) > disp[r["position"]]:
                continue
            gid = r["gsis_id"]
            if pd.notna(gid) and gid in id_team_2025:
                if id_team_2025[gid] != team:
                    p = by_id.loc[gid]
                    adds.append({"player": r["player_name"], "pos": r["position"],
                                 "from": id_team_2025[gid], "pg": round(float(p["reg_pg"]), 1),
                                 "ppr": round(float(p["ppr"]), 1)})
            else:
                rookies.append({"player": r["player_name"], "pos": r["position"]})
        # Losses: 2025 contributors for this team no longer on the roster.
        prior = by_id[(by_id["team_2025"] == team) & (by_id["ppr"] >= 40)]
        losses = []
        for gid, p in prior.iterrows():
            if gid not in roster_ids:
                losses.append({"player": p["player_display_name"], "pos": p["position"],
                               "pg": round(float(p["reg_pg"]), 1),
                               "ppr": round(float(p["ppr"]), 1)})
        adds.sort(key=lambda x: -x["pg"])
        losses.sort(key=lambda x: -x["ppr"])
        changes[team] = {"adds": adds, "losses": losses, "rookies": rookies}
    return changes


# ── Defensive player value (mirror of the offense functions) ─────────────────
def _defender_values(weekly_def: pd.DataFrame) -> pd.DataFrame:
    """Per-defender 2025 composite value, regressed to a per-game figure."""
    w = weekly_def.copy()
    w["bucket"] = w["position"].map(DEF_POS_BUCKET)
    w = w.dropna(subset=["bucket"])
    val = sum(w[c].fillna(0) * wt for c, wt in DEF_WEIGHTS.items())
    w = w.assign(val=val)
    g = w.groupby(["player_id", "player_display_name", "bucket", "team"]).agg(
        gp=("week", "nunique"),
        val=("val", "sum"),
    ).reset_index()
    repl = g["bucket"].map(DEF_REPL_PG).fillna(1.2)
    g["reg_pg"] = (g["val"] + repl * REG_GAMES) / (g["gp"] + REG_GAMES)
    return g


def _aggregate_def_by_id(defenders: pd.DataFrame) -> pd.DataFrame:
    """One row per defender (primary team + bucket = most games)."""
    primary = (defenders.sort_values("gp", ascending=False)
               .drop_duplicates("player_id")[["player_id", "team", "bucket"]]
               .rename(columns={"team": "team_2025"}))
    agg = defenders.groupby("player_id").agg(
        player_display_name=("player_display_name", "first"),
        gp=("gp", "sum"),
        val=("val", "sum"),
    ).reset_index().merge(primary, on="player_id", how="left")
    repl = agg["bucket"].map(DEF_REPL_PG).fillna(1.2)
    agg["reg_pg"] = (agg["val"] + repl * REG_GAMES) / (agg["gp"] + REG_GAMES)
    return agg.set_index("player_id")


def _def_index_2025(defenders: pd.DataFrame) -> dict:
    """Each team's 2025 defensive starters, weighted per bucket, summed."""
    out = {}
    for team, tdf in defenders.groupby("team"):
        s = 0.0
        for bucket in DEF_BUCKETS:
            ranked = tdf[tdf["bucket"] == bucket].sort_values("val", ascending=False)
            for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
                s += DEF_STARTER_WEIGHTS.get((bucket, rank), 0.0) * r["reg_pg"]
        out[team] = s
    return out


def _def_index_2026(depth: pd.DataFrame, pg_by_id: dict) -> dict:
    """Sum weighted per-game value of each team's projected 2026 defensive starters."""
    dfn = depth[(depth["side"] == "defense") & (depth["position"].isin(DEF_BUCKETS))]
    index = {}
    for team, tdf in dfn.groupby("team"):
        s = 0.0
        for _, r in tdf.iterrows():
            bucket, order = r["position"], int(r["depth_order"])
            w = DEF_STARTER_WEIGHTS.get((bucket, order), 0.0)
            if w == 0.0:
                continue
            gid = r["gsis_id"]
            if pd.notna(gid) and gid in pg_by_id:
                pg = pg_by_id[gid]
            else:
                pg = DEF_REPL_PG.get(bucket, 1.2) * DEF_ROOKIE_FACTOR
            s += w * pg
        index[team] = s
    return index


def def_roster_changes(depth: pd.DataFrame, defenders: pd.DataFrame) -> dict:
    """Per-team defensive acquisitions, losses and new starters vs 2025."""
    dfn = depth[(depth["side"] == "defense") & (depth["position"].isin(DEF_BUCKETS))]
    by_id = _aggregate_def_by_id(defenders)
    id_team_2025 = by_id["team_2025"].to_dict()
    changes = {}
    for team, tdf in dfn.groupby("team"):
        roster_ids = set(tdf["gsis_id"].dropna())
        adds, rookies = [], []
        for _, r in tdf.iterrows():
            if int(r["depth_order"]) > DEF_DISP[r["position"]]:
                continue
            gid = r["gsis_id"]
            if pd.notna(gid) and gid in id_team_2025:
                if id_team_2025[gid] != team:
                    p = by_id.loc[gid]
                    adds.append({"player": r["player_name"], "pos": r["position"],
                                 "from": id_team_2025[gid], "pg": round(float(p["reg_pg"]), 1),
                                 "val": round(float(p["val"]), 1)})
            else:
                rookies.append({"player": r["player_name"], "pos": r["position"]})
        prior = by_id[(by_id["team_2025"] == team) & (by_id["val"] >= 20)]
        losses = []
        for gid, p in prior.iterrows():
            if gid not in roster_ids:
                losses.append({"player": p["player_display_name"], "pos": p["bucket"],
                               "pg": round(float(p["reg_pg"]), 1),
                               "val": round(float(p["val"]), 1)})
        adds.sort(key=lambda x: -x["pg"])
        losses.sort(key=lambda x: -x["val"])
        changes[team] = {"adds": adds, "losses": losses, "rookies": rookies}
    return changes


# ── Projection ───────────────────────────────────────────────────────────────
def _calibrate(idx25: dict, target: pd.Series, teams: pd.DataFrame):
    """Least-squares fit of a 2025 roster index against a per-team 2025 target.

    Returns (slope, intercept, r2).
    """
    cal = pd.DataFrame({"team": list(idx25.keys()),
                        "index": [idx25[t] for t in idx25]})
    cal = cal.merge(teams[["team"]].assign(y=target.values), on="team", how="inner").dropna()
    slope, intercept = np.polyfit(cal["index"].to_numpy(), cal["y"].to_numpy(), 1)
    fit = intercept + slope * cal["index"].to_numpy()
    ss_res = ((cal["y"].to_numpy() - fit) ** 2).sum()
    ss_tot = ((cal["y"].to_numpy() - cal["y"].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return float(slope), float(intercept), float(r2)


def build_team_projections(ratings, depth, divisions, weekly, weekly_def=None,
                           win_totals=None):
    """Return a per-team table with predicted PPG, power rating and change info."""
    teams = divisions[["team_abbr", "division", "conference"]].rename(
        columns={"team_abbr": "team"}).copy()
    r25 = ratings[ratings["season"] == 2025][["team", "ppg", "oppg"]]
    teams = teams.merge(r25, on="team", how="left")

    players = _player_values(weekly)
    pg_by_id = _aggregate_by_id(players)["reg_pg"].to_dict()

    idx25 = _team_index_2025(players)
    idx26, _ = _team_index_2026(depth, pg_by_id)

    # Calibrate offensive PPG from the 2025 roster index.
    slope, intercept, r2 = _calibrate(idx25, teams["ppg"], teams)

    teams["roster_index"] = teams["team"].map(idx26).fillna(np.nan)
    teams["index_2025"] = teams["team"].map(idx25)
    # Anchor offense on last year's ACTUAL scoring (regressed toward the mean),
    # then adjust for roster turnover — exactly how defense is handled below.
    # The backtest (scripts/backtest_record_model.py) shows prior-year point
    # differential is the most accurate prior signal; the calibrated roster
    # index only supplies the *delta* from acquisitions/losses, not the level.
    lg_off = teams["ppg"].mean()
    baseline_off = lg_off + OFF_REGRESS * (teams["ppg"] - lg_off)
    raw_off_delta = slope * (teams["roster_index"] - teams["index_2025"])
    # Roster turnover is zero-sum across the league, so center the shift.
    off_delta = (raw_off_delta - raw_off_delta.mean()).fillna(0.0)
    teams["proj_off_ppg"] = baseline_off + off_delta
    teams["off_ppg_2025"] = teams["ppg"]
    teams["off_change"] = off_delta

    # ── Defense ──────────────────────────────────────────────────────────
    # Defensive box stats only weakly predict points allowed (R^2 ~ 0.1), so
    # unlike offense we do NOT predict the absolute level from the roster.
    # Instead we anchor on last year's actual points allowed (regressed toward
    # the mean) and adjust ONLY for roster turnover: how much a team's
    # defensive index moved vs 2025, mapped through the calibration slope and
    # centered so offseason moves redistribute rather than inflate league-wide.
    lg_oppg = teams["oppg"].mean()
    baseline = lg_oppg + DEF_REGRESS * (teams["oppg"] - lg_oppg)
    def_cal = {}
    if weekly_def is not None and not weekly_def.empty:
        defenders = _defender_values(weekly_def)
        dpg_by_id = _aggregate_def_by_id(defenders)["reg_pg"].to_dict()
        didx25 = _def_index_2025(defenders)
        didx26 = _def_index_2026(depth, dpg_by_id)
        d_slope, d_intercept, d_r2 = _calibrate(didx25, teams["oppg"], teams)

        teams["def_index"] = teams["team"].map(didx26)
        teams["def_index_2025"] = teams["team"].map(didx25)
        raw_delta = d_slope * (teams["def_index"] - teams["def_index_2025"])
        # Center so the leaguewide adjustment nets to zero (removes any
        # roster-matching bias); missing indices get no adjustment.
        delta = (raw_delta - raw_delta.mean()).fillna(0.0)
        teams["def_change"] = delta
        teams["proj_def_ppg"] = baseline + delta
        def_cal = {"slope": d_slope, "intercept": d_intercept, "r2": d_r2}
    else:
        teams["proj_def_ppg"] = baseline
        teams["def_change"] = 0.0

    teams["def_ppg_2025"] = teams["oppg"]

    teams["proj_net"] = teams["proj_off_ppg"] - teams["proj_def_ppg"]
    model_power = teams["proj_net"] - teams["proj_net"].mean()
    teams["model_power"] = model_power

    # ── Anchor to the market ─────────────────────────────────────────────
    # Convert each team's win total to an implied power rating by inverting the
    # game model against an average opponent (E[wins] ~ GAMES * Phi(power/SD)),
    # then blend it with the roster model.  The off/def PPG columns stay as the
    # roster story; only the power rating that drives the simulation is blended.
    market_used = False
    if win_totals is not None and not win_totals.empty and MARKET_WEIGHT > 0:
        wt = win_totals.rename(columns={c: c.lower() for c in win_totals.columns})
        wt_map = dict(zip(wt["team"].astype(str),
                          pd.to_numeric(wt["win_total"], errors="coerce")))
        wins_arr = np.array([wt_map.get(str(t), np.nan) for t in teams["team"]],
                            dtype=float)
        have = ~np.isnan(wins_arr)
        mp = np.asarray(model_power, dtype=float)
        implied = np.full(len(teams), np.nan, dtype=float)
        if have.any():
            frac = np.clip(wins_arr[have] / GAMES, 0.03, 0.97)
            imp = GAME_SD * _norm_ppf(frac)
            implied[have] = imp - imp.mean()
        blended = np.where(have,
                           MARKET_WEIGHT * np.nan_to_num(implied) + (1 - MARKET_WEIGHT) * mp,
                           mp)
        teams["power"] = blended - blended.mean()
        teams["market_power"] = implied
        market_used = True
    else:
        teams["power"] = model_power

    teams.attrs["calibration"] = {"slope": slope, "intercept": intercept, "r2": r2}
    teams.attrs["def_calibration"] = def_cal
    teams.attrs["market"] = {"used": market_used, "weight": MARKET_WEIGHT}
    return teams


# Schedule feeds occasionally use different abbreviations than our ratings/
# divisions (e.g. the Rams are "LA" on the schedule but "LAR" everywhere else),
# which would silently drop that team's games and give it a 0-win projection.
_SCHED_TEAM_ALIAS = {"LA": "LAR", "JAC": "JAX", "STL": "LAR", "SD": "LAC", "OAK": "LV"}


def _game_probs(schedule, power):
    games = schedule[(schedule["season"] == 2026) & (schedule["game_type"] == "REG")][
        ["week", "home_team", "away_team"]].copy()
    games["home_team"] = games["home_team"].replace(_SCHED_TEAM_ALIAS)
    games["away_team"] = games["away_team"].replace(_SCHED_TEAM_ALIAS)
    pmap = power.set_index("team")["power"]
    games = games[games["home_team"].isin(pmap.index) & games["away_team"].isin(pmap.index)]
    margin = games["home_team"].map(pmap) - games["away_team"].map(pmap) + HOME_ADV
    games["p_home"] = _norm_cdf((margin / GAME_SD).to_numpy())
    return games.reset_index(drop=True)


def project_season(ratings, depth, divisions, schedule, weekly, weekly_def=None,
                   win_totals=None, n_sims=N_SIMS, seed=7):
    """Full projection.

    Returns (team_table, games_table, changes):
      team_table adds proj_off_ppg / proj_def_ppg / off_change / def_change on
      top of proj_wins, playoff_pct, div_title_pct, exp_finish, win_dist, power.
      changes: dict team -> {"off": {...}, "def": {...}}.
    """
    power = build_team_projections(ratings, depth, divisions, weekly, weekly_def,
                                   win_totals=win_totals)
    cal = power.attrs.get("calibration", {})
    def_cal = power.attrs.get("def_calibration", {})
    market = power.attrs.get("market", {})
    games = _game_probs(schedule, power)

    off_changes = roster_changes(depth, _player_values(weekly))
    if weekly_def is not None and not weekly_def.empty:
        def_changes = def_roster_changes(depth, _defender_values(weekly_def))
    else:
        def_changes = {}
    changes = {t: {"off": off_changes.get(t, {"adds": [], "losses": [], "rookies": []}),
                   "def": def_changes.get(t, {"adds": [], "losses": [], "rookies": []})}
               for t in power["team"]}

    team_list = power["team"].tolist()
    idx = {t: i for i, t in enumerate(team_list)}
    n_teams = int(len(team_list))
    n_sims = int(n_sims)

    home_i = np.asarray(games["home_team"].map(idx).to_numpy(), dtype=np.int64)
    away_i = np.asarray(games["away_team"].map(idx).to_numpy(), dtype=np.int64)
    p_home = np.asarray(games["p_home"].to_numpy(), dtype=np.float64)
    n_games = int(len(p_home))

    exp_wins = np.zeros(n_teams, dtype=np.float64)
    np.add.at(exp_wins, home_i, p_home)
    np.add.at(exp_wins, away_i, 1.0 - p_home)

    rng = np.random.default_rng(int(seed))
    wins = np.zeros((n_sims, n_teams), dtype=np.int32)
    for g in range(n_games):
        # Scalar-size RNG per game keeps memory low and sidesteps any
        # size-tuple dtype quirks in newer numpy Generator.random.
        hw = rng.random(n_sims) < float(p_home[g])
        hi = int(home_i[g])
        ai = int(away_i[g])
        wins[hw, hi] += 1
        wins[~hw, ai] += 1

    noise = rng.random((n_sims, n_teams)) * 1e-3
    keyed = wins + noise

    div_of = power.set_index("team")["division"].to_dict()
    conf_of = power.set_index("team")["conference"].to_dict()

    exp_finish = np.zeros(n_teams)
    div_title = np.zeros(n_teams)
    playoff = np.zeros(n_teams)
    div_winner_mask = np.zeros((n_sims, n_teams), dtype=bool)

    for _div, members in _group(team_list, div_of).items():
        cols = [idx[t] for t in members]
        sub = keyed[:, cols]
        ranks = (sub[:, None, :] > sub[:, :, None]).sum(axis=2) + 1
        for k, t in enumerate(members):
            exp_finish[idx[t]] = ranks[:, k].mean()
            is_winner = ranks[:, k] == 1
            div_title[idx[t]] = is_winner.mean()
            div_winner_mask[is_winner, idx[t]] = True

    for _conf, members in _group(team_list, conf_of).items():
        cols = np.array([idx[t] for t in members])
        w_conf = keyed[:, cols]
        winners = div_winner_mask[:, cols]
        masked = np.where(winners, -1.0, w_conf)
        order = np.argsort(-masked, axis=1)
        wildcard = np.zeros_like(winners)
        rows = np.arange(w_conf.shape[0])[:, None]
        wildcard[rows, order[:, :3]] = True
        in_playoffs = winners | wildcard
        for k, t in enumerate(members):
            playoff[idx[t]] = in_playoffs[:, k].mean()

    win_dist = [np.bincount(np.asarray(wins[:, i], dtype=np.int64),
                            minlength=GAMES + 1) / n_sims
                for i in range(n_teams)]

    table = power[["team", "division", "conference", "power", "roster_index",
                   "proj_off_ppg", "proj_def_ppg", "off_ppg_2025", "off_change",
                   "def_ppg_2025", "def_change"]].copy()
    table["proj_wins"] = exp_wins.round(1)
    table["proj_losses"] = (GAMES - exp_wins).round(1)
    table["playoff_pct"] = playoff * 100
    table["div_title_pct"] = div_title * 100
    table["exp_finish"] = exp_finish.round(2)
    table["win_dist"] = win_dist
    table = table.sort_values("proj_wins", ascending=False).reset_index(drop=True)
    table.attrs["calibration"] = cal
    table.attrs["def_calibration"] = def_cal
    table.attrs["market"] = market
    return table, games, changes


def _group(team_list, mapping):
    out = {}
    for t in team_list:
        out.setdefault(mapping[t], []).append(t)
    return out
