"""2026 season projection model — team record & expected finish.

Approach (transparent, explainable):

1. Predicted offensive PPG from the roster.  Every skill player gets a
   regressed 2025 per-game PPR value (small samples pulled toward positional
   replacement).  Each team's projected starters — read from the updated
   post-offseason depth chart, so **acquisitions and losses are baked in** —
   are combined into an offensive "roster index".  A regression fit on 2025
   (actual offensive PPG vs the same index) maps that index to a predicted
   2026 offensive PPG.

2. Predicted defensive PPG allowed = league mean regressed from 2025 (player
   data here is offense-only, so defense leans on prior-year results).

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

DEF_REGRESS = 0.60       # 2025 points-allowed regression toward league mean
HOME_ADV = 1.6           # home-field advantage, points
GAME_SD = 13.2           # single-game point-margin standard deviation
N_SIMS = 20_000
GAMES = 17

_SQRT2 = np.sqrt(2.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / _SQRT2)))(x)


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


# ── Projection ───────────────────────────────────────────────────────────────
def build_team_projections(ratings, depth, divisions, weekly):
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
    cal = pd.DataFrame({
        "team": list(idx25.keys()),
        "index": [idx25[t] for t in idx25],
    }).merge(teams[["team", "ppg"]], on="team", how="inner").dropna()
    slope, intercept = np.polyfit(cal["index"].to_numpy(), cal["ppg"].to_numpy(), 1)
    fit = intercept + slope * cal["index"].to_numpy()
    r2 = 1.0 - ((cal["ppg"].to_numpy() - fit) ** 2).sum() / \
        ((cal["ppg"].to_numpy() - cal["ppg"].mean()) ** 2).sum()

    teams["roster_index"] = teams["team"].map(idx26).fillna(np.nan)
    teams["index_2025"] = teams["team"].map(idx25)
    teams["proj_off_ppg"] = intercept + slope * teams["roster_index"]
    # Fallback for any team with no index: prior-year offense.
    teams["proj_off_ppg"] = teams["proj_off_ppg"].fillna(teams["ppg"])
    teams["off_ppg_2025"] = teams["ppg"]
    teams["off_change"] = slope * (teams["roster_index"] - teams["index_2025"])

    lg_oppg = teams["oppg"].mean()
    teams["proj_def_ppg"] = lg_oppg + DEF_REGRESS * (teams["oppg"] - lg_oppg)

    teams["proj_net"] = teams["proj_off_ppg"] - teams["proj_def_ppg"]
    teams["power"] = teams["proj_net"] - teams["proj_net"].mean()

    teams.attrs["calibration"] = {"slope": slope, "intercept": intercept, "r2": r2}
    return teams


def _game_probs(schedule, power):
    games = schedule[(schedule["season"] == 2026) & (schedule["game_type"] == "REG")][
        ["week", "home_team", "away_team"]].copy()
    pmap = power.set_index("team")["power"]
    games = games[games["home_team"].isin(pmap.index) & games["away_team"].isin(pmap.index)]
    margin = games["home_team"].map(pmap) - games["away_team"].map(pmap) + HOME_ADV
    games["p_home"] = _norm_cdf((margin / GAME_SD).to_numpy())
    return games.reset_index(drop=True)


def project_season(ratings, depth, divisions, schedule, weekly,
                   n_sims=N_SIMS, seed=7):
    """Full projection.

    Returns (team_table, games_table, changes):
      team_table adds proj_off_ppg / proj_def_ppg / off_change on top of
      proj_wins, playoff_pct, div_title_pct, exp_finish, win_dist, power.
      changes: dict team -> {adds, losses, rookies}.
    """
    power = build_team_projections(ratings, depth, divisions, weekly)
    cal = power.attrs.get("calibration", {})
    games = _game_probs(schedule, power)
    changes = roster_changes(depth, _player_values(weekly))

    team_list = power["team"].tolist()
    idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)

    home_i = games["home_team"].map(idx).to_numpy()
    away_i = games["away_team"].map(idx).to_numpy()
    p_home = games["p_home"].to_numpy()

    exp_wins = np.zeros(n_teams)
    np.add.at(exp_wins, home_i, p_home)
    np.add.at(exp_wins, away_i, 1.0 - p_home)

    rng = np.random.default_rng(seed)
    n_games = len(games)
    home_win = rng.random((n_sims, n_games)) < p_home

    wins = np.zeros((n_sims, n_teams), dtype=np.int16)
    for g in range(n_games):
        hw = home_win[:, g]
        wins[hw, home_i[g]] += 1
        wins[~hw, away_i[g]] += 1

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

    win_dist = [np.bincount(wins[:, i], minlength=GAMES + 1) / n_sims
                for i in range(n_teams)]

    table = power[["team", "division", "conference", "power", "roster_index",
                   "proj_off_ppg", "proj_def_ppg", "off_ppg_2025", "off_change"]].copy()
    table["proj_wins"] = exp_wins.round(1)
    table["proj_losses"] = (GAMES - exp_wins).round(1)
    table["playoff_pct"] = playoff * 100
    table["div_title_pct"] = div_title * 100
    table["exp_finish"] = exp_finish.round(2)
    table["win_dist"] = win_dist
    table = table.sort_values("proj_wins", ascending=False).reset_index(drop=True)
    table.attrs["calibration"] = cal
    return table, games, changes


def _group(team_list, mapping):
    out = {}
    for t in team_list:
        out.setdefault(mapping[t], []).append(t)
    return out
