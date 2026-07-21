"""2026 season projection model — team record & expected finish.

Approach (transparent, explainable):

1. Power rating (points of margin vs a league-average team) blends two
   independent signals, each converted to a league z-score:
     - Prior-season strength: 2025 net points per game.
     - 2026 roster talent: weighted 2025 PPR of each team's projected offensive
       starters, taken from the updated (post-offseason) depth chart, so trades
       / signings / rookies move a team's rating.
   power = PWR_SD * (W_PRIOR * z(net_2025) + W_ROSTER * z(roster_2026))

2. Game model: for every 2026 scheduled game,
     margin_home = power_home - power_away + HOME_ADV
     P(home win) = Phi(margin_home / GAME_SD)         (normal CDF)

3. Projected record = sum of win probabilities over each team's 17 games.

4. Monte-Carlo simulation of the full 272-game slate yields the win-total
   distribution, expected division finish, division-title odds and playoff
   odds (simplified seeding: 4 division winners + 3 wild cards per conference,
   ties broken at random).

Limitations: defense is captured only through prior-year net margin (weekly
player data is offense-only), and rookie starters carry 0 prior PPR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Model constants ──────────────────────────────────────────────────────────
W_PRIOR = 0.55        # weight on 2025 net-margin signal
W_ROSTER = 0.45       # weight on 2026 roster-talent signal
PWR_SD = 5.5          # points; spread of team power ratings
HOME_ADV = 1.6        # home-field advantage, points
GAME_SD = 13.2        # single-game point-margin standard deviation
N_SIMS = 20_000
GAMES = 17

# Weight of each projected offensive starter's prior PPR in the talent score.
STARTER_WEIGHTS = {
    ("QB", 1): 1.00,
    ("RB", 1): 0.60, ("RB", 2): 0.30,
    ("WR", 1): 0.70, ("WR", 2): 0.50, ("WR", 3): 0.35,
    ("TE", 1): 0.45,
}

_SQRT2 = np.sqrt(2.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf
    vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / _SQRT2)))
    return vec(x)


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def build_power_ratings(ratings: pd.DataFrame, depth: pd.DataFrame,
                        divisions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team with power rating and component signals."""
    teams = divisions[["team_abbr", "division", "conference"]].rename(
        columns={"team_abbr": "team"}
    ).copy()

    # Prior-season net margin (2025).
    r25 = ratings[ratings["season"] == 2025][["team", "net_ppg"]]
    teams = teams.merge(r25, on="team", how="left")
    teams["net_ppg"] = teams["net_ppg"].fillna(0.0)

    # 2026 offensive roster talent from the updated depth chart.
    off = depth[depth["side"] == "offense"].copy()
    off["w"] = [
        STARTER_WEIGHTS.get((p, int(o)), 0.0)
        for p, o in zip(off["position"], off["depth_order"])
    ]
    off = off[off["w"] > 0]
    off["contrib"] = off["fantasy_pts"].astype(float) * off["w"]
    talent = off.groupby("team")["contrib"].sum().rename("roster_talent")
    teams = teams.merge(talent, on="team", how="left")
    teams["roster_talent"] = teams["roster_talent"].fillna(0.0)

    teams["z_net"] = _zscore(teams["net_ppg"])
    teams["z_roster"] = _zscore(teams["roster_talent"])
    teams["power"] = PWR_SD * (W_PRIOR * teams["z_net"] + W_ROSTER * teams["z_roster"])
    return teams


def _game_probs(schedule: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    """Attach home-win probability to each 2026 regular-season game."""
    games = schedule[
        (schedule["season"] == 2026) & (schedule["game_type"] == "REG")
    ][["week", "home_team", "away_team"]].copy()
    pmap = power.set_index("team")["power"]
    games = games[games["home_team"].isin(pmap.index) & games["away_team"].isin(pmap.index)]
    margin = games["home_team"].map(pmap) - games["away_team"].map(pmap) + HOME_ADV
    games["p_home"] = _norm_cdf((margin / GAME_SD).to_numpy())
    return games.reset_index(drop=True)


def project_season(ratings: pd.DataFrame, depth: pd.DataFrame,
                   divisions: pd.DataFrame, schedule: pd.DataFrame,
                   n_sims: int = N_SIMS, seed: int = 7):
    """Run the full projection.

    Returns (team_table, games_table):
      team_table columns: team, division, conference, power, roster_talent,
        proj_wins, proj_losses, playoff_pct, div_title_pct, exp_finish, win_dist
      games_table columns: week, home_team, away_team, p_home
    """
    power = build_power_ratings(ratings, depth, divisions)
    games = _game_probs(schedule, power)

    team_list = power["team"].tolist()
    idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)

    home_i = games["home_team"].map(idx).to_numpy()
    away_i = games["away_team"].map(idx).to_numpy()
    p_home = games["p_home"].to_numpy()

    # Analytic expected wins (stable point estimate for the projected record).
    exp_wins = np.zeros(n_teams)
    np.add.at(exp_wins, home_i, p_home)
    np.add.at(exp_wins, away_i, 1.0 - p_home)

    # ── Monte-Carlo ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_games = len(games)
    home_win = rng.random((n_sims, n_games)) < p_home  # (sims, games)

    wins = np.zeros((n_sims, n_teams), dtype=np.int16)
    for g in range(n_games):
        hw = home_win[:, g]
        wins[hw, home_i[g]] += 1
        wins[~hw, away_i[g]] += 1

    # Tiny noise breaks ties uniformly at random and stays stable per team.
    noise = rng.random((n_sims, n_teams)) * 1e-3
    keyed = wins + noise

    div_of = power.set_index("team")["division"].to_dict()
    conf_of = power.set_index("team")["conference"].to_dict()

    exp_finish = np.zeros(n_teams)
    div_title = np.zeros(n_teams)
    playoff = np.zeros(n_teams)

    # Division finish + division winners.
    div_winner_mask = np.zeros((n_sims, n_teams), dtype=bool)
    for _div, members in _group(team_list, div_of).items():
        cols = [idx[t] for t in members]
        sub = keyed[:, cols]                              # (sims, 4)
        # rank 1 = most wins; rank = #teammates strictly ahead + 1
        ranks = (sub[:, None, :] > sub[:, :, None]).sum(axis=2) + 1  # (sims, 4)
        for k, t in enumerate(members):
            exp_finish[idx[t]] = ranks[:, k].mean()
            is_winner = ranks[:, k] == 1
            div_title[idx[t]] = is_winner.mean()
            div_winner_mask[is_winner, idx[t]] = True

    # Playoffs: 4 division winners + 3 wild cards per conference.
    for _conf, members in _group(team_list, conf_of).items():
        cols = np.array([idx[t] for t in members])
        w_conf = keyed[:, cols]                           # (sims, 16)
        winners = div_winner_mask[:, cols]                # (sims, 16)
        # Wild cards: best non-winners by wins.
        masked = np.where(winners, -1.0, w_conf)
        order = np.argsort(-masked, axis=1)               # non-winners sorted desc
        wildcard = np.zeros_like(winners)
        rows = np.arange(w_conf.shape[0])[:, None]
        wildcard[rows, order[:, :3]] = True
        in_playoffs = winners | wildcard
        for k, t in enumerate(members):
            playoff[idx[t]] = in_playoffs[:, k].mean()

    win_dist = [np.bincount(wins[:, i], minlength=GAMES + 1) / n_sims
                for i in range(n_teams)]

    table = power[["team", "division", "conference", "power", "roster_talent"]].copy()
    table["proj_wins"] = exp_wins.round(1)
    table["proj_losses"] = (GAMES - exp_wins).round(1)
    table["playoff_pct"] = playoff * 100
    table["div_title_pct"] = div_title * 100
    table["exp_finish"] = exp_finish.round(2)
    table["win_dist"] = win_dist
    table = table.sort_values("proj_wins", ascending=False).reset_index(drop=True)
    return table, games


def _group(team_list, mapping):
    out = {}
    for t in team_list:
        out.setdefault(mapping[t], []).append(t)
    return out
