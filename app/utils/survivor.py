"""Survivor-pool planning for the CHOPPED league.

CHOPPED is a knockout pool: each week you name one team to win straight up, a
team can be used only ONCE all season, and one wrong pick (the mulligan) is
forgiven. Last entrant standing takes the pot.

The "use each team once" rule is what makes this more than a weekly win
probability lookup. Spending the best team on the board in Week 1 buys a ~92%
week and then leaves you short later, so the right question is never "who wins
this week" but "which assignment of teams to weeks survives the longest".

That is an assignment problem. Weeks are slots, teams are resources, and the
objective is to maximise the probability of winning every remaining week —
equivalently, to maximise the sum of log win probabilities, since surviving is
the product of independent weekly wins. scipy's Hungarian solver finds the exact
optimum in milliseconds, so the plan can be recomputed on every interaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Weight on the sportsbook when a line exists. Books close sharper than any
# public model, but our own projection correlates 0.975 with them across the 67
# priced 2026 games, so leaning fully on the market throws away a genuine second
# opinion — and would make the numbers jump discontinuously at Week 5, where the
# lines run out and the model has to carry the rest of the season alone.
MARKET_WEIGHT = 0.70

# Probability floor/ceiling. A model that reports 0.97 is claiming a level of
# certainty no NFL game supports (upsets happen at every price), and log(0)
# is undefined, so clamp before taking logs.
P_FLOOR, P_CEIL = 0.03, 0.97

# Cost used for a team/week pair that cannot be played (bye week, already used).
# A large finite number rather than inf: the Hungarian solver needs a real
# matrix, and inf propagates into NaN.
INFEASIBLE = 1e6


def _moneyline_to_prob(ml: pd.Series) -> pd.Series:
    """American odds -> implied probability (still carrying the vig)."""
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def market_probabilities(schedule: pd.DataFrame, season: int = 2026) -> pd.DataFrame:
    """Vig-free home win probability per game, where the book has posted a line.

    Both sides of a moneyline sum to more than 1 — that excess is the house
    margin. Dividing each side by the total removes it, which is what makes the
    number comparable to a model probability.
    """
    cols = {"season", "game_type", "week", "home_team", "away_team",
            "home_moneyline", "away_moneyline"}
    if not cols <= set(schedule.columns):
        return pd.DataFrame(columns=["week", "home_team", "away_team", "p_home_mkt"])

    s = schedule[(schedule["season"] == season)
                 & (schedule["game_type"] == "REG")].copy()
    home = _moneyline_to_prob(s["home_moneyline"])
    away = _moneyline_to_prob(s["away_moneyline"])
    total = home + away
    s["p_home_mkt"] = np.where(total > 0, home / total, np.nan)
    s = s.dropna(subset=["p_home_mkt"])
    return s[["week", "home_team", "away_team", "p_home_mkt"]].reset_index(drop=True)


def blend_probabilities(games: pd.DataFrame, schedule: pd.DataFrame,
                        season: int = 2026) -> pd.DataFrame:
    """Attach market probability and a blended p_home to the model's game table.

    Adds p_home_mkt (NaN when unpriced), p_home_blend, and has_market so callers
    can show which games are anchored to a real line.
    """
    out = games.copy()
    mkt = market_probabilities(schedule, season)
    out = out.merge(mkt, on=["week", "home_team", "away_team"], how="left")
    out["has_market"] = out["p_home_mkt"].notna()
    out["p_home_blend"] = np.where(
        out["has_market"],
        MARKET_WEIGHT * out["p_home_mkt"].fillna(0.0)
        + (1.0 - MARKET_WEIGHT) * out["p_home"],
        out["p_home"],
    ).clip(P_FLOOR, P_CEIL)
    return out


def team_week_table(games: pd.DataFrame, prob_col: str = "p_home_blend") -> pd.DataFrame:
    """One row per team per game: week, team, opponent, home/away, win prob.

    The planner needs to look up "what is TEAM's chance in WEEK", which the
    game-level table cannot answer without knowing which side they are on.
    """
    p = games[prob_col].clip(P_FLOOR, P_CEIL)
    home = pd.DataFrame({
        "week": games["week"], "team": games["home_team"],
        "opponent": games["away_team"], "is_home": True, "win_prob": p,
    })
    away = pd.DataFrame({
        "week": games["week"], "team": games["away_team"],
        "opponent": games["home_team"], "is_home": False, "win_prob": 1.0 - p,
    })
    return pd.concat([home, away], ignore_index=True).sort_values(["week", "team"])


def _cost_matrix(tw: pd.DataFrame, teams: list[str], weeks: list[int]) -> np.ndarray:
    """-log(win probability) per team/week, INFEASIBLE where the team is on bye."""
    grid = (tw.pivot_table(index="team", columns="week", values="win_prob",
                           aggfunc="max")
              .reindex(index=teams, columns=weeks))
    cost = -np.log(grid.to_numpy(dtype=float))
    return np.where(np.isfinite(cost), cost, INFEASIBLE)


def optimal_plan(tw: pd.DataFrame, used_teams: set[str], start_week: int,
                 end_week: int = 18, forced: tuple[int, str] | None = None
                 ) -> pd.DataFrame:
    """Assignment of one unused team to each remaining week, maximising survival.

    Maximising the product of weekly win probabilities is the same as minimising
    the sum of -log(p), which is a linear assignment problem over the
    team x week grid. `forced` pins one (week, team) pair so callers can price
    "what does this week's pick cost me later" by comparing plans.

    Returns week / team / opponent / is_home / win_prob, or an empty frame if
    no complete assignment exists.
    """
    weeks = [w for w in sorted(tw["week"].unique()) if start_week <= w <= end_week]
    teams = sorted(set(tw["team"].unique()) - set(used_teams))
    if not weeks or len(teams) < len(weeks):
        return pd.DataFrame(columns=["week", "team", "opponent", "is_home", "win_prob"])

    cost = _cost_matrix(tw, teams, weeks)

    if forced is not None:
        fweek, fteam = forced
        if fteam not in teams or fweek not in weeks:
            return pd.DataFrame(columns=["week", "team", "opponent", "is_home", "win_prob"])
        ti, wi = teams.index(fteam), weeks.index(fweek)
        # Pin the pair by making every alternative in that row and column
        # unusable, so the solver has no choice but to match them.
        keep = cost[ti, wi]
        cost[ti, :] = INFEASIBLE
        cost[:, wi] = INFEASIBLE
        cost[ti, wi] = keep

    rows, cols = linear_sum_assignment(cost)
    picks = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= INFEASIBLE:      # solver had to use a bye-week cell
            return pd.DataFrame(columns=["week", "team", "opponent", "is_home", "win_prob"])
        picks.append((weeks[c], teams[r]))

    plan = pd.DataFrame(picks, columns=["week", "team"]).sort_values("week")
    return plan.merge(tw, on=["week", "team"], how="left").reset_index(drop=True)


def survival_probability(plan: pd.DataFrame) -> float:
    """Chance of winning every week in the plan (no mulligan)."""
    return float(plan["win_prob"].prod()) if not plan.empty else 0.0


# Public survivor pools chase the biggest favourite, and they concentrate hard:
# the most popular pick in a typical week draws roughly a third of the field.
# Raising the favourite's edge over a coin flip to a power reproduces that shape
# from win probability alone, which is what lets the planner estimate a pool it
# cannot see. 3.0 puts the Week 1 chalk near 30%, matching published grids.
POPULARITY_CONCENTRATION = 3.0


def popularity_estimate(win_probs: pd.Series,
                        concentration: float = POPULARITY_CONCENTRATION) -> pd.Series:
    """Estimated share of a public pool on each team this week.

    A stand-in for real pick percentages, not a substitute: if you can read the
    actual numbers off a survivor grid, enter those instead. This only assumes
    the field piles onto favourites in proportion to how big a favourite they are.
    """
    edge = (win_probs - 0.5).clip(lower=0.0) ** concentration
    total = float(edge.sum())
    if total <= 0:
        return pd.Series(1.0 / len(win_probs), index=win_probs.index)
    return edge / total


# Grid resolution for the E[1/(1+X)] integral below. The integrand is a smooth
# polynomial on [0, 1], so the trapezoid rule converges fast; 2001 points holds
# the error well under a thousandth of the EV values being compared.
_EV_GRID = 2001


def pick_ev(win_probs: pd.Series, popularity: pd.Series, opponents: pd.Series,
            pool_size: int = 50) -> pd.Series:
    """Expected share of the pot for each pick, where 1.00 is the pool average.

    Survival is not the objective in a knockout pool — pot share is. Surviving a
    week in which most of the field also survived barely advances you, while
    surviving one that wipes out half the pool is worth far more than the extra
    risk cost. Both effects live in the denominator: the number of entries you
    would be splitting the pot with, given your team won.

    That is why an underdog can be the correct pick outright. If the chalk draws
    40% of the field, the weeks where it loses are exactly the weeks your survival
    is worth the most, and a lower win probability can more than pay for itself.

    The value of a pick is p_i * E[1/(1 + X_i)], where X_i is the number of OTHER
    entries that survive the week. Substituting E[1/(1+X)] with 1/(1+E[X]) looks
    harmless and is not: by Jensen's inequality it understates exactly the
    scenario a contrarian pick is buying — the week the chalk goes down and the
    field collapses. Computed that way the tool never recommends anything but the
    chalk. So take the expectation properly, via

        E[1/(1+X)] = integral of E[t^X] dt over [0, 1].

    The unit that factorises here is the GAME, not the entry. Entries are very
    far from independent: everyone on one side of a game and everyone on the
    other are perfectly anti-correlated, and that anti-correlation IS the
    contrarian edge. If you take the underdog and it wins, every entry on the
    favourite is eliminated with certainty — the collapse is not a tail scenario
    you hope for, it is a guarantee that comes attached to the pick. Treating
    entries as independent prices those backers as surviving at their own win
    probability and erases the effect completely.

    Games, by contrast, genuinely are independent, and each contributes a
    two-term generating function

        E[t^(survivors from game A-B)] = p_A * t^(n_A) + p_B * t^(n_B),

    so E[t^X] is a product over games. Your own game is not random once you
    condition on your team winning: it contributes t^(n_i - 1) exactly, the
    backers you must share with minus yourself.

    pool_size matters and has no sensible default that suits everyone: in a
    20-entry pool differentiation is most of the game, while in a 500-entry pool
    the field is so diluted that the raw win probability nearly always wins.
    """
    p = win_probs.astype(float)
    pop = popularity.reindex(p.index).fillna(0.0).astype(float)
    opp = opponents.reindex(p.index)

    # Entries per team. Round rather than floor so small shares are not erased,
    # and keep the total consistent with the pool size the user entered.
    n = np.maximum(np.round(pop.to_numpy() * max(int(pool_size), 2)), 0.0)

    # Pair the teams into games. Each team keeps a pointer to its game so the
    # "my own game" factor can be divided back out below.
    teams = list(p.index)
    pos = {tm: i for i, tm in enumerate(teams)}
    game_of = np.full(len(teams), -1, dtype=int)
    games: list[tuple[int, int | None]] = []
    for i, tm in enumerate(teams):
        if game_of[i] >= 0:
            continue
        o = opp.iloc[i]
        j = pos.get(o) if pd.notna(o) else None
        game_of[i] = len(games)
        if j is not None and j != i and game_of[j] < 0:
            game_of[j] = len(games)
            games.append((i, j))
        else:
            games.append((i, None))

    t = np.linspace(0.0, 1.0, _EV_GRID)
    log_t = np.log(np.clip(t, 1e-300, None))
    pv = p.to_numpy()

    def _log(x: float) -> float:
        return float(np.log(max(x, 1e-300)))

    # log E[t^X] for each game, via logaddexp so the two branches can be summed
    # without leaving log space.
    log_gf = np.empty((len(games), _EV_GRID))
    for g, (i, j) in enumerate(games):
        side_i = _log(pv[i]) + n[i] * log_t
        # An unpaired team means the opponent is outside this table, so nobody in
        # the pool is on them: their branch carries t^0 = 1.
        side_j = (_log(pv[j]) + n[j] * log_t) if j is not None else np.full_like(t, _log(1.0 - pv[i]))
        log_gf[g] = np.logaddexp(side_i, side_j)

    log_total = log_gf.sum(axis=0)
    raw = np.empty(len(teams))
    for i in range(len(teams)):
        # Swap this team's game out of the product for the conditioned version.
        log_own = log_total - log_gf[game_of[i]] + np.maximum(n[i] - 1.0, 0.0) * log_t
        raw[i] = pv[i] * float(np.trapz(np.exp(log_own), t))

    out = pd.Series(raw, index=p.index)
    # Normalise against the field's own average so 1.00 reads as "no better than
    # the pool", rather than as an unanchored ratio.
    baseline = float((pop * out).sum())
    return out / baseline if baseline > 0 else out


def week_options(tw: pd.DataFrame, used_teams: set[str], week: int,
                 end_week: int = 18, pool_size: int | None = None) -> pd.DataFrame:
    """Every legal pick this week, priced by what it costs the rest of the season.

    A team's weekly win probability alone is a trap in survivor: the safest team
    this week is often the one whose value is highest in some later week where
    you will have nothing else. For each candidate this pins it into the current
    week, re-solves the remaining season around that choice, and reports the
    resulting full-season survival. `cost_vs_best` is how much survival you give
    up relative to the unconstrained optimum — the honest price of the pick.

    Passing `pool_size` adds the other half of the picture: `popularity`, the
    share of the field expected on each team, and `pot_ev`, expected pot share
    where 1.00 is the pool average. Survival and pot share genuinely disagree
    sometimes, and neither is a strict improvement on the other, so both are
    reported rather than folded into one number.
    """
    best = optimal_plan(tw, used_teams, week, end_week)
    best_surv = survival_probability(best)

    avail = tw[(tw["week"] == week) & (~tw["team"].isin(used_teams))]
    rows = []
    for _, g in avail.iterrows():
        plan = optimal_plan(tw, used_teams, week, end_week,
                            forced=(week, g["team"]))
        surv = survival_probability(plan)
        rows.append({
            "team": g["team"], "opponent": g["opponent"], "is_home": g["is_home"],
            "win_prob": g["win_prob"],
            "season_survival": surv,
            "cost_vs_best": best_surv - surv,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if pool_size:
        # Price the field on the FULL slate, not just our legal picks. The pool's
        # money does not care which teams this entry has burned, and the EV
        # integral needs both sides of every game to pair them up correctly.
        # (It does assume the field has no used-team constraint of its own, which
        # is why popularity is worth overriding with real grid numbers when you
        # can read them.)
        slate = tw[tw["week"] == week].drop_duplicates("team").set_index("team")
        pop = popularity_estimate(slate["win_prob"])
        ev = pick_ev(slate["win_prob"], pop, slate["opponent"], pool_size)
        out["popularity"] = out["team"].map(pop)
        out["pot_ev"] = out["team"].map(ev)

    return out.sort_values(["season_survival", "win_prob"], ascending=False).reset_index(drop=True)
