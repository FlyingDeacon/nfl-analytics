"""Walk-forward backtest of the season-projection *game engine*.

The production model (app/utils/record_model.py) maps a roster index to a
predicted offensive PPG, then feeds a net-PPG power rating through a normal
single-game margin model and Monte-Carlo-simulates the season.  We cannot walk
the roster-index piece forward (no historical depth charts), but we CAN validate
the part that actually turns power ratings into wins — HOME_ADV, GAME_SD, and how
much a prior season should be regressed toward the mean.

For each target season S (2017-2025):
  * PRIOR  = regressed prior-season(s) net PPG  → power rating (centered).
  * Run season S's actual REG schedule through the model's formula
        P(home win) = Phi((power_home - power_away + HOME_ADV) / GAME_SD)
    and sum expected wins per team.
  * Score predicted wins vs ACTUAL REG wins that season.

Baselines it must beat to be worth anything:
  * league mean (8.5 wins for everyone)
  * naive: last season's actual win total, carried forward.

It also reports whether the simulated win-total spread is calibrated (predicted
SD of wins vs actual SD of wins across teams) — the "distributions too narrow"
question — and sweeps the regression coefficient / GAME_SD to show the tuning
surface.

    python scripts/backtest_record_model.py
"""
from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from app.utils.record_model import HOME_ADV, GAME_SD

SEASONS = list(range(2016, 2026))
_SQRT2 = sqrt(2.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + erf(float(v) / _SQRT2)) for v in np.ravel(x)])


def _reg_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Completed regular-season games with a numeric result (home margin)."""
    g = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()].copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    return g[["season", "week", "home_team", "away_team", "home_margin"]]


def _season_table(sched: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team): games, wins, net PPG — from raw REG scores."""
    g = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()].copy()
    recs = {}
    for _, r in g.iterrows():
        s = r["season"]
        h, a = r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        for t in (h, a):
            recs.setdefault((s, t), {"gp": 0, "w": 0.0, "pf": 0.0, "pa": 0.0})
        recs[(s, h)]["gp"] += 1; recs[(s, a)]["gp"] += 1
        recs[(s, h)]["pf"] += hs; recs[(s, h)]["pa"] += as_
        recs[(s, a)]["pf"] += as_; recs[(s, a)]["pa"] += hs
        if hs > as_:
            recs[(s, h)]["w"] += 1
        elif as_ > hs:
            recs[(s, a)]["w"] += 1
        else:
            recs[(s, h)]["w"] += 0.5; recs[(s, a)]["w"] += 0.5
    out = []
    for (s, t), d in recs.items():
        gp = d["gp"]
        out.append({"season": s, "team": t, "gp": gp, "wins": d["w"],
                    "net_ppg": (d["pf"] - d["pa"]) / gp})
    return pd.DataFrame(out)


def _prior_power(stats: pd.DataFrame, season: int, k: int, regress: float) -> dict:
    """Power rating for `season` = regressed avg net PPG of the prior `k` seasons.

    regress in [0,1]: fraction of the raw net-PPG signal kept (1=no reversion,
    0=everyone average). Recency-weighted if k>1 (most recent season weight k).
    """
    priors = [season - i for i in range(1, k + 1)]
    sub = stats[stats["season"].isin(priors)].copy()
    if sub.empty:
        return {}
    weight = {season - i: (k - i + 1) for i in range(1, k + 1)}
    sub["w"] = sub["season"].map(weight)
    agg = sub.groupby("team").apply(
        lambda d: np.average(d["net_ppg"], weights=d["w"])).to_dict()
    mean = np.mean(list(agg.values()))
    return {t: regress * (v - mean) for t, v in agg.items()}


def _predict_wins(sched: pd.DataFrame, season: int, power: dict,
                  home_adv: float, game_sd: float) -> dict:
    g = _reg_games(sched)
    g = g[g["season"] == season]
    g = g[g["home_team"].isin(power) & g["away_team"].isin(power)]
    exp = {t: 0.0 for t in power}
    margin = (g["home_team"].map(power).to_numpy()
              - g["away_team"].map(power).to_numpy() + home_adv)
    p_home = _norm_cdf(margin / game_sd)
    for ph, h, a in zip(p_home, g["home_team"], g["away_team"]):
        exp[h] += ph
        exp[a] += 1.0 - ph
    return exp


def _score(pred: dict, actual: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    a = actual.set_index("team")["wins"]
    common = [t for t in pred if t in a.index]
    p = np.array([pred[t] for t in common])
    y = np.array([a[t] for t in common])
    return float(np.abs(p - y).mean()), p, y


def main() -> None:
    sched = pd.read_csv(REPO_ROOT / "data" / "raw" / "schedules.csv", low_memory=False)
    stats = _season_table(sched)

    print(f"\nModel constants under test: HOME_ADV={HOME_ADV}  GAME_SD={GAME_SD}\n")

    # ── Headline: current model config vs baselines, walk-forward ────────────
    print("=" * 78)
    print("WALK-FORWARD WIN-TOTAL ACCURACY (MAE vs actual REG wins)")
    print("=" * 78)
    print(f"{'Season':>6} {'Model':>7} {'Naive':>7} {'Mean8.5':>8}   "
          f"{'ModelR':>7} {'PredSD':>7} {'ActSD':>7}")

    tgt = [s for s in SEASONS if s - 1 in SEASONS]
    m_all, n_all, mean_all = [], [], []
    corr_all, psd_all, asd_all = [], [], []
    for s in tgt:
        power = _prior_power(stats, s, k=1, regress=0.55)
        pred = _predict_wins(sched, s, power, HOME_ADV, GAME_SD)
        actual = stats[stats["season"] == s]
        mae, p, y = _score(pred, actual)

        prior = stats[stats["season"] == s - 1].set_index("team")["wins"]
        cur = actual.set_index("team")["wins"]
        common = [t for t in cur.index if t in prior.index]
        naive_mae = float(np.abs(cur[common].to_numpy() - prior[common].to_numpy()).mean())
        mean_mae = float(np.abs(cur.to_numpy() - 8.5).mean())

        corr = float(np.corrcoef(p, y)[0, 1])
        m_all.append(mae); n_all.append(naive_mae); mean_all.append(mean_mae)
        corr_all.append(corr); psd_all.append(p.std()); asd_all.append(y.std())
        print(f"{s:>6} {mae:>7.2f} {naive_mae:>7.2f} {mean_mae:>8.2f}   "
              f"{corr:>7.2f} {p.std():>7.2f} {y.std():>7.2f}")

    print("-" * 78)
    print(f"{'AVG':>6} {np.mean(m_all):>7.2f} {np.mean(n_all):>7.2f} "
          f"{np.mean(mean_all):>8.2f}   {np.mean(corr_all):>7.2f} "
          f"{np.mean(psd_all):>7.2f} {np.mean(asd_all):>7.2f}")
    print("\n(PredSD = spread of predicted wins across teams; ActSD = actual. "
          "PredSD << ActSD means the projection is too timid / compressed.)")

    # ── Tuning sweep: regression coefficient × GAME_SD ───────────────────────
    print("\n" + "=" * 78)
    print("TUNING SWEEP — avg walk-forward MAE (lower = better)")
    print("=" * 78)
    regress_grid = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    sd_grid = [11.0, 12.0, 13.2, 14.0]
    print("            " + "".join(f"SD={sd:>5}" for sd in sd_grid))
    best = (1e9, None)
    for reg in regress_grid:
        row = f"regress={reg:<4} "
        for sd in sd_grid:
            maes = []
            for s in tgt:
                power = _prior_power(stats, s, k=1, regress=reg)
                pred = _predict_wins(sched, s, power, HOME_ADV, sd)
                mae, _, _ = _score(pred, stats[stats["season"] == s])
                maes.append(mae)
            avg = float(np.mean(maes))
            if avg < best[0]:
                best = (avg, (reg, sd))
            row += f"{avg:>7.2f}"
        print(row)
    print(f"\nbest: regress={best[1][0]}  GAME_SD={best[1][1]}  MAE={best[0]:.3f}")

    # ── Multi-year prior (recency-weighted) vs single-year ───────────────────
    print("\n" + "=" * 78)
    print("PRIOR WINDOW — avg walk-forward MAE by #seasons of history")
    print("=" * 78)
    for k in (1, 2, 3):
        tgt_k = [s for s in SEASONS if s - k in SEASONS]
        maes = []
        for s in tgt_k:
            power = _prior_power(stats, s, k=k, regress=0.55)
            pred = _predict_wins(sched, s, power, HOME_ADV, GAME_SD)
            mae, _, _ = _score(pred, stats[stats["season"] == s])
            maes.append(mae)
        print(f"  k={k} season(s):  MAE={np.mean(maes):.3f}  (n={len(maes)} seasons)")

    print()


if __name__ == "__main__":
    main()
