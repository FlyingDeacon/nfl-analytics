"""How well the 2026 fantasy projection actually scored, read off the backtest.

The numbers come from baseline_metrics.json, written by
scripts/backtest_model.py. That is a *walk-forward* backtest: to score season S
it trains only on season pairs earlier than S and then projects S from S-1
features, so nothing it reports has seen the season it is being graded on. This
module only aggregates that file — it never recomputes a projection, so the page
cannot quietly disagree with the committed backtest.

Two things in the file need care, which is the whole reason this lives in one
place rather than being inlined on the page.

1. The "combined" block is a POOLING ARTIFACT for anything rank-shaped. It is
   computed by concatenating all five holdout seasons into one frame and then
   ranking that, so "the top 24" means the top 24 of ~1566 player-seasons rather
   than the top 24 of a season. It reports a 16.7% top-24 hit rate against
   37.5-58.3% in the individual seasons, and a mean absolute rank error of 295
   against ~59. Neither is a fact about the model. So rank-based metrics are
   averaged across the per-season blocks instead, and the combined block is used
   only for point errors (mean_abs_error_pts, rmse_pts), which carry no ranking
   and do pool correctly — 55.19 pooled against a 55.19 per-season mean.

2. The hit rates and the OVERALL rank correlation are cross-position, and the
   backtest ranks by `proj_vor`, which uses the older static replacement
   constant rather than the dynamic baseline the live page now derives. So those
   two are honest about the engine but not an exact description of today's board
   ordering, and callers should say so.

   Per-position rank correlation has no such caveat and is the number to trust.
   Subtracting a per-position replacement constant cannot reorder players within
   that position, so a positional rho is just predicted points against actual
   points — identical whichever baseline is in force.
"""
from __future__ import annotations

import json
from statistics import mean

from utils.nfl_data_core import get_base_dir

METRICS_FILE = "baseline_metrics.json"

# Metrics that are only meaningful inside a single season, so they are averaged
# over the per-season blocks rather than read from the pooled one.
_RANK_METRICS = ("spearman_vor_rank", "mean_abs_rank_error")

# Metrics that are per-player point errors, so pooling is sound.
_POINT_METRICS = ("mean_abs_error_pts", "rmse_pts", "mean_signed_bias_pts")


def _metrics_path():
    return get_base_dir() / METRICS_FILE


def load_metrics() -> dict:
    """Raw baseline_metrics.json, or {} when the backtest has never been run."""
    path = _metrics_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _avg(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else None


def accuracy(position: str = "All") -> dict:
    """Backtested accuracy for one position, or cross-position when "All".

    Returns {} if the backtest file is missing, so a fresh clone renders the
    page without this section rather than failing.

    Keys, when available:
        seasons            list of holdout seasons the figures average over
        n_per_season       graded players per season
        spearman           rank correlation, 1.0 = perfect ordering
        mean_abs_rank_err  average ranking places missed
        mae_pts            average absolute miss in fantasy points
        bias_pts           signed miss; positive means the model runs hot
        top24 / top50      share of the real top finishers it had in its own
                           top 24 / 50. Cross-position only, so absent for a
                           single position — the file does not break the hit
                           rates out by position, and a "top 24" inside one
                           position would not mean the same thing anyway.
    """
    m = load_metrics()
    per_season = m.get("per_season") or {}
    if not per_season:
        return {}

    seasons = sorted(per_season)

    def blocks() -> list:
        out = []
        for year in seasons:
            season = per_season[year]
            block = (season.get("overall") if position == "All"
                     else (season.get("by_position") or {}).get(position))
            if block and block.get("n", 0) >= 3:
                out.append(block)
        return out

    found = blocks()
    if not found:
        return {}

    out = {
        "seasons": [int(y) for y in seasons],
        "n_per_season": _avg([b.get("n") for b in found]),
    }
    for key, name in zip(_RANK_METRICS, ("spearman", "mean_abs_rank_err")):
        out[name] = _avg([b.get(key) for b in found])
    for key, name in zip(_POINT_METRICS, ("mae_pts", "rmse_pts", "bias_pts")):
        out[name] = _avg([b.get(key) for b in found])

    if position == "All":
        # Averaged over seasons, never taken from the pooled block — see the
        # module docstring for why the pooled hit rate is meaningless.
        for n in (24, 50):
            vals = [per_season[y].get(f"top{n}_hit_rate") for y in seasons]
            out[f"top{n}"] = _avg(vals)

    return out
