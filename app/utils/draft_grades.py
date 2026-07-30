"""Grade past ESPN drafts with this repo's projection model, and check both the
model and ESPN against what actually happened.

The comparison is deliberately preseason-only on both sides, so neither side
gets hindsight:

  - model_pts  build_predictions_core(..., as_of_season=S) trains only on
               season pairs < S and projects S off S-1 features, so nothing
               from season S leaks into the model's opinion of season S.
  - espn_pts   ESPN's own preseason projection for season S, taken from the
               league-scoped player endpoint (statSourceId 1).
  - actual_pts what the player really scored in season S (statSourceId 0).

Only QB/RB/WR/TE are graded: the projection model doesn't cover K or D/ST, so
including them would score one side on players the other never rated.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import _normalize_name, get_base_dir
from utils.espn_league import draft_picks_df, final_standings_df, load_league_season

MODEL_POS = ("QB", "RB", "WR", "TE")

# Minimum share of a draft's picks that need a usable ESPN projection before
# ESPN is worth grading at all for that season.
ESPN_COVERAGE_FLOOR = 0.60


@st.cache_data(show_spinner=False)
def model_projections(season: int) -> pd.DataFrame:
    """This repo's walk-forward projection for `season`, using only pre-season
    data. One row per player: name key, position, projected points."""
    import sys
    root = Path(get_base_dir())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.backtest_model import build_config, load_weekly, resolve_columns
    from model.projection import build_predictions_core

    weekly = load_weekly(root)
    config = build_config(*resolve_columns(weekly))
    proj, _ = build_predictions_core(weekly, config, as_of_season=season)
    if proj.empty:
        return pd.DataFrame()

    out = proj[[config.name_col, config.pos_col, "predicted_pts"]].copy()
    out.columns = ["player", "pos", "model_pts"]
    out["key"] = out["player"].map(_normalize_name)
    return out[["key", "pos", "model_pts"]]


@st.cache_data(show_spinner=False)
def graded_picks(season: int) -> pd.DataFrame:
    """Every pick of `season`'s draft with all three numbers attached."""
    data = load_league_season(season)
    picks = draft_picks_df(data, season)
    if picks.empty:
        return picks

    picks = picks[picks["pos"].isin(MODEL_POS)].copy()
    picks["key"] = picks["player"].map(_normalize_name)
    model = model_projections(season)
    if not model.empty:
        picks = picks.merge(model[["key", "model_pts"]], on="key", how="left")
    else:
        picks["model_pts"] = pd.NA

    # For older seasons ESPN stops retaining preseason projections and returns
    # 0.0 rather than null, which would otherwise be summed as a real opinion
    # of "this player will score nothing". Treat non-positive as missing.
    picks["espn_pts"] = picks["espn_proj_pts"].where(picks["espn_proj_pts"] > 0)

    # 2023 comes back with projections for ~12% of picks. A grade built from
    # that is noise dressed up as a number, so drop ESPN entirely below a
    # coverage floor rather than publish a misleading comparison.
    if picks["espn_pts"].notna().mean() < ESPN_COVERAGE_FLOOR:
        picks["espn_pts"] = float("nan")  # nan, not pd.NA, to keep the column numeric

    # The model projects season S off season S-1 production, so rookies (and
    # anyone with too little prior-season data) get no projection at all, while
    # ESPN rates them fine. Scoring each side on a different set of players
    # would quietly punish whoever drafted rookies, so only picks BOTH sides
    # rated — and that actually scored — count toward a grade. `comparable`
    # marks those; the rest stay in the frame so they can be shown and counted.
    ok = picks["model_pts"].notna() & picks["actual_pts"].notna()
    if picks["espn_pts"].notna().any():  # only demand ESPN when it exists at all
        ok &= picks["espn_pts"].notna()
    picks["comparable"] = ok
    return picks.drop(columns=["key"])


def _rank(series: pd.Series) -> pd.Series:
    """1 = best (highest points)."""
    return series.rank(ascending=False, method="min").astype("Int64")


@st.cache_data(show_spinner=False)
def team_draft_grades(season: int) -> pd.DataFrame:
    """Per-team draft haul for `season` under each opinion, next to the real
    final finish. Ranks are 1 = best."""
    picks = graded_picks(season)
    if picks.empty:
        return pd.DataFrame()

    skipped = (picks.loc[~picks["comparable"]]
               .groupby("team_id", as_index=False)["player"].count()
               .rename(columns={"player": "Unrated"}))

    agg = picks[picks["comparable"]].groupby(["team_id", "Team"], as_index=False).agg(
        model_pts=("model_pts", "sum"),
        espn_pts=("espn_pts", "sum"),
        actual_pts=("actual_pts", "sum"),
        graded=("player", "count"),
    )
    agg = agg.merge(skipped, on="team_id", how="left")
    agg["Unrated"] = agg["Unrated"].fillna(0).astype(int)

    # A groupby sum over an all-NaN column returns 0.0, which would read as a
    # real ESPN grade of zero. Restore it to "no opinion" instead.
    if not picks["espn_pts"].notna().any():
        agg["espn_pts"] = float("nan")
    agg["My Rank"] = _rank(agg["model_pts"])
    agg["ESPN Rank"] = _rank(agg["espn_pts"])
    agg["Actual Rank"] = _rank(agg["actual_pts"])

    standings = final_standings_df(load_league_season(season))
    if not standings.empty:
        agg = agg.merge(standings[["team_id", "Finish", "W", "L", "PF"]],
                        on="team_id", how="left")
    for c in ("model_pts", "espn_pts", "actual_pts"):
        agg[c] = agg[c].round(1)
    return agg.sort_values("My Rank").reset_index(drop=True)


def accuracy(season: int) -> dict:
    """How well each preseason opinion predicted the real season.

    Spearman correlation of each side's team ranking against the actual finish
    and against realized points. All three are "1 = best", so a POSITIVE
    correlation means the ranking predicted the outcome well; 0 means no signal.
    """
    from scipy.stats import spearmanr

    g = team_draft_grades(season)
    if g.empty or g["Finish"].isna().all():
        return {}

    out = {}
    for label, col in (("model", "My Rank"), ("espn", "ESPN Rank")):
        if g[col].isna().all() or g[col].nunique() < 2:
            continue
        out[label] = {
            "vs_finish": round(float(spearmanr(g[col], g["Finish"]).statistic), 3),
            "vs_points": round(float(spearmanr(g[col], g["Actual Rank"]).statistic), 3),
        }
    return out
