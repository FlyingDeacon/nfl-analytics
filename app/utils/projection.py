"""Shared entry point for the 2026 season projection.

Three pages are built on the same simulation — the standings table, the weekly
matchup board and the CHOPPED planner. project_season() is slow enough that the
spinner is visible, so each page running its own copy would both cost a wait per
navigation and risk the pages quoting subtly different numbers if their cache
keys ever drifted apart. Loading it here means one computation, one set of
numbers, and whichever page you open first pays for all three.

Cache keys are input file mtimes rather than the DataFrames themselves: hashing
a decade of weekly game logs on every rerun costs more than the lookup saves.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_ratings, load_schedules, load_divisions, load_weekly, load_weekly_def,
    get_base_dir, _file_mtime,
)
from utils.record_model import project_season
from utils.survivor import blend_probabilities, team_week_table

# Every file the projection reads. Order is fixed because the mtimes are passed
# positionally into the cached function below.
INPUTS = (
    "data/processed/team_ratings.csv",
    "data/raw/depth_charts.csv",
    "data/raw/nfl_divisions.csv",
    "data/raw/schedules.csv",
    "data/raw/weekly.csv",
    "data/raw/weekly_def.csv",
    "data/raw/win_totals_2026.csv",
)


def input_paths() -> list:
    """Absolute paths of the projection's inputs, for 'last updated' stamps."""
    base = get_base_dir()
    return [base / rel for rel in INPUTS]


def _keys() -> tuple:
    return tuple(_file_mtime(p) for p in input_paths())


@st.cache_data(show_spinner="Simulating the 2026 season…")
def _simulate(_ratings_m, _depth_m, _div_m, _sched_m, _weekly_m, _def_m, _wt_m):
    base = get_base_dir()
    win_totals_path = base / "data/raw/win_totals_2026.csv"
    return project_season(
        load_ratings(_mtime=_ratings_m),
        pd.read_csv(base / "data/raw/depth_charts.csv"),
        load_divisions(_mtime=_div_m),
        load_schedules(_mtime=_sched_m),
        load_weekly(_mtime=_weekly_m),
        load_weekly_def(_mtime=_def_m),
        win_totals=pd.read_csv(win_totals_path) if win_totals_path.exists() else None,
    )


@st.cache_data(show_spinner=False)
def _matchups(_ratings_m, _depth_m, _div_m, _sched_m, _weekly_m, _def_m, _wt_m):
    _, games, _ = _simulate(_ratings_m, _depth_m, _div_m, _sched_m,
                            _weekly_m, _def_m, _wt_m)
    blended = blend_probabilities(games, load_schedules(_mtime=_sched_m))
    return blended, team_week_table(blended)


def season_projection():
    """(table, games, changes) from the 2026 simulation."""
    return _simulate(*_keys())


def matchup_tables():
    """(blended games, team-week table) with market-blended win probabilities.

    `blended` is one row per game with p_home_blend / p_home_mkt / has_market;
    the team-week table is the same games exploded to one row per side, which is
    the shape the survivor planner needs.
    """
    return _matchups(*_keys())
