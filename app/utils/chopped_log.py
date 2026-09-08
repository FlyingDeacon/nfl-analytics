"""Persistent record of the picks both CHOPPED entries have made.

A survivor planner is only useful across a season if it remembers what has
already been spent, and a Streamlit multiselect forgets everything the moment
the server restarts. This keeps the picks in one small CSV so the used-team
list, the elimination status and the season history are all read off the same
record instead of being retyped every week.

Only the human decision is stored — week, entry, team. Whether that pick won is
derived from the schedule (see survivor.grade_picks), so a recorded result can
never drift from what actually happened on the field.

Streamlit Community Cloud rebuilds the container filesystem on every redeploy
and on idle restarts, so this file survives a browser refresh there but not a
reboot. The page offers a download/restore pair for exactly that; committing the
CSV to the repo is what makes a pick permanent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.data_loader import get_base_dir

COLUMNS = ["week", "entry", "team"]


def picks_path() -> Path:
    return get_base_dir() / "data" / "chopped_picks.csv"


def load_picks() -> pd.DataFrame:
    """Every pick on record, oldest first. Empty frame when nothing is logged.

    Deliberately not cached: it is a handful of rows, it changes from inside the
    app, and a stale cache here would show someone a team they had already used.
    """
    path = picks_path()
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS].dropna(subset=COLUMNS)
    df["week"] = df["week"].astype(int)
    # One pick per entry per week is the rule, so a duplicate is a bad write
    # rather than history worth keeping — the later row wins.
    df = df.drop_duplicates(subset=["week", "entry"], keep="last")
    return df.sort_values(["week", "entry"]).reset_index(drop=True)


def save_picks(picks: pd.DataFrame) -> None:
    path = picks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    picks[COLUMNS].sort_values(["week", "entry"]).to_csv(path, index=False)


def record_pick(week: int, entry: str, team: str) -> None:
    """Log a pick, replacing any existing one for that entry and week."""
    picks = load_picks()
    picks = picks[~((picks["week"] == int(week)) & (picks["entry"] == entry))]
    row = pd.DataFrame([{"week": int(week), "entry": entry, "team": team}])
    save_picks(pd.concat([picks, row], ignore_index=True))


def clear_pick(week: int, entry: str) -> None:
    """Unlock a pick so it can be made again."""
    picks = load_picks()
    save_picks(picks[~((picks["week"] == int(week)) & (picks["entry"] == entry))])
