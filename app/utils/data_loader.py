import re
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st


def get_base_dir() -> Path:
    # app/utils/data_loader.py → utils → app → NFL-Analytics (root)
    return Path(__file__).resolve().parent.parent.parent


@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    path = get_base_dir() / "data" / "processed" / "team_ratings.csv"
    if not path.exists():
        st.error(f"team_ratings.csv not found at {path}. Run build_team_ratings.py first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_teams() -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "teams.csv"
    if not path.exists():
        st.error(f"teams.csv not found at {path}. Run load_nfl_data.py first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_schedules() -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "schedules.csv"
    if not path.exists():
        st.error(f"schedules.csv not found at {path}. Run load_nfl_data.py first.")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_weekly() -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "weekly.csv"
    if not path.exists():
        st.warning("weekly.csv not found — player stats page unavailable.")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def get_logo(team_abbr: str, teams_df: pd.DataFrame) -> Optional[str]:
    abbr_col = "team_abbr" if "team_abbr" in teams_df.columns else "team"
    for col in ["team_logo_espn", "team_logo_wikipedia", "team_logo_squared", "team_logo"]:
        if col in teams_df.columns:
            match = teams_df[teams_df[abbr_col] == team_abbr]
            if not match.empty:
                val = match.iloc[0][col]
                return val if pd.notna(val) and str(val).strip() else None
    return None


def load_preseason_rankings() -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "preseason_rankings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = re.sub(r"[.\-']", "", name)
    return re.sub(r"\s+", " ", name).strip()


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add overall / offense / defense rank columns to a season slice."""
    out = df.copy()
    out["overall_rank"] = out["net_ppg"].rank(ascending=False, method="min").astype(int)
    out["offense_rank"] = out["ppg"].rank(ascending=False,  method="min").astype(int)
    out["defense_rank"] = out["oppg"].rank(ascending=True,   method="min").astype(int)
    return out
