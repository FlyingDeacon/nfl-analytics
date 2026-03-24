import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st


def get_base_dir() -> Path:
    # app/utils/data_loader.py → utils → app → NFL-Analytics (root)
    return Path(__file__).resolve().parent.parent.parent


def _run_nfl_download():
    base_dir = get_base_dir()
    src_dir = str(base_dir / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import load_nfl_data
    load_nfl_data.main()


def _ensure_raw_data():
    """Run the data pipeline if core raw CSVs are missing."""
    base_dir = get_base_dir()
    teams_path = base_dir / "data" / "raw" / "teams.csv"
    if not teams_path.exists():
        with st.spinner("Downloading NFL data for the first time (this may take a minute)..."):
            _run_nfl_download()
            src_dir = str(base_dir / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            import build_team_ratings
            build_team_ratings.main()


def _ensure_weekly_data():
    """Run NFL download if weekly.csv is missing (may exist separately from teams.csv)."""
    base_dir = get_base_dir()
    weekly_path = base_dir / "data" / "raw" / "weekly.csv"
    if not weekly_path.exists():
        with st.spinner("Downloading weekly player data (this may take a minute)..."):
            _run_nfl_download()


def _ensure_processed_data():
    """Run build_team_ratings if processed CSV is missing but raw data exists."""
    base_dir = get_base_dir()
    processed_path = base_dir / "data" / "processed" / "team_ratings.csv"
    schedules_path = base_dir / "data" / "raw" / "schedules.csv"
    if not processed_path.exists() and schedules_path.exists():
        with st.spinner("Building team ratings..."):
            src_dir = str(base_dir / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            import build_team_ratings
            build_team_ratings.main()


@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    _ensure_raw_data()
    _ensure_processed_data()
    path = get_base_dir() / "data" / "processed" / "team_ratings.csv"
    if not path.exists():
        st.error("Failed to build team_ratings.csv. Check logs.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_teams() -> pd.DataFrame:
    _ensure_raw_data()
    path = get_base_dir() / "data" / "raw" / "teams.csv"
    if not path.exists():
        st.error("Failed to download teams.csv. Check logs.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_schedules() -> pd.DataFrame:
    _ensure_raw_data()
    path = get_base_dir() / "data" / "raw" / "schedules.csv"
    if not path.exists():
        st.error("Failed to download schedules.csv. Check logs.")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_weekly() -> pd.DataFrame:
    _ensure_raw_data()
    _ensure_weekly_data()
    path = get_base_dir() / "data" / "raw" / "weekly.csv"
    if not path.exists():
        st.warning("weekly.csv could not be loaded. Check app logs.")
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
