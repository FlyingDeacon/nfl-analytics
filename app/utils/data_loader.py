import re
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
import urllib.request
import urllib.error


def get_base_dir() -> Path:
    # app/utils/data_loader.py → utils → app → NFL-Analytics (root)
    return Path(__file__).resolve().parent.parent.parent


# ── Canonical team abbreviation map ──────────────────────────────────────────
# Applied at load time in EVERY loader so stale caches can never produce wrong abbrs
ABBR_MAP = {
    "LA":  "LAR",   # Los Angeles Rams (old alias)
    "STL": "LAR",   # St. Louis Rams (relocated)
    "OAK": "LV",    # Oakland Raiders (relocated)
    "SD":  "LAC",   # San Diego Chargers (relocated)
}

# Exactly the 32 active franchises — any other team_abbr in teams.csv is dropped
ACTIVE_32 = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
    "DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA",
    "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
}


def _norm_abbr(series: pd.Series) -> pd.Series:
    """Replace legacy team abbreviations with canonical ones."""
    return series.map(lambda x: ABBR_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)


def _file_mtime(path: Path) -> float:
    """Return file mtime so cache_data re-runs when file changes on disk."""
    return path.stat().st_mtime if path.exists() else 0.0


# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_ratings(_mtime: float = 0.0) -> pd.DataFrame:
    path = get_base_dir() / "data" / "processed" / "team_ratings.csv"
    if not path.exists():
        st.error(f"team_ratings.csv not found at {path}.")
        st.stop()
    df = pd.read_csv(path)
    if "team" in df.columns:
        df["team"] = _norm_abbr(df["team"])
    return df


@st.cache_data(show_spinner=False)
def load_teams(_mtime: float = 0.0) -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "teams.csv"
    if not path.exists():
        st.error(f"teams.csv not found at {path}.")
        st.stop()
    df = pd.read_csv(path)
    # Normalise abbreviations
    if "team_abbr" in df.columns:
        df["team_abbr"] = _norm_abbr(df["team_abbr"])
    # Drop relocated/legacy franchises; keep only active 32
    if "team_abbr" in df.columns:
        df = df[df["team_abbr"].isin(ACTIVE_32)]
    # Drop any remaining duplicates (keep last = most recent entry)
    df = df.drop_duplicates(subset=["team_abbr"], keep="last")
    return df


@st.cache_data(show_spinner=False)
def load_schedules(_mtime: float = 0.0) -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "schedules.csv"
    if not path.exists():
        st.error(f"schedules.csv not found at {path}.")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    # Normalise team abbreviations in every relevant column
    for col in ("home_team", "away_team", "team", "posteam", "defteam"):
        if col in df.columns:
            df[col] = _norm_abbr(df[col])
    return df


@st.cache_data(show_spinner=False)
def load_weekly(_mtime: float = 0.0) -> pd.DataFrame:
    path = get_base_dir() / "data" / "raw" / "weekly.csv"
    if not path.exists():
        raw_dir = path.parent
        existing = list(raw_dir.iterdir()) if raw_dir.exists() else []
        st.warning(
            f"weekly.csv not found at {path}. "
            f"Base dir: {get_base_dir()}. "
            f"Files in data/raw/: {[f.name for f in existing]}"
        )
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ("recent_team", "team", "opponent_team"):
        if col in df.columns:
            df[col] = _norm_abbr(df[col])
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


@st.cache_data(show_spinner=False, ttl=86400)  # refresh once per day
def load_depth_charts(_mtime: float = 0.0) -> pd.DataFrame:
    """Load depth charts from local cache or nflverse GitHub releases.

    Data source: https://github.com/nflverse/nflverse-data
    Falls back to local CSV if network fetch fails.
    """
    base_dir = get_base_dir()
    local_path = base_dir / "data" / "raw" / "depth_charts.csv"

    # Try local cache first
    if local_path.exists():
        try:
            df = pd.read_csv(local_path)
            if not df.empty:
                return df
        except Exception:
            pass

    # Try fetching from nflverse GitHub release
    try:
        url = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2025.csv"
        with urllib.request.urlopen(url, timeout=10) as response:
            df = pd.read_csv(response)
            df.to_csv(local_path, index=False)
            return df
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        st.warning(f"Could not fetch latest depth charts: {e}. Using local cache if available.")
        if local_path.exists():
            return pd.read_csv(local_path)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_divisions(_mtime: float = 0.0) -> pd.DataFrame:
    """Load NFL divisions and conference structure (canonical 32-team reference)."""
    path = get_base_dir() / "data" / "raw" / "nfl_divisions.csv"
    if not path.exists():
        st.warning(f"nfl_divisions.csv not found at {path}.")
        return pd.DataFrame()
    return pd.read_csv(path)


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add overall / offense / defense rank columns to a season slice."""
    out = df.copy()
    out["overall_rank"] = out["net_ppg"].rank(ascending=False, method="min").astype(int)
    out["offense_rank"] = out["ppg"].rank(ascending=False,  method="min").astype(int)
    out["defense_rank"] = out["oppg"].rank(ascending=True,   method="min").astype(int)
    return out
