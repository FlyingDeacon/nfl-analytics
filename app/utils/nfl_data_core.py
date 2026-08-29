"""Streamlit-free data layer shared by the dashboard and the MCP server.

The dashboard wraps these functions in `st.cache_data` (see data_loader.py);
the MCP server imports them directly. Keeping team-abbreviation normalisation
and the DuckDB view definitions in one place is what guarantees a question
asked through Claude returns the same number the pages render.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def get_base_dir() -> Path:
    # app/utils/nfl_data_core.py -> utils -> app -> repo root
    return Path(__file__).resolve().parent.parent.parent


# ── Canonical team abbreviations ─────────────────────────────────────────────
ABBR_MAP = {
    "LA":  "LAR",   # Los Angeles Rams (old alias)
    "STL": "LAR",   # St. Louis Rams (relocated)
    "OAK": "LV",    # Oakland Raiders (relocated)
    "SD":  "LAC",   # San Diego Chargers (relocated)
}

ACTIVE_32 = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
    "DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA",
    "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
}


def norm_abbr(series: pd.Series) -> pd.Series:
    """Replace legacy team abbreviations with canonical ones."""
    return series.map(
        lambda x: ABBR_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x
    )


def normalize_name(name: str) -> str:
    """Lowercase a player name and drop suffixes/punctuation for matching."""
    name = str(name).lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = re.sub(r"[.\-']", "", name)
    return re.sub(r"\s+", " ", name).strip()


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ── Readers (plain pandas, no caching) ───────────────────────────────────────

def read_ratings() -> pd.DataFrame:
    df = pd.read_csv(get_base_dir() / "data/processed/team_ratings.csv")
    if "team" in df.columns:
        df["team"] = norm_abbr(df["team"])
    return df


def read_teams() -> pd.DataFrame:
    df = pd.read_csv(get_base_dir() / "data/raw/teams.csv")
    if "team_abbr" in df.columns:
        df["team_abbr"] = norm_abbr(df["team_abbr"])
        df = df[df["team_abbr"].isin(ACTIVE_32)]
    return df.drop_duplicates(subset=["team_abbr"], keep="last")


def read_schedules() -> pd.DataFrame:
    df = pd.read_csv(get_base_dir() / "data/raw/schedules.csv", low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ("home_team", "away_team", "team", "posteam", "defteam"):
        if col in df.columns:
            df[col] = norm_abbr(df[col])
    return df


def read_weekly() -> pd.DataFrame:
    path = get_base_dir() / "data/raw/weekly.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ("recent_team", "team", "opponent_team"):
        if col in df.columns:
            df[col] = norm_abbr(df[col])
    return df


def read_weekly_def() -> pd.DataFrame:
    path = get_base_dir() / "data/raw/weekly_def.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    if "team" in df.columns:
        df["team"] = norm_abbr(df["team"])
    return df


def read_divisions() -> pd.DataFrame:
    path = get_base_dir() / "data/raw/nfl_divisions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_depth_charts() -> pd.DataFrame:
    path = get_base_dir() / "data/raw/depth_charts.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ── Table documentation ──────────────────────────────────────────────────────
# Semantic notes the column names alone do not convey. Column lists themselves
# are generated from the live views, so they cannot drift.

TABLE_NOTES = {
    "weekly": (
        "One row per player per game, 2016-2025 regular AND postseason. "
        "Filter `season_type = 'REG'` unless the question is about playoffs. "
        "`player_display_name` is the readable name, `recent_team` is the team "
        "the player was on that week, `opponent_team` is who they faced. "
        "`fantasy_points` is STANDARD scoring and `fantasy_points_ppr` is full "
        "PPR; there is no half-PPR column, so compute it as "
        "`fantasy_points + 0.5 * receptions`. Position includes defensive and "
        "offensive-line positions, so filter to ('QB','RB','WR','TE') for "
        "fantasy questions."
    ),
    "schedules": (
        "One row per game, 2016-2026. 2026 rows are scheduled but unplayed, so "
        "`home_score`/`away_score`/`result` are NULL for them - filter "
        "`home_score IS NOT NULL` when asking about actual results. `result` is "
        "home_score minus away_score. `spread_line` and `total_line` are the "
        "closing Vegas lines. `game_type` is 'REG' or a playoff round."
    ),
    "team_ratings": (
        "One row per team per season, 2016-2025. `ppg` is points scored per "
        "game (offense), `oppg` is points allowed per game (defense, lower is "
        "better), `net_ppg` is ppg minus oppg."
    ),
    "teams": "Team metadata and logo URLs, one row per active franchise.",
    "divisions": "Conference and division for each of the 32 teams.",
    "depth_charts": "2025 depth chart entries, used to infer 2026 starters.",
    "weekly_def": "2025 per-defender box score stats (sacks, tackles, INTs).",
    "big_board_ppr": (
        "The 2026 fantasy projection model's output in full-PPR scoring - this "
        "is what the Fantasy Predictions page shows. `predicted_pts` is "
        "projected season total, `pred_ppg` per game, `proj_games` expected "
        "games played, `vor` value over replacement at the position, "
        "`espn_overall` is ESPN's consensus rank for comparison. Rank by `vor` "
        "for draft value and by `predicted_pts` for raw output."
    ),
    "big_board_half_ppr": "Same as big_board_ppr but half-PPR scoring.",
    "big_board_standard": "Same as big_board_ppr but standard (non-PPR) scoring.",
    "season_projections_2026": (
        "The 2026 season simulation, one row per team - what the Season "
        "Projections page shows. `proj_wins` is mean wins across 20k Monte "
        "Carlo sims, `playoff_pct` and `div_title_pct` are percentages (0-100), "
        "`power` is projected net points per game versus league average."
    ),
    "projected_games_2026": (
        "Every 2026 game with `p_home`, the model's probability the home team wins."
    ),
}


# ── DuckDB ───────────────────────────────────────────────────────────────────

MAX_ROWS = 200  # rows returned to the model per query


def build_frames() -> dict[str, pd.DataFrame]:
    """Load every dataset the dashboard exposes, keyed by view name."""
    base = get_base_dir()
    frames = {
        "weekly": read_weekly(),
        "schedules": read_schedules(),
        "team_ratings": read_ratings(),
        "teams": read_teams(),
        "divisions": read_divisions(),
        "weekly_def": read_weekly_def(),
        "depth_charts": read_depth_charts(),
    }

    for scoring in ("PPR", "Half_PPR", "Standard"):
        path = base / f"data/derived/big_board_{scoring}.parquet"
        if path.exists():
            frames[f"big_board_{scoring.lower()}"] = pd.read_parquet(path)

    table, games = _season_projection(frames)
    if table is not None:
        frames["season_projections_2026"] = table
        frames["projected_games_2026"] = games

    return frames


def _season_projection(frames: dict[str, pd.DataFrame]):
    """Run the 2026 Monte Carlo simulation, returning (team table, game table)."""
    try:
        from utils.record_model import project_season

        win_totals_path = get_base_dir() / "data/raw/win_totals_2026.csv"
        table, games, _changes = project_season(
            frames["team_ratings"],
            frames["depth_charts"],
            frames["divisions"],
            frames["schedules"],
            frames["weekly"],
            frames["weekly_def"],
            win_totals=pd.read_csv(win_totals_path) if win_totals_path.exists() else None,
        )
        # win_dist is a per-team probability array; it bloats every SELECT *
        # and the model cannot do anything useful with it.
        return table.drop(columns=["win_dist"], errors="ignore"), games
    except Exception:
        return None, None


def build_connection():
    """Return an in-memory DuckDB with a view over every dataset."""
    import duckdb

    con = duckdb.connect(":memory:")
    for name, df in build_frames().items():
        if df is None or df.empty:
            continue
        con.register(f"_df_{name}", df)
        con.execute(f'CREATE VIEW "{name}" AS SELECT * FROM _df_{name}')

    # Views are backed by in-memory frames, so queries never need the
    # filesystem or network.
    con.execute("SET enable_external_access = false")
    return con


def render_schema(con) -> str:
    """Build the table/column reference handed to the model."""
    # SHOW TABLES also lists the registered pandas frames backing each view;
    # only the views are part of the documented interface.
    names = [r[0] for r in con.execute("SHOW TABLES").fetchall() if not r[0].startswith("_df_")]

    blocks = []
    for name in names:
        cols = con.execute(f'DESCRIBE "{name}"').fetchall()
        rows = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        col_list = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        note = TABLE_NOTES.get(name, "")
        blocks.append(f"### {name}  ({rows:,} rows)\n{note}\nColumns: {col_list}")
    return "\n\n".join(blocks)


# ── Query tools ──────────────────────────────────────────────────────────────

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|install|load|pragma|export)\b",
    re.IGNORECASE,
)


def run_sql(con, sql: str) -> str:
    """Execute a read-only SELECT and return the rows as CSV."""
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        return "ERROR: only SELECT/WITH queries are allowed."
    if _FORBIDDEN.search(stripped):
        return "ERROR: query contains a non-read-only keyword."
    if ";" in stripped:
        return "ERROR: run one statement at a time."

    try:
        df = con.execute(stripped).fetch_df()
    except Exception as exc:
        return f"ERROR: {exc}"

    if df.empty:
        return "Query returned 0 rows."

    truncated = len(df) > MAX_ROWS
    out = df.head(MAX_ROWS).round(3).to_csv(index=False)
    if truncated:
        out += f"\n({len(df):,} rows matched; showing first {MAX_ROWS}.)"
    return out


def find_player(con, name: str) -> str:
    """Resolve a partial or misspelled name to the exact strings in the data."""
    target = normalize_name(name)
    hits = con.execute(
        """
        SELECT player_display_name AS player,
               MAX(position) AS pos,
               MAX(season)   AS last_season,
               COUNT(*)      AS games
        FROM weekly
        WHERE player_display_name IS NOT NULL
        GROUP BY player_display_name
        """
    ).fetch_df()

    norm = hits["player"].map(normalize_name)
    # Substring either direction catches "mahomes", "patrick mahomes ii" and
    # "Ja'Marr" alike without pulling in a fuzzy-matching dependency.
    mask = norm.str.contains(re.escape(target), regex=True) | norm.apply(
        lambda n: n in target and len(n) > 3
    )
    matches = hits[mask].sort_values("last_season", ascending=False).head(15)

    if matches.empty:
        return f"No player matching '{name}'. They may not appear in weekly data (2016-2025)."
    return matches.to_csv(index=False)
