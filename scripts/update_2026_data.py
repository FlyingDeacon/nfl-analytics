"""Refresh the 2026 schedule and 2026 depth charts from nflverse.

Two independent updates, both writing into data/raw/ in the exact schemas the
Streamlit pages already consume:

  1. schedules.csv  — append the 2026 season games (Schedule page).
  2. depth_charts.csv — rebuild the custom offense/defense depth chart from the
     latest 2026 nflverse snapshot (reflects offseason moves), with 2025
     regular-season GP + PPR points merged in (Team Profile page).

Sources (nflverse-data / nfldata):
  - https://github.com/nflverse/nfldata/raw/master/data/games.csv
  - https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv

Run:  .venv/bin/python scripts/update_2026_data.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
DEPTH_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_2026.csv"
)

SEASON = 2026

# Legacy → canonical abbreviations (mirror data_loader.ABBR_MAP + WSH).
ABBR_MAP = {"LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC", "WSH": "WAS"}


def _norm_abbr(s: pd.Series) -> pd.Series:
    return s.map(lambda x: ABBR_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)


def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = re.sub(r"[.\-']", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ── 1. Schedule ──────────────────────────────────────────────────────────────
def update_schedule() -> None:
    path = RAW / "schedules.csv"
    existing = pd.read_csv(path, low_memory=False)
    existing.columns = [c.lower().strip() for c in existing.columns]

    games = pd.read_csv(GAMES_URL, low_memory=False)
    games.columns = [c.lower().strip() for c in games.columns]

    new = games[games["season"] == SEASON].copy()
    # Keep only the columns the existing file already has, in the same order.
    cols = [c for c in existing.columns if c in new.columns]
    new = new[cols]

    # Drop any prior 2026 rows (idempotent) then append the fresh ones.
    kept = existing[existing["season"] != SEASON]
    out = pd.concat([kept, new], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"schedules.csv: +{len(new)} games for {SEASON} "
          f"(total {len(out)} rows, seasons "
          f"{int(out['season'].min())}–{int(out['season'].max())})")


# ── 2. Depth chart ───────────────────────────────────────────────────────────
OFF_POS = {"QB", "RB", "WR", "TE"}
DL_POS = {"LDE", "RDE", "NT", "LDT", "RDT", "DE", "DT"}
LB_POS = {"WLB", "SLB", "MLB", "LILB", "RILB", "LOLB", "ROLB", "ILB", "OLB", "LB"}
DB_POS = {"LCB", "RCB", "NB", "SS", "FS", "CB", "S", "DB"}


def _season_stats() -> pd.DataFrame:
    """2025 regular-season GP + PPR points, keyed by gsis id and normalized name."""
    w = pd.read_csv(RAW / "weekly.csv", low_memory=False)
    w.columns = [c.lower().strip() for c in w.columns]
    w = w[w["season"] == 2025]
    if "season_type" in w.columns:
        w = w[w["season_type"] == "REG"]
    g = w.groupby(["player_id", "player_display_name"]).agg(
        GP=("week", "nunique"),
        fantasy_pts=("fantasy_points_ppr", "sum"),
    ).reset_index()
    g["fantasy_pts"] = g["fantasy_pts"].round(1)
    g["_key"] = g["player_display_name"].map(_normalize_name)
    by_id = g[["player_id", "GP", "fantasy_pts"]].rename(columns={"player_id": "gsis_id"})
    by_name = g[["_key", "GP", "fantasy_pts"]].drop_duplicates("_key", keep=False)
    return by_id, by_name


def update_depth_chart() -> None:
    dc = pd.read_csv(DEPTH_URL, low_memory=False)
    dc["dt"] = pd.to_datetime(dc["dt"])
    # Latest snapshot per team.
    snap = dc[dc["dt"] == dc.groupby("team")["dt"].transform("max")].copy()
    snap = snap.dropna(subset=["player_name"])
    snap["team"] = _norm_abbr(snap["team"])

    rows = []
    for team, tdf in snap.groupby("team"):
        # Offense: QB/RB/WR/TE, depth_order = pos_rank within position.
        off = tdf[(tdf["pos_grp"] == "3WR 1TE") & (tdf["pos_abb"].isin(OFF_POS))]
        for _, r in off.iterrows():
            rows.append((team, r["pos_abb"], int(r["pos_rank"]),
                         r["player_name"], r["gsis_id"], "offense"))

        # Defense: collapse detailed positions into DL / LB / DB buckets.
        dfn = tdf[tdf["pos_grp"].isin(["Base 3-4 D", "Base 4-3 D"])]
        for bucket, pos_set in (("DL", DL_POS), ("LB", LB_POS), ("DB", DB_POS)):
            b = dfn[dfn["pos_abb"].isin(pos_set)].sort_values(["pos_rank", "pos_slot"])
            for order, (_, r) in enumerate(b.iterrows(), start=1):
                rows.append((team, bucket, order, r["player_name"], r["gsis_id"], "defense"))

    out = pd.DataFrame(rows, columns=["team", "position", "depth_order",
                                      "player_name", "gsis_id", "side"])
    out["season"] = SEASON
    out["jersey_number"] = 0

    # Merge 2025 stats on gsis id (robust to nicknames / team changes),
    # then fill any id-less gaps by normalized name.
    by_id, by_name = _season_stats()
    out = out.merge(by_id, on="gsis_id", how="left")
    miss = out["GP"].isna()
    if miss.any():
        fill = out.loc[miss, ["player_name"]].assign(
            _key=out.loc[miss, "player_name"].map(_normalize_name)
        ).merge(by_name, on="_key", how="left")
        out.loc[miss, "GP"] = fill["GP"].values
        out.loc[miss, "fantasy_pts"] = fill["fantasy_pts"].values
    out["GP"] = out["GP"].fillna(0).astype(int)
    out["fantasy_pts"] = out["fantasy_pts"].fillna(0.0)

    out = out[["team", "season", "position", "depth_order", "player_name",
               "gsis_id", "jersey_number", "side", "GP", "fantasy_pts"]]
    out = out.sort_values(["team", "side", "position", "depth_order"]).reset_index(drop=True)
    out.to_csv(RAW / "depth_charts.csv", index=False)
    print(f"depth_charts.csv: rebuilt for {SEASON} — {len(out)} rows, "
          f"{out['team'].nunique()} teams "
          f"(offense {int((out['side'] == 'offense').sum())}, "
          f"defense {int((out['side'] == 'defense').sum())})")


# ── 3. Defensive box stats ───────────────────────────────────────────────────
DEF_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_2025.csv"
)

DEF_COLS = [
    "player_id", "player_display_name", "position", "team", "week",
    "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss",
    "def_fumbles_forced", "def_sacks", "def_qb_hits", "def_interceptions",
    "def_pass_defended", "def_tds", "def_fumble_recovery_opp",
]


def update_def_stats() -> None:
    """Save 2025 regular-season per-defender box stats for the record model."""
    df = pd.read_csv(DEF_STATS_URL, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    for c in DEF_COLS:
        if c not in df.columns:
            df[c] = 0
    out = df[DEF_COLS].copy()
    # Keep only rows with any defensive production.
    stat_cols = [c for c in DEF_COLS if c.startswith("def_")]
    out = out[out[stat_cols].fillna(0).abs().sum(axis=1) > 0]
    out["team"] = _norm_abbr(out["team"])
    out.to_csv(RAW / "weekly_def.csv", index=False)
    print(f"weekly_def.csv: {len(out)} defender-weeks, "
          f"{out['player_id'].nunique()} defenders")


# ── 4. Headshots ─────────────────────────────────────────────────────────────
ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    f"rosters/roster_{SEASON}.csv"
)


def update_headshots() -> None:
    """Save current-season headshots keyed by gsis id.

    The app previously read headshot_url straight out of weekly.csv, which only
    runs through 2025 — so every picture was the last one taken before a player's
    most recent logged game. Players who changed teams in the offseason showed up
    in their old uniform, and rookies with no NFL snaps had no picture at all.
    The roster release is keyed to the current season and carries both.
    """
    df = pd.read_csv(ROSTER_URL, low_memory=False)
    df = df.dropna(subset=["headshot_url", "gsis_id"])
    # Weekly roster rows: keep each player's latest, which is his current club.
    if "week" in df.columns:
        df = df.sort_values("week")
    out = (df.groupby("gsis_id")
             .agg(player_name=("full_name", "last"),
                  team=("team", "last"),
                  position=("position", "last"),
                  headshot_url=("headshot_url", "last"))
             .reset_index())
    out["team"] = _norm_abbr(out["team"])
    out["season"] = SEASON
    out = out[["gsis_id", "season", "player_name", "team", "position", "headshot_url"]]
    out = out.sort_values("player_name").reset_index(drop=True)
    out.to_csv(RAW / "headshots.csv", index=False)
    print(f"headshots.csv: {len(out)} players for {SEASON}")


if __name__ == "__main__":
    update_schedule()
    update_def_stats()
    update_depth_chart()
    update_headshots()
