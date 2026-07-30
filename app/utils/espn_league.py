"""ESPN Fantasy Football league integration.

Reads league credentials from st.secrets["espn"] (see .streamlit/secrets.toml)
and pulls team / standings / roster / matchup data straight from ESPN's
private fantasy API, so the "Fantasy Football League" page stays current as
the real season plays out — no manual re-export needed.

Private leagues need the espn_s2 + SWID cookies from a logged-in browser
session at fantasy.espn.com (DevTools -> Application/Storage -> Cookies).
Public leagues work with just a league_id + season.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd
import streamlit as st

_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}"

# ESPN's internal roster-slot IDs -> readable labels (confirmed against this
# league's own rosterSettings.lineupSlotCounts).
SLOT_MAP = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "OP", 16: "D/ST", 17: "K",
    20: "BE", 21: "IR", 23: "FLEX",
}

# ESPN's internal *default* position IDs (player.defaultPositionId) — a
# different id space than the roster-slot ids above.
POS_ID_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}

# ESPN's internal pro-team IDs -> our standard abbreviations.
PRO_TEAM_MAP = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV",
    14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG",
    20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF",
    26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}


def espn_configured() -> bool:
    """True once league_id / season / espn_s2 / swid are present in secrets."""
    try:
        cfg = st.secrets.get("espn", {})
    except Exception:
        return False
    return bool(cfg.get("league_id")) and bool(cfg.get("season"))


def _get(season: int, league_id, espn_s2: str, swid: str, views: list) -> Optional[dict]:
    url = _BASE.format(season=season, league_id=league_id)
    params = "&".join(f"view={v}" for v in views)
    req = urllib.request.Request(
        f"{url}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cookie": f"espn_s2={espn_s2}; SWID={swid}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            st.error("ESPN rejected the request (401/403) — your espn_s2 / SWID "
                      "cookies have likely expired. Grab fresh ones from a "
                      "logged-in browser session and update .streamlit/secrets.toml.")
        else:
            st.error(f"ESPN API error {e.code} for league {league_id}.")
        return None
    except Exception as e:
        st.error(f"Could not reach ESPN: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=900)
def load_league(_ttl_bucket: int = 0) -> Optional[dict]:
    """Raw league payload: teams, rosters, standings, settings, matchup schedule.

    Cached for 15 minutes so normal browsing doesn't hammer ESPN; use the
    sidebar "Refresh Data" button to force an immediate re-fetch.
    """
    if not espn_configured():
        return None
    cfg = st.secrets["espn"]
    return _get(
        cfg["season"], cfg["league_id"],
        cfg.get("espn_s2", ""), cfg.get("swid", ""),
        views=["mTeam", "mRoster", "mSettings", "mStandings", "mMatchupScore"],
    )


def league_name(data: dict) -> str:
    return (data or {}).get("settings", {}).get("name", "").strip()


def draft_completed(data: dict) -> bool:
    return bool((data or {}).get("draftDetail", {}).get("drafted"))


def _owner_names(data: dict) -> dict:
    members = {m["id"]: m.get("displayName", "") for m in data.get("members", [])}
    out = {}
    for t in data.get("teams", []):
        names = [members.get(o, "") for o in t.get("owners", [])]
        out[t["id"]] = ", ".join(n for n in names if n)
    return out


def team_logos(data: dict) -> dict:
    """team_id -> logo URL."""
    return {t["id"]: t.get("logo", "") for t in (data or {}).get("teams", [])}


def team_names(data: dict) -> dict:
    """team_id -> display name."""
    return {t["id"]: t.get("name", "") for t in (data or {}).get("teams", [])}


def standings_df(data: dict) -> pd.DataFrame:
    """One row per team: record, points for/against, streak — sorted by wins."""
    if not data:
        return pd.DataFrame()
    owners = _owner_names(data)
    rows = []
    for t in data.get("teams", []):
        rec = t.get("record", {}).get("overall", {})
        streak_len = rec.get("streakLength", 0)
        streak_type = rec.get("streakType", "NONE")
        rows.append({
            "team_id": t["id"],
            "Logo": t.get("logo", ""),
            "Team": t.get("name", ""),
            "Owner": owners.get(t["id"], ""),
            "W": rec.get("wins", 0),
            "L": rec.get("losses", 0),
            "T": rec.get("ties", 0),
            "PF": round(rec.get("pointsFor", 0.0), 1),
            "PA": round(rec.get("pointsAgainst", 0.0), 1),
            "Streak": f"{'W' if streak_type == 'WIN' else 'L' if streak_type == 'LOSS' else '-'}{streak_len}"
                      if streak_len else "-",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["W", "PF"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def roster_df(data: dict, team_id: int) -> pd.DataFrame:
    """One row per rostered player on a given team (empty until the draft happens)."""
    if not data:
        return pd.DataFrame()
    team = next((t for t in data.get("teams", []) if t["id"] == team_id), None)
    if not team:
        return pd.DataFrame()
    rows = []
    for entry in team.get("roster", {}).get("entries", []):
        pool = entry.get("playerPoolEntry", {})
        p = pool.get("player", {})
        rows.append({
            "Slot": SLOT_MAP.get(entry.get("lineupSlotId"), str(entry.get("lineupSlotId", ""))),
            "Player": p.get("fullName", ""),
            "Pos": POS_ID_MAP.get(p.get("defaultPositionId", 0), ""),
            "NFL Team": PRO_TEAM_MAP.get(p.get("proTeamId", 0), "FA"),
            "Injured": "Yes" if p.get("injured") else "",
        })
    return pd.DataFrame(rows)


def matchups_df(data: dict) -> pd.DataFrame:
    """One row per matchup across the whole season schedule."""
    if not data:
        return pd.DataFrame()
    names = team_names(data)
    rows = []
    for m in data.get("schedule", []):
        home = m.get("home", {})
        away = m.get("away", {})
        rows.append({
            "week": m.get("matchupPeriodId"),
            "home_id": home.get("teamId"),
            "home_team": names.get(home.get("teamId"), ""),
            "home_pts": round(home.get("totalPoints", 0.0), 1),
            "away_id": away.get("teamId"),
            "away_team": names.get(away.get("teamId"), "") if away else "BYE",
            "away_pts": round(away.get("totalPoints", 0.0), 1) if away else None,
            "winner": m.get("winner", "UNDECIDED"),
        })
    return pd.DataFrame(rows)
