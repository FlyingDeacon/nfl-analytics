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

import base64
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


# Real managers behind ESPN's display names. Keyed on displayName rather than
# team name because teams get renamed most seasons while the account behind them
# doesn't — and Owner is what the all-time history aggregates on.
# Anyone missing here falls through to their ESPN handle.
OWNER_NAMES = {
    "flying deacon":     "Blake",
    "soypapad":          "Doug",
    "espnfan6929404931": "Cameron",
    "ayweav":            "Alaina",
    "milesiscool":       "Miles",
    "adeselms":          "Alyse",
    "meggie jeffreys":   "Meggie",
    "espnfan1656601808": "Alyssa",
    "compgeek52":        "Stephen",
    "espnfan7346069141": "Kristen",
    "espnfan5882418016": "James",
}


def _real_name(display_name: str) -> str:
    return OWNER_NAMES.get(display_name.strip().lower(), display_name)


# Results the league settled outside ESPN's bracket. rankCalculatedFinal has room
# for exactly one champion, so a shared title shows the co-champ as runner-up
# forever — including in the all-time title count. Keyed by owner real name.
CO_CHAMPIONS = {
    2022: {"Stephen"},   # season cut short by the Damar Hamlin game; title shared
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


@st.cache_data(show_spinner=False, ttl=86400)
def load_league_season(season: int) -> Optional[dict]:
    """Full payload for one season, including the draft. Cached for a day —
    completed seasons are immutable, so there's nothing to re-poll for."""
    if not espn_configured():
        return None
    cfg = st.secrets["espn"]
    return _get(
        season, cfg["league_id"], cfg.get("espn_s2", ""), cfg.get("swid", ""),
        views=["mTeam", "mRoster", "mSettings", "mStandings", "mMatchupScore",
               "mDraftDetail"],
    )


def previous_seasons(data: dict) -> list:
    """Completed seasons this league has played, oldest first."""
    return sorted((data or {}).get("status", {}).get("previousSeasons", []))


@st.cache_data(show_spinner=False, ttl=86400)
def season_players(season: int) -> pd.DataFrame:
    """Every player ESPN knows about for a season, with the two preseason
    opinions we care about and what actually happened:

      espn_proj_pts  — ESPN's own preseason projection (statSourceId 1)
      espn_rank      — ESPN's preseason PPR draft rank
      actual_pts     — what the player really scored (statSourceId 0)

    Used to name draft picks and to grade past drafts against ESPN.
    """
    if not espn_configured():
        return pd.DataFrame()
    cfg = st.secrets["espn"]
    # Must be league-scoped: the generic /players endpoint has no scoring
    # context, so every appliedTotal comes back null.
    url = _BASE.format(season=season, league_id=cfg["league_id"]) + "?view=kona_player_info"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Cookie": f"espn_s2={cfg.get('espn_s2','')}; SWID={cfg.get('swid','')}",
        "x-fantasy-filter": json.dumps({"players": {
            "limit": 1500,
            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "PPR"},
        }}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
    except Exception as e:
        st.warning(f"Couldn't load {season} player details from ESPN: {e}")
        return pd.DataFrame()

    rows = []
    for item in (payload if isinstance(payload, list) else payload.get("players", [])):
        p = item.get("player", item)
        proj = actual = None
        for s in p.get("stats", []):
            if s.get("statSplitTypeId") == 0 and s.get("seasonId") == season:
                if s.get("statSourceId") == 1:
                    proj = s.get("appliedTotal")
                elif s.get("statSourceId") == 0:
                    actual = s.get("appliedTotal")
        rows.append({
            "player_id": p.get("id"),
            "player": p.get("fullName", ""),
            "pos": POS_ID_MAP.get(p.get("defaultPositionId", 0), ""),
            "nfl_team": PRO_TEAM_MAP.get(p.get("proTeamId", 0), "FA"),
            "espn_proj_pts": round(proj, 1) if proj is not None else None,
            "espn_rank": (p.get("draftRanksByRankType") or {}).get("PPR", {}).get("rank"),
            "actual_pts": round(actual, 1) if actual is not None else None,
        })
    return pd.DataFrame(rows)


def final_standings_df(data: dict) -> pd.DataFrame:
    """Completed-season standings ordered by true final finish.

    ESPN keeps two different ranks: playoffSeed is where a team finished the
    regular season, rankCalculatedFinal is where it ended up after the
    playoffs. They often disagree — that gap is the whole story of a season.
    """
    if not data:
        return pd.DataFrame()
    owners = _owner_names(data)
    rows = []
    for t in data.get("teams", []):
        rec = t.get("record", {}).get("overall", {})
        rows.append({
            "Finish": t.get("rankCalculatedFinal") or t.get("playoffSeed"),
            "Seed": t.get("playoffSeed"),
            "Logo": _logo(t),
            "Team": t.get("name", ""),
            "Owner": owners.get(t["id"], ""),
            "W": rec.get("wins", 0),
            "L": rec.get("losses", 0),
            "T": rec.get("ties", 0),
            "PF": round(rec.get("pointsFor", 0.0), 1),
            "PA": round(rec.get("pointsAgainst", 0.0), 1),
            "team_id": t["id"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    co = CO_CHAMPIONS.get(int(data.get("seasonId") or 0), set())
    if co:
        df.loc[df["Owner"].isin(co), "Finish"] = 1

    return (df.sort_values(["Finish", "PF"], ascending=[True, False])
              .reset_index(drop=True))


def draft_picks_df(data: dict, season: int) -> pd.DataFrame:
    """Every pick of a completed draft, with player names resolved."""
    picks = (data or {}).get("draftDetail", {}).get("picks", [])
    if not picks:
        return pd.DataFrame()
    names = team_names(data)
    df = pd.DataFrame([{
        "Rd": p.get("roundId"),
        "Pick": p.get("roundPickNumber"),
        "Overall": p.get("overallPickNumber"),
        "team_id": p.get("teamId"),
        "Team": names.get(p.get("teamId"), ""),
        "player_id": p.get("playerId"),
        "Keeper": bool(p.get("keeper")),
    } for p in picks])

    pool = season_players(season)
    if not pool.empty:
        df = df.merge(
            pool[["player_id", "player", "pos", "nfl_team",
                  "espn_proj_pts", "espn_rank", "actual_pts"]],
            on="player_id", how="left",
        )
    return df.sort_values("Overall").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def all_time_df(seasons: tuple) -> pd.DataFrame:
    """Franchise history across every completed season, keyed by owner.

    Owners are the stable identity here — team names get renamed most years,
    so aggregating on them would split one manager into several franchises.
    """
    rows = []
    for season in seasons:
        data = load_league_season(season)
        standings = final_standings_df(data)
        for _, t in standings.iterrows():
            rows.append({
                "Owner": t["Owner"], "Season": season, "Team": t["Team"],
                "Finish": t["Finish"], "W": t["W"], "L": t["L"], "T": t["T"],
                "PF": t["PF"], "PA": t["PA"],
            })
    hist = pd.DataFrame(rows)
    if hist.empty:
        return hist

    agg = hist.groupby("Owner", as_index=False).agg(
        Seasons=("Season", "count"), W=("W", "sum"), L=("L", "sum"), T=("T", "sum"),
        PF=("PF", "sum"), PA=("PA", "sum"),
        Best=("Finish", "min"), Worst=("Finish", "max"), AvgFinish=("Finish", "mean"),
    )
    agg["Titles"] = agg["Owner"].map(
        hist[hist["Finish"] == 1].groupby("Owner").size()).fillna(0).astype(int)
    agg["Win%"] = (agg["W"] / (agg["W"] + agg["L"] + agg["T"]).clip(lower=1)).round(3)
    agg["AvgFinish"] = agg["AvgFinish"].round(2)
    agg["PF"] = agg["PF"].round(1)
    agg["PA"] = agg["PA"].round(1)
    return agg.sort_values(["Titles", "Win%"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def h2h_df(seasons: tuple) -> pd.DataFrame:
    """All-time head-to-head record between every pair of owners."""
    tally: dict = {}
    for season in seasons:
        data = load_league_season(season)
        owners = _owner_names(data)
        for m in (data or {}).get("schedule", []):
            home, away = m.get("home") or {}, m.get("away") or {}
            hid, aid = home.get("teamId"), away.get("teamId")
            winner = m.get("winner", "UNDECIDED")
            if not hid or not aid or winner == "UNDECIDED":
                continue
            ho, ao = owners.get(hid, ""), owners.get(aid, "")
            if not ho or not ao or ho == ao:
                continue
            for a, b, won in ((ho, ao, winner == "HOME"), (ao, ho, winner == "AWAY")):
                rec = tally.setdefault((a, b), [0, 0])
                rec[0 if won else 1] += 1

    rows = [{"Owner": a, "Opponent": b, "W": w, "L": l,
             "Win%": round(w / max(w + l, 1), 3)}
            for (a, b), (w, l) in tally.items()]
    return pd.DataFrame(rows)


def league_name(data: dict) -> str:
    return (data or {}).get("settings", {}).get("name", "").strip()


def draft_completed(data: dict) -> bool:
    return bool((data or {}).get("draftDetail", {}).get("drafted"))


def _owner_names(data: dict) -> dict:
    members = {m["id"]: _real_name(m.get("displayName", ""))
               for m in data.get("members", [])}
    out = {}
    for t in data.get("teams", []):
        names = [members.get(o, "") for o in t.get("owners", [])]
        out[t["id"]] = ", ".join(n for n in names if n)
    return out


# Host ESPN serves user-uploaded team logos from. Unlike the stock logo packs on
# g.espncdn.com, it 401s without the league cookies — and a browser won't attach
# those to a cross-site <img> request, so those logos silently break.
_UPLOAD_HOST = "mystique-api.fantasy.espn.com"


@st.cache_data(show_spinner=False, ttl=86400)
def _logo_data_uri(url: str) -> str:
    """Fetch an authenticated custom logo here and inline it as a data URI."""
    cfg = st.secrets["espn"]
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Cookie": f"espn_s2={cfg.get('espn_s2', '')}; SWID={cfg.get('swid', '')}",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            mime = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
            if mime == "image/jpg":
                mime = "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(r.read()).decode()
    except Exception:
        return ""


def _logo(team: dict) -> str:
    """Team logo URL, with custom uploads inlined so they actually render."""
    url = team.get("logo", "") or ""
    return _logo_data_uri(url) if _UPLOAD_HOST in url else url


def team_logos(data: dict) -> dict:
    """team_id -> logo URL."""
    return {t["id"]: _logo(t) for t in (data or {}).get("teams", [])}


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
            "Logo": _logo(t),
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


def _scoring_format(settings: dict) -> str:
    """PPR / Half PPR / Standard, read off the league's points-per-reception rule."""
    items = settings.get("scoringSettings", {}).get("scoringItems", [])
    ppr = next((i.get("points", 0.0) for i in items if i.get("statId") == 53), 0.0)
    if ppr >= 1.0:
        return "PPR"
    return "Half PPR" if ppr > 0 else "Standard"


def draft_setup(data: dict) -> Optional[dict]:
    """League facts the practice draft simulator needs to mirror the real draft:
    team count, rounds, snake vs linear, per-slot team names, scoring format and
    the logged-in user's own draft slot.

    Slots are 1-based positions in ESPN's draftSettings.pickOrder (a list of
    team ids), so slot 1 is whoever holds the first overall pick.
    """
    if not data:
        return None
    settings = data.get("settings", {})
    draft = settings.get("draftSettings", {})
    names = team_names(data)

    order = [tid for tid in draft.get("pickOrder", []) if tid in names]
    if not order:  # order isn't published until the commissioner sets it
        order = sorted(names)
    slot_names = {i: names.get(tid, f"Team {i}") for i, tid in enumerate(order, start=1)}

    # Each roster spot is one draft round, except IR (slot 21), which isn't drafted.
    counts = settings.get("rosterSettings", {}).get("lineupSlotCounts", {})
    rounds = sum(int(v) for k, v in counts.items() if str(k) != "21")

    my_swid = st.secrets["espn"].get("swid", "")
    my_team = next(
        (t["id"] for t in data.get("teams", []) if my_swid in t.get("owners", [])), None
    )

    return {
        "league_name": league_name(data),
        "teams": len(order),
        "rounds": rounds,
        "snake": draft.get("type", "SNAKE") == "SNAKE",
        "slot_names": slot_names,
        "my_slot": order.index(my_team) + 1 if my_team in order else None,
        "scoring": _scoring_format(settings),
    }


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
