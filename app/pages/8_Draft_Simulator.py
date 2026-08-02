from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.styles import NFL_CSS
from utils.nav import render_sidebar_nav
from utils.data_loader import load_teams, load_weekly, get_logo, get_base_dir, _file_mtime
from utils.espn_league import (espn_configured, load_league, draft_setup, live_draft,
                               load_league_season, previous_seasons, season_players,
                               team_names)
from utils.draft_live import (resolve_players, replay, available, id_to_player,
                              normalize_pos, pick_made)

st.set_page_config(page_title="Draft Simulator · NFL", page_icon="🏈", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

render_sidebar_nav(current_page="8_Draft_Simulator")

if st.button("← Back to Fantasy Football", key="ds_back_btn"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🏈</div>
    <div>
        <div class="title">Practice Draft Simulator</div>
        <div class="subtitle">Robots draft ESPN best-available · you draft from your own big board</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Big board loading (built on demand) ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BIG_BOARD_DIR = ROOT_DIR / "data" / "derived"
BUILD_SCRIPT = ROOT_DIR / "scripts" / "build_big_boards.py"
SCHEDULE_CSV = ROOT_DIR / "data" / "raw" / "schedules.csv"
SCORING_FORMATS = ["PPR", "Half PPR", "Standard"]
DRAFT_SEASON = 2026

# Board team abbreviations vs. nflverse schedule abbreviations (only differences).
_SCHED_TEAM_ALIAS = {"LAR": "LA", "LA": "LA", "JAC": "JAX"}


@st.cache_data(show_spinner=False)
def _bye_weeks() -> dict:
    """Map each team abbreviation → its 2026 regular-season bye week.

    Derived from the schedule: a team's bye is the one regular-season week it
    has no game. Returns {} if the schedule is unavailable so the sim degrades
    gracefully (bye columns just show blank).
    """
    try:
        s = pd.read_csv(SCHEDULE_CSV)
    except Exception:
        return {}
    s = s[(s["season"] == DRAFT_SEASON) & (s.get("game_type", "REG") == "REG")]
    if s.empty:
        return {}
    all_weeks = set(range(1, int(s["week"].max()) + 1))
    byes: dict = {}
    for t in set(s["home_team"]) | set(s["away_team"]):
        played = set(s[(s["home_team"] == t) | (s["away_team"] == t)]["week"])
        missing = sorted(all_weeks - played)
        byes[str(t)] = int(missing[0]) if missing else None
    return byes


@st.cache_data(show_spinner=False)
def _headshots() -> dict:
    """Map player_display_name → most recent headshot_url (from weekly.csv).

    Rookies with no NFL games yet simply won't have an entry — the image
    column just renders blank for them.
    """
    _base = get_base_dir()
    wk = load_weekly(_mtime=_file_mtime(_base / "data" / "raw" / "weekly.csv"))
    if wk.empty or "headshot_url" not in wk.columns:
        return {}
    wk = wk.sort_values(["season", "week"])
    return wk.groupby("player_display_name")["headshot_url"].last().dropna().to_dict()


@st.cache_data(show_spinner=False)
def _team_logos() -> dict:
    """Map team abbreviation → logo URL."""
    _base = get_base_dir()
    teams_df = load_teams(_mtime=_file_mtime(_base / "data" / "raw" / "teams.csv"))
    abbr_col = "team_abbr" if "team_abbr" in teams_df.columns else "team"
    logos: dict = {}
    for t in teams_df[abbr_col].dropna().unique():
        url = get_logo(t, teams_df)
        if url:
            logos[str(t)] = url
    return logos


def _team_bye(team: str, byes: dict) -> int | None:
    """Bye week for a board team abbr, resolving the LA/LAR-style aliases."""
    t = str(team)
    return byes.get(t, byes.get(_SCHED_TEAM_ALIAS.get(t, t)))


def _board_path(scoring: str) -> Path:
    return BIG_BOARD_DIR / f"big_board_{scoring.replace(' ', '_')}.parquet"


def _build_board(scoring: str) -> bool:
    """Generate the big board parquet headlessly (same pipeline as the
    Predictions page). Returns True on success."""
    try:
        subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--scoring", scoring],
            cwd=str(ROOT_DIR), check=True, capture_output=True, timeout=300,
        )
    except Exception:
        return False
    return _board_path(scoring).exists()


# Positions the finished board must contain; a board missing any of these is
# stale (built before that position was added) and gets rebuilt automatically.
_REQUIRED_BOARD_POS = ("QB", "RB", "WR", "TE", "DEF", "K")


def _board_is_current(path: Path) -> bool:
    """True if the parquet exists and already carries every required position
    (e.g. defenses/kickers). Guards against a board written by older code."""
    if not path.exists():
        return False
    try:
        have = set(pd.read_parquet(path, columns=["pos"])["pos"].unique())
    except Exception:
        return False
    return all(p in have for p in _REQUIRED_BOARD_POS)


def _ensure_board(scoring: str) -> Path | None:
    """Return the parquet path, (re)building it when it's missing or stale.

    A stale board is one produced before defenses/kickers existed — it would
    silently lack a required position. The rebuild runs build_big_boards.py in a
    fresh subprocess, so it uses the *current* pipeline even if this Streamlit
    process is still holding older module code in memory.
    """
    path = _board_path(scoring)
    if _board_is_current(path):
        return path
    with st.spinner(f"Building your {scoring} big board…"):
        ok = _build_board(scoring)
    return path if (ok and _board_is_current(path)) else (path if path.exists() else None)


@st.cache_data(show_spinner=False)
def _load_board(path: str, mtime: float) -> pd.DataFrame:
    # mtime is part of the cache key so a rebuilt board invalidates the cache.
    df = pd.read_parquet(path)
    # VOR-sorted board = the user's big board order.
    df = df.sort_values("vor", ascending=False).reset_index(drop=True)
    df["my_rank"] = range(1, len(df) + 1)
    return df


# ── Draft settings ───────────────────────────────────────────────────────────
POSITIONS = ("QB", "RB", "WR", "TE", "DEF", "K")
POS_CAPS = {"QB": 2, "RB": 8, "WR": 8, "TE": 3, "DEF": 2, "K": 2}  # max per roster
REQ_MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DEF": 1, "K": 1}   # starters that must be filled
STARTER_TARGET = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DEF": 1, "K": 1}  # weekly starters (+1 FLEX)
STARTER_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "K"]
FLEX_POS = ("RB", "WR", "TE")
FLEX_TOTAL = STARTER_TARGET["RB"] + STARTER_TARGET["WR"] + STARTER_TARGET["TE"] + 1  # 6

# Robot draft variance: how many ESPN-rank "spots" of gaussian noise to add to
# each available player before a robot takes its best-available. 0 = pure chalk
# (always the ADP top); larger = more real-draft reaches and slides.
_ROBOT_SIGMA_BY_STYLE = {"Chalk (strict ADP)": 0.0, "Realistic": 6.0, "Wild": 12.0}

settings_locked = st.session_state.get("dr_started", False)

# Real-league setup (team names, pick order, roster size, scoring) so a mock can
# mirror your actual draft instead of using generic "Team 1..N" placeholders.
_league_setup = None
if espn_configured():
    _lg_data = load_league()
    if _lg_data:
        _league_setup = draft_setup(_lg_data)

with st.expander("⚙️ Draft Settings", expanded=not settings_locked):
    league_opts = ["Custom mock"]
    if _league_setup:
        league_opts.append(_league_setup["league_name"] or "My ESPN league")
    league_choice = st.selectbox(
        "League", league_opts, index=len(league_opts) - 1,
        disabled=settings_locked or len(league_opts) == 1, key="ds_league",
        help="Pick your ESPN league to inherit its real team names, draft order, "
             "roster size and scoring. Custom mock lets you set everything by hand.",
    )
    use_league = _league_setup is not None and league_choice != "Custom mock"

    if use_league:
        _s = _league_setup
        teams, rounds, scoring = _s["teams"], _s["rounds"], _s["scoring"]
        order_type = "Snake" if _s["snake"] else "Linear"
        st.caption(
            f"**{teams} teams** · **{rounds} rounds** · **{order_type}** · "
            f"**{scoring}** — pulled live from ESPN."
        )
        slot = st.selectbox(
            "Your draft slot", list(range(1, teams + 1)),
            index=(_s["my_slot"] or 1) - 1, disabled=settings_locked, key="ds_slot_lg",
            format_func=lambda x: f"{x}. {_s['slot_names'].get(x, '')}"
                                  + ("  ← you" if x == _s["my_slot"] else ""),
            help="Defaults to your real slot in the league's pick order — change it "
                 "to practice drafting from somewhere else.",
        )
        slot_names = dict(_s["slot_names"])
    else:
        c1, c2, c3 = st.columns(3)
        scoring = c1.selectbox("Scoring", SCORING_FORMATS, index=0,
                               disabled=settings_locked, key="ds_scoring")
        teams = c2.selectbox("Teams", [8, 9, 10, 11, 12, 13, 14], index=4,
                             disabled=settings_locked, key="ds_teams")
        rounds = c3.selectbox("Rounds", list(range(10, 21)), index=6,
                              disabled=settings_locked, key="ds_rounds")

        c4, c5 = st.columns(2)
        slot = c4.selectbox("Your draft slot", list(range(1, teams + 1)), index=min(5, teams - 1),
                            disabled=settings_locked, key="ds_slot")
        order_type = c5.radio("Draft order", ["Snake", "Linear"], horizontal=True,
                              disabled=settings_locked, key="ds_order_type")
        slot_names = {}

    # Live mirrors the real draft, so it only makes sense against a real league.
    _mode_opts = ["Standard (robots auto-draft)", "Manual (you pick for every team)"]
    if use_league:
        _mode_opts.append("Live (mirror my real ESPN draft)")
    draft_mode = st.radio(
        "Draft mode", _mode_opts,
        horizontal=True, disabled=settings_locked, key="ds_mode",
        help="Manual lets you make every team's selection yourself — useful for "
             "running a mock where you control the whole room. Live stops "
             "simulating entirely and follows your real ESPN draft pick by pick, "
             "so the board and the suggestions track the actual room.",
    )

    robot_style = st.radio(
        "Robot draft style",
        list(_ROBOT_SIGMA_BY_STYLE.keys()), index=1,
        horizontal=True, disabled=settings_locked, key="ds_robot_style",
        help="How far the robots stray from ESPN best-available. Realistic adds "
             "draft-day variance (reaches and slides); Wild swings harder; Chalk "
             "always takes the ADP top. Roster rules are always respected.",
    )


def _snake_order(teams: int, rounds: int, snake: bool) -> list:
    """Return a flat list of team slots (1-based) for each overall pick."""
    order = []
    for rd in range(rounds):
        seq = list(range(1, teams + 1))
        if snake and rd % 2 == 1:
            seq = seq[::-1]
        order.extend(seq)
    return order


# ── Draft Watch (read-only ESPN sync) ────────────────────────────────────────
# Mirrors a real ESPN draft onto the big board without touching the simulator's
# own state. Two jobs: watch your live draft, and — by replaying a season that
# already finished — prove the player-id linking actually holds *before* draft
# day, when a silent miss would be expensive.
@st.cache_data(show_spinner=False)
def _resolve_links(board_path: str, mtime: float, season: int) -> pd.DataFrame:
    """Board <-> ESPN playerId links. Keyed on the board's mtime so a rebuilt
    board re-links automatically."""
    return resolve_players(_load_board(board_path, mtime), season_players(season))


with st.expander("🔍 Draft Watch (beta) — read-only ESPN sync", expanded=False):
    if not espn_configured():
        st.info("Add your ESPN league to `.streamlit/secrets.toml` to use Draft Watch.")
    elif not _board_is_current(_board_path(scoring)):
        st.warning(f"No **{scoring}** big board yet — hit Start Draft once (or open "
                   "the Predictions page) to build it, then come back.")
    else:
        _done = previous_seasons(_lg_data or {})
        _wseasons = [DRAFT_SEASON] + [s for s in reversed(_done) if s != DRAFT_SEASON]
        wcol1, wcol2 = st.columns([3, 1])
        wseason = wcol1.selectbox(
            "Season", _wseasons, index=0, key="ds_watch_season",
            format_func=lambda s: f"{s} (live)" if s == DRAFT_SEASON else f"{s} (replay)",
            help="Live watches this year's draft as it happens. A past season replays "
                 "a finished draft — the dry run that tells you the linking is sound.",
        )
        if wcol2.button("↻ Refresh", use_container_width=True, key="ds_watch_refresh"):
            live_draft.clear()
            st.rerun()

        _wpath = _board_path(scoring)
        _wboard = _load_board(str(_wpath), _wpath.stat().st_mtime)
        links = _resolve_links(str(_wpath), _wpath.stat().st_mtime, wseason)

        # Coverage first: an unlinked board player can never be marked drafted,
        # so he would stay "available" all draft and keep getting recommended.
        _miss = links[links["player_id"].isna()]
        _soft = links["method"].isin(["fuzzy", "alias", "name"]).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Board players linked", f"{len(links) - len(_miss)}/{len(links)}")
        m2.metric("Non-exact links", int(_soft))
        m3.metric("Unlinked", len(_miss))

        if len(_miss) and wseason == DRAFT_SEASON:
            st.error(
                f"{len(_miss)} board player(s) have no ESPN id. If one of them is "
                "drafted, Draft Watch won't see it and he'll stay on the board. Add "
                "the ESPN spelling to `NAME_ALIASES` in `app/utils/draft_live.py`."
            )
        elif len(_miss):
            st.info(
                f"{len(_miss)} board player(s) are absent from {wseason}'s ESPN pool — "
                "on a replay these are nearly all rookies ESPN had no record of back "
                "then. Only the live season's count speaks to draft-day safety."
            )
        if len(_miss):
            st.dataframe(_miss[["player", "pos", "team", "suspect"]],
                         hide_index=True, use_container_width=True)
        if _soft:
            with st.popover(f"Review {int(_soft)} non-exact link(s)"):
                st.dataframe(
                    links[links["method"].isin(["fuzzy", "alias", "name"])]
                    [["player", "pos", "team", "espn_name", "method", "score"]],
                    hide_index=True, use_container_width=True)

        # Pick feed. Completed seasons carry draftDetail in the season payload;
        # the current one needs the narrow, pollable endpoint.
        if wseason == DRAFT_SEASON:
            _wdata, _detail = _lg_data, live_draft()
            if _detail is None:
                st.warning("Couldn't reach ESPN just now — try Refresh. If this keeps "
                           "up, your espn_s2 / SWID cookies have probably expired.")
        else:
            _wdata = load_league_season(wseason)
            _detail = (_wdata or {}).get("draftDetail")

        # ESPN publishes every slot up front with playerId -1, so the array
        # length is the size of the draft and replay() reports the progress.
        _slots = (_detail or {}).get("picks", [])
        rep = replay(_slots, links, _wboard)
        if rep.empty:
            st.info(f"No picks yet — ESPN has {len(_slots)} slots queued and fills "
                    "them in as the draft happens.")
        else:
            _names = team_names(_wdata) if _wdata else {}
            _off = int((~rep["on_board"]).sum())
            st.caption(
                f"**{len(rep)} of {len(_slots)} picks made** · {len(rep) - _off} on your "
                f"big board · {_off} off it"
                + ("  ·  draft complete" if (_detail or {}).get("drafted")
                   else "  ·  **in progress**" if (_detail or {}).get("inProgress") else "")
            )

            show = rep.iloc[::-1].head(15).copy()
            show["Team"] = show["team_id"].map(lambda t: _names.get(t, f"Team {t}"))
            show["Player"] = show.apply(
                lambda r: r["player"] if r["on_board"] else "— off board —", axis=1)
            st.dataframe(
                show[["overall", "round", "Team", "Player", "pos", "my_rank", "vor"]]
                .rename(columns={"overall": "Pick", "round": "Rd", "pos": "Pos",
                                 "my_rank": "My Rank", "vor": "VOR"}),
                hide_index=True, use_container_width=True,
            )

            st.markdown("**Best available on your board**")
            st.dataframe(
                available(_wboard, rep).head(10)
                [["my_rank", "player", "pos", "team", "vor"]]
                .rename(columns={"my_rank": "My Rank", "player": "Player", "pos": "Pos",
                                 "team": "Team", "vor": "VOR"}),
                hide_index=True, use_container_width=True,
            )


# ── Start / reset ────────────────────────────────────────────────────────────
cstart, creset = st.columns([1, 1])
if not settings_locked:
    _mode = ("live" if draft_mode.startswith("Live")
             else "manual" if draft_mode.startswith("Manual") else "robots")
    if cstart.button("🔴 Connect Live Draft" if _mode == "live" else "🚀 Start Draft",
                     type="primary", use_container_width=True, key="ds_start"):
        path = _ensure_board(scoring)
        if path is None:
            st.error(
                "Couldn't build the big board. Open **🔮 2026 Fantasy Predictions** "
                "once to generate it, then come back."
            )
            st.stop()
        board = _load_board(str(path), path.stat().st_mtime)

        dr_order = _snake_order(teams, rounds, order_type == "Snake")
        live_slots: dict = {}
        if _mode == "live":
            # ESPN publishes every pick slot with its owning team before anyone
            # picks, so take the running order straight from that instead of
            # re-deriving snake/linear here. Theirs already accounts for traded
            # picks and any custom order the commissioner set.
            _detail = live_draft()
            _slots = sorted((_detail or {}).get("picks", []),
                            key=lambda p: p.get("overallPickNumber") or 0)
            live_slots = {tid: i for i, tid
                          in enumerate(_league_setup["pick_order"], start=1)}
            _seq = [live_slots.get(p.get("teamId")) for p in _slots]
            if not _seq or any(s is None for s in _seq):
                st.error(
                    "ESPN hasn't published a usable pick order for this draft yet. "
                    "That normally appears once the commissioner sets the draft "
                    "order — try again closer to draft day, or run a Standard mock "
                    "in the meantime."
                )
                st.stop()
            dr_order = _seq
            rounds = -(-len(_seq) // teams)

        st.session_state.dr_started = True
        st.session_state.dr_scoring = scoring
        st.session_state.dr_teams = teams
        st.session_state.dr_rounds = rounds
        st.session_state.dr_slot = slot
        st.session_state.dr_team_names = slot_names
        st.session_state.dr_mode = _mode
        st.session_state.dr_robot_sigma = _ROBOT_SIGMA_BY_STYLE[robot_style]
        st.session_state.dr_snake = (order_type == "Snake")
        st.session_state.dr_order = dr_order
        st.session_state.dr_board = board.to_dict("records")
        st.session_state.dr_drafted = {}          # player -> team slot
        st.session_state.dr_picks = []            # list of pick dicts
        st.session_state.dr_pick_idx = 0
        st.session_state.dr_live_slots = live_slots   # ESPN team id -> draft slot
        st.session_state.dr_live_seen = -1            # picks seen on the last sync
        st.session_state.dr_live_at = 0.0             # time of last good sync
        st.session_state.pop("ds_roster_view", None)
        st.rerun()
    st.stop()

if creset.button("🔄 Reset Draft", use_container_width=True, key="ds_reset"):
    for k in ("dr_started", "dr_order", "dr_board", "dr_drafted",
              "dr_picks", "dr_pick_idx", "dr_mode", "dr_robot_sigma",
              "dr_team_names", "ds_roster_view", "dr_live_slots",
              "dr_live_seen", "dr_live_at", "dr_live_stale", "dr_live_complete"):
        st.session_state.pop(k, None)
    st.rerun()

# ── Live draft state ─────────────────────────────────────────────────────────
teams = st.session_state.dr_teams
rounds = st.session_state.dr_rounds
user_slot = st.session_state.dr_slot
order = st.session_state.dr_order
total_picks = len(order)
board = pd.DataFrame(st.session_state.dr_board)
drafted = st.session_state.dr_drafted
mode = st.session_state.get("dr_mode", "robots")
manual = mode == "manual"     # you pick for every team
live = mode == "live"         # every pick comes from the real ESPN draft
team_names_by_slot = st.session_state.get("dr_team_names", {})


def _team_label(slot: int, short: bool = False) -> str:
    """Real league team name when the mock is league-backed, else 'Team N'."""
    name = team_names_by_slot.get(slot)
    if not name:
        return f"T{slot}" if short else f"Team {slot}"
    return name[:14] if short else name

# Attach each player's 2026 bye week (from the schedule) to the board so it can
# be shown and factored into suggestions. Recomputed per rerun since dr_board is
# persisted without it.
_BYES = _bye_weeks()
board["bye"] = board["team"].map(lambda t: _team_bye(t, _BYES))
_BYE_BY_PLAYER = dict(zip(board["player"], board["bye"]))

# Player headshots + team logos, used to replace name-only / abbreviation-only
# columns with images throughout the board and roster tables.
_HEADSHOTS = _headshots()
_TEAM_LOGOS = _team_logos()


def _bye_label(player: str) -> str:
    b = _BYE_BY_PLAYER.get(player)
    return "" if b is None or pd.isna(b) else str(int(b))


def _team_roster(slot: int) -> list:
    """Positions drafted by a given team slot (in pick order)."""
    return [p["pos"] for p in st.session_state.dr_picks if p["team"] == slot]


def _pos_counts(slot: int) -> dict:
    roster = _team_roster(slot)
    return {p: roster.count(p) for p in POSITIONS}


def _needed_positions(slot: int) -> list:
    """Required starter positions this team has not yet filled."""
    counts = _pos_counts(slot)
    return [p for p in POSITIONS if counts[p] < REQ_MIN[p]]


def _is_must_fill(slot: int) -> bool:
    """True when remaining picks are exactly enough to fill the required
    starters — so the team must draft a needed position now or it can never
    ice a legal lineup."""
    counts = _pos_counts(slot)
    picks_left = rounds - len(_team_roster(slot))
    unmet = sum(max(0, REQ_MIN[p] - counts[p]) for p in POSITIONS)
    return picks_left <= unmet


# ══════════════════════════════════════════════════════════════════════════════
# SUGGESTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# The engine answers "who would I most regret losing?", not "who is best?".
# Every input collapses into a single score:
#
#   score = VOR
#         + _VONA_W × (VOR − expected best at that position at your next pick)
#         + roster-construction bonuses, each sized to the live VOR spread
#
# The middle term is what makes this more than a sorted list. The robots pick by
# ESPN rank plus gaussian noise (see _robot_pick), so the picks between now and
# your next turn can be *simulated* rather than guessed at. That yields both the
# odds each player survives and how much value is left at his position when you
# are back on the clock — which is why a player the field ranks far below you
# stops looking like a bargain: he'll still be sitting there next turn.

# Bonus weights, expressed as MULTIPLES of the live VOR spread rather than as
# absolute points. A flat "+30 VOR" bonus is a rounding error in round 1 and a
# landslide in round 12, when the whole remaining board sits within a few
# points; scaling keeps each nudge worth the same *relative* amount throughout.
_NEED_W  = 0.90   # fills an open STARTING slot
_FLEX_W  = 0.42   # fills the FLEX once core RB/WR/TE are set
_STACK_W = 1.35   # QB2 / TE2 / K2 taken before you need the depth
_BYE_W   = 0.35   # bye-week congestion, per colliding player
_RUN_W   = 0.60   # position coming off the board faster than expected
_TIER_W  = 0.50   # last player left in his tier
_CORR_W  = 0.15   # pass-catcher stacked with your QB (or vice versa)
_CUFF_W  = 0.35   # handcuff behind an RB you already roster
_AVAIL_W = 0.12   # projected-games nudge — VOR already prices games, so this
                  # only ever acts as a tiebreaker between similar players
_REACH_W = 0.24   # per round, for taking your first QB/TE before its window

_VONA_W  = 0.65   # weight on scarcity vs raw value. Drafting on pure scarcity
                  # over-corrects and grabs mediocre players from thin
                  # positions, so raw value keeps the larger share.
_K_PRIORITY = 500.0  # once it's the kicker's turn, lift it above bench value
                     # (K/DEF VOR is deeply negative, so the need bonus alone
                     # can't out-rank a high-VOR bench player) — beats any
                     # realistic VOR gap but stays below _BLOCK
_BLOCK      = 1e6    # effectively removes a player from consideration
_MIN_SCALE  = 8.0    # floor under the VOR spread, so the bonuses don't vanish
                     # once the late-round board flattens out
_SURVIVAL_SIMS = 400
_TIER_MIN_GAP  = 6.0 # VOR points — the smallest gap that may start a new tier

# Positional draft windows distilled from recent (2023-24) championship-roster
# ADP trends: title teams anchored rounds 1-3 with elite RB/WR, built RB/WR
# depth through the mid rounds, treated an every-week QB as a round ~5-11 value
# (unless it was a truly top-tier arm), took a top-3 TE at a mid pick or else
# waited, and left kickers for the final two rounds. We only *penalize
# reaching* — taking your first QB or TE well before its window — so a genuinely
# elite player (huge VOR) can still beat the penalty, but the engine won't burn
# a premium pick on a replaceable starter at a scarce single-slot position.
_REACH_ROUND = {"QB": 4, "TE": 3}   # earliest round to prioritize QB1 / TE1


def _vor_scale(avail: pd.DataFrame) -> float:
    """Spread of the talent still on the board — the unit every bonus is in."""
    v = avail["vor"].to_numpy(dtype=float)
    if v.size == 0:
        return _MIN_SCALE
    top = np.sort(v)[::-1][: max(12, 3 * teams)]
    return max(float(top.std()), _MIN_SCALE)


def _picks_until_next(slot: int, idx: int) -> int:
    """How many picks other teams make before this slot is back on the clock."""
    for n, s in enumerate(order[idx + 1:], start=1):
        if s == slot:
            return n - 1
    return 0


def _survival(avail: pd.DataFrame, n_picks: int) -> np.ndarray:
    """P(each available player is still there at your next pick).

    The robots' selection rule is known exactly — ESPN overall rank plus
    gaussian noise — so rather than guessing at ADP we simulate that same rule.
    Each trial draws one noisy ranking of the board and assumes the next
    `n_picks` selections come off the top of it; a player survives when he falls
    outside that cut. K/DEF sit at rank 1e9 and so always survive, matching how
    the robots actually defer them. Seeded on the pick index so the number a
    user sees doesn't flicker between Streamlit reruns of the same pick.
    """
    n = len(avail)
    if n == 0:
        return np.zeros(0)
    if n_picks <= 0:
        return np.ones(n)
    if n_picks >= n:
        return np.zeros(n)
    espn = pd.to_numeric(avail["espn_overall"], errors="coerce").fillna(1e9).to_numpy(dtype=float)
    sigma = float(st.session_state.get("dr_robot_sigma", 6.0))
    if sigma <= 0:                       # chalk mode: the cut is deterministic
        return (np.argsort(np.argsort(espn)) >= n_picks).astype(float)
    rng = np.random.default_rng(1_000 * int(st.session_state.dr_pick_idx) + n_picks)
    noisy = espn[None, :] + rng.normal(0.0, sigma, size=(_SURVIVAL_SIMS, n))
    cut = np.partition(noisy, n_picks - 1, axis=1)[:, n_picks - 1]
    return (noisy > cut[:, None]).mean(axis=0)


def _next_turn_replacement(avail: pd.DataFrame, surv: np.ndarray) -> dict:
    """Expected VOR of the best player left at each position at your next pick.

    Walks each position best-first and accumulates
    ``Σ vor_i · P(i survives) · Π_{j<i} P(j is gone)`` — the expected value of
    whoever is still on top of that position when you pick again. Subtracting it
    from a candidate's VOR gives his value over *next-turn* replacement, i.e.
    what taking him now actually buys you over waiting.
    """
    out = {}
    v_all = avail["vor"].to_numpy(dtype=float)
    pos_all = avail["pos"].to_numpy()
    for p in POSITIONS:
        idx = np.flatnonzero(pos_all == p)
        if idx.size == 0:
            out[p] = 0.0
            continue
        idx = idx[np.argsort(v_all[idx])[::-1]]
        v, s = v_all[idx], surv[idx]
        gone = np.concatenate(([1.0], np.cumprod(1.0 - s)[:-1]))
        out[p] = float((v * s * gone).sum())
    return out


def _position_tiers(avail: pd.DataFrame) -> dict:
    """Group each position's remaining players into tiers → {player: (tier, size, cliff)}.

    Drafters think in tiers, not on a continuous ladder: six WRs within four
    points are interchangeable, so the right move is to take the lone RB sitting
    above a 40-point drop. A tier break is a player-to-player gap that is
    unusually large *for that position right now*, which adapts to both the
    scoring format and how picked-over the position already is.
    """
    out: dict = {}
    v_all = avail["vor"].to_numpy(dtype=float)
    pos_all = avail["pos"].to_numpy()
    names = avail["player"].to_numpy()
    for p in POSITIONS:
        idx = np.flatnonzero(pos_all == p)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(v_all[idx])[::-1]]
        v = v_all[idx]
        gaps = -np.diff(v)
        thresh = max(_TIER_MIN_GAP, 2.0 * float(np.median(gaps))) if gaps.size else np.inf
        tier = np.zeros(v.size, dtype=int)
        for b in np.flatnonzero(gaps > thresh):
            tier[b + 1:] += 1
        for t in range(int(tier.max()) + 1):
            members = np.flatnonzero(tier == t)
            below = np.flatnonzero(tier == t + 1)
            floor_v = float(v[below[0]]) if below.size else 0.0
            for m in members:
                out[names[idx[m]]] = (t + 1, int(members.size), float(v[m] - floor_v))
    return out


def _run_heat(avail: pd.DataFrame) -> dict:
    """How much faster than expected each position is coming off the board.

    ESPN rank — which the survival model runs on — is a static pre-draft view
    and cannot see a live positional run. Comparing the last round of actual
    picks against the composition of the players who *should* have gone next
    catches the case where RBs are flying and the cliff arrives early.
    """
    look = max(6, teams)
    recent = st.session_state.dr_picks[-look:]
    if len(recent) < look:
        return {p: 0.0 for p in POSITIONS}
    espn = pd.to_numeric(avail["espn_overall"], errors="coerce").fillna(1e9).to_numpy(dtype=float)
    expected_pool = avail.iloc[np.argsort(espn)[:look]]["pos"]
    return {
        p: float(np.clip(
            (sum(1 for r in recent if r["pos"] == p) / look) - float((expected_pool == p).mean()),
            0.0, 0.25) / 0.25)
        for p in POSITIONS
    }


def _roster_state(slot: int) -> dict:
    """Snapshot a team's roster construction: what starters remain, flex status,
    and how many picks are left."""
    c = _pos_counts(slot)
    core = {p: max(0, STARTER_TARGET[p] - c[p]) for p in POSITIONS}
    flex_filled = (c["RB"] + c["WR"] + c["TE"]) >= FLEX_TOTAL
    starters_left = sum(core.values()) + (0 if flex_filled else 1)
    picks_left = rounds - len(_team_roster(slot))
    return {"c": c, "core": core, "flex_filled": flex_filled,
            "starters_left": starters_left, "picks_left": picks_left}


def _suggest_pick(slot: int, pool: pd.DataFrame, top_n: int = 3) -> list:
    """Scarcity- and roster-aware recommendations, best first.

    On top of the value/scarcity core described above, the score carries:
      • a bonus for filling an open starting slot (or the FLEX once core is set)
      • a penalty for stacking single-slot positions (QB2/TE2/K2) too early
      • a "last man in his tier" bonus, and urgency when that position is on a run
      • bye-week congestion, counted across the whole roster rather than just
        within the position, since four starters idle in week 9 is the real pain
      • QB↔pass-catcher stacks and RB handcuffs, for bench picks only
      • K/DEF held back until they're the last starters to fill
      • a hard block on non-starters once remaining picks equal remaining
        starter slots, so a legal lineup is always reachable
    """
    if pool.empty:
        return []
    st_ = _roster_state(slot)
    c, core = st_["c"], st_["core"]
    flex_filled, picks_left = st_["flex_filled"], st_["picks_left"]
    force = picks_left <= st_["starters_left"]
    others_done = all(core[p] == 0 for p in ("QB", "RB", "WR", "TE")) and flex_filled
    cur_round = rounds - picks_left + 1
    late_time = others_done or picks_left <= max(2, st_["starters_left"])

    # Scarcity is modelled over the WHOLE remaining board, not just the players
    # this roster may legally take — the robots are under no such restriction.
    full = board[~board["player"].isin(drafted.keys())]
    gap = _picks_until_next(slot, st.session_state.dr_pick_idx)
    surv = _survival(full, gap)
    surv_by_player = dict(zip(full["player"], surv))
    repl = _next_turn_replacement(full, surv)
    tiers = _position_tiers(full)
    heat = _run_heat(full)
    scale = _vor_scale(full)

    mine = [p for p in st.session_state.dr_picks if p["team"] == slot]
    qb_teams = {p["team_abbr"] for p in mine if p["pos"] == "QB"}
    catcher_teams = {p["team_abbr"] for p in mine if p["pos"] in ("WR", "TE")}
    rb_teams = {p["team_abbr"] for p in mine if p["pos"] == "RB"}
    bye_load: dict = {}
    for p in mine:
        b = _BYE_BY_PLAYER.get(p["player"])
        if b is not None and not pd.isna(b):
            bye_load[int(b)] = bye_load.get(int(b), 0) + 1

    def _score(row) -> float:
        pos, name = row["pos"], row["player"]
        core_need = core[pos] > 0
        depth = not core_need
        flex_ok = depth and pos in FLEX_POS and not flex_filled
        # Scarcity only counts where the roster can still use it. A picked-over
        # TE room is no reason to take a second TE you'll never start, so VONA is
        # damped (not zeroed — bench depth is still injury insurance) once the
        # position has no starting or FLEX slot left to fill.
        vona = _VONA_W * (float(row["vor"]) - repl.get(pos, 0.0))
        s = float(row["vor"]) + (vona if (core_need or flex_ok) else 0.30 * vona)

        if core_need:
            s += _NEED_W * scale
        elif flex_ok:
            s += _FLEX_W * scale
        if pos in ("QB", "TE", "K", "DEF") and c[pos] >= STARTER_TARGET[pos]:
            s -= _STACK_W * scale

        tier, tier_size, cliff = tiers.get(name, (1, 99, 0.0))
        if tier_size == 1 and cliff >= 0.5 * scale:
            s += _TIER_W * scale
        s += _RUN_W * scale * heat.get(pos, 0.0)

        # Bye congestion: a depth piece sharing its starter's week off, plus a
        # roster-wide penalty once a single week claims four or more players.
        b = _BYE_BY_PLAYER.get(name)
        if b is not None and not pd.isna(b):
            b = int(b)
            same_pos = any(_BYE_BY_PLAYER.get(p["player"]) == b
                           for p in mine if p["pos"] == pos)
            pen = (0.6 if (depth and same_pos) else 0.0)
            pen += 0.6 * max(0, bye_load.get(b, 0) + 1 - 3)
            s -= _BYE_W * scale * pen

        # Correlation and handcuffs only make sense once starters are covered.
        if depth:
            abbr = row.get("team", "")
            if (pos in ("WR", "TE") and abbr in qb_teams) or (pos == "QB" and abbr in catcher_teams):
                s += _CORR_W * scale
            if pos == "RB" and abbr in rb_teams:
                s += _CUFF_W * scale

        # Availability tiebreaker only — proj_games is already inside VOR.
        g = float(row.get("proj_games") or 0.0)
        if g:
            s += _AVAIL_W * scale * float(np.clip((g - 15.5) / 3.0, -1.0, 1.0))

        if pos in _REACH_ROUND and core_need:
            early = _REACH_ROUND[pos] - cur_round
            if early > 0:
                s -= _REACH_W * scale * early
        if pos in ("K", "DEF") and core_need:
            s += -_BLOCK if not late_time else _K_PRIORITY
        if force and not core_need:
            s -= _BLOCK
        return s

    scored = pool.assign(_score=pool.apply(_score, axis=1))
    scored = scored.sort_values(["_score", "vor"], ascending=[False, False]).head(top_n)

    out = []
    for _, row in scored.iterrows():
        pos, name = row["pos"], row["player"]
        if force and core[pos] > 0:
            bits = [f"required — you must fill {pos} to ice a legal lineup"]
        elif core[pos] > 0:
            bits = [f"best value at {pos}, an open starting spot"]
        elif pos in FLEX_POS and not flex_filled:
            bits = [f"fills your FLEX with the best remaining {pos}"]
        else:
            bits = ["best value left for your bench"]
        tier, tier_size, cliff = tiers.get(name, (1, 99, 0.0))
        if tier_size == 1 and cliff >= 0.5 * scale:
            bits.append(f"last {pos} in tier {tier} — the next one is {cliff:.0f} VOR worse")
        if gap > 0:
            bits.append(f"{surv_by_player.get(name, 1.0) * 100:.0f}% to survive "
                        f"the {gap} picks until your next turn")
        if heat.get(pos, 0.0) > 0.35:
            bits.append(f"{pos} run underway")
        out.append({"player": name, "pos": pos, "reason": "; ".join(bits)})
    return out


def _robot_pick(slot: int) -> dict | None:
    """Best available for a robot: ESPN overall (asc, NaN last), tiebreak VOR.
    Respects positional caps and a must-fill-starters guard so every robot team
    finishes with a full, legal lineup (QB/RB/RB/WR/WR/TE/FLEX/DEF/K).

    Draft-day variance: each available player's ESPN rank is jittered by gaussian
    noise (σ = dr_robot_sigma, set by the "Robot draft style" setting) before the
    best-available is taken. This produces realistic reaches and slides — players
    ranked close together frequently swap, while big rank gaps rarely do — instead
    of a perfectly chalky ADP order every time. Noise never moves the NaN-ranked
    K/DEF (parked at 1e9), and the must-fill guard still guarantees legal rosters.
    """
    counts = _pos_counts(slot)

    avail = board[~board["player"].isin(drafted.keys())].copy()
    if avail.empty:
        return None

    # Enforce position caps.
    avail = avail[avail["pos"].apply(lambda p: counts.get(p, 0) < POS_CAPS.get(p, 99))]
    if _is_must_fill(slot):
        needed = _needed_positions(slot)
        need_pool = avail[avail["pos"].isin(needed)]
        if not need_pool.empty:
            avail = need_pool
    if avail.empty:
        avail = board[~board["player"].isin(drafted.keys())].copy()

    # ESPN best-player-available with rank asc, NaN (K/DEF) pushed to the bottom.
    espn = pd.to_numeric(avail["espn_overall"], errors="coerce").fillna(1e9).to_numpy(dtype=float)
    sigma = float(st.session_state.get("dr_robot_sigma", 6.0))
    if sigma > 0:
        espn = espn + np.random.normal(0.0, sigma, size=len(espn))
    avail = avail.assign(_espn=espn)
    avail = avail.sort_values(["_espn", "vor"], ascending=[True, False])
    return avail.iloc[0].to_dict()


def _record_pick(slot: int, row: dict) -> None:
    idx = st.session_state.dr_pick_idx
    rd = idx // teams + 1
    st.session_state.dr_picks.append({
        "overall": idx + 1, "round": rd, "team": slot,
        "player": row["player"], "pos": row["pos"], "team_abbr": row.get("team", ""),
        "is_user": slot == user_slot,
    })
    st.session_state.dr_drafted[row["player"]] = slot
    st.session_state.dr_pick_idx += 1


@st.cache_data(show_spinner=False)
def _espn_meta(season: int) -> dict:
    """ESPN playerId -> (name, board-style position) for the whole player pool.

    Covers the picks that land outside our top few hundred. They carry no board
    value, but they still occupy a roster spot, and leaving them out would make
    the position counts wrong for whoever made the pick — which in turn would
    mislead the must-fill-starters guard.
    """
    pool = season_players(season)
    if pool.empty:
        return {}
    return {int(r.player_id): (r.player, normalize_pos(r.pos))
            for r in pool.itertuples() if pd.notna(r.player_id)}


def _sync_live() -> bool:
    """Pull the real draft from ESPN and rebuild local state from it.

    Rebuilt from scratch on every change rather than appended to: ESPN hands
    over the whole draft each poll, picks can arrive out of order, a poll can be
    missed and a commissioner can undo. Replaying a couple hundred picks costs
    nothing and, unlike appending, can never drift out of step with the room.

    Returns True when the pick count moved, so the caller knows to redraw.
    """
    detail = live_draft()
    if detail is None:
        # Hold the last good state. A failed poll must never look like an empty
        # draft, or the board would spring back to full mid-round.
        st.session_state.dr_live_stale = True
        return False
    st.session_state.dr_live_stale = False
    st.session_state.dr_live_at = time.time()
    st.session_state.dr_live_complete = bool(detail.get("drafted"))

    made = [p for p in detail.get("picks", []) if pick_made(p)]
    if len(made) == st.session_state.get("dr_live_seen", -1):
        return False

    _wpath = _board_path(st.session_state.dr_scoring)
    id_map = id_to_player(_resolve_links(str(_wpath), _wpath.stat().st_mtime, DRAFT_SEASON))
    meta = _espn_meta(DRAFT_SEASON)
    binfo = {r["player"]: r for r in st.session_state.dr_board}
    slot_of = st.session_state.get("dr_live_slots", {})

    picks_out, drafted_out = [], {}
    for p in sorted(made, key=lambda x: x.get("overallPickNumber") or 0):
        pid = int(p["playerId"])
        slot = slot_of.get(p.get("teamId"), 0)
        name = id_map.get(pid)
        if name is not None:
            row = binfo[name]
            pos, abbr, on_board = row["pos"], row.get("team", ""), True
        else:
            name, pos = meta.get(pid, ("", ""))
            abbr, on_board = "", False
            if pos not in POSITIONS:
                # Punter / IDP / someone ESPN knows and we don't. The slot is
                # spent either way, but there's no roster position to credit.
                continue
        picks_out.append({
            "overall": p.get("overallPickNumber"), "round": p.get("roundId"),
            "team": slot, "player": name or f"ESPN #{pid}", "pos": pos,
            "team_abbr": abbr, "is_user": slot == user_slot,
        })
        if on_board:
            drafted_out[name] = slot

    st.session_state.dr_picks = picks_out
    st.session_state.dr_drafted = drafted_out
    # Progress is what ESPN has actually filled in, including the picks we
    # skipped above — otherwise the clock would drift behind the real room.
    st.session_state.dr_pick_idx = len(made)
    st.session_state.dr_live_seen = len(made)
    return True


@st.fragment(run_every="4s")
def _live_poll() -> None:
    """Poll ESPN on a timer without redrawing the page for nothing.

    The fragment reruns by itself every few seconds; only an actual change in
    the pick count escalates to a full rerun, so the board and suggestions stay
    put between picks instead of flickering on every tick.
    """
    if _sync_live():
        st.rerun(scope="app")


def _advance_robots() -> None:
    """Run robot picks until it is the user's turn or the draft ends.

    In manual mode the user makes every team's selection, so no robot ever
    picks — the on-clock team is always whoever's slot is up next. In live mode
    every pick, including yours, comes from ESPN.
    """
    if manual or live:
        return
    while (st.session_state.dr_pick_idx < total_picks
           and order[st.session_state.dr_pick_idx] != user_slot):
        slot = order[st.session_state.dr_pick_idx]
        pick = _robot_pick(slot)
        if pick is None:
            st.session_state.dr_pick_idx = total_picks
            break
        _record_pick(slot, pick)


# ── Roster construction + draft grading ──────────────────────────────────────
_VOR_BY_PLAYER = dict(zip(board["player"], board["vor"]))


def _roster_row(slot_name: str, p: dict | None) -> dict:
    """One roster-table row: slot, headshot, player, position, team logo, bye."""
    if p is None:
        return {"Slot": slot_name, "Headshot": "", "Player": "—", "Pos": "",
                "Team": "", "Bye": ""}
    return {
        "Slot": slot_name,
        "Headshot": _HEADSHOTS.get(p["player"], ""),
        "Player": p["player"],
        "Pos": p["pos"],
        "Team": _TEAM_LOGOS.get(p.get("team_abbr", ""), ""),
        "Bye": _bye_label(p["player"]),
    }


def _roster_slot_rows(picks_list: list) -> list:
    """Greedy starter fill (QB/RB×2/WR×2/TE/FLEX/K), remainder to bench (BE)."""
    remaining = list(picks_list)
    rows = []
    for slot_name in STARTER_SLOTS:
        allowed = FLEX_POS if slot_name == "FLEX" else (slot_name,)
        take = next((p for p in remaining if p["pos"] in allowed), None)
        if take:
            remaining.remove(take)
        rows.append(_roster_row(slot_name, take))
    for p in remaining:
        rows.append(_roster_row("BE", p))
    return rows


# Shared column_config for any dataframe rendering headshot/team-logo columns.
_IMG_COLS = {
    "Headshot": st.column_config.ImageColumn("📷", width="small"),
    "Team":     st.column_config.ImageColumn("Team", width="small"),
}
# Roster tables (right panel + per-team inspector) — squeeze the image/bye
# columns down to their minimum width and give the freed-up space to Player.
_ROSTER_COLS = {
    **_IMG_COLS,
    "Bye":    st.column_config.TextColumn("Bye", width="small"),
    "Player": st.column_config.TextColumn("Player", width=225),
}


def _lineup_value(slot: int) -> dict:
    """Best starting-lineup VOR + bench VOR for a team, plus unfilled starters.

    Fills the required starters with each position's highest-VOR players, then
    the FLEX with the best remaining RB/WR/TE — the same lineup that decides a
    real matchup — so a draft is judged on the strength of what it can start.
    """
    picks = [p for p in st.session_state.dr_picks if p["team"] == slot]
    by_pos = {p_: [] for p_ in POSITIONS}
    for p in picks:
        by_pos[p["pos"]].append(float(_VOR_BY_PLAYER.get(p["player"], 0.0)))
    for p_ in by_pos:
        by_pos[p_].sort(reverse=True)

    used = {p_: 0 for p_ in POSITIONS}
    starter = 0.0
    for p_, n in STARTER_TARGET.items():
        for _ in range(n):
            if used[p_] < len(by_pos[p_]):
                starter += by_pos[p_][used[p_]]
                used[p_] += 1
    # FLEX: best remaining RB/WR/TE.
    flex_best, flex_pos = None, None
    for p_ in FLEX_POS:
        if used[p_] < len(by_pos[p_]):
            v = by_pos[p_][used[p_]]
            if flex_best is None or v > flex_best:
                flex_best, flex_pos = v, p_
    if flex_pos is not None:
        starter += flex_best
        used[flex_pos] += 1

    bench = sum(v for p_ in POSITIONS for v in by_pos[p_][used[p_]:])
    missing = sum(max(0, REQ_MIN[p_] - len(by_pos[p_])) for p_ in POSITIONS)
    return {"starter": starter, "bench": bench, "n": len(picks), "missing": missing}


# Letter bands over a team's z-score vs the rest of the league (graded on a
# curve so grades stay meaningful regardless of scoring format / league size).
_GRADE_BANDS = [
    (1.30, "A+"), (1.00, "A"), (0.70, "A-"),
    (0.40, "B+"), (0.15, "B"), (-0.10, "B-"),
    (-0.35, "C+"), (-0.60, "C"), (-0.85, "C-"),
    (-1.10, "D+"), (-1.40, "D"), (-1.70, "D-"),
]


def _league_grades() -> dict:
    """Grade every team A+..F. Score = best-lineup VOR + a slice of bench depth,
    minus a heavy penalty per unfilled required starter, then curved by z-score
    across the league."""
    lv = {s: _lineup_value(s) for s in range(1, teams + 1)}
    scores = {s: v["starter"] + 0.15 * v["bench"] - 60.0 * v["missing"]
              for s, v in lv.items()}
    vals = np.array(list(scores.values()), dtype=float)
    mean = float(vals.mean())
    std = float(vals.std()) or 1.0
    grades = {}
    for s, sc in scores.items():
        z = (sc - mean) / std
        letter = "F"
        for thresh, lt in _GRADE_BANDS:
            if z >= thresh:
                letter = lt
                break
        grades[s] = {"grade": letter, "score": sc, "z": z, "lv": lv[s]}
    return grades


if live:
    _sync_live()
_advance_robots()

pick_idx = st.session_state.dr_pick_idx
drafted = st.session_state.dr_drafted
done = pick_idx >= total_picks or (live and st.session_state.get("dr_live_complete"))
# Team currently on the clock (always the user in standard mode after robots
# advance; any team in manual mode). This is who the pick UI drafts for.
active_slot = None if done else order[pick_idx]

# ── Header status ────────────────────────────────────────────────────────────
if done:
    st.success("🏁 Draft complete! Review your roster below.")
else:
    cur_round = pick_idx // teams + 1
    on_clock = active_slot
    if manual:
        who = f"🟠 **Manual mode** — you're picking for **{_team_label(on_clock)}**"
    elif live:
        who = ("🟢 **You're on the clock!**" if on_clock == user_slot
               else f"⏳ {_team_label(on_clock)} is picking…")
    else:
        who = ("🟢 **You're on the clock!**" if on_clock == user_slot
               else f"{_team_label(on_clock)} is picking…")
    st.markdown(
        f"**Pick {pick_idx + 1} / {total_picks}**  ·  Round {cur_round}  ·  "
        f"Your slot: **{user_slot}** ({_team_label(user_slot)})  ·  {who}"
    )

# Live mode runs on ESPN's clock, so keep a poller alive and be explicit about
# how fresh the board is — a silently stale board is worse than a slow one.
if live and not done:
    _live_poll()
    if st.session_state.get("dr_live_stale"):
        _ago = int(time.time() - (st.session_state.get("dr_live_at") or time.time()))
        st.warning(
            f"⚠️ Can't reach ESPN — showing the board as of {_ago}s ago. Picks made "
            "since then are missing. If this persists your espn_s2 / SWID cookies "
            "have probably expired."
        )
    else:
        st.caption("🔴 Live — following your ESPN draft room, refreshing every 4s.")

left, right = st.columns([1.4, 1])

# ── Available board + user pick control ──────────────────────────────────────
with left:
    st.markdown("#### 📋 Your Big Board — Best Available")
    avail = board[~board["player"].isin(drafted.keys())].copy()

    fcol1, fcol2 = st.columns([1, 2])
    pos_filter = fcol1.selectbox("Position", ["All", *POSITIONS], key="ds_pos_filter")
    view = avail if pos_filter == "All" else avail[avail["pos"] == pos_filter]

    # Show the full available board (scrollable). Capping at the top 50 by VOR
    # hid every kicker, since their VOR ranks them ~150+ — use the "K" position
    # filter (or scroll) to reach them.
    show = view[["my_rank", "espn_overall", "player", "pos", "team", "bye", "vor",
                 "predicted_pts", "proj_games", "round_grade"]].copy()
    show.insert(2, "headshot", view["player"].map(_HEADSHOTS).fillna(""))
    show["team"] = view["team"].map(_TEAM_LOGOS).fillna("")
    show.columns = ["My Rank", "ESPN Rank", "Headshot", "Player", "Pos", "Team", "Bye",
                    "VOR", "Proj Pts", "Proj G", "Grade"]
    st.dataframe(show, hide_index=True, use_container_width=True, height=430,
                 column_config=_IMG_COLS)

    if not done:
        # Live mode always advises *you*, even while someone else is on the
        # clock — that waiting stretch is exactly when you want to know what
        # survives to your next turn. Elsewhere the advice follows the clock.
        advice_slot = user_slot if live else active_slot
        counts = _pos_counts(advice_slot)
        needed = _needed_positions(advice_slot)
        must = _is_must_fill(advice_slot)
        who_txt = _team_label(advice_slot) if manual and advice_slot != user_slot else "you"

        # Selectable pool = available players this team can still roster (caps).
        pool = avail[avail["pos"].apply(lambda p: counts.get(p, 0) < POS_CAPS.get(p, 99))].copy()
        if must and needed:
            forced = pool[pool["pos"].isin(needed)]
            if not forced.empty:
                pool = forced
                st.warning(
                    f"⚠️ Roster requirement — {who_txt} still need **{', '.join(needed)}**. "
                    "Only those positions are selectable so the roster finishes with a full lineup."
                )

        # Suggestions are computed over the full legal pool (roster-aware).
        sugs = _suggest_pick(advice_slot, pool)
        sug = sugs[0] if sugs else None
        if sug is not None:
            st.info(f"💡 **Suggested pick:** {sug['player']} ({sug['pos']}) — {sug['reason']}")
        if len(sugs) > 1:
            st.caption("**Also considered** — " + "  ·  ".join(
                f"{s['player']} ({s['pos']}): {s['reason']}" for s in sugs[1:]))

        # Apply the position filter, but never let it push outside the legal pool.
        pool_view = pool if pos_filter == "All" else pool[pool["pos"] == pos_filter]
        if pool_view.empty:
            pool_view = pool

        pick_opts = pool_view["player"].tolist()
        if live:
            # No pick control: ESPN owns every selection, including yours. Making
            # one here too would put local state out of step with the real room.
            st.caption(
                f"🔴 Draft in your ESPN room — it lands here within a few seconds. "
                f"Suggestions above are for **{_team_label(user_slot)}**"
                + ("." if active_slot == user_slot else
                   f", {_picks_until_next(user_slot, pick_idx)} picks from your next turn.")
            )
        elif pick_opts:
            labels = {
                r["player"]: f'#{r["my_rank"]}  {r["player"]} ({r["pos"]}) · VOR {r["vor"]}'
                for _, r in pool_view.iterrows()
            }
            default_idx = pick_opts.index(sug["player"]) if sug and sug["player"] in pick_opts else 0
            # Pick-scoped keys so a stale selection can never carry across picks
            # (which previously let a non-required player slip through the guard).
            pick_label = "Make your pick" if not manual else f"Make {_team_label(active_slot)}'s pick"
            choice = fcol2.selectbox(pick_label, pick_opts, index=default_idx,
                                     format_func=lambda p: labels.get(p, p),
                                     key=f"ds_user_choice_{pick_idx}")
            if st.button(f"✅ Draft {choice}", type="primary", use_container_width=True,
                         key=f"ds_draft_btn_{pick_idx}"):
                row = pool_view[pool_view["player"] == choice].iloc[0].to_dict()
                _record_pick(active_slot, row)
                _advance_robots()
                st.rerun()

# ── Your roster + recent picks ───────────────────────────────────────────────
with right:
    rhdr, rsel = st.columns([2, 1])
    with rsel:
        # Defaults to your own team; use the dropdown to inspect anyone else's
        # roster (most useful in manual mode, where you pick for every team).
        roster_slot = st.selectbox(
            "View roster", list(range(1, teams + 1)),
            index=user_slot - 1,
            format_func=lambda s: f"You — {_team_label(s)}" if s == user_slot else _team_label(s),
            key="ds_roster_view", label_visibility="collapsed",
        )
    with rhdr:
        st.markdown("#### 🧢 Your Roster" if roster_slot == user_slot
                    else f"#### 🧢 {_team_label(roster_slot)}")
    my_picks = [p for p in st.session_state.dr_picks if p["team"] == roster_slot]

    st.dataframe(
        pd.DataFrame(_roster_slot_rows(my_picks)),
        hide_index=True, use_container_width=True, height=340,
        column_config=_ROSTER_COLS,
    )

    _still_need = _needed_positions(roster_slot)
    if _still_need:
        st.caption(f"⚠️ Still to fill: {', '.join(_still_need)}")
    else:
        st.caption("✅ All required starter positions filled.")

    st.markdown("#### 🕒 Recent Picks")
    recent = st.session_state.dr_picks[-10:][::-1]
    if recent:
        rdf = pd.DataFrame([{
            "Pk": p["overall"],
            "Team": ("You" if p["is_user"] else _team_label(p["team"], short=True)),
            "Player": f'{p["player"]} ({p["pos"]})',
        } for p in recent])
        st.dataframe(rdf, hide_index=True, use_container_width=True, height=300)
    else:
        st.caption("No picks yet.")

# ── Team rosters & draft grades ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Team Rosters & Draft Grades")
st.caption(
    "Grades weigh each team's best startable lineup (VOR) plus a slice of bench "
    "depth, curved across the league. They're most meaningful once the draft is "
    "complete."
)

grades = _league_grades()

# League-wide grade table (best startable roster first).
gtable = []
for s in range(1, teams + 1):
    v = grades[s]["lv"]
    gtable.append({
        "Team": f"You — {_team_label(s)}" if s == user_slot else _team_label(s),
        "Grade": grades[s]["grade"],
        "Starter VOR": round(v["starter"], 1),
        "Bench VOR": round(v["bench"], 1),
        "Players": v["n"],
    })
gdf = pd.DataFrame(gtable).sort_values("Starter VOR", ascending=False)
st.dataframe(gdf, hide_index=True, use_container_width=True)

# Per-team inspector.
sel = st.selectbox(
    "Inspect a team's picks", list(range(1, teams + 1)),
    index=user_slot - 1,
    format_func=lambda s: f"You — {_team_label(s)}" if s == user_slot else _team_label(s),
    key="ds_team_view",
)
sel_picks = [p for p in st.session_state.dr_picks if p["team"] == sel]
who = "You" if sel == user_slot else _team_label(sel)
st.markdown(f"**{who} — Draft Grade: {grades[sel]['grade']}**")

vcol1, vcol2 = st.columns(2)
with vcol1:
    st.caption("Starting lineup")
    st.dataframe(
        pd.DataFrame(_roster_slot_rows(sel_picks)),
        hide_index=True, use_container_width=True,
        column_config=_ROSTER_COLS,
    )
with vcol2:
    st.caption("Picks in order")
    if sel_picks:
        st.dataframe(
            pd.DataFrame([{
                "Rd": p["round"], "Pk": p["overall"],
                "Player": f'{p["player"]} ({p["pos"]})',
            } for p in sel_picks]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.caption("No picks yet.")
