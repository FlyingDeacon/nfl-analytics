from __future__ import annotations

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.styles import NFL_CSS
from utils.nav import render_sidebar_nav
from utils.data_loader import load_teams, load_weekly, get_logo, get_base_dir, _file_mtime
from utils.espn_league import espn_configured, load_league, draft_setup

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

    draft_mode = st.radio(
        "Draft mode",
        ["Standard (robots auto-draft)", "Manual (you pick for every team)"],
        horizontal=True, disabled=settings_locked, key="ds_mode",
        help="Manual lets you make every team's selection yourself — useful for "
             "running a mock where you control the whole room.",
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


# ── Start / reset ────────────────────────────────────────────────────────────
cstart, creset = st.columns([1, 1])
if not settings_locked:
    if cstart.button("🚀 Start Draft", type="primary", use_container_width=True, key="ds_start"):
        path = _ensure_board(scoring)
        if path is None:
            st.error(
                "Couldn't build the big board. Open **🔮 2026 Fantasy Predictions** "
                "once to generate it, then come back."
            )
            st.stop()
        board = _load_board(str(path), path.stat().st_mtime)
        st.session_state.dr_started = True
        st.session_state.dr_scoring = scoring
        st.session_state.dr_teams = teams
        st.session_state.dr_rounds = rounds
        st.session_state.dr_slot = slot
        st.session_state.dr_team_names = slot_names
        st.session_state.dr_manual = draft_mode.startswith("Manual")
        st.session_state.dr_robot_sigma = _ROBOT_SIGMA_BY_STYLE[robot_style]
        st.session_state.dr_snake = (order_type == "Snake")
        st.session_state.dr_order = _snake_order(teams, rounds, order_type == "Snake")
        st.session_state.dr_board = board.to_dict("records")
        st.session_state.dr_drafted = {}          # player -> team slot
        st.session_state.dr_picks = []            # list of pick dicts
        st.session_state.dr_pick_idx = 0
        st.session_state.pop("ds_roster_view", None)
        st.rerun()
    st.stop()

if creset.button("🔄 Reset Draft", use_container_width=True, key="ds_reset"):
    for k in ("dr_started", "dr_order", "dr_board", "dr_drafted",
              "dr_picks", "dr_pick_idx", "dr_manual", "dr_robot_sigma",
              "dr_team_names", "ds_roster_view"):
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
manual = st.session_state.get("dr_manual", False)
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


# Suggestion-engine weights (in VOR points). Tuned so a full starting lineup
# (QB/RB×2/WR×2/TE/FLEX/K) gets built at sensible times: RB/WR early, a QB and
# TE mid-draft, FLEX/bench with best value, and the kicker in the final rounds.
_NEED_BONUS  = 30.0   # lift for a player who fills an open STARTING slot
_FLEX_BONUS  = 14.0   # lift for filling the FLEX once core RB/WR/TE are set
_STACK_PEN   = 45.0   # push down QB2 / TE2 / K2 taken before you need depth
_K_PRIORITY  = 500.0  # once it's the kicker's turn, lift it above bench value
                      # (K's VOR is deeply negative, so the need bonus alone
                      # can't out-rank a high-VOR bench player) — beats any
                      # realistic VOR gap but stays below _BLOCK
_BYE_PEN     = 12.0   # nudge down a depth pick that shares a bye with a player
                      # already at that position — keeps starters/backups from
                      # being off the same week. Deliberately small (in VOR
                      # points) so a clearly superior player overrides it, and
                      # only applied to non-starter-need picks so it mostly
                      # shapes bench/FLEX choices, not required starters
_BLOCK       = 1e6    # effectively removes a player from consideration

# Positional draft windows distilled from recent (2023-24) championship-roster
# ADP trends: title teams anchored rounds 1-3 with elite RB/WR, built RB/WR
# depth through the mid rounds, treated an every-week QB as a round ~5-11 value
# (unless it was a truly top-tier arm), took a top-3 TE at a mid pick or else
# waited, and left kickers for the final two rounds. We only *penalize
# reaching* — taking your first QB or TE well before its window — so a genuinely
# elite player (huge VOR) can still beat the penalty, but the engine won't burn
# a premium pick on a replaceable starter at a scarce single-slot position.
_REACH_ROUND = {"QB": 4, "TE": 3}   # earliest round to prioritize QB1 / TE1
_REACH_PEN   = 8.0                   # per-round penalty for reaching early


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


def _suggest_pick(slot: int, avail: pd.DataFrame) -> dict | None:
    """Roster-aware value-based recommendation.

    Scores every available player as VOR plus roster-construction adjustments:
      • bonus for filling an open starting slot (or the FLEX once core is set)
      • penalty for stacking single-slot positions (QB2/TE2/K2) before you need
        bench depth
      • the kicker (a required starter) is held back until it's the last starter
        to fill or the draft is running tight, then surfaced like any other need
      • positional timing from recent championship rosters: reaching for your
        first QB/TE before its draft window is penalized (elite VOR can override)
      • when remaining picks equal remaining starter slots, only starter-filling
        players are considered so you never miss a required position
    """
    if avail.empty:
        return None
    st_ = _roster_state(slot)
    c, core = st_["c"], st_["core"]
    flex_filled, picks_left = st_["flex_filled"], st_["picks_left"]
    force = picks_left <= st_["starters_left"]
    others_done = all(core[p] == 0 for p in ("QB", "RB", "WR", "TE")) and flex_filled
    cur_round = rounds - picks_left + 1

    # Bye weeks already on this roster, per position — used to avoid stacking
    # depth at a position that would all sit out the same week.
    team_byes: dict = {}
    for p in st.session_state.dr_picks:
        if p["team"] == slot:
            b = _BYE_BY_PLAYER.get(p["player"])
            if b is not None and not pd.isna(b):
                team_byes.setdefault(p["pos"], set()).add(int(b))

    def _score(row) -> float:
        pos = row["pos"]
        s = float(row["vor"])
        core_need = core[pos] > 0
        flex_ok = (not core_need) and (pos in FLEX_POS) and (not flex_filled)
        if core_need:
            s += _NEED_BONUS
        elif flex_ok:
            s += _FLEX_BONUS
        if pos in ("QB", "TE", "K", "DEF") and c[pos] >= STARTER_TARGET[pos]:
            s -= _STACK_PEN
        # Bye-week fit: only for depth picks (not an open starting need), nudge
        # down a player who'd share a bye with someone already at that position.
        # Small enough that a high-VOR player still wins the slot.
        if not core_need:
            b = _BYE_BY_PLAYER.get(row["player"])
            if b is not None and not pd.isna(b) and int(b) in team_byes.get(pos, set()):
                s -= _BYE_PEN
        # Winning-strategy timing: don't reach for your first QB/TE too early.
        if pos in _REACH_ROUND and core_need:
            early = _REACH_ROUND[pos] - cur_round
            if early > 0:
                s -= _REACH_PEN * early
        # Kicker and defense are required starters but streamed last: surface
        # them once they're the final starters to fill (skill positions + FLEX
        # all set) or the draft is running tight — the same "fill your starter"
        # logic that promotes QB, just later.
        late_time = others_done or picks_left <= max(2, st_["starters_left"])
        if pos in ("K", "DEF") and core_need:
            if not late_time:
                s -= _BLOCK           # hold K/DEF until it's their turn
            else:
                s += _K_PRIORITY      # their turn: rank ahead of bench depth
                                      # (K/DEF VOR is deeply negative, so lift it)
        if force and not core_need:
            s -= _BLOCK               # must fill a required starter now
        return s

    scored = avail.assign(_score=avail.apply(_score, axis=1))
    scored = scored.sort_values(["_score", "vor"], ascending=[False, False])
    best = scored.iloc[0]
    pos = best["pos"]

    if force and core[pos] > 0:
        reason = f"required — you must fill {pos} to ice a legal starting lineup"
    elif core[pos] > 0:
        reason = f"best value at {pos}, an open starting spot on your roster"
    elif pos in FLEX_POS and not flex_filled:
        reason = f"fills your FLEX with the best remaining {pos} value"
    else:
        reason = "best value available for your bench (VOR)"
    return {"player": best["player"], "pos": pos, "reason": reason}


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


def _advance_robots() -> None:
    """Run robot picks until it is the user's turn or the draft ends.

    In manual mode the user makes every team's selection, so no robot ever
    picks — the on-clock team is always whoever's slot is up next.
    """
    if manual:
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


_advance_robots()

pick_idx = st.session_state.dr_pick_idx
drafted = st.session_state.dr_drafted
done = pick_idx >= total_picks
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
    else:
        who = ("🟢 **You're on the clock!**" if on_clock == user_slot
               else f"{_team_label(on_clock)} is picking…")
    st.markdown(
        f"**Pick {pick_idx + 1} / {total_picks}**  ·  Round {cur_round}  ·  "
        f"Your slot: **{user_slot}** ({_team_label(user_slot)})  ·  {who}"
    )

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
        counts = _pos_counts(active_slot)
        needed = _needed_positions(active_slot)
        must = _is_must_fill(active_slot)
        who_txt = _team_label(active_slot) if manual and active_slot != user_slot else "you"

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

        # Suggestion is computed over the full legal pool (roster-aware).
        sug = _suggest_pick(active_slot, pool)
        if sug is not None:
            st.info(f"💡 **Suggested pick:** {sug['player']} ({sug['pos']}) — {sug['reason']}")

        # Apply the position filter, but never let it push outside the legal pool.
        pool_view = pool if pos_filter == "All" else pool[pool["pos"] == pos_filter]
        if pool_view.empty:
            pool_view = pool

        pick_opts = pool_view["player"].tolist()
        if pick_opts:
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
