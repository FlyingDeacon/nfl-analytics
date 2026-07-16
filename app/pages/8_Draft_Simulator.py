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
SCORING_FORMATS = ["PPR", "Half PPR", "Standard"]


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


def _ensure_board(scoring: str) -> Path | None:
    """Return the parquet path, building it first if it does not exist yet."""
    path = _board_path(scoring)
    if path.exists():
        return path
    with st.spinner(f"Building your {scoring} big board (first run only)…"):
        ok = _build_board(scoring)
    return path if ok else None


@st.cache_data(show_spinner=False)
def _load_board(path: str, mtime: float) -> pd.DataFrame:
    # mtime is part of the cache key so a rebuilt board invalidates the cache.
    df = pd.read_parquet(path)
    # VOR-sorted board = the user's big board order.
    df = df.sort_values("vor", ascending=False).reset_index(drop=True)
    df["my_rank"] = range(1, len(df) + 1)
    return df


# ── Draft settings ───────────────────────────────────────────────────────────
POSITIONS = ("QB", "RB", "WR", "TE", "K")
POS_CAPS = {"QB": 2, "RB": 8, "WR": 8, "TE": 3, "K": 2}  # max per roster
REQ_MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1}   # starters that must be filled
STARTER_TARGET = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1}  # weekly starters (+1 FLEX)
STARTER_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K"]
FLEX_POS = ("RB", "WR", "TE")
FLEX_TOTAL = STARTER_TARGET["RB"] + STARTER_TARGET["WR"] + STARTER_TARGET["TE"] + 1  # 6

settings_locked = st.session_state.get("dr_started", False)

with st.expander("⚙️ Draft Settings", expanded=not settings_locked):
    c1, c2, c3 = st.columns(3)
    scoring = c1.selectbox("Scoring", SCORING_FORMATS, index=0,
                           disabled=settings_locked, key="ds_scoring")
    teams = c2.selectbox("Teams", [8, 10, 12, 14], index=2,
                         disabled=settings_locked, key="ds_teams")
    rounds = c3.selectbox("Rounds", list(range(10, 21)), index=5,
                          disabled=settings_locked, key="ds_rounds")

    c4, c5 = st.columns(2)
    slot = c4.selectbox("Your draft slot", list(range(1, teams + 1)), index=min(5, teams - 1),
                        disabled=settings_locked, key="ds_slot")
    order_type = c5.radio("Draft order", ["Snake", "Linear"], horizontal=True,
                          disabled=settings_locked, key="ds_order_type")


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
        st.session_state.dr_snake = (order_type == "Snake")
        st.session_state.dr_order = _snake_order(teams, rounds, order_type == "Snake")
        st.session_state.dr_board = board.to_dict("records")
        st.session_state.dr_drafted = {}          # player -> team slot
        st.session_state.dr_picks = []            # list of pick dicts
        st.session_state.dr_pick_idx = 0
        st.rerun()
    st.stop()

if creset.button("🔄 Reset Draft", use_container_width=True, key="ds_reset"):
    for k in ("dr_started", "dr_order", "dr_board", "dr_drafted",
              "dr_picks", "dr_pick_idx"):
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
_BLOCK       = 1e6    # effectively removes a player from consideration


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

    def _score(row) -> float:
        pos = row["pos"]
        s = float(row["vor"])
        core_need = core[pos] > 0
        flex_ok = (not core_need) and (pos in FLEX_POS) and (not flex_filled)
        if core_need:
            s += _NEED_BONUS
        elif flex_ok:
            s += _FLEX_BONUS
        if pos in ("QB", "TE", "K") and c[pos] >= STARTER_TARGET[pos]:
            s -= _STACK_PEN
        # Kicker is a required starter: surface it once it's the last starter to
        # fill (skill positions + FLEX all set) or the draft is running tight —
        # the same "fill your starter" logic that promotes QB, just later.
        k_time = others_done or picks_left <= max(2, st_["starters_left"])
        if pos == "K" and core_need:
            if not k_time:
                s -= _BLOCK           # hold the kicker until it's its turn
            else:
                s += _K_PRIORITY      # its turn: rank ahead of bench depth
                                      # (K's VOR is deeply negative, so lift it)
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
    finishes with a full, legal lineup (QB/RB/RB/WR/WR/TE/FLEX)."""
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

    # ESPN best-player-available: rank asc with NaN pushed to the bottom.
    espn = pd.to_numeric(avail["espn_overall"], errors="coerce")
    avail = avail.assign(_espn=espn.fillna(1e9))
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
    """Run robot picks until it is the user's turn or the draft ends."""
    while (st.session_state.dr_pick_idx < total_picks
           and order[st.session_state.dr_pick_idx] != user_slot):
        slot = order[st.session_state.dr_pick_idx]
        pick = _robot_pick(slot)
        if pick is None:
            st.session_state.dr_pick_idx = total_picks
            break
        _record_pick(slot, pick)


_advance_robots()

pick_idx = st.session_state.dr_pick_idx
drafted = st.session_state.dr_drafted
done = pick_idx >= total_picks

# ── Header status ────────────────────────────────────────────────────────────
if done:
    st.success("🏁 Draft complete! Review your roster below.")
else:
    cur_round = pick_idx // teams + 1
    on_clock = order[pick_idx]
    who = "🟢 **You're on the clock!**" if on_clock == user_slot else f"Team {on_clock} is picking…"
    st.markdown(
        f"**Pick {pick_idx + 1} / {total_picks}**  ·  Round {cur_round}  ·  "
        f"Your slot: **{user_slot}**  ·  {who}"
    )

left, right = st.columns([1.4, 1])

# ── Available board + user pick control ──────────────────────────────────────
with left:
    st.markdown("#### 📋 Your Big Board — Best Available")
    avail = board[~board["player"].isin(drafted.keys())].copy()

    fcol1, fcol2 = st.columns([1, 2])
    pos_filter = fcol1.selectbox("Position", ["All", *POSITIONS], key="ds_pos_filter")
    view = avail if pos_filter == "All" else avail[avail["pos"] == pos_filter]

    show = view.head(50)[["my_rank", "player", "pos", "team", "vor",
                          "predicted_pts", "proj_games", "espn_overall", "round_grade"]].copy()
    show.columns = ["My Rank", "Player", "Pos", "Team", "VOR",
                    "Proj Pts", "Proj G", "ESPN Rank", "Grade"]
    st.dataframe(show, hide_index=True, use_container_width=True, height=430)

    if not done and order[pick_idx] == user_slot:
        counts = _pos_counts(user_slot)
        needed = _needed_positions(user_slot)
        must = _is_must_fill(user_slot)

        # Selectable pool = available players you can still roster (respect caps).
        pool = avail[avail["pos"].apply(lambda p: counts.get(p, 0) < POS_CAPS.get(p, 99))].copy()
        if must and needed:
            forced = pool[pool["pos"].isin(needed)]
            if not forced.empty:
                pool = forced
                st.warning(
                    f"⚠️ Roster requirement — you still need **{', '.join(needed)}**. "
                    "Only those positions are selectable so you finish with a full lineup."
                )

        # Suggestion is computed over the full legal pool (roster-aware).
        sug = _suggest_pick(user_slot, pool)
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
            choice = fcol2.selectbox("Make your pick", pick_opts, index=default_idx,
                                     format_func=lambda p: labels.get(p, p),
                                     key=f"ds_user_choice_{pick_idx}")
            if st.button(f"✅ Draft {choice}", type="primary", use_container_width=True,
                         key=f"ds_draft_btn_{pick_idx}"):
                row = pool_view[pool_view["player"] == choice].iloc[0].to_dict()
                _record_pick(user_slot, row)
                _advance_robots()
                st.rerun()

# ── Your roster + recent picks ───────────────────────────────────────────────
with right:
    st.markdown("#### 🧢 Your Roster")
    my_picks = [p for p in st.session_state.dr_picks if p["team"] == user_slot]

    # Fill starter slots greedily, remainder to bench.
    remaining = list(my_picks)
    roster_rows = []
    for slot_name in STARTER_SLOTS:
        allowed = FLEX_POS if slot_name == "FLEX" else (slot_name,)
        take = next((p for p in remaining if p["pos"] in allowed), None)
        if take:
            remaining.remove(take)
            roster_rows.append((slot_name, f'{take["player"]} ({take["pos"]})'))
        else:
            roster_rows.append((slot_name, "—"))
    for p in remaining:
        roster_rows.append(("BE", f'{p["player"]} ({p["pos"]})'))

    st.dataframe(
        pd.DataFrame(roster_rows, columns=["Slot", "Player"]),
        hide_index=True, use_container_width=True, height=340,
    )

    _still_need = _needed_positions(user_slot)
    if _still_need:
        st.caption(f"⚠️ Still to fill: {', '.join(_still_need)}")
    else:
        st.caption("✅ All required starter positions filled.")

    st.markdown("#### 🕒 Recent Picks")
    recent = st.session_state.dr_picks[-10:][::-1]
    if recent:
        rdf = pd.DataFrame([{
            "Pk": p["overall"],
            "Team": ("You" if p["is_user"] else f'T{p["team"]}'),
            "Player": f'{p["player"]} ({p["pos"]})',
        } for p in recent])
        st.dataframe(rdf, hide_index=True, use_container_width=True, height=300)
    else:
        st.caption("No picks yet.")
