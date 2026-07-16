from __future__ import annotations

import sys
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

# ── Load persisted big boards ────────────────────────────────────────────────
BIG_BOARD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "derived"


@st.cache_data(show_spinner=False)
def _available_boards() -> dict:
    """Return {scoring_label -> parquet path} for every persisted big board."""
    out: dict = {}
    for p in sorted(BIG_BOARD_DIR.glob("big_board_*.parquet")):
        label = p.stem.replace("big_board_", "").replace("_", " ")
        out[label] = str(p)
    return out


@st.cache_data(show_spinner=False)
def _load_board(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # VOR-sorted board = the user's big board order.
    df = df.sort_values("vor", ascending=False).reset_index(drop=True)
    df["my_rank"] = range(1, len(df) + 1)
    return df


boards = _available_boards()
if not boards:
    st.warning(
        "No big board found yet. Open **🔮 2026 Fantasy Predictions** first — "
        "visiting that page saves the board the simulator drafts from."
    )
    if st.button("Go to Fantasy Predictions", key="ds_goto_pred"):
        st.switch_page("pages/7_Fantasy_Predictions.py")
    st.stop()

# ── Draft settings ───────────────────────────────────────────────────────────
POSITIONS = ("QB", "RB", "WR", "TE")
POS_CAPS = {"QB": 2, "RB": 8, "WR": 8, "TE": 3}          # max per roster
REQ_MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}           # starters that must be filled
STARTER_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
FLEX_POS = ("RB", "WR", "TE")

settings_locked = st.session_state.get("dr_started", False)

with st.expander("⚙️ Draft Settings", expanded=not settings_locked):
    c1, c2, c3 = st.columns(3)
    scoring_opts = list(boards.keys())
    default_scoring = "PPR" if "PPR" in scoring_opts else scoring_opts[0]
    scoring = c1.selectbox("Scoring", scoring_opts,
                           index=scoring_opts.index(default_scoring),
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
        board = _load_board(boards[scoring])
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


def _robot_pick(slot: int) -> dict | None:
    """Best available for a robot: ESPN overall (asc, NaN last), tiebreak VOR.
    Respects positional caps and a must-fill-starters guard late in the draft."""
    roster = _team_roster(slot)
    counts = {p: roster.count(p) for p in POSITIONS}
    picks_left = rounds - len(roster)
    unmet = sum(max(0, REQ_MIN[p] - counts[p]) for p in POSITIONS)
    must_fill = picks_left <= unmet

    avail = board[~board["player"].isin(drafted.keys())].copy()
    if avail.empty:
        return None

    # Enforce position caps.
    avail = avail[avail["pos"].apply(lambda p: counts.get(p, 0) < POS_CAPS.get(p, 99))]
    if must_fill:
        needed = [p for p in POSITIONS if counts[p] < REQ_MIN[p]]
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
        pick_opts = view["player"].tolist()
        if pick_opts:
            labels = {
                r["player"]: f'#{r["my_rank"]}  {r["player"]} ({r["pos"]}) · VOR {r["vor"]}'
                for _, r in view.iterrows()
            }
            choice = fcol2.selectbox("Make your pick", pick_opts,
                                     format_func=lambda p: labels.get(p, p), key="ds_user_choice")
            if st.button(f"✅ Draft {choice}", type="primary", use_container_width=True, key="ds_draft_btn"):
                row = view[view["player"] == choice].iloc[0].to_dict()
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
