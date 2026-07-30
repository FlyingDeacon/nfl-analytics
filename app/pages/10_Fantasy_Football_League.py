import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.styles import NFL_CSS
from utils.nav import render_sidebar_nav
from utils.espn_league import (
    espn_configured, load_league, league_name, draft_completed,
    standings_df, roster_df, matchups_df, team_names,
)

st.set_page_config(page_title="Fantasy League · NFL", page_icon="🏆", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

render_sidebar_nav(current_page="10_Fantasy_Football_League")

if st.button("← Back to Fantasy Football", key="fl_back_btn"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🏆</div>
    <div>
        <div class="title">Fantasy Football League</div>
        <div class="subtitle">Live standings, rosters &amp; matchups from your ESPN league</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

_IMG_COL = {"Logo": st.column_config.ImageColumn("", width="small")}

# ── Not configured yet: walk the user through it ─────────────────────────────
if not espn_configured():
    st.info("This page isn't connected to an ESPN league yet.")
    st.markdown("""
**To connect your ESPN league, add these to `.streamlit/secrets.toml`:**

```toml
[espn]
league_id = 1234567890
season = 2026
espn_s2 = "..."     # only needed for private leagues
swid = "{...}"      # only needed for private leagues
```

- **League ID**: the number in your league's URL
  (`fantasy.espn.com/football/team?leagueId=1234567890`).
- **espn_s2 / SWID**: cookies from a logged-in browser session at
  fantasy.espn.com — DevTools → Application/Storage → Cookies →
  `fantasy.espn.com`.
""")
    st.stop()

with st.spinner("Loading league data from ESPN..."):
    data = load_league()

if not data:
    st.stop()  # load_league() already surfaced an st.error

name = league_name(data) or "Your League"
teams_ct = len(data.get("teams", []))
season = st.secrets["espn"].get("season", "")

st.markdown(f"#### {name} · {season} · {teams_ct} teams")

if not draft_completed(data):
    st.warning("The draft for this league hasn't happened yet — standings will "
                "show all-zero records and rosters will be empty until it does. "
                "This page will fill in automatically once the season starts.")

tab_standings, tab_rosters, tab_schedule = st.tabs(["📊 Standings", "🧢 Rosters", "📅 Schedule"])

# ── Standings ─────────────────────────────────────────────────────────────────
with tab_standings:
    sdf = standings_df(data)
    if sdf.empty:
        st.info("No standings available.")
    else:
        show = sdf[["Rank", "Logo", "Team", "Owner", "W", "L", "T", "PF", "PA", "Streak"]]
        st.dataframe(show, hide_index=True, use_container_width=True, column_config=_IMG_COL)

# ── Rosters ───────────────────────────────────────────────────────────────────
with tab_rosters:
    names = team_names(data)
    team_ids = sorted(names.keys())

    # Default to the team owned by the logged-in SWID, if we can find it.
    my_swid = st.secrets["espn"].get("swid", "")
    my_team_id = next(
        (t["id"] for t in data.get("teams", []) if my_swid in t.get("owners", [])),
        team_ids[0] if team_ids else None,
    )

    rhdr, rsel = st.columns([2, 1])
    with rsel:
        sel_team = st.selectbox(
            "View roster", team_ids,
            index=team_ids.index(my_team_id) if my_team_id in team_ids else 0,
            format_func=lambda tid: f"You — {names[tid]}" if tid == my_team_id else names[tid],
            key="fl_roster_team", label_visibility="collapsed",
        )
    with rhdr:
        st.markdown(f"#### {'Your Roster — ' if sel_team == my_team_id else ''}{names.get(sel_team, '')}")

    rdf = roster_df(data, sel_team)
    if rdf.empty:
        st.info("No players rostered yet (draft hasn't happened).")
    else:
        st.dataframe(rdf, hide_index=True, use_container_width=True)

# ── Schedule / matchups ────────────────────────────────────────────────────────
with tab_schedule:
    mdf = matchups_df(data)
    if mdf.empty:
        st.info("No schedule available.")
    else:
        weeks = sorted(mdf["week"].dropna().unique())
        cur_week = data.get("scoringPeriodId", weeks[0] if weeks else 1)
        sel_week = st.selectbox(
            "Week", weeks,
            index=weeks.index(cur_week) if cur_week in weeks else 0,
            key="fl_sched_week",
        )
        wk = mdf[mdf["week"] == sel_week]
        for _, m in wk.iterrows():
            c1, c2, c3 = st.columns([3, 1, 3])
            c1.markdown(f"**{m['home_team']}**")
            c1.caption(f"{m['home_pts']:g} pts")
            c2.markdown("<div style='text-align:center'>vs</div>", unsafe_allow_html=True)
            if m["away_team"] == "BYE":
                c3.markdown("*BYE*")
            else:
                c3.markdown(f"**{m['away_team']}**")
                c3.caption(f"{m['away_pts']:g} pts")
            st.divider()
