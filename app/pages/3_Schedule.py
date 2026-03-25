import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_schedules, load_teams, get_logo
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Schedule · NFL", page_icon="📅", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="3_Schedule")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">📅</div>
    <div>
        <div class="title">Schedule</div>
        <div class="subtitle">Game results and upcoming matchups</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

schedules = load_schedules()
teams     = load_teams()

# ── Sidebar ──────────────────────────────────────────────────────────────────
if "sch_v" not in st.session_state:
    st.session_state["sch_v"] = 0
_v = st.session_state["sch_v"]

seasons = sorted(schedules["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key=f"sch_season_{_v}")

abbr_col = "team_abbr" if "team_abbr" in teams.columns else "team"
all_teams = sorted(teams[abbr_col].dropna().unique().tolist())
sel_team  = st.sidebar.selectbox("Filter by Team", ["All Teams"] + all_teams, key=f"sch_team_{_v}")

view = st.sidebar.radio("View", ["Results", "Upcoming"], key=f"sch_view_{_v}")

if st.sidebar.button("Reset Filters", key="sch_reset", use_container_width=True):
    st.session_state["sch_v"] = _v + 1
    st.rerun()

# ── Filter schedule ───────────────────────────────────────────────────────────
df = schedules[schedules["season"] == sel_season].copy()

# Detect score / result columns
home_score_col = next((c for c in ["home_score", "home_points"] if c in df.columns), None)
away_score_col = next((c for c in ["away_score", "away_points"] if c in df.columns), None)
home_team_col  = next((c for c in ["home_team"] if c in df.columns), None)
away_team_col  = next((c for c in ["away_team"] if c in df.columns), None)
week_col       = next((c for c in ["week", "game_week"] if c in df.columns), None)
gametime_col   = next((c for c in ["gametime", "game_time", "start_time"] if c in df.columns), None)
gameday_col    = next((c for c in ["gameday", "game_date", "date"] if c in df.columns), None)

if home_score_col and away_score_col:
    completed   = df[df[home_score_col].notna() & df[away_score_col].notna()].copy()
    upcoming    = df[df[home_score_col].isna()].copy()
else:
    # Fall back to result column if scores not present
    result_col = next((c for c in ["result", "game_result"] if c in df.columns), None)
    if result_col:
        completed = df[df[result_col].notna()].copy()
        upcoming  = df[df[result_col].isna()].copy()
    else:
        completed = df.copy()
        upcoming  = pd.DataFrame()

# Filter by team
def filter_by_team(frame, team):
    if team == "All Teams" or not home_team_col or not away_team_col:
        return frame
    return frame[(frame[home_team_col] == team) | (frame[away_team_col] == team)]

# ── Helper: render a score badge ─────────────────────────────────────────────
def logo_tag(abbr, w=28):
    url = get_logo(abbr, teams)
    return f'<img src="{url}" width="{w}" style="vertical-align:middle; margin:0 4px;">' if url else f"<b>{abbr}</b>"

def score_badge(home, away, h_score, a_score):
    h_win = h_score > a_score
    hw = "font-weight:700; color:#1e1e2e;" if h_win else "color:#8b8fa8;"
    aw = "font-weight:700; color:#1e1e2e;" if not h_win else "color:#8b8fa8;"
    return (
        f'<span style="{hw}">{logo_tag(home)} {home} {int(h_score)}</span>'
        f' <span style="color:#c8cdd8; padding:0 6px;">vs</span>'
        f' <span style="{aw}">{int(a_score)} {away} {logo_tag(away)}</span>'
    )

# ── RESULTS ──────────────────────────────────────────────────────────────────
if view == "Results":
    st.markdown("### Results")
    res = filter_by_team(completed, sel_team)

    if week_col:
        weeks = sorted(res[week_col].dropna().unique())
        sel_week = st.selectbox("Week", ["All Weeks"] + [int(w) for w in weeks], key=f"sch_week_res_{_v}")
        if sel_week != "All Weeks":
            res = res[res[week_col].astype(float) == float(sel_week)]

    if res.empty:
        st.info("No completed games for the selected filters.")
    elif home_score_col and away_score_col and home_team_col and away_team_col:
        for _, row in res.iterrows():
            badge = score_badge(
                row[home_team_col], row[away_team_col],
                row[home_score_col], row[away_score_col],
            )
            date_str = f" — {row[gameday_col]}" if gameday_col and pd.notna(row.get(gameday_col)) else ""
            week_str = f"Week {int(row[week_col])}" if week_col and pd.notna(row.get(week_col)) else ""
            st.markdown(
                f'<div style="padding:10px 0; border-bottom:1px solid #e2e5ef;">'
                f'<span style="color:#8b8fa8; font-size:0.8rem; margin-right:12px;">'
                f'{week_str}{date_str}</span>{badge}</div>',
                unsafe_allow_html=True,
            )
    else:
        show_cols = [c for c in [week_col, gameday_col, home_team_col, away_team_col,
                                  home_score_col, away_score_col] if c]
        st.dataframe(res[show_cols].reset_index(drop=True), hide_index=True,
                     use_container_width=True)

# ── UPCOMING ─────────────────────────────────────────────────────────────────
else:
    st.markdown("### Upcoming Games")
    upc = filter_by_team(upcoming, sel_team)

    if upc.empty:
        st.info("No upcoming games found for the selected filters.")
    elif home_team_col and away_team_col:
        if week_col:
            weeks = sorted(upc[week_col].dropna().unique())
            sel_week = st.selectbox("Week", ["All Weeks"] + [int(w) for w in weeks], key=f"sch_week_upc_{_v}")
            if sel_week != "All Weeks":
                upc = upc[upc[week_col].astype(float) == float(sel_week)]

        for _, row in upc.iterrows():
            home = row.get(home_team_col, "?")
            away = row.get(away_team_col, "?")
            date_str = f"{row[gameday_col]}" if gameday_col and pd.notna(row.get(gameday_col)) else "TBD"
            time_str = f" · {row[gametime_col]}" if gametime_col and pd.notna(row.get(gametime_col)) else ""
            week_str = f"Week {int(row[week_col])}" if week_col and pd.notna(row.get(week_col)) else ""

            st.markdown(
                f'<div style="padding:10px 0; border-bottom:1px solid #e2e5ef;">'
                f'<span style="color:#8b8fa8; font-size:0.8rem; margin-right:12px;">'
                f'{week_str} · {date_str}{time_str}</span>'
                f'{logo_tag(home)} <b>{home}</b>'
                f' <span style="color:#4f46e5; padding:0 8px; font-family:\'DM Sans\',sans-serif;">vs</span>'
                f' <b>{away}</b> {logo_tag(away)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.dataframe(upc.reset_index(drop=True), hide_index=True, use_container_width=True)
