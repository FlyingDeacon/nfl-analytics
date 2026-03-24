import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Player Stats · NFL", page_icon="🏃", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="2_Player_Stats")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🏃</div>
    <div>
        <div class="title">Player Stats</div>
        <div class="subtitle">Season leaders in passing, rushing, and receiving</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

weekly = load_weekly()
teams  = load_teams()

if weekly.empty:
    st.warning("No weekly player data found. Run load_nfl_data.py first.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
if "ps_v" not in st.session_state:
    st.session_state["ps_v"] = 0
_v = st.session_state["ps_v"]

seasons = sorted(weekly["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key=f"ps_season_{_v}")
stat_type  = st.sidebar.radio("Stat Category", ["Passing", "Rushing", "Receiving"], key=f"ps_stat_{_v}")
top_n      = st.sidebar.slider("Show top N players", 5, 50, 20, key=f"ps_top_n_{_v}")

# Player autocomplete
name_col_detect = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
if name_col_detect:
    season_players = sorted(
        weekly[weekly["season"] == sel_season][name_col_detect].dropna().unique().tolist()
    )
    sel_player = st.sidebar.selectbox(
        "Search Player", ["All Players"] + season_players, key=f"ps_player_{_v}"
    )
else:
    sel_player = "All Players"

if st.sidebar.button("Reset Filters", key="ps_reset", use_container_width=True):
    st.session_state["ps_v"] = _v + 1
    st.rerun()

df = weekly[weekly["season"] == sel_season].copy()

# Apply player filter
if sel_player != "All Players" and name_col_detect:
    df = df[df[name_col_detect] == sel_player]

# ── Aggregate season totals ───────────────────────────────────────────────────
STAT_CONFIG = {
    "Passing": {
        "cols":  ["passing_yards", "passing_tds", "interceptions", "attempts", "completions"],
        "sort":  "passing_yards",
        "label": "Passing Yards",
        "color": "#4f46e5",
    },
    "Rushing": {
        "cols":  ["rushing_yards", "rushing_tds", "carries"],
        "sort":  "rushing_yards",
        "label": "Rushing Yards",
        "color": "#10b981",
    },
    "Receiving": {
        "cols":  ["receiving_yards", "receiving_tds", "receptions", "targets"],
        "sort":  "receiving_yards",
        "label": "Receiving Yards",
        "color": "#f59e0b",
    },
}

cfg  = STAT_CONFIG[stat_type]
name_col  = next((c for c in ["player_display_name", "player_name", "name"] if c in df.columns), None)
team_col  = next((c for c in ["recent_team", "posteam", "team"] if c in df.columns), None)
pos_col   = next((c for c in ["position", "pos"] if c in df.columns), None)

avail_cols = [c for c in cfg["cols"] if c in df.columns]
if not avail_cols or name_col is None:
    st.error(f"Required columns not found in weekly data. Available: {list(df.columns[:20])}")
    st.stop()

group_cols = [name_col]
if team_col:  group_cols.append(team_col)
if pos_col:   group_cols.append(pos_col)

agg_df = (
    df.groupby(group_cols, as_index=False)[avail_cols]
    .sum()
    .sort_values(cfg["sort"], ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
agg_df.insert(0, "Rank", range(1, len(agg_df) + 1))

# ── Top stat cards ────────────────────────────────────────────────────────────
st.markdown(f"### {sel_season} {stat_type} Leaders")

if len(agg_df) >= 3:
    c1, c2, c3 = st.columns(3)
    for i, (col, label) in enumerate(zip(
        [c1, c2, c3],
        ["🥇 #1", "🥈 #2", "🥉 #3"],
    )):
        row = agg_df.iloc[i]
        player = row[name_col]
        team   = row[team_col] if team_col else "—"
        val    = int(row[cfg["sort"]])
        logo_html = ""
        url = get_logo(team, teams)
        if url:
            logo_html = f'<img src="{url}" width="40" style="margin:6px 0;">'
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">{label}</div>
                {logo_html}
                <div class="value" style="font-size:1.15rem;">{player}</div>
                <div class="sub">{team} &nbsp;·&nbsp; {val:,} {cfg["label"]}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.markdown(f"#### Top {top_n} by {cfg['label']}")

bar_colors = [TEAM_COLORS.get(row.get(team_col, ""), cfg["color"])
              for _, row in agg_df.iterrows()]

fig = px.bar(
    agg_df, x=name_col, y=cfg["sort"],
    color_discrete_sequence=[cfg["color"]],
    hover_data=avail_cols,
)
fig.update_traces(marker_color=bar_colors)

y_max = agg_df[cfg["sort"]].max()
logo_h = y_max * 0.13
fig.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{sel_season} {stat_type} Leaders — {cfg['label']}",
    xaxis_title="Player", yaxis_title=cfg["label"],
    showlegend=False,
    xaxis_tickangle=-35,
    yaxis_range=[0, y_max * 1.22],
)
if team_col:
    for idx, (_, row) in enumerate(agg_df.iterrows()):
        url = get_logo(row[team_col], teams)
        if url:
            fig.add_layout_image(dict(
                source=url, xref="x", yref="y",
                x=idx, y=row[cfg["sort"]] + y_max * 0.01,
                sizex=0.8, sizey=logo_h,
                xanchor="center", yanchor="bottom", layer="above",
            ))
st.plotly_chart(fig, use_container_width=True)

# ── Full table ────────────────────────────────────────────────────────────────
st.markdown("#### Full Leaderboard")
display_cols = ["Rank", name_col] + ([team_col] if team_col else []) + \
               ([pos_col] if pos_col else []) + avail_cols
st.dataframe(
    agg_df[display_cols].rename(columns={
        name_col: "Player",
        team_col: "Team",
        pos_col:  "Pos",
    }),
    hide_index=True,
    use_container_width=True,
)
