import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Player Search · NFL", page_icon="🔍", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="6_Player_Search")

# ── Back button ─────────────────────────────────────────────────────────────
if st.button("← Back to Player Stats", key="back_btn", help="Return to Player Stats"):
    st.switch_page("pages/2_Player_Stats.py")

st.markdown("<style>div[data-testid='stButton']:has(button:contains('Back')) { max-width: 150px; }</style>", unsafe_allow_html=True)

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔍</div>
    <div>
        <div class="title">Player Search</div>
        <div class="subtitle">Detailed individual player stats and rankings</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────────────────────
weekly = load_weekly()
teams = load_teams()

if weekly.empty:
    st.warning("No weekly player data found. Run load_nfl_data.py first.")
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
if "ps_search_v" not in st.session_state:
    st.session_state["ps_search_v"] = 0
_v = st.session_state["ps_search_v"]

seasons = sorted(weekly["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key=f"ps_search_season_{_v}")

# Get player list
name_col_detect = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
pos_col = next((c for c in ["position", "pos"] if c in weekly.columns), None)
team_col = next((c for c in ["recent_team", "posteam", "team"] if c in weekly.columns), None)

if not name_col_detect:
    st.error("Player name column not found")
    st.stop()

# Filter by season
season_data = weekly[weekly["season"] == sel_season].copy()

# Get unique players for the season
season_players = sorted(season_data[name_col_detect].dropna().unique().tolist())

# Searchable player selector
sel_player = st.sidebar.selectbox(
    "Search & Select Player",
    season_players,
    key=f"ps_search_player_{_v}",
    help="Start typing to search for a player"
)

if not sel_player:
    st.sidebar.warning("Please select a player")
    st.stop()

# Get all rows for the selected player in this season
player_data_all = season_data[season_data[name_col_detect] == sel_player].copy()

if player_data_all.empty:
    st.error("No data found for this player")
    st.stop()

# Use the most recent record (typically has most complete info)
player_row = player_data_all.iloc[-1]

# ── Player Info ──────────────────────────────────────────────────────────────
player_name = player_row[name_col_detect]
player_team = player_row[team_col] if team_col else "—"
player_pos = player_row[pos_col] if pos_col else "—"
player_pic = player_row.get("headshot_url", "") if "headshot_url" in player_row.index else ""

# Get team logo
team_logo_url = get_logo(player_team, teams)

st.markdown("### Player Card")

# Calculate fantasy position rank before rendering
pos_players = season_data[season_data[pos_col] == player_pos].copy() if pos_col else pd.DataFrame()
fantasy_rank = "—"

if not pos_players.empty and "fantasy_points_ppr" in player_data_all.columns:
    player_totals = pos_players.groupby(name_col_detect)["fantasy_points_ppr"].sum().sort_values(ascending=False)
    if player_name in player_totals.index:
        fantasy_rank = list(player_totals.index).index(player_name) + 1
        total_at_pos = len(player_totals)
        fantasy_rank = f"{fantasy_rank}/{total_at_pos}"

# Build player image HTML
if player_pic and pd.notna(player_pic):
    player_img_html = f'<img src="{player_pic}" style="width: 100%; height: 100%; max-width: 30vw; object-fit: cover; display: block; border-radius: 8px;">'
else:
    player_img_html = f'<div style="width: 100%; max-width: 30vw; height: 100%; display: flex; align-items: center; justify-content: center; color: var(--muted); background: var(--card-bg); border-radius: 8px;">No photo</div>'

# Build team logo HTML
if team_logo_url:
    team_logo_html = f'<img src="{team_logo_url}" style="width: 60px; height: 60px; object-fit: contain; margin-top: 6px;">'
else:
    team_logo_html = f'<span style="font-size: 1rem; font-weight: 600; color: var(--text);">{player_team}</span>'

card_style = "padding: 16px 12px; text-align: center; min-height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center;"
label_style = "font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;"
value_style = "font-size: 1.2rem; font-weight: 700; color: var(--text); line-height: 1.3; word-break: break-word;"

st.markdown(f"""
<div style="display: flex; gap: 20px; align-items: stretch; margin-bottom: 24px;">
    <!-- Player photo -->
    <div style="flex: 0 0 auto; display: flex;">
        {player_img_html}
    </div>
    <!-- Info cards -->
    <div style="flex: 1; display: flex; flex-direction: column; gap: 12px;">
        <!-- Row 1: Name and Position -->
        <div style="display: flex; gap: 12px;">
            <div class="stat-card" style="flex: 1; {card_style}">
                <div style="{label_style}">Name</div>
                <div style="{value_style}">{player_name}</div>
            </div>
            <div class="stat-card" style="flex: 1; {card_style}">
                <div style="{label_style}">Position</div>
                <div style="{value_style}">{player_pos}</div>
            </div>
        </div>
        <!-- Row 2: Team (fixed small width) and Pos Rank -->
        <div style="display: flex; gap: 12px;">
            <div class="stat-card" style="flex: 0 0 130px; {card_style}">
                <div style="{label_style}">Team</div>
                {team_logo_html}
            </div>
            <div class="stat-card" style="flex: 1; {card_style}">
                <div style="{label_style}">Pos. Rank (PPR)</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: var(--text);">{fantasy_rank}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Aggregate Stats ──────────────────────────────────────────────────────────
st.markdown("### Season Statistics")

# Group by player to get season totals
group_cols = [name_col_detect]
if team_col:
    group_cols.append(team_col)
if pos_col:
    group_cols.append(pos_col)

# Identify stat columns available
passing_cols = [c for c in ["passing_yards", "passing_tds", "interceptions", "attempts", "completions"] if c in player_data_all.columns]
rushing_cols = [c for c in ["rushing_yards", "rushing_tds", "carries"] if c in player_data_all.columns]
receiving_cols = [c for c in ["receiving_yards", "receiving_tds", "receptions", "targets"] if c in player_data_all.columns]

agg_player = player_data_all[group_cols + passing_cols + rushing_cols + receiving_cols].sum()

# Display stats in columns
if passing_cols:
    st.markdown("#### Passing")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.metric("Passing Yards", f"{int(agg_player.get('passing_yards', 0)):,}")
        with col2:
            st.metric("Touchdowns", int(agg_player.get("passing_tds", 0)))
        with col3:
            st.metric("Interceptions", int(agg_player.get("interceptions", 0)))
    except:
        pass

if rushing_cols:
    st.markdown("#### Rushing")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.metric("Rushing Yards", f"{int(agg_player.get('rushing_yards', 0)):,}")
        with col2:
            st.metric("Touchdowns", int(agg_player.get("rushing_tds", 0)))
        with col3:
            st.metric("Carries", int(agg_player.get("carries", 0)))
    except:
        pass

if receiving_cols:
    st.markdown("#### Receiving")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.metric("Receiving Yards", f"{int(agg_player.get('receiving_yards', 0)):,}")
        with col2:
            st.metric("Touchdowns", int(agg_player.get("receiving_tds", 0)))
        with col3:
            st.metric("Receptions", int(agg_player.get("receptions", 0)))
    except:
        pass

st.markdown("<br>", unsafe_allow_html=True)

# ── Player Rankings ──────────────────────────────────────────────────────────
st.markdown("### Player Rankings (vs. Position Group)")

# Calculate rankings by position
rankings_data = []

if player_pos and pos_col:
    pos_players = season_data[season_data[pos_col] == player_pos].copy()
    
    # Create ranking metrics
    ranking_metrics = [
        ("passing_yards", "Passing Yards"),
        ("rushing_yards", "Rushing Yards"),
        ("receiving_yards", "Receiving Yards"),
        ("receptions", "Receptions"),
        ("passing_tds", "Passing TDs"),
        ("rushing_tds", "Rushing TDs"),
        ("receiving_tds", "Receiving TDs"),
    ]
    
    rank_cols = []
    for col_name, label in ranking_metrics:
        if col_name in pos_players.columns:
            # Sum by player
            player_totals = pos_players.groupby(name_col_detect)[col_name].sum().sort_values(ascending=False)
            
            # Find player's rank
            if player_name in player_totals.index:
                rank = list(player_totals.index).index(player_name) + 1
                total_at_pos = len(player_totals)
                player_val = int(player_totals[player_name])
                rank_cols.append({
                    "Metric": label,
                    "Value": f"{player_val:,}",
                    "Rank": f"{rank} of {total_at_pos}",
                })
    
    if rank_cols:
        rank_df = pd.DataFrame(rank_cols)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    else:
        st.info("No ranking data available for this position")
else:
    st.info("Position information not available for rankings")

st.markdown("<br>", unsafe_allow_html=True)

# ── Weekly Breakdown ─────────────────────────────────────────────────────────
st.markdown("### Weekly Breakdown")

# Show week-by-week stats
week_col = next((c for c in ["week", "wk"] if c in player_data_all.columns), None)

if week_col:
    display_cols = [week_col]
    
    # Add available stat columns
    available_stats = [
        "passing_yards", "passing_tds", "interceptions",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions", "targets"
    ]
    
    for col in available_stats:
        if col in player_data_all.columns:
            display_cols.append(col)
    
    week_data = player_data_all[display_cols].copy()
    week_data = week_data.rename(columns={
        week_col: "Week",
        "passing_yards": "Pass Yds",
        "passing_tds": "Pass TDs",
        "interceptions": "INTs",
        "rushing_yards": "Rush Yds",
        "rushing_tds": "Rush TDs",
        "receiving_yards": "Rec Yds",
        "receiving_tds": "Rec TDs",
        "receptions": "Receptions",
        "targets": "Targets",
    })
    
    # Fill NaN with 0 for display
    week_data = week_data.fillna(0)
    
    st.dataframe(week_data, use_container_width=True, hide_index=True)
else:
    st.info("Week information not available")
