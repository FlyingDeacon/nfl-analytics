import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_ratings, load_teams
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Historical · NFL", page_icon="📈", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="4_Historical")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">📈</div>
    <div>
        <div class="title">Historical Trends</div>
        <div class="subtitle">Season-over-season team performance</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

ratings = load_ratings()
teams   = load_teams()

if "hist_v" not in st.session_state:
    st.session_state["hist_v"] = 0
_v = st.session_state["hist_v"]

all_teams = sorted(ratings["team"].dropna().unique().tolist())
sel_teams = st.sidebar.multiselect(
    "Select Teams (up to 8)",
    all_teams,
    default=all_teams[:4],
    max_selections=8,
    key=f"hist_teams_{_v}",
)

metric = st.sidebar.selectbox("Metric", ["net_ppg", "ppg", "oppg"], key=f"hist_metric_{_v}")
METRIC_LABELS = {"net_ppg": "Net PPG", "ppg": "Offensive PPG", "oppg": "Defensive PPG Allowed"}

seasons = sorted(ratings["season"].dropna().unique().astype(int))
season_range = st.sidebar.select_slider(
    "Season Range",
    options=seasons,
    value=(min(seasons), max(seasons)),
    key=f"hist_range_{_v}",
)

if st.sidebar.button("Reset Filters", key="hist_reset", use_container_width=True):
    st.session_state["hist_v"] = _v + 1
    st.rerun()

# ── Filter ────────────────────────────────────────────────────────────────────
df = ratings[
    (ratings["season"] >= season_range[0]) &
    (ratings["season"] <= season_range[1])
].copy()

if not sel_teams:
    st.info("Select at least one team from the sidebar.")
    st.stop()

df = df[df["team"].isin(sel_teams)]

# ── Line chart: selected teams over time ──────────────────────────────────────
st.markdown(f"### {METRIC_LABELS[metric]} Over Time")

fig = go.Figure()
for team in sel_teams:
    t_df = df[df["team"] == team].sort_values("season")
    if t_df.empty:
        continue
    color = TEAM_COLORS.get(team, "#4f46e5")
    fig.add_trace(go.Scatter(
        x=t_df["season"], y=t_df[metric].round(1),
        mode="lines+markers",
        name=team,
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color,
                    line=dict(color="#ffffff", width=1.5)),
        hovertemplate=f"<b>{team}</b><br>Season: %{{x}}<br>{METRIC_LABELS[metric]}: %{{y:.1f}}<extra></extra>",
    ))

fig.add_hline(y=0, line_dash="dot", line_color="rgba(79,70,229,0.2)")
fig.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{METRIC_LABELS[metric]} by Season",
    xaxis_title="Season",
    yaxis_title=METRIC_LABELS[metric],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Rank Progression Bump Chart ───────────────────────────────────────────────
st.markdown("### Rank Progression")
st.caption(f"How selected teams rank in {METRIC_LABELS[metric]} each season (1 = best)")

# Build league-wide ranks per season
rank_asc = (metric == "oppg")  # lower OPPG = better (rank 1)
season_mask = (
    (ratings["season"] >= season_range[0]) &
    (ratings["season"] <= season_range[1])
)
rank_df = ratings[season_mask].copy()
rank_df["rank"] = rank_df.groupby("season")[metric].rank(
    ascending=rank_asc, method="min"
).astype(int)

rank_filtered = rank_df[rank_df["team"].isin(sel_teams)].sort_values("season")
n_teams_total = rank_df["team"].nunique()

if rank_filtered.empty:
    st.info("No rank data for selected teams/seasons.")
else:
    fig2 = go.Figure()
    for team in sel_teams:
        t_df = rank_filtered[rank_filtered["team"] == team]
        if t_df.empty:
            continue
        color = TEAM_COLORS.get(team, "#4f46e5")
        fig2.add_trace(go.Scatter(
            x=t_df["season"],
            y=t_df["rank"],
            mode="lines+markers+text",
            name=team,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color="#ffffff", width=1.5)),
            text=t_df["rank"].astype(str),
            textposition="top center",
            textfont=dict(size=10, color=color),
            hovertemplate=(
                f"<b>{team}</b><br>Season: %{{x}}<br>"
                f"Rank: %{{y}} / {n_teams_total}<br>"
                f"{METRIC_LABELS[metric]}: %{{customdata:.1f}}<extra></extra>"
            ),
            customdata=t_df[metric].round(1),
        ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{METRIC_LABELS[metric]} Rank by Season (1 = Best)",
        xaxis_title="Season",
        yaxis_title="Rank",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig2.update_yaxes(autorange="reversed", tickmode="linear", dtick=2)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Season summary table ──────────────────────────────────────────────────────
st.markdown("### Season Summary Table")

summary = (
    ratings[
        (ratings["season"] >= season_range[0]) &
        (ratings["season"] <= season_range[1]) &
        (ratings["team"].isin(sel_teams))
    ]
    [["team", "season", "games", "ppg", "oppg", "net_ppg"]]
    .sort_values(["team", "season"])
    .copy()
)
for c in ["ppg", "oppg", "net_ppg"]:
    summary[c] = summary[c].round(1)

summary.columns = ["Team", "Season", "Games", "PPG", "OPPG", "Net PPG"]
st.dataframe(summary, hide_index=True, use_container_width=True)
