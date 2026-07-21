import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import (
    load_ratings, load_teams, load_schedules, load_divisions,
    get_logo, get_base_dir, _file_mtime,
)
from utils.record_model import project_season
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Season Projections · NFL", page_icon="🔮", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)
render_sidebar_nav(current_page="9_Record_Predictions")

if st.button("← Back to Team Ratings", key="back_to_ratings"):
    st.switch_page("pages/1_Team_Ratings.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔮</div>
    <div>
        <div class="title">2026 Season Projections</div>
        <div class="subtitle">Projected record & expected finish · powered by 2025 results + updated rosters</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Load data + run projection (cached on file mtimes) ───────────────────────
_base = get_base_dir()
ratings   = load_ratings(_mtime=_file_mtime(_base / "data/processed/team_ratings.csv"))
teams_df  = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))
schedules = load_schedules(_mtime=_file_mtime(_base / "data/raw/schedules.csv"))
divisions = load_divisions(_mtime=_file_mtime(_base / "data/raw/nfl_divisions.csv"))
depth = pd.read_csv(_base / "data/raw/depth_charts.csv")


@st.cache_data(show_spinner="Simulating the 2026 season…")
def _run(_r_m, _d_m, _s_m):
    depth_df = pd.read_csv(_base / "data/raw/depth_charts.csv")
    return project_season(ratings, depth_df, divisions, schedules)


table, games = _run(
    _file_mtime(_base / "data/processed/team_ratings.csv"),
    _file_mtime(_base / "data/raw/depth_charts.csv"),
    _file_mtime(_base / "data/raw/schedules.csv"),
)

ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def _rec(w):
    wi = int(round(w))
    return f"{wi}–{17 - wi}"


# ══════════════════════════════════════════════════════════════════════════════
# LEAGUE PROJECTIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏆 League Projections")

conf_opt = st.radio("Conference", ["All", "AFC", "NFC"], horizontal=True, key="rp_conf")
league = table if conf_opt == "All" else table[table["conference"] == conf_opt]
league = league.reset_index(drop=True)

disp = pd.DataFrame({
    "Logo": [get_logo(t, teams_df) for t in league["team"]],
    "Team": league["team"],
    "Division": league["division"],
    "Proj Record": [_rec(w) for w in league["proj_wins"]],
    "Proj Wins": league["proj_wins"].round(1),
    "Power": league["power"].round(1),
    "Make Playoffs": league["playoff_pct"].round(0),
    "Win Division": league["div_title_pct"].round(0),
    "Proj Finish": league["exp_finish"].round(2),
})
st.dataframe(
    disp,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Logo": st.column_config.ImageColumn("", width="small"),
        "Proj Wins": st.column_config.NumberColumn("Proj Wins", format="%.1f"),
        "Power": st.column_config.NumberColumn("Power", format="%+.1f",
                    help="Projected point margin vs a league-average team"),
        "Make Playoffs": st.column_config.NumberColumn("Playoffs %", format="%d%%"),
        "Win Division": st.column_config.NumberColumn("Div Title %", format="%d%%"),
        "Proj Finish": st.column_config.NumberColumn("Avg Div Finish", format="%.2f",
                    help="Average finishing place within the division across simulations"),
    },
)
st.caption("Sorted by projected wins · 20,000 simulated seasons")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PROJECTED DIVISION STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Projected Division Standings")

div_order = ["AFC East", "AFC North", "AFC South", "AFC West",
             "NFC East", "NFC North", "NFC South", "NFC West"]
grid = st.columns(2)
for i, dv in enumerate(div_order):
    d = table[table["division"] == dv].sort_values("proj_wins", ascending=False).reset_index(drop=True)
    if d.empty:
        continue
    html = f'<div style="margin-bottom:6px;"><b style="color:#4a4e69;">{dv}</b></div>'
    html += '<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.9rem;margin-bottom:18px;">'
    for place, (_, r) in enumerate(d.iterrows(), start=1):
        url = get_logo(r["team"], teams_df)
        logo = f'<img src="{url}" width="22" style="vertical-align:middle;margin-right:6px;">' if url else ""
        lead = "font-weight:700;" if place == 1 else ""
        html += (
            f'<tr style="border-bottom:1px solid #eceef4;{lead}">'
            f'<td style="padding:6px 4px;color:#8b8fa8;width:28px;">{ORD[place]}</td>'
            f'<td style="padding:6px 4px;">{logo}{r["team"]}</td>'
            f'<td style="padding:6px 4px;text-align:right;">{_rec(r["proj_wins"])}</td>'
            f'<td style="padding:6px 4px;text-align:right;color:#4f46e5;">{r["playoff_pct"]:.0f}% PO</td>'
            f'</tr>'
        )
    html += "</table>"
    grid[i % 2].markdown(html, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TEAM FOCUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 Team Deep Dive")

team_names = table.sort_values("team")["team"].tolist()
default = st.session_state.get("profile_team", team_names[0])
sel = st.selectbox("Select a team", team_names,
                   index=team_names.index(default) if default in team_names else 0,
                   key="rp_team")
trow = table[table["team"] == sel].iloc[0]
tcolor = TEAM_COLORS.get(sel, "#4f46e5")

m1, m2, m3, m4 = st.columns(4)
cards = [
    (m1, "Projected Record", _rec(trow["proj_wins"]), f"{trow['proj_wins']:.1f} expected wins"),
    (m2, "Power Rating", f"{trow['power']:+.1f}", "pts vs avg team"),
    (m3, "Make Playoffs", f"{trow['playoff_pct']:.0f}%", "across 20k sims"),
    (m4, "Win Division", f"{trow['div_title_pct']:.0f}%", f"proj finish {ORD.get(int(round(trow['exp_finish'])), '—')}"),
]
for col, label, val, sub in cards:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:18px 12px;">
            <div class="label">{label}</div>
            <div class="value" style="font-size:2rem;font-weight:800;color:{tcolor};margin:6px 0 4px;">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1, 1])

# ── Win-total distribution ────────────────────────────────────────────────────
with left:
    st.markdown("#### Win Total Distribution")
    dist = np.asarray(trow["win_dist"])
    fig = go.Figure(go.Bar(
        x=list(range(len(dist))), y=dist * 100,
        marker_color=tcolor,
        hovertemplate="%{x} wins: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                      xaxis_title="Wins", yaxis_title="Probability (%)")
    fig.add_vline(x=float(trow["proj_wins"]), line_dash="dot",
                  line_color="rgba(79,70,229,0.5)")
    st.plotly_chart(fig, use_container_width=True)

# ── Game-by-game schedule with win probability ───────────────────────────────
with right:
    st.markdown("#### 2026 Schedule & Win Odds")
    g = games[(games["home_team"] == sel) | (games["away_team"] == sel)].copy()
    g["is_home"] = g["home_team"] == sel
    g["opp"] = np.where(g["is_home"], g["away_team"], g["home_team"])
    g["win_prob"] = np.where(g["is_home"], g["p_home"], 1.0 - g["p_home"])
    g = g.sort_values("week")
    sched = pd.DataFrame({
        "Wk": g["week"].astype(int),
        "": ["vs" if h else "@" for h in g["is_home"]],
        "Opp Logo": [get_logo(o, teams_df) for o in g["opp"]],
        "Opponent": g["opp"].values,
        "Win %": (g["win_prob"] * 100).round(0).values,
    })
    st.dataframe(
        sched, hide_index=True, use_container_width=True, height=460,
        column_config={
            "Opp Logo": st.column_config.ImageColumn("", width="small"),
            "Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100,
                        format="%d%%"),
        },
    )

# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ How these projections are built"):
    st.markdown("""
Each team gets a **power rating** (projected point margin vs a league-average
opponent) that blends two signals:

- **Prior-season strength** — 2025 net points per game.
- **2026 roster talent** — the weighted 2025 PPR production of each team's
  projected offensive starters, pulled from the **updated post-offseason depth
  chart**, so trades, signings and rookies move the needle.

For every 2026 scheduled game the win probability comes from the rating gap plus
home-field advantage, run through a normal single-game margin model. The full
272-game slate is then **simulated 20,000 times** to produce win-total
distributions, division-title odds, playoff odds and expected finish
(4 division winners + 3 wild cards per conference, ties broken at random).

*Limitations:* defense enters only through prior-year net margin (player-level
data here is offense-only), and rookie starters carry no prior production.
    """)
