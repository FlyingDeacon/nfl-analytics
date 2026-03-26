import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_ratings, load_teams, load_schedules, load_weekly, get_logo, add_ranks
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Team Profile · NFL", page_icon="🏟️", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)
render_sidebar_nav(current_page="8_Team_Profile")

# ── Back button ──────────────────────────────────────────────────────────────
if st.button("← Back to Team Ratings", key="back_to_ratings"):
    st.switch_page("pages/1_Team_Ratings.py")

# ── Load data ────────────────────────────────────────────────────────────────
ratings   = load_ratings()
teams_df  = load_teams()
schedules = load_schedules()
weekly    = load_weekly()

# ── Sidebar: season + single team selector ───────────────────────────────────
seasons = sorted(ratings["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key="tp_season")

team_list = sorted(ratings[ratings["season"] == sel_season]["team"].unique().tolist())

# Honour pre-selection passed from Team Ratings (via session_state)
default_team = st.session_state.get("profile_team", team_list[0])
if default_team not in team_list:
    default_team = team_list[0]

sel_team = st.sidebar.selectbox(
    "Select Team",
    team_list,
    index=team_list.index(default_team),
    key="tp_team",
)
st.session_state["profile_team"] = sel_team

# ── Team metadata ────────────────────────────────────────────────────────────
abbr_col  = "team_abbr" if "team_abbr" in teams_df.columns else "team"
team_row  = teams_df[teams_df[abbr_col] == sel_team]
team_name = team_row.iloc[0]["team_name"]      if not team_row.empty else sel_team
team_div  = team_row.iloc[0]["team_division"]  if not team_row.empty else ""
logo_url  = get_logo(sel_team, teams_df)
team_color = TEAM_COLORS.get(sel_team, "#4f46e5")

# ── Page header ──────────────────────────────────────────────────────────────
logo_html = f'<img src="{logo_url}" width="72" style="margin-right:18px;object-fit:contain;">' if logo_url else ""
st.markdown(f"""
<div class="nfl-page-header" style="align-items:center;">
    {logo_html}
    <div>
        <div class="title">{team_name}</div>
        <div class="subtitle">{team_div} &nbsp;·&nbsp; {sel_season} Season Profile</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Team ratings for selected season ─────────────────────────────────────────
season_ratings = add_ranks(ratings[ratings["season"] == sel_season].copy())
team_stats = season_ratings[season_ratings["team"] == sel_team]

if team_stats.empty:
    st.warning(f"No rating data found for {sel_team} in {sel_season}.")
    st.stop()

row = team_stats.iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# STAT CARDS — PPG / OPPG / Net Rating with ranks
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns(3)
card_defs = [
    (c1, "Offensive PPG",  f"{row['ppg']:.1f}",      f"Rank #{int(row['offense_rank'])} of 32",  "⚔️"),
    (c2, "Defensive PPG",  f"{row['oppg']:.1f}",     f"Rank #{int(row['defense_rank'])} of 32",  "🛡️"),
    (c3, "Net Rating",     f"{row['net_ppg']:+.1f}",  f"Rank #{int(row['overall_rank'])} of 32", "📈"),
]
for col, label, val, rank_str, icon in card_defs:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:20px 12px;">
            <div class="label">{icon} {label}</div>
            <div class="value" style="font-size:2.2rem;font-weight:800;color:{team_color};margin:8px 0 4px;">{val}</div>
            <div class="sub" style="font-size:0.9rem;">{rank_str}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RECORD + DIVISION STANDING
# ══════════════════════════════════════════════════════════════════════════════

reg_games = schedules[
    (schedules["season"] == sel_season) &
    (schedules["game_type"] == "REG")
].copy()
reg_games["home_score"] = pd.to_numeric(reg_games.get("home_score", pd.Series(dtype=float)), errors="coerce")
reg_games["away_score"] = pd.to_numeric(reg_games.get("away_score", pd.Series(dtype=float)), errors="coerce")

def calc_record(df, team):
    """Return (W, L, T) for a team from a games dataframe."""
    home = df[df["home_team"] == team][["home_score", "away_score"]].rename(
        columns={"home_score": "pf", "away_score": "pa"}
    )
    away = df[df["away_team"] == team][["away_score", "home_score"]].rename(
        columns={"away_score": "pf", "home_score": "pa"}
    )
    both = pd.concat([home, away]).dropna()
    if both.empty:
        return 0, 0, 0
    w = int((both["pf"] > both["pa"]).sum())
    l = int((both["pf"] < both["pa"]).sum())
    t = int((both["pf"] == both["pa"]).sum())
    return w, l, t

# Overall record
ow, ol, ot = calc_record(reg_games, sel_team)
overall_rec = f"{ow}–{ol}" + (f"–{ot}" if ot else "")

# Division opponents
div_teams = (
    teams_df[teams_df["team_division"] == team_div][abbr_col].tolist()
    if team_div else []
)
div_opponents = [t for t in div_teams if t != sel_team]

div_games = reg_games[
    ((reg_games["home_team"] == sel_team) & (reg_games["away_team"].isin(div_opponents))) |
    ((reg_games["away_team"] == sel_team) & (reg_games["home_team"].isin(div_opponents)))
]
dw, dl, dt = calc_record(div_games, sel_team)
div_rec = f"{dw}–{dl}" + (f"–{dt}" if dt else "")

# Division standings
div_standing_rows = []
for t in div_opponents + [sel_team]:
    tw, tl, tt = calc_record(reg_games, t)
    total_g = tw + tl + tt
    div_standing_rows.append({
        "team": t,
        "W": tw, "L": tl, "T": tt,
        "win_pct": round((tw + 0.5 * tt) / max(total_g, 1), 3),
    })
div_standing_df = (
    pd.DataFrame(div_standing_rows)
    .sort_values(["W", "win_pct"], ascending=False)
    .reset_index(drop=True)
)
div_standing_df.insert(0, "Place", range(1, len(div_standing_df) + 1))

place_row = div_standing_df[div_standing_df["team"] == sel_team]
team_place = int(place_row.iloc[0]["Place"]) if not place_row.empty else 0
ordinals   = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
place_str  = ordinals.get(team_place, f"{team_place}th")

st.markdown("### 📋 Season Record & Division Standing")
r1, r2, r3 = st.columns(3)
record_cards = [
    (r1, "Overall Record",   overall_rec, f"{int(row['games'])} games played"),
    (r2, "Division Record",  div_rec,     team_div),
    (r3, "Division Standing", place_str,  team_div),
]
for col, label, val, sub in record_cards:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:20px 12px;">
            <div class="label">{label}</div>
            <div class="value" style="font-size:2.2rem;font-weight:800;color:{team_color};margin:8px 0 4px;">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Division standings table
st.markdown(f"#### {team_div} Standings — {sel_season}")

def _logo_cell(abbr):
    url = get_logo(abbr, teams_df)
    return f'<img src="{url}" width="26" style="vertical-align:middle;margin-right:6px;">' if url else ""

standings_html = """
<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.92rem;">
<thead>
  <tr style="border-bottom:2px solid #e2e5ef;color:#4a4e69;text-align:left;">
    <th style="padding:8px 6px;">Place</th>
    <th style="padding:8px 6px;">Team</th>
    <th style="padding:8px 6px;text-align:center;">W</th>
    <th style="padding:8px 6px;text-align:center;">L</th>
    <th style="padding:8px 6px;text-align:center;">T</th>
    <th style="padding:8px 6px;text-align:center;">Win %</th>
  </tr>
</thead>
<tbody>
"""
for _, srow in div_standing_df.iterrows():
    t       = srow["team"]
    is_sel  = t == sel_team
    bg      = f"background:{team_color}18;" if is_sel else ""
    fw      = "font-weight:700;" if is_sel else ""
    logo    = _logo_cell(t)
    rec_str = f"{int(srow['W'])}–{int(srow['L'])}" + (f"–{int(srow['T'])}" if srow['T'] else "")
    standings_html += f"""
  <tr style="{bg}{fw}border-bottom:1px solid #e8eaef;">
    <td style="padding:8px 6px;">{ordinals.get(int(srow['Place']), str(int(srow['Place'])))}</td>
    <td style="padding:8px 6px;">{logo}{t}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['W'])}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['L'])}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['T'])}</td>
    <td style="padding:8px 6px;text-align:center;">{srow['win_pct']:.3f}</td>
  </tr>"""
standings_html += "</tbody></table>"
st.markdown(standings_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# OFFENSIVE DEPTH CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏈 Offensive Depth Chart")
st.caption(f"Season totals · {sel_season} regular season · sorted by PPR fantasy points")

if weekly.empty:
    st.warning("Weekly player data not loaded.")
else:
    off_data = weekly[
        (weekly["recent_team"] == sel_team) &
        (weekly["season"] == sel_season) &
        (weekly.get("season_type", pd.Series("REG", index=weekly.index)) == "REG") &
        (weekly["position"].isin(["QB", "RB", "WR", "TE", "FB"]))
    ].copy()

    if "season_type" in weekly.columns:
        off_data = off_data[off_data["season_type"] == "REG"]

    if off_data.empty:
        st.info(f"No offensive player data found for {sel_team} in {sel_season}.")
    else:
        sum_cols = [
            "fantasy_points_ppr", "carries", "rushing_yards", "rushing_tds",
            "receptions", "targets", "receiving_yards", "receiving_tds",
            "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        ]
        sum_cols = [c for c in sum_cols if c in off_data.columns]

        # Count games separately then merge
        gp_df = (
            off_data.groupby(["player_display_name", "position"], as_index=False)
            ["week"].count().rename(columns={"week": "GP"})
        )
        agg_df = off_data.groupby(
            ["player_display_name", "position"], as_index=False
        )[sum_cols].sum()
        off_agg = agg_df.merge(gp_df, on=["player_display_name", "position"], how="left")
        off_agg["PPR Pts"] = off_agg["fantasy_points_ppr"].round(1)
        off_agg["PPG"]     = (off_agg["fantasy_points_ppr"] / off_agg["GP"].clip(lower=1)).round(1)

        pos_order = {"QB": 0, "RB": 1, "FB": 2, "WR": 3, "TE": 4}
        off_agg["_sort"] = off_agg["position"].map(pos_order).fillna(99)
        off_agg = off_agg.sort_values(["_sort", "PPR Pts"], ascending=[True, False]).drop(columns=["_sort"]).reset_index(drop=True)

        pos_configs = {
            "QB": {
                "label": "Quarterbacks",
                "cols":  ["player_display_name", "GP", "PPR Pts", "PPG",
                          "completions", "attempts", "passing_yards", "passing_tds",
                          "interceptions", "carries", "rushing_yards", "rushing_tds"],
                "rename": {
                    "player_display_name": "Player",
                    "completions": "CMP", "attempts": "ATT",
                    "passing_yards": "Pass Yds", "passing_tds": "Pass TD",
                    "interceptions": "INT", "carries": "Rush Att",
                    "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
                },
            },
            "RB": {
                "label": "Running Backs",
                "cols":  ["player_display_name", "GP", "PPR Pts", "PPG",
                          "carries", "rushing_yards", "rushing_tds",
                          "receptions", "targets", "receiving_yards", "receiving_tds"],
                "rename": {
                    "player_display_name": "Player",
                    "carries": "Carries", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
                    "receptions": "Rec", "targets": "Tgt",
                    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
                },
            },
            "FB": {
                "label": "Fullbacks",
                "cols":  ["player_display_name", "GP", "PPR Pts", "PPG",
                          "carries", "rushing_yards", "rushing_tds",
                          "receptions", "targets", "receiving_yards", "receiving_tds"],
                "rename": {
                    "player_display_name": "Player",
                    "carries": "Carries", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
                    "receptions": "Rec", "targets": "Tgt",
                    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
                },
            },
            "WR": {
                "label": "Wide Receivers",
                "cols":  ["player_display_name", "GP", "PPR Pts", "PPG",
                          "targets", "receptions", "receiving_yards", "receiving_tds",
                          "carries", "rushing_yards"],
                "rename": {
                    "player_display_name": "Player",
                    "targets": "Tgt", "receptions": "Rec",
                    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
                    "carries": "Rush Att", "rushing_yards": "Rush Yds",
                },
            },
            "TE": {
                "label": "Tight Ends",
                "cols":  ["player_display_name", "GP", "PPR Pts", "PPG",
                          "targets", "receptions", "receiving_yards", "receiving_tds"],
                "rename": {
                    "player_display_name": "Player",
                    "targets": "Tgt", "receptions": "Rec",
                    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
                },
            },
        }

        for pos, cfg in pos_configs.items():
            pos_df = off_agg[off_agg["position"] == pos].copy()
            if pos_df.empty:
                continue
            st.markdown(f"**{cfg['label']}**")
            show_cols = [c for c in cfg["cols"] if c in pos_df.columns]
            # Round float cols
            for fc in pos_df[show_cols].select_dtypes("float").columns:
                pos_df[fc] = pos_df[fc].round(0).astype(int, errors="ignore")

            st.dataframe(
                pos_df[show_cols].rename(columns=cfg["rename"]),
                hide_index=True,
                use_container_width=True,
                column_config={"PPG": st.column_config.NumberColumn(format="%.1f")},
            )
            st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE DEPTH CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🛡️ Defensive Depth Chart")

DEF_POSITIONS = {"CB", "S", "SS", "FS", "LB", "ILB", "OLB", "MLB",
                 "DE", "DT", "NT", "DL", "DB", "EDGE"}

if not weekly.empty and "position" in weekly.columns:
    def_data = weekly[
        (weekly["recent_team"] == sel_team) &
        (weekly["season"] == sel_season)
    ].copy()
    if "season_type" in def_data.columns:
        def_data = def_data[def_data["season_type"] == "REG"]
    def_data = def_data[def_data["position"].isin(DEF_POSITIONS)]

    if not def_data.empty:
        # Defensive players may have minimal fantasy stats; just show games + position
        def_gp = (
            def_data.groupby(["player_display_name", "position"], as_index=False)
            ["week"].count().rename(columns={"week": "GP"})
            .sort_values(["position", "GP"], ascending=[True, False])
            .reset_index(drop=True)
        )
        # Add headshot if available
        st.caption("Games played by defensive position — individual defensive stats (sacks, tackles, INTs) require an additional data source.")

        def_pos_groups = {
            "Defensive Line": ["DE", "DT", "NT", "DL", "EDGE"],
            "Linebackers":    ["LB", "ILB", "OLB", "MLB"],
            "Defensive Backs":["CB", "S", "SS", "FS", "DB"],
        }
        for group_label, positions in def_pos_groups.items():
            g_df = def_gp[def_gp["position"].isin(positions)].copy()
            if g_df.empty:
                continue
            st.markdown(f"**{group_label}**")
            st.dataframe(
                g_df.rename(columns={"player_display_name": "Player", "position": "Pos"}),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info(
            "Individual defensive player stats (tackles, sacks, INTs) are not included "
            "in the current data source. This section will be expanded in a future update "
            "once defensive game-log data is added."
        )
else:
    st.info("Weekly data not available.")

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    f"**Data source** — Season ratings from processed team stats · "
    f"Records calculated from {sel_season} regular season schedule · "
    "Offensive depth chart from weekly player game logs."
)
