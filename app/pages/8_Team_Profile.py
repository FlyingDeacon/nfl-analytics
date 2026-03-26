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

# ── Load depth charts CSV ─────────────────────────────────────────────────────
@st.cache_data
def load_depth_charts():
    p = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "depth_charts.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

# PFR roster team code map for depth chart fallback
PFR_TEAM_CODES = {
    'ARI': 'ari', 'ATL': 'atl', 'BAL': 'rav', 'BUF': 'buf', 'CAR': 'car',
    'CHI': 'chi', 'CIN': 'cin', 'CLE': 'cle', 'DAL': 'dal', 'DEN': 'den',
    'DET': 'det', 'GB': 'gnb', 'HOU': 'hou', 'IND': 'ind', 'JAX': 'jac',
    'KC': 'kan', 'LAR': 'ram', 'LA': 'ram', 'LAC': 'lac', 'LV': 'rai',
    'MIA': 'mia', 'MIN': 'min', 'NE': 'nwe', 'NO': 'nor', 'NYG': 'nyg',
    'NYJ': 'nyj', 'PHI': 'phi', 'PIT': 'pit', 'SEA': 'sea', 'SF': 'sfo',
    'TB': 'tam', 'TEN': 'ten', 'WAS': 'wsh'
}

@st.cache_data(show_spinner=False)
def load_pfr_depth_chart(team_abbr: str, season: int) -> pd.DataFrame:
    import requests

    pfr_code = PFR_TEAM_CODES.get(team_abbr.upper(), team_abbr.lower())
    url = f"https://www.pro-football-reference.com/teams/{pfr_code}/{season}_roster.htm"

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; NFL-Analytics/1.0; +https://github.com/)' }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text, attrs={'id': 'roster'})
        if not tables:
            return pd.DataFrame()

        roster_df = tables[0].copy()
        if 'Player' not in roster_df.columns or 'Pos' not in roster_df.columns:
            return pd.DataFrame()

        # Normalize to depth chart format used in this app
        roster_df = roster_df.loc[:, roster_df.columns.intersection(['No.', 'Player', 'Pos'])]
        roster_df = roster_df.rename(columns={'No.': 'jersey_number', 'Player': 'player_name', 'Pos': 'position'})
        roster_df['team'] = team_abbr
        roster_df['season'] = season

        # Depth order per position from roster listing order
        roster_df['depth_order'] = roster_df.groupby('position').cumcount() + 1

        # set side by pos type
        offense = {'QB', 'RB', 'FB', 'WR', 'TE'}
        defense = {'DE', 'DT', 'NT', 'DL', 'EDGE', 'LB', 'ILB', 'OLB', 'MLB', 'CB', 'S', 'SS', 'FS', 'DB'}
        roster_df['side'] = roster_df['position'].apply(
            lambda p: 'offense' if p in offense else ('defense' if p in defense else 'offense')
        )

        return roster_df

    except Exception as e:
        st.warning(f"Unable to load PFR depth chart for {team_abbr} {season}: {e}")
        return pd.DataFrame()


st.set_page_config(page_title="Team Profile · NFL", page_icon="🏟️", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)
render_sidebar_nav(current_page="8_Team_Profile")

# ── Back button ──────────────────────────────────────────────────────────────
if st.button("← Back to Team Ratings", key="back_to_ratings"):
    st.switch_page("pages/1_Team_Ratings.py")

# ── Load data ────────────────────────────────────────────────────────────────
ratings      = load_ratings()
teams_df     = load_teams()
schedules    = load_schedules()
weekly       = load_weekly()
depth_charts = load_depth_charts()

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
# DEPTH CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Build weekly 2025 REG aggregated stats for this team (for merging into depth chart)
_wk = weekly.copy() if not weekly.empty else pd.DataFrame()
if not _wk.empty:
    if "season_type" in _wk.columns:
        _wk = _wk[_wk["season_type"] == "REG"]
    _wk_team = _wk[
        (_wk["recent_team"] == sel_team) & (_wk["season"] == sel_season)
    ].copy()
    sum_cols = [
        "fantasy_points_ppr", "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    ]
    sum_cols = [c for c in sum_cols if c in _wk_team.columns]
    if not _wk_team.empty and sum_cols:
        _gp = (
            _wk_team.groupby("player_display_name", as_index=False)["week"]
            .count().rename(columns={"week": "GP"})
        )
        _agg = _wk_team.groupby("player_display_name", as_index=False)[sum_cols].sum()
        _stats = _agg.merge(_gp, on="player_display_name", how="left")
        _stats["PPR Pts"] = _stats["fantasy_points_ppr"].round(1)
        _stats["PPG"]     = (_stats["fantasy_points_ppr"] / _stats["GP"].clip(lower=1)).round(1)
    else:
        _stats = pd.DataFrame()
else:
    _wk_team = pd.DataFrame()
    _stats   = pd.DataFrame()


def _merge_stats(dc_pos_df):
    """Merge depth chart rows with weekly stats by player name."""
    if _stats.empty:
        return dc_pos_df

    left_key = "player_name" if "player_name" in dc_pos_df.columns else "Player"
    merged = dc_pos_df.merge(
        _stats.drop(columns=["fantasy_points_ppr"], errors="ignore"),
        left_on=left_key,
        right_on="player_display_name",
        how="left",
        validate="m:m",
    )

    # Keep existing depth chart name column as-is (Player or player_name)
    if left_key != "player_name" and "player_name" in merged.columns:
        merged = merged.drop(columns=["player_name"], errors="ignore")

    return merged.drop(columns=["player_display_name"], errors="ignore")


# Get team's depth chart rows from the CSV (fallback to PFR roster if missing)
_dc = depth_charts[(depth_charts["team"] == sel_team)].copy() if not depth_charts.empty else pd.DataFrame()

if _dc.empty:
    # PFR team roster page for depth chart (example: Panthers 2025)
    _dc = load_pfr_depth_chart(sel_team, sel_season)

# preserve season label for caption
if not _dc.empty and "season" in _dc.columns:
    dc_season = int(_dc["season"].mode()[0])
else:
    dc_season = sel_season

# ══════════════════════════════════════════════════════════════════════════════
# OFFENSIVE DEPTH CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏈 Offensive Depth Chart")
st.caption(
    f"Projected 2026 roster depth · "
    f"2025 season stats shown where available (REG season)"
)

OFF_POS_GROUPS = [
    ("QB",  "Quarterbacks"),
    ("RB",  "Running Backs"),
    ("FB",  "Fullbacks"),
    ("WR",  "Wide Receivers"),
    ("TE",  "Tight Ends"),
]

OFF_STAT_COLS = {
    "QB": {
        "cols":   ["#", "Player", "GP", "PPR Pts", "PPG",
                   "completions", "attempts", "passing_yards", "passing_tds",
                   "interceptions", "carries", "rushing_yards", "rushing_tds"],
        "rename": {
            "completions": "CMP", "attempts": "ATT",
            "passing_yards": "Pass Yds", "passing_tds": "Pass TD",
            "interceptions": "INT", "carries": "Rush Att",
            "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
        },
    },
    "RB": {
        "cols":   ["#", "Player", "GP", "PPR Pts", "PPG",
                   "carries", "rushing_yards", "rushing_tds",
                   "receptions", "targets", "receiving_yards", "receiving_tds"],
        "rename": {
            "carries": "Car", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
            "receptions": "Rec", "targets": "Tgt",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
        },
    },
    "FB": {
        "cols":   ["#", "Player", "GP", "PPR Pts", "PPG",
                   "carries", "rushing_yards", "receptions", "receiving_yards"],
        "rename": {
            "carries": "Car", "rushing_yards": "Rush Yds",
            "receptions": "Rec", "receiving_yards": "Rec Yds",
        },
    },
    "WR": {
        "cols":   ["#", "Player", "GP", "PPR Pts", "PPG",
                   "targets", "receptions", "receiving_yards", "receiving_tds",
                   "carries", "rushing_yards"],
        "rename": {
            "targets": "Tgt", "receptions": "Rec",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
            "carries": "Rush Att", "rushing_yards": "Rush Yds",
        },
    },
    "TE": {
        "cols":   ["#", "Player", "GP", "PPR Pts", "PPG",
                   "targets", "receptions", "receiving_yards", "receiving_tds"],
        "rename": {
            "targets": "Tgt", "receptions": "Rec",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
        },
    },
}

if _dc.empty:
    # Fallback: use weekly data only, sorted by PPR pts
    st.info("Depth chart data not available — showing 2025 season stats.")
    if not _wk_team.empty:
        _fb = _wk_team[_wk_team["position"].isin(["QB","RB","FB","WR","TE"])].copy()
        if not _stats.empty:
            _fb = _stats.copy()
        st.dataframe(_fb, hide_index=True, use_container_width=True)
else:
    dc_off = _dc[_dc["side"] == "offense"].copy()

    for pos, label in OFF_POS_GROUPS:
        pos_df = dc_off[dc_off["position"] == pos].sort_values("depth_order").copy()
        if pos_df.empty:
            continue

        # Rename columns before merge
        pos_df = pos_df.rename(columns={
            "jersey_number": "#",
            "player_name":   "Player",
        })

        # Merge season stats
        merged = _merge_stats(pos_df)

        cfg    = OFF_STAT_COLS.get(pos, {"cols": ["#", "Player"], "rename": {}})
        show   = [c for c in cfg["cols"] if c in merged.columns]

        # Round float stat columns
        for fc in merged[show].select_dtypes("float").columns:
            if fc not in ("#", "PPG"):
                merged[fc] = merged[fc].fillna(0).round(0).astype(int, errors="ignore")
        if "PPG" in merged.columns:
            merged["PPG"] = merged["PPG"].fillna("—")
        if "PPR Pts" in merged.columns:
            merged["PPR Pts"] = merged["PPR Pts"].fillna("—")

        disp = merged[show].rename(columns=cfg["rename"])

        st.markdown(f"**{label}**")
        st.dataframe(
            disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PPG":     st.column_config.TextColumn("PPG"),
                "PPR Pts": st.column_config.TextColumn("PPR Pts"),
                "#":       st.column_config.NumberColumn("#", width="small"),
            },
        )
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE DEPTH CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🛡️ Defensive Depth Chart")
st.caption("Projected 2026 starters and key contributors by unit")

DEF_POS_GROUPS = [
    ("Defensive Line", ["DE", "DT", "NT", "DL", "EDGE"]),
    ("Linebackers",    ["LB", "ILB", "OLB", "MLB"]),
    ("Defensive Backs",["CB", "S", "SS", "FS", "DB"]),
]

if _dc.empty:
    st.info("Depth chart data not available for this team.")
else:
    dc_def = _dc[_dc["side"] == "defense"].copy()

    if dc_def.empty:
        st.info("No defensive depth chart data available.")
    else:
        for group_label, positions in DEF_POS_GROUPS:
            g_df = dc_def[dc_def["position"].isin(positions)].sort_values(
                ["position", "depth_order"]
            ).copy()
            if g_df.empty:
                continue
            disp = g_df[["jersey_number", "player_name", "position", "depth_order"]].rename(columns={
                "jersey_number": "#",
                "player_name":   "Player",
                "position":      "Pos",
                "depth_order":   "Depth",
            })
            st.markdown(f"**{group_label}**")
            st.dataframe(
                disp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "#":     st.column_config.NumberColumn("#", width="small"),
                    "Depth": st.column_config.NumberColumn("Depth", width="small"),
                },
            )
            st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    f"**Data sources** — Season ratings from processed team stats · "
    f"Records calculated from {sel_season} regular season schedule · "
    "Depth chart based on projected 2026 rosters · "
    "Season stats from 2025 regular season game logs."
)
